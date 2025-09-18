"""Cliente robusto para a API IFData do Banco Central do Brasil."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin

import pandas as pd
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

try:  # pragma: no cover - dependência opcional tratada dinamicamente
    import requests_cache
except ImportError:  # pragma: no cover
    requests_cache = None

from configuracao import (
    BASE_URL,
    CACHE_BACKEND,
    CACHE_EXPIRATION,
    CACHE_NAME,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP,
    ENDPOINTS,
    MAX_PAGE_SIZE,
)


_LOGGER = logging.getLogger(__name__)


class BCBIFDataError(RuntimeError):
    """Erro base para o cliente IFData."""


class BCBIFDataHTTPError(BCBIFDataError):
    """Erro HTTP com informações adicionais."""

    def __init__(self, status_code: int, message: str, *, suggestion: str | None = None, details: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.suggestion = suggestion
        self.details = details

    def __str__(self) -> str:  # pragma: no cover - representação amigável
        base = f"HTTP {self.status_code}: {super().__str__()}"
        if self.suggestion:
            base += f" | Sugestão: {self.suggestion}"
        if self.details:
            base += f" | Detalhes: {self.details}"
        return base


@dataclass
class _FunctionCall:
    endpoint: str
    params: Dict[str, Any]


class BCBIFDataClient:
    """Cliente para consumo da API IFData.

    O cliente foi projetado para oferecer uma interface de alto nível, com
    cache opcional, auto-descoberta de campos, tratamento de erros detalhado e
    ferramentas auxiliares para exportação e análises.
    """

    def __init__(
        self,
        *,
        cache: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
        base_url: str = BASE_URL,
        cache_expiration: int = CACHE_EXPIRATION,
        max_workers: int = DEFAULT_MAX_WORKERS,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.logger = logger or _LOGGER
        self.session = session or self._construir_sessao(cache, cache_expiration)
        self._metadata_cache: Optional[str] = None
        self._ultimo_periodo: Optional[str] = None

    # ------------------------------------------------------------------
    # Sessão HTTP e utilidades básicas
    # ------------------------------------------------------------------
    def _construir_sessao(self, cache: bool, cache_expiration: int) -> requests.Session:
        if cache and requests_cache is not None:
            session: requests.Session = requests_cache.CachedSession(
                cache_name=CACHE_NAME,
                backend=CACHE_BACKEND,
                expire_after=cache_expiration,
            )
            self.logger.debug("Sessão com cache ativado (backend=%s).", CACHE_BACKEND)
        else:
            session = requests.Session()
            if cache and requests_cache is None:
                self.logger.warning(
                    "requests-cache não disponível; prosseguindo sem cache em memória."
                )

        retry_strategy = Retry(
            total=DEFAULT_MAX_RETRIES,
            connect=DEFAULT_MAX_RETRIES,
            read=DEFAULT_MAX_RETRIES,
            backoff_factor=DEFAULT_BACKOFF_FACTOR,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _resolver_url(self, endpoint: str) -> str:
        return urljoin(self.base_url, endpoint)

    def _extrair_mensagem_erro(self, response: requests.Response) -> Tuple[str, Optional[str]]:
        mensagem = response.text
        sugestao = None
        try:
            payload = response.json()
        except ValueError:
            return mensagem, sugestao

        if isinstance(payload, dict):
            erro = payload.get("error")
            if isinstance(erro, dict):
                mensagem = erro.get("message", mensagem)
                detalhes = erro.get("details")
                if detalhes:
                    sugestao = detalhes
        return mensagem, sugestao

    def _sugerir_parametros(self, endpoint: str) -> str:
        definicao = ENDPOINTS.get(endpoint.split("(")[0], None)
        if not definicao or not definicao.parametros_obrigatorios:
            return ""
        return (
            "Parâmetros obrigatórios: "
            + ", ".join(f"{nome}" for nome in definicao.parametros_obrigatorios)
        )

    def _tratar_http_error(self, response: requests.Response, endpoint: str) -> None:
        mensagem, detalhes = self._extrair_mensagem_erro(response)
        sugestao = self._sugerir_parametros(endpoint)
        raise BCBIFDataHTTPError(
            response.status_code,
            mensagem,
            suggestion=sugestao or None,
            details=detalhes,
        )

    def _formatar_valor(self, valor: Any) -> Any:
        if isinstance(valor, str):
            # Ajuste para strings já encapsuladas
            if not (valor.startswith("'") and valor.endswith("'")):
                return f"'{valor}'"
            return valor
        if isinstance(valor, datetime):
            return f"datetime'{valor.isoformat()}'"
        if isinstance(valor, bool):
            return "true" if valor else "false"
        return valor

    def _formatar_filtro(self, filtros: Any) -> str:
        if filtros is None:
            return ""
        if isinstance(filtros, str):
            return filtros
        condicoes: List[str] = []
        if isinstance(filtros, dict):
            for chave, valor in filtros.items():
                condicoes.append(f"{chave} eq {self._formatar_valor(valor)}")
        elif isinstance(filtros, Iterable):
            for item in filtros:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    campo, operador, valor = item
                    condicoes.append(f"{campo} {operador} {self._formatar_valor(valor)}")
                else:
                    condicoes.append(str(item))
        else:
            condicoes.append(str(filtros))
        return " and ".join(condicoes)

    def _preparar_chamada(self, nome: str, parametros: Dict[str, Any] | None) -> _FunctionCall:
        parametros = parametros or {}
        if parametros:
            placeholders = ",".join(f"{chave}=@{chave}" for chave in parametros)
            endpoint = f"{nome}({placeholders})"
        else:
            endpoint = nome
        params = {f"@{chave}": self._formatar_valor(valor) for chave, valor in parametros.items()}
        return _FunctionCall(endpoint=endpoint, params=params)

    # Retry manual adicional para garantir robustez
    @retry(
        reraise=True,
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BACKOFF_FACTOR, min=DEFAULT_BACKOFF_FACTOR, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    )
    def _executar_requisicao(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params}
        params.setdefault("$format", "json")
        url = self._resolver_url(endpoint)
        self.logger.debug("Requisição GET %s params=%s", url, params)
        try:
            resposta = self.session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            self.logger.error("Timeout acessando %s: %s", endpoint, exc)
            raise
        except requests.RequestException as exc:
            self.logger.error("Erro de rede em %s: %s", endpoint, exc)
            raise

        if resposta.status_code >= 400:
            self.logger.error(
                "Erro HTTP %s ao acessar %s | params=%s | corpo=%s",
                resposta.status_code,
                endpoint,
                params,
                resposta.text,
            )
            self._tratar_http_error(resposta, endpoint)

        try:
            return resposta.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - resposta inesperada
            raise BCBIFDataError("Resposta JSON inválida do serviço IFData") from exc

    def _coletar_registros(self, endpoint: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        parametros = params or {}
        todos: List[Dict[str, Any]] = []
        proximo_endpoint = endpoint
        proximo_params = parametros

        while True:
            payload = self._executar_requisicao(proximo_endpoint, proximo_params)
            valores = payload.get("value")
            if valores is None:
                if payload:
                    todos.append(payload)
                break
            todos.extend(valores)
            next_link = payload.get("@odata.nextLink")
            if not next_link:
                break
            if next_link.startswith(self.base_url):
                next_link = next_link[len(self.base_url) :]
            proximo_endpoint = next_link
            proximo_params = {}
        return todos

    def _coletar_dataframe(
        self,
        nome_endpoint: str,
        parametros: Dict[str, Any] | None = None,
        filtros: Any | None = None,
        params_extra: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        chamada = self._preparar_chamada(nome_endpoint, parametros)
        parametros_query = {**chamada.params}
        if filtros:
            parametros_query["$filter"] = self._formatar_filtro(filtros)
        if params_extra:
            parametros_query.update(params_extra)
        registros = self._coletar_registros(chamada.endpoint, parametros_query)
        return self._registros_para_dataframe(registros)

    def _registros_para_dataframe(self, registros: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        if not registros:
            return pd.DataFrame()
        df = pd.DataFrame(registros)
        # Conversão de colunas para tipos apropriados
        for coluna in df.columns:
            if df[coluna].dropna().empty:
                continue
            if df[coluna].dtype == object:
                df[coluna] = df[coluna].apply(self._converter_valor)
        return df

    def _converter_valor(self, valor: Any) -> Any:
        if isinstance(valor, str):
            valor = valor.strip()
            # Datas ISO
            data_regex = re.compile(r"^\d{4}-\d{2}-\d{2}")
            if data_regex.match(valor):
                try:
                    return date_parser.parse(valor)
                except (ValueError, OverflowError):
                    return valor
            # Números
            valor_normalizado = valor.replace(".", "").replace(",", ".")
            if valor_normalizado.replace("-", "", 1).replace(".", "", 1).isdigit():
                try:
                    numero = float(valor_normalizado)
                    if numero.is_integer():
                        return int(numero)
                    return numero
                except ValueError:
                    return valor
        return valor

    # ------------------------------------------------------------------
    # Métodos públicos principais
    # ------------------------------------------------------------------
    def listar_relatorios(self) -> pd.DataFrame:
        registros = self._coletar_registros("ListaDeRelatorio", {"$top": DEFAULT_TOP})
        df = self._registros_para_dataframe(registros)
        if "AnoMes" in df.columns:
            df = df.sort_values("AnoMes", ascending=False)
        return df.reset_index(drop=True)

    def obter_cadastro_instituicoes(
        self,
        ano_mes: Optional[str] = None,
        *,
        tipo: Optional[Union[str, int]] = None,
        uf: Optional[str] = None,
        situacao: Optional[str] = None,
        filtros: Any | None = None,
        tamanho_pagina: int = MAX_PAGE_SIZE,
    ) -> pd.DataFrame:
        ano_mes = self._normalizar_ano_mes(ano_mes)
        filtros_combinados = self._combinar_filtros_cadastro(tipo=tipo, uf=uf, situacao=situacao, filtros=filtros)
        df = self._coletar_dataframe(
            "IfDataCadastro",
            {"AnoMes": ano_mes},
            filtros=filtros_combinados,
            params_extra={"$top": tamanho_pagina},
        )
        return df

    def obter_dados_financeiros(
        self,
        ano_mes: str,
        tipo_inst: Union[int, str],
        relatorio: Union[int, str],
        *,
        filtros: Any | None = None,
        tamanho_pagina: int = MAX_PAGE_SIZE,
    ) -> pd.DataFrame:
        ano_mes_normalizado = self._normalizar_ano_mes(ano_mes)
        parametros = {
            "AnoMes": ano_mes_normalizado,
            "TipoInstituicao": tipo_inst,
            "Relatorio": relatorio,
        }
        df = self._coletar_dataframe(
            "IfDataValores",
            parametros,
            filtros=filtros,
            params_extra={"$top": tamanho_pagina},
        )
        return df

    def buscar_por_instituicao(
        self,
        nome_ou_cnpj: str,
        *,
        periodo: Optional[Union[str, Sequence[str], Tuple[str, str]]] = None,
        tipo_inst: Optional[Union[int, str]] = None,
        relatorio: Optional[Union[int, str]] = None,
    ) -> pd.DataFrame:
        periodos = self._normalizar_periodo(periodo)
        cadastros: List[pd.DataFrame] = []
        for periodo_ref in periodos:
            cadastros.append(
                self.obter_cadastro_instituicoes(periodo_ref)
            )
        if not cadastros:
            return pd.DataFrame()
        cadastro_df = pd.concat(cadastros, ignore_index=True).drop_duplicates()
        nome_coluna = self._resolver_coluna(cadastro_df.columns, ["NomeInstituicao", "Nome", "NomeEntidade"])
        codigo_coluna = self._resolver_coluna(cadastro_df.columns, ["CodInst", "Codigo", "CodInstituicao"])
        cnpj_coluna = self._resolver_coluna(cadastro_df.columns, ["Cnpj", "CNPJ", "CnpjBase"])

        filtro_series = pd.Series([False] * len(cadastro_df))
        busca_normalizada = nome_ou_cnpj.strip().lower()
        if nome_coluna:
            filtro_series |= cadastro_df[nome_coluna].astype(str).str.lower().str.contains(busca_normalizada, na=False)
        if busca_normalizada.isdigit():
            if codigo_coluna:
                filtro_series |= cadastro_df[codigo_coluna].astype(str) == busca_normalizada
            if cnpj_coluna:
                filtro_series |= cadastro_df[cnpj_coluna].astype(str).str.replace(r"\D", "", regex=True) == busca_normalizada
        cadastro_filtrado = cadastro_df.loc[filtro_series]
        if cadastro_filtrado.empty:
            return cadastro_filtrado

        resultados: List[pd.DataFrame] = []
        for _, linha in cadastro_filtrado.iterrows():
            codigo = linha.get(codigo_coluna)
            tipo_para_busca = tipo_inst
            if tipo_para_busca is None:
                tipo_para_busca = linha.get(
                    self._resolver_coluna(cadastro_filtrado.columns, ["TipoInstituicao", "TipoInst", "Tipo"])
                )
            relatorio_para_busca = relatorio
            if relatorio_para_busca is None:
                relatorio_para_busca = self._selecionar_relatorio_padrao(tipo_para_busca)
            filtros = {"CodInst": codigo} if codigo is not None else None
            for periodo_ref in periodos:
                try:
                    dados = self.obter_dados_financeiros(
                        periodo_ref,
                        tipo_inst=tipo_para_busca,
                        relatorio=relatorio_para_busca,
                        filtros=filtros,
                    )
                    if not dados.empty:
                        resultados.append(dados.assign(AnoMes=periodo_ref))
                except BCBIFDataHTTPError as erro:
                    self.logger.warning(
                        "Falha ao obter dados financeiros para %s (%s): %s",
                        linha.get(nome_coluna),
                        codigo,
                        erro,
                    )
        if not resultados:
            return pd.DataFrame()
        return pd.concat(resultados, ignore_index=True)

    def consulta_periodo(
        self,
        inicio: str,
        fim: str,
        tipo_inst: Union[int, str],
        relatorio: Union[int, str],
        *,
        filtros: Any | None = None,
        paralelo: bool = True,
    ) -> pd.DataFrame:
        periodos = self._gerar_periodo(inicio, fim)
        resultados: List[pd.DataFrame] = []

        if paralelo and len(periodos) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tarefas = {
                    executor.submit(
                        self.obter_dados_financeiros, periodo, tipo_inst, relatorio, filtros=filtros
                    ): periodo
                    for periodo in periodos
                }
                for tarefa in tqdm(as_completed(tarefas), total=len(tarefas), desc="Consultando período"):
                    periodo = tarefas[tarefa]
                    try:
                        dados = tarefa.result()
                        if not dados.empty:
                            dados = dados.assign(AnoMes=periodo)
                            resultados.append(dados)
                    except BCBIFDataError as erro:
                        self.logger.error("Falha na consulta do período %s: %s", periodo, erro)
        else:
            for periodo in tqdm(periodos, desc="Consultando período"):
                try:
                    dados = self.obter_dados_financeiros(periodo, tipo_inst, relatorio, filtros=filtros)
                except BCBIFDataError as erro:
                    self.logger.error("Falha na consulta do período %s: %s", periodo, erro)
                    continue
                if not dados.empty:
                    dados = dados.assign(AnoMes=periodo)
                    resultados.append(dados)

        if not resultados:
            return pd.DataFrame()
        return pd.concat(resultados, ignore_index=True)

    def exportar_excel(
        self,
        dados: Union[pd.DataFrame, Dict[str, pd.DataFrame], Sequence[pd.DataFrame]],
        arquivo: Union[str, Path],
        *,
        abas: Optional[Sequence[str]] = None,
        incluir_indice: bool = False,
    ) -> Path:
        destino = Path(arquivo)
        if destino.suffix.lower() != ".xlsx":
            destino = destino.with_suffix(".xlsx")
        destino.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(dados, pd.DataFrame):
            frames = {abas[0] if abas else "Dados": dados}
        elif isinstance(dados, dict):
            frames = dados
        else:
            frames = {}
            abas = list(abas or [])
            for idx, frame in enumerate(dados):
                nome_aba = abas[idx] if idx < len(abas) else f"Aba_{idx+1}"
                frames[nome_aba] = frame

        with pd.ExcelWriter(destino, engine="openpyxl") as writer:
            for nome_aba, frame in frames.items():
                frame.to_excel(writer, sheet_name=nome_aba[:31], index=incluir_indice)
        self.logger.info("Dados exportados para %s", destino)
        return destino

    def gerar_relatorio_automatico(
        self,
        periodo: Union[str, Sequence[str], Tuple[str, str]],
        *,
        tipos_inst: Sequence[Union[int, str]] = (1, 2, 3),
        relatorio: Optional[Union[int, str]] = None,
        diretorio_saida: Path | None = None,
    ) -> Dict[str, Any]:
        periodos = self._normalizar_periodo(periodo)
        if relatorio is None:
            relatorio = self._selecionar_relatorio_padrao(tipos_inst[0])
        diretorio = Path(diretorio_saida or DEFAULT_OUTPUT_DIR)
        diretorio.mkdir(parents=True, exist_ok=True)

        dados_por_tipo: Dict[Union[int, str], pd.DataFrame] = {}
        for tipo in tipos_inst:
            dados = self.consulta_periodo(periodos[0], periodos[-1], tipo, relatorio)
            if not dados.empty:
                dados_por_tipo[tipo] = dados

        if not dados_por_tipo:
            raise BCBIFDataError("Nenhum dado retornado para o período/tipos informados.")

        resumo_list: List[pd.DataFrame] = []
        for tipo, dados in dados_por_tipo.items():
            coluna_valor = self._resolver_coluna(
                dados.columns,
                ["Saldo", "Valor", "SaldoFinal", "SaldoConta", "Montante"]
            )
            if not coluna_valor:
                continue
            agrupado = (
                dados.groupby("AnoMes")[coluna_valor]
                .sum(min_count=1)
                .reset_index()
                .rename(columns={coluna_valor: "ValorAgregado"})
                .assign(TipoInstituicao=tipo)
            )
            resumo_list.append(agrupado)

        if not resumo_list:
            raise BCBIFDataError("Não foi possível determinar coluna de valores para agregação.")

        resumo_df = pd.concat(resumo_list, ignore_index=True)
        tabela_pivot = resumo_df.pivot_table(
            index="AnoMes", columns="TipoInstituicao", values="ValorAgregado", aggfunc="sum"
        )
        tabela_pivot = tabela_pivot.sort_index()

        figura_path = diretorio / f"grafico_{periodos[0]}_{periodos[-1]}.png"
        plt.figure(figsize=(10, 6))
        for coluna in tabela_pivot.columns:
            plt.plot(tabela_pivot.index, tabela_pivot[coluna], marker="o", label=f"Tipo {coluna}")
        plt.title("Evolução por Tipo de Instituição")
        plt.xlabel("Período (Ano/Mês)")
        plt.ylabel("Valor agregado")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figura_path)
        plt.close()

        arquivo_excel = diretorio / f"relatorio_ifdata_{periodos[0]}_{periodos[-1]}.xlsx"
        abas = {"Resumo": resumo_df, **{f"Tipo_{tipo}": dados for tipo, dados in dados_por_tipo.items()}}
        self.exportar_excel(abas, arquivo_excel)

        return {
            "periodos": periodos,
            "relatorio": relatorio,
            "dados": dados_por_tipo,
            "resumo": resumo_df,
            "arquivo_excel": arquivo_excel,
            "grafico": figura_path,
        }

    # ------------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------------
    def _resolver_coluna(self, colunas: Iterable[str], candidatos: Sequence[str]) -> Optional[str]:
        conjunto = {coluna.lower(): coluna for coluna in colunas}
        for candidato in candidatos:
            chave = candidato.lower()
            if chave in conjunto:
                return conjunto[chave]
        for candidato in candidatos:
            for coluna in colunas:
                if coluna.lower().startswith(candidato.lower()):
                    return coluna
        return None

    def _combinar_filtros_cadastro(
        self,
        *,
        tipo: Optional[Union[str, int]],
        uf: Optional[str],
        situacao: Optional[str],
        filtros: Any | None,
    ) -> Any:
        condicoes: List[Tuple[str, str, Any]] = []
        if filtros:
            if isinstance(filtros, (list, tuple)):
                condicoes.extend(tuple(filtros))
            elif isinstance(filtros, dict):
                condicoes.extend((chave, "eq", valor) for chave, valor in filtros.items())
            else:
                condicoes.append((str(filtros), "", ""))
        if tipo is not None:
            condicoes.append(("TipoInstituicao", "eq", tipo))
        if uf is not None:
            condicoes.append(("Uf", "eq", uf))
        if situacao is not None:
            condicoes.append(("Situacao", "eq", situacao))
        condicoes_validas = [c for c in condicoes if c[1]]
        if not condicoes_validas:
            return None
        return [(campo, operador, valor) for campo, operador, valor in condicoes_validas]

    def _normalizar_ano_mes(self, ano_mes: Optional[str]) -> str:
        if ano_mes:
            ano_mes = ano_mes.replace("-", "")
            if not re.fullmatch(r"\d{6}", ano_mes):
                raise ValueError("ano_mes deve estar no formato AAAAMM")
            return ano_mes
        if self._ultimo_periodo:
            return self._ultimo_periodo
        try:
            relatorios = self.listar_relatorios()
        except BCBIFDataError:
            data_atual = datetime.utcnow()
            ano_mes_atual = data_atual.strftime("%Y%m")
            self._ultimo_periodo = ano_mes_atual
            return ano_mes_atual
        if "AnoMes" in relatorios.columns and not relatorios.empty:
            ultimo = str(relatorios["AnoMes"].iloc[0])
        else:
            ultimo = datetime.utcnow().strftime("%Y%m")
        self._ultimo_periodo = ultimo
        return ultimo

    def _normalizar_periodo(
        self,
        periodo: Optional[Union[str, Sequence[str], Tuple[str, str]]],
    ) -> List[str]:
        if periodo is None:
            return [self._normalizar_ano_mes(None)]
        if isinstance(periodo, str):
            return [self._normalizar_ano_mes(periodo)]
        if isinstance(periodo, tuple) and len(periodo) == 2:
            inicio, fim = periodo
            return self._gerar_periodo(inicio, fim)
        return [self._normalizar_ano_mes(item) for item in periodo]

    def _gerar_periodo(self, inicio: str, fim: str) -> List[str]:
        inicio_norm = self._normalizar_ano_mes(inicio)
        fim_norm = self._normalizar_ano_mes(fim)
        ano_inicio, mes_inicio = int(inicio_norm[:4]), int(inicio_norm[4:])
        ano_fim, mes_fim = int(fim_norm[:4]), int(fim_norm[4:])
        data_inicio = datetime(ano_inicio, mes_inicio, 1)
        data_fim = datetime(ano_fim, mes_fim, 1)
        if data_inicio > data_fim:
            raise ValueError("Data inicial não pode ser maior que a final")
        periodos: List[str] = []
        data_corrente = data_inicio
        while data_corrente <= data_fim:
            periodos.append(data_corrente.strftime("%Y%m"))
            data_corrente += relativedelta(months=1)
        return periodos

    def _selecionar_relatorio_padrao(self, tipo_inst: Union[int, str, None]) -> Union[int, str]:
        relatorios = self.listar_relatorios()
        if relatorios.empty:
            raise BCBIFDataError("Não foi possível obter lista de relatórios.")
        if tipo_inst is not None:
            coluna_tipo = self._resolver_coluna(relatorios.columns, ["TipoInstituicao", "Tipo"])
            if coluna_tipo and coluna_tipo in relatorios.columns:
                filtrado = relatorios[relatorios[coluna_tipo] == tipo_inst]
                if not filtrado.empty:
                    return filtrado.iloc[0][self._resolver_coluna(relatorios.columns, ["Relatorio", "Codigo", "Id"])]
        coluna_relatorio = self._resolver_coluna(relatorios.columns, ["Relatorio", "Codigo", "Id"])
        return relatorios.iloc[0][coluna_relatorio]

    def obter_metadata(self) -> Optional[str]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        try:
            resposta = self.session.get(self._resolver_url("$metadata"), timeout=self.timeout)
            if resposta.status_code == 200:
                self._metadata_cache = resposta.text
                return self._metadata_cache
            self.logger.warning("Falha ao obter metadata: HTTP %s", resposta.status_code)
        except requests.RequestException as exc:  # pragma: no cover - dependência externa
            self.logger.warning("Não foi possível obter metadata: %s", exc)
        self._metadata_cache = None
        return None


__all__ = ["BCBIFDataClient", "BCBIFDataError", "BCBIFDataHTTPError"]
