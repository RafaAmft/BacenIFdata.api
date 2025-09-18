"""Testes unitários para o cliente BCBIFDataClient."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from bcb_ifdata_client import BCBIFDataClient


@pytest.fixture()
def client_sem_cache() -> BCBIFDataClient:
    return BCBIFDataClient(cache=False)


def test_listar_relatorios_ordena_por_ano_mes(monkeypatch, client_sem_cache: BCBIFDataClient) -> None:
    dados = [
        {"Relatorio": 1, "Descricao": "Rel 1", "AnoMes": "202201"},
        {"Relatorio": 2, "Descricao": "Rel 2", "AnoMes": "202203"},
    ]

    def falso_coletar(endpoint: str, params: Dict[str, Any] | None = None):
        assert endpoint == "ListaDeRelatorio"
        return dados

    monkeypatch.setattr(client_sem_cache, "_coletar_registros", falso_coletar)

    resultado = client_sem_cache.listar_relatorios()
    assert list(resultado["AnoMes"]) == ["202203", "202201"]


def test_obter_cadastro_combina_filtros(monkeypatch, client_sem_cache: BCBIFDataClient) -> None:
    capturado: Dict[str, Any] = {}

    def falso_dataframe(endpoint: str, parametros: Dict[str, Any] | None = None, **kwargs: Any):
        capturado.update({"endpoint": endpoint, "parametros": parametros, **kwargs})
        return pd.DataFrame(
            [
                {"CodInst": "1", "NomeInstituicao": "Banco A", "Uf": "SP", "Situacao": "Ativo"},
                {"CodInst": "2", "NomeInstituicao": "Banco B", "Uf": "RJ", "Situacao": "Liquidacao"},
            ]
        )

    monkeypatch.setattr(client_sem_cache, "_coletar_dataframe", falso_dataframe)

    client_sem_cache.obter_cadastro_instituicoes(
        "202201", tipo=1, uf="SP", situacao="Ativo", filtros={"NomeInstituicao": "Banco"}
    )

    assert capturado["endpoint"] == "IfDataCadastro"
    assert capturado["parametros"] == {"AnoMes": "202201"}
    assert ("NomeInstituicao", "eq", "Banco") in capturado["filtros"]
    assert ("TipoInstituicao", "eq", 1) in capturado["filtros"]


def test_obter_dados_financeiros_constroi_parametros(monkeypatch, client_sem_cache: BCBIFDataClient) -> None:
    capturado: Dict[str, Any] = {}

    def falso_dataframe(endpoint: str, parametros: Dict[str, Any] | None = None, **kwargs: Any):
        capturado.update({"endpoint": endpoint, "parametros": parametros, **kwargs})
        return pd.DataFrame([{"CodInst": "1", "Saldo": 10}])

    monkeypatch.setattr(client_sem_cache, "_coletar_dataframe", falso_dataframe)

    client_sem_cache.obter_dados_financeiros("202201", tipo_inst=1, relatorio=123, filtros={"CodInst": "1"})

    assert capturado["endpoint"] == "IfDataValores"
    assert capturado["parametros"] == {
        "AnoMes": "202201",
        "TipoInstituicao": 1,
        "Relatorio": 123,
    }
    assert capturado["filtros"] == {"CodInst": "1"}


def test_consulta_periodo_agrega_resultados(monkeypatch, client_sem_cache: BCBIFDataClient) -> None:
    chamadas: list[str] = []

    def falso_obter(ano_mes: str, tipo_inst: int, relatorio: int, filtros: Any | None = None):
        chamadas.append(ano_mes)
        return pd.DataFrame(
            [
                {"CodInst": "1", "Saldo": 100, "AnoMes": ano_mes},
            ]
        )

    monkeypatch.setattr(client_sem_cache, "obter_dados_financeiros", falso_obter)

    resultado = client_sem_cache.consulta_periodo("202201", "202203", 1, 99, paralelo=False)

    assert chamadas == ["202201", "202202", "202203"]
    assert len(resultado) == 3


def test_exportar_excel_cria_arquivo(tmp_path: Path, client_sem_cache: BCBIFDataClient) -> None:
    arquivo = tmp_path / "saida.xlsx"
    dados = {
        "Resumo": pd.DataFrame({"AnoMes": ["202201"], "Saldo": [100]}),
        "Detalhes": pd.DataFrame({"CodInst": ["1"], "Saldo": [100]}),
    }
    caminho = client_sem_cache.exportar_excel(dados, arquivo)
    assert caminho.exists()


def test_gerar_relatorio_automatico(monkeypatch, tmp_path: Path, client_sem_cache: BCBIFDataClient) -> None:
    def falso_listar() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Relatorio": 1, "Descricao": "Rel", "AnoMes": "202201", "TipoInstituicao": 1},
                {"Relatorio": 2, "Descricao": "Rel 2", "AnoMes": "202201", "TipoInstituicao": 2},
            ]
        )

    def falso_consulta_periodo(inicio: str, fim: str, tipo_inst: int, relatorio: int, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"AnoMes": "202201", "Saldo": 100 * tipo_inst},
                {"AnoMes": "202202", "Saldo": 150 * tipo_inst},
            ]
        )

    def falso_exportar(dados: Dict[str, pd.DataFrame], arquivo: Path, **kwargs: Any) -> Path:
        arquivo = tmp_path / arquivo.name
        with pd.ExcelWriter(arquivo, engine="openpyxl") as writer:
            for nome, df in dados.items():
                df.to_excel(writer, sheet_name=nome[:31], index=False)
        return arquivo

    monkeypatch.setattr(client_sem_cache, "listar_relatorios", falso_listar)
    monkeypatch.setattr(client_sem_cache, "consulta_periodo", falso_consulta_periodo)
    monkeypatch.setattr(client_sem_cache, "exportar_excel", falso_exportar)

    resultado = client_sem_cache.gerar_relatorio_automatico(("202201", "202202"), tipos_inst=(1, 2), diretorio_saida=tmp_path)

    assert set(resultado.keys()) == {"periodos", "relatorio", "dados", "resumo", "arquivo_excel", "grafico"}
    assert resultado["arquivo_excel"].exists()
    assert Path(resultado["grafico"]).exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
