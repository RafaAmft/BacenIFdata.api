"""Exemplos de uso avançado do cliente BCBIFDataClient."""
from __future__ import annotations

import logging

import pandas as pd

from bcb_ifdata_client import BCBIFDataClient, BCBIFDataError

logging.basicConfig(level=logging.INFO)


def exemplo_listar_relatorios(client: BCBIFDataClient) -> pd.DataFrame:
    """Obtém e exibe a lista completa de relatórios disponíveis."""
    relatorios = client.listar_relatorios()
    print("Relatórios disponíveis:")
    print(relatorios.head())
    return relatorios


def exemplo_cadastro(client: BCBIFDataClient, ano_mes: str) -> pd.DataFrame:
    """Demonstra filtros combinados para o cadastro de instituições."""
    cadastro = client.obter_cadastro_instituicoes(
        ano_mes,
        tipo=1,
        uf="SP",
        situacao="Ativo",
        filtros={"NomeInstituicao": "COOPERATIVA"},
    )
    print("Cadastro filtrado:")
    print(cadastro.head())
    return cadastro


def exemplo_valores(client: BCBIFDataClient, ano_mes: str, relatorio: int) -> pd.DataFrame:
    """Coleta valores financeiros de uma instituição específica."""
    valores = client.obter_dados_financeiros(
        ano_mes,
        tipo_inst=1,
        relatorio=relatorio,
        filtros={"CodInst": "00000000"},
    )
    print("Valores financeiros:")
    print(valores.head())
    return valores


def exemplo_busca_instituicao(client: BCBIFDataClient, termo: str) -> pd.DataFrame:
    """Busca uma instituição pelo nome, retornando dados consolidados."""
    dados = client.buscar_por_instituicao(termo, periodo=("202201", "202203"))
    print(f"Dados encontrados para {termo}:")
    print(dados.head())
    return dados


def exemplo_relatorio_automatico(client: BCBIFDataClient) -> None:
    """Gera um relatório completo com exportação e gráficos."""
    resultado = client.gerar_relatorio_automatico(("202201", "202203"), tipos_inst=(1, 2))
    print("Relatório gerado:")
    print(f"Excel: {resultado['arquivo_excel']}")
    print(f"Gráfico: {resultado['grafico']}")


def main() -> None:
    client = BCBIFDataClient(cache=True)
    try:
        relatorios = exemplo_listar_relatorios(client)
        if relatorios.empty:
            print("Nenhum relatório retornado pela API.")
            return
        relatorio_padrao = relatorios.iloc[0]["Relatorio"] if "Relatorio" in relatorios.columns else 0
        ano_mes = str(relatorios.iloc[0].get("AnoMes", "202201"))
        exemplo_cadastro(client, ano_mes)
        exemplo_valores(client, ano_mes, relatorio_padrao)
        exemplo_busca_instituicao(client, "BANCO")
        exemplo_relatorio_automatico(client)
    except BCBIFDataError as erro:
        print("Falha ao executar exemplos:", erro)


if __name__ == "__main__":
    main()
