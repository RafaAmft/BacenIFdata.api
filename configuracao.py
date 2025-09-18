"""Configurações e constantes utilizadas pelo cliente IFData."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# URL base oficial do serviço OData do IFData.
BASE_URL: str = "https://olinda.bcb.gov.br/olinda/servico/IFDATA/versao/v1/odata/"

# Parâmetros padrão de rede
DEFAULT_TIMEOUT: int = 30  # segundos
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
DEFAULT_MAX_WORKERS: int = 4

# Configuração de cache
CACHE_NAME: str = "bcb_ifdata_cache"
CACHE_BACKEND: str = "memory"
CACHE_EXPIRATION: int = 3600  # segundos

# Opções diversas
DEFAULT_TOP: int = 100
MAX_PAGE_SIZE: int = 100

# Caminhos padrão para exportação de relatórios
DEFAULT_OUTPUT_DIR: Path = Path("saidas")


@dataclass(frozen=True)
class EndpointDefinition:
    """Estrutura utilitária para mapear endpoints conhecidos."""

    nome: str
    parametros_obrigatorios: tuple[str, ...]
    descricao: str = ""


ENDPOINTS = {
    "ListaDeRelatorio": EndpointDefinition(
        nome="ListaDeRelatorio",
        parametros_obrigatorios=tuple(),
        descricao="Relação de relatórios e respectivas descrições",
    ),
    "IfDataCadastro": EndpointDefinition(
        nome="IfDataCadastro",
        parametros_obrigatorios=("AnoMes",),
        descricao="Cadastro das instituições financeiras cadastradas no IFData",
    ),
    "IfDataValores": EndpointDefinition(
        nome="IfDataValores",
        parametros_obrigatorios=("AnoMes", "TipoInstituicao", "Relatorio"),
        descricao="Valores financeiros associados a um relatório e instituição",
    ),
}


__all__ = [
    "BASE_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_FACTOR",
    "DEFAULT_MAX_WORKERS",
    "CACHE_NAME",
    "CACHE_BACKEND",
    "CACHE_EXPIRATION",
    "DEFAULT_TOP",
    "MAX_PAGE_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "EndpointDefinition",
    "ENDPOINTS",
]
