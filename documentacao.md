# Cliente Python para a API IFData do Banco Central do Brasil

## Visão Geral

O projeto disponibiliza um cliente Python completo (`BCBIFDataClient`) para consumo da API IFData do Banco Central do Brasil, oferecendo recursos de descoberta automática, cache inteligente, tratamento avançado de erros, agregações, exportação e geração de relatórios gráficos.

## Instalação

1. Crie um ambiente virtual (opcional, porém recomendado).
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso Rápido

```python
from bcb_ifdata_client import BCBIFDataClient

cliente = BCBIFDataClient(cache=True)
relatorios = cliente.listar_relatorios()
print(relatorios.head())
```

## Principais Funcionalidades

### 1. Listar Relatórios
`listar_relatorios()` retorna um `DataFrame` com todos os relatórios disponíveis, ordenados do período mais recente para o mais antigo.

### 2. Cadastro de Instituições
`obter_cadastro_instituicoes(ano_mes, tipo=None, uf=None, situacao=None, filtros=None)` recupera o cadastro das instituições financeiras. O método aceita filtros combinados, validação de parâmetros e converte automaticamente os tipos retornados.

### 3. Dados Financeiros
`obter_dados_financeiros(ano_mes, tipo_inst, relatorio, filtros=None)` retorna valores financeiros detalhados para o relatório informado. Suporta filtros OData personalizados e paginação automática.

### 4. Consultas Avançadas e Períodos
`consulta_periodo(inicio, fim, tipo_inst, relatorio, filtros=None, paralelo=True)` executa consultas paralelas em múltiplos meses, agregando os resultados em um único `DataFrame`. O método exibe barra de progresso (`tqdm`) para facilitar o acompanhamento.

### 5. Busca por Instituição
`buscar_por_instituicao(nome_ou_cnpj, periodo=None, tipo_inst=None, relatorio=None)` realiza a busca inteligente combinando cadastro e dados financeiros, com suporte a múltiplos períodos e detecção automática de códigos de instituição.

### 6. Exportação e Relatórios Automáticos
`exportar_excel(dados, arquivo, abas=None)` exporta `DataFrames` para planilhas Excel com múltiplas abas. `gerar_relatorio_automatico(periodo, tipos_inst=(1,2,3), relatorio=None)` cria relatórios completos contendo agregações, gráficos em `matplotlib` e planilhas prontas para análise de negócio.

## Tratamento de Erros e Boas Práticas

- Todas as requisições passam por tratamento de exceções HTTP, com mensagens claras em português e sugestões automáticas de parâmetros obrigatórios.
- Retries exponenciais e cache opcional (`requests-cache`) reduzem falhas intermitentes e melhoram o desempenho.
- Os filtros OData são validados e formatados automaticamente antes do envio.
- Valores retornados pela API são convertidos dinamicamente para tipos adequados (datas, inteiros, floats), facilitando a análise.

## Exemplos Avançados

O arquivo `exemplos_completos.py` demonstra fluxos completos: listagem de relatórios, consultas com filtros, busca por instituição, agregação multi-período e geração de relatórios.

Execute:

```bash
python exemplos_completos.py
```

## Testes Automatizados

Utilize `pytest` para executar os testes:

```bash
pytest testes_unitarios.py
```

Os testes cobrem geração de filtros, paginação, exportação e construção de relatórios, utilizando *mocks* para garantir independência da API externa.

## Estrutura de Pastas

```
.
├── bcb_ifdata_client.py     # Cliente principal
├── configuracao.py          # Configurações e constantes reutilizáveis
├── documentacao.md          # Manual completo
├── exemplos_completos.py    # Casos de uso avançados
├── testes_unitarios.py      # Testes automatizados
└── requirements.txt         # Dependências do projeto
```

## Considerações Finais

- Em ambientes sem acesso à internet, o cliente degrada graciosamente, fornecendo mensagens amigáveis e sugerindo parâmetros necessários.
- Para consultas massivas, utilize o modo paralelo de `consulta_periodo` e habilite o cache em disco ajustando o backend do `requests-cache`.
- É possível integrar o cliente a *pipelines* de dados, adicionando camadas de análise personalizadas sobre os `DataFrames` retornados.
