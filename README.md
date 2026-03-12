# Sistema Operacional de Hidrologia e Previsao

Base inicial para um sistema operacional de hidrologia e previsao orientado por artefatos locais, com foco em simplicidade, auditabilidade e crescimento incremental.

## Objetivo

Organizar o repositorio para suportar:

- coleta de dados observados e de previsao;
- QC automatico e revisao manual;
- montagem e execucao de runs hidrologicos;
- armazenamento local de historico e de runs;
- visualizacao operacional em Streamlit;
- consumo espacial complementar em QGIS;
- geracao de relatorios.

Nesta fase, o repositorio entrega estrutura, contratos, stubs, schemas SQL e documentacao. Integracoes externas, regras completas de QC, execucao real do MGB e UI final ainda nao estao implementadas.

O inventario inicial de estacoes do historico e mantido em `data/interim/history_station_inventory.csv` e carregado automaticamente durante o bootstrap do banco historico, que calcula `station_uid` como `1000000000 + codigo` para ANA e `2000000000 + codigo` para INMET, convertendo letras do codigo para numeros (`A=1`, `B=2`, etc.).

## Filosofia

- artefatos locais primeiro;
- um banco historico persistente em `data/history.sqlite`;
- um arquivo SQLite por run em `data/runs/`;
- rasters e vetores fora do banco, com paths relativos e metadados;
- Streamlit como interface principal;
- QGIS como cliente complementar sobre artefatos gerados.

## Componentes principais

- `apps/ops_dashboard/`
  Dashboard operacional em Streamlit, ainda em formato placeholder.
- `apps/mgb_runner/`
  Runner dedicado para o MGB Windows-only, com preparacao de comando e dry-run stubados.
- `apps/qgis_project/`
  Convencoes e instrucoes para abrir artefatos do pipeline no QGIS.
- `src/`
  Modulos por dominio, organizados em ingestao, QC, modelo, storage, reporting e utilitarios comuns.
- `sql/`
  Schemas explicitos de `history.sqlite` e `run.sqlite`.
- `docs/`
  Documentacao arquitetural, de modelo de dados, workflows e operacao.

## Banco historico vs banco de run

- `data/history.sqlite`
  Guarda metadados de estacoes, series observadas, flags, edicoes e o catalogo dos runs.
- `data/runs/<run_id>.sqlite`
  Guarda o estado de um run especifico, lineage, inputs, outputs, flags, edicoes, assets e relatorios.

Cada run manual gera um novo arquivo SQLite com referencia ao run automatico de origem. O run automatico nao e editado em lugar.

## Papel do Streamlit, QGIS e MGB runner

- `Streamlit`
  Interface principal para triagem, revisao rapida, navegacao entre runs e sumarios.
- `QGIS`
  Ferramenta complementar para inspecao espacial de GeoPackages e GeoTIFFs produzidos pelo pipeline.
- `MGB runner`
  Camada isolada para preparar e futuramente executar o modelo Windows-only sem misturar esse acoplamento com o restante do sistema.

## Estrutura do repositorio

```text
.
|-- apps/
|   |-- mgb_runner/
|   |-- ops_dashboard/
|   `-- qgis_project/
|-- config/
|-- data/
|   |-- interim/
|   |-- runs/
|   |-- spatial/
|   `-- timeseries/
|-- docs/
|-- sql/
|-- src/
|   |-- common/
|   |-- ingest/
|   |-- legacy/
|   |-- model/
|   |-- qc/
|   |-- reporting/
|   `-- storage/
`-- tests/
```

## Como comecar

Crie um ambiente virtual e instale as dependencias minimas:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Comandos iniciais uteis:

```bash
python src/storage/db_bootstrap.py --history
python src/storage/db_bootstrap.py --run-id 20260310T120000
streamlit run apps/ops_dashboard/app.py
python apps/mgb_runner/main.py --run-db data/runs/20260310T120000.sqlite --dry-run
pytest
```
