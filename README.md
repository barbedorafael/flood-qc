# Sistema Operacional de Hidrologia e Previsao

Base operacional local-first para hidrologia, chuva e previsao no RS, orientada por artefatos locais, SQLite e scripts diretos em Python.

## Estado atual

O repositorio ja possui base funcional para:

- bootstrap de `data/history.sqlite` e `data/runs/<run_id>.sqlite`;
- ingest de observados ANA para `rain`, `level` e `flow`;
- ingest de grade ECMWF e registro do GRIB canonico no historico;
- preparacao de metadados e chuva horaria para o MGB;
- execucao real ou dry-run do runner do MGB;
- dashboard Streamlit para monitoramento, series MGB e preview/correcao manual de forecast ECMWF.

Ainda estao pendentes nesta fase:

- ingest operacional de chuva do INMET;
- QC automatico de observados;
- correcao manual de chuva observada;
- materializacao completa de runs operacionais em `data/runs/`;
- geracao de relatorios operacionais.

## Principios

- artefatos locais primeiro;
- SQLite como baseline operacional;
- um banco historico persistente em `data/history.sqlite`;
- um arquivo SQLite por run em `data/runs/`;
- rasters e vetores fora do banco, com paths relativos e metadados;
- Streamlit como interface principal;
- QGIS como cliente complementar sobre artefatos gerados.

## Layout principal

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

Importante: `data/spatial/`, `data/timeseries/` e `data/runs/` sao o layout alvo canonico. O repositorio ainda preserva artefatos legados fora dessa convencao.

## Runtime e configuracao

- Contrato oficial de runtime: `Python >= 3.11`
- Configuracao canonica nesta fase:
  - `config/default.yaml`
  - `config/custom.yaml`
- A avaliacao de migracao da configuracao para `.toml` segue em aberto, sem mudanca de contrato por enquanto.

O inventario inicial de estacoes fica em `data/interim/history_station_inventory.csv`. Durante o bootstrap do historico, o sistema calcula `station_uid` como `1000000000 + codigo` para ANA e `2000000000 + codigo` para INMET, convertendo letras do codigo para numeros (`A=1`, `B=2`, etc.).

## Setup local

Crie um ambiente virtual e instale as dependencias para uso completo local:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,data,geo,ui]
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev,data,geo,ui]
```

## Entry points canonicos

```bash
python src/storage/db_bootstrap.py --history
python src/ingest/fetch_observed_ana.py
python src/ingest/fetch_observed_inmet.py
python src/ingest/forecast_grid.py
python src/model/prepare_mgb_meta.py
python src/model/prepare_mgb_rainfall.py
python src/model/run_mgb.py --dry-run
streamlit run apps/ops_dashboard/app.py
```

Para rodar a ingestao INMET, defina `INMET_API_KEY` no ambiente ou preencha `.env` a partir de `.env.example`.

## Componentes principais

- `apps/ops_dashboard/`
  Dashboard operacional em Streamlit para monitoramento, series observadas, series MGB e preview/correcao de forecast ECMWF.
- `apps/mgb_runner/`
  Artefatos locais do MGB (`Input`, `Output` e `.exe`). O codigo do runner fica em `src/model/`.
- `apps/qgis_project/`
  Material de apoio para consumo espacial complementar.
- `src/`
  Modulos por dominio, separados entre ingestao, QC, modelo, storage, reporting e utilitarios comuns.
- `sql/`
  Schemas explicitos de `history.sqlite` e `run.sqlite`.
- `docs/`
  Arquitetura, modelo de dados, operacao e workflows.

## Banco historico vs banco de run

- `data/history.sqlite`
  Guarda metadados de estacoes, observados, flags, edicoes e catalogo de runs.
- `data/runs/<run_id>.sqlite`
  Guarda o estado fechado de um run especifico.

O schema de run existe e o bootstrap esta implementado, mas a montagem operacional completa do run ainda nao esta concluida nesta fase.
