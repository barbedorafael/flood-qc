# Operacao e convencoes

## Setup local

1. Criar ambiente virtual com `Python 3.11+`.
2. Instalar dependencias com `pip install -e .[dev,data,geo,ui]`.
3. Ajustar `config/default.yaml` quando necessario para defaults operacionais.
4. Usar `config/custom.yaml` apenas para overrides locais.

Setup tipico em Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,data,geo,ui]
```

Setup tipico em Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev,data,geo,ui]
```

## Configuracao operacional

O runtime atual le exclusivamente:

- `config/default.yaml`
- `config/custom.yaml`

Esse continua sendo o contrato canonico de configuracao nesta fase. A eventual migracao para `.toml` segue em avaliacao.

## Entry points usuais

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

`python src/ingest/fetch_observed_inmet.py` requer `INMET_API_KEY` no ambiente local ou em `.env`.

## Convencoes de nomes

- `run_id`: preferencialmente `YYYYMMDDTHHMMSS`
- `history.sqlite`: banco historico unico
- `data/runs/<run_id>.sqlite`: um arquivo por run
- assets externos com paths relativos sempre que possivel

## Estados de maturidade

- `raw`: dado ingerido sem revisao completa
- `curated`: dado tratado por regras automaticas ou pre-processamento
- `approved`: dado liberado para uso operacional

O schema e o consumo do dashboard ja respeitam essa convencao, embora o fluxo automatico de promocao entre estados ainda esteja pendente.

## Artefato completo vs run

O fluxo operacional atual usa diretamente os artefatos completos do runner:

- `apps/mgb_runner/Output/QTUDO_Inercial_Atual.MGB`
- `apps/mgb_runner/Output/YTUDO.MGB`
- `apps/mgb_runner/Input/PARHIG.hig`
- `apps/mgb_runner/Input/MINI.gtp`

O schema de run continua previsto para guardar o subset operacional e o contexto fechado do ciclo, mas essa etapa ainda nao esta completa no pipeline atual.

## Paths de raster e vetores

- guardar path relativo no banco sempre que possivel
- nao armazenar raster como blob em SQLite
- preservar `data/spatial/` como destino canonico de camadas tratadas, mesmo que parte do consumo atual ainda use artefatos legados

## Edicao destrutiva e auditoria

- nao sobrescrever dado de origem
- registrar flags e edicoes de forma append-only quando aplicavel
- criar run manual derivado em vez de alterar um run automatico em lugar

Toda transformacao relevante deve explicitar:

- origem do dado ou asset
- momento da alteracao
- operador ou servico responsavel
- motivo da alteracao
- relacao com o run impactado, quando houver
