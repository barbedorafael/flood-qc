# Flood QC - Rio Grande do Sul

Ferramenta em Python para **QC e preparo de dados de entrada/saída** do fluxo de previsão de cheias no RS.

Arquitetura intencionalmente enxuta:
- scripts em `src/`;
- configurações em `config/`;
- saídas em `data/`.

## O que já está implementado

- Inventário mínimo de estações (`src/station_inventory.py`).
- Coleta de telemetria ANA (chuva/nível/vazão) (`src/fetch_data.py`).
- Cálculo de chuva acumulada por estação + interpolação IDW (`src/accumulate_interpolate.py`).
- Dashboard exploratório em desenvolvimento (`app.py`).

## Próximos passos

- Implementar coleta de previsão de chuva (1, 3, 10, 30 dias).
- Implementar módulo de QC para modelagem (outliers, nodata, gaps curtos, flags).
- Gerar produto consolidado "pronto para modelagem" com rastreabilidade de processamento.
- Implementar relatório de estado por bacia.

## Estrutura do repositório

- `src/` scripts principais da ferramenta.
- `config/` parâmetros operacionais.
- `data/spatial/` arquivos-fonte de estações (entrada).
- `data/` saídas intermediárias e prontas para uso.
  - `estacoes_nivel.csv`, `estacoes_pluv.csv`
  - `telemetria/*.csv`
  - `accum/*.parquet`
  - `interp/*.tif`
  - `reports/<run_id>/*.json`

## Requisitos

- Python 3.10+
- Pacotes principais:
  - `pandas`
  - `requests`
  - `pyarrow`
  - `numpy`
  - `rasterio`
  - `pyyaml`
- Pacotes para o dashboard:
  - `streamlit`
  - `plotly`
  - `folium`
  - `streamlit-folium`
  - `branca`
  - `affine`

Exemplo de instalação rápida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas requests pyarrow numpy rasterio pyyaml streamlit plotly folium streamlit-folium branca affine
```

## Fluxo recomendado de execução

Execute na raiz do repositório:

```bash
python src/station_inventory.py
python src/fetch_data.py
python src/accumulate_interpolate.py
```

Para replay de evento:

```bash
# ajuste o replay em config/run.yaml (mode/event_name/reference_time_utc)
python src/fetch_data.py
python src/accumulate_interpolate.py
```

visualização:

```bash
streamlit run app.py
```

## Configuração

Arquivos:
- `config/default.yaml`: padrão geral.
- `config/run.yaml`: ajustes do dia (operacional/replay, data de referência).
- `config/events/*.yaml`: cenários históricos.
- `config/basins.yaml`: IDs de bacias e bacias com estatística detalhada.
- `config/qc.yaml`: thresholds de QC.

Parâmetros mais úteis:
- `run.reference_time_utc`: data/hora do run (se `null`, usa hora UTC atual).
- `windows.forecast_days`: janelas de previsão (1, 3, 10, 30 etc).
- `windows.accum_hours`: janelas de acumulado (24, 72, 240, 720 etc).
- `interpolation.grid_res_deg` e `interpolation.power`: parâmetros IDW.
- `basins.selected_ids` e `basins.detailed_stats_ids`.

Exemplo mínimo em `config/run.yaml`:

```yaml
run:
  mode: "event_replay"
  reference_time_utc: "2024-05-05T12:00:00Z"
  event_name: "maio_2024"
```

## Scripts e exemplos

### 1) `src/station_inventory.py`

Lê arquivos brutos de estações em `data/spatial/` e gera versões mínimas em `data/`.

Entradas esperadas:
- `data/spatial/EstacoesNivel.csv`
- `data/spatial/EstacoesPluv.csv`

Saídas:
- `data/estacoes_nivel.csv`
- `data/estacoes_pluv.csv`

Colunas mantidas:
- `ID`, `CODIGO`, `LAT`, `LON`, `ALT`, `NOME`

### 2) `src/fetch_data.py`

Busca dados recentes na API de telemetria ANA para as estações listadas em `data/estacoes_*.csv`.

Saída por estação:
- `data/telemetria/{CODIGO}.csv`

Campos gerados:
- `station_id`, `datetime`, `level`, `rain`, `flow`

Observações:
- Janela de coleta vem de `config/default.yaml` (`ingest.request_days`).
- O script limpa `data/telemetria/` no início do run e grava um CSV por estação.
- Dentro de cada arquivo, os registros são ordenados por tempo e deduplicados por `station_id + datetime`.

### 3) `src/accumulate_interpolate.py`

Processa a telemetria de chuva e gera:
1) séries acumuladas por estação;
2) rasters interpolados por IDW.

Horizontes atuais:
- 24h, 72h (3 dias), 240h (10 dias), 720h (30 dias)

Saídas:
- `data/accum/{CODIGO}.parquet`
- `data/interp/accum_{horizonte}.tif`

Observações:
- Resample horário com soma de chuva.
- Valores faltantes são tratados como `0` no cálculo de acumulado.
- Horizontes e parâmetros IDW vêm da configuração (`windows` e `interpolation`).

### 4) `app.py`

Dashboard Streamlit para inspeção rápida:
- mapa de estações;
- série temporal (chuva e nível);
- camada raster interpolada com transparência ajustável.

## Contrato de dados (estado atual)

### `data/estacoes_nivel.csv` e `data/estacoes_pluv.csv`
- Separador `;`
- Colunas mínimas: `CODIGO`, `LAT`, `LON` (ideal incluir `NOME`)

### `data/telemetria/*.csv`
- Colunas: `station_id`, `datetime`, `rain`, `level`, `flow`
- Frequência original da API (depois agregada para horário no script de acumulado)

### `data/accum/*.parquet`
- Colunas: `datetime`, `station_id`, `rain_acc_24h`, `rain_acc_72h`, `rain_acc_240h`, `rain_acc_720h`

### `data/interp/*.tif`
- COG em `EPSG:4326`
- Tags incluem horizonte e horário de referência

## JSONs para relatório

Cada run gera pasta `data/reports/<run_id>/` com:
- `config_snapshot.json`: configuração efetivamente usada no run.
- `fetch_data_summary.json`: status por estação na coleta ANA.
- `accumulate_interpolate_summary.json`: resumo de acumulados e camadas raster geradas.
- `basin_stats.json`: lista de bacias selecionadas e marcação de análise detalhada (estrutura base para relatório).

Exemplo de `fetch_data_summary.json`:

```json
{
  "step": "fetch_data",
  "run_id": "20260202T140000Z",
  "mode": "operational",
  "reference_time_utc": "2026-02-02T14:00:00Z",
  "stations_total": 312,
  "stations_ok": 287,
  "stations_no_data": 20,
  "stations_error": 5
}
```

Exemplo de `accumulate_interpolate_summary.json`:

```json
{
  "step": "accumulate_interpolate",
  "run_id": "20260202T140000Z",
  "accum_horizons_h": {"24h": 24, "72h": 72, "240h": 240, "720h": 720},
  "grid_res_deg": 0.1,
  "idw_power": 2.0,
  "stations_with_accum": 287,
  "layers_generated": ["accum_24h.tif", "accum_72h.tif", "accum_240h.tif", "accum_720h.tif"]
}
```

Exemplo de `basin_stats.json`:

```json
{
  "step": "accumulate_interpolate",
  "run_id": "20260202T140000Z",
  "reference_time_utc": "2026-02-02T14:00:00Z",
  "basins": [
    {"basin_id": "7601", "detailed": true, "status": "pending_stats_implementation"},
    {"basin_id": "7612", "detailed": false, "status": "pending_stats_implementation"}
  ]
}
```

## Ajustes operacionais

- Janela de coleta ANA: `config/default.yaml` → `ingest.request_days`.
- Janelas de acumulado: `config/default.yaml` → `windows.accum_hours`.
- Janelas de previsão: `config/default.yaml` → `windows.forecast_days`.
- Data de referência do run: `config/run.yaml` → `run.reference_time_utc`.
- Bacias com análise detalhada: `config/basins.yaml` → `basins.detailed_stats_ids`.
- Parâmetros de interpolação: `config/default.yaml` → `interpolation`.
