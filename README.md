# Flood QC - Rio Grande do Sul

Ferramenta em Python para **QC e preparo de dados de entrada/saída** do fluxo de previsão de cheias no RS.

Arquitetura intencionalmente enxuta:
- scripts em `src/`;
- configurações em `config/`;
- saídas em `data/`.

## O que já está implementado

- Inventário mínimo de estações (`src/station_inventory.py`).
- Coleta de telemetria ANA (chuva/nível/vazão) (`src/fetch_data.py`).
- Cálculo de chuva acumulada por estação (`src/accumulate.py`).
- Interpolação IDW a partir dos acumulados (`src/interpolate.py`).
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
  - `accum/*.csv`
  - `interp/*.tif`
  - `reports/<run_id>/*.json`

## Requisitos

- Python 3.10+
- Pacotes principais:
  - `pandas`
  - `requests`
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
pip install pandas requests numpy rasterio pyyaml streamlit plotly folium streamlit-folium branca affine
```

## Fluxo recomendado de execução

Execute na raiz do repositório:

```bash
python src/station_inventory.py
python src/fetch_data.py
python src/accumulate.py
python src/interpolate.py
```

Compatibilidade: `python src/accumulate_interpolate.py` ainda funciona e executa as duas etapas em sequência.

Para replay de evento:

```bash
# ajuste o replay em config/run.yaml (mode/event_name/reference_time)
python src/fetch_data.py
python src/accumulate.py
python src/interpolate.py
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
- `run.reference_time`: data/hora do run (se `null`, usa hora atual truncada para a hora cheia).
- `windows.forecast_days`: janelas de previsão (1, 3, 10, 30 etc).
- `windows.accum_hours`: janelas de acumulado (24, 72, 240, 720 etc).
- `interpolation.grid_res_deg` e `interpolation.power`: parâmetros IDW.
- `qc.level.min_cm`, `qc.level.max_cm`, `qc.level.max_step_cm_h`: limites de nível em centímetros.
- `basins.selected_ids` e `basins.detailed_stats_ids`.

Observação:
- Todos os horários do projeto usam `America/Sao_Paulo`.

Exemplo mínimo em `config/run.yaml`:

```yaml
run:
  mode: "event_replay"
  reference_time: "2024-05-05T09:00:00"
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
- `level` é mantido em centímetros (`cm`) conforme retorno da telemetria ANA.

### 3) `src/accumulate.py`

Processa a telemetria de chuva e gera séries acumuladas por estação.

Horizontes atuais:
- 24h, 72h (3 dias), 240h (10 dias), 720h (30 dias)

Saídas:
- `data/accum/{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv`

Observações:
- Resample horário com soma de chuva.
- Valores faltantes são tratados como `0` no cálculo de acumulado.
- Horizontes vêm da configuração (`windows.accum_hours`).
- `yyyymmdd/hhmm` representam `reference_time - horizonte`.

### 4) `src/interpolate.py`

Consome os acumulados e gera rasters interpolados por IDW.

Saídas:
- `data/interp/accum_{yyyymmdd}_{hhmm}_{horizonte}.tif`

Observações:
- Lê somente CSVs no padrão `{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv`.
- Parâmetros IDW vêm de `interpolation.grid_res_deg` e `interpolation.power`.

### 5) `app.py`

Dashboard Streamlit para inspeção rápida:
- mapa de estações;
- série temporal (chuva em mm e nível em cm);
- camada raster interpolada com transparência ajustável.

## Contrato de dados (estado atual)

### `data/estacoes_nivel.csv` e `data/estacoes_pluv.csv`
- Separador `;`
- Colunas mínimas: `CODIGO`, `LAT`, `LON` (ideal incluir `NOME`)

### `data/telemetria/*.csv`
- Colunas: `station_id`, `datetime`, `rain`, `level`, `flow`
- Unidades: `rain` em `mm`, `level` em `cm`, `flow` em `m3/s`
- Frequência original da API (depois agregada para horário no script de acumulado)

### `data/accum/*.csv`
- Nome: `{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv`
- Colunas: `station_id`, `reference_time`, `window_start`, `station_latest_time`, `horizon_label`, `horizon_hours`, `rain_acc_mm`

### `data/interp/*.tif`
- COG em `EPSG:4326`
- Tags incluem horizonte e horário de referência

## JSONs para relatório

Cada run gera pasta `data/reports/<run_id>/` com:
- `config_snapshot.json`: configuração efetivamente usada no run.
- `fetch_data_summary.json`: status por estação na coleta ANA.
- `accumulate_summary.json`: resumo da etapa de acumulado por estação.
- `interpolate_summary.json`: resumo da etapa de interpolação raster.
- `basin_stats.json`: lista de bacias selecionadas e marcação de análise detalhada (estrutura base para relatório).

Exemplo de `fetch_data_summary.json`:

```json
{
  "step": "fetch_data",
  "run_id": "20260202T140000",
  "mode": "operational",
  "reference_time": "2026-02-02T14:00:00",
  "stations_total": 312,
  "stations_ok": 287,
  "stations_no_data": 20,
  "stations_error": 5
}
```

Exemplo de `accumulate_summary.json`:

```json
{
  "step": "accumulate",
  "run_id": "20260202T140000",
  "accum_horizons_h": {"24h": 24, "72h": 72, "240h": 240, "720h": 720},
  "telemetry_files_found": 287,
  "stations_with_accum": 287,
  "accum_files_generated": ["83970000_20260201_1400_24h.csv", "..."],
  "accum_filename_pattern": "{station}_{yyyymmdd}_{hhmm}_{horizon}.csv"
}
```

Exemplo de `interpolate_summary.json`:

```json
{
  "step": "interpolate",
  "run_id": "20260202T140000",
  "accum_horizons_h": {"24h": 24, "72h": 72, "240h": 240, "720h": 720},
  "grid_res_deg": 0.1,
  "idw_power": 2.0,
  "layers_generated": ["accum_20260201_1400_24h.tif", "accum_20260130_1400_72h.tif"],
  "accum_input_pattern": "{station}_{yyyymmdd}_{hhmm}_{horizon}.csv",
  "interp_output_pattern": "accum_{yyyymmdd}_{hhmm}_{horizon}.tif"
}
```

Exemplo de `basin_stats.json`:

```json
{
  "step": "interpolate",
  "run_id": "20260202T140000",
  "reference_time": "2026-02-02T14:00:00",
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
- Data de referência do run: `config/run.yaml` → `run.reference_time`.
- Bacias com análise detalhada: `config/basins.yaml` → `basins.detailed_stats_ids`.
- Parâmetros de interpolação: `config/default.yaml` → `interpolation`.
- Limites de nível (cm): `config/qc.yaml` → `qc.level` (`min_cm`, `max_cm`, `max_step_cm_h`).
