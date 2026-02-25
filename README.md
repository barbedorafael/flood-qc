# Flood QC - Rio Grande do Sul

Ferramenta em Python para **QC e preparo de dados de entrada/saída** do fluxo de previsão de cheias no RS.

Arquitetura intencionalmente enxuta:
- scripts em `src/`;
- configurações em `config/`;
- saídas em `data/`.

## O que já está implementado

- Inventário mínimo de estações (`src/station_inventory.py`).
- Coleta de telemetria ANA (chuva/nível/vazão) (`src/fetch_data.py`).
- Cálculo de chuva acumulada em CSV consolidado (`src/accumulate.py`).
- Interpolação IDW a partir do CSV consolidado (`src/interpolate.py`).
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
  - `accum/acc_*.csv`
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

Processa a telemetria de chuva e gera um único CSV consolidado para todas as estações.

Horizontes atuais:
- 24h, 72h (3 dias), 240h (10 dias), 720h (30 dias)

Saídas:
- `data/accum/acc_{yyyymmdd}_{hhmm}.csv`

Observações:
- O script constrói um dataframe único com colunas:
  `station_id`, `dt_start`, `dt_end`, `horizon_h`, `rain_acc_mm`.
- Para cada estação, `dt_end` é a última data/hora disponível da estação.
- `dt_start = dt_end - horizon_h`.
- `rain_acc_mm` é a soma no intervalo `(dt_start, dt_end]`, arredondada para 1 casa decimal.
- Horizontes vêm da configuração (`windows.accum_hours`).
- `yyyymmdd/hhmm` no nome do arquivo representam `max(dt_end)` do dataframe consolidado.

### 4) `src/interpolate.py`

Consome o CSV consolidado de acumulados e gera rasters interpolados por IDW.

Saídas:
- `data/interp/accum_{yyyymmdd}_{hhmm}_{horizonte}.tif`

Observações:
- Lê o arquivo no padrão `acc_{yyyymmdd}_{hhmm}.csv` mais recente
  (priorizando `<= run.reference_time`; se não houver, usa o mais recente disponível).
- Filtra o dataframe por `horizon_h` para montar cada camada.
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
- Nome: `acc_{yyyymmdd}_{hhmm}.csv`
- Colunas: `station_id`, `dt_start`, `dt_end`, `horizon_h`, `rain_acc_mm`

### `data/interp/*.tif`
- COG em `EPSG:4326`
- Tags incluem horizonte e horário de referência

## JSONs para relatório

Cada run gera pasta `data/reports/<run_id>/` com:
- `config_snapshot.json`: configuração efetivamente usada no run.
- `fetch_data_summary.json`: status por estação na coleta ANA.
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

Exemplo de `interpolate_summary.json`:

```json
{
  "step": "interpolate",
  "run_id": "20260202T140000",
  "reference_time": "2026-02-02T14:00:00",
  "runtime_reference_time": "2026-02-02T14:00:00",
  "accum_horizons_h": {"24h": 24, "72h": 72, "240h": 240, "720h": 720},
  "grid_res_deg": 0.1,
  "idw_power": 2.0,
  "layers_generated": ["accum_20260202_1400_24h.tif", "accum_20260202_1400_72h.tif"],
  "used_accum_files_by_horizon": {"24h": ["acc_20260202_1400.csv"], "72h": ["acc_20260202_1400.csv"]},
  "accum_input_file": "acc_20260202_1400.csv",
  "accum_input_pattern": "acc_{yyyymmdd}_{hhmm}.csv",
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
