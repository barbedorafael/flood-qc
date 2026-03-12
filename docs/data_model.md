# Modelo conceitual de dados

## Entidades principais

### Historico (`data/history.sqlite`)

- `provider`: catalogo de origens como ANA, INMET e providers de grade.
- `variable`: catalogo de variaveis e unidade padrao.
- `station`: cadastro canonico unificado de estacoes, uma linha por `provider_code + station_code`.
- `asset`: ponteiro generico para arquivos externos como CSVs, GRIB2, rasters e GPKGs.
- `ingest_batch`: agrupador de uma coleta ou importacao.
- `observed_series`: definicao canonica de uma serie observada tratada.
- `observed_value`: valores temporais em formato long.
- `qc_flag`: flags de qualidade sem sobrescrever o dado original.
- `manual_edit`: trilha append-only de alteracoes.
- `run_catalog`: indice de runs existentes/publicados.

### Run (`data/runs/<run_id>.sqlite`)

- `run`: cabecalho do run com `parent_run_id` quando derivado.
- `run_input_series`: copia local das series usadas no run.
- `run_input_value`: copia local dos valores usados no run.
- `run_asset`: assets associados ao run, incluindo forecast original e editado.
- `derived_series`: series derivadas dentro do run.
- `derived_value`: valores derivados com suporte a janela temporal e horizonte.
- `model_execution`: execucao do modelo e metadados do setup espacial externo.
- `mgb_output_series`: normalizacao da malha completa do MGB por variavel, celula e `prev_flag`.
- `mgb_output_value`: serie temporal dos outputs do MGB em formato long.
- `qc_flag`: flags locais ao run.
- `manual_edit`: ajustes manuais locais ao run.
- `report_artifact`: produtos e relatorios gerados a partir do run.

## Separacao entre historico e execucao

### Historico

Guarda o estado consolidado e reusavel do sistema:

- cadastro unificado de estacoes;
- inventario local consolidado por provider;
- observados em formato long;
- assets externos estaveis e arquivos brutos/originais;
- batches de ingestao;
- flags persistentes e trilha de edicao;
- catalogo de runs.

### Run

Guarda o contexto fechado de uma execucao especifica:

- cabecalho do run e relacao com o run pai;
- copia local dos inputs realmente usados;
- assets ligados ao run;
- produtos derivados operacionais;
- execucao do modelo;
- outputs completos do MGB normalizados no proprio SQLite;
- relatorios e flags locais.

## Convencoes de representacao

### Estacoes

No schema novo:

- `station` guarda uma linha por estacao operacional local;
- `provider_code + station_code` formam a identidade logica da estacao;
- `station_code` ANA permanece sem zero a esquerda no catalogo canonico.

O inventario inicial fica em `data/interim/history_station_inventory.csv`. O bootstrap do banco historico calcula `station_uid` como base do provider (`1000000000` para ANA, `2000000000` para INMET) somada ao `station_code`, convertendo letras para numeros (`A=1`, `B=2`, etc.), e carrega o inventario no banco.

### Series observadas

Observados entram em formato long:

- cada variavel observavel vira uma linha em `observed_series`;
- cada ponto temporal vira uma linha em `observed_value`.

### Derivados operacionais

Derivados operacionais sao representados por:

- `derived_series` define a serie derivada;
- `derived_value` guarda `window_start`, `window_end`, `horizon_h` e `value`.

### Assets de grade e raster

Grades e rasters continuam fora do SQLite. O banco guarda apenas metadados e paths relativos.

- no historico, o GRIB2 original entra em `asset`;
- no run, `run_asset` aponta para o original e para o editado quando existir;
- rasters interpolados entram como `run_asset` com metadados em `metadata_json`.

### Setup espacial do MGB

O cadastro espacial do MGB nao entra nem no historico nem no run. Ele fica em um GPKG externo em `data/spatial/<mgb_setup>.gpkg`.

O banco do run guarda apenas a referencia desse setup em `model_execution.setup_gpkg_path`.

### Outputs do MGB

Os outputs completos do MGB entram normalizados:

- `mgb_output_series`: uma serie por `variable_code + cell_id + prev_flag`;
- `mgb_output_value`: um valor por `series_id + dt`.

Isso deixa o run auto-suficiente e facilita consultas por celula, variavel e tempo.

## Schemas implementados

Os schemas canonicos ficam em:

- `sql/history_schema.sql`
- `sql/run_schema.sql`
