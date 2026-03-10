# Modelo conceitual de dados

## Entidades principais

- `station`: cadastro canonico de estacoes.
- `station_alias`: aliases e codigos externos.
- `observed_series`: definicao de series observadas por estacao e variavel.
- `observed_value`: valores observados indexados no tempo.
- `qc_flag`: flags automaticas ou manuais.
- `manual_edit`: trilha append-only de alteracoes.
- `external_asset`: ponteiro para arquivos externos.
- `run_catalog`: indice de runs no historico.
- `run_metadata`: cabecalho do run.
- `run_lineage`: relacao entre run automatico e run manual derivado.
- `input_series` e `input_value`: entradas tabulares do run.
- `output_series` e `output_value`: saidas tabulares do run.
- `asset_ref`: ponteiro para rasters, vetores e arquivos auxiliares do run.
- `report_artifact`: produtos publicados a partir do run.

## Separacao entre historico e execucao

### Historico (`data/history.sqlite`)

Guarda os observados e o estado consolidado do sistema ao longo do tempo:

- cadastro de estacoes;
- series observadas tratadas;
- flags persistentes;
- edicoes registradas;
- indice de runs publicados.

### Run (`data/runs/<run_id>.sqlite`)

Guarda o contexto de uma execucao especifica:

- identificacao e lineage;
- inputs materializados do run;
- outputs tabulares e referencias a assets pesados;
- flags e edicoes locais ao run;
- relatorios associados.

## Convencoes de representacao

### Estacoes

Cada estacao tem um `station_code` canonico, nome, origem e tipo. Aliases externos ficam em tabela propria para evitar perda de rastreabilidade.

### Series temporais

Series menores e operacionais podem ficar em tabelas `*_value`. Assets maiores ou produtos de grade ficam como arquivos externos referenciados em `external_asset` ou `asset_ref`.

### Flags de QC

Flags nao sobrescrevem valores. Elas apontam para um `scope`, um `reference_id`, a regra aplicada e a severidade.

### Edicoes

Toda edicao manual e append-only. O valor antigo e o novo valor sao registrados com motivo, editor e timestamp.

### Assets de grade e raster

Devem ser persistidos como arquivos em `data/spatial/` ou em paths associados a um run. O banco guarda formato, path relativo, estado e descricao.

### Runs

Runs automaticos e manuais compartilham a tabela `run_metadata`. Quando um run manual nasce de um automatico, a relacao fica em `run_lineage`.

### Inputs e outputs do modelo

Inputs tabulares cabem no SQLite do run. Outputs muito grandes podem permanecer externos, desde que cadastrados em `asset_ref`.

### Relatorios

Cada relatorio publicado e um registro em `report_artifact`, com format, path relativo e observacao opcional.

## Schemas iniciais

Os schemas canonicos ficam em:

- `sql/history_schema.sql`
- `sql/run_schema.sql`