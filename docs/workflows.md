# Workflows operacionais

## Ingestao

1. Coletar observados e previsoes de fontes externas.
2. Gravar os arquivos brutos em `data/interim/`.
3. Registrar assets e batches de ingestao no historico.
4. Materializar observados em `observed_series` e `observed_value` no formato long.

## QC automatico

1. Aplicar regras por variavel e por serie.
2. Registrar flags em `qc_flag` sem alterar o dado original.
3. Promover dados entre `raw`, `curated` e `approved` conforme o processo.

## Revisao manual

1. Operador inspeciona flags e series.
2. Ajustes sao registrados como `manual_edit`.
3. Nenhum run automatico e editado em lugar.

## Montagem do run

1. Criar um novo `data/runs/<run_id>.sqlite`.
2. Registrar o cabecalho em `run` com `parent_run_id` quando houver derivacao.
3. Copiar os inputs aprovados para `run_input_series` e `run_input_value`.
4. Registrar assets usados no run em `run_asset`, incluindo forecast original e editado quando aplicavel.
5. Registrar derivados operacionais em `derived_series` e `derived_value`.

## Execucao do modelo

1. O runner do MGB le o banco do run.
2. Registra `model_execution`, incluindo `setup_gpkg_path` para o catalogo espacial externo.
3. Em fase futura, executa o modelo e grava a malha completa em `mgb_output_series` e `mgb_output_value`.

## QC de outputs

1. Validar coerencia minima dos resultados.
2. Registrar flags e comparacoes com observados.
3. Marcar outputs para revisao ou publicacao.

## Geracao de relatorio

1. Consolidar sumarios e produtos do run.
2. Registrar relatorios em `report_artifact`.
3. Publicar referencias no `run_catalog` do historico quando aplicavel.

## Dia normal vs dia de evento

### Dia normal

- run automatico diario;
- pouca ou nenhuma intervencao manual;
- foco em monitoramento e verificacao rapida.

### Dia de evento

- mais de um run ao longo do dia;
- criacao de runs manuais derivados;
- maior uso do dashboard e do QGIS para triagem e comparacao.

## Run automatico vs run revisado

- run automatico: gerado pela rotina padrao do dia;
- run revisado: novo arquivo SQLite derivado, com `parent_run_id` apontando para o automatico.