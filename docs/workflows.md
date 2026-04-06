# Workflows operacionais

## Ingestao

1. Coletar observados e previsoes de fontes externas.
2. Gravar os arquivos brutos em `data/interim/`.
3. Registrar o historico operacional da coleta em `logs/`.
4. Materializar observados em `observed_series` e `observed_value` no formato long.

## QC automatico

1. Aplicar regras por variavel e por serie.
2. Registrar flags em `qc_flag` sem alterar o dado original.
3. Promover dados entre `raw`, `curated` e `approved` conforme o processo.
4. Liberar os insumos aprovados para a execucao automatica do modelo.

## Execucao do modelo

1. Preparar os arquivos de input necessarios para o MGB a partir dos insumos aprovados.
2. Copiar `apps/mgb_runner/Input` para `C:/mgb-hora/Input`, recriar `C:/mgb-hora/Output` e executar o `.exe` local sem parametros.
3. Espelhar `C:/mgb-hora/Output` de volta para `apps/mgb_runner/Output`.
4. Usar diretamente os binarios `QTUDO_Inercial_Atual.MGB` e `YTUDO.MGB`, junto de `PARHIG.hig` e `MINI.gtp`, como base para visualizacao, triagem e selecao do subset operacional.

## QC de outputs

1. Validar coerencia minima dos resultados.
2. Registrar flags e comparacoes com observados.
3. Marcar outputs/celulas/series para composicao do run operacional.

## Montagem do run

1. Criar um novo `data/runs/<run_id>.sqlite` para o run automatico ou derivado.
2. Registrar o cabecalho em `run`, com `parent_run_id` quando houver derivacao.
3. Copiar os inputs aprovados efetivamente usados para `run_input_series` e `run_input_value`.
4. Registrar assets usados no run em `run_asset`, incluindo o artefato completo de output e outros arquivos auxiliares relevantes.
5. Materializar no run apenas o subset operacional dos outputs do modelo em `mgb_output_series` e `mgb_output_value`.
6. Registrar derivados operacionais em `derived_series` e `derived_value`.

## Revisao manual

1. Operador inspeciona flags, series e o subset do run apoiado pelo dashboard e pelos binarios completos do MGB.
2. Ajustes sao registrados como `manual_edit` em um run derivado quando houver intervencao manual.
3. Nenhum run automatico e editado em lugar.

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

- run automatico: gerado apos QC automatico, execucao do modelo e materializacao do subset operacional do dia;
- run revisado: novo arquivo SQLite derivado, com `parent_run_id` apontando para o automatico ja executado.
