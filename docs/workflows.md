# Workflows operacionais

## Ingestao

1. Coletar observados e previsoes de fontes externas.
2. Gravar os arquivos brutos em `data/interim/`.
3. Registrar origem e metadados para futura auditoria.

## QC automatico

1. Aplicar regras por variavel e por serie.
2. Registrar flags sem alterar o dado original.
3. Marcar o estado dos dados como `raw`, `curated` ou `approved`.

## Revisao manual

1. Operador inspeciona flags e series.
2. Ajustes sao registrados como `manual_edit`.
3. Nenhum run automatico e editado em lugar.

## Montagem do run

1. Criar um novo `run.sqlite` em `data/runs/`.
2. Registrar lineage e referencia temporal.
3. Materializar os inputs aprovados do modelo.

## Execucao do modelo

1. O runner do MGB le o banco do run.
2. Prepara um plano de execucao com executavel e diretorio de trabalho.
3. Em fase futura, executa o modelo e registra outputs e metadados.

## QC de outputs

1. Validar coerencia minima dos resultados.
2. Registrar flags e comparacoes com observados.
3. Marcar outputs para revisao ou publicacao.

## Geracao de relatorio

1. Consolidar sumarios e produtos do run.
2. Registrar relatorios no `run.sqlite`.
3. Publicar referencias no catalogo historico quando aplicavel.

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