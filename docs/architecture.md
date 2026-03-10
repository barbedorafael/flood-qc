# Arquitetura

## Visao geral

A arquitetura desta base e monorepo, local-first e orientada por arquivos. O sistema assume uma maquina Windows operada remotamente pela equipe, sem dependencia de backend central para o fluxo principal.

Os componentes sao:

- `apps/ops_dashboard`: interface operacional principal em Streamlit.
- `apps/mgb_runner`: encapsulamento do modelo MGB como processo externo.
- `apps/qgis_project`: convencoes para consumo espacial complementar.
- `src/`: logica por dominio, separada entre ingestao, QC, modelo, storage e reporting.
- `sql/`: schemas explicitos de SQLite.

## Decisoes arquiteturais

### SQLite simples

Foi adotado SQLite simples como baseline para reduzir dependencia operacional, facilitar backup/copias e manter o sistema auditavel em maquina local. O desenho preserva a possibilidade de evoluir para SpatiaLite depois, mas sem acoplar a primeira versao a uma stack espacial mais pesada.

### Banco historico + banco por run

O historico persistente fica em `data/history.sqlite` e concentra metadados, observados, flags, edicoes e catalogo de runs. Cada run possui seu proprio arquivo em `data/runs/<run_id>.sqlite`, contendo lineage, inputs, outputs, assets e relatorios associados.

### Rasters e vetores fora do banco

Rasters e vetores nao entram como blob em SQLite. O banco guarda apenas metadados e paths relativos. Isso simplifica o consumo por QGIS, evita inflar os bancos e deixa os artefatos mais portaveis.

### Streamlit como UI principal

O Streamlit foi escolhido para ser a interface operacional principal porque permite navegar rapidamente por runs, revisar flags, sumarizar resultados e centralizar a triagem sem introduzir backend web adicional.

### QGIS como ferramenta complementar

QGIS nao e o centro da arquitetura. Ele entra como cliente sobre GeoPackages e GeoTIFFs produzidos pelo pipeline, principalmente para inspecao espacial, comparacao visual e apoio a edicoes externas quando necessario.

### MGB isolado em um runner proprio

O MGB e Windows-only e tem acoplamentos especificos de executavel, diretoria de trabalho e arquivos de entrada. Por isso ele fica encapsulado em `apps/mgb_runner`, separado da logica de ingestao, QC e reporting.

## Trade-offs

- simplificar infraestrutura agora em vez de maximizar flexibilidade prematura.
- arquivos SQLite e artefatos locais em vez de servicos centralizados.
- contratos claros e stubs em vez de integrar APIs e modelo antes de estabilizar a estrutura.

## Fluxo entre historico, runs e outputs

1. A ingestao coleta dados externos e grava artefatos brutos em `data/interim/`.
2. Series tratadas e aprovadas passam a ser referenciadas a partir de `data/timeseries/` e indexadas no historico.
3. Um run e criado em `data/runs/<run_id>.sqlite`.
4. O run referencia inputs, registra lineage e prepara a execucao do modelo.
5. Outputs, flags e relatorios ficam vinculados ao banco do run.
6. O historico recebe o catalogo do run e, quando apropriado, metadados de publicacao.