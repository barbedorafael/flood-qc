# Arquitetura

## Visao geral

A arquitetura desta base e monorepo, local-first e orientada por arquivos. O sistema assume uma maquina Windows operada remotamente pela equipe, sem dependencia de backend central para o fluxo principal.

Os componentes sao:

- `apps/ops_dashboard`: interface operacional principal em Streamlit.
- `apps/mgb_runner`: artefatos locais do MGB (`Input`, `Output`, `.exe`).
- `apps/qgis_project`: convencoes para consumo espacial complementar.
- `src/`: logica por dominio, separada entre ingestao, QC, modelo, storage e reporting.
- `sql/`: schemas explicitos de SQLite.

## Decisoes arquiteturais

### SQLite simples

Foi adotado SQLite simples como baseline para reduzir dependencia operacional, facilitar backup/copias e manter o sistema auditavel em maquina local. O desenho preserva a possibilidade de evoluir para SpatiaLite depois, mas sem acoplar a primeira versao a uma stack espacial mais pesada.

### Banco historico + banco por run

O historico persistente fica em `data/history.sqlite` e concentra metadados, observados, flags, edicoes, assets externos e catalogo de runs. Cada run possui seu proprio arquivo em `data/runs/<run_id>.sqlite`, contendo copia dos inputs usados, derivados operacionais, subset operacional dos outputs do modelo, assets e relatorios associados.

### Observados em formato long

O historico padroniza observados em formato long, com uma serie por variavel e uma tabela de valores temporais. Isso reduz ambiguidade entre providers, facilita QC, aprovacao e extensao para novas variaveis meteorologicas.

### Rasters e vetores fora do banco

Rasters e vetores nao entram como blob em SQLite. O banco guarda apenas metadados e paths relativos. Isso simplifica o consumo por QGIS, evita inflar os bancos e deixa os artefatos mais portaveis.

### Setup espacial do MGB fora dos bancos

O cadastro espacial completo do MGB fica em um GPKG externo em `data/spatial/`. O banco do run guarda apenas a referencia a esse setup, suficiente para ligar celulas e outputs do modelo sem normalizar geometrias em SQLite.

### Artefato completo de outputs fora do run

O output completo do MGB nao precisa morar dentro do banco do run. O contrato operacional passa a considerar os binarios canonicos do runner (`QTUDO_Inercial_Atual.MGB` e `YTUDO.MGB`) como artefatos completos de output para visualizacao e triagem. O run materializa apenas o subset operacional efetivamente usado no ciclo do dia ou em uma derivacao manual.

### Streamlit como UI principal

O Streamlit foi escolhido para ser a interface operacional principal porque permite navegar rapidamente por runs, revisar flags, sumarizar resultados e centralizar a triagem sem introduzir backend web adicional.

### QGIS como ferramenta complementar

QGIS nao e o centro da arquitetura. Ele entra como cliente sobre GeoPackages e GeoTIFFs produzidos pelo pipeline, principalmente para inspecao espacial, comparacao visual e apoio a edicoes externas quando necessario.

### MGB isolado em um runner proprio

O MGB e Windows-only e tem acoplamentos especificos de executavel, diretoria de trabalho e arquivos de entrada. Por isso a logica do runner fica em `src/model/`, enquanto os artefatos operacionais ficam em `apps/mgb_runner`, separados da logica de ingestao, QC e reporting.

## Trade-offs

- simplificar infraestrutura agora em vez de maximizar flexibilidade prematura.
- arquivos SQLite e artefatos locais em vez de servicos centralizados.
- artefato completo do MGB em binarios fora do run e subset operacional dentro do run para manter o fluxo mais enxuto e aderente ao uso real.
- geometria e setup espacial do MGB fora dos bancos para evitar duplicacao e acoplamento espacial desnecessario.

## Fluxo entre historico, runs e outputs

1. A ingestao coleta dados externos e grava artefatos brutos em `data/interim/`.
2. O QC automatico registra flags e promove os dados usados como insumo do modelo.
3. O runner do MGB executa a simulacao com os arquivos de input preparados para o ciclo operacional.
4. O output completo do modelo permanece nos binarios do runner e e lido diretamente pelo dashboard.
5. Um run automatico e materializado em `data/runs/<run_id>.sqlite`, copiando os inputs usados, referenciando os assets relevantes e registrando apenas o subset operacional dos outputs.
6. Quando houver intervencao humana, um run manual derivado e criado com `parent_run_id` apontando para o automatico.
7. O historico recebe o catalogo do run e, quando apropriado, metadados de publicacao.
