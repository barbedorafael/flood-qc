# Layout operacional de `data/`

Estrutura alvo desta base:

- `data/history.sqlite`
  Banco historico persistente.
- `data/interim/`
  Arquivos coletados de APIs e outros artefatos intermediarios de ingestao.
- `data/timeseries/`
  Series temporais tratadas e prontas para uso operacional.
- `data/spatial/`
  Camadas espaciais tratadas, estaveis e reutilizaveis.
- `data/runs/`
  Um arquivo SQLite por run.

Observacao:

O repositorio ja contem diretorios e arquivos legados fora dessa convencao. Eles nao foram apagados nesta etapa por seguranca de dados e para preservar historico local.