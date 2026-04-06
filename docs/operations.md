# Operacao e convencoes

## Setup local

1. Criar ambiente virtual com Python 3.10+.
2. Instalar dependencias com `pip install -e .[dev]` e, quando necessario, `.[ui]`.
3. Ajustar `config/default.yaml` com os defaults operacionais do ingest.
4. Usar `config/custom.yaml` apenas para overrides locais.
5. Rodar `python src/ingest/fetch_observed_ana.py` sem argumentos.

## Config do ingest

O ingest novo le exclusivamente:

- `config/default.yaml`
- `config/custom.yaml`

Os paths canonicos e a URL base da ANA ficam fixos em codigo.

## Convencoes de nomes

- `run_id`: preferencialmente `YYYYMMDDTHHMMSS`.
- `history.sqlite`: banco historico unico.
- `data/runs/<run_id>.sqlite`: um arquivo por run.
- assets externos com paths relativos sempre que possivel.

## Raw, curated, approved

- `raw`: dado ingerido sem revisao completa.
- `curated`: dado tratado por regras automaticas ou pre-processamento.
- `approved`: dado liberado para uso operacional e para alimentar a execucao automatica do modelo.

## Artefato completo vs run

- `apps/mgb_runner/Output/QTUDO_Inercial_Atual.MGB` e `apps/mgb_runner/Output/YTUDO.MGB`: artefatos completos dos outputs do modelo usados pelo dashboard.
- `apps/mgb_runner/Input/PARHIG.hig` e `apps/mgb_runner/Input/MINI.gtp`: metadados e mapeamento para leitura direta desses binarios.
- `data/runs/<run_id>.sqlite`: contexto operacional do run, com subset dos outputs realmente carregados para analise, rastreio e revisao.

## Como armazenar paths de raster

Guardar path relativo ao repositorio ou ao diretoria do run no banco. Nao armazenar o arquivo raster como blob em SQLite.

## Como evitar edicao destrutiva

- nao sobrescrever dados de origem;
- registrar flags e edicoes em tabelas append-only;
- criar novo run manual derivado em vez de editar um automatico ja executado.

## Provenance e audit trail

Toda transformacao relevante deve registrar:

- origem do dado ou asset;
- momento da alteracao;
- operador ou servico responsavel;
- motivo da alteracao;
- relacao com o run impactado.
