# Operacao e convencoes

## Setup local

1. Criar ambiente virtual com Python 3.10+.
2. Instalar dependencias com `pip install -e .[dev]` e, quando necessario, `.[ui]`.
3. Copiar `config/system.example.yaml` para `config/system.yaml` quando houver paths locais a configurar.
4. Ajustar `.env` apenas para overrides de maquina.

## Convencoes de nomes

- `run_id`: preferencialmente `YYYYMMDDTHHMMSS`.
- `history.sqlite`: banco historico unico.
- `data/runs/<run_id>.sqlite`: um arquivo por run.
- assets externos com paths relativos sempre que possivel.

## Raw, curated, approved

- `raw`: dado ingerido sem revisao completa.
- `curated`: dado tratado por regras automaticas ou pre-processamento.
- `approved`: dado liberado para uso operacional ou para composicao de run.

## Como armazenar paths de raster

Guardar path relativo ao repositorio ou ao diretoria do run no banco. Nao armazenar o arquivo raster como blob em SQLite.

## Como evitar edicao destrutiva

- nao sobrescrever dados de origem;
- registrar flags e edicoes em tabelas append-only;
- criar novo run manual em vez de editar um automatico existente.

## Provenance e audit trail

Toda transformacao relevante deve registrar:

- origem do dado ou asset;
- momento da alteracao;
- operador ou servico responsavel;
- motivo da alteracao;
- relacao com o run impactado.

## Observacao sobre dados legados

O repositorio ainda contem diretorios historicos fora do layout alvo. Eles foram mantidos nesta etapa para evitar perda de dados locais e facilitar migracao gradual do acervo existente.