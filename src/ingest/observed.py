from __future__ import annotations

from common.models import RunMetadata, TimeSeriesRecord


def collect_observed_timeseries(run: RunMetadata) -> list[TimeSeriesRecord]:
    """Coleta series observadas para o run informado.

    TODO:
    - implementar conectores reais de ANA, INMET e outras fontes;
    - persistir os arquivos brutos em `data/interim/`;
    - registrar metadados e lineage no banco historico.
    """
    raise NotImplementedError("Coleta de observados ainda nao implementada.")