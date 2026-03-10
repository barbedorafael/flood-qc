from __future__ import annotations

from common.models import RasterAsset, RunMetadata


def collect_forecast_grids(run: RunMetadata) -> list[RasterAsset]:
    """Coleta e prepara previsoes meteorologicas em grade.

    TODO:
    - implementar download ou leitura local das grades;
    - registrar transformacoes espaciais aplicadas;
    - gravar referencias em `data/interim/` e `data/spatial/`.
    """
    raise NotImplementedError("Coleta de grades de previsao ainda nao implementada.")