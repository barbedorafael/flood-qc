from __future__ import annotations

from common.models import ModelInput, RunMetadata


def assemble_model_inputs(run: RunMetadata) -> list[ModelInput]:
    """Monta os inputs necessarios para um run do modelo.

    TODO:
    - consolidar series aprovadas;
    - apontar rasters e vetores necessarios;
    - validar completude antes da execucao do modelo.
    """
    raise NotImplementedError("Montagem de inputs do modelo ainda nao implementada.")