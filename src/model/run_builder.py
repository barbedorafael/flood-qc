from __future__ import annotations

from common.models import ModelInput, RunMetadata


def assemble_model_inputs(run: RunMetadata) -> list[ModelInput]:
    """Materializa o run operacional a partir dos insumos aprovados e do subset selecionado.

    TODO:
    - consolidar series aprovadas efetivamente usadas na execucao;
    - apontar rasters, vetores e artefatos auxiliares necessarios;
    - selecionar o subset operacional dos outputs completos do modelo;
    - validar completude antes da revisao/publicacao do run.
    """
    raise NotImplementedError("Montagem de inputs do modelo ainda nao implementada.")
