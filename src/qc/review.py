from __future__ import annotations

from common.models import ManualEdit, RunMetadata


def register_manual_review(run: RunMetadata, edits: list[ManualEdit]) -> None:
    """Registra revisoes manuais sem alterar a origem em lugar.

    TODO:
    - persistir o log append-only no banco do run;
    - opcionalmente propagar aprovacoes para o historico;
    - validar autoria, motivo e timestamps.
    """
    raise NotImplementedError("Revisao manual ainda nao implementada.")