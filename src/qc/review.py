from __future__ import annotations

from common.models import ManualEdit, RunMetadata


def register_manual_review(run: RunMetadata, edits: list[ManualEdit]) -> None:
    """Registra revisoes manuais sobre um run ja executado e materializado.

    TODO:
    - persistir o log append-only no banco do run derivado;
    - impedir alteracao em lugar do run automatico de origem;
    - opcionalmente propagar aprovacoes para o historico;
    - validar autoria, motivo e timestamps.
    """
    raise NotImplementedError("Revisao manual ainda nao implementada.")
