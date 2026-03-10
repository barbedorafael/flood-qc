from __future__ import annotations

from common.models import ReportArtifact, RunMetadata


def build_run_reports(run: RunMetadata) -> list[ReportArtifact]:
    """Gera os artefatos de relatorio do run.

    TODO:
    - consolidar sumarios operacionais;
    - gerar saídas em formatos acordados;
    - registrar relatorios no banco do run.
    """
    raise NotImplementedError("Geracao de relatorios ainda nao implementada.")