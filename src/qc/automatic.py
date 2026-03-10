from __future__ import annotations

from common.models import QcFlag, RunMetadata


def apply_automatic_qc(run: RunMetadata) -> list[QcFlag]:
    """Executa regras automaticas de QC para o contexto do run.

    TODO:
    - implementar verificacoes por variavel;
    - registrar flags no historico e no run;
    - diferenciar severidades e criterios de bloqueio.
    """
    raise NotImplementedError("QC automatico ainda nao implementado.")