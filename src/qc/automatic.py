from __future__ import annotations

from common.models import QcFlag, RunMetadata


def apply_automatic_qc(run: RunMetadata) -> list[QcFlag]:
    """Executa regras automaticas de QC antes da execucao do modelo.

    TODO:
    - implementar verificacoes por variavel;
    - registrar flags no historico e, quando fizer sentido, no contexto operacional do run;
    - promover dados entre `raw`, `curated` e `approved`;
    - diferenciar severidades e criterios de bloqueio para liberar insumos ao modelo.
    """
    raise NotImplementedError("QC automatico ainda nao implementado.")
