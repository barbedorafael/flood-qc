from __future__ import annotations

from pathlib import Path

from common.models import CommandPlan, ModelOutput, RunMetadata


def prepare_mgb_execution(run: RunMetadata, executable_path: str, workdir: str) -> CommandPlan:
    """Prepara o comando do MGB sem executa-lo.

    TODO:
    - validar existencia do executavel e do diretorio de trabalho;
    - materializar inputs do run em layout compativel com o MGB;
    - registrar o plano no banco do run.
    """
    return CommandPlan(
        command=[executable_path, "--run-id", run.run_id],
        working_directory=str(Path(workdir)),
        metadata={"run_id": run.run_id, "reference_time": run.reference_time},
    )


def execute_mgb_plan(plan: CommandPlan, *, dry_run: bool = False) -> ModelOutput:
    """Executa ou simula a execucao do plano do MGB.

    Nesta etapa, apenas dry-run e suportado para documentar o contrato.
    """
    if dry_run:
        return ModelOutput(
            output_name="mgb_dry_run",
            description="Dry-run do modelo MGB; nenhuma simulacao foi executada.",
            asset_refs=[],
        )
    raise NotImplementedError("Execucao real do MGB ainda nao implementada.")