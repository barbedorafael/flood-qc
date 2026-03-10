from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.models import RunMetadata
from common.settings import load_settings
from model.mgb_execution import execute_mgb_plan, prepare_mgb_execution

from contracts import MgbRunRequest, MgbRunResponse


def build_request(run_db: Path) -> MgbRunRequest:
    settings = load_settings()
    mgb_settings = settings.get("mgb", {})
    return MgbRunRequest(
        run_db=run_db.as_posix(),
        executable_path=str(mgb_settings.get("executable_path", "MGB.exe")),
        working_directory=str(mgb_settings.get("workdir", run_db.parent)),
    )


def run(request: MgbRunRequest, *, dry_run: bool = False) -> MgbRunResponse:
    run_id = Path(request.run_db).stem
    metadata = RunMetadata(run_id=run_id, reference_time=run_id)
    plan = prepare_mgb_execution(metadata, request.executable_path, request.working_directory)
    result = execute_mgb_plan(plan, dry_run=dry_run)
    summary = json.dumps(
        {
            "command": plan.command,
            "working_directory": plan.working_directory,
            "description": result.description,
            "run_db": request.run_db,
        },
        indent=2,
    )
    return MgbRunResponse(status="dry_run" if dry_run else "pending", summary=summary)