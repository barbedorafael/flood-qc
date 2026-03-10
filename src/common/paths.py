from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
SQL_DIR = REPO_ROOT / "sql"


def history_db_path() -> Path:
    return DATA_DIR / "history.sqlite"


def runs_dir() -> Path:
    return DATA_DIR / "runs"


def interim_dir() -> Path:
    return DATA_DIR / "interim"


def timeseries_dir() -> Path:
    return DATA_DIR / "timeseries"


def spatial_dir() -> Path:
    return DATA_DIR / "spatial"


def build_run_db_path(run_id: str) -> Path:
    return runs_dir() / f"{run_id}.sqlite"


def ensure_standard_dirs() -> None:
    for path in (interim_dir(), timeseries_dir(), spatial_dir(), runs_dir()):
        path.mkdir(parents=True, exist_ok=True)


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()