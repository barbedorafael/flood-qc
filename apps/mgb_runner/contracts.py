from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MgbRunRequest:
    run_db: str
    executable_path: str
    working_directory: str


@dataclass(slots=True)
class MgbRunResponse:
    status: str
    summary: str