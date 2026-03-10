from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml

from common.paths import CONFIG_DIR


CONFIG_FILES = ("default.yaml", "run.yaml", "qc.yaml", "basins.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config invalido em {path}: esperado um objeto YAML.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_env_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    env_map = {
        "FLOODQC_HISTORY_DB": ("paths", "history_db"),
        "FLOODQC_RUNS_DIR": ("paths", "runs_dir"),
        "FLOODQC_INTERIM_DIR": ("paths", "interim_dir"),
        "FLOODQC_TIMESERIES_DIR": ("paths", "timeseries_dir"),
        "FLOODQC_SPATIAL_DIR": ("paths", "spatial_dir"),
        "FLOODQC_LOG_DIR": ("paths", "logs_dir"),
        "FLOODQC_MGB_EXECUTABLE": ("mgb", "executable_path"),
        "FLOODQC_MGB_WORKDIR": ("mgb", "workdir"),
    }
    for env_name, keys in env_map.items():
        value = os.getenv(env_name)
        if not value:
            continue
        scope = settings.setdefault(keys[0], {})
        scope[keys[1]] = value
    return settings


def load_settings(config_dir: Path | None = None) -> dict[str, Any]:
    config_dir = config_dir or CONFIG_DIR
    settings: dict[str, Any] = {}
    for file_name in CONFIG_FILES:
        settings = _deep_merge(settings, _load_yaml(config_dir / file_name))

    system_path = config_dir / "system.yaml"
    if system_path.exists():
        settings = _deep_merge(settings, _load_yaml(system_path))

    return _apply_env_overrides(settings)


def read_logging_config(config_dir: Path | None = None) -> dict[str, Any]:
    config_dir = config_dir or CONFIG_DIR
    return _load_yaml(config_dir / "logging.yaml")