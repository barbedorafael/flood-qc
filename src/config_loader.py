from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "PyYAML não encontrado. Instale com: pip install pyyaml"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_TIMEZONE = ZoneInfo("America/Sao_Paulo")

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "run": {
        "mode": "operational",
        "reference_time": None,
        "event_name": None,
    },
    "ingest": {
        "ana_base_url": "http://telemetriaws1.ana.gov.br/serviceana.asmx/DadosHidrometeorologicos",
        "request_days": 3,
        "timeout_seconds": 15,
    },
    "windows": {
        "forecast_days": [1, 3, 10, 30],
        "accum_hours": [24, 72, 240, 720],
    },
    "interpolation": {
        "method": "idw",
        "grid_res_deg": 0.1,
        "power": 2.0,
    },
    "paths": {
        "station_files": [
            "data/estacoes_nivel.csv",
            "data/estacoes_pluv.csv",
        ],
        "telemetry_dir": "data/telemetria",
        "accum_dir": "data/accum",
        "interp_dir": "data/interp",
    },
    "outputs": {
        "reports_base_dir": "data/reports",
        "write_summary_json": True,
        "write_station_json": True,
        "write_basin_json": True,
        "write_config_snapshot": True,
    },
    "basins": {
        "selected_ids": [],
        "detailed_stats_ids": [],
        "names": {},
    },
    "qc": {
        "rain": {"min_mm_h": 0.0, "max_mm_h": 120.0},
        "level": {"min_m": -2.0, "max_m": 20.0, "max_step_m_h": 1.5},
        "flow": {"min_m3s": 0.0, "max_m3s": 50000.0},
        "gap_fill": {"max_gap_hours": 3},
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config inválido em {path}: esperado um objeto YAML no topo.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(base_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_reference_time(value: Any) -> datetime:
    if value in (None, "", "now"):
        now = datetime.now(DEFAULT_TIMEZONE)
        return now.replace(minute=0, second=0, microsecond=0, tzinfo=None)

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)

    if dt.tzinfo is not None:
        dt = dt.astimezone(DEFAULT_TIMEZONE).replace(tzinfo=None)
    return dt


def _build_accum_horizons(hours: list[Any]) -> dict[str, int]:
    horizons: dict[str, int] = {}
    for raw in hours:
        try:
            h = int(raw)
        except (TypeError, ValueError):
            continue
        if h <= 0:
            continue
        horizons[f"{h}h"] = h
    if not horizons:
        horizons = {"24h": 24, "72h": 72, "240h": 240, "720h": 720}
    return horizons


def resolve_path(raw_path: str, *, root: Path = REPO_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path)


def resolve_paths(raw_paths: list[str], *, root: Path = REPO_ROOT) -> list[Path]:
    return [resolve_path(p, root=root) for p in raw_paths]


def get_report_dir(config: dict[str, Any], *, root: Path = REPO_ROOT) -> Path:
    base = config.get("outputs", {}).get("reports_base_dir", "data/reports")
    return resolve_path(base, root=root) / config["runtime"]["run_id"]


def get_runtime_reference_time(config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    value = runtime.get("reference_time")
    if value not in (None, ""):
        return str(value)
    legacy_value = runtime.get("reference_time_utc")
    if legacy_value not in (None, ""):
        return str(legacy_value)
    raise KeyError("runtime.reference_time não encontrado.")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Tipo não serializável em JSON: {type(obj)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_runtime_config(
    *,
    config_dir: Path | None = None,
    event_name: str | None = None,
) -> dict[str, Any]:
    config_dir = config_dir or DEFAULT_CONFIG_DIR

    config = deepcopy(DEFAULT_CONFIG)
    for name in ("default.yaml", "run.yaml", "basins.yaml", "qc.yaml"):
        config = deep_merge(config, _load_yaml(config_dir / name))

    configured_event = config.get("run", {}).get("event_name")
    mode = str(config.get("run", {}).get("mode", "operational")).strip().lower()
    effective_event = event_name or configured_event

    should_load_event = bool(effective_event) and (mode == "event_replay" or event_name is not None)
    if should_load_event:
        event_path = config_dir / "events" / f"{effective_event}.yaml"
        config = deep_merge(config, _load_yaml(event_path))
        config.setdefault("run", {})["event_name"] = effective_event

    run_config = config.setdefault("run", {})
    legacy_reference_time = run_config.pop("reference_time_utc", None)
    if run_config.get("reference_time") in (None, "") and legacy_reference_time not in (None, ""):
        run_config["reference_time"] = legacy_reference_time

    reference_dt = _parse_reference_time(run_config.get("reference_time"))
    run_config["reference_time"] = reference_dt.isoformat()
    run_id = reference_dt.strftime("%Y%m%dT%H%M%S")
    if config.get("run", {}).get("event_name"):
        run_id = f"{run_id}_{config['run']['event_name']}"

    config["runtime"] = {
        "reference_time": reference_dt.isoformat(),
        "run_id": run_id,
        "loaded_at": datetime.now(DEFAULT_TIMEZONE).replace(microsecond=0, tzinfo=None).isoformat(),
        "config_dir": str(config_dir),
        "accum_horizons_h": _build_accum_horizons(config.get("windows", {}).get("accum_hours", [])),
    }

    if config.get("outputs", {}).get("write_config_snapshot", True):
        report_dir = get_report_dir(config)
        write_json(report_dir / "config_snapshot.json", config)

    return config
