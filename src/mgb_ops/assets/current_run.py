"""The mutable current-run artifact and its SQLite persistence contract."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping
from uuid import uuid4

from mgb_ops.edit.forcing import ForecastCorrectionInstruction, validate_instruction
from mgb_ops.qc.precipitation import PrecipitationQCPolicy


_SCHEMA_TABLES = {
    "current_run", "forecast_scenario", "forecast_correction", "observed_replacement",
    "station_exclusion", "current_execution", "published_cache_reference",
}
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_time(value: object, *, name: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO timestamp.") from exc


@dataclass(frozen=True, slots=True)
class ObservedReplacement:
    station_id: str
    observed_at: str
    value: float | None


@dataclass(frozen=True, slots=True)
class StationExclusion:
    station_id: str
    start_time: str
    end_time: str


@dataclass(frozen=True, slots=True)
class ForecastScenarioReference:
    scenario_id: str
    position: int
    kind: str
    label: str
    provider_code: str | None = None
    source_asset_id: str | None = None
    source_asset_path: str | None = None
    corrections: tuple[ForecastCorrectionInstruction, ...] = ()


@dataclass(frozen=True, slots=True)
class CurrentExecution:
    status: str = "never"
    execution_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CurrentRunArtifact:
    reference_time: str
    forecast_end_exclusive: str
    timestep_hours: int
    mgb_settings: dict[str, Any]
    spatial_settings: dict[str, Any]
    interpolation_settings: dict[str, Any]
    review_window: dict[str, Any]
    observed_providers: tuple[str, ...]
    precipitation_qc_settings: dict[str, Any] = field(default_factory=lambda: PrecipitationQCPolicy().as_dict())
    saved_run_id: str | None = None
    saved_run_description: str | None = None
    responsible_person: str | None = None
    reason: str | None = None
    scenarios: tuple[ForecastScenarioReference, ...] = ()
    observed_replacements: tuple[ObservedReplacement, ...] = ()
    station_exclusions: tuple[StationExclusion, ...] = ()
    execution: CurrentExecution = field(default_factory=CurrentExecution)
    published_caches: Mapping[str, str] = field(default_factory=dict)

    def validate(self, *, require_reason: bool = False) -> "CurrentRunArtifact":
        reference = _parse_time(self.reference_time, name="reference_time")
        forecast_end = _parse_time(self.forecast_end_exclusive, name="forecast_end_exclusive")
        if not reference < forecast_end:
            raise ValueError("Artifact timing must satisfy reference_time < forecast_end_exclusive.")
        if not isinstance(self.timestep_hours, int) or isinstance(self.timestep_hours, bool) or self.timestep_hours < 1:
            raise ValueError("timestep_hours must be >= 1.")
        if not self.observed_providers or any(not str(value).strip() for value in self.observed_providers):
            raise ValueError("Current artifact requires at least one observed provider.")
        PrecipitationQCPolicy.from_mapping(self.precipitation_qc_settings)
        if require_reason and (not str(self.responsible_person or "").strip() or not str(self.reason or "").strip()):
            raise ValueError("responsible_person and reason are required for operational updates.")
        if (self.saved_run_id is None) != (self.saved_run_description is None):
            raise ValueError("Saved-run ID and description must either both be set or both be null.")
        if self.saved_run_id is not None:
            validate_run_id(self.saved_run_id)
            if not str(self.saved_run_description).strip():
                raise ValueError("Saved-run description must be non-empty.")
        ids = [item.scenario_id for item in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Forecast scenario IDs must be unique.")
        if [item.position for item in self.scenarios] != list(range(len(self.scenarios))):
            raise ValueError("Forecast scenario positions must be contiguous and start at zero.")
        for scenario in self.scenarios:
            if scenario.kind not in {"zero", "raw", "corrected"}:
                raise ValueError(f"Unsupported scenario kind: {scenario.kind!r}.")
            if scenario.kind == "zero" and scenario.source_asset_id is not None:
                raise ValueError("Zero scenario cannot reference a forecast asset.")
            if scenario.kind != "zero" and not scenario.source_asset_id:
                raise ValueError(f"{scenario.scenario_id}: source_asset_id is required.")
            for correction in scenario.corrections:
                if correction.t1_step <= correction.t0_step:
                    raise ValueError("Correction window must satisfy t0_step < t1_step.")
                validate_instruction(correction)
                if correction.asset_id != scenario.source_asset_id:
                    raise ValueError(f"{scenario.scenario_id}: correction asset differs from scenario asset.")
        review_start = review_end = None
        if self.review_window:
            if "start" not in self.review_window or "end" not in self.review_window:
                raise ValueError("review_window requires start and end timestamps.")
            review_start = _parse_time(self.review_window["start"], name="review_window.start")
            review_end = _parse_time(self.review_window["end"], name="review_window.end")
            if review_end <= review_start:
                raise ValueError("review_window must satisfy start < end for (start, end].")
        replacement_keys: list[tuple[str, datetime]] = []
        for item in self.observed_replacements:
            if not str(item.station_id).strip():
                raise ValueError("Observed replacements require a station_id.")
            observed_at = _parse_time(item.observed_at, name="observed_replacement.observed_at")
            if observed_at.minute or observed_at.second or observed_at.microsecond or observed_at.hour % self.timestep_hours:
                raise ValueError("Observed replacement timestamps must align to the normalized timestep.")
            if review_start is not None and not (review_start < observed_at <= review_end):
                raise ValueError("Observed replacements must fall within the selected (start, end] review window.")
            if item.value is not None and (isinstance(item.value, bool) or not isinstance(item.value, (int, float)) or not math.isfinite(float(item.value)) or float(item.value) < 0):
                raise ValueError("Observed replacement values must be finite non-negative numbers or NULL.")
            replacement_keys.append((str(item.station_id), observed_at))
        if len(replacement_keys) != len(set(replacement_keys)):
            raise ValueError("Observed replacements must be unique per station and timestamp.")
        exclusion_keys: list[tuple[str, datetime, datetime]] = []
        for item in self.station_exclusions:
            start = _parse_time(item.start_time, name="station_exclusion.start_time")
            end = _parse_time(item.end_time, name="station_exclusion.end_time")
            if not str(item.station_id).strip() or end <= start:
                raise ValueError("Station exclusions must use a station and a non-empty (start, end] window.")
            if review_start is None or start != review_start or end != review_end:
                raise ValueError("Station exclusions must exactly match the selected review window.")
            exclusion_keys.append((str(item.station_id), start, end))
        if len(exclusion_keys) != len(set(exclusion_keys)):
            raise ValueError("Station exclusions must be unique.")
        return self


@dataclass(frozen=True, slots=True)
class SavedRunArtifact:
    run_id: str
    description: str
    path: Path


def validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not _SAFE_RUN_ID.fullmatch(value) or value in {".", ".."} or ".." in value:
        raise ValueError("run_id must be path-safe and contain only letters, numbers, '.', '_' or '-'.")
    return value


class CurrentRunRepository:
    """Typed repository for the one mutable artifact database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._migrate_columns()
        self.validate_database(self.database_path)

    def __enter__(self) -> "CurrentRunRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _migrate_columns(self) -> None:
        found = {str(row[1]) for row in self.connection.execute("PRAGMA table_info(current_run)")}
        additions = {
            "precipitation_qc_settings_json": "TEXT NOT NULL DEFAULT '{\"threshold_mm\":50.0,\"sequence_lower_mm\":0.0,\"sequence_min_length\":5,\"sequence_upper_mm\":1.0}'",
            "saved_run_id": "TEXT",
            "saved_run_description": "TEXT",
        }
        with self.connection:
            for name, definition in additions.items():
                if name not in found:
                    self.connection.execute(f"ALTER TABLE current_run ADD COLUMN {name} {definition}")

    @classmethod
    def validate_database(cls, database_path: Path) -> None:
        path = Path(database_path)
        if not path.exists():
            raise FileNotFoundError(f"Current artifact does not exist: {path}")
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            found = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if found != _SCHEMA_TABLES:
            raise RuntimeError(f"Current artifact has incompatible tables: expected {sorted(_SCHEMA_TABLES)}, found {sorted(found)}.")

    def load(self) -> CurrentRunArtifact:
        row = self.connection.execute("SELECT * FROM current_run WHERE singleton=1").fetchone()
        if row is None:
            raise RuntimeError("Current artifact has not been created.")
        correction_rows = self.connection.execute("SELECT * FROM forecast_correction ORDER BY scenario_id, position").fetchall()
        corrections: dict[str, list[ForecastCorrectionInstruction]] = {}
        for item in correction_rows:
            corrections.setdefault(str(item["scenario_id"]), []).append(ForecastCorrectionInstruction(
                asset_id=self._scenario_asset(str(item["scenario_id"])), t0_step=int(item["t0_step"]), t1_step=int(item["t1_step"]),
                shift_lat=float(item["shift_lat"]), shift_lon=float(item["shift_lon"]), rotation_deg=float(item["rotation_deg"]), multiplication_factor=float(item["multiplication_factor"]),
            ))
        scenarios = tuple(ForecastScenarioReference(
            scenario_id=str(item["scenario_id"]), position=int(item["position"]), kind=str(item["scenario_kind"]), label=str(item["label"]),
            provider_code=item["provider_code"], source_asset_id=item["source_asset_id"], source_asset_path=item["source_asset_path"],
            corrections=tuple(corrections.get(str(item["scenario_id"]), ())),
        ) for item in self.connection.execute("SELECT * FROM forecast_scenario ORDER BY position"))
        execution = self.connection.execute("SELECT * FROM current_execution WHERE singleton=1").fetchone()
        return CurrentRunArtifact(
            reference_time=str(row["reference_time"]), forecast_end_exclusive=str(row["forecast_end_exclusive"]), timestep_hours=int(row["timestep_hours"]),
            mgb_settings=json.loads(row["mgb_settings_json"]), spatial_settings=json.loads(row["spatial_settings_json"]), interpolation_settings=json.loads(row["interpolation_settings_json"]), review_window=json.loads(row["review_window_json"]), observed_providers=tuple(json.loads(row["observed_providers_json"])),
            precipitation_qc_settings=json.loads(row["precipitation_qc_settings_json"]), saved_run_id=row["saved_run_id"], saved_run_description=row["saved_run_description"],
            responsible_person=row["responsible_person"], reason=row["reason"], scenarios=scenarios,
            observed_replacements=tuple(ObservedReplacement(str(x["station_id"]), str(x["observed_at"]), x["value"]) for x in self.connection.execute("SELECT * FROM observed_replacement ORDER BY station_id, observed_at")),
            station_exclusions=tuple(StationExclusion(str(x["station_id"]), str(x["start_time"]), str(x["end_time"])) for x in self.connection.execute("SELECT * FROM station_exclusion ORDER BY station_id, start_time, end_time")),
            execution=CurrentExecution(status=str(execution["status"]), execution_id=execution["execution_id"], started_at=execution["started_at"], finished_at=execution["finished_at"], error_message=execution["error_message"]) if execution else CurrentExecution(),
            published_caches={str(x["scenario_id"]): str(x["relative_path"]) for x in self.connection.execute("SELECT scenario_id, relative_path FROM published_cache_reference")},
        ).validate()

    def _scenario_asset(self, scenario_id: str) -> str:
        row = self.connection.execute("SELECT source_asset_id FROM forecast_scenario WHERE scenario_id=?", (scenario_id,)).fetchone()
        return str(row[0]) if row and row[0] else ""

    def replace(self, artifact: CurrentRunArtifact, *, require_reason: bool = False) -> CurrentRunArtifact:
        artifact.validate(require_reason=require_reason)
        now = datetime.now().isoformat(timespec="seconds")
        ids = {scenario.scenario_id for scenario in artifact.scenarios}
        with self.connection:
            self.connection.execute("DELETE FROM forecast_scenario")
            self.connection.execute("DELETE FROM observed_replacement")
            self.connection.execute("DELETE FROM station_exclusion")
            self.connection.execute("DELETE FROM current_run")
            self.connection.execute("""INSERT INTO current_run (
                singleton,reference_time,forecast_end_exclusive,timestep_hours,
                mgb_settings_json,spatial_settings_json,interpolation_settings_json,review_window_json,
                observed_providers_json,precipitation_qc_settings_json,saved_run_id,saved_run_description,
                responsible_person,reason,created_at,updated_at) VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                artifact.reference_time, artifact.forecast_end_exclusive, artifact.timestep_hours,
                _json(artifact.mgb_settings), _json(artifact.spatial_settings), _json(artifact.interpolation_settings), _json(artifact.review_window), _json(list(artifact.observed_providers)), _json(PrecipitationQCPolicy.from_mapping(artifact.precipitation_qc_settings).as_dict()),
                artifact.saved_run_id, artifact.saved_run_description, artifact.responsible_person, artifact.reason, now, now,
            ))
            self.connection.executemany("INSERT INTO forecast_scenario (scenario_id,position,scenario_kind,label,provider_code,source_asset_id,source_asset_path) VALUES (?,?,?,?,?,?,?)", [(s.scenario_id,s.position,s.kind,s.label,s.provider_code,s.source_asset_id,s.source_asset_path) for s in artifact.scenarios])
            self.connection.executemany("INSERT INTO forecast_correction (scenario_id,position,t0_step,t1_step,shift_lat,shift_lon,rotation_deg,multiplication_factor,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)", [(s.scenario_id,index,c.t0_step,c.t1_step,c.shift_lat,c.shift_lon,c.rotation_deg,c.multiplication_factor,"{}") for s in artifact.scenarios for index,c in enumerate(s.corrections)])
            self.connection.executemany("INSERT INTO observed_replacement VALUES (?,?,?)", [(x.station_id,x.observed_at,x.value) for x in artifact.observed_replacements])
            self.connection.executemany("INSERT INTO station_exclusion VALUES (?,?,?)", [(x.station_id,x.start_time,x.end_time) for x in artifact.station_exclusions])
            self.connection.executemany("INSERT INTO published_cache_reference (scenario_id,relative_path) VALUES (?,?)", [(scenario_id, path) for scenario_id, path in artifact.published_caches.items() if scenario_id in ids])
        return self.load()

    def replace_scenarios(self, scenarios: Iterable[ForecastScenarioReference], *, responsible_person: str | None = None, reason: str | None = None) -> CurrentRunArtifact:
        current = self.load()
        return self.replace(replace(current, scenarios=tuple(scenarios), responsible_person=responsible_person if responsible_person is not None else current.responsible_person, reason=reason if reason is not None else current.reason), require_reason=True)

    def replace_observed_instructions(self, replacements: Iterable[ObservedReplacement], exclusions: Iterable[StationExclusion], *, responsible_person: str | None = None, reason: str | None = None) -> CurrentRunArtifact:
        current = self.load()
        return self.replace(replace(current, observed_replacements=tuple(replacements), station_exclusions=tuple(exclusions), responsible_person=responsible_person if responsible_person is not None else current.responsible_person, reason=reason if reason is not None else current.reason), require_reason=True)

    def set_execution(self, execution: CurrentExecution, *, caches: Mapping[str, str] | None = None) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO current_execution VALUES (1,?,?,?,?,?)", (execution.status, execution.execution_id, execution.started_at, execution.finished_at, execution.error_message))
            if caches is not None:
                self.connection.execute("DELETE FROM published_cache_reference")
                self.connection.executemany("INSERT INTO published_cache_reference (scenario_id,relative_path) VALUES (?,?)", list(caches.items()))


def create_current_artifact(database_path: Path, artifact: CurrentRunArtifact, *, schema_path: Path | None = None) -> CurrentRunArtifact:
    """Create atomically on first use; update an existing stable artifact transactionally."""
    from mgb_ops.assets.schemas import RUN_SCHEMA_PATH
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with CurrentRunRepository(path) as repository:
            return repository.replace(artifact)
    seeded = artifact
    if not str(seeded.reason or "").strip():
        seeded = replace(seeded, reason="default precipitation replacements")
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with sqlite3.connect(temp) as connection:
            connection.executescript(Path(schema_path or RUN_SCHEMA_PATH).read_text(encoding="utf-8"))
            connection.execute("INSERT INTO current_execution (singleton,status) VALUES (1,'never')")
        with CurrentRunRepository(temp) as repository:
            result = repository.replace(seeded)
        os.replace(temp, path)
        return result
    finally:
        temp.unlink(missing_ok=True)


def load_current_artifact(database_path: Path) -> CurrentRunArtifact:
    with CurrentRunRepository(database_path) as repository:
        return repository.load()


def update_current_artifact(database_path: Path, artifact: CurrentRunArtifact, *, require_reason: bool = False) -> CurrentRunArtifact:
    with CurrentRunRepository(database_path) as repository:
        return repository.replace(artifact, require_reason=require_reason)


def validate_current_artifact(database_path: Path) -> CurrentRunArtifact:
    return load_current_artifact(database_path)


def replace_forecast_scenarios(database_path: Path, scenarios: Iterable[ForecastScenarioReference], *, responsible_person: str, reason: str) -> CurrentRunArtifact:
    with CurrentRunRepository(database_path) as repository:
        return repository.replace_scenarios(scenarios, responsible_person=responsible_person, reason=reason)


def replace_observed_instructions(database_path: Path, replacements: Iterable[ObservedReplacement], exclusions: Iterable[StationExclusion], *, responsible_person: str, reason: str) -> CurrentRunArtifact:
    with CurrentRunRepository(database_path) as repository:
        return repository.replace_observed_instructions(replacements, exclusions, responsible_person=responsible_person, reason=reason)


def archive_current_artifact(database_path: Path, destination: Path) -> Path:
    source, target = Path(database_path), Path(destination)
    if target.exists():
        raise FileExistsError(f"Saved artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with sqlite3.connect(source) as source_connection, sqlite3.connect(temp) as target_connection:
            source_connection.backup(target_connection)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def save_current_artifact(database_path: Path, runs_dir: Path, *, run_id: str, description: str) -> SavedRunArtifact:
    safe_id = validate_run_id(run_id)
    detail = str(description or "").strip()
    if not detail:
        raise ValueError("description must be non-empty.")
    target = Path(runs_dir) / f"{safe_id}.sqlite"
    archive_current_artifact(database_path, target)
    try:
        with CurrentRunRepository(target) as repository:
            saved = replace(repository.load(), saved_run_id=safe_id, saved_run_description=detail)
            repository.replace(saved)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return SavedRunArtifact(safe_id, detail, target)


save_current_artifact_copy = archive_current_artifact
