"""Load preferred rainfall and apply current-run or session-draft instructions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from mgb_ops.assets.current_run import CurrentRunArtifact, ObservedReplacement, StationExclusion
from mgb_ops.assets.history_queries import open_history_read_only, read_observed_values, read_rain_series, select_preferred_series_rows
from mgb_ops.utils.time import validate_timestep_hours


class InsufficientObservationSourceError(ValueError):
    """Raised when an expected timestep has no effective source station."""

    def __init__(self, observed_at: datetime | pd.Timestamp) -> None:
        self.observed_at = pd.Timestamp(observed_at).to_pydatetime()
        super().__init__(f"No usable observed rainfall source at timestep {pd.Timestamp(observed_at).isoformat()}.")


def _naive_timestamp(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp


def load_preferred_rainfall_observations(
    history_path: Path,
    *,
    start_time: datetime,
    end_time: datetime,
    providers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read unmodified latest preferred rainfall observations for ``(start, end]``."""
    start, end = _naive_timestamp(start_time), _naive_timestamp(end_time)
    if end <= start:
        raise ValueError("Observation window must satisfy start < end for (start, end].")
    with open_history_read_only(Path(history_path)) as connection:
        preferred = select_preferred_series_rows(read_rain_series(connection))
        if providers is not None:
            requested = {str(value).strip().lower() for value in providers}
            preferred = preferred[preferred["provider_code"].astype(str).str.lower().isin(requested)]
        values = read_observed_values(
            connection,
            preferred["series_id"].astype(str).tolist(),
            start_time=start.to_pydatetime(),
            end_time=end.to_pydatetime(),
            end_inclusive=True,
        )
    columns = ["series_id", "station_id", "provider_code", "lat", "lon", "observed_at", "value"]
    if values.empty:
        return pd.DataFrame(columns=columns)
    result = values.merge(preferred[["series_id", "provider_code", "lat", "lon"]], on="series_id", how="inner")
    result["observed_at"] = pd.to_datetime(result["observed_at"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result[(result["observed_at"] > start) & (result["observed_at"] <= end)]
    return result[columns].sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def load_effective_observations(
    history_path: Path,
    *,
    start_time: datetime,
    end_time: datetime,
    timestep_hours: int,
    artifact: CurrentRunArtifact | None = None,
    replacements: Iterable[ObservedReplacement] | None = None,
    exclusions: Iterable[StationExclusion] | None = None,
    providers: Iterable[str] | None = None,
    require_complete_source: bool = True,
) -> pd.DataFrame:
    """Return preferred rainfall for ``(start, end]`` after replacement and masks."""
    step_hours = validate_timestep_hours(timestep_hours)
    start, end = _naive_timestamp(start_time), _naive_timestamp(end_time)
    step = pd.Timedelta(hours=step_hours)
    if end <= start or (end - start) % step:
        raise ValueError("Observation window must contain an exact positive number of normalized timesteps.")
    if artifact is not None:
        replacements = artifact.observed_replacements if replacements is None else replacements
        exclusions = artifact.station_exclusions if exclusions is None else exclusions
        providers = artifact.observed_providers if providers is None else providers
    replacement_items = tuple(replacements or ())
    exclusion_items = tuple(exclusions or ())
    raw = load_preferred_rainfall_observations(history_path, start_time=start.to_pydatetime(), end_time=end.to_pydatetime(), providers=providers)
    preferred_meta = raw.drop_duplicates("station_id").set_index("station_id") if not raw.empty else pd.DataFrame()
    result = raw.copy()
    result["raw_value"] = result["value"]
    result["replacement_state"] = "raw"
    result["excluded"] = False
    for instruction in replacement_items:
        timestamp = _naive_timestamp(instruction.observed_at)
        if not (start < timestamp <= end):
            continue
        station_id = str(instruction.station_id)
        mask = (result["station_id"].astype(str) == station_id) & (result["observed_at"] == timestamp)
        if not mask.any():
            if preferred_meta.empty or station_id not in preferred_meta.index:
                raise ValueError(f"Replacement station is not a preferred rainfall station: {station_id}")
            meta = preferred_meta.loc[station_id]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            result = pd.concat([result, pd.DataFrame([{
                "series_id": meta["series_id"], "station_id": station_id,
                "provider_code": meta["provider_code"], "lat": meta["lat"], "lon": meta["lon"],
                "observed_at": timestamp, "value": instruction.value, "raw_value": float("nan"),
                "replacement_state": "null" if instruction.value is None else "numeric", "excluded": False,
            }])], ignore_index=True)
        else:
            result.loc[mask, "value"] = instruction.value
            result.loc[mask, "replacement_state"] = "null" if instruction.value is None else "numeric"
    for instruction in exclusion_items:
        exclusion_start, exclusion_end = _naive_timestamp(instruction.start_time), _naive_timestamp(instruction.end_time)
        mask = ((result["station_id"].astype(str) == str(instruction.station_id)) &
                (result["observed_at"] > exclusion_start) & (result["observed_at"] <= exclusion_end))
        result.loc[mask, "excluded"] = True
        result.loc[mask, "value"] = float("nan")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.sort_values(["observed_at", "station_id"]).reset_index(drop=True)
    if require_complete_source:
        usable = set(result.loc[result["value"].notna(), "observed_at"])
        for timestamp in pd.date_range(start + step, end, freq=step):
            if timestamp not in usable:
                raise InsufficientObservationSourceError(timestamp)
    return result


load_effective_rainfall_observations = load_effective_observations

__all__ = ["InsufficientObservationSourceError", "load_effective_observations", "load_effective_rainfall_observations", "load_preferred_rainfall_observations"]
