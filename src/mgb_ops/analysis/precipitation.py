from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd

from mgb_ops.analysis.observations import load_effective_observations, load_preferred_rainfall_observations
from mgb_ops.assets.current_run import CurrentRunArtifact, ObservedReplacement, StationExclusion
from mgb_ops.assets.grid_transforms import idw_interpolate, interpolate_station_values
from mgb_ops.assets.spatial_grid import PrecipitationGrid, RegularGridSpec


@dataclass(frozen=True, slots=True)
class ObservationPolicyDraft:
    replacements: tuple[ObservedReplacement, ...] = ()
    exclusions: tuple[StationExclusion, ...] = ()


@dataclass(frozen=True, slots=True)
class LeaveOneOutResult:
    status: str
    station_id: str
    grid: PrecipitationGrid | None
    observed_total: float | None
    predicted_total: float | None
    residual: float | None
    absolute_error: float | None
    expected_timesteps: int
    target_timesteps: int
    predicted_timesteps: int
    message: str | None = None

    @property
    def expected_coverage(self) -> int:
        return self.expected_timesteps

    @property
    def target_coverage(self) -> int:
        return self.target_timesteps

    @property
    def predicted_coverage(self) -> int:
        return self.predicted_timesteps

    @property
    def target_complete(self) -> bool:
        return self.target_timesteps == self.expected_timesteps

    @property
    def prediction_complete(self) -> bool:
        return self.predicted_timesteps == self.expected_timesteps

    @property
    def sufficient_source(self) -> bool:
        return self.status == "ok"


def _draft_instructions(policy_draft: object | None) -> tuple[tuple[ObservedReplacement, ...], tuple[StationExclusion, ...]]:
    if policy_draft is None:
        return (), ()
    if isinstance(policy_draft, Mapping):
        return tuple(policy_draft.get("replacements", ())), tuple(policy_draft.get("exclusions", ()))
    return tuple(getattr(policy_draft, "replacements", getattr(policy_draft, "observed_replacements", ()))), tuple(getattr(policy_draft, "exclusions", getattr(policy_draft, "station_exclusions", ())))


def accumulate_observed_rainfall(
    database_path: Path,
    *,
    start_time: datetime,
    end_time: datetime,
    timestep_hours: int = 1,
    artifact: CurrentRunArtifact | None = None,
    replacements: Iterable[ObservedReplacement] | None = None,
    exclusions: Iterable[StationExclusion] | None = None,
    providers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Accumulate effective preferred rainfall over ``(start, end]``."""
    values = load_effective_observations(
        database_path, start_time=start_time, end_time=end_time,
        timestep_hours=timestep_hours, artifact=artifact, replacements=replacements,
        exclusions=exclusions, providers=providers, require_complete_source=False,
    )
    usable = values.dropna(subset=["value"])
    if usable.empty:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "value"])
    return (usable.groupby(["station_id", "lat", "lon"], as_index=False)["value"]
            .sum(min_count=1).sort_values("station_id").reset_index(drop=True))


def observed_rainfall_grid(
    database_path: Path,
    *,
    grid: RegularGridSpec,
    start_time: datetime,
    end_time: datetime,
    nearest_stations: int = 5,
    power: float = 2.0,
    timestep_hours: int = 1,
    artifact: CurrentRunArtifact | None = None,
    replacements: Iterable[ObservedReplacement] | None = None,
    exclusions: Iterable[StationExclusion] | None = None,
    providers: Iterable[str] | None = None,
) -> PrecipitationGrid:
    effective = load_effective_observations(
        database_path, start_time=start_time, end_time=end_time,
        timestep_hours=timestep_hours, artifact=artifact, replacements=replacements,
        exclusions=exclusions, providers=providers, require_complete_source=True,
    )
    fields = []
    for _, timestep in effective.groupby("observed_at", sort=True):
        fields.append(interpolate_station_values(timestep.dropna(subset=["value"]), grid, nearest_stations=nearest_stations, power=power))
    values = np.sum(np.stack(fields), axis=0)
    return PrecipitationGrid(
        values=values, latitudes=grid.latitudes, longitudes=grid.longitudes,
        bounds=grid.effective_bbox, start_time=start_time, end_time=end_time,
        units="mm", source="observed",
    )


def accumulated_leave_one_out_analysis(
    history_path: Path,
    *,
    start_time: datetime,
    end_time: datetime,
    policy_draft: ObservationPolicyDraft | Mapping[str, Any] | object | None = None,
    selected_station: str,
    grid: RegularGridSpec,
    nearest_stations: int = 5,
    power: float = 2.0,
    timestep_hours: int = 1,
    replacements: Iterable[ObservedReplacement] | None = None,
    exclusions: Iterable[StationExclusion] | None = None,
    providers: Iterable[str] | None = None,
) -> LeaveOneOutResult:
    """Accumulate IDW fields while retaining an omitted station as unmodified truth."""
    if not str(selected_station or "").strip():
        raise ValueError("selected_station is required.")
    draft_replacements, draft_exclusions = _draft_instructions(policy_draft)
    if replacements is None:
        replacements = draft_replacements
    if exclusions is None:
        exclusions = draft_exclusions
    step = pd.Timedelta(hours=timestep_hours)
    expected_index = pd.date_range(pd.Timestamp(start_time) + step, pd.Timestamp(end_time), freq=step)
    if len(expected_index) < 1 or expected_index[-1] != pd.Timestamp(end_time):
        raise ValueError("Review window must contain exact normalized timesteps.")
    raw = load_preferred_rainfall_observations(history_path, start_time=start_time, end_time=end_time, providers=providers)
    truth = raw[(raw["station_id"].astype(str) == str(selected_station)) & raw["value"].notna()].copy()
    if truth.empty and not (raw["station_id"].astype(str) == str(selected_station)).any():
        raise ValueError(f"Selected station is not available in preferred rainfall history: {selected_station}")
    target_meta = raw[raw["station_id"].astype(str) == str(selected_station)].iloc[0]
    truth_by_time = truth.groupby("observed_at")["value"].sum(min_count=1)
    effective = load_effective_observations(
        history_path, start_time=start_time, end_time=end_time, timestep_hours=timestep_hours,
        replacements=replacements, exclusions=exclusions, providers=providers,
        require_complete_source=False,
    )
    sources = effective[(effective["station_id"].astype(str) != str(selected_station)) & effective["value"].notna()]
    fields: list[np.ndarray] = []
    predicted_values: list[float] = []
    for timestamp in expected_index:
        timestep = sources[sources["observed_at"] == timestamp]
        if timestep.empty:
            return LeaveOneOutResult(
                "insufficient_source", str(selected_station), None,
                float(truth_by_time.sum()) if len(truth_by_time) else None, None, None, None,
                len(expected_index), int(truth_by_time.index.isin(expected_index).sum()), len(predicted_values),
                f"No remaining source station at timestep {timestamp.isoformat()}.",
            )
        fields.append(interpolate_station_values(timestep, grid, nearest_stations=nearest_stations, power=power))
        predicted_values.append(float(idw_interpolate(
            timestep["lon"].to_numpy(), timestep["lat"].to_numpy(), timestep["value"].to_numpy(),
            np.array([float(target_meta["lon"])]), np.array([float(target_meta["lat"])]),
            nearest_stations=nearest_stations, power=power,
        )[0]))
    observed_total = float(truth_by_time.sum()) if len(truth_by_time) else None
    predicted_total = float(np.sum(predicted_values))
    target_count = int(truth_by_time.index.isin(expected_index).sum())
    predicted_count = len(predicted_values)
    complete = target_count == len(expected_index) and predicted_count == len(expected_index)
    residual = predicted_total - observed_total if complete and observed_total is not None else None
    accumulated = PrecipitationGrid(
        values=np.sum(np.stack(fields), axis=0), latitudes=grid.latitudes, longitudes=grid.longitudes,
        bounds=grid.effective_bbox, start_time=start_time, end_time=end_time, units="mm", source="observed_leave_one_out",
    )
    return LeaveOneOutResult(
        "ok", str(selected_station), accumulated, observed_total, predicted_total,
        residual, abs(residual) if residual is not None else None,
        len(expected_index), target_count, predicted_count,
    )


analyze_accumulated_leave_one_out = accumulated_leave_one_out_analysis
leave_one_out_precipitation = accumulated_leave_one_out_analysis

__all__ = [
    "LeaveOneOutResult", "ObservationPolicyDraft", "accumulate_observed_rainfall",
    "accumulated_leave_one_out_analysis", "analyze_accumulated_leave_one_out",
    "leave_one_out_precipitation", "observed_rainfall_grid",
]
