"""Reusable precipitation suspect-value detection with no persistence concerns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from typing import Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class PrecipitationQCPolicy:
    threshold_mm: float = 50.0
    sequence_min_length: int = 5
    sequence_lower_mm: float = 0.0
    sequence_upper_mm: float = 1.0

    def validate(self) -> "PrecipitationQCPolicy":
        numeric = (self.threshold_mm, self.sequence_lower_mm, self.sequence_upper_mm)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in numeric):
            raise ValueError("Precipitation QC bounds must be finite numbers.")
        if self.threshold_mm < 0:
            raise ValueError("threshold_mm must be >= 0.")
        if not isinstance(self.sequence_min_length, int) or isinstance(self.sequence_min_length, bool) or self.sequence_min_length < 1:
            raise ValueError("sequence_min_length must be an integer >= 1.")
        if self.sequence_lower_mm < 0 or self.sequence_lower_mm >= self.sequence_upper_mm:
            raise ValueError("Sequence bounds must satisfy 0 <= lower < upper.")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "PrecipitationQCPolicy":
        raw = dict(value or {})
        return cls(
            threshold_mm=float(raw.get("threshold_mm", 50.0)),
            sequence_min_length=int(raw.get("sequence_min_length", 5)),
            sequence_lower_mm=float(raw.get("sequence_lower_mm", 0.0)),
            sequence_upper_mm=float(raw.get("sequence_upper_mm", 1.0)),
        ).validate()

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold_mm": float(self.threshold_mm),
            "sequence_min_length": int(self.sequence_min_length),
            "sequence_lower_mm": float(self.sequence_lower_mm),
            "sequence_upper_mm": float(self.sequence_upper_mm),
        }


@dataclass(frozen=True, slots=True)
class PrecipitationQCMatch:
    station_id: str
    observed_at: str
    value: float
    match_type: str
    sequence_id: str | None = None


@dataclass(frozen=True, slots=True)
class PrecipitationQCResult:
    matches: tuple[PrecipitationQCMatch, ...]

    @property
    def threshold_matches(self) -> tuple[PrecipitationQCMatch, ...]:
        return tuple(item for item in self.matches if item.match_type == "threshold")

    @property
    def sequence_matches(self) -> tuple[PrecipitationQCMatch, ...]:
        return tuple(item for item in self.matches if item.match_type == "sequence")

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.sequence_id for item in self.sequence_matches if item.sequence_id))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"station_id": x.station_id, "observed_at": x.observed_at, "value": x.value,
              "match_type": x.match_type, "sequence_id": x.sequence_id} for x in self.matches],
            columns=["station_id", "observed_at", "value", "match_type", "sequence_id"],
        )


def detect_suspect_precipitation(
    observations: pd.DataFrame,
    *,
    timestep_hours: int = 1,
    policy: PrecipitationQCPolicy | Mapping[str, object] | None = None,
) -> PrecipitationQCResult:
    """Detect values above threshold and consecutive strictly bounded low runs."""
    resolved = policy if isinstance(policy, PrecipitationQCPolicy) else PrecipitationQCPolicy.from_mapping(policy)
    resolved.validate()
    if not isinstance(timestep_hours, int) or isinstance(timestep_hours, bool) or timestep_hours < 1:
        raise ValueError("timestep_hours must be an integer >= 1.")
    required = {"station_id", "observed_at", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Observations are missing columns: {sorted(missing)}")
    frame = observations[["station_id", "observed_at", "value"]].copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["observed_at", "value"]).sort_values(["station_id", "observed_at"])
    matches: list[PrecipitationQCMatch] = []
    for row in frame.loc[frame["value"] > resolved.threshold_mm].itertuples(index=False):
        matches.append(PrecipitationQCMatch(row.station_id, row.observed_at.isoformat(), float(row.value), "threshold"))
    step = pd.Timedelta(timedelta(hours=timestep_hours))
    for station_id, station in frame.groupby("station_id", sort=True):
        low = station[(station["value"] > resolved.sequence_lower_mm) & (station["value"] < resolved.sequence_upper_mm)]
        groups, current, previous = [], [], None
        for row in low.itertuples(index=False):
            if previous is None or row.observed_at - previous == step:
                current.append(row)
            else:
                groups.append(current)
                current = [row]
            previous = row.observed_at
        if current:
            groups.append(current)
        for sequence_number, group in enumerate((x for x in groups if len(x) >= resolved.sequence_min_length), 1):
            sequence_id = f"{station_id}:{group[0].observed_at.isoformat()}:{sequence_number}"
            matches.extend(PrecipitationQCMatch(str(station_id), row.observed_at.isoformat(), float(row.value), "sequence", sequence_id) for row in group)
    matches.sort(key=lambda x: (x.station_id, x.observed_at, x.match_type))
    return PrecipitationQCResult(tuple(matches))


scan_precipitation_qc = detect_suspect_precipitation

__all__ = ["PrecipitationQCMatch", "PrecipitationQCPolicy", "PrecipitationQCResult", "detect_suspect_precipitation", "scan_precipitation_qc"]
