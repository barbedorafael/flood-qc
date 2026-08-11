"""Artifact-owned forecast correction draft helpers."""
from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd
from apps.ops_dashboard.services import forecast as dashboard_forecast
from mgb_ops.edit.forcing import ForecastCorrectionInstruction

FORECAST_EDIT_COLUMNS = ["correction_id", "asset_id", "t0_step", "t1_step", "shift_lat", "shift_lon", "rotation_deg", "multiplication_factor", "metadata_json", "remove"]
FORECAST_EDIT_NUMERIC_COLUMNS = ["t0_step", "t1_step", "shift_lat", "shift_lon", "rotation_deg", "multiplication_factor"]

def normalize_forecast_edit_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    normalized = pd.DataFrame() if frame is None else frame.copy()
    for column in FORECAST_EDIT_COLUMNS:
        if column not in normalized: normalized[column] = pd.NA
    normalized["correction_id"] = pd.to_numeric(normalized["correction_id"], errors="coerce").astype("Int64")
    for column in FORECAST_EDIT_NUMERIC_COLUMNS: normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["asset_id"] = normalized["asset_id"].fillna("").astype(str)
    normalized["metadata_json"] = normalized["metadata_json"].fillna("{}").astype(str)
    normalized["remove"] = normalized["remove"].fillna(False).astype(bool)
    return normalized[FORECAST_EDIT_COLUMNS]

def empty_forecast_edit_frame() -> pd.DataFrame: return normalize_forecast_edit_frame(None)

def build_forecast_edit_row(*, asset_id: str, t0_step: int, t1_step: int, shift_lat: float, shift_lon: float, rotation_deg: float, multiplication_factor: float, metadata: dict[str, Any] | None = None, **_: Any) -> dict[str, object]:
    return {"correction_id": pd.NA, "asset_id": asset_id, "t0_step": int(t0_step), "t1_step": int(t1_step), "shift_lat": float(shift_lat), "shift_lon": float(shift_lon), "rotation_deg": float(rotation_deg), "multiplication_factor": float(multiplication_factor), "metadata_json": json.dumps(metadata or {}, sort_keys=True), "remove": False}

def validate_forecast_edit_draft(asset_id: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    active = normalize_forecast_edit_frame(frame).loc[lambda x: ~x["remove"]].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(active.itertuples(index=False), 1):
        if pd.isna(row.t0_step) or pd.isna(row.t1_step) or pd.isna(row.multiplication_factor): raise ValueError(f"Row {index}: t0_step, t1_step, and multiplication_factor are required.")
        if int(row.t0_step) < 0 or int(row.t1_step) <= int(row.t0_step): raise ValueError(f"Row {index}: correction window must satisfy 0 <= t0_step < t1_step.")
        if not np.isfinite(float(row.multiplication_factor)) or float(row.multiplication_factor) <= 0: raise ValueError(f"Row {index}: multiplication_factor must be > 0.")
        try: metadata = json.loads(str(row.metadata_json or "{}"))
        except json.JSONDecodeError as exc: raise ValueError(f"Row {index}: invalid metadata_json.") from exc
        rows.append({"asset_id": asset_id, "t0_step": int(row.t0_step), "t1_step": int(row.t1_step), "shift_lat": float(0 if pd.isna(row.shift_lat) else row.shift_lat), "shift_lon": float(0 if pd.isna(row.shift_lon) else row.shift_lon), "rotation_deg": float(0 if pd.isna(row.rotation_deg) else row.rotation_deg), "multiplication_factor": float(row.multiplication_factor), "metadata": metadata})
    rows.sort(key=lambda row: (row["t0_step"], row["t1_step"]))
    for previous, current in zip(rows, rows[1:]):
        if current["t0_step"] < previous["t1_step"]: raise ValueError(f"Overlapping grid corrections: [{previous['t0_step']}, {previous['t1_step']}] x [{current['t0_step']}, {current['t1_step']}].")
    return rows

def build_forecast_instruction_from_request(request: dashboard_forecast.ForecastPreviewRequest) -> ForecastCorrectionInstruction:
    return ForecastCorrectionInstruction(asset_id=request.asset_id, t0_step=request.t0_step, t1_step=request.t1_step, shift_lat=request.shift_lat, shift_lon=request.shift_lon, rotation_deg=request.rotation_deg, multiplication_factor=request.multiplication_factor)

__all__ = ["FORECAST_EDIT_COLUMNS", "build_forecast_edit_row", "build_forecast_instruction_from_request", "empty_forecast_edit_frame", "normalize_forecast_edit_frame", "validate_forecast_edit_draft"]
