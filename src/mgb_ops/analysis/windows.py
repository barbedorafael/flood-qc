from __future__ import annotations

from datetime import datetime, time, timedelta

from mgb_ops.assets.types import AnalysisWindow


def build_analysis_window(
    reference_time: datetime,
    *,
    observed_horizon_days: int,
    forecast_horizon_days: int,
) -> AnalysisWindow:
    """Build the read-only current/forecast interval from explicit inputs."""
    if observed_horizon_days < 0:
        raise ValueError("observed_horizon_days must be >= 0.")
    if forecast_horizon_days < 0:
        raise ValueError("forecast_horizon_days must be >= 0.")
    reference_date = reference_time.date()
    return AnalysisWindow(
        start_time=datetime.combine(
            reference_date - timedelta(days=observed_horizon_days), time.min
        ),
        cutoff_time=reference_time,
        forecast_end_exclusive=datetime.combine(
            reference_date + timedelta(days=forecast_horizon_days + 1), time.min
        ),
    )


__all__ = ["build_analysis_window"]
