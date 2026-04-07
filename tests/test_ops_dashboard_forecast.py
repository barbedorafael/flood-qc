from __future__ import annotations

from datetime import datetime

import numpy as np

from ingest.forecast_grid import TpGribMessage
from qc import ecmwf_forecast_correction
from reporting import ops_dashboard_forecast
from storage.db_bootstrap import initialize_history_db
from storage.history_repository import HistoryRepository


def _message(step_hours: int, value: float) -> TpGribMessage:
    return TpGribMessage(
        valid_time=datetime(2026, 3, 11, step_hours, 0, 0),
        step_hours=step_hours,
        latitudes=np.array([-29.5, -30.5], dtype=np.float64),
        longitudes=np.array([-51.5, -50.5], dtype=np.float64),
        values_mm=np.full((2, 2), value, dtype=np.float64),
    )


def test_ops_dashboard_forecast_lists_steps_and_builds_previews(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "history.sqlite"
    forecast_path = tmp_path / "forecast.grib2"
    forecast_path.write_bytes(b"fake grib")
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.rsbuf",
            asset_kind="forecast_grib_rs_buffered",
            format="GRIB2",
            relative_path=str(forecast_path),
            provider_code="ecmwf",
            valid_from="2026-03-11T00:00:00",
            valid_to="2026-03-12T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z"},
        )

    messages = [_message(0, 0.0), _message(3, 6.0), _message(6, 10.0)]
    monkeypatch.setattr(ops_dashboard_forecast, "read_tp_grib_messages", lambda _: messages)

    assets = ops_dashboard_forecast.list_forecast_assets(db_path)
    steps = ops_dashboard_forecast.list_forecast_steps("ecmwf.ifs.fc.20260311T000000Z.rsbuf", db_path)
    accum_preview = ops_dashboard_forecast.build_forecast_preview(
        "ecmwf.ifs.fc.20260311T000000Z.rsbuf",
        t0_step=0,
        t1_step=3,
        database_path=db_path,
    )
    incr_preview = ops_dashboard_forecast.build_forecast_preview(
        "ecmwf.ifs.fc.20260311T000000Z.rsbuf",
        t0_step=3,
        t1_step=6,
        database_path=db_path,
    )

    assert assets["asset_id"].tolist() == ["ecmwf.ifs.fc.20260311T000000Z.rsbuf"]
    assert steps["step_hours"].tolist() == [0, 3, 6]
    assert np.allclose(accum_preview.data, 6.0)
    assert np.allclose(incr_preview.data, 4.0)


def test_ops_dashboard_forecast_applies_preview_correction() -> None:
    preview = ops_dashboard_forecast.ForecastPreview(
        asset_id="asset",
        relative_path="forecast.grib2",
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        latitudes=np.array([-29.5, -30.5], dtype=np.float64),
        longitudes=np.array([-51.5, -50.5], dtype=np.float64),
        t0_step=0,
        t1_step=3,
        mode_label="acumulado_nativo",
        title="teste",
    )
    instruction = ecmwf_forecast_correction.ForecastCorrectionInstruction(
        asset_id="asset",
        t0_step=0,
        t1_step=3,
        shift_lat=1.0,
        shift_lon=0.0,
        rotation_deg=0.0,
        multiplication_factor=2.0,
    )

    corrected = ops_dashboard_forecast.apply_preview_corrections(preview, [instruction])

    assert corrected.data[0, 0] == 0.0
    assert corrected.data[1, 0] == 2.0
