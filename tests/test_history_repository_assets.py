from __future__ import annotations

from datetime import datetime

from storage.db_bootstrap import initialize_history_db
from storage.history_repository import HistoryRepository


def test_history_repository_upserts_and_finds_ecmwf_asset(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        asset = repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.rsbuf",
            asset_kind="forecast_grib_rs_buffered",
            format="GRIB2",
            relative_path="data/interim/ecmwf/fc_2026-03-11_00_IFS_rsbuf.grib2",
            provider_code="ecmwf",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-26T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z"},
        )
        same_path = repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.rsbuf",
            asset_kind="forecast_grib_rs_buffered",
            format="GRIB2",
            relative_path="data/interim/ecmwf/fc_2026-03-11_00_IFS_rsbuf.grib2",
            provider_code="ecmwf",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-27T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z", "bbox": [-72.0, -44.0, -36.0, -17.0]},
        )
        found = repository.find_latest_ecmwf_asset(
            datetime(2026, 3, 11, 12, 0, 0),
            asset_kind="forecast_grib_rs_buffered",
        )
        listed = repository.list_ecmwf_assets(asset_kind="forecast_grib_rs_buffered")

    assert asset["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.rsbuf"
    assert same_path["valid_to"] == "2026-03-27T00:00:00"
    assert found is not None
    assert found["relative_path"] == "data/interim/ecmwf/fc_2026-03-11_00_IFS_rsbuf.grib2"
    assert listed[0]["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.rsbuf"


def test_history_repository_persists_forecast_manual_edits(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.rsbuf",
            asset_kind="forecast_grib_rs_buffered",
            format="GRIB2",
            relative_path="data/interim/ecmwf/fc_2026-03-11_00_IFS_rsbuf.grib2",
            provider_code="ecmwf",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-26T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z"},
        )
        inserted = repository.insert_forecast_manual_edit(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.rsbuf",
            t0_step=0,
            t1_step=24,
            shift_lat=2.0,
            shift_lon=-1.0,
            rotation_deg=5.0,
            multiplication_factor=1.2,
            editor="tester",
            reason="ajuste operacional",
            metadata={"mode_label": "acumulado_nativo"},
        )
        listed = repository.list_forecast_manual_edits("ecmwf.ifs.fc.20260311T000000Z.rsbuf")

    assert inserted["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.rsbuf"
    assert inserted["t0_step"] == 0
    assert inserted["t1_step"] == 24
    assert inserted["shift_lat"] == 2.0
    assert inserted["shift_lon"] == -1.0
    assert inserted["rotation_deg"] == 5.0
    assert inserted["multiplication_factor"] == 1.2
    assert listed[0]["reason"] == "ajuste operacional"
