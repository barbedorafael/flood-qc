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

    assert asset["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.rsbuf"
    assert same_path["valid_to"] == "2026-03-27T00:00:00"
    assert found is not None
    assert found["relative_path"] == "data/interim/ecmwf/fc_2026-03-11_00_IFS_rsbuf.grib2"
