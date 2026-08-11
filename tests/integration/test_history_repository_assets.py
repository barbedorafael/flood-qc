from __future__ import annotations

from datetime import datetime

import pytest

from db_helpers import initialize_history_db
from mgb_ops.assets.history import HistoryRepository


def test_history_repository_upserts_and_finds_ecmwf_asset(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        asset = repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.precipitation_grid",
            asset_kind="forecast_precipitation_grid",
            format="NetCDF",
            relative_path="data/downloads/ecmwf/fc_2026-03-11_00_IFS_precipitation_grid.nc",
            provider_code="ecmwf",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-26T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z"},
        )
        same_path = repository.upsert_asset(
            asset_id="ecmwf.ifs.fc.20260311T000000Z.precipitation_grid",
            asset_kind="forecast_precipitation_grid",
            format="NetCDF",
            relative_path="data/downloads/ecmwf/fc_2026-03-11_00_IFS_precipitation_grid.nc",
            provider_code="ecmwf",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-27T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z", "bbox": [-72.0, -44.0, -36.0, -17.0]},
        )
        found = repository.find_latest_asset(
            datetime(2026, 3, 11, 12, 0, 0),
            provider_code="ecmwf",
            asset_kind="forecast_precipitation_grid",
        )
        listed = repository.list_assets(provider_code="ecmwf", asset_kind="forecast_precipitation_grid")

    assert asset["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.precipitation_grid"
    assert same_path["valid_to"] == "2026-03-27T00:00:00"
    assert found is not None
    assert found["relative_path"] == "data/downloads/ecmwf/fc_2026-03-11_00_IFS_precipitation_grid.nc"
    assert listed[0]["asset_id"] == "ecmwf.ifs.fc.20260311T000000Z.precipitation_grid"


def test_history_repository_lists_and_finds_generic_non_ecmwf_asset(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        repository.upsert_asset(
            asset_id="noaa.test.fc.20260311T000000Z.precipitation_grid",
            asset_kind="forecast_precipitation_grid",
            format="NetCDF",
            relative_path="data/downloads/noaa/fc_2026-03-11_00_GFS_precipitation_grid.nc",
            provider_code="noaa",
            valid_from="2026-03-11T03:00:00",
            valid_to="2026-03-12T00:00:00",
            metadata={"cycle_time": "2026-03-11T00:00:00Z"},
        )

        listed = repository.list_assets(provider_code="noaa", asset_kind="forecast_precipitation_grid")
        found = repository.find_latest_asset(
            datetime(2026, 3, 11, 12, 0, 0),
            provider_code="noaa",
            asset_kind="forecast_precipitation_grid",
        )

    assert [asset["asset_id"] for asset in listed] == ["noaa.test.fc.20260311T000000Z.precipitation_grid"]
    assert found is not None
    assert found["provider_code"] == "noaa"




def test_history_repository_does_not_expose_operational_edit_storage(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)
    with HistoryRepository(db_path) as repository:
        assert not hasattr(repository, "list_forecast_manual_edits")
        assert not hasattr(repository, "replace_forecast_manual_edits")
        tables = {row[0] for row in repository.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"manual_edit", "qc_flag", "run_catalog"}.isdisjoint(tables)
