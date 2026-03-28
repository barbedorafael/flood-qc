from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from ingest import forecast_grid
from storage.db_bootstrap import initialize_history_db


class FakeTemporaryDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def test_ingest_forecast_grids_stores_only_cropped_asset(tmp_path, monkeypatch) -> None:
    history_db = tmp_path / "history.sqlite"
    initialize_history_db(history_db)
    temp_dir = tmp_path / "temp_download"

    monkeypatch.setattr(
        forecast_grid.tempfile,
        "TemporaryDirectory",
        lambda prefix="": FakeTemporaryDirectory(temp_dir),
    )

    def fake_download(target_path: Path, *, reference_time: datetime) -> None:
        target_path.write_bytes(b"raw-grib")

    def fake_crop(source_path: Path, target_path: Path, *, bbox) -> None:
        assert source_path.exists()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"cropped-grib")

    monkeypatch.setattr(forecast_grid, "download_ecmwf_grib_to_path", fake_download)
    monkeypatch.setattr(forecast_grid, "crop_grib_to_bbox", fake_crop)
    monkeypatch.setattr(
        forecast_grid,
        "extract_valid_time_bounds",
        lambda _: (datetime(2026, 3, 11, 3, 0, 0), datetime(2026, 3, 26, 0, 0, 0)),
    )

    summary = forecast_grid.ingest_forecast_grids(
        history_db,
        reference_time=datetime(2026, 3, 11, 23, 0, 0),
        interim_dir=tmp_path / "data" / "interim",
        logs_dir=tmp_path / "logs",
    )

    assert summary.asset_path.exists()
    assert not temp_dir.exists()

    with sqlite3.connect(history_db) as connection:
        row = connection.execute(
            """
            SELECT asset_kind, format, provider_code, relative_path, valid_from, valid_to
            FROM asset
            WHERE provider_code = 'ecmwf'
            """
        ).fetchone()

    assert row == (
        forecast_grid.ECMWF_ASSET_KIND,
        "GRIB2",
        "ecmwf",
        summary.asset_path.as_posix(),
        "2026-03-11T03:00:00",
        "2026-03-26T00:00:00",
    )
