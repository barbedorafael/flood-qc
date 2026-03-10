from __future__ import annotations

from common.paths import build_run_db_path, history_db_path, interim_dir, runs_dir, spatial_dir, timeseries_dir


def test_standard_paths_are_under_data() -> None:
    assert history_db_path().as_posix().endswith("data/history.sqlite")
    assert runs_dir().as_posix().endswith("data/runs")
    assert interim_dir().as_posix().endswith("data/interim")
    assert timeseries_dir().as_posix().endswith("data/timeseries")
    assert spatial_dir().as_posix().endswith("data/spatial")


def test_run_path_uses_single_sqlite_file() -> None:
    assert build_run_db_path("20260310T120000").as_posix().endswith("data/runs/20260310T120000.sqlite")