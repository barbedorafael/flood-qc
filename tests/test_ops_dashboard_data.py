from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from reporting import ops_dashboard_data
from storage.db_bootstrap import apply_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_SCHEMA_PATH = REPO_ROOT / "sql" / "history_schema.sql"
MODEL_OUTPUTS_SCHEMA_PATH = REPO_ROOT / "sql" / "model_outputs_schema.sql"


def initialize_history_db(path: Path) -> Path:
    apply_schema(path, HISTORY_SCHEMA_PATH)
    return path


def initialize_model_outputs_db(path: Path) -> Path:
    apply_schema(path, MODEL_OUTPUTS_SCHEMA_PATH)
    return path


def insert_station(connection: sqlite3.Connection, *, station_uid: int, station_code: str, station_name: str) -> None:
    connection.execute(
        """
        INSERT INTO station (
            station_uid,
            station_code,
            station_name,
            provider_code,
            latitude,
            longitude,
            altitude_m
        ) VALUES (?, ?, ?, 'ana', -29.5, -53.5, 10)
        """,
        (station_uid, station_code, station_name),
    )


def insert_observed_series(
    connection: sqlite3.Connection,
    *,
    series_id: str,
    station_uid: int,
    variable_code: str,
    state: str,
    created_at: str = "2026-03-17 12:00:00",
) -> None:
    connection.execute(
        """
        INSERT INTO observed_series (series_id, station_uid, variable_code, state, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (series_id, station_uid, variable_code, state, created_at),
    )


def insert_observed_value(connection: sqlite3.Connection, *, series_id: str, observed_at: str, value: float | None) -> None:
    connection.execute(
        "INSERT INTO observed_value (series_id, observed_at, value) VALUES (?, ?, ?)",
        (series_id, observed_at, value),
    )


def test_select_preferred_series_rows_uses_state_precedence() -> None:
    series = pd.DataFrame(
        [
            {"series_id": "rain.raw", "station_uid": 1, "variable_code": "rain", "state": "raw", "created_at": "2026-01-01 00:00:00"},
            {"series_id": "rain.curated", "station_uid": 1, "variable_code": "rain", "state": "curated", "created_at": "2026-01-02 00:00:00"},
            {"series_id": "rain.approved", "station_uid": 1, "variable_code": "rain", "state": "approved", "created_at": "2026-01-03 00:00:00"},
            {"series_id": "level.raw", "station_uid": 1, "variable_code": "level", "state": "raw", "created_at": "2026-01-01 00:00:00"},
        ]
    )

    preferred = ops_dashboard_data.select_preferred_series_rows(series)

    assert preferred["series_id"].tolist() == ["level.raw", "rain.approved"]


def test_derive_station_kind_from_variable_coverage() -> None:
    assert ops_dashboard_data.derive_station_kind(["rain"]) == "chuva"
    assert ops_dashboard_data.derive_station_kind(["level"]) == "nivel"
    assert ops_dashboard_data.derive_station_kind(["flow"]) == "nivel"
    assert ops_dashboard_data.derive_station_kind(["rain", "flow"]) == "misto"


def test_load_station_catalog_classifies_status_from_observed_values(tmp_path) -> None:
    db_path = initialize_history_db(tmp_path / "history.sqlite")
    now = datetime(2026, 3, 17, 12, 0, 0)

    with sqlite3.connect(db_path) as connection:
        insert_station(connection, station_uid=1001, station_code="1001", station_name="OK")
        insert_station(connection, station_uid=1002, station_code="1002", station_name="ISSUE")
        insert_station(connection, station_uid=1003, station_code="1003", station_name="NODATA")

        insert_observed_series(connection, series_id="1001.rain.raw", station_uid=1001, variable_code="rain", state="raw")
        insert_observed_series(connection, series_id="1002.rain.raw", station_uid=1002, variable_code="rain", state="raw")
        insert_observed_series(connection, series_id="1003.rain.raw", station_uid=1003, variable_code="rain", state="raw")

        insert_observed_value(connection, series_id="1001.rain.raw", observed_at="2026-03-16 00:00:00", value=5.0)
        insert_observed_value(connection, series_id="1002.rain.raw", observed_at="2026-03-16 00:00:00", value=None)
        insert_observed_value(connection, series_id="1003.rain.raw", observed_at="2026-01-01 00:00:00", value=2.0)
        connection.commit()

    catalog = ops_dashboard_data.load_station_catalog(db_path, days=30, now=now)
    status_by_station = dict(zip(catalog["station_uid"], catalog["status"]))

    assert status_by_station == {
        1001: "ok",
        1002: "data_issue",
        1003: "no_data",
    }
    assert set(catalog.columns).issuperset(
        {"station_uid", "station_code", "provider_code", "station_name", "lat", "lon", "kind", "status", "status_reason"}
    )


def test_load_observed_series_returns_only_preferred_state_for_station(tmp_path) -> None:
    db_path = initialize_history_db(tmp_path / "history.sqlite")
    now = datetime(2026, 3, 17, 12, 0, 0)

    with sqlite3.connect(db_path) as connection:
        insert_station(connection, station_uid=1001, station_code="1001", station_name="TESTE")
        insert_observed_series(
            connection,
            series_id="1001.rain.raw",
            station_uid=1001,
            variable_code="rain",
            state="raw",
            created_at="2026-03-10 00:00:00",
        )
        insert_observed_series(
            connection,
            series_id="1001.rain.curated",
            station_uid=1001,
            variable_code="rain",
            state="curated",
            created_at="2026-03-11 00:00:00",
        )
        insert_observed_series(connection, series_id="1001.level.raw", station_uid=1001, variable_code="level", state="raw")

        insert_observed_value(connection, series_id="1001.rain.raw", observed_at="2026-03-16 01:00:00", value=1.0)
        insert_observed_value(connection, series_id="1001.rain.curated", observed_at="2026-03-16 01:00:00", value=2.5)
        insert_observed_value(connection, series_id="1001.level.raw", observed_at="2026-03-16 01:00:00", value=120.0)
        connection.commit()

    observed = ops_dashboard_data.load_observed_series(1001, db_path, days=30, now=now)

    assert observed.to_dict(orient="records") == [
        {"datetime": pd.Timestamp("2026-03-16 01:00:00"), "variable_code": "level", "value": 120.0},
        {"datetime": pd.Timestamp("2026-03-16 01:00:00"), "variable_code": "rain", "value": 2.5},
    ]


def test_load_mgb_series_splits_current_and_forecast(tmp_path) -> None:
    db_path = initialize_model_outputs_db(tmp_path / "model_outputs.sqlite")

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO metadata (
                reference_time,
                reference_date,
                window_start,
                window_end_exclusive,
                dt_seconds,
                nc,
                nt_current,
                nt_forecast
            ) VALUES ('2026-02-15T00:00:00', '2026-02-15', '2026-01-16T00:00:00', '2026-03-02T00:00:00', 3600, 1, 2, 2)
            """
        )
        connection.execute("INSERT INTO variable (variable_code, display_name, unit) VALUES ('q', 'QTUDO', 'm3/s')")
        connection.execute(
            "INSERT INTO output_series (series_id, variable_code, mini_id, prev_flag, unit) VALUES ('0539.q.sim', 'q', 539, 0, 'm3/s')"
        )
        connection.execute(
            "INSERT INTO output_series (series_id, variable_code, mini_id, prev_flag, unit) VALUES ('0539.q.for', 'q', 539, 1, 'm3/s')"
        )
        connection.executemany(
            "INSERT INTO output_value (series_id, dt, value) VALUES (?, ?, ?)",
            [
                ("0539.q.sim", "2026-01-01T00:00:00", 1.0),
                ("0539.q.sim", "2026-02-15T00:00:00", 2.0),
                ("0539.q.for", "2026-02-15T01:00:00", 3.0),
                ("0539.q.for", "2026-02-15T02:00:00", 4.0),
            ],
        )
        connection.commit()

    series = ops_dashboard_data.load_mgb_series(539, "q", db_path, days_window=10)

    assert series["prev_flag"].tolist() == [0, 1, 1]
    assert series["value"].tolist() == [2.0, 3.0, 4.0]
    assert series["display_name"].tolist() == ["QTUDO", "QTUDO", "QTUDO"]


def test_list_accumulation_rasters_catalogs_expected_horizons(tmp_path) -> None:
    (tmp_path / "accum_72h.tif").touch()
    (tmp_path / "accum_24h.tif").touch()
    (tmp_path / "accum_720h.tif").touch()
    (tmp_path / "accum_240h.tif").touch()
    (tmp_path / "other.tif").touch()

    catalog = ops_dashboard_data.list_accumulation_rasters(tmp_path)

    assert [item["horizon_label"] for item in catalog] == ["24h", "72h", "240h", "720h"]
    assert [item["name"] for item in catalog] == ["accum_24h", "accum_72h", "accum_240h", "accum_720h"]
