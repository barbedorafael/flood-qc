from __future__ import annotations

import sqlite3
from pathlib import Path

from storage.db_bootstrap import initialize_history_db, initialize_run_db

REPO_ROOT = Path(__file__).resolve().parents[1]


def _list_tables(database_path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def _list_columns(database_path, table_name: str) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def test_initialize_history_db(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)
    tables = _list_tables(db_path)
    assert {
        "provider",
        "variable",
        "station",
        "asset",
        "observed_series",
        "observed_value",
        "run_catalog",
    }.issubset(tables)
    assert "ingest_batch" not in tables

    with sqlite3.connect(db_path) as connection:
        providers = {
            row[0] for row in connection.execute("SELECT provider_code FROM provider").fetchall()
        }
        variables = {
            row[0] for row in connection.execute("SELECT variable_code FROM variable").fetchall()
        }

    station_columns = _list_columns(db_path, "station")
    assert {
        "station_uid",
        "station_code",
        "station_name",
        "provider_code",
        "latitude",
        "longitude",
        "altitude_m",
        "created_at",
    }.issubset(station_columns)
    with sqlite3.connect(db_path) as connection:
        altitude_type = connection.execute(
            "SELECT type FROM pragma_table_info('station') WHERE name = 'altitude_m'"
        ).fetchone()[0]
    assert altitude_type == "INTEGER"

    assert {"ana", "inmet", "ecmwf"}.issubset(providers)
    assert {"rain", "level", "flow"}.issubset(variables)

    observed_series_columns = _list_columns(db_path, "observed_series")
    assert {"series_id", "station_uid", "variable_code", "state", "created_at"}.issubset(observed_series_columns)
    assert "provider_code" not in observed_series_columns
    assert "unit" not in observed_series_columns
    assert "source_asset_id" not in observed_series_columns
    assert "ingest_batch_id" not in observed_series_columns


def test_history_station_inventory_csv_loads(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with sqlite3.connect(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM station").fetchone()[0]
        ana_total = connection.execute(
            "SELECT COUNT(*) FROM station WHERE provider_code = 'ana'"
        ).fetchone()[0]
        inmet_total = connection.execute(
            "SELECT COUNT(*) FROM station WHERE provider_code = 'inmet'"
        ).fetchone()[0]
        distinct_uid = connection.execute(
            "SELECT COUNT(DISTINCT station_uid) FROM station"
        ).fetchone()[0]
        distinct_station = connection.execute(
            "SELECT COUNT(DISTINCT provider_code || '|' || station_code) FROM station"
        ).fetchone()[0]
        ana_sample = connection.execute(
            "SELECT station_name, latitude, longitude, altitude_m, typeof(altitude_m) FROM station "
            "WHERE provider_code = 'ana' AND station_code = '2650035'"
        ).fetchone()
        fallback_sample = connection.execute(
            "SELECT station_name, latitude, longitude, altitude_m, typeof(altitude_m) FROM station "
            "WHERE provider_code = 'ana' AND station_code = '74320000'"
        ).fetchone()
        inmet_sample = connection.execute(
            "SELECT station_name, latitude, longitude, altitude_m, typeof(altitude_m) FROM station "
            "WHERE provider_code = 'inmet' AND station_code = 'A840'"
        ).fetchone()
        computed_uids = dict(
            connection.execute(
                "SELECT station_code, station_uid FROM station "
                "WHERE station_code IN ('71200000', '2650035', 'A801', 'B807')"
            ).fetchall()
        )
        padded_code = connection.execute(
            "SELECT COUNT(*) FROM station WHERE station_code IN ('02650035', '0A801')"
        ).fetchone()[0]

    assert total == 343
    assert ana_total == 292
    assert inmet_total == 51
    assert distinct_uid == total
    assert distinct_station == total
    assert ana_sample == ("UHE ITA CACADOR PLU", -26.8192, -50.9856, 960, "integer")
    assert fallback_sample == ("PONTE DO SARGENTO", -26.6822, -53.2861, 0, "integer")
    assert inmet_sample == ("BENTO GONCALVES", -29.1645, -51.5342, 623, "integer")
    assert computed_uids == {
        "71200000": 1071200000,
        "2650035": 1002650035,
        "A801": 2000001801,
        "B807": 2000002807,
    }
    assert padded_code == 0


def test_initialize_run_db(tmp_path) -> None:
    db_path = tmp_path / "20260310T120000.sqlite"
    initialize_run_db("20260310T120000", db_path)
    tables = _list_tables(db_path)
    assert {
        "run",
        "run_input_series",
        "run_input_value",
        "run_asset",
        "derived_series",
        "derived_value",
        "model_execution",
        "mgb_output_series",
        "mgb_output_value",
        "report_artifact",
    }.issubset(tables)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT run_id FROM run").fetchone()
    assert row[0] == "20260310T120000"
