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


def _apply_sql(database_path, sql_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.executescript(sql_path.read_text(encoding="utf-8"))
        connection.commit()


def test_initialize_history_db(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)
    tables = _list_tables(db_path)
    assert {
        "provider",
        "variable",
        "station",
        "asset",
        "ingest_batch",
        "observed_series",
        "observed_value",
        "run_catalog",
    }.issubset(tables)

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
    assert "station_type" not in station_columns

    assert {"ana", "inmet", "ecmwf"}.issubset(providers)
    assert {"rain", "level"}.issubset(variables)


def test_history_station_inventory_seed_loads(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)
    _apply_sql(db_path, REPO_ROOT / "sql" / "history_station_inventory_seed.sql")

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
        ana_match = connection.execute(
            "SELECT station_name, altitude_m FROM station WHERE provider_code = 'ana' AND station_code = '2650035'"
        ).fetchone()
        local_fallback = connection.execute(
            "SELECT station_name, altitude_m FROM station WHERE provider_code = 'ana' AND station_code = '74320000'"
        ).fetchone()
        inmet_fallback = connection.execute(
            "SELECT station_name, altitude_m FROM station WHERE provider_code = 'inmet' AND station_code = 'B807'"
        ).fetchone()
        padded_code = connection.execute(
            "SELECT COUNT(*) FROM station WHERE station_code = '02650035'"
        ).fetchone()[0]

    assert total == 343
    assert ana_total == 292
    assert inmet_total == 51
    assert distinct_uid == total
    assert distinct_station == total
    assert ana_match == ("UHE ITÁ CAÇADOR PLU", 960.0)
    assert local_fallback == ("PONTE DO SARGENTO", 0.0)
    assert inmet_fallback == ("PORTO ALEGRE- BELEM NOVO", 2.0)
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
