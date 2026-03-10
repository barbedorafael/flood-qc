from __future__ import annotations

import sqlite3

from storage.db_bootstrap import initialize_history_db, initialize_run_db


def _list_tables(database_path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def test_initialize_history_db(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)
    tables = _list_tables(db_path)
    assert {
        "provider",
        "variable",
        "station",
        "station_alias",
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

    assert {"ana", "inmet", "forecast_provider_x", "mgb_setup_ref"}.issubset(providers)
    assert {"rain", "level", "flow", "rain_accum"}.issubset(variables)


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