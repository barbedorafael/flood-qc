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
    assert {"station", "observed_series", "run_catalog"}.issubset(tables)


def test_initialize_run_db(tmp_path) -> None:
    db_path = tmp_path / "20260310T120000.sqlite"
    initialize_run_db("20260310T120000", db_path)
    tables = _list_tables(db_path)
    assert {"run_metadata", "run_lineage", "input_series", "output_series"}.issubset(tables)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT run_id FROM run_metadata").fetchone()
    assert row[0] == "20260310T120000"