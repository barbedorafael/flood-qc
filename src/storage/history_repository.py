from __future__ import annotations

import sqlite3
from pathlib import Path


def build_observed_series_id(station_uid: int, variable_code: str, state: str = "raw") -> str:
    return f"obs.{station_uid}.{variable_code}.{state}"


class HistoryRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._validate_expected_schema()

    def __enter__(self) -> HistoryRepository:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _validate_expected_schema(self) -> None:
        observed_series_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(observed_series)").fetchall()
        }
        expected_observed_series_columns = {
            "series_id",
            "station_uid",
            "variable_code",
            "state",
            "created_at",
        }
        if observed_series_columns != expected_observed_series_columns:
            raise RuntimeError(
                "Banco historico incompatível com o schema atual de observed_series. "
                f"Esperado {sorted(expected_observed_series_columns)}, encontrado {sorted(observed_series_columns)}. "
                f"Apague o arquivo {self.database_path} e rode `python src/storage/db_bootstrap.py --history` "
                "para recriar o banco."
            )

        variable_codes = {
            row["variable_code"]
            for row in self.connection.execute("SELECT variable_code FROM variable").fetchall()
        }
        expected_variables = {"rain", "level", "flow"}
        if not expected_variables.issubset(variable_codes):
            raise RuntimeError(
                "Banco historico incompatível com o catalogo atual de variaveis. "
                f"Esperado pelo menos {sorted(expected_variables)}, encontrado {sorted(variable_codes)}. "
                f"Apague o arquivo {self.database_path} e rode `python src/storage/db_bootstrap.py --history` "
                "para recriar o banco."
            )

    def get_provider_stations(self, provider_code: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT station_uid, station_code, station_name, provider_code
            FROM station
            WHERE provider_code = ?
            ORDER BY station_code
            """,
            (provider_code,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _get_observed_series_id(self, station_uid: int, variable_code: str, state: str) -> str | None:
        row = self.connection.execute(
            """
            SELECT series_id
            FROM observed_series
            WHERE station_uid = ? AND variable_code = ? AND state = ?
            """,
            (station_uid, variable_code, state),
        ).fetchone()
        if row is None:
            return None
        return str(row["series_id"])

    def ensure_observed_series(self, station_uid: int, variable_code: str, state: str = "raw") -> str:
        existing_series_id = self._get_observed_series_id(station_uid, variable_code, state)
        if existing_series_id is not None:
            return existing_series_id

        series_id = build_observed_series_id(station_uid, variable_code, state)
        self.connection.execute(
            """
            INSERT INTO observed_series (
                series_id,
                station_uid,
                variable_code,
                state
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(station_uid, variable_code, state) DO NOTHING
            """,
            (series_id, station_uid, variable_code, state),
        )
        self.connection.commit()
        ensured_series_id = self._get_observed_series_id(station_uid, variable_code, state)
        if ensured_series_id is None:
            raise RuntimeError(
                "Falha ao garantir observed_series "
                f"station_uid={station_uid} variable_code={variable_code} state={state}."
            )
        return ensured_series_id

    def upsert_observed_values(self, series_id: str, rows: list[tuple[str, float]]) -> int:
        if not rows:
            return 0
        self.connection.executemany(
            """
            INSERT INTO observed_value (
                series_id,
                observed_at,
                value
            ) VALUES (?, ?, ?)
            ON CONFLICT(series_id, observed_at) DO UPDATE SET
                value = excluded.value
            """,
            [(series_id, observed_at, value) for observed_at, value in rows],
        )
        self.connection.commit()
        return len(rows)
