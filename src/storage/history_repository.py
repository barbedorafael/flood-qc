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

    def __enter__(self) -> HistoryRepository:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

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

    def ensure_observed_series(self, station_uid: int, variable_code: str, state: str = "raw") -> str:
        series_id = build_observed_series_id(station_uid, variable_code, state)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO observed_series (
                series_id,
                station_uid,
                variable_code,
                state
            ) VALUES (?, ?, ?, ?)
            """,
            (series_id, station_uid, variable_code, state),
        )
        self.connection.commit()
        return series_id

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
