from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def build_observed_series_id(station_uid: int, variable_code: str, state: str = "raw") -> str:
    return f"{station_uid}.{variable_code}.{state}"


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
        asset_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(asset)").fetchall()
        }
        expected_asset_columns = {
            "asset_id",
            "asset_kind",
            "format",
            "relative_path",
            "provider_code",
            "checksum",
            "valid_from",
            "valid_to",
            "metadata_json",
            "created_at",
        }
        if asset_columns != expected_asset_columns:
            raise RuntimeError(
                "Banco historico incompatível com o schema atual de asset. "
                f"Esperado {sorted(expected_asset_columns)}, encontrado {sorted(asset_columns)}. "
                f"Apague o arquivo {self.database_path} e rode `python src/storage/db_bootstrap.py --history` "
                "para recriar o banco."
            )

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

        provider_codes = {
            row["provider_code"]
            for row in self.connection.execute("SELECT provider_code FROM provider").fetchall()
        }
        expected_providers = {"ana", "inmet", "ecmwf"}
        if not expected_providers.issubset(provider_codes):
            raise RuntimeError(
                "Banco historico incompatível com o catalogo atual de providers. "
                f"Esperado pelo menos {sorted(expected_providers)}, encontrado {sorted(provider_codes)}. "
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

    def get_asset_by_relative_path(self, relative_path: str) -> dict | None:
        row = self.connection.execute(
            """
            SELECT
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json,
                created_at
            FROM asset
            WHERE relative_path = ?
            """,
            (relative_path,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def upsert_asset(
        self,
        *,
        asset_id: str,
        asset_kind: str,
        format: str,
        relative_path: str,
        provider_code: str | None,
        checksum: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        metadata_json = json.dumps(metadata or {}, sort_keys=True, ensure_ascii=True)
        self.connection.execute(
            """
            INSERT INTO asset (
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
                asset_id = excluded.asset_id,
                asset_kind = excluded.asset_kind,
                format = excluded.format,
                provider_code = excluded.provider_code,
                checksum = excluded.checksum,
                valid_from = excluded.valid_from,
                valid_to = excluded.valid_to,
                metadata_json = excluded.metadata_json
            """,
            (
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json,
            ),
        )
        self.connection.commit()
        ensured_asset = self.get_asset_by_relative_path(relative_path)
        if ensured_asset is None:
            raise RuntimeError(f"Falha ao garantir asset relative_path={relative_path}.")
        return ensured_asset

    def find_latest_ecmwf_asset(self, reference_time: datetime | str, *, asset_kind: str) -> dict | None:
        if isinstance(reference_time, datetime):
            reference_text = reference_time.isoformat(timespec="seconds")
        else:
            reference_text = str(reference_time)
        row = self.connection.execute(
            """
            SELECT
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json,
                created_at
            FROM asset
            WHERE provider_code = 'ecmwf'
              AND asset_kind = ?
              AND valid_from IS NOT NULL
              AND valid_to IS NOT NULL
              AND valid_from <= ?
              AND valid_to >= ?
            ORDER BY valid_from DESC, created_at DESC
            LIMIT 1
            """,
            (asset_kind, reference_text, reference_text),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
