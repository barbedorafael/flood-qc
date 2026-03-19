from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.prepare_mgb_rainfall import (
    _select_preferred_series_rows,
    build_idw_neighbors,
    interpolate_station_chunk,
    prepare_mgb_rainfall,
    read_mini_centroids,
    temporarily_normalize_rain_to_hourly,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "sql" / "history_schema.sql"


def build_history_db(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.executemany(
            "INSERT INTO station (station_uid, station_code, station_name, provider_code, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1001, "1001", "P1", "ana", -29.0, -50.0),
                (1002, "1002", "P2", "ana", -29.0, -51.0),
            ],
        )
    return path


def write_parhig(path: Path, *, start_time: datetime, nt: int, nc: int, dt_seconds: int = 3600) -> None:
    path.write_text(
        "\n".join(
            [
                "ARQUIVO DE INFORMACOES GERAIS PARA O MODELO DE GRANDES BACIAS",
                "!",
                "       DIA       MES       ANO      HORA          !INICIO DA SIMULACAO",
                f"        {start_time.day:02d}       {start_time.month:02d}       {start_time.year:04d}        {start_time.hour:02d}",
                "",
                "        NT        DT       !NUMERO DE INTERVALOS DE TEMPO E TAMANHO DO INTERVALO EM SEGUNDOS",
                f"{nt:10d}     {dt_seconds}.",
                "",
                "        NC        NU        NB      NCLI     !NUMERO DE CELULAS, USOS, BACIAS E POSTOS CLIMA",
                f"         {nc}         1         1         1",
            ]
        )
        + "\n",
        encoding="latin-1",
    )


def write_mini(path: Path, rows: list[tuple[int, float, float]]) -> None:
    lines = ["CatID Mini Xcen Ycen"]
    for idx, (mini_id, lon, lat) in enumerate(rows, start=1):
        lines.append(f"{idx} {mini_id} {lon:.4f} {lat:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def insert_series(connection: sqlite3.Connection, *, series_id: str, station_uid: int, state: str, created_at: str) -> None:
    connection.execute(
        "INSERT INTO observed_series (series_id, station_uid, variable_code, state, created_at) VALUES (?, ?, 'rain', ?, ?)",
        (series_id, station_uid, state, created_at),
    )


def insert_values(connection: sqlite3.Connection, *, series_id: str, rows: list[tuple[str, float]]) -> None:
    connection.executemany(
        "INSERT INTO observed_value (series_id, observed_at, value) VALUES (?, ?, ?)",
        [(series_id, observed_at, value) for observed_at, value in rows],
    )


def test_select_preferred_series_rows_prefers_approved_then_curated_then_raw() -> None:
    frame = pd.DataFrame(
        [
            {"series_id": "1001.rain.raw", "station_uid": 1001, "state": "raw", "created_at": "2026-03-10 00:00:00"},
            {"series_id": "1001.rain.approved", "station_uid": 1001, "state": "approved", "created_at": "2026-03-11 00:00:00"},
            {"series_id": "1002.rain.raw", "station_uid": 1002, "state": "raw", "created_at": "2026-03-10 00:00:00"},
        ]
    )

    preferred = _select_preferred_series_rows(frame)

    assert preferred["series_id"].tolist() == ["1001.rain.approved", "1002.rain.raw"]


def test_temporarily_normalize_rain_to_hourly_closes_on_the_hour() -> None:
    frame = pd.DataFrame(
        [
            {"station_uid": 1001, "observed_at": "2026-03-10 00:15:00", "value": 1.0},
            {"station_uid": 1001, "observed_at": "2026-03-10 00:30:00", "value": 2.0},
            {"station_uid": 1001, "observed_at": "2026-03-10 00:45:00", "value": 3.0},
            {"station_uid": 1001, "observed_at": "2026-03-10 01:00:00", "value": 4.0},
        ]
    )

    hourly, used_normalization = temporarily_normalize_rain_to_hourly(frame)

    assert used_normalization is True
    assert hourly.to_dict("records") == [
        {"station_uid": 1001, "observed_at": pd.Timestamp("2026-03-10 01:00:00"), "value": 10.0}
    ]


def test_read_mini_centroids_requires_expected_columns(tmp_path) -> None:
    mini_path = tmp_path / "MINI.gtp"
    mini_path.write_text("CatID Mini\n1 10\n", encoding="latin-1")

    with pytest.raises(ValueError, match="missing required columns"):
        read_mini_centroids(mini_path, nc=1)


def test_build_idw_neighbors_and_interpolate_chunk_support_configurable_k() -> None:
    mini_df = pd.DataFrame([{"mini_id": 10, "lon": -50.5, "lat": -29.0}])
    station_df = pd.DataFrame(
        [
            {"station_uid": 1001, "lon": -50.0, "lat": -29.0},
            {"station_uid": 1002, "lon": -51.0, "lat": -29.0},
        ]
    )
    station_chunk = np.array([[2.0], [10.0]], dtype=np.float64)

    nearest_idx, weights = build_idw_neighbors(mini_df, station_df, nearest_stations=2, power=1.0)
    interpolated = interpolate_station_chunk(station_chunk, nearest_idx=nearest_idx, weights=weights)

    assert interpolated.shape == (1, 1)
    assert interpolated[0, 0] == 6.0


def test_prepare_mgb_rainfall_writes_expected_chuvabin(tmp_path, monkeypatch) -> None:
    history_db = build_history_db(tmp_path / "history.sqlite")
    with sqlite3.connect(history_db) as connection:
        insert_series(connection, series_id="1001.rain.approved", station_uid=1001, state="approved", created_at="2026-03-11 00:00:00")
        insert_series(connection, series_id="1002.rain.approved", station_uid=1002, state="approved", created_at="2026-03-11 00:00:00")
        insert_values(
            connection,
            series_id="1001.rain.approved",
            rows=[("2026-03-10 00:00", 1.0), ("2026-03-10 01:00", 2.0), ("2026-03-10 02:00", 3.0)],
        )
        insert_values(
            connection,
            series_id="1002.rain.approved",
            rows=[("2026-03-10 00:00", 10.0), ("2026-03-10 01:00", 20.0), ("2026-03-10 02:00", 30.0)],
        )
        connection.commit()

    parhig_path = tmp_path / "PARHIG.hig"
    mini_path = tmp_path / "MINI.gtp"
    output_path = tmp_path / "chuvabin.hig"
    write_parhig(parhig_path, start_time=datetime(2026, 3, 10, 0, 0, 0), nt=3, nc=2)
    write_mini(mini_path, [(101, -50.0, -29.0), (202, -51.0, -29.0)])

    monkeypatch.setattr("model.prepare_mgb_rainfall.build_execution_id", lambda: "20260311T120000")

    summary = prepare_mgb_rainfall(
        history_db=history_db,
        parhig_path=parhig_path,
        mini_gtp_path=mini_path,
        output_path=output_path,
        nearest_stations=1,
        power=2.0,
        chunk_hours=2,
        logs_dir=tmp_path / "logs",
    )

    data = np.fromfile(output_path, dtype=np.float32).reshape((2, 3), order="F")
    assert summary.station_count == 2
    assert summary.nt == 3
    assert summary.nc == 2
    assert np.allclose(data, np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32))

    log_path = tmp_path / "logs" / "prepare_mgb_rainfall" / "20260311T120000.log"
    assert log_path.exists()
    assert "rainfall_prepare_done" in log_path.read_text(encoding="utf-8")


def test_prepare_mgb_rainfall_fails_when_any_mini_hour_has_no_coverage(tmp_path, monkeypatch) -> None:
    history_db = build_history_db(tmp_path / "history.sqlite")
    with sqlite3.connect(history_db) as connection:
        insert_series(connection, series_id="1001.rain.approved", station_uid=1001, state="approved", created_at="2026-03-11 00:00:00")
        insert_values(
            connection,
            series_id="1001.rain.approved",
            rows=[("2026-03-10 00:00", 1.0), ("2026-03-10 01:00", 2.0)],
        )
        connection.commit()

    parhig_path = tmp_path / "PARHIG.hig"
    mini_path = tmp_path / "MINI.gtp"
    output_path = tmp_path / "chuvabin.hig"
    write_parhig(parhig_path, start_time=datetime(2026, 3, 10, 0, 0, 0), nt=3, nc=1)
    write_mini(mini_path, [(101, -50.0, -29.0)])

    monkeypatch.setattr("model.prepare_mgb_rainfall.build_execution_id", lambda: "20260311T120000")

    with pytest.raises(ValueError, match="without rainfall coverage"):
        prepare_mgb_rainfall(
            history_db=history_db,
            parhig_path=parhig_path,
            mini_gtp_path=mini_path,
            output_path=output_path,
            nearest_stations=1,
            power=2.0,
            chunk_hours=2,
            logs_dir=tmp_path / "logs",
        )
