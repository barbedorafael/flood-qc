from __future__ import annotations

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from model.prepare_mgb_rainfall import extend_station_matrix_with_forecast, prepare_mgb_rainfall


PARHIG_TEMPLATE = """\
ARQUIVO DE INFORMACOES GERAIS PARA O MODELO DE GRANDES BACIAS
!
Projeto Teste
!
       DIA       MES       ANO      HORA          !INICIO DA SIMULACAO
        09       03       2026        00

        NT        DT       !NUMERO DE INTERVALOS DE TEMPO E TAMANHO DO INTERVALO EM SEGUNDOS
       121     3600.

        NC        NU        NB      NCLI     !NUMERO DE CELULAS, USOS, BACIAS E POSTOS CLIMA
         2         1         1         1
"""


MINI_TEMPLATE = """\
Mini Xcen Ycen
1 -51.5 -29.5
2 -52.5 -30.5
"""


def test_extend_station_matrix_with_forecast_zeroes_future_block() -> None:
    station_matrix = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

    extended = extend_station_matrix_with_forecast(
        station_matrix,
        total_nt=4,
        forecast_nt=2,
        use_forecast_data=False,
    )

    assert extended.tolist() == [[1.0, 2.0, 0.0, 0.0]]


def test_extend_station_matrix_with_forecast_rejects_unimplemented_forecast() -> None:
    station_matrix = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)

    with pytest.raises(NotImplementedError, match="Forecast rainfall ingestion is not implemented yet"):
        extend_station_matrix_with_forecast(
            station_matrix,
            total_nt=4,
            forecast_nt=2,
            use_forecast_data=True,
        )


def test_prepare_mgb_rainfall_zeroes_forecast_period(tmp_path, monkeypatch) -> None:
    history_db = tmp_path / "history.sqlite"
    parhig_path = tmp_path / "PARHIG.hig"
    mini_gtp_path = tmp_path / "MINI.gtp"
    output_path = tmp_path / "CHUVABIN.hig"
    parhig_path.write_text(PARHIG_TEMPLATE, encoding="latin-1")
    mini_gtp_path.write_text(MINI_TEMPLATE, encoding="latin-1")
    history_db.write_bytes(b"sqlite placeholder")

    monkeypatch.setattr(
        "model.prepare_mgb_rainfall.load_settings",
        lambda: {
            "run": {"reference_time": "2026-03-11"},
            "ingest": {"request_days": 7, "timeout_seconds": 15},
            "summaries": {"forecast_days": [1], "accum_hours": [24], "selected_mini_ids": []},
            "mgb": {
                "input_days_before": 2,
                "output_days_before": 30,
                "forecast_horizon_days": 2,
                "use_forecast_data": False,
            },
            "rainfall_interpolation": {"nearest_stations": 1, "power": 2.0},
        },
    )
    monkeypatch.setattr("model.prepare_mgb_rainfall.build_execution_id", lambda: "20260311T230000")
    monkeypatch.setattr("model.prepare_mgb_rainfall.default_logs_dir", lambda: tmp_path / "logs")

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("model.prepare_mgb_rainfall._connect_history_read_only", lambda _: FakeConnection())
    monkeypatch.setattr(
        "model.prepare_mgb_rainfall.load_preferred_rain_stations",
        lambda _: pd.DataFrame(
            {
                "series_id": ["s1"],
                "station_uid": [1],
                "state": ["raw"],
                "created_at": [""],
                "lat": [-29.5],
                "lon": [-51.5],
            }
        ),
    )
    monkeypatch.setattr(
        "model.prepare_mgb_rainfall.load_rain_values",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "station_uid": [1, 1, 1],
                "observed_at": [
                    "2026-03-09 00:00",
                    "2026-03-10 00:00",
                    "2026-03-11 23:00",
                ],
                "value": [1.0, 2.0, 3.0],
            }
        ),
    )

    captured: dict[str, np.ndarray] = {}

    def fake_write_chuvabin_atomic(output_path, *, station_matrix, nearest_idx, weights, chunk_hours):
        captured["matrix"] = station_matrix.copy()
        np.asarray(station_matrix, dtype=np.float32).tofile(output_path)

    monkeypatch.setattr("model.prepare_mgb_rainfall.write_chuvabin_atomic", fake_write_chuvabin_atomic)

    summary = prepare_mgb_rainfall(
        history_db=history_db,
        parhig_path=parhig_path,
        mini_gtp_path=mini_gtp_path,
        output_path=output_path,
        nearest_stations=1,
        power=2.0,
        chunk_hours=24,
        logs_dir=tmp_path / "logs",
    )

    matrix = captured["matrix"]
    assert summary.nt == 121
    assert summary.forecast_hours == 49
    assert matrix.shape == (1, 121)
    assert matrix[0, 71] == 3.0
    assert np.allclose(matrix[0, 72:], 0.0)
    assert output_path.exists()
