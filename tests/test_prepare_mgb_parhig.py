from __future__ import annotations

from datetime import datetime
from pathlib import Path

from model.prepare_mgb_parhig import build_parhig_window, rewrite_parhig_from_config


PARHIG_TEMPLATE = """\
ARQUIVO DE INFORMACOES GERAIS PARA O MODELO DE GRANDES BACIAS
!
Projeto Teste
!
       DIA       MES       ANO      HORA          !INICIO DA SIMULACAO
        01       01       2018        01

        NT        DT       !NUMERO DE INTERVALOS DE TEMPO E TAMANHO DO INTERVALO EM SEGUNDOS
         1     3600.

        NC        NU        NB      NCLI     !NUMERO DE CELULAS, USOS, BACIAS E POSTOS CLIMA
         2         1         1         1

linha final preservada
"""


def test_build_parhig_window_uses_midnight_start_and_hourly_nt() -> None:
    reference_time = datetime(2026, 3, 11, 12, 0, 0)

    start_time, nt = build_parhig_window(reference_time, input_days_before=2)

    assert start_time == datetime(2026, 3, 9, 0, 0, 0)
    assert nt == 61


def test_rewrite_parhig_from_config_updates_only_time_blocks(tmp_path, monkeypatch) -> None:
    parhig_path = tmp_path / "PARHIG.hig"
    parhig_path.write_text(PARHIG_TEMPLATE, encoding="latin-1")

    monkeypatch.setattr(
        "model.prepare_mgb_parhig.load_settings",
        lambda: {
            "run": {"reference_time": "2026-03-11T12:00:00"},
            "ingest": {"request_days": 7, "timeout_seconds": 15},
            "summaries": {"forecast_days": [1], "accum_hours": [24], "selected_mini_ids": []},
            "mgb": {"input_days_before": 2, "output_days_before": 30, "output_days_after": 15},
            "rainfall_interpolation": {"nearest_stations": 5, "power": 2.0},
        },
    )
    monkeypatch.setattr("model.prepare_mgb_parhig.build_execution_id", lambda: "20260311T120000")
    monkeypatch.setattr("model.prepare_mgb_parhig.default_logs_dir", lambda: tmp_path / "logs")

    summary = rewrite_parhig_from_config(parhig_path=parhig_path, logs_dir=tmp_path / "logs")

    assert summary.start_time == datetime(2026, 3, 9, 0, 0, 0)
    assert summary.nt == 61
    assert summary.dt_seconds == 3600

    updated = parhig_path.read_text(encoding="latin-1")
    assert "        09       03       2026        00" in updated
    assert "        61     3600." in updated
    assert "linha final preservada" in updated

    log_path = tmp_path / "logs" / "prepare_mgb_parhig" / "20260311T120000.log"
    assert log_path.exists()
    assert "parhig_updated" in log_path.read_text(encoding="utf-8")
