from __future__ import annotations

import sqlite3
from datetime import datetime

from ingest.fetch_observed_ana import ingest_observed_ana
from storage.db_bootstrap import initialize_history_db
from storage.history_repository import HistoryRepository, build_observed_series_id


SAMPLE_ANA_XML = """\
<root>
  <DadosHidrometereologicos>
    <CodEstacao>74100000</CodEstacao>
    <DataHora>2026-03-10 00:00:00</DataHora>
    <Nivel>100</Nivel>
    <Chuva>1.0</Chuva>
    <Vazao>10.0</Vazao>
  </DadosHidrometereologicos>
  <DadosHidrometereologicos>
    <CodEstacao>74100000</CodEstacao>
    <DataHora>2026-03-10 00:00:00</DataHora>
    <Nivel>101</Nivel>
    <Chuva>2.0</Chuva>
    <Vazao>11.0</Vazao>
  </DadosHidrometereologicos>
  <DadosHidrometereologicos>
    <CodEstacao>74100000</CodEstacao>
    <DataHora>2026-03-10 02:00:00</DataHora>
    <Nivel>105</Nivel>
    <Chuva></Chuva>
    <Vazao>12.0</Vazao>
  </DadosHidrometereologicos>
</root>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_history_repository_observed_series_and_values(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    with HistoryRepository(db_path) as repository:
        station = repository.get_provider_stations("ana")[0]
        series_id = repository.ensure_observed_series(station["station_uid"], "rain")
        repeated_series_id = repository.ensure_observed_series(station["station_uid"], "rain")
        written = repository.upsert_observed_values(
            series_id,
            [("2026-03-10 00:00:00", 1.0), ("2026-03-10 01:00:00", 2.0)],
        )
        updated = repository.upsert_observed_values(
            series_id,
            [("2026-03-10 00:00:00", 3.5)],
        )

    with sqlite3.connect(db_path) as connection:
        series_total = connection.execute("SELECT COUNT(*) FROM observed_series").fetchone()[0]
        values = connection.execute(
            "SELECT observed_at, value FROM observed_value WHERE series_id = ? ORDER BY observed_at",
            (series_id,),
        ).fetchall()

    assert series_id == repeated_series_id
    assert series_id == build_observed_series_id(station["station_uid"], "rain")
    assert written == 2
    assert updated == 1
    assert series_total == 1
    assert values == [("2026-03-10 00:00:00", 3.5), ("2026-03-10 01:00:00", 2.0)]


def test_fetch_observed_ana_persists_values_and_logs(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "history.sqlite"
    initialize_history_db(db_path)

    def fake_get(*args, **kwargs):
        return FakeResponse(SAMPLE_ANA_XML)

    monkeypatch.setattr("ingest.fetch_observed_ana.requests.get", fake_get)

    summary = ingest_observed_ana(
        db_path,
        base_url="http://example.test/ana",
        reference_time=datetime(2026, 3, 11, 0, 0, 0),
        request_days=1,
        timeout_seconds=5,
        station_codes=["74100000"],
        interim_dir=tmp_path / "interim",
        logs_dir=tmp_path / "logs",
    )

    with sqlite3.connect(db_path) as connection:
        series_rows = connection.execute(
            "SELECT series_id, variable_code FROM observed_series ORDER BY variable_code"
        ).fetchall()
        rain_values = connection.execute(
            "SELECT observed_at, value FROM observed_value "
            "WHERE series_id = 'obs.1074100000.rain.raw' ORDER BY observed_at"
        ).fetchall()
        level_values = connection.execute(
            "SELECT observed_at, value FROM observed_value "
            "WHERE series_id = 'obs.1074100000.level.raw' ORDER BY observed_at"
        ).fetchall()
        flow_values = connection.execute(
            "SELECT observed_at, value FROM observed_value "
            "WHERE series_id = 'obs.1074100000.flow.raw' ORDER BY observed_at"
        ).fetchall()

    raw_xml_files = list((tmp_path / "interim" / "ana" / "raw" / "74100000").glob("*.xml"))
    log_file = tmp_path / "logs" / "ana" / "20260311T000000.log"

    assert summary == {
        "run_id": "20260311T000000",
        "stations_total": 1,
        "stations_ok": 1,
        "stations_no_data": 0,
        "stations_error": 0,
    }
    assert series_rows == [
        ("obs.1074100000.flow.raw", "flow"),
        ("obs.1074100000.level.raw", "level"),
        ("obs.1074100000.rain.raw", "rain"),
    ]
    assert rain_values == [("2026-03-10 00:00:00", 2.0)]
    assert level_values == [("2026-03-10 00:00:00", 101.0), ("2026-03-10 02:00:00", 105.0)]
    assert flow_values == [("2026-03-10 00:00:00", 11.0), ("2026-03-10 02:00:00", 12.0)]
    assert len(raw_xml_files) == 1
    assert raw_xml_files[0].read_text(encoding="utf-8") == SAMPLE_ANA_XML
    assert log_file.exists()
    assert "station=74100000" in log_file.read_text(encoding="utf-8")
    assert not (tmp_path / "reports").exists()


def test_history_repository_rejects_old_observed_schema(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE observed_series (
                series_id TEXT PRIMARY KEY,
                station_uid INTEGER NOT NULL,
                provider_code TEXT NOT NULL,
                variable_code TEXT NOT NULL,
                unit TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'raw',
                source_asset_id TEXT,
                ingest_batch_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE variable (
                variable_code TEXT PRIMARY KEY,
                variable_name TEXT NOT NULL,
                default_unit TEXT NOT NULL,
                description TEXT
            );
            INSERT INTO variable (variable_code, variable_name, default_unit, description) VALUES
                ('rain', 'Precipitacao observada', 'mm', ''),
                ('level', 'Nivel observado', 'cm', '');
            """
        )

    try:
        HistoryRepository(db_path)
    except RuntimeError as exc:
        assert "Banco historico incompat" in str(exc)
        assert "db_bootstrap.py --history" in str(exc)
    else:
        raise AssertionError("Era esperado erro para schema antigo de observed_series.")
