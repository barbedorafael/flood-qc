from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import history_db_path, interim_dir as default_interim_dir, logs_dir as default_logs_dir
from common.settings import load_settings
from storage.history_repository import HistoryRepository


TIMEZONE = ZoneInfo("America/Sao_Paulo")
REQUEST_WINDOW_HOURS = 24
DEFAULT_ANA_BASE_URL = "http://telemetriaws1.ana.gov.br/serviceana.asmx/DadosHidrometeorologicos"
OBSERVED_VARIABLES = ("rain", "level", "flow")

def resolve_reference_time(raw_value: str | None) -> datetime:
    if raw_value in (None, "", "now"):
        now = datetime.now(TIMEZONE)
        return now.replace(minute=0, second=0, microsecond=0, tzinfo=None)

    text = str(raw_value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    reference_time = datetime.fromisoformat(text)
    if reference_time.tzinfo is not None:
        reference_time = reference_time.astimezone(TIMEZONE).replace(tzinfo=None)
    return reference_time


def build_run_id(reference_time: datetime) -> str:
    return reference_time.strftime("%Y%m%dT%H%M%S")



def configure_run_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("floodqc.ingest.ana")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().replace(",", ".")
    if not normalized or normalized == "-":
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def normalize_ana_station_code(station_code: str | None) -> str | None:
    if station_code is None:
        return None
    normalized = str(station_code).strip()
    if not normalized:
        return None
    return normalized.lstrip("0") or "0"


def iter_data_nodes(root: ET.Element):
    for element in root.iter():
        if element.tag.endswith("DadosHidrometereologicos"):
            yield element


def parse_response(text: str):
    import pandas as pd

    root = ET.fromstring(text)
    records: list[dict] = []
    for data in iter_data_nodes(root):
        station_code = normalize_ana_station_code(data.findtext("CodEstacao"))
        observed_text = data.findtext("DataHora")
        if not station_code or not observed_text:
            continue

        try:
            observed_at = pd.to_datetime(observed_text.strip(), errors="raise")
        except (TypeError, ValueError):
            continue

        if getattr(observed_at, "tzinfo", None) is not None:
            observed_at = observed_at.tz_convert(TIMEZONE).tz_localize(None)

        record = {
            "station_code": station_code,
            "observed_at": observed_at,
            "rain": parse_float(data.findtext("Chuva")),
            "level": parse_float(data.findtext("Nivel")),
            "flow": parse_float(data.findtext("Vazao")),
        }
        if any(record[variable] is not None for variable in OBSERVED_VARIABLES):
            records.append(record)

    if not records:
        return pd.DataFrame(columns=["station_code", "observed_at", *OBSERVED_VARIABLES])

    frame = pd.DataFrame.from_records(records)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    return frame


def fetch_station_xml(
    station_code: str,
    *,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
    timeout_seconds: float,
) -> str:
    params = {
        "codEstacao": station_code,
        "dataInicio": start_time.strftime("%d/%m/%Y %H:%M:%S"),
        "dataFim": end_time.strftime("%d/%m/%Y %H:%M:%S"),
    }
    response = requests.get(base_url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def iter_request_windows(reference_time: datetime, request_days: int):
    start_time = reference_time - timedelta(days=request_days)
    window_delta = timedelta(hours=REQUEST_WINDOW_HOURS)
    one_second = timedelta(seconds=1)

    for window_index in range(request_days):
        window_start = start_time + (window_index * window_delta)
        window_end = window_start + window_delta - one_second
        if window_index == request_days - 1:
            window_end = reference_time
        yield window_start, window_end


def save_raw_xml(
    xml_text: str,
    *,
    raw_root_dir: Path,
    station_code: str,
    start_time: datetime,
    end_time: datetime,
) -> Path:
    station_dir = raw_root_dir / station_code
    station_dir.mkdir(parents=True, exist_ok=True)
    file_path = station_dir / f"{start_time:%Y%m%dT%H%M%S}__{end_time:%Y%m%dT%H%M%S}.xml"
    file_path.write_text(xml_text, encoding="utf-8")
    return file_path


def persist_station_frame(
    repository: HistoryRepository,
    station_uid: int,
    frame,
    *,
    state: str = "raw",
) -> dict[str, int]:
    if frame.empty:
        return {variable: 0 for variable in OBSERVED_VARIABLES}

    station_frame = frame.sort_values("observed_at").drop_duplicates(subset=["observed_at"], keep="last")
    counts: dict[str, int] = {}
    for variable in OBSERVED_VARIABLES:
        variable_frame = station_frame.loc[station_frame[variable].notna(), ["observed_at", variable]]
        if variable_frame.empty:
            counts[variable] = 0
            continue
        rows = [
            (observed_at.strftime("%Y-%m-%d %H:%M:%S"), float(value))
            for observed_at, value in zip(variable_frame["observed_at"], variable_frame[variable])
        ]
        series_id = repository.ensure_observed_series(station_uid, variable, state)
        counts[variable] = repository.upsert_observed_values(series_id, rows)
    return counts


def ingest_observed_ana(
    database_path: Path,
    *,
    base_url: str,
    reference_time: datetime,
    request_days: int,
    timeout_seconds: float,
    station_codes: list[str] | None = None,
    interim_dir: Path,
    logs_dir: Path,
) -> dict[str, object]:
    if request_days < 1:
        raise ValueError("request_days deve ser >= 1.")
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Banco historico nao encontrado: {database_path}")

    run_id = build_run_id(reference_time)
    logger = configure_run_logger(logs_dir / "ana" / f"{run_id}.log")
    raw_root_dir = interim_dir / "ana" / "raw"
    raw_root_dir.mkdir(parents=True, exist_ok=True)

    with HistoryRepository(database_path) as repository:
        stations = repository.get_provider_stations("ana")
        if station_codes:
            allowed_codes = {normalize_ana_station_code(code) for code in station_codes}
            stations = [station for station in stations if station["station_code"] in allowed_codes]
        if not stations:
            raise ValueError("Nenhuma estacao ANA encontrada para ingestao.")

        summary = {
            "run_id": run_id,
            "stations_total": len(stations),
            "stations_ok": 0,
            "stations_no_data": 0,
            "stations_error": 0,
        }

        for station in stations:
            station_code = station["station_code"]
            station_uid = station["station_uid"]
            station_written = {variable: 0 for variable in OBSERVED_VARIABLES}
            station_error = False

            for start_time, end_time in iter_request_windows(reference_time, request_days):
                try:
                    xml_text = fetch_station_xml(
                        station_code,
                        start_time=start_time,
                        end_time=end_time,
                        base_url=base_url,
                        timeout_seconds=timeout_seconds,
                    )
                    raw_path = save_raw_xml(
                        xml_text,
                        raw_root_dir=raw_root_dir,
                        station_code=station_code,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    frame = parse_response(xml_text)
                    returned_codes = {
                        code for code in frame["station_code"].dropna().astype(str).unique().tolist()
                    } if not frame.empty else set()
                    if returned_codes and returned_codes != {station_code}:
                        raise ValueError(
                            f"Resposta da ANA retornou codigos inesperados para {station_code}: {sorted(returned_codes)}"
                        )
                    counts = persist_station_frame(repository, station_uid, frame)
                    for variable, count in counts.items():
                        station_written[variable] += count
                    logger.info(
                        "station=%s station_uid=%s window_start=%s window_end=%s records=%s rain=%s level=%s flow=%s raw_xml=%s",
                        station_code,
                        station_uid,
                        start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        len(frame),
                        counts["rain"],
                        counts["level"],
                        counts["flow"],
                        raw_path,
                    )
                except (requests.RequestException, ET.ParseError, ValueError) as exc:
                    station_error = True
                    logger.error(
                        "station=%s station_uid=%s window_start=%s window_end=%s error=%s",
                        station_code,
                        station_uid,
                        start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        exc,
                    )
                    break

            total_written = sum(station_written.values())
            if station_error:
                summary["stations_error"] += 1
            elif total_written == 0:
                summary["stations_no_data"] += 1
            else:
                summary["stations_ok"] += 1

        logger.info(
            "run_id=%s stations_total=%s stations_ok=%s stations_no_data=%s stations_error=%s",
            summary["run_id"],
            summary["stations_total"],
            summary["stations_ok"],
            summary["stations_no_data"],
            summary["stations_error"],
        )
        return summary


def main() -> int:
    settings = load_settings()
    ingest_settings = settings["ingest"]
    reference_time = resolve_reference_time(settings["run"]["reference_time"])

    ingest_observed_ana(
        history_db_path(),
        base_url=DEFAULT_ANA_BASE_URL,
        reference_time=reference_time,
        request_days=int(ingest_settings["request_days"]),
        timeout_seconds=float(ingest_settings["timeout_seconds"]),
        station_codes=None,
        interim_dir=default_interim_dir(),
        logs_dir=default_logs_dir(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


