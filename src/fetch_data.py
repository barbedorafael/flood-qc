from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd
import requests
import xml.etree.ElementTree as ET

from config_loader import (
    load_runtime_config,
    resolve_paths,
    resolve_path,
    get_report_dir,
    write_json,
    get_runtime_reference_time,
    DEFAULT_TIMEZONE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_STATION_FILES = [
    DATA_DIR / "estacoes_nivel.csv",
    DATA_DIR / "estacoes_pluv.csv",
]
REQUEST_WINDOW_HOURS = 24


def load_station_codes(station_files: list[Path]) -> list[str]:
    codes: set[str] = set()
    for path in station_files:
        if not path.exists():
            print(f"Arquivo não encontrado: {path}")
            continue
        df = pd.read_csv(path, sep=";", encoding="utf-8", usecols=["CODIGO"])
        codes.update(df["CODIGO"].astype(str).str.strip())
    return sorted(codes)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iter_data_nodes(root: ET.Element) -> Iterable[ET.Element]:
    for elem in root.iter():
        if elem.tag.endswith("DadosHidrometereologicos"):
            yield elem


def parse_response(text: str) -> pd.DataFrame:
    root = ET.fromstring(text)
    records = []
    for data in iter_data_nodes(root):
        station = data.findtext("CodEstacao")
        when = data.findtext("DataHora")
        if not station or not when:
            continue
        try:
            timestamp = pd.to_datetime(when.strip(), errors="raise")
        except ValueError:
            continue
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(DEFAULT_TIMEZONE).tz_localize(None)
        entry = {
            "station_id": station.strip(),
            "datetime": timestamp,
            "level": parse_float(data.findtext("Nivel")),
            "rain": parse_float(data.findtext("Chuva")),
            "flow": parse_float(data.findtext("Vazao")),
        }
        if any(v is not None for v in (entry["level"], entry["rain"], entry["flow"])):
            records.append(entry)
    if records:
        return pd.DataFrame.from_records(records)
    return pd.DataFrame(columns=["station_id", "datetime", "level", "rain", "flow"])


def fetch_station_data(
    station: str,
    *,
    start_time: datetime,
    end_time: datetime,
    base_url: str,
    timeout_seconds: float,
) -> pd.DataFrame:
    params = {
        "codEstacao": station,
        "dataInicio": start_time.strftime("%d/%m/%Y %H:%M:%S"),
        "dataFim": end_time.strftime("%d/%m/%Y %H:%M:%S"),
    }
    response = requests.get(base_url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_response(response.text)


def iter_request_windows(reference_time: datetime, request_days: int) -> Iterable[tuple[datetime, datetime]]:
    """
    Divide o intervalo total em janelas de 24h sem sobreposição.
    """
    start_time = reference_time - timedelta(days=request_days)
    window_delta = timedelta(hours=REQUEST_WINDOW_HOURS)
    one_second = timedelta(seconds=1)

    for window_index in range(request_days):
        window_start = start_time + (window_index * window_delta)
        window_end = window_start + window_delta - one_second
        if window_index == request_days - 1:
            window_end = reference_time
        yield window_start, window_end


def clear_telemetry_dir(output_dir: Path) -> int:
    """
    Remove todos os arquivos/subpastas de telemetria no início do run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            removed += 1
        else:
            item.unlink(missing_ok=True)
            removed += 1
    return removed


def persist_station_data(station: str, df: pd.DataFrame, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{station}.csv"

    df = (
        df.sort_values("datetime")
        .drop_duplicates(["station_id", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        return 0
    df.to_csv(file_path, mode="a", header=not file_path.exists(), index=False, encoding="utf-8")
    return len(df)


def main(*, config_dir: str | Path | None = None, event_name: str | None = None) -> None:
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    config = load_runtime_config(config_dir=resolved_config_dir, event_name=event_name)

    station_files = config.get("paths", {}).get("station_files", [])
    stations = load_station_codes(resolve_paths(station_files) if station_files else DEFAULT_STATION_FILES)
    if not stations:
        print("Não há estações carregadas; execute primeiro src/station_inventory.py")
        return

    base_url = str(config.get("ingest", {}).get("ana_base_url"))
    request_days = int(config.get("ingest", {}).get("request_days", 3))
    if request_days < 1:
        print("Valor inválido para ingest.request_days: use um inteiro >= 1.")
        return
    timeout_seconds = float(config.get("ingest", {}).get("timeout_seconds", 15))
    output_dir = resolve_path(config.get("paths", {}).get("telemetry_dir", "data/telemetria"))
    reference_time = pd.to_datetime(get_runtime_reference_time(config)).to_pydatetime()
    if pd.isna(reference_time):
        raise ValueError("runtime.reference_time inválido.")
    request_windows = list(iter_request_windows(reference_time, request_days))
    include_station_details = bool(config.get("outputs", {}).get("write_station_json", True))
    removed_entries = clear_telemetry_dir(output_dir)
    print(f"Diretório de telemetria limpo: {output_dir} ({removed_entries} entradas removidas)")

    summary = {
        "step": "fetch_data",
        "run_id": config["runtime"]["run_id"],
        "mode": config.get("run", {}).get("mode", "operational"),
        "event_name": config.get("run", {}).get("event_name"),
        "reference_time": get_runtime_reference_time(config),
        "request_days": request_days,
        "request_window_hours": REQUEST_WINDOW_HOURS,
        "stations_total": len(stations),
        "stations_ok": 0,
        "stations_no_data": 0,
        "stations_error": 0,
        "telemetry_dir": str(output_dir),
        "telemetry_dir_removed_entries": removed_entries,
    }
    if include_station_details:
        summary["details"] = []

    for station in stations:
        records_fetched = 0
        records_written = 0
        try:
            for start_time, end_time in request_windows:
                df = fetch_station_data(
                    station,
                    start_time=start_time,
                    end_time=end_time,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
                records_fetched += len(df)
                records_written += persist_station_data(station, df, output_dir)
        except requests.RequestException as exc:
            print(f"Falha na estação {station}: {exc}")
            summary["stations_error"] += 1
            if include_station_details:
                summary["details"].append(
                    {
                        "station_id": station,
                        "status": "error",
                        "message": str(exc),
                        "records_fetched": int(records_fetched),
                        "records_total_file": int(records_written),
                    }
                )
            continue
        if records_written == 0:
            print(f"Sem dados novos para {station}")
            summary["stations_no_data"] += 1
            if include_station_details:
                summary["details"].append(
                    {"station_id": station, "status": "no_data", "records_fetched": 0, "records_total_file": 0}
                )
            continue

        print(f"{station}: {records_written} registros novos salvos em {output_dir / (station + '.csv')}")
        summary["stations_ok"] += 1
        if include_station_details:
            summary["details"].append(
                {
                    "station_id": station,
                    "status": "ok",
                    "records_fetched": int(records_fetched),
                    "records_total_file": int(records_written),
                }
            )

    if config.get("outputs", {}).get("write_summary_json", True):
        report_dir = get_report_dir(config)
        write_json(report_dir / "fetch_data_summary.json", summary)


if __name__ == "__main__":
    main()
