from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd
import requests
import xml.etree.ElementTree as ET

from config_loader import load_runtime_config, resolve_paths, resolve_path, get_report_dir, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_STATION_FILES = [
    DATA_DIR / "estacoes_nivel.csv",
    DATA_DIR / "estacoes_pluv.csv",
]


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
    reference_time_utc: datetime,
    request_days: int,
    base_url: str,
    timeout_seconds: float,
) -> pd.DataFrame:
    now = reference_time_utc
    params = {
        "codEstacao": station,
        "dataInicio": (now - timedelta(days=request_days)).strftime("%d/%m/%Y %H:%M:%S"),
        "dataFim": now.strftime("%d/%m/%Y %H:%M:%S"),
    }
    response = requests.get(base_url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_response(response.text)


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
    df.to_csv(file_path, index=False, encoding="utf-8")
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
    timeout_seconds = float(config.get("ingest", {}).get("timeout_seconds", 15))
    output_dir = resolve_path(config.get("paths", {}).get("telemetry_dir", "data/telemetria"))
    reference_time_utc = pd.to_datetime(config["runtime"]["reference_time_utc"]).to_pydatetime()
    include_station_details = bool(config.get("outputs", {}).get("write_station_json", True))
    removed_entries = clear_telemetry_dir(output_dir)
    print(f"Diretório de telemetria limpo: {output_dir} ({removed_entries} entradas removidas)")

    summary = {
        "step": "fetch_data",
        "run_id": config["runtime"]["run_id"],
        "mode": config.get("run", {}).get("mode", "operational"),
        "event_name": config.get("run", {}).get("event_name"),
        "reference_time_utc": config["runtime"]["reference_time_utc"],
        "request_days": request_days,
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
        try:
            df = fetch_station_data(
                station,
                reference_time_utc=reference_time_utc,
                request_days=request_days,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        except requests.RequestException as exc:
            print(f"Falha na estação {station}: {exc}")
            summary["stations_error"] += 1
            if include_station_details:
                summary["details"].append({"station_id": station, "status": "error", "message": str(exc)})
            continue
        if df.empty:
            print(f"Sem dados novos para {station}")
            summary["stations_no_data"] += 1
            if include_station_details:
                summary["details"].append(
                    {"station_id": station, "status": "no_data", "records_fetched": 0}
                )
            continue

        total_records = persist_station_data(station, df, output_dir)
        print(f"{station}: {len(df)} registros novos salvos em {output_dir / (station + '.csv')}")
        summary["stations_ok"] += 1
        if include_station_details:
            summary["details"].append(
                {
                    "station_id": station,
                    "status": "ok",
                    "records_fetched": int(len(df)),
                    "records_total_file": int(total_records),
                }
            )

    if config.get("outputs", {}).get("write_summary_json", True):
        report_dir = get_report_dir(config)
        write_json(report_dir / "fetch_data_summary.json", summary)


if __name__ == "__main__":
    main()
