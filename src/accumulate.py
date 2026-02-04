from __future__ import annotations

"""
Gera acumulados de chuva por estação em CSV (um arquivo por estação+horizonte).

Entradas:
- data/telemetria/{CODIGO}.csv (campos: station_id, datetime, rain)

Saídas:
- data/accum/{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv
  (yyyymmdd/hhmm representam reference_time - horizonte)
"""

from datetime import datetime
from typing import Any
from pathlib import Path

import pandas as pd

from config_loader import load_runtime_config, resolve_path, get_report_dir, write_json, get_runtime_reference_time

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TELEM_DIR = DATA_DIR / "telemetria"
DEFAULT_ACCUM_DIR = DATA_DIR / "accum"
DEFAULT_HORIZONS_H = {"24h": 24, "72h": 72, "240h": 240, "720h": 720}
LOCAL_TZ = datetime.now().astimezone().tzinfo


def to_local_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        if LOCAL_TZ is not None:
            ts = ts.tz_convert(LOCAL_TZ)
        ts = ts.tz_localize(None)
    return ts


def to_local_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        if LOCAL_TZ is not None:
            parsed = parsed.dt.tz_convert(LOCAL_TZ)
        parsed = parsed.dt.tz_localize(None)
    return parsed


def compute_accum(df: pd.DataFrame, horizons_h: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = to_local_series(df["datetime"])
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return df

    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").fillna(0.0)
    df = df.sort_values("datetime").set_index("datetime")

    # Resample para 1h somando chuva; faltantes viram 0 para acumular corretamente.
    hourly = df.resample("1h").agg({"rain": "sum"}).fillna(0.0)
    for label, hours in horizons_h.items():
        hourly[f"rain_acc_{label}"] = hourly["rain"].rolling(f"{hours}h", min_periods=1).sum()

    hourly = hourly.drop(columns=["rain"])
    hourly["station_id"] = df["station_id"].iloc[0]
    return hourly.reset_index()


def build_accum_filename(
    station_id: str,
    *,
    reference_time: pd.Timestamp,
    horizon_hours: int,
    horizon_label: str,
) -> str:
    window_start = reference_time - pd.Timedelta(hours=horizon_hours)
    return f"{station_id}_{window_start:%Y%m%d}_{window_start:%H%M}_{horizon_label}.csv"


def save_station_horizon_csv(
    *,
    station_id: str,
    horizon_label: str,
    horizon_hours: int,
    reference_time: pd.Timestamp,
    station_latest_time: pd.Timestamp,
    rain_acc_value: float,
    accum_dir: Path,
) -> Path:
    accum_dir.mkdir(parents=True, exist_ok=True)
    out_name = build_accum_filename(
        station_id,
        reference_time=reference_time,
        horizon_hours=horizon_hours,
        horizon_label=horizon_label,
    )
    out_path = accum_dir / out_name
    window_start = reference_time - pd.Timedelta(hours=horizon_hours)

    payload = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "reference_time": reference_time.isoformat(),
                "window_start": window_start.isoformat(),
                "station_latest_time": to_local_timestamp(station_latest_time).isoformat(),
                "horizon_label": horizon_label,
                "horizon_hours": int(horizon_hours),
                "rain_acc_mm": float(rain_acc_value),
            }
        ]
    )
    payload.to_csv(out_path, index=False)
    return out_path


def main(*, config_dir: str | Path | None = None, event_name: str | None = None) -> None:
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    config = load_runtime_config(config_dir=resolved_config_dir, event_name=event_name)

    horizons_h = config.get("runtime", {}).get("accum_horizons_h", DEFAULT_HORIZONS_H)
    telem_dir = resolve_path(config.get("paths", {}).get("telemetry_dir", str(DEFAULT_TELEM_DIR)))
    accum_dir = resolve_path(config.get("paths", {}).get("accum_dir", str(DEFAULT_ACCUM_DIR)))
    reference_time = to_local_timestamp(get_runtime_reference_time(config))
    if pd.isna(reference_time):
        raise ValueError("runtime.reference_time inválido.")

    telemetry_files = sorted(telem_dir.glob("*.csv"))
    stations_with_accum = 0
    accum_files_generated: list[str] = []

    for telemetry_path in telemetry_files:
        try:
            df = pd.read_csv(telemetry_path, usecols=["station_id", "datetime", "rain"])
        except (FileNotFoundError, ValueError):
            continue
        if df.empty:
            continue

        df["datetime"] = to_local_series(df["datetime"])
        df = df[df["datetime"] <= reference_time]
        if df.empty:
            continue

        accum_df = compute_accum(df, horizons_h)
        if accum_df.empty:
            continue

        latest_row = accum_df.sort_values("datetime").tail(1).iloc[0]
        station_id = str(latest_row["station_id"])
        station_latest_time = to_local_timestamp(latest_row["datetime"])

        wrote_any = False
        for horizon_label, horizon_hours in horizons_h.items():
            col = f"rain_acc_{horizon_label}"
            if col not in latest_row or pd.isna(latest_row[col]):
                continue
            out_path = save_station_horizon_csv(
                station_id=station_id,
                horizon_label=horizon_label,
                horizon_hours=int(horizon_hours),
                reference_time=reference_time,
                station_latest_time=station_latest_time,
                rain_acc_value=float(latest_row[col]),
                accum_dir=accum_dir,
            )
            accum_files_generated.append(out_path.name)
            wrote_any = True

        if wrote_any:
            stations_with_accum += 1

    if config.get("outputs", {}).get("write_summary_json", True):
        summary = {
            "step": "accumulate",
            "run_id": config["runtime"]["run_id"],
            "mode": config.get("run", {}).get("mode", "operational"),
            "event_name": config.get("run", {}).get("event_name"),
            "reference_time": reference_time.isoformat(),
            "accum_horizons_h": horizons_h,
            "telemetry_files_found": len(telemetry_files),
            "stations_with_accum": stations_with_accum,
            "accum_files_generated": accum_files_generated,
            "accum_filename_pattern": "{station}_{yyyymmdd}_{hhmm}_{horizon}.csv",
        }
        report_dir = get_report_dir(config)
        write_json(report_dir / "accumulate_summary.json", summary)

    if not accum_files_generated:
        print("Nenhum acumulado gerado (sem dados de telemetria para o horário de referência).")
        return

    print(f"Acumulados salvos em CSV em {accum_dir} ({len(accum_files_generated)} arquivos).")


if __name__ == "__main__":
    main()
