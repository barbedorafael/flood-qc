from __future__ import annotations

"""
Gera acumulados de chuva por estação em CSV (um arquivo por estação+horizonte).

Entradas:
- data/telemetria/{CODIGO}.csv (campos: station_id, datetime, rain)

Saídas:
- data/accum/{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv
  (yyyymmdd/hhmm representam reference_time_utc - horizonte)
"""

from pathlib import Path

import pandas as pd

from config_loader import load_runtime_config, resolve_path, get_report_dir, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TELEM_DIR = DATA_DIR / "telemetria"
DEFAULT_ACCUM_DIR = DATA_DIR / "accum"
DEFAULT_HORIZONS_H = {"24h": 24, "72h": 72, "240h": 240, "720h": 720}


def compute_accum(df: pd.DataFrame, horizons_h: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return df

    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").fillna(0.0)
    df = df.sort_values("datetime").set_index("datetime")

    # Resample para 1H somando chuva; faltantes viram 0 para acumular corretamente.
    hourly = df.resample("1H").agg({"rain": "sum"}).fillna(0.0)
    for label, hours in horizons_h.items():
        hourly[f"rain_acc_{label}"] = hourly["rain"].rolling(f"{hours}H", min_periods=1).sum()

    hourly = hourly.drop(columns=["rain"])
    hourly["station_id"] = df["station_id"].iloc[0]
    return hourly.reset_index()


def build_accum_filename(
    station_id: str,
    *,
    reference_time_utc: pd.Timestamp,
    horizon_hours: int,
    horizon_label: str,
) -> str:
    window_start_utc = reference_time_utc - pd.Timedelta(hours=horizon_hours)
    return f"{station_id}_{window_start_utc:%Y%m%d}_{window_start_utc:%H%M}_{horizon_label}.csv"


def save_station_horizon_csv(
    *,
    station_id: str,
    horizon_label: str,
    horizon_hours: int,
    reference_time_utc: pd.Timestamp,
    station_latest_time_utc: pd.Timestamp,
    rain_acc_value: float,
    accum_dir: Path,
) -> Path:
    accum_dir.mkdir(parents=True, exist_ok=True)
    out_name = build_accum_filename(
        station_id,
        reference_time_utc=reference_time_utc,
        horizon_hours=horizon_hours,
        horizon_label=horizon_label,
    )
    out_path = accum_dir / out_name

    payload = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "reference_time_utc": reference_time_utc.isoformat(),
                "window_start_utc": (reference_time_utc - pd.Timedelta(hours=horizon_hours)).isoformat(),
                "station_latest_time_utc": pd.to_datetime(station_latest_time_utc, utc=True).isoformat(),
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
    reference_time_utc = pd.to_datetime(config["runtime"]["reference_time_utc"], utc=True)

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

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df[df["datetime"] <= reference_time_utc]
        if df.empty:
            continue

        accum_df = compute_accum(df, horizons_h)
        if accum_df.empty:
            continue

        latest_row = accum_df.sort_values("datetime").tail(1).iloc[0]
        station_id = str(latest_row["station_id"])
        station_latest_time_utc = pd.to_datetime(latest_row["datetime"], utc=True)

        wrote_any = False
        for horizon_label, horizon_hours in horizons_h.items():
            col = f"rain_acc_{horizon_label}"
            if col not in latest_row or pd.isna(latest_row[col]):
                continue
            out_path = save_station_horizon_csv(
                station_id=station_id,
                horizon_label=horizon_label,
                horizon_hours=int(horizon_hours),
                reference_time_utc=reference_time_utc,
                station_latest_time_utc=station_latest_time_utc,
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
            "reference_time_utc": config["runtime"]["reference_time_utc"],
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
