from __future__ import annotations

"""
Gera um único CSV de acumulados de chuva para todas as estações.

Entradas:
- data/telemetria/{CODIGO}.csv (campos: station_id, datetime, rain)

Saída:
- data/accum/acc_{yyyymmdd}_{hhmm}.csv
  Colunas: station_id, dt_start, dt_end, horizon_h, rain_acc_mm
"""

from pathlib import Path

import pandas as pd

from config_loader import (
    DEFAULT_TIMEZONE,
    load_runtime_config,
    resolve_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TELEM_DIR = DATA_DIR / "telemetria"
DEFAULT_ACCUM_DIR = DATA_DIR / "accum"
DEFAULT_HORIZONS_H = {"24h": 24, "72h": 72, "240h": 240, "720h": 720}
LOCAL_TZ = DEFAULT_TIMEZONE
OUTPUT_COLUMNS = ["station_id", "dt_start", "dt_end", "horizon_h", "rain_acc_mm"]


def to_local_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TZ)
        parsed = parsed.dt.tz_localize(None)
    return parsed


def normalize_horizons(config_horizons: object) -> list[int]:
    if isinstance(config_horizons, dict):
        raw_values = list(config_horizons.values())
    elif isinstance(config_horizons, (list, tuple)):
        raw_values = list(config_horizons)
    else:
        raw_values = list(DEFAULT_HORIZONS_H.values())

    horizons: list[int] = []
    for value in raw_values:
        try:
            hours = int(value)
        except (TypeError, ValueError):
            continue
        if hours > 0 and hours not in horizons:
            horizons.append(hours)
    if not horizons:
        horizons = list(DEFAULT_HORIZONS_H.values())
    return horizons


def build_station_accum_rows(df: pd.DataFrame, horizons_h: list[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.copy()
    df["datetime"] = to_local_series(df["datetime"])
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["rain"] = pd.to_numeric(df["rain"], errors="coerce").fillna(0.0)
    df = df.sort_values("datetime")

    station_id = str(df["station_id"].iloc[0])
    dt_end = df["datetime"].max()

    rows: list[dict[str, object]] = []
    for horizon_h in horizons_h:
        dt_start = dt_end - pd.Timedelta(hours=horizon_h)
        mask = (df["datetime"] > dt_start) & (df["datetime"] <= dt_end)
        rain_acc_mm = round(float(df.loc[mask, "rain"].sum()), 1)
        rows.append(
            {
                "station_id": station_id,
                "dt_start": dt_start,
                "dt_end": dt_end,
                "horizon_h": int(horizon_h),
                "rain_acc_mm": rain_acc_mm,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main(*, config_dir: str | Path | None = None, event_name: str | None = None) -> None:
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    config = load_runtime_config(config_dir=resolved_config_dir, event_name=event_name)

    telem_dir = resolve_path(config.get("paths", {}).get("telemetry_dir", str(DEFAULT_TELEM_DIR)))
    accum_dir = resolve_path(config.get("paths", {}).get("accum_dir", str(DEFAULT_ACCUM_DIR)))
    horizons_h = normalize_horizons(config.get("runtime", {}).get("accum_horizons_h", DEFAULT_HORIZONS_H))

    station_rows: list[pd.DataFrame] = []
    telemetry_files = sorted(telem_dir.glob("*.csv"))
    for telemetry_path in telemetry_files:
        try:
            df = pd.read_csv(telemetry_path, usecols=["station_id", "datetime", "rain"])
        except (FileNotFoundError, ValueError):
            continue
        if df.empty:
            continue

        rows_df = build_station_accum_rows(df, horizons_h)
        if rows_df.empty:
            continue
        station_rows.append(rows_df)

    if not station_rows:
        print("Nenhum acumulado gerado (sem dados válidos em telemetria).")
        return

    accum_df = pd.concat(station_rows, ignore_index=True)
    accum_df = accum_df.sort_values(["station_id", "horizon_h"]).reset_index(drop=True)
    accum_df["rain_acc_mm"] = pd.to_numeric(accum_df["rain_acc_mm"], errors="coerce").round(1)

    dt_end_max = pd.to_datetime(accum_df["dt_end"], errors="coerce").max()
    if pd.isna(dt_end_max):
        print("Nenhum acumulado gerado (dt_end inválido).")
        return

    out_name = f"acc_{dt_end_max:%Y%m%d}_{dt_end_max:%H%M}.csv"
    out_path = accum_dir / out_name
    accum_dir.mkdir(parents=True, exist_ok=True)
    accum_df.to_csv(out_path, index=False, float_format="%.1f")

    print(f"Acumulado salvo em {out_path} ({len(accum_df)} linhas).")


if __name__ == "__main__":
    main()
