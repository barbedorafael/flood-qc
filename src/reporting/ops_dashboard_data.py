from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common.paths import history_db_path, interim_dir


STATE_PRIORITY = {"approved": 0, "curated": 1, "raw": 2}
ACCUM_RASTER_PATTERN = re.compile(r"^accum_(\d+)h\.tif$", re.IGNORECASE)
LEGACY_RIVERS_GEOJSON_PATH = Path("data/legacy/app_layers/rios_mini.geojson")


def _ensure_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def select_preferred_series_rows(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return series_df.copy()

    ranked = series_df.copy()
    ranked["state_rank"] = ranked["state"].map(STATE_PRIORITY).fillna(len(STATE_PRIORITY)).astype(int)
    if "created_at" not in ranked.columns:
        ranked["created_at"] = ""
    else:
        ranked["created_at"] = ranked["created_at"].fillna("")

    ranked = ranked.sort_values(
        ["station_uid", "variable_code", "state_rank", "created_at"],
        ascending=[True, True, True, False],
    )
    preferred = ranked.drop_duplicates(subset=["station_uid", "variable_code"], keep="first")
    return preferred.drop(columns=["state_rank"], errors="ignore").reset_index(drop=True)


def derive_station_kind(variable_codes: Iterable[str]) -> str:
    codes = {str(code).strip().lower() for code in variable_codes if str(code).strip()}
    has_rain = "rain" in codes
    has_stage = bool({"level", "flow"} & codes)

    if has_rain and has_stage:
        return "misto"
    if has_rain:
        return "chuva"
    if has_stage:
        return "nivel"
    return "sem_dados"


def summarize_station_status(values_df: pd.DataFrame, *, days: int) -> dict[str, object]:
    if values_df.empty:
        return {
            "status": "no_data",
            "status_reason": f"sem registros nos ultimos {days} dias",
            "rows_recent": 0,
        }

    non_null_count = int(values_df["value"].notna().sum())
    if non_null_count == 0:
        return {
            "status": "data_issue",
            "status_reason": "somente valores nulos no periodo",
            "rows_recent": int(len(values_df)),
        }

    return {
        "status": "ok",
        "status_reason": "",
        "rows_recent": int(len(values_df)),
    }


def compute_rain_summary(rain_df: pd.DataFrame) -> dict[str, float]:
    if rain_df.empty:
        return {
            "rain_mean_mm_h": float("nan"),
            "rain_acc_24h_mm": float("nan"),
            "rain_p90_mm_h": float("nan"),
        }

    ordered = rain_df.sort_values("datetime").copy()
    ordered["value"] = pd.to_numeric(ordered["value"], errors="coerce")
    valid = ordered.dropna(subset=["value"])
    if valid.empty:
        return {
            "rain_mean_mm_h": float("nan"),
            "rain_acc_24h_mm": float("nan"),
            "rain_p90_mm_h": float("nan"),
        }

    latest_time = valid["datetime"].max()
    rain_24h = valid.loc[valid["datetime"] >= latest_time - timedelta(hours=24), "value"].sum(min_count=1)
    return {
        "rain_mean_mm_h": float(valid["value"].mean()),
        "rain_acc_24h_mm": float(rain_24h) if pd.notna(rain_24h) else float("nan"),
        "rain_p90_mm_h": float(valid["value"].quantile(0.9)),
    }


def load_station_catalog(
    database_path: Path | None = None,
    *,
    days: int = 30,
    now: datetime | None = None,
) -> pd.DataFrame:
    history_path = database_path or history_db_path()
    cutoff = (now or datetime.utcnow()) - timedelta(days=days)
    cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    with _connect(history_path) as connection:
        stations = pd.read_sql_query(
            """
            SELECT
                station_uid,
                station_code,
                provider_code,
                station_name,
                latitude AS lat,
                longitude AS lon
            FROM station
            WHERE latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY provider_code, station_code
            """,
            connection,
        )
        series = pd.read_sql_query(
            """
            SELECT
                series_id,
                station_uid,
                variable_code,
                state,
                created_at
            FROM observed_series
            """,
            connection,
        )
        recent_values = pd.read_sql_query(
            """
            SELECT
                os.series_id,
                os.station_uid,
                os.variable_code,
                ov.observed_at,
                ov.value
            FROM observed_series os
            JOIN observed_value ov ON ov.series_id = os.series_id
            WHERE ov.observed_at >= ?
            """,
            connection,
            params=(cutoff_text,),
        )

    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_uid",
                "station_code",
                "provider_code",
                "station_name",
                "lat",
                "lon",
                "kind",
                "status",
                "status_reason",
            ]
        )

    preferred_series = select_preferred_series_rows(series)
    preferred_ids = set(preferred_series["series_id"].tolist())
    recent_values = recent_values[recent_values["series_id"].isin(preferred_ids)].copy()
    recent_values["datetime"] = _ensure_datetime_series(recent_values["observed_at"])
    recent_values["value"] = pd.to_numeric(recent_values["value"], errors="coerce")
    recent_values = recent_values.dropna(subset=["datetime"])

    coverage = (
        preferred_series.groupby("station_uid")["variable_code"]
        .agg(list)
        .reset_index(name="variable_codes")
    )
    coverage["kind"] = coverage["variable_codes"].apply(derive_station_kind)

    metrics_rows: list[dict[str, object]] = []
    for station_uid, station_values in recent_values.groupby("station_uid", sort=False):
        status_summary = summarize_station_status(station_values, days=days)
        rain_summary = compute_rain_summary(station_values[station_values["variable_code"] == "rain"])
        metrics_rows.append(
            {
                "station_uid": int(station_uid),
                **status_summary,
                **rain_summary,
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    merged = stations.merge(coverage[["station_uid", "kind"]], on="station_uid", how="left")
    merged = merged.merge(metrics, on="station_uid", how="left")
    merged["kind"] = merged["kind"].fillna("sem_dados")
    merged["status"] = merged["status"].fillna("no_data")
    merged["status_reason"] = merged["status_reason"].fillna(f"sem registros nos ultimos {days} dias")
    merged["rows_recent"] = merged["rows_recent"].fillna(0).astype(int)

    for column in ("rain_mean_mm_h", "rain_acc_24h_mm", "rain_p90_mm_h"):
        if column not in merged:
            merged[column] = np.nan

    return merged.sort_values(["provider_code", "station_code"]).reset_index(drop=True)


def load_observed_series(
    station_uid: int,
    database_path: Path | None = None,
    *,
    days: int = 30,
    now: datetime | None = None,
) -> pd.DataFrame:
    history_path = database_path or history_db_path()
    cutoff = (now or datetime.utcnow()) - timedelta(days=days)

    with _connect(history_path) as connection:
        series = pd.read_sql_query(
            """
            SELECT
                series_id,
                station_uid,
                variable_code,
                state,
                created_at
            FROM observed_series
            WHERE station_uid = ?
            """,
            connection,
            params=(int(station_uid),),
        )

        if series.empty:
            return pd.DataFrame(columns=["datetime", "variable_code", "value"])

        preferred = select_preferred_series_rows(series)
        placeholders = ",".join("?" for _ in preferred["series_id"])
        values = pd.read_sql_query(
            f"""
            SELECT
                os.variable_code,
                ov.observed_at AS datetime,
                ov.value
            FROM observed_value ov
            JOIN observed_series os ON os.series_id = ov.series_id
            WHERE ov.series_id IN ({placeholders})
              AND ov.observed_at >= ?
            ORDER BY ov.observed_at
            """,
            connection,
            params=(*preferred["series_id"].tolist(), cutoff.strftime("%Y-%m-%d %H:%M:%S")),
        )

    values["datetime"] = _ensure_datetime_series(values["datetime"])
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    values = values.dropna(subset=["datetime"]).sort_values(["datetime", "variable_code"])
    return values.reset_index(drop=True)


def compute_observed_metrics(observed_df: pd.DataFrame) -> dict[str, object]:
    if observed_df.empty:
        return {
            "latest_time": None,
            "rain_12h": float("nan"),
            "rain_24h": float("nan"),
            "rain_72h": float("nan"),
            "level_current": float("nan"),
            "flow_current": float("nan"),
        }

    ordered = observed_df.copy()
    ordered["datetime"] = _ensure_datetime_series(ordered["datetime"])
    ordered["value"] = pd.to_numeric(ordered["value"], errors="coerce")
    ordered = ordered.dropna(subset=["datetime"]).sort_values("datetime")
    if ordered.empty:
        return {
            "latest_time": None,
            "rain_12h": float("nan"),
            "rain_24h": float("nan"),
            "rain_72h": float("nan"),
            "level_current": float("nan"),
            "flow_current": float("nan"),
        }

    latest_time = ordered["datetime"].max()
    rain_df = ordered[ordered["variable_code"] == "rain"].dropna(subset=["value"])

    def accumulate(hours: int) -> float:
        if rain_df.empty:
            return float("nan")
        value = rain_df.loc[rain_df["datetime"] >= latest_time - timedelta(hours=hours), "value"].sum(min_count=1)
        return float(value) if pd.notna(value) else float("nan")

    def latest_variable(variable_code: str) -> float:
        subset = ordered[(ordered["variable_code"] == variable_code) & ordered["value"].notna()]
        if subset.empty:
            return float("nan")
        return float(subset.sort_values("datetime").iloc[-1]["value"])

    return {
        "latest_time": latest_time,
        "rain_12h": accumulate(12),
        "rain_24h": accumulate(24),
        "rain_72h": accumulate(72),
        "level_current": latest_variable("level"),
        "flow_current": latest_variable("flow"),
    }


def load_model_metadata(database_path: Path | None = None) -> dict[str, object]:
    model_db_path = database_path or interim_dir() / "model_outputs.sqlite"
    with _connect(model_db_path) as connection:
        row = connection.execute(
            """
            SELECT
                reference_time,
                reference_date,
                window_start,
                window_end_exclusive,
                dt_seconds,
                nc,
                nt_current,
                nt_forecast
            FROM metadata
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return {}
    metadata = dict(row)
    for key in ("reference_time", "window_start", "window_end_exclusive"):
        metadata[key] = pd.to_datetime(metadata[key], errors="coerce")
    metadata["reference_date"] = pd.to_datetime(metadata["reference_date"], errors="coerce")
    return metadata


def list_model_variables(database_path: Path | None = None) -> pd.DataFrame:
    model_db_path = database_path or interim_dir() / "model_outputs.sqlite"
    with _connect(model_db_path) as connection:
        variables = pd.read_sql_query(
            """
            SELECT
                variable_code,
                display_name,
                unit
            FROM variable
            ORDER BY variable_code
            """,
            connection,
        )
    return variables


def load_mgb_series(
    mini_id: int,
    variable_code: str,
    database_path: Path | None = None,
    *,
    days_window: int = 30,
) -> pd.DataFrame:
    model_db_path = database_path or interim_dir() / "model_outputs.sqlite"
    with _connect(model_db_path) as connection:
        df = pd.read_sql_query(
            """
            SELECT
                ov.dt,
                os.prev_flag,
                ov.value,
                os.variable_code,
                v.display_name,
                os.unit
            FROM output_series os
            JOIN output_value ov ON ov.series_id = os.series_id
            JOIN variable v ON v.variable_code = os.variable_code
            WHERE os.mini_id = ?
              AND os.variable_code = ?
            ORDER BY ov.dt
            """,
            connection,
            params=(int(mini_id), str(variable_code)),
        )

    if df.empty:
        return pd.DataFrame(columns=["dt", "prev_flag", "value", "variable_code", "display_name", "unit"])

    df["dt"] = _ensure_datetime_series(df["dt"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["prev_flag"] = pd.to_numeric(df["prev_flag"], errors="coerce").fillna(0).astype(int)
    current_df = df[df["prev_flag"] == 0].copy()
    forecast_df = df[df["prev_flag"] == 1].copy()

    if not current_df.empty and days_window > 0:
        cutoff = current_df["dt"].max() - timedelta(days=days_window)
        current_df = current_df[current_df["dt"] >= cutoff]

    out = pd.concat([current_df, forecast_df], ignore_index=True)
    return out.sort_values("dt").reset_index(drop=True)


def list_accumulation_rasters(base_dir: Path | None = None) -> list[dict[str, object]]:
    search_dir = base_dir or interim_dir()
    rasters: list[dict[str, object]] = []
    for path in sorted(search_dir.glob("accum_*h.tif")):
        match = ACCUM_RASTER_PATTERN.match(path.name)
        if not match:
            continue
        horizon_hours = int(match.group(1))
        rasters.append(
            {
                "name": path.stem,
                "path": path,
                "horizon_hours": horizon_hours,
                "horizon_label": f"{horizon_hours}h",
            }
        )
    return sorted(rasters, key=lambda item: int(item["horizon_hours"]))


def load_raster_data(path: Path, *, max_size: int = 600) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Leitura de raster requer rasterio. Instale as dependencias de UI/geo antes de abrir o dashboard."
        ) from exc

    with rasterio.open(path) as src:
        scale = min(max_size / src.height, max_size / src.width, 1.0)
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))
        data = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype("float32")
        data[data <= 0] = np.nan
        data[data <= -1e20] = np.nan
        bounds = src.bounds
    return data, (bounds.left, bounds.bottom, bounds.right, bounds.top)


def load_rivers_layer_geojson(path: Path | None = None) -> dict | None:
    target_path = path or LEGACY_RIVERS_GEOJSON_PATH
    if not target_path.exists():
        return None
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        return None

    for feature in payload.get("features", []):
        props = feature.setdefault("properties", {})
        mini_raw = props.get("mini_id")
        try:
            mini_id = int(mini_raw)
        except (TypeError, ValueError):
            props["click_id"] = "MINI|"
            continue
        props["mini_id"] = mini_id
        props["click_id"] = f"MINI|{mini_id}"
    return payload
