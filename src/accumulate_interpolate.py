from __future__ import annotations

"""
Gera acumulados de chuva (24h, 72h, 240h, 720h) por estação e interpolação IDW simples.

Entradas:
- data/telemetria/{CODIGO}.csv      (campos: station_id, datetime, rain)
- data/estacoes_*.csv               (campos: CODIGO, LAT, LON)

Saídas:
- data/accum/{CODIGO}.parquet       (séries com colunas rain_acc_{h})
- data/interp/accum_{h}.tif         (COG idw em EPSG:4326 com tags horizon/ref_time)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from config_loader import load_runtime_config, resolve_paths, resolve_path, get_report_dir, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TELEM_DIR = DATA_DIR / "telemetria"
DEFAULT_ACCUM_DIR = DATA_DIR / "accum"
DEFAULT_INTERP_DIR = DATA_DIR / "interp"
DEFAULT_STATION_FILES = [DATA_DIR / "estacoes_nivel.csv", DATA_DIR / "estacoes_pluv.csv"]
DEFAULT_HORIZONS_H = {"24h": 24, "72h": 72, "240h": 240, "720h": 720}


def load_stations(station_files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in station_files:
        if path.exists():
            frames.append(pd.read_csv(path, sep=";", encoding="utf-8"))
    if not frames:
        raise FileNotFoundError("Nenhum CSV de estação encontrado em data/estacoes_*.csv")
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="CODIGO")
        .rename(columns={"CODIGO": "station_id", "LAT": "lat", "LON": "lon"})
    )
    return df[["station_id", "lat", "lon"]].dropna()


def compute_accum(df: pd.DataFrame, horizons_h: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    # resample para 1H somando chuva; faltantes viram 0 para acumular corretamente
    hourly = df.resample("1H").agg({"rain": "sum"}).fillna(0.0)
    for label, hours in horizons_h.items():
        hourly[f"rain_acc_{label}"] = hourly["rain"].rolling(f"{hours}H", min_periods=1).sum()
    hourly = hourly.drop(columns=["rain"])  # manter só acumulados
    hourly["station_id"] = df["station_id"].iloc[0]
    return hourly.reset_index()


def save_station_accum(station: str, accum_df: pd.DataFrame, accum_dir: Path) -> None:
    accum_dir.mkdir(parents=True, exist_ok=True)
    out_path = accum_dir / f"{station}.parquet"
    accum_df.to_parquet(out_path, index=False)


def idw_interpolate(
    points: pd.DataFrame, value_col: str, grid_res: float = 0.1, power: float = 2.0
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Interpola valores via IDW em uma grade regular (lat/lon) e retorna raster + transform.
    Mantém resolução em graus; grade vai de (lon_min, lat_max) no canto superior esquerdo.
    """
    lats = points["lat"].to_numpy()
    lons = points["lon"].to_numpy()
    vals = points[value_col].to_numpy()

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # linha 0 no raster precisa ser o norte; por isso lat decrescente
    grid_lat = np.arange(lat_max, lat_min - grid_res, -grid_res)
    grid_lon = np.arange(lon_min, lon_max + grid_res, grid_res)
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

    interpolated = np.zeros_like(lat_grid, dtype=float)
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            dy = lat_grid[i, j] - lats
            dx = lon_grid[i, j] - lons
            dist = np.hypot(dx, dy)
            dist[dist == 0] = 1e-6  # evita divisão por zero
            weights = 1 / (dist ** power)
            interpolated[i, j] = np.sum(weights * vals) / np.sum(weights)

    transform = from_origin(lon_min, lat_max, grid_res, grid_res)
    return interpolated.astype("float32"), transform


def save_interp_cog(
    array: np.ndarray, transform, out_path: Path, *, horizon: str, ref_time: pd.Timestamp, grid_res: float
) -> None:
    profile = {
        "driver": "COG",
        "dtype": "float32",
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "DEFLATE",
        "blocksize": 256,
        "nodata": np.nan,
        "BIGTIFF": "IF_SAFER",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.update_tags(horizon=horizon, ref_time=pd.to_datetime(ref_time).isoformat(), grid_res_deg=str(grid_res))


def build_interp_layers(
    stations: pd.DataFrame,
    accum_latest: pd.DataFrame,
    horizons_h: dict[str, int],
    interp_dir: Path,
    grid_res: float = 0.1,
    power: float = 2.0,
) -> list[str]:
    interp_dir.mkdir(parents=True, exist_ok=True)
    merged = stations.merge(accum_latest, on="station_id", how="inner")
    ref_time = merged["datetime"].max()
    generated_layers: list[str] = []
    for label in horizons_h.keys():
        col = f"rain_acc_{label}"
        if col not in merged.columns:
            continue
        points = merged[["lat", "lon", col]].dropna()
        if points.empty:
            continue
        raster, transform = idw_interpolate(points, col, grid_res=grid_res, power=power)
        out_path = interp_dir / f"accum_{label}.tif"
        save_interp_cog(raster, transform, out_path, horizon=label, ref_time=ref_time, grid_res=grid_res)
        generated_layers.append(out_path.name)
    return generated_layers


def main(*, config_dir: str | Path | None = None, event_name: str | None = None) -> None:
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    config = load_runtime_config(config_dir=resolved_config_dir, event_name=event_name)

    horizons_h = config.get("runtime", {}).get("accum_horizons_h", DEFAULT_HORIZONS_H)
    station_files = config.get("paths", {}).get("station_files", [])
    stations = load_stations(resolve_paths(station_files) if station_files else DEFAULT_STATION_FILES)
    telem_dir = resolve_path(config.get("paths", {}).get("telemetry_dir", str(DEFAULT_TELEM_DIR)))
    accum_dir = resolve_path(config.get("paths", {}).get("accum_dir", str(DEFAULT_ACCUM_DIR)))
    interp_dir = resolve_path(config.get("paths", {}).get("interp_dir", str(DEFAULT_INTERP_DIR)))
    grid_res = float(config.get("interpolation", {}).get("grid_res_deg", 0.1))
    power = float(config.get("interpolation", {}).get("power", 2.0))
    reference_time_utc = pd.to_datetime(config["runtime"]["reference_time_utc"])
    selected_basins = list(config.get("basins", {}).get("selected_ids", []))
    detailed_basins = set(config.get("basins", {}).get("detailed_stats_ids", []))

    latest_records = []
    processed_station_count = 0
    telemetry_files = sorted(telem_dir.glob("*.csv"))

    for telemetry_path in telemetry_files:
        df = pd.read_csv(telemetry_path, usecols=["station_id", "datetime", "rain"])
        if df.empty:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df[df["datetime"] <= reference_time_utc]
        if df.empty:
            continue
        accum_df = compute_accum(df, horizons_h)
        if accum_df.empty:
            continue
        station_id = accum_df["station_id"].iloc[0]
        save_station_accum(station_id, accum_df, accum_dir)
        latest_records.append(accum_df.sort_values("datetime").tail(1))
        processed_station_count += 1

    if not latest_records:
        print("Nenhum acumulado gerado (sem dados de telemetria).")
        return

    latest = pd.concat(latest_records, ignore_index=True)
    generated_layers = build_interp_layers(
        stations, latest, horizons_h=horizons_h, interp_dir=interp_dir, grid_res=grid_res, power=power
    )

    if config.get("outputs", {}).get("write_summary_json", True):
        summary = {
            "step": "accumulate_interpolate",
            "run_id": config["runtime"]["run_id"],
            "mode": config.get("run", {}).get("mode", "operational"),
            "event_name": config.get("run", {}).get("event_name"),
            "reference_time_utc": config["runtime"]["reference_time_utc"],
            "accum_horizons_h": horizons_h,
            "grid_res_deg": grid_res,
            "idw_power": power,
            "stations_with_accum": processed_station_count,
            "layers_generated": generated_layers,
            "selected_basins_count": len(selected_basins),
        }
        report_dir = get_report_dir(config)
        write_json(report_dir / "accumulate_interpolate_summary.json", summary)
        if config.get("outputs", {}).get("write_basin_json", True):
            basin_stats = {
                "schema_version": config.get("schema_version", "1.0"),
                "step": "accumulate_interpolate",
                "run_id": config["runtime"]["run_id"],
                "reference_time_utc": config["runtime"]["reference_time_utc"],
                "basins": [
                    {
                        "basin_id": basin_id,
                        "detailed": basin_id in detailed_basins,
                        "status": "pending_stats_implementation",
                    }
                    for basin_id in selected_basins
                ],
            }
            write_json(report_dir / "basin_stats.json", basin_stats)

    print("Acumulados salvos em data/accum/ e grades IDW em data/interp/")


if __name__ == "__main__":
    main()
