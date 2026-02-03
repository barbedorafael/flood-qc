from __future__ import annotations

"""
Gera rasters interpolados (IDW) a partir dos CSVs acumulados por estação.

Entradas:
- data/estacoes_*.csv
- data/accum/{estacao}_{yyyymmdd}_{hhmm}_{horizonte}.csv

Saídas:
- data/interp/accum_{yyyymmdd}_{hhmm}_{horizonte}.tif
  (yyyymmdd/hhmm representam reference_time_utc - horizonte)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from config_loader import load_runtime_config, resolve_paths, resolve_path, get_report_dir, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
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
    df = df[["station_id", "lat", "lon"]].dropna()
    df["station_id"] = df["station_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


def build_window_start_utc(reference_time_utc: pd.Timestamp, horizon_hours: int) -> pd.Timestamp:
    return reference_time_utc - pd.Timedelta(hours=horizon_hours)


def build_accum_glob_pattern(reference_time_utc: pd.Timestamp, horizon_hours: int, horizon_label: str) -> str:
    window_start_utc = build_window_start_utc(reference_time_utc, horizon_hours)
    return f"*_{window_start_utc:%Y%m%d}_{window_start_utc:%H%M}_{horizon_label}.csv"


def build_interp_filename(reference_time_utc: pd.Timestamp, horizon_hours: int, horizon_label: str) -> str:
    window_start_utc = build_window_start_utc(reference_time_utc, horizon_hours)
    return f"accum_{window_start_utc:%Y%m%d}_{window_start_utc:%H%M}_{horizon_label}.tif"


def load_horizon_accum(
    accum_dir: Path,
    *,
    reference_time_utc: pd.Timestamp,
    horizon_label: str,
    horizon_hours: int,
) -> tuple[pd.DataFrame, list[str]]:
    pattern = build_accum_glob_pattern(reference_time_utc, horizon_hours, horizon_label)
    matched_files = sorted(accum_dir.glob(pattern))
    if not matched_files:
        return pd.DataFrame(columns=["station_id", "rain_acc_mm"]), []

    rows: list[pd.DataFrame] = []
    used_files: list[str] = []
    for csv_path in matched_files:
        try:
            df = pd.read_csv(csv_path, usecols=["station_id", "rain_acc_mm"])
        except (FileNotFoundError, ValueError):
            continue
        if df.empty:
            continue

        df["station_id"] = df["station_id"].astype(str)
        df["rain_acc_mm"] = pd.to_numeric(df["rain_acc_mm"], errors="coerce")
        df = df.dropna(subset=["station_id", "rain_acc_mm"])
        if df.empty:
            continue

        rows.append(df[["station_id", "rain_acc_mm"]])
        used_files.append(csv_path.name)

    if not rows:
        return pd.DataFrame(columns=["station_id", "rain_acc_mm"]), []

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset="station_id", keep="last")
    return out, used_files


def idw_interpolate(
    points: pd.DataFrame, value_col: str, grid_res: float = 0.1, power: float = 2.0
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Interpola valores via IDW em uma grade regular (lat/lon) e retorna raster + transform.
    Mantém resolução em graus; amostra o IDW nos centros dos pixels.
    """
    lats = points["lat"].to_numpy()
    lons = points["lon"].to_numpy()
    vals = points[value_col].to_numpy()

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Linha 0 no raster precisa ser o norte; por isso lat decrescente.
    grid_lat = np.arange(lat_max, lat_min - grid_res, -grid_res)
    grid_lon = np.arange(lon_min, lon_max + grid_res, grid_res)
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

    interpolated = np.zeros_like(lat_grid, dtype=float)
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            dy = lat_grid[i, j] - lats
            dx = lon_grid[i, j] - lons
            dist = np.hypot(dx, dy)
            dist[dist == 0] = 1e-6
            weights = 1 / (dist ** power)
            interpolated[i, j] = np.sum(weights * vals) / np.sum(weights)

    transform = from_origin(
        lon_min - (grid_res / 2),
        lat_max + (grid_res / 2),
        grid_res,
        grid_res,
    )
    return interpolated.astype("float32"), transform


def save_interp_cog(
    array: np.ndarray,
    transform,
    out_path: Path,
    *,
    horizon: str,
    ref_time: pd.Timestamp,
    window_start_utc: pd.Timestamp,
    grid_res: float,
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
        dst.update_tags(
            horizon=horizon,
            ref_time=pd.to_datetime(ref_time, utc=True).isoformat(),
            window_start_utc=pd.to_datetime(window_start_utc, utc=True).isoformat(),
            grid_res_deg=str(grid_res),
        )


def build_interp_layers(
    *,
    stations: pd.DataFrame,
    horizons_h: dict[str, int],
    accum_dir: Path,
    interp_dir: Path,
    reference_time_utc: pd.Timestamp,
    grid_res: float = 0.1,
    power: float = 2.0,
) -> tuple[list[str], dict[str, list[str]]]:
    interp_dir.mkdir(parents=True, exist_ok=True)

    generated_layers: list[str] = []
    used_accum_files_by_horizon: dict[str, list[str]] = {}

    for horizon_label, horizon_hours in horizons_h.items():
        accum_rows, used_files = load_horizon_accum(
            accum_dir,
            reference_time_utc=reference_time_utc,
            horizon_label=horizon_label,
            horizon_hours=int(horizon_hours),
        )
        used_accum_files_by_horizon[horizon_label] = used_files
        if accum_rows.empty:
            continue

        merged = stations.merge(accum_rows, on="station_id", how="inner")
        points = merged[["lat", "lon", "rain_acc_mm"]].dropna()
        if points.empty:
            continue

        raster, transform = idw_interpolate(points, "rain_acc_mm", grid_res=grid_res, power=power)
        out_name = build_interp_filename(reference_time_utc, int(horizon_hours), horizon_label)
        out_path = interp_dir / out_name
        save_interp_cog(
            raster,
            transform,
            out_path,
            horizon=horizon_label,
            ref_time=reference_time_utc,
            window_start_utc=build_window_start_utc(reference_time_utc, int(horizon_hours)),
            grid_res=grid_res,
        )
        generated_layers.append(out_name)

    return generated_layers, used_accum_files_by_horizon


def main(*, config_dir: str | Path | None = None, event_name: str | None = None) -> None:
    resolved_config_dir = Path(config_dir) if config_dir is not None else None
    config = load_runtime_config(config_dir=resolved_config_dir, event_name=event_name)

    horizons_h = config.get("runtime", {}).get("accum_horizons_h", DEFAULT_HORIZONS_H)
    station_files = config.get("paths", {}).get("station_files", [])
    stations = load_stations(resolve_paths(station_files) if station_files else DEFAULT_STATION_FILES)

    accum_dir = resolve_path(config.get("paths", {}).get("accum_dir", str(DEFAULT_ACCUM_DIR)))
    interp_dir = resolve_path(config.get("paths", {}).get("interp_dir", str(DEFAULT_INTERP_DIR)))
    grid_res = float(config.get("interpolation", {}).get("grid_res_deg", 0.1))
    power = float(config.get("interpolation", {}).get("power", 2.0))
    reference_time_utc = pd.to_datetime(config["runtime"]["reference_time_utc"], utc=True)
    selected_basins = list(config.get("basins", {}).get("selected_ids", []))
    detailed_basins = set(config.get("basins", {}).get("detailed_stats_ids", []))

    generated_layers, used_accum_files_by_horizon = build_interp_layers(
        stations=stations,
        horizons_h=horizons_h,
        accum_dir=accum_dir,
        interp_dir=interp_dir,
        reference_time_utc=reference_time_utc,
        grid_res=grid_res,
        power=power,
    )

    if config.get("outputs", {}).get("write_summary_json", True):
        summary = {
            "step": "interpolate",
            "run_id": config["runtime"]["run_id"],
            "mode": config.get("run", {}).get("mode", "operational"),
            "event_name": config.get("run", {}).get("event_name"),
            "reference_time_utc": config["runtime"]["reference_time_utc"],
            "accum_horizons_h": horizons_h,
            "grid_res_deg": grid_res,
            "idw_power": power,
            "layers_generated": generated_layers,
            "used_accum_files_by_horizon": used_accum_files_by_horizon,
            "accum_input_pattern": "{station}_{yyyymmdd}_{hhmm}_{horizon}.csv",
            "interp_output_pattern": "accum_{yyyymmdd}_{hhmm}_{horizon}.tif",
            "selected_basins_count": len(selected_basins),
        }
        report_dir = get_report_dir(config)
        write_json(report_dir / "interpolate_summary.json", summary)

        if config.get("outputs", {}).get("write_basin_json", True):
            basin_stats = {
                "schema_version": config.get("schema_version", "1.0"),
                "step": "interpolate",
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

    if not generated_layers:
        print("Nenhuma camada interpolada gerada (sem CSVs acumulados válidos para interpolação).")
        return

    print(f"Grades IDW salvas em {interp_dir} ({len(generated_layers)} camadas).")


if __name__ == "__main__":
    main()
