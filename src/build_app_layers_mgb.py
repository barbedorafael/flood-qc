from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SPATIAL_DIR = DATA_DIR / "spatial"
APP_LAYERS_DIR = DATA_DIR / "app_layers"
MGB_PROCESSED_DIR = DATA_DIR / "mgb-hora" / "processed"

DEFAULT_RIOS_SHP = SPATIAL_DIR / "Rios.shp"
DEFAULT_Q_META = MGB_PROCESSED_DIR / "q_meta.json"
DEFAULT_OUT_GEOJSON = APP_LAYERS_DIR / "rios_mini.geojson"
DEFAULT_OUT_META = APP_LAYERS_DIR / "rios_mini_meta.json"


def to_repo_relative_posix(path: Path) -> str:
    path_abs = path.resolve()
    repo_abs = REPO_ROOT.resolve()
    try:
        rel = path_abs.relative_to(repo_abs)
    except ValueError:
        rel = path_abs
    return rel.as_posix()


def read_nc_from_q_meta(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    nc = payload.get("nc")
    if nc is None:
        return None
    return int(nc)


def build_rivers_layer(
    rios_shp_path: Path = DEFAULT_RIOS_SHP,
    q_meta_path: Path = DEFAULT_Q_META,
    out_geojson_path: Path = DEFAULT_OUT_GEOJSON,
    out_meta_path: Path = DEFAULT_OUT_META,
) -> None:
    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script requires geopandas/shapely. Install with: pip install geopandas shapely"
        ) from exc

    if not rios_shp_path.exists():
        raise FileNotFoundError(f"Missing shapefile: {rios_shp_path}")

    gdf = gpd.read_file(rios_shp_path)
    input_rows = int(len(gdf))
    if "Mini" not in gdf.columns:
        raise ValueError(f"Rios.shp missing required column 'Mini': {rios_shp_path}")

    gdf["mini_id"] = pd.to_numeric(gdf["Mini"], errors="coerce")
    gdf = gdf.dropna(subset=["mini_id", "geometry"]).copy()
    gdf["mini_id"] = gdf["mini_id"].astype("int32")
    gdf = gdf[gdf["mini_id"] > 0].copy()

    counts = (
        gdf.groupby("mini_id", as_index=False)
        .size()
        .rename(columns={"size": "parts_count"})
    )
    duplicated_mini_count = int((counts["parts_count"] > 1).sum())

    dissolved = gdf[["mini_id", "geometry"]].dissolve(by="mini_id", as_index=False)
    rivers_layer = dissolved.merge(counts, on="mini_id", how="left")
    rivers_layer["parts_count"] = rivers_layer["parts_count"].fillna(1).astype("int32")
    rivers_layer = rivers_layer.sort_values("mini_id").reset_index(drop=True)

    out_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    rivers_layer.to_file(out_geojson_path, driver="GeoJSON")

    unique_mini = int(rivers_layer["mini_id"].nunique())
    nc = read_nc_from_q_meta(q_meta_path)
    missing_mini_count = None
    missing_mini_sample: list[int] = []
    if nc is not None:
        existing = set(rivers_layer["mini_id"].astype(int).tolist())
        expected = set(range(1, int(nc) + 1))
        missing = sorted(expected - existing)
        missing_mini_count = int(len(missing))
        missing_mini_sample = missing[:20]

    payload = {
        "layer_name": "rios_mini",
        "input_file": to_repo_relative_posix(rios_shp_path),
        "output_geojson": to_repo_relative_posix(out_geojson_path),
        "q_meta_file": to_repo_relative_posix(q_meta_path),
        "input_rows": input_rows,
        "output_rows": int(len(rivers_layer)),
        "unique_mini": unique_mini,
        "duplicated_mini_count": duplicated_mini_count,
        "nc_reference": int(nc) if nc is not None else None,
        "missing_mini_count": missing_mini_count,
        "missing_mini_sample": missing_mini_sample,
        "columns": ["mini_id", "parts_count", "geometry"],
    }
    out_meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"GeoJSON written: {to_repo_relative_posix(out_geojson_path)}")
    print(f"Metadata written: {to_repo_relative_posix(out_meta_path)}")


def main() -> None:
    parser = ArgumentParser(description="Build app layer GeoJSON from Rios.shp grouped by mini_id.")
    parser.add_argument(
        "--rios-shp",
        type=Path,
        default=DEFAULT_RIOS_SHP,
        help="Path to Rios.shp (default: data/spatial/Rios.shp).",
    )
    parser.add_argument(
        "--q-meta",
        type=Path,
        default=DEFAULT_Q_META,
        help="Path to q_meta.json used for NC reference (default: data/mgb-hora/processed/q_meta.json).",
    )
    parser.add_argument(
        "--out-geojson",
        type=Path,
        default=DEFAULT_OUT_GEOJSON,
        help="Output GeoJSON path (default: data/app_layers/rios_mini.geojson).",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=DEFAULT_OUT_META,
        help="Output metadata JSON path (default: data/app_layers/rios_mini_meta.json).",
    )
    args = parser.parse_args()

    build_rivers_layer(
        rios_shp_path=args.rios_shp,
        q_meta_path=args.q_meta,
        out_geojson_path=args.out_geojson,
        out_meta_path=args.out_meta,
    )


if __name__ == "__main__":
    main()
