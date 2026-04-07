from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.raster_layers import ImageOverlay

from common.paths import REPO_ROOT, history_db_path
from common.time_utils import TIMEZONE
from ingest.forecast_grid import ECMWF_ASSET_KIND, TpGribMessage, read_tp_grib_messages
from qc.ecmwf_forecast_correction import ForecastCorrectionInstruction, apply_correction_sequence
from reporting import ops_dashboard_data, ops_dashboard_map


@dataclass(frozen=True, slots=True)
class ForecastPreview:
    asset_id: str
    relative_path: str
    data: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    t0_step: int
    t1_step: int
    mode_label: str
    title: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(np.min(self.longitudes)),
            float(np.min(self.latitudes)),
            float(np.max(self.longitudes)),
            float(np.max(self.latitudes)),
        )


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_repo_path(relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _read_asset_row(asset_id: str, *, database_path: Path | None = None) -> dict[str, object]:
    history_path = database_path or history_db_path()
    with _connect(history_path) as connection:
        row = connection.execute(
            """
            SELECT
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json,
                created_at
            FROM asset
            WHERE asset_id = ?
            """,
            (asset_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Asset ECMWF {asset_id!r} nao encontrado no history.sqlite.")
    return dict(row)


def _to_local_time(raw_value: datetime) -> datetime:
    return raw_value.replace(tzinfo=timezone.utc).astimezone(TIMEZONE).replace(tzinfo=None)


def _load_local_messages(asset_id: str, *, database_path: Path | None = None) -> tuple[dict[str, object], list[TpGribMessage]]:
    asset_row = _read_asset_row(asset_id, database_path=database_path)
    if str(asset_row["asset_kind"]) != ECMWF_ASSET_KIND:
        raise ValueError(f"Asset {asset_id!r} nao e do tipo {ECMWF_ASSET_KIND!r}.")

    source_path = _resolve_repo_path(str(asset_row["relative_path"]))
    messages = read_tp_grib_messages(source_path)
    localized = [
        TpGribMessage(
            valid_time=_to_local_time(message.valid_time),
            step_hours=int(message.step_hours),
            latitudes=np.asarray(message.latitudes, dtype=np.float64),
            longitudes=np.asarray(message.longitudes, dtype=np.float64),
            values_mm=np.asarray(message.values_mm, dtype=np.float64),
        )
        for message in messages
    ]
    return asset_row, localized


def list_forecast_assets(database_path: Path | None = None) -> pd.DataFrame:
    history_path = database_path or history_db_path()
    with _connect(history_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                asset_id,
                asset_kind,
                format,
                relative_path,
                provider_code,
                checksum,
                valid_from,
                valid_to,
                metadata_json,
                created_at
            FROM asset
            WHERE provider_code = 'ecmwf'
              AND asset_kind = ?
            ORDER BY COALESCE(valid_from, created_at) DESC, created_at DESC
            """,
            connection,
            params=(ECMWF_ASSET_KIND,),
        )
    if frame.empty:
        return frame

    frame["metadata"] = frame["metadata_json"].apply(lambda value: json.loads(value) if value else {})
    frame["cycle_time"] = frame["metadata"].apply(lambda value: value.get("cycle_time") if isinstance(value, dict) else None)
    frame["display_label"] = frame.apply(
        lambda row: f"{row['asset_id']} | ciclo {row['cycle_time'] or row['valid_from'] or 'sem ciclo'}",
        axis=1,
    )
    return frame.drop(columns=["metadata"])


def list_forecast_steps(asset_id: str, database_path: Path | None = None) -> pd.DataFrame:
    _, messages = _load_local_messages(asset_id, database_path=database_path)
    rows = [
        {
            "step_hours": int(message.step_hours),
            "valid_time": pd.Timestamp(message.valid_time),
            "label": f"t={int(message.step_hours)}h | {message.valid_time.strftime('%d/%m %H:%M')}",
        }
        for message in messages
    ]
    return pd.DataFrame(rows).sort_values("step_hours").reset_index(drop=True)


def build_forecast_preview(
    asset_id: str,
    *,
    t0_step: int,
    t1_step: int,
    database_path: Path | None = None,
) -> ForecastPreview:
    asset_row, messages = _load_local_messages(asset_id, database_path=database_path)
    if not messages:
        raise ValueError(f"Asset ECMWF {asset_id!r} nao possui mensagens tp.")

    message_by_step = {int(message.step_hours): message for message in messages}
    if t0_step not in message_by_step:
        raise ValueError(f"t0_step={t0_step} nao existe no GRIB selecionado.")
    if t1_step not in message_by_step:
        raise ValueError(f"t1_step={t1_step} nao existe no GRIB selecionado.")
    if t1_step < t0_step:
        raise ValueError("t1_step must be >= t0_step.")

    base_step = min(message_by_step)
    end_message = message_by_step[int(t1_step)]
    start_message = message_by_step[int(t0_step)]

    if int(t0_step) == base_step:
        data = np.asarray(end_message.values_mm, dtype=np.float64).copy()
        mode_label = "acumulado_nativo"
        title = f"Acumulado ECMWF ate t={t1_step}h"
    else:
        data = np.asarray(end_message.values_mm, dtype=np.float64) - np.asarray(start_message.values_mm, dtype=np.float64)
        data = np.where(np.isfinite(data), data, np.nan)
        data[data < 0.0] = 0.0
        mode_label = "incremental"
        title = f"Incremental ECMWF entre t={t0_step}h e t={t1_step}h"

    return ForecastPreview(
        asset_id=str(asset_row["asset_id"]),
        relative_path=str(asset_row["relative_path"]),
        data=data,
        latitudes=np.asarray(end_message.latitudes, dtype=np.float64),
        longitudes=np.asarray(end_message.longitudes, dtype=np.float64),
        t0_step=int(t0_step),
        t1_step=int(t1_step),
        mode_label=mode_label,
        title=title,
    )


def apply_preview_corrections(
    preview: ForecastPreview,
    instructions: list[ForecastCorrectionInstruction],
) -> ForecastPreview:
    corrected_data = apply_correction_sequence(preview.data, instructions)
    return replace(preview, data=corrected_data, title=f"{preview.title} | corrigido")


def export_preview_raster(preview: ForecastPreview, target_path: Path) -> Path:
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Exportacao de raster de forecast requer rasterio. Instale as dependencias geo/ui antes de usar a aba ECMWF."
        ) from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = preview.bounds
    rows, cols = preview.data.shape
    transform = from_bounds(west, south, east, north, cols, rows)
    data = np.flipud(np.asarray(preview.data, dtype=np.float32))

    with rasterio.open(
        target_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return target_path


def build_forecast_map(preview: ForecastPreview, *, opacity: float = 0.7) -> folium.Map:
    west, south, east, north = preview.bounds
    center = [float((south + north) / 2.0), float((west + east) / 2.0)]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB Positron", control_scale=True)

    rivers_geojson = ops_dashboard_data.load_rivers_layer_geojson()
    if rivers_geojson and rivers_geojson.get("features"):
        folium.GeoJson(
            rivers_geojson,
            style_function=lambda _: {"color": "#1971c2", "weight": 1.0, "opacity": 0.35},
            name="Rios MGB",
        ).add_to(fmap)

    finite_values = np.asarray(preview.data, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size > 0:
        vmin, vmax = np.nanpercentile(preview.data, [5, 95])
        overlay = ImageOverlay(
            name=preview.title,
            image=np.asarray(preview.data, dtype=np.float64),
            bounds=[[south, west], [north, east]],
            opacity=float(opacity),
            interactive=False,
            cross_origin=False,
            mercator_project=False,
            colormap=ops_dashboard_map.color_ramp_factory(float(vmin), float(vmax), float(opacity)),
        )
        layer = folium.FeatureGroup(name=preview.title, show=True)
        overlay.add_to(layer)
        layer.add_to(fmap)
        ops_dashboard_map.add_legend(fmap, float(vmin), float(vmax), horizon_label=preview.title)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def build_forecast_map_artifacts(
    preview: ForecastPreview,
    *,
    opacity: float = 0.7,
    component_key: str = "forecast-preview-map",
) -> ops_dashboard_map.MapRenderArtifacts:
    fmap = build_forecast_map(preview, opacity=opacity)
    return ops_dashboard_map.build_map_render_artifacts(fmap, component_key=component_key)
