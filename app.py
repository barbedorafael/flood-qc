from __future__ import annotations

"""
Dashboard Streamlit para explorar estações e grades interpoladas de chuva,
com resumo de disponibilidade dos postos e visualizações temáticas.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import re
from typing import Optional

import numpy as np
import pandas as pd
from affine import Affine
import folium
import branca.colormap as cm
from branca.element import MacroElement, Template
from folium.raster_layers import ImageOverlay
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
import streamlit as st
from streamlit_folium import st_folium
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
TELEMETRIA_DIR = DATA_DIR / "telemetria"
INTERP_DIR = DATA_DIR / "interp"
DAYS_WINDOW = 30
NO_DATA_COLOR = "#e64980"
DATA_ISSUE_COLOR = "#f08c00"
KIND_COLORS = {"nível": "#0b7285", "chuva": "#364fc7"}
STATION_VIEW_OPTIONS: dict[str, dict[str, str]] = {
    "Tipo de estação": {},
    "Chuva média (mm/h) - 30 dias": {
        "column": "rain_mean_mm_h",
        "legend": "Chuva média (mm/h)",
    },
    "Chuva acumulada 24h (mm)": {
        "column": "rain_acc_24h_mm",
        "legend": "Chuva acumulada 24h (mm)",
    },
    "Chuva p90 (mm/h) - 30 dias": {
        "column": "rain_p90_mm_h",
        "legend": "Chuva p90 (mm/h)",
    },
}

# Paleta fixa Blues (claro→escuro)
BLUES = np.array(
    [
        (239, 243, 255),
        (198, 219, 239),
        (158, 202, 225),
        (107, 174, 214),
        (66, 146, 198),
        (33, 113, 181),
        (8, 81, 156),
        (8, 48, 107),
    ],
    dtype=float,
) / 255.0


# ---------- Estilos ----------
st.set_page_config(page_title="Explorador de Estações RS", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', 'Helvetica Neue', sans-serif;
        background: radial-gradient(circle at 10% 20%, #f2f7fb, #e8f1f7 40%, #e5ecf3 100%);
    }
    .metric-card {
        background: #0b7285;
        color: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Utilidades de leitura ----------
@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_name, kind in [("estacoes_nivel.csv", "nível"), ("estacoes_pluv.csv", "chuva")]:
        path = DATA_DIR / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        keep = {"CODIGO": "station_id", "LAT": "lat", "LON": "lon", "NOME": "name"}
        df = df.rename(columns=keep)
        df = df[list(keep.values())].copy()
        df["kind"] = kind
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "name", "kind"])
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="station_id")
    for col in ["lat", "lon"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["station_id"] = merged["station_id"].astype(str).str.strip()
    merged["name"] = merged["name"].fillna("").astype(str).str.strip()
    merged = merged.dropna(subset=["lat", "lon"])
    return merged


@st.cache_data(show_spinner=False)
def load_timeseries(station_id: str, days: int = 30) -> pd.DataFrame:
    csv_path = TELEMETRIA_DIR / f"{station_id}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["station_id", "datetime", "rain", "level", "flow"])
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["rain"] = pd.to_numeric(df.get("rain"), errors="coerce")
    df["level"] = pd.to_numeric(df.get("level"), errors="coerce")
    df["flow"] = pd.to_numeric(df.get("flow"), errors="coerce")
    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df["datetime"] >= cutoff].sort_values("datetime")
    return df


@st.cache_data(show_spinner=False)
def list_rasters() -> list[dict]:
    rasters = []
    for tif in sorted(INTERP_DIR.glob("*.tif")):
        try:
            with rasterio.open(tif) as src:
                tags = src.tags()
                rasters.append(
                    {
                        "path": tif,
                        "name": tif.stem,
                        "shape": src.shape,
                        "tags": tags,
                    }
                )
        except rasterio.RasterioError:
            continue
    return rasters


@st.cache_data(show_spinner=False)
def load_raster_data(path: Path, max_size: int = 600) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Lê raster e devolve array float32 e bounds (west, south, east, north).
    Downsample automático para no máx. max_size pixels em cada dimensão.
    Mascara valores <= 0 como NaN (chuva zero não deve aparecer).
    """
    with rasterio.open(path) as src:
        scale = min(max_size / src.height, max_size / src.width, 1.0)
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))
        data = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        )
        data = data.astype("float32")
        data[data <= 0] = np.nan  # mascara chuva zero/negativa
        data[data <= -1e20] = np.nan  # tolera nodata

        # Ajusta transform para o raster reamostrado
        scale_x = src.width / out_w
        scale_y = src.height / out_h
        new_transform = src.transform * Affine.scale(scale_x, scale_y)
        west, south, east, north = array_bounds(out_h, out_w, new_transform)
    return data, (west, south, east, north)


@st.cache_data(show_spinner=False)
def load_station_health(station_ids: tuple[str, ...], days: int = 30) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows: list[dict[str, object]] = []

    for station_id in station_ids:
        csv_path = TELEMETRIA_DIR / f"{station_id}.csv"
        if not csv_path.exists():
            rows.append(
                {
                    "station_id": station_id,
                    "status": "no_data",
                    "status_reason": "arquivo ausente",
                    "rows_recent": 0,
                    "rain_mean_mm_h": np.nan,
                    "rain_acc_24h_mm": np.nan,
                    "rain_p90_mm_h": np.nan,
                }
            )
            continue

        try:
            df = pd.read_csv(csv_path, usecols=["datetime", "rain", "level", "flow"])
        except Exception:
            rows.append(
                {
                    "station_id": station_id,
                    "status": "data_issue",
                    "status_reason": "erro de leitura",
                    "rows_recent": 0,
                    "rain_mean_mm_h": np.nan,
                    "rain_acc_24h_mm": np.nan,
                    "rain_p90_mm_h": np.nan,
                }
            )
            continue

        if df.empty:
            rows.append(
                {
                    "station_id": station_id,
                    "status": "no_data",
                    "status_reason": "arquivo vazio",
                    "rows_recent": 0,
                    "rain_mean_mm_h": np.nan,
                    "rain_acc_24h_mm": np.nan,
                    "rain_p90_mm_h": np.nan,
                }
            )
            continue

        total_rows = len(df)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        invalid_dt = int(df["datetime"].isna().sum())
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        recent = df[df["datetime"] >= cutoff].copy()

        if recent.empty:
            rows.append(
                {
                    "station_id": station_id,
                    "status": "no_data",
                    "status_reason": f"sem registros nos últimos {days} dias",
                    "rows_recent": 0,
                    "rain_mean_mm_h": np.nan,
                    "rain_acc_24h_mm": np.nan,
                    "rain_p90_mm_h": np.nan,
                }
            )
            continue

        recent["rain"] = pd.to_numeric(recent["rain"], errors="coerce")
        recent["level"] = pd.to_numeric(recent["level"], errors="coerce")
        recent["flow"] = pd.to_numeric(recent["flow"], errors="coerce")

        rain_valid = recent["rain"].dropna()
        level_valid = recent["level"].dropna()
        flow_valid = recent["flow"].dropna()
        duplicate_time_mask = recent.duplicated(subset=["datetime"], keep=False)
        duplicate_nonzero_rain_mask = duplicate_time_mask & recent["rain"].fillna(0).abs().gt(0)
        duplicate_ratio = float(duplicate_nonzero_rain_mask.sum() / max(len(recent), 1))

        issues = []
        if invalid_dt > 0:
            issues.append("datetime inválido")
        if (invalid_dt / max(total_rows, 1)) > 0.2:
            issues.append("muitos datetime inválidos")
        if duplicate_ratio > 0.2:
            issues.append("muitos horários repetidos com chuva > 0")
        if rain_valid.empty and level_valid.empty and flow_valid.empty:
            issues.append("sem variáveis válidas")

        latest_time = recent["datetime"].max()
        rain_24h = recent.loc[
            recent["datetime"] >= latest_time - timedelta(hours=24), "rain"
        ].sum(min_count=1)

        rows.append(
            {
                "station_id": station_id,
                "status": "data_issue" if issues else "ok",
                "status_reason": "; ".join(issues) if issues else "",
                "rows_recent": int(len(recent)),
                "rain_mean_mm_h": float(rain_valid.mean()) if not rain_valid.empty else np.nan,
                "rain_acc_24h_mm": float(rain_24h) if pd.notna(rain_24h) else np.nan,
                "rain_p90_mm_h": float(rain_valid.quantile(0.9)) if not rain_valid.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def merge_station_context(stations: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if stations.empty:
        out = stations.copy()
        out["status"] = pd.Series(dtype="object")
        out["status_reason"] = pd.Series(dtype="object")
        out["rain_mean_mm_h"] = pd.Series(dtype="float64")
        out["rain_acc_24h_mm"] = pd.Series(dtype="float64")
        out["rain_p90_mm_h"] = pd.Series(dtype="float64")
        return out
    health = load_station_health(tuple(stations["station_id"].tolist()), days=days)
    merged = stations.merge(health, on="station_id", how="left")
    merged["status"] = merged["status"].fillna("no_data")
    merged["status_reason"] = merged["status_reason"].fillna("")
    return merged


def format_mm(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f} mm"


def network_summary(stations: pd.DataFrame) -> dict[str, float]:
    if stations.empty or "status" not in stations:
        return {
            "total": 0.0,
            "with_data": 0.0,
            "no_data": 0.0,
            "data_issue": 0.0,
            "rain_mean_24h": np.nan,
            "rain_median_24h": np.nan,
            "rain_p90_24h": np.nan,
        }

    total = float(len(stations))
    no_data = float((stations["status"] == "no_data").sum())
    data_issue = float((stations["status"] == "data_issue").sum())
    with_data = float((stations["status"] == "ok").sum())

    rain_values = stations.loc[stations["status"] == "ok", "rain_acc_24h_mm"].dropna()
    if rain_values.empty:
        rain_values = stations["rain_acc_24h_mm"].dropna()

    return {
        "total": total,
        "with_data": with_data,
        "no_data": no_data,
        "data_issue": data_issue,
        "rain_mean_24h": float(rain_values.mean()) if not rain_values.empty else np.nan,
        "rain_median_24h": float(rain_values.median()) if not rain_values.empty else np.nan,
        "rain_p90_24h": float(rain_values.quantile(0.9)) if not rain_values.empty else np.nan,
    }


def render_network_summary(stations: pd.DataFrame) -> None:
    summary = network_summary(stations)
    cols = st.columns(7)
    cols[0].metric("Postos totais", f"{int(summary['total'])}")
    cols[1].metric("Com dados", f"{int(summary['with_data'])}")
    cols[2].metric("Sem dados", f"{int(summary['no_data'])}")
    cols[3].metric("Falha de dados", f"{int(summary['data_issue'])}")
    cols[4].metric("Média chuva 24h", format_mm(summary["rain_mean_24h"]))
    cols[5].metric("Mediana chuva 24h", format_mm(summary["rain_median_24h"]))
    cols[6].metric("P90 chuva 24h", format_mm(summary["rain_p90_24h"]))


def color_ramp_factory(vmin: float, vmax: float, alpha: float):
    stops = np.linspace(0, 1, len(BLUES))

    def cmap(val: float):
        if val is None or np.isnan(val):
            return (0, 0, 0, 0)
        span = vmax - vmin
        t = 0.5 if span <= 0 else (val - vmin) / span
        t = float(np.clip(t, 0.0, 1.0))
        r = float(np.interp(t, stops, BLUES[:, 0]))
        g = float(np.interp(t, stops, BLUES[:, 1]))
        b = float(np.interp(t, stops, BLUES[:, 2]))
        return (r, g, b, alpha)

    return cmap


# ---------- Componentes de UI ----------
def add_legend(fmap: folium.Map, vmin: float, vmax: float, *, horizon_label: Optional[str]) -> None:
    colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b in BLUES]
    colormap = cm.LinearColormap(colors=colors_hex, vmin=float(vmin), vmax=float(vmax))
    horizon_text = horizon_label or "período selecionado"
    colormap.caption = f"Chuva acumulada das últimas {horizon_text}"
    colormap.add_to(fmap)


def extract_horizon_label(layer_name: Optional[str]) -> Optional[str]:
    if not layer_name:
        return None
    match = re.search(r"_(\d+h)$", layer_name)
    if match:
        return match.group(1)
    return None


def sanitize_hex_color(value: str) -> str:
    if value.startswith("#") and len(value) >= 7:
        return value[:7]
    return value


def add_station_legend(
    fmap: folium.Map, *, metric_values: pd.Series, legend_title: str
) -> Optional[cm.LinearColormap]:
    clean = pd.to_numeric(metric_values, errors="coerce").dropna()
    if clean.empty:
        return None

    vmin, vmax = np.nanpercentile(clean, [10, 90])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if abs(vmax - vmin) < 1e-9:
        vmin -= 0.5
        vmax += 0.5

    station_colormap = cm.LinearColormap(
        colors=["#f7fbff", "#9ecae1", "#4292c6", "#08519c"],
        vmin=float(vmin),
        vmax=float(vmax),
    )
    station_colormap.caption = legend_title
    station_colormap.add_to(fmap)
    return station_colormap


def format_popup_metric(value: object, suffix: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f} {suffix}"


def render_selected_station_context(row: Optional[pd.Series], station_id: str) -> None:
    if row is None:
        st.markdown(f"### Posto selecionado: {station_id}")
        return

    status_labels = {
        "ok": "ok",
        "data_issue": "falha de dados",
        "no_data": "sem dados",
    }
    station_name = str(row.get("name", "")).strip() or "sem nome"
    status = status_labels.get(str(row.get("status", "no_data")), "sem dados")
    kind = str(row.get("kind", "—"))
    reason = str(row.get("status_reason", "")).strip()

    st.markdown(f"### Posto selecionado: {station_name}")
    st.caption(f"Código: {station_id} | Tipo: {kind} | Status: {status}")
    st.caption(
        " | ".join(
            [
                f"Chuva média: {format_popup_metric(row.get('rain_mean_mm_h', np.nan), 'mm/h')}",
                f"Chuva acum. 24h: {format_popup_metric(row.get('rain_acc_24h_mm', np.nan), 'mm')}",
                f"Chuva p90: {format_popup_metric(row.get('rain_p90_mm_h', np.nan), 'mm/h')}",
            ]
        )
    )
    if reason:
        st.caption(f"Obs: {reason}")


class RasterClickPopup(MacroElement):
    def __init__(self, data: np.ndarray, bounds: tuple[float, float, float, float], layer_name: str) -> None:
        super().__init__()
        west, south, east, north = bounds
        payload = np.where(np.isnan(data), None, data).tolist()
        self._name = "RasterClickPopup"
        self.data = json.dumps(payload)
        self.south = south
        self.west = west
        self.north = north
        self.east = east
        self.layer_name = json.dumps(layer_name)
        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            var rasterData = {{this.data}};
            var rasterBounds = {south: {{this.south}}, west: {{this.west}}, north: {{this.north}}, east: {{this.east}}};
            (function attachRasterClick() {
                var mapRef = window.map;
                if (!mapRef) {
                    setTimeout(attachRasterClick, 50);
                    return;
                }
                if (mapRef._rasterClickHandler) {
                    mapRef.off('click', mapRef._rasterClickHandler);
                }
                mapRef._rasterClickHandler = function(e) {
                    var lat = e.latlng.lat;
                    var lng = e.latlng.lng;
                    if (lat < rasterBounds.south || lat > rasterBounds.north || lng < rasterBounds.west || lng > rasterBounds.east) {
                        return;
                    }
                    var rows = rasterData.length;
                    var cols = rasterData[0].length;
                    var row = Math.floor((rasterBounds.north - lat) / (rasterBounds.north - rasterBounds.south) * (rows - 1));
                    var col = Math.floor((lng - rasterBounds.west) / (rasterBounds.east - rasterBounds.west) * (cols - 1));
                    var val = rasterData[row][col];
                    if (val === null || isNaN(val) || val <= 0) {
                        return;
                    }
                    var layerName = {{this.layer_name}};
                    var html = `<b>${layerName}</b><br>Lat: ${lat.toFixed(4)}<br>Lon: ${lng.toFixed(4)}<br>Valor: ${val.toFixed(1)} mm`;
                    L.popup().setLatLng(e.latlng).setContent(html).openOn(mapRef);
                };
                mapRef.on('click', mapRef._rasterClickHandler);
            })();
            {% endmacro %}
            """
        )


def build_map(
    selected_layer: Optional[str],
    opacity: float,
    stations: pd.DataFrame,
    station_view_mode: str,
) -> folium.Map:
    center = (
        [stations["lat"].mean(), stations["lon"].mean()]
        if not stations.empty
        else [-29.7, -53.3]
    )
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB Positron", control_scale=True)

    catalog = {layer["name"]: layer for layer in list_rasters()}
    if selected_layer:
        meta = catalog.get(selected_layer)
        if meta:
            data, (west, south, east, north) = load_raster_data(meta["path"])
            finite_values = data[np.isfinite(data)]
            if finite_values.size > 0:
                vmin, vmax = np.nanpercentile(data, [5, 95])
                overlay = ImageOverlay(
                    name=f"Raster {selected_layer}",
                    image=data,
                    bounds=[[south, west], [north, east]],
                    opacity=opacity,
                    interactive=False,
                    cross_origin=False,
                    mercator_project=False,
                    colormap=color_ramp_factory(vmin, vmax, opacity),
                )
                raster_group = folium.FeatureGroup(name="Chuva interpolada", show=True)
                overlay.add_to(raster_group)
                raster_group.add_to(fmap)
                add_legend(
                    fmap,
                    vmin,
                    vmax,
                    horizon_label=extract_horizon_label(selected_layer),
                )
                RasterClickPopup(data, (west, south, east, north), selected_layer).add_to(fmap)

    station_view_cfg = STATION_VIEW_OPTIONS.get(station_view_mode, {})
    station_metric_col = station_view_cfg.get("column")
    station_colormap = None
    if station_metric_col:
        station_colormap = add_station_legend(
            fmap,
            metric_values=stations.loc[stations["status"] != "no_data", station_metric_col],
            legend_title=station_view_cfg["legend"],
        )

    station_layer = folium.FeatureGroup(name="Postos com dados", show=True)
    no_data_layer = folium.FeatureGroup(name="Postos sem dados", show=True)
    for row in stations.itertuples():
        station_name = row.name if row.name else "sem nome"
        tooltip = f"{row.station_id} — {station_name}"
        status = getattr(row, "status", "no_data")
        popup_html = station_name

        if status == "no_data":
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=6,
                color=NO_DATA_COLOR,
                fill=True,
                fill_color=NO_DATA_COLOR,
                weight=1,
                fill_opacity=0.55,
                tooltip=tooltip,
                popup=popup_html,
            ).add_to(no_data_layer)
            continue

        if station_metric_col and station_colormap is not None:
            metric_value = getattr(row, station_metric_col, np.nan)
            if pd.notna(metric_value):
                marker_color = sanitize_hex_color(station_colormap(float(metric_value)))
            elif status == "data_issue":
                marker_color = DATA_ISSUE_COLOR
            else:
                marker_color = "#adb5bd"
        else:
            marker_color = KIND_COLORS.get(getattr(row, "kind", ""), "#364fc7")
            if status == "data_issue":
                marker_color = DATA_ISSUE_COLOR

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            weight=1,
            fill_opacity=0.9 if status == "ok" else 0.75,
            tooltip=tooltip,
            popup=popup_html,
        ).add_to(station_layer)

    station_layer.add_to(fmap)
    no_data_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def time_series_chart(df: pd.DataFrame, station_id: str, days: int):
    if df.empty:
        st.info("Sem dados para esta estação/intervalo.")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.35, 0.65],
    )

    fig.add_bar(
        x=df["datetime"],
        y=df["rain"],
        name="Chuva (mm)",
        marker_color="#4dabf7",
        opacity=0.9,
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=df["datetime"],
        y=df["level"],
        name="Nível (cm)",
        mode="lines+markers",
        line=dict(color="#0b7285", width=2),
        marker=dict(size=4),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(t=30, r=20, l=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        xaxis=dict(title=""),
        xaxis2=dict(title="Data/hora (UTC)"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart-{station_id}-{days}")


def metric_cards(df: pd.DataFrame):
    if df.empty:
        st.write("Nenhum dado recente.")
        return
    last = df.dropna(subset=["datetime"]).sort_values("datetime").tail(1).iloc[0]
    rain_24h = df[df["datetime"] >= df["datetime"].max() - timedelta(hours=24)]["rain"].sum(min_count=1)
    cols = st.columns(3)
    cols[0].markdown(f"<div class='metric-card'><div>Última leitura</div><div style='font-size:1.4rem'>{last['datetime']:%d/%m %H:%M}</div></div>", unsafe_allow_html=True)
    rain_txt = f"{rain_24h:.1f} mm" if pd.notna(rain_24h) else "—"
    cols[1].markdown(f"<div class='metric-card'><div>Chuva 24h</div><div style='font-size:1.4rem'>{rain_txt}</div></div>", unsafe_allow_html=True)
    level_txt = f"{last['level']:.1f} cm" if pd.notna(last.get('level')) else "—"
    cols[2].markdown(f"<div class='metric-card'><div>Nível</div><div style='font-size:1.4rem'>{level_txt}</div></div>", unsafe_allow_html=True)


stations_df = load_stations()
stations_context_df = merge_station_context(stations_df, days=DAYS_WINDOW)
st.title("Sistema de Alerta de Cheias RS — Explorer")
render_network_summary(stations_context_df)
st.caption(
    "Resumo dos últimos 30 dias. Clique em um posto no mapa para ver a série temporal de chuva e nível."
)

with st.sidebar:
    st.subheader("Controles")
    station_view_mode = st.selectbox(
        "Visualização dos postos",
        options=list(STATION_VIEW_OPTIONS.keys()),
        index=0,
    )
    available_rasters = [r["name"] for r in list_rasters()]
    selected_layer = st.selectbox("Raster interpolado", options=["(nenhum)"] + available_rasters, index=1 if available_rasters else 0)
    selected_layer = None if selected_layer == "(nenhum)" else selected_layer
    opacity = st.slider("Transparência raster", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    st.markdown(
        "**Camadas:** postos sem dados aparecem em rosa fosco e podem ser ligados/desligados no controle do mapa."
    )

base_map = build_map(selected_layer, opacity, stations_context_df, station_view_mode)
map_state = st_folium(
    base_map,
    height=620,
    use_container_width=True,
    key="map",
    returned_objects=["last_object_clicked_tooltip"],
)

clicked_tooltip = (map_state or {}).get("last_object_clicked_tooltip")
default_station = None
if not stations_context_df.empty:
    preferred = stations_context_df[stations_context_df["status"] != "no_data"]
    default_station = (
        preferred["station_id"].iloc[0]
        if not preferred.empty
        else stations_context_df["station_id"].iloc[0]
    )
station_id = None
if clicked_tooltip:
    station_id = clicked_tooltip.split(" — ")[0]
station_id = station_id or st.session_state.get("station_id", default_station)
if station_id:
    st.session_state["station_id"] = station_id

series = load_timeseries(station_id, days=DAYS_WINDOW) if station_id else pd.DataFrame()
selected_station_row: Optional[pd.Series] = None
if station_id and not stations_context_df.empty:
    selected = stations_context_df[stations_context_df["station_id"] == station_id]
    if not selected.empty:
        selected_station_row = selected.iloc[0]

left, right = st.columns([0.45, 0.55])
with left:
    if station_id:
        render_selected_station_context(selected_station_row, station_id)
    else:
        st.markdown("### Estação selecionada: —")
    if station_id:
        metric_cards(series)
    else:
        st.info("Selecione uma estação no mapa.")

with right:
    if station_id:
        time_series_chart(series, station_id, DAYS_WINDOW)
