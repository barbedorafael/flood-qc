from __future__ import annotations

"""
Dashboard Streamlit para explorar estações e grades interpoladas de chuva.
Funcionalidades:
- Mapa interativo (Folium) com clique em estações para mostrar séries.
- Gráfico Plotly com chuva (barras) e nível (linha) dos últimos 30 dias.
- Camada raster única (data/processed/interp) com liga/desliga e transparência ajustável.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
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
PROC_DIR = DATA_DIR / "processed"
TELEMETRIA_DIR = PROC_DIR / "telemetria"
INTERP_DIR = PROC_DIR / "interp"
DAYS_WINDOW = 30

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
        path = PROC_DIR / csv_name
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
    merged["name"] = merged["name"].fillna("").astype(str).str.strip()
    merged = merged.dropna(subset=["lat", "lon"])
    return merged


@st.cache_data(show_spinner=False)
def load_timeseries(station_id: str, days: int = 30) -> pd.DataFrame:
    path = TELEMETRIA_DIR / f"{station_id}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["station_id", "datetime", "rain", "level", "flow"])
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
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
def add_legend(fmap: folium.Map, vmin: float, vmax: float) -> None:
    colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b in BLUES]
    colormap = cm.LinearColormap(colors=colors_hex, vmin=float(vmin), vmax=float(vmax))
    colormap.caption = "Chuva acumulada (mm)"
    colormap.add_to(fmap)


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


def build_map(selected_layer: Optional[str], opacity: float, stations: pd.DataFrame) -> folium.Map:
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
            raster_group = folium.FeatureGroup(name=selected_layer, show=True)
            overlay.add_to(raster_group)
            raster_group.add_to(fmap)
            add_legend(fmap, vmin, vmax)
            RasterClickPopup(data, (west, south, east, north), selected_layer).add_to(fmap)

    station_layer = folium.FeatureGroup(name="Estações", show=True)
    for row in stations.itertuples():
        tooltip = f"{row.station_id} — {row.name}"
        popup_html = f"<b>{row.station_id}</b><br/>{row.name}"
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color="#0b7285" if row.kind == "nível" else "#364fc7",
            fill=True,
            fill_color="#0b7285" if row.kind == "nível" else "#364fc7",
            weight=1,
            fill_opacity=0.9,
            tooltip=tooltip,
            popup=popup_html,
        ).add_to(station_layer)
    station_layer.add_to(fmap)
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
        name="Nível (m)",
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
    rain_24h = df[df["datetime"] >= df["datetime"].max() - timedelta(hours=24)]["rain"].sum()
    cols = st.columns(3)
    cols[0].markdown(f"<div class='metric-card'><div>Última leitura</div><div style='font-size:1.4rem'>{last['datetime']:%d/%m %H:%M}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'><div>Chuva 24h</div><div style='font-size:1.4rem'>{rain_24h:.1f} mm</div></div>", unsafe_allow_html=True)
    level_txt = f"{last['level']:.2f} m" if pd.notna(last.get('level')) else "—"
    cols[2].markdown(f"<div class='metric-card'><div>Nível</div><div style='font-size:1.4rem'>{level_txt}</div></div>", unsafe_allow_html=True)


stations_df = load_stations()
st.title("Sistema de Alerta de Cheias RS — Explorer")
st.caption("Clique em uma estação para ver chuva (barras) e nível (linha) dos últimos 30 dias. Camadas raster de chuva interpolada podem ser ligadas/desligadas e ter transparência ajustada.")

with st.sidebar:
    st.subheader("Controles")
    available_rasters = [r["name"] for r in list_rasters()]
    selected_layer = st.selectbox("Raster interpolado", options=["(nenhum)"] + available_rasters, index=1 if available_rasters else 0)
    selected_layer = None if selected_layer == "(nenhum)" else selected_layer
    opacity = st.slider("Transparência raster", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    st.markdown("**Legenda:** círculos verde-azulados = estações nível; azul escuro = estações chuva.")

base_map = build_map(selected_layer, opacity, stations_df)
map_state = st_folium(
    base_map,
    height=620,
    use_container_width=True,
    key="map",
    returned_objects=["last_object_clicked_tooltip"],
)

clicked_tooltip = (map_state or {}).get("last_object_clicked_tooltip")
default_station = stations_df["station_id"].iloc[0] if not stations_df.empty else None
station_id = None
if clicked_tooltip:
    station_id = clicked_tooltip.split(" — ")[0]
station_id = station_id or st.session_state.get("station_id", default_station)
if station_id:
    st.session_state["station_id"] = station_id

left, right = st.columns([0.45, 0.55])
with left:
    st.markdown(f"### Estação selecionada: {station_id or '—'}")
    if station_id:
        series = load_timeseries(station_id, days=DAYS_WINDOW)
        metric_cards(series)
    else:
        st.info("Selecione uma estação no mapa.")

with right:
    if station_id:
        series = load_timeseries(station_id, days=DAYS_WINDOW)
        time_series_chart(series, station_id, DAYS_WINDOW)
