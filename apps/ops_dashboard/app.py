from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Dependencias de UI nao encontradas. Instale com: "
        "pip install streamlit plotly folium streamlit-folium branca rasterio"
    ) from exc

from common.paths import history_db_path
from reporting import ops_dashboard_data, ops_dashboard_map


DAYS_WINDOW = 30
MGB_COLORS = {"q": "#1864ab", "y": "#2b8a3e"}


st.set_page_config(page_title="Hidrologia Operacional", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Helvetica Neue', sans-serif;
        background: radial-gradient(circle at 10% 20%, #f2f7fb, #e8f1f7 40%, #e5ecf3 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_station_catalog(days: int) -> pd.DataFrame:
    return ops_dashboard_data.load_station_catalog(days=days)


@st.cache_data(show_spinner=False)
def get_observed_series(station_uid: int, days: int) -> pd.DataFrame:
    return ops_dashboard_data.load_observed_series(station_uid=station_uid, days=days)


@st.cache_data(show_spinner=False)
def get_model_metadata() -> dict[str, object]:
    return ops_dashboard_data.load_model_metadata()


@st.cache_data(show_spinner=False)
def get_model_variables() -> pd.DataFrame:
    return ops_dashboard_data.list_model_variables()


@st.cache_data(show_spinner=False)
def get_mgb_series(mini_id: int, variable_code: str, days_window: int) -> pd.DataFrame:
    return ops_dashboard_data.load_mgb_series(mini_id=mini_id, variable_code=variable_code, days_window=days_window)


@st.cache_data(show_spinner=False)
def get_accumulation_rasters() -> list[dict[str, object]]:
    return ops_dashboard_data.list_accumulation_rasters()


@st.cache_data(show_spinner=False)
def get_rivers_geojson() -> dict | None:
    return ops_dashboard_data.load_rivers_layer_geojson()


@st.cache_resource
def get_map_artifacts(
    map_cache_key: str,
    selected_layer_name: Optional[str],
    opacity: float,
) -> ops_dashboard_map.MapRenderArtifacts:
    del map_cache_key
    stations_df = ops_dashboard_data.load_station_catalog(days=DAYS_WINDOW)
    rivers_geojson = ops_dashboard_data.load_rivers_layer_geojson()
    raster_catalog = {str(item["name"]): item for item in ops_dashboard_data.list_accumulation_rasters()}
    base_map = ops_dashboard_map.build_ops_map(selected_layer_name, opacity, stations_df, rivers_geojson, raster_catalog)
    return ops_dashboard_map.build_map_render_artifacts(base_map)


def format_mm(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "indisponivel"
    return f"{float(value):.1f} mm"


def format_value(value: float | int | None, unit: str) -> str:
    if value is None or pd.isna(value):
        return "indisponivel"
    return f"{float(value):.2f} {unit}"


def network_summary(stations: pd.DataFrame) -> dict[str, float]:
    if stations.empty:
        return {
            "total": 0.0,
            "with_data": 0.0,
            "no_data": 0.0,
            "data_issue": 0.0,
            "rain_mean_24h": np.nan,
            "rain_p90_24h": np.nan,
        }

    rain_values = stations.loc[stations["status"] == "ok", "rain_acc_24h_mm"].dropna()
    if rain_values.empty:
        rain_values = stations["rain_acc_24h_mm"].dropna()

    return {
        "total": float(len(stations)),
        "with_data": float((stations["status"] == "ok").sum()),
        "no_data": float((stations["status"] == "no_data").sum()),
        "data_issue": float((stations["status"] == "data_issue").sum()),
        "rain_mean_24h": float(rain_values.mean()) if not rain_values.empty else np.nan,
        "rain_p90_24h": float(rain_values.quantile(0.9)) if not rain_values.empty else np.nan,
    }


def render_network_summary(stations: pd.DataFrame) -> None:
    summary = network_summary(stations)
    cols = st.columns(6)
    cols[0].metric("Postos totais", f"{int(summary['total'])}")
    cols[1].metric("Com dados", f"{int(summary['with_data'])}")
    cols[2].metric("Sem dados", f"{int(summary['no_data'])}")
    cols[3].metric("Falha de dados", f"{int(summary['data_issue'])}")
    cols[4].metric("Media chuva 24h", format_mm(summary["rain_mean_24h"]))
    cols[5].metric("P90 chuva 24h", format_mm(summary["rain_p90_24h"]))


def render_selected_station_context(row: Optional[pd.Series]) -> None:
    if row is None:
        st.markdown("### Posto selecionado: -")
        st.caption("Clique em um posto no mapa para carregar os dados observados.")
        return

    kind_labels = {"chuva": "chuva", "nivel": "nivel", "misto": "misto", "sem_dados": "sem cobertura"}
    status_labels = {"ok": "ok", "data_issue": "falha de dados", "no_data": "sem dados"}

    station_name = str(row["station_name"]).strip() or "sem nome"
    st.markdown(f"### Posto selecionado: {station_name}")
    st.caption(
        "UID: {uid} | Codigo: {provider}:{code} | Tipo: {kind} | Status: {status}".format(
            uid=int(row["station_uid"]),
            provider=str(row["provider_code"]).upper(),
            code=str(row["station_code"]),
            kind=kind_labels.get(str(row["kind"]), "sem cobertura"),
            status=status_labels.get(str(row["status"]), "sem dados"),
        )
    )
    reason = str(row.get("status_reason", "")).strip()
    if reason:
        st.caption(f"Obs: {reason}")


def metric_cards(observed_df: pd.DataFrame) -> None:
    if observed_df.empty:
        st.info("Sem dados observados para esta estacao na janela selecionada.")
        return

    metrics = ops_dashboard_data.compute_observed_metrics(observed_df)
    latest_time = metrics["latest_time"]
    if latest_time is not None and not pd.isna(latest_time):
        st.markdown(f"**Ultima leitura:** {latest_time:%d/%m %H:%M}")

    st.markdown("**Chuvas acumuladas**")
    rain_table = pd.DataFrame(
        {"Valor": [format_mm(metrics["rain_12h"]), format_mm(metrics["rain_24h"]), format_mm(metrics["rain_72h"])]},
        index=["12h", "24h", "72h"],
    )
    st.table(rain_table)

    st.markdown("**Estado hidrologico**")
    hydro_table = pd.DataFrame(
        {"Valor": [format_value(metrics["level_current"], "cm"), format_value(metrics["flow_current"], "m3/s")]},
        index=["nivel atual", "vazao atual"],
    )
    st.table(hydro_table)


def time_series_chart(df: pd.DataFrame, station_uid: int, days: int) -> None:
    if df.empty:
        st.info("Sem dados para esta estacao/intervalo.")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.35, 0.65],
    )

    rain_df = df[df["variable_code"] == "rain"].dropna(subset=["value"])
    level_df = df[df["variable_code"] == "level"].dropna(subset=["value"])
    flow_df = df[df["variable_code"] == "flow"].dropna(subset=["value"])

    if not rain_df.empty:
        fig.add_bar(
            x=rain_df["datetime"],
            y=rain_df["value"],
            name="Chuva (mm)",
            marker_color="#4dabf7",
            opacity=0.9,
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(text="Chuva indisponivel", xref="paper", yref="paper", x=0.5, y=0.88, showarrow=False)

    if not level_df.empty:
        fig.add_scatter(
            x=level_df["datetime"],
            y=level_df["value"],
            name="Nivel (cm)",
            mode="lines+markers",
            line=dict(color="#0b7285", width=2),
            marker=dict(size=4),
            row=2,
            col=1,
        )
    if not flow_df.empty:
        fig.add_scatter(
            x=flow_df["datetime"],
            y=flow_df["value"],
            name="Vazao (m3/s)",
            mode="lines",
            line=dict(color="#e67700", width=2),
            row=2,
            col=1,
        )
    if level_df.empty and flow_df.empty:
        fig.add_annotation(text="Nivel e vazao indisponiveis", xref="paper", yref="paper", x=0.5, y=0.22, showarrow=False)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(t=30, r=20, l=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        xaxis=dict(title=""),
        xaxis2=dict(title="Data/hora"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart-{station_uid}-{days}")


def render_selected_mini_context(
    mini_id: Optional[int],
    variable_display: str,
    metadata: dict[str, object],
    days_window: int,
) -> None:
    if mini_id is None:
        st.markdown("### Mini selecionada: -")
        st.caption("Clique em uma geometria de rio no mapa para carregar a serie do modelo.")
        return

    st.markdown(f"### Mini selecionada: {mini_id}")
    reference_time = metadata.get("reference_time")
    reference_label = reference_time.strftime("%d/%m/%Y %H:%M") if isinstance(reference_time, pd.Timestamp) else "-"
    st.caption(f"Variavel: {variable_display} | Janela: ultimos {days_window} dias + previsao")
    st.caption(f"Fonte: data/interim/model_outputs.sqlite | Reference time: {reference_label}")


def mgb_metric_cards(df: pd.DataFrame, variable_display: str, unit: str) -> None:
    if df.empty:
        st.info("Sem dados do modelo para a mini selecionada.")
        return

    current_df = df[df["prev_flag"] == 0].copy()
    forecast_df = df[df["prev_flag"] == 1].copy()

    col1, col2 = st.columns(2)
    col1.metric("Pontos atuais", f"{len(current_df)}")
    col2.metric("Pontos previsao", f"{len(forecast_df)}")

    if not current_df.empty:
        last_current = current_df.sort_values("dt").iloc[-1]
        st.caption(f"Ultimo atual: {last_current['dt']:%d/%m/%Y %H:%M} | {variable_display}={last_current['value']:.3f} {unit}")
    if not forecast_df.empty:
        first_forecast = forecast_df.sort_values("dt").iloc[0]
        st.caption(f"Inicio previsao: {first_forecast['dt']:%d/%m/%Y %H:%M} | {variable_display}={first_forecast['value']:.3f} {unit}")


def mgb_time_series_chart(df: pd.DataFrame, mini_id: int, variable_code: str, variable_display: str, unit: str, days_window: int) -> None:
    if df.empty:
        st.info("Sem serie do modelo para a mini selecionada.")
        return

    current_df = df[df["prev_flag"] == 0]
    forecast_df = df[df["prev_flag"] == 1]

    fig = go.Figure()
    base_color = MGB_COLORS.get(variable_code, "#1864ab")
    forecast_color = "#e67700"

    if not current_df.empty:
        fig.add_trace(
            go.Scatter(
                x=current_df["dt"],
                y=current_df["value"],
                mode="lines",
                name=f"{variable_display} atual",
                line=dict(color=base_color, width=2),
            )
        )
    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["dt"],
                y=forecast_df["value"],
                mode="lines",
                name=f"{variable_display} previsao",
                line=dict(color=forecast_color, width=2, dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(t=30, r=20, l=10, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Data/hora"),
        yaxis=dict(title=f"{variable_display} ({unit})"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"mgb-chart-{variable_code}-{mini_id}-{days_window}")


def compute_map_cache_key(selected_layer_name: Optional[str], opacity: float) -> str:
    accumulation_rasters = get_accumulation_rasters()
    raster_catalog = {str(item["name"]): item for item in accumulation_rasters}
    raster_version = "no-raster"
    if selected_layer_name:
        meta = raster_catalog.get(selected_layer_name)
        if meta:
            raster_version = ops_dashboard_map.build_file_version(Path(str(meta["path"])))

    return ops_dashboard_map.build_map_cache_key(
        selected_layer_name=selected_layer_name,
        opacity=opacity,
        history_version=ops_dashboard_map.build_file_version(history_db_path()),
        rivers_version=ops_dashboard_map.build_file_version(ops_dashboard_data.LEGACY_RIVERS_GEOJSON_PATH),
        raster_version=raster_version,
        station_uid=st.session_state.get("station_uid"),
        mini_id=st.session_state.get("mini_id"),
    )


@st.fragment
def render_map_fragment(map_artifacts: ops_dashboard_map.MapRenderArtifacts) -> None:
    map_state = ops_dashboard_map.render_map_component(
        map_artifacts,
        height=620,
        use_container_width=True,
    )
    click_token = ops_dashboard_map.parse_click_token((map_state or {}).get("last_object_clicked_tooltip"))
    if ops_dashboard_map.update_selection_from_click_token(click_token, st.session_state):
        st.rerun()


def ensure_default_selection(stations_df: pd.DataFrame) -> None:
    if "station_uid" not in st.session_state and not stations_df.empty:
        preferred = stations_df[stations_df["status"] != "no_data"]
        default_station_uid = int(preferred["station_uid"].iloc[0]) if not preferred.empty else int(stations_df["station_uid"].iloc[0])
        st.session_state["station_uid"] = default_station_uid


def main() -> None:
    stations_df = get_station_catalog(DAYS_WINDOW)
    rivers_geojson = get_rivers_geojson()
    model_metadata = get_model_metadata()
    model_variables = get_model_variables()
    accumulation_rasters = get_accumulation_rasters()
    raster_catalog = {str(item["name"]): item for item in accumulation_rasters}

    ensure_default_selection(stations_df)

    st.title("Sistema de Alerta de Cheias RS - Explorer")
    render_network_summary(stations_df)
    st.caption("Clique em um posto para dados observados ou em uma geometria de rio para series do modelo MGB.")

    with st.sidebar:
        st.subheader("Controles")
        selected_layer_name = st.selectbox(
            "Raster de chuva acumulada",
            options=["(nenhum)"] + [str(item["name"]) for item in accumulation_rasters],
            format_func=lambda option: "(nenhum)" if option == "(nenhum)" else str(raster_catalog[option]["horizon_label"]),
            index=1 if accumulation_rasters else 0,
        )
        selected_layer_name = None if selected_layer_name == "(nenhum)" else selected_layer_name
        opacity = st.slider("Transparencia raster", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        st.markdown("**Camadas:** postos, rios MGB e raster de chuva acumulada sobrepostos no mapa.")

    map_cache_key = compute_map_cache_key(selected_layer_name, opacity)
    map_warning: Optional[str] = None
    try:
        map_artifacts = get_map_artifacts(map_cache_key, selected_layer_name, opacity)
    except RuntimeError as exc:
        map_warning = str(exc)
        fallback_key = compute_map_cache_key(None, opacity)
        map_artifacts = get_map_artifacts(fallback_key, None, opacity)

    if map_warning:
        st.warning(map_warning)

    render_map_fragment(map_artifacts)

    station_uid = st.session_state.get("station_uid")
    station_uid = int(station_uid) if station_uid is not None else None
    mini_id = st.session_state.get("mini_id")
    mini_id = int(mini_id) if mini_id is not None else None

    observed_series = get_observed_series(station_uid, DAYS_WINDOW) if station_uid is not None else pd.DataFrame()
    selected_station_row: Optional[pd.Series] = None
    if station_uid is not None and not stations_df.empty:
        selected = stations_df[stations_df["station_uid"] == station_uid]
        if not selected.empty:
            selected_station_row = selected.iloc[0]

    st.subheader("Dados observados")
    left, right = st.columns([0.45, 0.55])
    with left:
        render_selected_station_context(selected_station_row)
        if station_uid is not None:
            metric_cards(observed_series)
        else:
            st.info("Selecione um posto no mapa.")

    with right:
        if station_uid is not None:
            time_series_chart(observed_series, station_uid, DAYS_WINDOW)
        else:
            st.info("Sem posto selecionado.")

    st.markdown("---")
    st.subheader("Outputs MGB")

    if rivers_geojson is None:
        st.warning("Camada de rios nao encontrada em data/legacy/app_layers/rios_mini.geojson.")

    if model_variables.empty:
        st.warning("Banco data/interim/model_outputs.sqlite sem variaveis cadastradas.")
        available_variable_codes: list[str] = []
    else:
        available_variable_codes = model_variables["variable_code"].tolist()

    selected_variable_code = (
        st.selectbox(
            "Variavel MGB",
            options=available_variable_codes,
            format_func=lambda code: f"{model_variables.loc[model_variables['variable_code'] == code, 'display_name'].iloc[0]} ({code})",
            key="mgb_variable",
        )
        if available_variable_codes
        else None
    )

    model_series = pd.DataFrame(columns=["dt", "prev_flag", "value", "variable_code", "display_name", "unit"])
    if mini_id is not None and selected_variable_code is not None:
        model_series = get_mgb_series(mini_id=mini_id, variable_code=selected_variable_code, days_window=DAYS_WINDOW)

    variable_display = selected_variable_code or "-"
    variable_unit = "-"
    if selected_variable_code is not None and not model_variables.empty:
        row = model_variables[model_variables["variable_code"] == selected_variable_code].iloc[0]
        variable_display = str(row["display_name"])
        variable_unit = str(row["unit"])

    mgb_left, mgb_right = st.columns([0.35, 0.65])
    with mgb_left:
        render_selected_mini_context(mini_id, variable_display, model_metadata, DAYS_WINDOW)
        if mini_id is None:
            st.info("Selecione uma mini no mapa.")
        else:
            mgb_metric_cards(model_series, variable_display, variable_unit)

    with mgb_right:
        if mini_id is None or selected_variable_code is None:
            st.info("Sem mini/variavel selecionada.")
        else:
            mgb_time_series_chart(
                model_series,
                mini_id=mini_id,
                variable_code=selected_variable_code,
                variable_display=variable_display,
                unit=variable_unit,
                days_window=DAYS_WINDOW,
            )


if __name__ == "__main__":
    main()
