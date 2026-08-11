"""Observed Precipitation QC tab composed around session-local DashboardState."""
from __future__ import annotations

from typing import Any

import pandas as pd
import panel as pn

from apps.ops_dashboard.state import DashboardState
from apps.ops_dashboard.services import deckgl as dashboard_map


def _observed_qc_view(state: DashboardState) -> pn.viewable.Viewable:
    review_start = pn.widgets.DatetimeInput(name="Review window start", value=state.review_window_start)
    review_end = pn.widgets.DatetimeInput(name="Review window end", value=state.review_window_end)
    threshold = pn.widgets.FloatInput(name="Threshold (mm)", value=state.qc_threshold_mm, start=0)
    sequence_length = pn.widgets.IntInput(name="Minimum sequence length", value=state.qc_sequence_min_length, start=1)
    sequence_lower = pn.widgets.FloatInput(name="Sequence lower bound (mm)", value=state.qc_sequence_lower_mm, start=0)
    sequence_upper = pn.widgets.FloatInput(name="Sequence upper bound (mm)", value=state.qc_sequence_upper_mm, start=0)
    scan = pn.widgets.Button(name="Scan observations", button_type="primary", icon="search")
    status = pn.pane.Alert("", alert_type="info", visible=False)
    table = pn.widgets.Tabulator(
        state.detected_flags.copy(), name="Detected precipitation flags",
        pagination="local", page_size=20, selectable=1, header_filters=True,
        sizing_mode="stretch_width", height=420,
    )

    def show_message(kind: str = "warning", text: str | None = None) -> None:
        status.object = text if text is not None else state.message
        status.alert_type = kind if text is not None else state.message_kind
        status.visible = bool(status.object)

    def scan_flags(_: Any) -> None:
        state.review_window_start, state.review_window_end = review_start.value, review_end.value
        state.qc_threshold_mm = threshold.value
        state.qc_sequence_min_length = sequence_length.value
        state.qc_sequence_lower_mm = sequence_lower.value
        state.qc_sequence_upper_mm = sequence_upper.value
        try:
            table.value = state.scan_observed_precipitation().copy()
            show_message()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            show_message("warning", str(exc))

    scan.on_click(scan_flags)

    def selected_row() -> pd.Series:
        if not table.selection:
            raise ValueError("Select a flagged observation first.")
        return table.value.iloc[int(table.selection[0])]

    def select_flag(event: Any) -> None:
        if event.new:
            row = table.value.iloc[int(event.new[0])]
            state.qc_selected_station = str(row.station_id)
            state.qc_selected_sequence = None if pd.isna(row.sequence_id) else str(row.sequence_id)

    table.param.watch(select_flag, "selection")
    custom_value = pn.widgets.FloatInput(name="Custom replacement (mm)", value=0.0, start=0)
    null_button = pn.widgets.Button(name="Apply NULL", button_type="warning")
    value_button = pn.widgets.Button(name="Apply custom value", button_type="light")
    sequence_zero = pn.widgets.Button(name="Replace sequence with 0.0", button_type="light")
    sequence_custom = pn.widgets.Button(name="Replace sequence with custom value", button_type="light")
    remove_button = pn.widgets.Button(name="Remove instruction", button_type="light")
    exclude_button = pn.widgets.Button(name="Exclude station for review window", button_type="warning")
    include_button = pn.widgets.Button(name="Include station", button_type="light")

    def edit(action) -> None:
        try:
            row = selected_row()
            action(row)
            table.value = state.scan_observed_precipitation().copy()
            show_message()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            show_message("warning", str(exc))

    null_button.on_click(lambda _: edit(lambda row: state.set_observed_replacement(row.station_id, row.observed_at, None)))
    value_button.on_click(lambda _: edit(lambda row: state.set_observed_replacement(row.station_id, row.observed_at, custom_value.value)))
    sequence_zero.on_click(lambda _: edit(lambda row: state.replace_selected_sequence(None if pd.isna(row.sequence_id) else str(row.sequence_id), 0.0)))
    sequence_custom.on_click(lambda _: edit(lambda row: state.replace_selected_sequence(None if pd.isna(row.sequence_id) else str(row.sequence_id), custom_value.value)))
    remove_button.on_click(lambda _: edit(lambda row: state.remove_observed_replacement(row.station_id, row.observed_at)))
    exclude_button.on_click(lambda _: edit(lambda row: state.exclude_station(str(row.station_id))))
    include_button.on_click(lambda _: edit(lambda row: state.include_station(str(row.station_id))))

    def map_pane(artifacts):
        if artifacts is None:
            return pn.pane.Alert("Scan observations to build the QC station map.", alert_type="info")
        return pn.pane.DeckGL(artifacts.spec, height=480, sizing_mode="stretch_width")

    qc_map = pn.bind(map_pane, state.param.qc_map_artifacts)
    loo_station = pn.widgets.Select(name="Leave-one-out station", options={})
    run_loo = pn.widgets.Button(name="Run leave-one-out", button_type="primary")

    def station_options(_: Any = None) -> None:
        if state.stations.empty:
            loo_station.options = {}
            return
        loo_station.options = {f"{getattr(row, 'station_name', row.station_id)} ({row.station_id})": str(row.station_id) for row in state.stations.itertuples()}

    station_options()

    def diagnostic(result):
        if result is None:
            return pn.pane.Alert("Choose a station and run leave-one-out analysis.", alert_type="info")
        if not result.sufficient_source:
            return pn.pane.Alert(result.message, alert_type="warning")
        residual = "suppressed" if result.residual is None else f"{result.residual:.2f} mm"
        error = "suppressed" if result.absolute_error is None else f"{result.absolute_error:.2f} mm"
        layer, _, legend = dashboard_map.build_raster_layer(
            result.grid, layer_id="qc-leave-one-out-raster", layer_name="Accumulated leave-one-out precipitation", opacity=0.75,
        )
        layers = [layer] if layer is not None else []
        selected = state.stations[state.stations["station_id"].astype(str) == str(result.station_id)] if not state.stations.empty else pd.DataFrame()
        if not selected.empty:
            row = selected.iloc[0]
            layers.append({
                "@@type": "GeoJsonLayer", "id": "qc-omitted-station", "pickable": True,
                "data": {"type": "FeatureCollection", "features": [{
                    "type": "Feature", "geometry": {"type": "Point", "coordinates": [float(row.lon), float(row.lat)]},
                    "properties": {"station_id": result.station_id, "label": "Omitted target station"},
                }]},
                "pointType": "circle", "filled": True, "getFillColor": [220, 38, 38, 255],
                "getLineColor": [255, 255, 255, 255], "getLineWidth": 3,
                "getPointRadius": 9500,
            })
        raster = pn.pane.DeckGL({
            "initialViewState": dashboard_map.default_view_state(bounds=result.grid.bounds),
            "controller": True, "layers": layers,
        }, height=480, sizing_mode="stretch_width")
        metrics = pn.pane.Markdown(
            f"Observed total: **{result.observed_total if result.observed_total is not None else 'missing'} mm**  \n"
            f"Predicted total: **{result.predicted_total:.2f} mm**  \n"
            f"Residual (predicted − observed): **{residual}**  \n"
            f"Absolute error: **{error}**  \n"
            f"Coverage expected / target / predicted: **{result.expected_timesteps} / {result.target_timesteps} / {result.predicted_timesteps}**"
        )
        legend_pane = pn.pane.Markdown(dashboard_map.build_raster_legend_html(legend)) if legend is not None else pn.Spacer()
        return pn.Column(metrics, raster, legend_pane)

    diagnostic_view = pn.bind(diagnostic, state.param.qc_diagnostic_result)

    def run_diagnostic(_: Any) -> None:
        try:
            state.qc_selected_station = loo_station.value
            state.run_leave_one_out()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            show_message("warning", str(exc))

    run_loo.on_click(run_diagnostic)
    responsible = pn.widgets.TextInput(name="Responsible person", value=state.responsible_person)
    reason = pn.widgets.TextInput(name="Run reason", value=state.update_reason)
    run_id = pn.widgets.TextInput(name="Saved run ID", value=state.save_run_id)
    description = pn.widgets.TextAreaInput(name="Saved run description", value=state.save_run_description)
    update = pn.widgets.Button(name="Update Current Run", button_type="primary", icon="device-floppy")
    save = pn.widgets.Button(name="Save Run", button_type="success", icon="archive")
    conflicting = [scan, null_button, value_button, sequence_zero, sequence_custom, remove_button, exclude_button, include_button, run_loo, update, save]
    def disable_conflicting(event: Any) -> None:
        for widget in conflicting:
            widget.disabled = bool(event.new)
    state.param.watch(disable_conflicting, "execution_active")
    for widget in conflicting:
        widget.disabled = bool(state.execution_active)

    def update_current(_: Any) -> None:
        state.responsible_person, state.update_reason = responsible.value, reason.value
        try:
            state.update_current_run()
            show_message()
        except (RuntimeError, ValueError) as exc:
            show_message("warning", str(exc))

    def save_copy(_: Any) -> None:
        state.save_run_id, state.save_run_description = run_id.value, description.value
        try:
            state.save_run()
            show_message()
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            show_message("warning", str(exc))

    update.on_click(update_current)
    save.on_click(save_copy)
    policy = pn.Card(
        pn.Row(review_start, review_end), pn.Row(threshold, sequence_length),
        pn.Row(sequence_lower, sequence_upper), scan, title="Review window and QC policy",
    )
    edits = pn.Card(
        custom_value, pn.Row(null_button, value_button), pn.Row(sequence_zero, sequence_custom),
        pn.Row(remove_button, exclude_button, include_button), title="Session instruction draft",
    )
    analysis = pn.Card(pn.Row(loo_station, run_loo), diagnostic_view, title="Accumulated leave-one-out analysis")
    persistence = pn.Card(pn.Row(responsible, reason), pn.Row(run_id, description), pn.Row(update, save), title="Artifact update and optional saved copy")
    return pn.Column(policy, status, table, edits, pn.Card(qc_map, title="QC station map"), analysis, persistence, sizing_mode="stretch_width")


__all__ = ["_observed_qc_view"]
