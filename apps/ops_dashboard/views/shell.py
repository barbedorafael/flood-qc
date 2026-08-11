"""Responsive dashboard shell and top-level view placement."""
from __future__ import annotations

import json
import panel as pn

from apps.ops_dashboard.state import DashboardState
from apps.ops_dashboard.views.forecast import _forecast_view
from apps.ops_dashboard.views.monitoring import _monitoring_view
from apps.ops_dashboard.views.observed_qc import _observed_qc_view
from apps.ops_dashboard.views.summaries import _network_summary


def _build_template(state: DashboardState) -> pn.template.base.BasicTemplate:
    chart_start = pn.widgets.DatetimeInput.from_param(
        state.param.start_time, name="Chart start time", sizing_mode="stretch_width"
    )
    refresh = pn.widgets.Button(
        name="Refresh data", button_type="primary", icon="refresh", sizing_mode="stretch_width",
    )
    refreshed = pn.bind(
        lambda value: pn.pane.Markdown(f"Last session refresh:  \n{value}" if value else "Not refreshed yet."),
        state.param.last_refresh_at,
    )
    warnings = pn.bind(
        lambda values: pn.Column(*[pn.pane.Alert(value, alert_type="warning") for value in values], sizing_mode="stretch_width"),
        state.param.warnings,
    )
    execution = pn.bind(
        lambda active, progress, status, error: pn.Column(
            pn.indicators.Progress(name="MGB execution", value=progress, max=100, visible=active or status in {"failed", "completed"}),
            pn.pane.Alert(error or status, alert_type="danger" if error else "info", visible=bool(error) or active),
        ),
        state.param.execution_active, state.param.execution_progress,
        state.param.execution_status, state.param.execution_error,
    )
    tabs = pn.Tabs(
        ("Monitoring", _monitoring_view(state)),
        ("Forecast", _forecast_view(state)),
        ("Observed Precipitation QC", _observed_qc_view(state)),
        dynamic=True, sizing_mode="stretch_width",
    )
    template = pn.template.FastListTemplate(
        title="Operational Hydrology",
        sidebar=[pn.pane.Markdown("## Controls"), chart_start, refresh, refreshed, pn.layout.Divider(), warnings],
        main=[
            pn.pane.Markdown("# Operational MGB System\nObserved and forecasted hydrological data for the operation of MGB results."),
            pn.bind(lambda stations: _network_summary(stations, state.window.cutoff_time), state.param.stations),
            tabs,
        ],
        sidebar_width=320, accent_base_color="#1864ab", header_background="#1864ab",
    )
    summary = pn.pane.JSON({}, name="Validated current artifact", depth=2, sizing_mode="stretch_width")
    modal_status = pn.pane.Alert("", alert_type="warning", visible=False)
    run = pn.widgets.Button(name="Run", button_type="primary", icon="player-play")
    cancel = pn.widgets.Button(name="Cancel", button_type="light")
    template.modal.append(pn.Column(
        pn.pane.Markdown("## Confirm artifact-backed Refresh\nReview the validated current artifact before starting the asynchronous MGB pipeline."),
        summary, execution, modal_status, pn.Row(cancel, run), sizing_mode="stretch_width",
    ))

    def open_confirmation(_: object) -> None:
        try:
            summary.object = state.artifact_summary()
            modal_status.visible = False
            template.open_modal()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            modal_status.object = str(exc)
            modal_status.visible = True
            template.open_modal()

    def start_execution(_: object) -> None:
        try:
            state.run_current_artifact_async()
            template.close_modal()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            modal_status.object = str(exc)
            modal_status.visible = True

    refresh.on_click(open_confirmation)
    run.on_click(start_execution)
    cancel.on_click(lambda _: template.close_modal())
    state.param.watch(lambda event: setattr(refresh, "disabled", bool(event.new)), "execution_active")
    state.param.watch(lambda event: setattr(run, "disabled", bool(event.new)), "execution_active")
    template.state = state
    template.refresh_confirmation = {"summary": summary, "run": run, "cancel": cancel, "status": modal_status}
    return template
