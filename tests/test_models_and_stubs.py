from __future__ import annotations

import pytest

from common.models import DataState, RunKind, RunMetadata, RunStatus, TimeSeriesRecord
from ingest.forecast_grid import collect_forecast_grids
from ingest.observed import collect_observed_timeseries
from model.run_builder import assemble_model_inputs
from qc.automatic import apply_automatic_qc
from qc.review import register_manual_review
from reporting.reports import build_run_reports


@pytest.fixture
def run_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="20260310T120000",
        reference_time="2026-03-10T12:00:00",
        run_kind=RunKind.AUTOMATIC,
        status=RunStatus.DRAFT,
    )


def test_dataclass_instantiation() -> None:
    record = TimeSeriesRecord(
        series_id="ana.level.123",
        station_code="123",
        variable="level",
        unit="cm",
        state=DataState.RAW,
    )
    assert record.station_code == "123"


def test_stubs_raise_not_implemented(run_metadata: RunMetadata) -> None:
    with pytest.raises(NotImplementedError):
        collect_observed_timeseries(run_metadata)
    with pytest.raises(NotImplementedError):
        collect_forecast_grids(run_metadata)
    with pytest.raises(NotImplementedError):
        apply_automatic_qc(run_metadata)
    with pytest.raises(NotImplementedError):
        register_manual_review(run_metadata, [])
    with pytest.raises(NotImplementedError):
        assemble_model_inputs(run_metadata)
    with pytest.raises(NotImplementedError):
        build_run_reports(run_metadata)