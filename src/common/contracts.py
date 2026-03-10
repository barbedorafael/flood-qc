from __future__ import annotations

from typing import Iterable, Protocol

from common.models import (
    CommandPlan,
    ManualEdit,
    ModelInput,
    ModelOutput,
    QcFlag,
    RasterAsset,
    ReportArtifact,
    RunMetadata,
    TimeSeriesRecord,
)


class ObservationCollector(Protocol):
    """Contrato para coletores de series observadas."""

    def collect(self, run: RunMetadata) -> Iterable[TimeSeriesRecord]:
        ...


class ForecastGridCollector(Protocol):
    """Contrato para coletores de previsao em grade."""

    def collect(self, run: RunMetadata) -> Iterable[RasterAsset]:
        ...


class AutomaticQcProcessor(Protocol):
    """Contrato para avaliacao automatica de qualidade."""

    def run(self, run: RunMetadata) -> Iterable[QcFlag]:
        ...


class ManualReviewService(Protocol):
    """Contrato para registrar ajustes manuais sem sobrescrever a origem."""

    def apply(self, run: RunMetadata, edits: Iterable[ManualEdit]) -> None:
        ...


class RunAssembler(Protocol):
    """Contrato para montagem dos inputs do run do modelo."""

    def build(self, run: RunMetadata) -> Iterable[ModelInput]:
        ...


class ModelExecutor(Protocol):
    """Contrato para preparar e executar o modelo externo."""

    def prepare(self, run: RunMetadata) -> CommandPlan:
        ...

    def execute(self, plan: CommandPlan, *, dry_run: bool = False) -> ModelOutput:
        ...


class PostProcessor(Protocol):
    """Contrato para consolidacao dos outputs apos a execucao."""

    def process(self, run: RunMetadata) -> Iterable[ModelOutput]:
        ...


class ReportBuilder(Protocol):
    """Contrato para geracao de artefatos de relatorio."""

    def build(self, run: RunMetadata) -> Iterable[ReportArtifact]:
        ...