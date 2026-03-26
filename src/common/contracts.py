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
    """Contrato para avaliacao automatica de qualidade antes da execucao do modelo."""

    def run(self, run: RunMetadata) -> Iterable[QcFlag]:
        ...


class ManualReviewService(Protocol):
    """Contrato para registrar ajustes manuais sobre um run ja materializado."""

    def apply(self, run: RunMetadata, edits: Iterable[ManualEdit]) -> None:
        ...


class RunAssembler(Protocol):
    """Contrato para materializar o run operacional a partir dos insumos e outputs selecionados."""

    def build(self, run: RunMetadata) -> Iterable[ModelInput]:
        ...


class ModelExecutor(Protocol):
    """Contrato para preparar e executar o modelo externo a partir dos arquivos de input."""

    def prepare(self, run: RunMetadata) -> CommandPlan:
        ...

    def execute(self, plan: CommandPlan, *, dry_run: bool = False) -> ModelOutput:
        ...


class PostProcessor(Protocol):
    """Contrato para exportar o output completo e preparar o subset operacional do run."""

    def process(self, run: RunMetadata) -> Iterable[ModelOutput]:
        ...


class ReportBuilder(Protocol):
    """Contrato para geracao de artefatos de relatorio."""

    def build(self, run: RunMetadata) -> Iterable[ReportArtifact]:
        ...
