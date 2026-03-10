from __future__ import annotations

from pathlib import Path


class HistoryRepository:
    """Acesso futuro ao banco historico.

    TODO:
    - implementar leitura e escrita das estacoes;
    - persistir series observadas e flags de QC;
    - indexar runs publicados para consulta posterior.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> None:
        raise NotImplementedError("Repositorio historico ainda nao implementado.")