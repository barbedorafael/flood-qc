from __future__ import annotations

from pathlib import Path


class RunRepository:
    """Acesso futuro ao arquivo SQLite de um run.

    TODO:
    - registrar lineage do run;
    - persistir inputs, outputs, flags e edicoes;
    - registrar assets e relatorios associados.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> None:
        raise NotImplementedError("Repositorio de run ainda nao implementado.")