from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import SQL_DIR, build_run_db_path, history_db_path


def apply_schema(database_path: Path, schema_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")
    with sqlite3.connect(database_path) as connection:
        connection.executescript(schema_sql)
        connection.commit()


def initialize_history_db(database_path: Path | None = None) -> Path:
    target = database_path or history_db_path()
    apply_schema(target, SQL_DIR / "history_schema.sql")
    return target


def initialize_run_db(run_id: str, database_path: Path | None = None) -> Path:
    target = database_path or build_run_db_path(run_id)
    apply_schema(target, SQL_DIR / "run_schema.sql")
    with sqlite3.connect(target) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO run (run_id, reference_time, run_kind, status, parent_run_id, operator, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, run_id, "automatic", "draft", None, None, None),
        )
        connection.commit()
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inicializa bancos SQLite do repositorio.")
    parser.add_argument("--history", action="store_true", help="Inicializa `data/history.sqlite`.")
    parser.add_argument("--history-path", type=Path, default=None, help="Path alternativo para o banco historico.")
    parser.add_argument("--run-id", type=str, default=None, help="Identificador do run a ser criado.")
    parser.add_argument("--run-path", type=Path, default=None, help="Path alternativo para o banco do run.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.history:
        path = initialize_history_db(args.history_path)
        print(path)

    if args.run_id:
        path = initialize_run_db(args.run_id, args.run_path)
        print(path)

    if not args.history and not args.run_id:
        parser.error("Informe --history e/ou --run-id.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())