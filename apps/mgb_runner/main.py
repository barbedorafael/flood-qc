from __future__ import annotations

import argparse
from pathlib import Path

from runner import build_request, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runner inicial do MGB.")
    parser.add_argument("--run-db", required=True, type=Path, help="Arquivo SQLite do run.")
    parser.add_argument("--dry-run", action="store_true", help="Nao executa o binario real.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    response = run(build_request(args.run_db), dry_run=args.dry_run)
    print(response.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())