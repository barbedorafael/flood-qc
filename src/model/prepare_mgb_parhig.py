from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import logs_dir as default_logs_dir
from common.settings import load_settings
from ingest.fetch_observed_ana import resolve_reference_time

DEFAULT_PARHIG = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "PARHIG.hig"
DEFAULT_DT_SECONDS = 3600
LOGGER_NAME = "floodqc.model.prepare_mgb_parhig"


@dataclass(frozen=True, slots=True)
class ParhigUpdateSummary:
    parhig_path: Path
    reference_time: datetime
    start_time: datetime
    nt: int
    dt_seconds: int
    input_days_before: int


def script_stem() -> str:
    return Path(__file__).stem


def build_execution_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def configure_run_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _extract_numbers(text: str) -> list[str]:
    import re

    return re.findall(r"[-+]?\d+(?:[.,]\d+)?", text)


def _next_data_line_index(lines: list[str], start_idx: int) -> int:
    for idx in range(start_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("!"):
            continue
        return idx
    raise ValueError("Could not find a data line after the header line.")


def _find_header_index(lines: list[str], required_tokens: tuple[str, ...]) -> int:
    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()
        if all(token in upper for token in required_tokens):
            return idx
    raise ValueError(f"Could not find header containing tokens: {required_tokens}")


def _format_start_time_line(start_time: datetime) -> str:
    return f"        {start_time.day:02d}       {start_time.month:02d}       {start_time.year:04d}        {start_time.hour:02d}"


def _format_nt_dt_line(nt: int, dt_seconds: int) -> str:
    return f"{nt:10d}     {dt_seconds}."


def build_parhig_window(reference_time: datetime, *, input_days_before: int) -> tuple[datetime, int]:
    if input_days_before < 1:
        raise ValueError("input_days_before must be >= 1.")
    if reference_time.minute != 0 or reference_time.second != 0 or reference_time.microsecond != 0:
        raise ValueError("reference_time must be aligned to the hour for PARHIG hourly inputs.")

    start_date = reference_time.date() - timedelta(days=input_days_before)
    start_time = datetime.combine(start_date, time.min)
    nt = int((reference_time - start_time).total_seconds() // DEFAULT_DT_SECONDS) + 1
    if nt < 1:
        raise ValueError(f"Invalid NT calculated from reference_time={reference_time} and start_time={start_time}.")
    return start_time, nt


def update_parhig_text(text: str, *, start_time: datetime, nt: int, dt_seconds: int = DEFAULT_DT_SECONDS) -> str:
    lines = text.splitlines()
    start_header_idx = _find_header_index(lines, ("DIA", "MES", "ANO", "HORA"))
    nt_header_idx = _find_header_index(lines, ("NT", "DT"))

    start_data_idx = _next_data_line_index(lines, start_header_idx)
    nt_data_idx = _next_data_line_index(lines, nt_header_idx)

    lines[start_data_idx] = _format_start_time_line(start_time)
    lines[nt_data_idx] = _format_nt_dt_line(nt, dt_seconds)
    return "\n".join(lines) + "\n"


def read_time_settings_from_parhig(parhig_path: Path) -> tuple[datetime, int, int]:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    start_time: datetime | None = None
    nt: int | None = None
    dt_seconds: int | None = None

    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()
        if start_time is None and all(token in upper for token in ("DIA", "MES", "ANO", "HORA")):
            numbers = _extract_numbers(lines[_next_data_line_index(lines, idx)])
            if len(numbers) >= 4:
                day = int(float(numbers[0].replace(",", ".")))
                month = int(float(numbers[1].replace(",", ".")))
                year = int(float(numbers[2].replace(",", ".")))
                hour = int(float(numbers[3].replace(",", ".")))
                start_time = datetime(year, month, day, hour)
        if nt is None and dt_seconds is None and "NT" in upper and "DT" in upper:
            numbers = _extract_numbers(lines[_next_data_line_index(lines, idx)])
            if len(numbers) >= 2:
                nt = int(float(numbers[0].replace(",", ".")))
                dt_seconds = int(float(numbers[1].replace(",", ".")))
        if start_time is not None and nt is not None and dt_seconds is not None:
            break

    if start_time is None or nt is None or dt_seconds is None:
        raise ValueError(
            f"Could not read timing from {parhig_path}. Expected PARHIG to provide DIA/MES/ANO/HORA and NT/DT."
        )
    if nt <= 0 or dt_seconds <= 0:
        raise ValueError(f"Invalid PARHIG timing values: nt={nt}, dt_seconds={dt_seconds}")
    return start_time, nt, dt_seconds


def read_nc_from_parhig(parhig_path: Path) -> int:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    for idx, raw_line in enumerate(lines):
        if "NC" in raw_line.upper() and "NU" in raw_line.upper():
            numbers = _extract_numbers(lines[_next_data_line_index(lines, idx)])
            if numbers:
                nc = int(float(numbers[0].replace(",", ".")))
                if nc > 0:
                    return nc
    raise ValueError(f"Could not read NC from {parhig_path}")


def rewrite_parhig_from_config(
    *,
    parhig_path: Path = DEFAULT_PARHIG,
    logs_dir: Path = default_logs_dir(),
) -> ParhigUpdateSummary:
    settings = load_settings()
    reference_time = resolve_reference_time(settings["run"]["reference_time"])
    input_days_before = int(settings["mgb"]["input_days_before"])
    start_time, nt = build_parhig_window(reference_time, input_days_before=input_days_before)
    execution_id = build_execution_id()
    logger = configure_run_logger(logs_dir / script_stem() / f"{execution_id}.log")

    original_text = parhig_path.read_text(encoding="latin-1")
    updated_text = update_parhig_text(original_text, start_time=start_time, nt=nt, dt_seconds=DEFAULT_DT_SECONDS)
    parhig_path.write_text(updated_text, encoding="latin-1")

    logger.info(
        "parhig_updated path=%s reference_time=%s start_time=%s nt=%s dt_seconds=%s input_days_before=%s",
        parhig_path,
        reference_time.isoformat(timespec="seconds"),
        start_time.isoformat(timespec="seconds"),
        nt,
        DEFAULT_DT_SECONDS,
        input_days_before,
    )
    return ParhigUpdateSummary(
        parhig_path=parhig_path,
        reference_time=reference_time,
        start_time=start_time,
        nt=nt,
        dt_seconds=DEFAULT_DT_SECONDS,
        input_days_before=input_days_before,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reescreve o PARHIG.hig a partir da configuracao do run.")
    parser.add_argument("--parhig", type=Path, default=DEFAULT_PARHIG, help="Arquivo PARHIG.hig a reescrever.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = rewrite_parhig_from_config(parhig_path=args.parhig)
    print(
        "parhig_ready "
        f"path={summary.parhig_path} "
        f"reference_time={summary.reference_time.isoformat(timespec='seconds')} "
        f"start_time={summary.start_time.isoformat(timespec='seconds')} "
        f"nt={summary.nt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
