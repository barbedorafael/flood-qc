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
from common.time_utils import resolve_reference_time

DEFAULT_PARHIG = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "PARHIG.hig"
DEFAULT_PREVISAO_META = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "Previsao.meta"
DEFAULT_MINI_GTP = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "MINI.gtp"
DEFAULT_DT_SECONDS = 3600
LOGGER_NAME = "floodqc.model.prepare_mgb_meta"


@dataclass(frozen=True, slots=True)
class MgbWindow:
    reference_time: datetime
    start_time: datetime
    forecast_start_time: datetime
    forecast_nt: int
    nt: int
    dt_seconds: int
    input_days_before: int
    forecast_horizon_days: int


@dataclass(frozen=True, slots=True)
class MgbMetaUpdateSummary:
    parhig_path: Path
    previsao_meta_path: Path
    mini_gtp_path: Path
    reference_time: datetime
    start_time: datetime
    forecast_start_time: datetime
    forecast_nt: int
    nt: int
    nc: int
    dt_seconds: int
    input_days_before: int
    forecast_horizon_days: int


@dataclass(frozen=True, slots=True)
class MiniCentroid:
    mini_id: int
    lon: float
    lat: float


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


def build_mgb_window(
    reference_time: datetime,
    *,
    input_days_before: int,
    forecast_horizon_days: int,
) -> MgbWindow:
    if input_days_before < 1:
        raise ValueError("input_days_before must be >= 1.")
    if forecast_horizon_days < 1:
        raise ValueError("forecast_horizon_days must be >= 1.")
    if reference_time.minute != 0 or reference_time.second != 0 or reference_time.microsecond != 0:
        raise ValueError("reference_time must be aligned to the hour for MGB hourly inputs.")

    start_date = reference_time.date() - timedelta(days=input_days_before)
    start_time = datetime.combine(start_date, time.min)
    forecast_start_time = reference_time + timedelta(hours=1)
    forecast_nt = forecast_horizon_days * 24 + 1
    forecast_end_time = forecast_start_time + timedelta(hours=forecast_nt - 1)
    nt = int((forecast_end_time - start_time).total_seconds() // DEFAULT_DT_SECONDS) + 1
    if nt < 1:
        raise ValueError(f"Invalid NT calculated from reference_time={reference_time} and start_time={start_time}.")
    return MgbWindow(
        reference_time=reference_time,
        start_time=start_time,
        forecast_start_time=forecast_start_time,
        forecast_nt=forecast_nt,
        nt=nt,
        dt_seconds=DEFAULT_DT_SECONDS,
        input_days_before=input_days_before,
        forecast_horizon_days=forecast_horizon_days,
    )


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


def read_mini_centroids(mini_gtp_path: Path, *, nc: int) -> list[MiniCentroid]:
    header: list[str] | None = None
    rows: list[MiniCentroid] = []

    with mini_gtp_path.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if header is None:
                header = parts
                required = {"Mini", "Xcen", "Ycen"}
                if not required.issubset(header):
                    raise ValueError(f"MINI.gtp missing required columns {sorted(required)}: {mini_gtp_path}")
                continue

            mini_idx = header.index("Mini")
            xcen_idx = header.index("Xcen")
            ycen_idx = header.index("Ycen")
            if len(parts) <= max(mini_idx, xcen_idx, ycen_idx):
                raise ValueError(f"Invalid MINI.gtp row: {raw_line.rstrip()}")
            rows.append(
                MiniCentroid(
                    mini_id=int(float(parts[mini_idx].replace(",", "."))),
                    lon=float(parts[xcen_idx].replace(",", ".")),
                    lat=float(parts[ycen_idx].replace(",", ".")),
                )
            )
            if len(rows) == nc:
                break

    if len(rows) < nc:
        raise ValueError(f"MINI.gtp has {len(rows)} rows, smaller than NC={nc}")

    seen: set[int] = set()
    duplicates: list[int] = []
    for row in rows:
        if row.mini_id in seen:
            duplicates.append(row.mini_id)
        seen.add(row.mini_id)
    if duplicates:
        raise ValueError(f"MINI.gtp has duplicated Mini ids (sample: {duplicates[:5]})")
    return rows


def _read_tagged_value(lines: list[str], label: str) -> str:
    for idx, raw_line in enumerate(lines):
        if raw_line.strip().upper() == label.upper():
            return lines[_next_data_line_index(lines, idx)].strip()
    raise ValueError(f"Could not find section {label!r} in Previsao.meta template.")


def build_previsao_meta_text(
    template_text: str,
    *,
    forecast_start_time: datetime,
    forecast_nt: int,
    mini_centroids: list[MiniCentroid],
) -> str:
    lines = template_text.splitlines()
    project_name = _read_tagged_value(lines, "!Projeto")
    binary_name = _read_tagged_value(lines, "!Nome do Arquivo Binario")

    text_lines = [
        "!Metadados de Arquivo de Previsao de Chuva",
        "!Projeto",
        project_name,
        "",
        "!Nome do Arquivo Binario",
        binary_name,
        "",
        "!Data de Inicio",
        "!ANO   MES   DIA  HORA   HORIZ(NT)",
        f"{forecast_start_time.year:4d}    {forecast_start_time.month:02d}    {forecast_start_time.day:02d}    {forecast_start_time.hour:02d}     {forecast_nt}",
        "",
        "!Numero de Mini-Bacias",
        f" {len(mini_centroids):4d} ",
        "",
        "!Lista de coordenadas dos centroides",
        "!  IC\t LAT_dg    LONG_dg",
    ]
    for cell_index, centroid in enumerate(mini_centroids, start=1):
        text_lines.append(f"{cell_index:5d}  {centroid.lat:10.5f}  {centroid.lon:10.5f}")
    return "\n".join(text_lines) + "\n"


def rewrite_mgb_meta_from_config(
    *,
    parhig_path: Path = DEFAULT_PARHIG,
    previsao_meta_path: Path = DEFAULT_PREVISAO_META,
    mini_gtp_path: Path = DEFAULT_MINI_GTP,
    logs_dir: Path = default_logs_dir(),
) -> MgbMetaUpdateSummary:
    settings = load_settings()
    reference_time = resolve_reference_time(settings["run"]["reference_time"])
    input_days_before = int(settings["mgb"]["input_days_before"])
    forecast_horizon_days = int(settings["mgb"]["forecast_horizon_days"])
    window = build_mgb_window(
        reference_time,
        input_days_before=input_days_before,
        forecast_horizon_days=forecast_horizon_days,
    )
    execution_id = build_execution_id()
    logger = configure_run_logger(logs_dir / script_stem() / f"{execution_id}.log")

    original_parhig_text = parhig_path.read_text(encoding="latin-1")
    updated_parhig_text = update_parhig_text(
        original_parhig_text,
        start_time=window.start_time,
        nt=window.nt,
        dt_seconds=window.dt_seconds,
    )
    parhig_path.write_text(updated_parhig_text, encoding="latin-1")

    nc = read_nc_from_parhig(parhig_path)
    mini_centroids = read_mini_centroids(mini_gtp_path, nc=nc)
    previsao_template = previsao_meta_path.read_text(encoding="latin-1")
    previsao_text = build_previsao_meta_text(
        previsao_template,
        forecast_start_time=window.forecast_start_time,
        forecast_nt=window.forecast_nt,
        mini_centroids=mini_centroids,
    )
    previsao_meta_path.write_text(previsao_text, encoding="latin-1")

    logger.info(
        "mgb_meta_updated parhig=%s previsao_meta=%s reference_time=%s start_time=%s forecast_start_time=%s nt=%s forecast_nt=%s dt_seconds=%s nc=%s input_days_before=%s forecast_horizon_days=%s",
        parhig_path,
        previsao_meta_path,
        window.reference_time.isoformat(timespec="seconds"),
        window.start_time.isoformat(timespec="seconds"),
        window.forecast_start_time.isoformat(timespec="seconds"),
        window.nt,
        window.forecast_nt,
        window.dt_seconds,
        nc,
        input_days_before,
        forecast_horizon_days,
    )
    return MgbMetaUpdateSummary(
        parhig_path=parhig_path,
        previsao_meta_path=previsao_meta_path,
        mini_gtp_path=mini_gtp_path,
        reference_time=window.reference_time,
        start_time=window.start_time,
        forecast_start_time=window.forecast_start_time,
        forecast_nt=window.forecast_nt,
        nt=window.nt,
        nc=nc,
        dt_seconds=window.dt_seconds,
        input_days_before=input_days_before,
        forecast_horizon_days=forecast_horizon_days,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reescreve PARHIG.hig e Previsao.meta a partir da configuracao do run.")
    parser.add_argument("--parhig", type=Path, default=DEFAULT_PARHIG, help="Arquivo PARHIG.hig a reescrever.")
    parser.add_argument(
        "--previsao-meta",
        type=Path,
        default=DEFAULT_PREVISAO_META,
        help="Arquivo Previsao.meta a reescrever.",
    )
    parser.add_argument("--mini-gtp", type=Path, default=DEFAULT_MINI_GTP, help="Arquivo MINI.gtp.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = rewrite_mgb_meta_from_config(
        parhig_path=args.parhig,
        previsao_meta_path=args.previsao_meta,
        mini_gtp_path=args.mini_gtp,
    )
    print(
        "mgb_meta_ready "
        f"parhig={summary.parhig_path} "
        f"previsao_meta={summary.previsao_meta_path} "
        f"reference_time={summary.reference_time.isoformat(timespec='seconds')} "
        f"start_time={summary.start_time.isoformat(timespec='seconds')} "
        f"forecast_start_time={summary.forecast_start_time.isoformat(timespec='seconds')} "
        f"nt={summary.nt} "
        f"forecast_nt={summary.forecast_nt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
