from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import SQL_DIR, interim_dir
from common.settings import load_settings


DEFAULT_PARHIG = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "PARHIG.hig"
DEFAULT_MINI_GTP = REPO_ROOT / "apps" / "mgb_runner" / "Input" / "MINI.gtp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apps" / "mgb_runner" / "Output"
DEFAULT_OUTPUT_DB = interim_dir() / "model_outputs.sqlite"
DEFAULT_SCHEMA_PATH = SQL_DIR / "model_outputs_schema.sql"
DEFAULT_CHUNK_HOURS = 42200
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


@dataclass(frozen=True, slots=True)
class VariableSpec:
    variable_code: str
    display_name: str
    unit: str


@dataclass(frozen=True, slots=True)
class OutputSource:
    variable_code: str
    prev_flag: int
    path: Path
    nt: int
    global_start_offset: int


@dataclass(frozen=True, slots=True)
class ExportWindow:
    reference_time: datetime
    reference_date: date
    window_start: datetime
    window_end_exclusive: datetime


@dataclass(frozen=True, slots=True)
class ExportSummary:
    database_path: Path
    reference_time: datetime
    window_start: datetime
    window_end_exclusive: datetime
    nc: int
    nt_current: int
    nt_forecast: int
    series_count: int
    value_count: int


VARIABLE_SPECS = (
    VariableSpec(variable_code="QTUDO", display_name="QTUDO", unit="m3/s"),
    VariableSpec(variable_code="YTUDO", display_name="YTUDO", unit="m"),
)


def _extract_numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def _next_data_line(lines: list[str], start_idx: int) -> str:
    for raw_line in lines[start_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        return raw_line
    raise ValueError("Could not find a data line after the header line.")


def _first_int(text: str) -> int:
    numbers = _extract_numbers(text)
    if not numbers:
        raise ValueError(f"No integer found in line: {text!r}")
    return int(float(numbers[0].replace(",", ".")))


def _isoformat_seconds(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def apply_schema(database_path: Path, schema_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(schema_sql)
        connection.commit()
    finally:
        connection.close()


def read_nc_from_parhig(parhig_path: Path) -> int:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()
        if "NC" in upper and "NU" in upper:
            return _first_int(_next_data_line(lines, idx))
    raise ValueError(f"Could not read NC from {parhig_path}")


def read_time_settings_from_parhig(parhig_path: Path) -> tuple[datetime, int]:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    start_time: datetime | None = None
    dt_seconds: int | None = None

    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()

        if start_time is None and all(token in upper for token in ("DIA", "MES", "ANO", "HORA")):
            numbers = _extract_numbers(_next_data_line(lines, idx))
            if len(numbers) >= 4:
                day = int(float(numbers[0].replace(",", ".")))
                month = int(float(numbers[1].replace(",", ".")))
                year = int(float(numbers[2].replace(",", ".")))
                hour = int(float(numbers[3].replace(",", ".")))
                start_time = datetime(year, month, day, hour)

        if dt_seconds is None and "NT" in upper and "DT" in upper:
            numbers = _extract_numbers(_next_data_line(lines, idx))
            if len(numbers) >= 2:
                dt_seconds = int(float(numbers[1].replace(",", ".")))

        if start_time is not None and dt_seconds is not None:
            break

    if start_time is None or dt_seconds is None:
        raise ValueError(
            f"Could not read start_time/dt_seconds from {parhig_path}. "
            "Expected PARHIG to provide DIA/MES/ANO/HORA and NT/DT."
        )
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")
    return start_time, dt_seconds


def read_mini_ids(mini_gtp_path: Path, *, nc: int) -> list[int]:
    header: list[str] | None = None
    mini_column_index: int | None = None
    mini_ids: list[int] = []

    with mini_gtp_path.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if header is None:
                header = parts
                if "Mini" not in header:
                    raise ValueError(f"MINI.gtp missing required column 'Mini': {mini_gtp_path}")
                mini_column_index = header.index("Mini")
                continue

            assert mini_column_index is not None
            if len(parts) <= mini_column_index:
                raise ValueError(f"Invalid MINI.gtp row: {raw_line.rstrip()}")
            mini_ids.append(int(float(parts[mini_column_index].replace(",", "."))))
            if len(mini_ids) == nc:
                break

    if len(mini_ids) < nc:
        raise ValueError(f"MINI.gtp has {len(mini_ids)} rows, smaller than NC={nc}")

    seen: set[int] = set()
    duplicated: list[int] = []
    for mini_id in mini_ids:
        if mini_id in seen and mini_id not in duplicated:
            duplicated.append(mini_id)
        seen.add(mini_id)
    if duplicated:
        sample = ", ".join(str(value) for value in duplicated[:5])
        raise ValueError(f"MINI.gtp has duplicated Mini ids (sample: {sample})")

    return mini_ids


def infer_nt_from_binary(file_path: Path, *, nc: int) -> int:
    size_bytes = file_path.stat().st_size
    if size_bytes % 4 != 0:
        raise ValueError(
            f"Invalid binary size in {file_path.name}: {size_bytes} bytes is not divisible by 4 (float32)."
        )
    total_floats = size_bytes // 4
    if total_floats % nc != 0:
        raise ValueError(
            f"Invalid shape in {file_path.name}: {total_floats} float32 values are not divisible by NC={nc}."
        )
    nt = total_floats // nc
    if nt <= 0:
        raise ValueError(f"Invalid NT inferred for {file_path.name}: NT={nt}")
    return int(nt)


def discover_output_sources(output_dir: Path, *, nc: int) -> dict[str, dict[int, OutputSource]]:
    sources: dict[str, dict[int, OutputSource]] = {}

    for spec in VARIABLE_SPECS:
        matches = [
            path
            for path in output_dir.iterdir()
            if path.is_file() and path.name.upper().startswith(spec.variable_code)
        ]
        if not matches:
            raise FileNotFoundError(f"No files found for {spec.variable_code} in {output_dir}")

        classified: dict[int, list[Path]] = {0: [], 1: []}
        for path in matches:
            prev_flag = 1 if "PREV" in path.stem.upper() else 0
            classified[prev_flag].append(path)

        if len(classified[0]) != 1 or len(classified[1]) != 1:
            names = sorted(path.name for path in matches)
            raise FileNotFoundError(
                f"Expected exactly one current file and one forecast file for {spec.variable_code} in {output_dir}. "
                f"Found: {names}"
            )

        sources[spec.variable_code] = {
            prev_flag: OutputSource(
                variable_code=spec.variable_code,
                prev_flag=prev_flag,
                path=classified[prev_flag][0],
                nt=infer_nt_from_binary(classified[prev_flag][0], nc=nc),
                global_start_offset=0,
            )
            for prev_flag in (0, 1)
        }

    return sources


def validate_source_lengths(
    sources: dict[str, dict[int, OutputSource]]
) -> tuple[dict[str, dict[int, OutputSource]], int, int]:
    nt_current_values = {variable_code: source_map[0].nt for variable_code, source_map in sources.items()}
    nt_forecast_values = {variable_code: source_map[1].nt for variable_code, source_map in sources.items()}

    nt_current_set = set(nt_current_values.values())
    if len(nt_current_set) != 1:
        raise ValueError(f"Inconsistent NT for current outputs: {nt_current_values}")

    nt_forecast_set = set(nt_forecast_values.values())
    if len(nt_forecast_set) != 1:
        raise ValueError(f"Inconsistent NT for forecast outputs: {nt_forecast_values}")

    nt_current = nt_current_set.pop()
    nt_forecast = nt_forecast_set.pop()

    normalized: dict[str, dict[int, OutputSource]] = {}
    for variable_code, source_map in sources.items():
        normalized[variable_code] = {
            0: OutputSource(
                variable_code=variable_code,
                prev_flag=0,
                path=source_map[0].path,
                nt=source_map[0].nt,
                global_start_offset=0,
            ),
            1: OutputSource(
                variable_code=variable_code,
                prev_flag=1,
                path=source_map[1].path,
                nt=source_map[1].nt,
                global_start_offset=nt_current,
            ),
        }

    return normalized, nt_current, nt_forecast


def build_export_window(reference_time: datetime, *, output_days_before: int, output_days_after: int) -> ExportWindow:
    reference_date = reference_time.date()
    window_start = datetime.combine(reference_date - timedelta(days=output_days_before), time.min)
    window_end_exclusive = datetime.combine(reference_date + timedelta(days=output_days_after + 1), time.min)
    return ExportWindow(
        reference_time=reference_time,
        reference_date=reference_date,
        window_start=window_start,
        window_end_exclusive=window_end_exclusive,
    )


def compute_global_row_bounds(
    *,
    start_time: datetime,
    dt_seconds: int,
    window_start: datetime,
    window_end_exclusive: datetime,
    total_nt: int,
) -> tuple[int, int]:
    start_delta_seconds = int((window_start - start_time).total_seconds())
    end_delta_seconds = int((window_end_exclusive - start_time).total_seconds())
    start_offset = max(0, _ceil_div(start_delta_seconds, dt_seconds))
    end_offset = min(total_nt, _ceil_div(end_delta_seconds, dt_seconds))
    if start_offset >= end_offset:
        raise ValueError("Configured window does not intersect available MGB outputs.")
    return start_offset, end_offset


def build_series_rows(
    mini_ids: list[int],
) -> tuple[list[tuple[int, str, int, int, str]], dict[tuple[str, int], dict[int, int]]]:
    rows: list[tuple[int, str, int, int, str]] = []
    lookup: dict[tuple[str, int], dict[int, int]] = {}
    series_id = 1

    for spec in VARIABLE_SPECS:
        for prev_flag in (0, 1):
            mapping: dict[int, int] = {}
            for mini_id in mini_ids:
                rows.append((series_id, spec.variable_code, mini_id, prev_flag, spec.unit))
                mapping[mini_id] = series_id
                series_id += 1
            lookup[(spec.variable_code, prev_flag)] = mapping

    return rows, lookup


def iter_value_rows(
    values_chunk: np.ndarray,
    *,
    dt_values: list[str],
    mini_ids: list[int],
    series_ids_by_mini: dict[int, int],
):
    for column_index, dt_value in enumerate(dt_values):
        column = values_chunk[:, column_index]
        for row_index, mini_id in enumerate(mini_ids):
            raw_value = float(column[row_index])
            value = raw_value if np.isfinite(raw_value) else None
            yield (series_ids_by_mini[mini_id], dt_value, value)


def load_output_window_from_settings() -> tuple[int, int]:
    settings = load_settings()
    mgb_settings = settings["mgb"]
    return int(mgb_settings["output_days_before"]), int(mgb_settings["output_days_after"])


def write_output_database(
    *,
    database_path: Path,
    schema_path: Path,
    mini_ids: list[int],
    sources: dict[str, dict[int, OutputSource]],
    start_time: datetime,
    dt_seconds: int,
    export_window: ExportWindow,
    nt_current: int,
    nt_forecast: int,
    chunk_hours: int,
) -> ExportSummary:
    apply_schema(database_path, schema_path)

    total_nt = nt_current + nt_forecast
    global_start_offset, global_end_offset = compute_global_row_bounds(
        start_time=start_time,
        dt_seconds=dt_seconds,
        window_start=export_window.window_start,
        window_end_exclusive=export_window.window_end_exclusive,
        total_nt=total_nt,
    )

    series_rows, series_lookup = build_series_rows(mini_ids)
    value_count = 0
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO metadata (reference_time, reference_date, window_start, window_end_exclusive, dt_seconds, nc, nt_current, nt_forecast) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _isoformat_seconds(export_window.reference_time),
                export_window.reference_date.isoformat(),
                _isoformat_seconds(export_window.window_start),
                _isoformat_seconds(export_window.window_end_exclusive),
                dt_seconds,
                len(mini_ids),
                nt_current,
                nt_forecast,
            ),
        )
        connection.executemany(
            "INSERT INTO variable (variable_code, display_name, unit) VALUES (?, ?, ?)",
            [(spec.variable_code, spec.display_name, spec.unit) for spec in VARIABLE_SPECS],
        )
        connection.executemany(
            "INSERT INTO output_series (series_id, variable_code, mini_id, prev_flag, unit) VALUES (?, ?, ?, ?, ?)",
            series_rows,
        )

        for spec in VARIABLE_SPECS:
            for prev_flag in (0, 1):
                source = sources[spec.variable_code][prev_flag]
                source_global_start = source.global_start_offset
                source_global_end = source.global_start_offset + source.nt
                overlap_start = max(global_start_offset, source_global_start)
                overlap_end = min(global_end_offset, source_global_end)
                if overlap_start >= overlap_end:
                    continue

                local_start = overlap_start - source_global_start
                local_end = overlap_end - source_global_start
                matrix = np.memmap(source.path, dtype=np.float32, mode="r", shape=(len(mini_ids), source.nt))
                series_ids_by_mini = series_lookup[(spec.variable_code, prev_flag)]

                try:
                    for chunk_start in range(local_start, local_end, chunk_hours):
                        chunk_end = min(chunk_start + chunk_hours, local_end)
                        values_chunk = np.asarray(matrix[:, chunk_start:chunk_end], dtype=np.float32)
                        dt_values = [
                            _isoformat_seconds(
                                start_time + timedelta(seconds=(source_global_start + offset) * dt_seconds)
                            )
                            for offset in range(chunk_start, chunk_end)
                        ]
                        connection.executemany(
                            "INSERT INTO output_value (series_id, dt, value) VALUES (?, ?, ?)",
                            iter_value_rows(
                                values_chunk,
                                dt_values=dt_values,
                                mini_ids=mini_ids,
                                series_ids_by_mini=series_ids_by_mini,
                            ),
                        )
                        value_count += len(mini_ids) * (chunk_end - chunk_start)
                finally:
                    del matrix

        connection.commit()
    finally:
        connection.close()

    return ExportSummary(
        database_path=database_path,
        reference_time=export_window.reference_time,
        window_start=export_window.window_start,
        window_end_exclusive=export_window.window_end_exclusive,
        nc=len(mini_ids),
        nt_current=nt_current,
        nt_forecast=nt_forecast,
        series_count=len(series_rows),
        value_count=value_count,
    )


def export_mgb_outputs(
    *,
    parhig_path: Path = DEFAULT_PARHIG,
    mini_gtp_path: Path = DEFAULT_MINI_GTP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_db_path: Path = DEFAULT_OUTPUT_DB,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_days_before: int | None = None,
    output_days_after: int | None = None,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
) -> ExportSummary:
    if chunk_hours <= 0:
        raise ValueError(f"chunk_hours must be > 0, got {chunk_hours}")

    if output_days_before is None or output_days_after is None:
        default_before, default_after = load_output_window_from_settings()
        if output_days_before is None:
            output_days_before = default_before
        if output_days_after is None:
            output_days_after = default_after

    if output_days_before < 0 or output_days_after < 0:
        raise ValueError("output_days_before and output_days_after must be >= 0.")

    nc = read_nc_from_parhig(parhig_path)
    start_time, dt_seconds = read_time_settings_from_parhig(parhig_path)
    mini_ids = read_mini_ids(mini_gtp_path, nc=nc)
    raw_sources = discover_output_sources(output_dir, nc=nc)
    sources, nt_current, nt_forecast = validate_source_lengths(raw_sources)

    reference_time = start_time + timedelta(seconds=(nt_current - 1) * dt_seconds)
    export_window = build_export_window(
        reference_time,
        output_days_before=output_days_before,
        output_days_after=output_days_after,
    )

    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_db_path = output_db_path.with_name(f"{output_db_path.stem}.{uuid4().hex[:8]}.tmp{output_db_path.suffix}")

    try:
        summary = write_output_database(
            database_path=temp_db_path,
            schema_path=schema_path,
            mini_ids=mini_ids,
            sources=sources,
            start_time=start_time,
            dt_seconds=dt_seconds,
            export_window=export_window,
            nt_current=nt_current,
            nt_forecast=nt_forecast,
            chunk_hours=chunk_hours,
        )
        temp_db_path.replace(output_db_path)
    except Exception:
        if temp_db_path.exists():
            temp_db_path.unlink()
        raise

    return ExportSummary(
        database_path=output_db_path,
        reference_time=summary.reference_time,
        window_start=summary.window_start,
        window_end_exclusive=summary.window_end_exclusive,
        nc=summary.nc,
        nt_current=summary.nt_current,
        nt_forecast=summary.nt_forecast,
        series_count=summary.series_count,
        value_count=summary.value_count,
    )


def main() -> int:
    summary = export_mgb_outputs()
    print(summary.database_path)
    print(f"reference_time={_isoformat_seconds(summary.reference_time)}")
    print(f"window_start={_isoformat_seconds(summary.window_start)}")
    print(f"window_end_exclusive={_isoformat_seconds(summary.window_end_exclusive)}")
    print(f"series_count={summary.series_count}")
    print(f"value_count={summary.value_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
