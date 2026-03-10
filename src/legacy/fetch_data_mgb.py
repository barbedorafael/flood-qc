from __future__ import annotations

"""
Read MGB-Hora outputs (QTUDO*/YTUDO*) and write consolidated wide parquet files.

Pipeline:
1) read NC/start_time/dt_seconds from PARHIG.hig;
2) read Mini ids from MINI.gtp in row order;
3) infer NT from each binary file size (float32);
4) write one parquet per variable (QTUDO, YTUDO) with columns:
   prev, dt, <mini_id_1>, <mini_id_2>, ...;
5) write q_meta.json and y_meta.json in processed/.
"""

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MGB_ROOT = REPO_ROOT / "data" / "mgb-hora"
DEFAULT_PARHIG = MGB_ROOT / "Input" / "PARHIG.hig"
DEFAULT_MINI_GTP = MGB_ROOT / "Input" / "MINI.gtp"
DEFAULT_OUTPUT_DIR = MGB_ROOT / "Output"
DEFAULT_PROCESSED_DIR = MGB_ROOT / "processed"
DEFAULT_CHUNK_NT = 42200 # 30 days of hourly data, adjust as needed for memory constraints


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:[.,]\d+)?", text)


def _first_int(text: str) -> int:
    numbers = _extract_numbers(text)
    if not numbers:
        raise ValueError(f"No integer found in line: {text!r}")
    return int(float(numbers[0].replace(",", ".")))


def _next_data_line(lines: list[str], start_idx: int) -> str:
    for raw_line in lines[start_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        return raw_line
    raise ValueError("Could not find a data line after the header line.")


def _to_repo_relative_posix(path: Path) -> str:
    path_abs = path.resolve()
    repo_abs = REPO_ROOT.resolve()
    try:
        rel = path_abs.relative_to(repo_abs)
    except ValueError:
        rel = path_abs
    return rel.as_posix()


def read_nc_from_parhig(parhig_path: Path = DEFAULT_PARHIG) -> int:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()
        if "NC" in upper and "NU" in upper:
            return _first_int(_next_data_line(lines, idx))
    raise ValueError(f"Could not read NC from {parhig_path}")


def read_time_settings_from_parhig(parhig_path: Path = DEFAULT_PARHIG) -> tuple[datetime | None, int | None]:
    lines = parhig_path.read_text(encoding="latin-1").splitlines()
    start_time: datetime | None = None
    dt_seconds: int | None = None

    for idx, raw_line in enumerate(lines):
        upper = raw_line.upper()

        if start_time is None and all(token in upper for token in ("DIA", "MES", "ANO", "HORA")):
            data_line = _next_data_line(lines, idx)
            numbers = _extract_numbers(data_line)
            if len(numbers) >= 4:
                day = int(float(numbers[0].replace(",", ".")))
                month = int(float(numbers[1].replace(",", ".")))
                year = int(float(numbers[2].replace(",", ".")))
                hour = int(float(numbers[3].replace(",", ".")))
                start_time = datetime(year, month, day, hour)

        if dt_seconds is None and "NT" in upper and "DT" in upper:
            data_line = _next_data_line(lines, idx)
            numbers = _extract_numbers(data_line)
            if len(numbers) >= 2:
                dt_seconds = int(float(numbers[1].replace(",", ".")))

        if start_time is not None and dt_seconds is not None:
            break

    return start_time, dt_seconds


def read_mini_ids(mini_gtp_path: Path = DEFAULT_MINI_GTP, *, nc: int) -> tuple[np.ndarray, bool]:
    mini_raw = pd.read_csv(mini_gtp_path, sep=r"\s+", engine="python")
    if "Mini" not in mini_raw.columns:
        raise ValueError(f"MINI.gtp missing required column 'Mini': {mini_gtp_path}")
    if len(mini_raw) < nc:
        raise ValueError(f"MINI.gtp has {len(mini_raw)} rows, smaller than NC={nc}")

    mini_series = pd.to_numeric(mini_raw.iloc[:nc]["Mini"], errors="raise").astype(np.int64)
    duplicated = mini_series[mini_series.duplicated()].unique()
    if len(duplicated) > 0:
        sample = ", ".join(str(int(v)) for v in duplicated[:5])
        raise ValueError(f"MINI.gtp has duplicated Mini ids (sample: {sample})")

    mini_ids = mini_series.to_numpy(dtype=np.int32)
    is_sequential = bool((mini_ids == np.arange(1, nc + 1, dtype=np.int32)).all())
    return mini_ids, is_sequential


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


def discover_variable_files(output_dir: Path, *, variable: str, nc: int) -> list[dict[str, Any]]:
    files = [path for path in output_dir.glob(f"{variable}*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found for {variable} in {output_dir}")

    sources: list[dict[str, Any]] = []
    for file_path in files:
        nt = infer_nt_from_binary(file_path, nc=nc)
        sources.append(
            {
                "path": file_path,
                "prev": "prev" in file_path.stem.lower(),
                "nt": nt,
            }
        )

    # Required order: non-forecast first, forecast files last.
    sources.sort(key=lambda item: (bool(item["prev"]), str(item["path"].name).lower()))
    return sources


def _build_chunk_frame(
    values_chunk: np.ndarray,
    *,
    mini_columns: list[str],
    prev_flag: int,
    start_time: datetime,
    dt_seconds: int,
    global_time_offset: int,
) -> pd.DataFrame:
    rows = int(values_chunk.shape[1])
    values_by_time = np.asarray(values_chunk.T, dtype=np.float32)
    frame = pd.DataFrame(values_by_time, columns=mini_columns, copy=False)

    time_offsets = np.arange(global_time_offset, global_time_offset + rows, dtype=np.int64)
    dt_col = np.datetime64(start_time, "ns") + time_offsets * np.timedelta64(dt_seconds, "s")
    frame.insert(0, "dt", dt_col.astype("datetime64[ns]"))
    frame.insert(0, "prev", np.full(rows, int(prev_flag), dtype=np.int8))
    return frame


def write_variable_parquet(
    *,
    variable: str,
    sources: list[dict[str, Any]],
    mini_ids: np.ndarray,
    nc: int,
    start_time: datetime,
    dt_seconds: int,
    output_file: Path,
    chunk_nt: int,
) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Parquet export requires pyarrow. Install with: pip install pyarrow") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    mini_columns = [str(int(mini_id)) for mini_id in mini_ids]
    writer: Any | None = None
    nt_written = 0

    try:
        for source in sources:
            nt = int(source["nt"])
            prev_flag = 1 if bool(source["prev"]) else 0
            matrix = np.memmap(source["path"], dtype=np.float32, mode="r", shape=(nc, nt))

            for chunk_start in range(0, nt, chunk_nt):
                chunk_end = min(chunk_start + chunk_nt, nt)
                chunk_values = matrix[:, chunk_start:chunk_end]

                chunk_df = _build_chunk_frame(
                    chunk_values,
                    mini_columns=mini_columns,
                    prev_flag=prev_flag,
                    start_time=start_time,
                    dt_seconds=dt_seconds,
                    global_time_offset=nt_written,
                )
                chunk_table = pa.Table.from_pandas(chunk_df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_file.as_posix(), chunk_table.schema, compression="snappy")
                writer.write_table(chunk_table)
                nt_written += int(chunk_end - chunk_start)

            del matrix
    finally:
        if writer is not None:
            writer.close()

    if nt_written <= 0:
        raise ValueError(f"{variable} produced no rows for parquet export.")
    return nt_written


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def build_metadata_payload(
    *,
    variable: str,
    nc: int,
    nt: int,
    start_time: datetime,
    dt_seconds: int,
    output_file: Path,
    sources: list[dict[str, Any]],
    mini_columns_count: int,
    mini_sequential_1_to_nc: bool,
) -> dict[str, Any]:
    return {
        "variable": variable,
        "nc": int(nc),
        "nt": int(nt),
        "start_time": start_time.isoformat(),
        "dt_seconds": int(dt_seconds),
        "output_file": _to_repo_relative_posix(output_file),
        "source_files": [
            {
                "file": _to_repo_relative_posix(source["path"]),
                "prev": bool(source["prev"]),
                "nt": int(source["nt"]),
            }
            for source in sources
        ],
        "mini_columns_count": int(mini_columns_count),
        "mini_sequential_1_to_nc": bool(mini_sequential_1_to_nc),
    }


def main(
    *,
    parhig_path: Path = DEFAULT_PARHIG,
    mini_gtp_path: Path = DEFAULT_MINI_GTP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    chunk_nt: int = DEFAULT_CHUNK_NT,
) -> None:
    if chunk_nt <= 0:
        raise ValueError(f"chunk_nt must be > 0, got {chunk_nt}")

    nc = read_nc_from_parhig(parhig_path)
    start_time, dt_seconds = read_time_settings_from_parhig(parhig_path)
    if start_time is None or dt_seconds is None:
        raise ValueError(
            f"Could not read start_time/dt_seconds from {parhig_path}. "
            "Expected PARHIG to provide DIA/MES/ANO/HORA and NT/DT."
        )
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")

    mini_ids, mini_is_sequential = read_mini_ids(mini_gtp_path, nc=nc)
    processed_dir.mkdir(parents=True, exist_ok=True)

    variable_targets = (
        ("QTUDO", processed_dir / "QTUDO.parquet", processed_dir / "q_meta.json"),
        ("YTUDO", processed_dir / "YTUDO.parquet", processed_dir / "y_meta.json"),
    )

    for variable, parquet_path, meta_path in variable_targets:
        sources = discover_variable_files(output_dir, variable=variable, nc=nc)
        nt_written = write_variable_parquet(
            variable=variable,
            sources=sources,
            mini_ids=mini_ids,
            nc=nc,
            start_time=start_time,
            dt_seconds=dt_seconds,
            output_file=parquet_path,
            chunk_nt=chunk_nt,
        )

        expected_nt = int(sum(int(source["nt"]) for source in sources))
        if nt_written != expected_nt:
            raise RuntimeError(
                f"{variable} row mismatch after write: expected {expected_nt}, got {nt_written}."
            )

        metadata = build_metadata_payload(
            variable=variable,
            nc=nc,
            nt=nt_written,
            start_time=start_time,
            dt_seconds=dt_seconds,
            output_file=parquet_path,
            sources=sources,
            mini_columns_count=len(mini_ids),
            mini_sequential_1_to_nc=mini_is_sequential,
        )
        write_metadata(meta_path, metadata)

        print(f"{variable}: NC={nc} NT={nt_written}")
        print(f"  parquet: {_to_repo_relative_posix(parquet_path)}")
        print(f"  meta: {_to_repo_relative_posix(meta_path)}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Read MGB-Hora outputs and write consolidated QTUDO/YTUDO parquet tables."
    )
    parser.add_argument(
        "--chunk-nt",
        type=int,
        default=DEFAULT_CHUNK_NT,
        help="Rows per write chunk in parquet export (default: 720).",
    )
    args = parser.parse_args()
    main(chunk_nt=args.chunk_nt)
