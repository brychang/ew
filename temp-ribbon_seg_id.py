from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume

DEFAULT_CV_PATH = (
    "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/"
    "stroeh_mouse_retina"
)
REQUIRED_COLUMNS = ("centroid_x", "centroid_y", "centroid_z")
_VOL: CloudVolume | None = None


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    rows: list[tuple[int, int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel centroid-to-segmentation lookup for ribbon synapses with "
            "chunked outputs and deterministic merge."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ribbon_v2_info.df"),
        help="Input table path (CSV-like file with index column and centroid columns).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ribbon_seg_id_runs"),
        help="Directory for run outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of process workers.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Rows per chunk written to disk.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--cv-path",
        type=str,
        default=DEFAULT_CV_PATH,
        help="CloudVolume segmentation path.",
    )
    parser.add_argument(
        "--mip",
        type=int,
        default=0,
        help="CloudVolume mip level for lookup.",
    )
    parser.add_argument(
        "--use-https",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable HTTPS requests in CloudVolume.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Chunk and final output table format.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")


def load_input_table(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path, header=0, index_col=0)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if limit is not None:
        df = df.head(limit)

    result = df.reset_index(names="ribbon_id")[
        ["ribbon_id", "centroid_x", "centroid_y", "centroid_z"]
    ].copy()
    for col in REQUIRED_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="raise")

    # CloudVolume indexing expects integer voxel coordinates.
    for col in REQUIRED_COLUMNS:
        result[col] = result[col].astype(np.int64)

    result.insert(0, "source_row", np.arange(len(result), dtype=np.int64))
    return result


def build_tasks(df: pd.DataFrame, chunk_size: int) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    total = len(df)
    for chunk_id, start in enumerate(range(0, total, chunk_size)):
        stop = min(start + chunk_size, total)
        chunk = df.iloc[start:stop]
        rows = list(
            zip(
                chunk["source_row"].tolist(),
                chunk["ribbon_id"].tolist(),
                chunk["centroid_x"].tolist(),
                chunk["centroid_y"].tolist(),
                chunk["centroid_z"].tolist(),
                strict=True,
            )
        )
        tasks.append(ChunkTask(chunk_id=chunk_id, rows=rows))
    return tasks


def init_worker(cv_path: str, mip: int, use_https: bool) -> None:
    global _VOL
    _VOL = CloudVolume(cv_path, mip=mip, use_https=use_https)


def _as_scalar_int(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def run_chunk(task: ChunkTask) -> tuple[int, list[tuple[int, int, int, int, int, int]]]:
    if _VOL is None:
        raise RuntimeError("Worker CloudVolume client was not initialized")

    out_rows: list[tuple[int, int, int, int, int, int]] = []
    for source_row, ribbon_id, x, y, z in task.rows:
        seg_id = _VOL[(int(x), int(y), int(z))]
        out_rows.append(
            (
                int(source_row),
                int(ribbon_id),
                int(x),
                int(y),
                int(z),
                _as_scalar_int(seg_id),
            )
        )
    return task.chunk_id, out_rows


def write_table(df: pd.DataFrame, path: Path, output_format: str) -> None:
    if output_format == "csv":
        df.to_csv(path, index=False)
        return
    df.to_parquet(path, index=False)


def read_table(path: Path, output_format: str) -> pd.DataFrame:
    if output_format == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def parallel_lookup(
    tasks: list[ChunkTask],
    workers: int,
    cv_path: str,
    mip: int,
    use_https: bool,
    chunk_dir: Path,
    output_format: str,
) -> dict[int, Path]:
    chunk_paths: dict[int, Path] = {}
    rows_done = 0
    start_time = time.time()
    total_rows = sum(len(t.rows) for t in tasks)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(cv_path, mip, use_https),
    ) as executor:
        future_to_task = {executor.submit(run_chunk, task): task for task in tasks}
        done, not_done = wait(future_to_task, return_when=FIRST_EXCEPTION)

        for fut in done:
            exc = fut.exception()
            if exc is not None:
                failing_task = future_to_task[fut]
                for pending in not_done:
                    pending.cancel()
                raise RuntimeError(
                    f"Fail-fast: chunk {failing_task.chunk_id} failed with error: {exc}"
                ) from exc

        for fut in future_to_task:
            chunk_id, rows = fut.result()
            out_df = pd.DataFrame(
                rows,
                columns=[
                    "source_row",
                    "ribbon_id",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                    "segmentation_id",
                ],
            )
            ext = "csv" if output_format == "csv" else "parquet"
            chunk_path = chunk_dir / f"chunk_{chunk_id:05d}.{ext}"
            write_table(out_df, chunk_path, output_format)
            chunk_paths[chunk_id] = chunk_path

            rows_done += len(out_df)
            elapsed = max(1e-6, time.time() - start_time)
            rate = rows_done / elapsed
            log(
                f"Completed chunk {chunk_id + 1}/{len(tasks)}; "
                f"rows {rows_done}/{total_rows}; rate {rate:.1f} rows/s"
            )

    return chunk_paths


def merge_chunks(
    chunk_paths: dict[int, Path],
    total_input_rows: int,
    final_path: Path,
    output_format: str,
) -> pd.DataFrame:
    if not chunk_paths:
        raise ValueError("No chunk output files were generated")

    dfs = [read_table(chunk_paths[i], output_format) for i in sorted(chunk_paths)]
    merged = pd.concat(dfs, ignore_index=True)
    merged.sort_values("source_row", inplace=True, kind="mergesort")
    merged.reset_index(drop=True, inplace=True)

    if len(merged) != total_input_rows:
        raise RuntimeError(
            f"Row-count mismatch after merge: expected {total_input_rows}, got {len(merged)}"
        )
    if merged["source_row"].duplicated().any():
        dup_n = int(merged["source_row"].duplicated().sum())
        raise RuntimeError(f"Duplicate source_row values after merge: {dup_n}")
    if merged["source_row"].isna().any():
        raise RuntimeError("Missing source_row values after merge")
    if merged["segmentation_id"].isna().any():
        raise RuntimeError("Missing segmentation_id values after merge")

    write_table(merged, final_path, output_format)
    return merged


def prepare_run_dirs(
    base_output_dir: Path, output_format: str
) -> tuple[Path, Path, Path]:
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_output_dir / run_name
    chunk_dir = run_dir / "chunks"
    run_dir.mkdir(parents=True, exist_ok=False)
    chunk_dir.mkdir(parents=True, exist_ok=False)
    ext = "csv" if output_format == "csv" else "parquet"
    final_path = run_dir / f"ribbon_with_seg_ids.{ext}"
    return run_dir, chunk_dir, final_path


def main() -> None:
    args = parse_args()
    ensure_valid_args(args)

    log("Loading input table")
    input_df = load_input_table(args.input, args.limit)
    total_rows = len(input_df)
    if total_rows == 0:
        raise ValueError("No rows found in input table after filtering/limit")

    task_count = math.ceil(total_rows / args.chunk_size)
    log(
        f"Prepared {total_rows} rows, {task_count} chunks, "
        f"workers={args.workers}, format={args.format}"
    )
    tasks = build_tasks(input_df, args.chunk_size)

    run_dir, chunk_dir, final_path = prepare_run_dirs(args.output_dir, args.format)
    log(f"Run directory: {run_dir}")

    started = time.time()
    chunk_paths = parallel_lookup(
        tasks=tasks,
        workers=args.workers,
        cv_path=args.cv_path,
        mip=args.mip,
        use_https=args.use_https,
        chunk_dir=chunk_dir,
        output_format=args.format,
    )
    merged = merge_chunks(
        chunk_paths=chunk_paths,
        total_input_rows=total_rows,
        final_path=final_path,
        output_format=args.format,
    )
    elapsed = time.time() - started
    rate = len(merged) / max(1e-6, elapsed)
    log(f"Finished. Rows={len(merged)} elapsed={elapsed:.1f}s rate={rate:.1f} rows/s")
    log(f"Final output: {final_path}")


if __name__ == "__main__":
    main()
