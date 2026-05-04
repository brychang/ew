from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from caveclient import CAVEclient

DATASET = "stroeh_mouse_retina"
DEFAULT_INPUT_CSV = Path("data/ewii_20260414.csv")
DEFAULT_OUTPUT_CSV = Path("data/ewii_20260414_latest_per_user_parallel.csv")
TS_UNIT = "ms"

# Global client, one per process (set by init_worker).
_CLIENT: CAVEclient | None = None


@dataclass
class WorkerResult:
    seg_id: int
    rows: list[dict[str, Any]]
    error: str | None = None


def init_worker(dataset: str) -> None:
    """Initialize one CAVE client per process."""
    global _CLIENT
    _CLIENT = CAVEclient(dataset)


def fetch_latest_per_user(seg_id: int) -> WorkerResult:
    """Fetch one seg ID and keep one operation per user at max timestamp."""
    global _CLIENT
    if _CLIENT is None:
        # Fallback for direct execution without worker initializer.
        _CLIENT = CAVEclient(DATASET)

    try:
        tcl = _CLIENT.chunkedgraph.get_tabular_change_log([seg_id])[seg_id]
    except Exception as exc:  # noqa: BLE001 - we want to preserve per-ID failures.
        return WorkerResult(seg_id=seg_id, rows=[], error=str(exc))

    if tcl.empty:
        return WorkerResult(seg_id=seg_id, rows=[])

    tcl = tcl.copy()
    tcl["ts_utc"] = pd.to_datetime(tcl["timestamp"], unit=TS_UNIT, utc=True)

    latest_per_user = (
        tcl.sort_values(["user_id", "timestamp", "operation_id"])
        .groupby("user_id", as_index=False)
        .tail(1)
        .sort_values("timestamp")
    )

    rows: list[dict[str, Any]] = []
    for _, row in latest_per_user.iterrows():
        rows.append(
            {
                "Final SegID": seg_id,
                "operation_id": row["operation_id"],
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "ts_utc": row["ts_utc"],
                "after_root_ids": row["after_root_ids"],
            }
        )

    return WorkerResult(seg_id=seg_id, rows=rows)


def load_target_seg_ids(csv_path: Path) -> list[int]:
    """Load Final SegIDs where both proofreader fields are non-empty."""
    spreadsheet = pd.read_csv(csv_path, dtype={"Final SegID": "string"})

    for col in ["611 Proofreader", "Proofreader 2"]:
        spreadsheet[col] = spreadsheet[col].fillna("").str.strip()

    filtered = spreadsheet[
        (spreadsheet["611 Proofreader"] != "")
        & (spreadsheet["Proofreader 2"] != "")
        & spreadsheet["Final SegID"].notna()
    ]

    seg_series = (
        pd.to_numeric(filtered["Final SegID"], errors="coerce").dropna().astype("int64")
    )
    # Keep ordering stable while removing duplicates.
    seg_ids = list(dict.fromkeys(seg_series.tolist()))
    return seg_ids


def run_parallel(
    seg_ids: list[int], workers: int, dataset: str
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Run one-by-one TCL fetches in parallel and collect successes + failures."""
    total = len(seg_ids)
    all_rows: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []

    if total == 0:
        return pd.DataFrame(
            columns=[
                "Final SegID",
                "operation_id",
                "user_id",
                "user_name",
                "ts_utc",
                "after_root_ids",
            ]
        ), failures

    completed = 0
    with mp.Pool(
        processes=workers, initializer=init_worker, initargs=(dataset,)
    ) as pool:
        for result in pool.imap_unordered(fetch_latest_per_user, seg_ids, chunksize=8):
            completed += 1
            if result.error is not None:
                failures.append((result.seg_id, result.error))
            else:
                all_rows.extend(result.rows)

            if completed % 25 == 0 or completed == total:
                print(f"Progress: {completed}/{total} seg IDs processed")

    out_df = pd.DataFrame(
        all_rows,
        columns=[
            "Final SegID",
            "operation_id",
            "user_id",
            "user_name",
            "ts_utc",
            "after_root_ids",
        ],
    )
    return out_df, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch tabular change logs one seg ID at a time, in parallel, and keep "
            "the latest operation per unique user."
        )
    )
    parser.add_argument("--dataset", default=DATASET, help="CAVE dataset name")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_CSV, help="Input spreadsheet CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV for aggregated rows",
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path("data/ewii_20260414_latest_per_user_parallel_failures.csv"),
        help="Output CSV for seg IDs that failed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (mp.cpu_count() or 2) - 1)),
        help="Number of worker processes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seg_ids = load_target_seg_ids(args.input)

    print(f"Loaded {len(seg_ids)} eligible seg IDs from {args.input}")
    print(f"Using {args.workers} worker processes")

    final_results, failures = run_parallel(
        seg_ids=seg_ids, workers=args.workers, dataset=args.dataset
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(args.output, index=False)

    failures_df = pd.DataFrame(failures, columns=["Final SegID", "error"])
    failures_df.to_csv(args.failures_output, index=False)

    print(f"Wrote {len(final_results)} rows to {args.output}")
    print(f"Failed seg IDs: {len(failures)} (saved to {args.failures_output})")


if __name__ == "__main__":
    main()
