# %%
import ast
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Callable

import pandas as pd
from caveclient import CAVEclient

# %%
DATASET = "stroeh_mouse_retina"
_THREAD_LOCAL = local()
MAX_WORKERS = int(os.getenv("EWII_MAX_WORKERS", "8"))

# %%
csv_path = Path("data/ewii_20260414_latest_per_user_parallel.csv")
column_name = "Final SegID"

# %%
unique_final_segids: set[str] = set()

with csv_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None or column_name not in reader.fieldnames:
        raise ValueError(
            f"Column '{column_name}' not found. Available columns: {reader.fieldnames}"
        )

    for row in reader:
        value = (row.get(column_name) or "").strip()
        if value:
            unique_final_segids.add(value)

print(f"Unique final segids: {len(unique_final_segids):,}")


# %%
def load_target_seg_ids(csv_path: Path) -> list[int]:
    """Load Final SegIDs where both proofreader fields are non-empty."""
    spreadsheet = pd.read_csv(
        csv_path, dtype={"Final SegID": "string"}, low_memory=False
    )

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


def load_proofreader_name_map(mapping_csv_path: Path) -> dict[str, str]:
    """Map 611 Proofreader names to user_name prefixes.

    The `matched` column is intentionally ignored because some rows have been
    filled in manually and are now valid regardless of that flag.
    """
    mapping = pd.read_csv(mapping_csv_path, dtype="string", keep_default_na=False)
    mapping["611 Proofreader"] = mapping["611 Proofreader"].str.strip()
    mapping["matching_user_name"] = mapping["matching_user_name"].str.strip()

    mapping = mapping[
        (mapping["611 Proofreader"] != "") & (mapping["matching_user_name"] != "")
    ]

    # Preserve the last entry for any duplicated proofreader spelling.
    return dict(zip(mapping["611 Proofreader"], mapping["matching_user_name"]))


def load_target_proofreaders(csv_path: Path, mapping_csv_path: Path) -> pd.DataFrame:
    """Return filtered rows with the 611 proofreader and mapped user_name."""
    spreadsheet = pd.read_csv(
        csv_path,
        dtype={"Final SegID": "string", "611 Proofreader": "string"},
        low_memory=False,
    )

    for col in ["611 Proofreader", "Proofreader 2"]:
        spreadsheet[col] = spreadsheet[col].fillna("").str.strip()

    filtered = spreadsheet[
        (spreadsheet["611 Proofreader"] != "")
        & (spreadsheet["Proofreader 2"] != "")
        & spreadsheet["Final SegID"].notna()
    ].copy()

    filtered["Final SegID"] = pd.to_numeric(
        filtered["Final SegID"], errors="coerce"
    ).astype("Int64")
    filtered = filtered.dropna(subset=["Final SegID"])

    # Keep one row per seg_id while preserving the original order.
    filtered = filtered.drop_duplicates(subset=["Final SegID"], keep="first")

    proofreader_to_user_name = load_proofreader_name_map(mapping_csv_path)
    filtered["user_name"] = filtered["611 Proofreader"].map(proofreader_to_user_name)

    return filtered[["Final SegID", "611 Proofreader", "user_name"]]


def parse_after_root_ids(value: object) -> list[int]:
    """Parse the serialized after_root_ids field into a Python list."""
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [int(text)] if text.isdigit() else [text]

    if isinstance(parsed, list):
        return parsed
    if parsed is None:
        return []
    return [parsed]


def get_client() -> CAVEclient:
    """Return a thread-local CAVE client."""
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = CAVEclient(DATASET)
        _THREAD_LOCAL.client = client
    return client


@lru_cache(maxsize=200_000)
def get_leaves_cached(root_id: int) -> frozenset[int]:
    """Fetch and cache leaves for a root ID to avoid duplicate API calls."""
    client = get_client()
    leaves = client.chunkedgraph.get_leaves(root_id=int(root_id))
    return frozenset(int(leaf) for leaf in leaves)


@lru_cache(maxsize=50_000)
def get_original_ids_cached(root_id: int) -> tuple[int, ...]:
    """Fetch and cache sorted original IDs from a lineage graph."""
    client = get_client()
    lineage_graph = client.chunkedgraph.get_lineage_graph(int(root_id))

    sources = {int(link["source"]) for link in lineage_graph.get("links", [])}
    targets = {int(link["target"]) for link in lineage_graph.get("links", [])}
    original_ids = sorted(sources - targets)

    if not original_ids:
        return (int(root_id),)

    return tuple(original_ids)


def compute_overlap_metrics(
    before_root_id: int, after_root_id: int
) -> dict[str, object]:
    """Compute precision, recall, and IoU between two root IDs using leaf overlap."""
    before_leaves = get_leaves_cached(int(before_root_id))
    after_leaves = get_leaves_cached(int(after_root_id))

    tp = len(before_leaves & after_leaves)
    fp = len(before_leaves - after_leaves)
    fn = len(after_leaves - before_leaves)
    union = tp + fp + fn

    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def score_best_after_segment(task: dict[str, object]) -> dict[str, object]:
    """Score a task against all after segments and keep the best IoU match."""
    final_seg_id = int(task["Final SegID"])
    after_segments = task.get("after_segments") or []

    result: dict[str, object] = {
        "Final SegID": final_seg_id,
        "611 Proofreader": task.get("611 Proofreader", ""),
        "user_name": task.get("user_name", ""),
        "after_segments": after_segments,
        "after_segment_count": len(after_segments),
        "best_after_segment": pd.NA,
        "precision": pd.NA,
        "recall": pd.NA,
        "iou": pd.NA,
        "hit": False,
        "error": pd.NA,
    }

    if not after_segments:
        return result

    try:
        best_metrics: dict[str, object] | None = None
        best_after_segment: int | None = None
        best_error: str | None = None

        for after_segment in after_segments:
            try:
                # Precision/recall convention: predicted=pass 1, true=final.
                metrics = compute_overlap_metrics(int(after_segment), final_seg_id)
            except Exception as exc:  # noqa: BLE001 - keep per-candidate failures local.
                best_error = str(exc)
                continue

            metrics_iou = float(metrics["iou"])
            if best_metrics is None or metrics_iou > float(best_metrics["iou"]):
                best_metrics = metrics
                best_after_segment = int(after_segment)

        if best_metrics is None or best_after_segment is None:
            result["error"] = best_error or "No after segments could be scored"
            return result

        result.update(
            {
                "best_after_segment": best_after_segment,
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "iou": best_metrics["iou"],
                "hit": True,
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 - preserve one-row failure isolation.
        result["error"] = str(exc)
        return result


def get_original_ids(root_id: int) -> list[int]:
    """Return original/root-source IDs for a given root from its lineage graph."""
    return list(get_original_ids_cached(int(root_id)))


def score_tasks_parallel(
    task_records: list[dict[str, object]],
    scorer: Callable[[dict[str, object]], dict[str, object]],
    max_workers: int = MAX_WORKERS,
    progress_label: str = "Scoring",
    progress_every: int = 25,
) -> list[dict[str, object]]:
    """Run independent row-level scoring in parallel."""
    if not task_records:
        return []

    total = len(task_records)
    workers = max(1, int(max_workers))
    progress_every = max(1, int(progress_every))
    print(
        f"{progress_label}: starting {total:,} tasks with {workers} workers...",
        flush=True,
    )

    ordered_results: list[dict[str, object] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(scorer, task): index
            for index, task in enumerate(task_records)
        }

        for completed_count, future in enumerate(
            as_completed(future_to_index), start=1
        ):
            index = future_to_index[future]
            ordered_results[index] = future.result()

            if completed_count % progress_every == 0 or completed_count == total:
                print(
                    f"{progress_label}: {completed_count:,}/{total:,} done",
                    flush=True,
                )

    return [result for result in ordered_results if result is not None]


def score_best_original_segment(task: dict[str, object]) -> dict[str, object]:
    """Score best-after segment against original IDs and keep best IoU match."""
    final_seg_id = int(task["Final SegID"])
    best_after_segment = task.get("best_after_segment")

    result: dict[str, object] = {
        "Final SegID": final_seg_id,
        "611 Proofreader": task.get("611 Proofreader", ""),
        "user_name": task.get("user_name", ""),
        "best_after_segment": best_after_segment,
        "original_ids": [],
        "original_id_count": 0,
        "best_original_id": pd.NA,
        "after_to_original_precision": pd.NA,
        "after_to_original_recall": pd.NA,
        "after_to_original_iou": pd.NA,
        "hit": False,
        "error": pd.NA,
    }

    if best_after_segment is None or pd.isna(best_after_segment):
        result["error"] = "Missing best_after_segment"
        return result

    try:
        best_after_segment_int = int(best_after_segment)
        original_ids = get_original_ids(best_after_segment_int)
        result["original_ids"] = original_ids
        result["original_id_count"] = len(original_ids)

        best_metrics: dict[str, object] | None = None
        best_original_id: int | None = None
        best_error: str | None = None

        for original_id in original_ids:
            try:
                # Precision/recall convention: predicted=original, true=pass 1.
                metrics = compute_overlap_metrics(
                    int(original_id), best_after_segment_int
                )
            except Exception as exc:  # noqa: BLE001 - keep per-candidate failures local.
                best_error = str(exc)
                continue

            metrics_iou = float(metrics["iou"])
            if best_metrics is None or metrics_iou > float(best_metrics["iou"]):
                best_metrics = metrics
                best_original_id = int(original_id)

        if best_metrics is None or best_original_id is None:
            result["error"] = best_error or "No original IDs could be scored"
            return result

        result.update(
            {
                "best_original_id": best_original_id,
                "after_to_original_precision": best_metrics["precision"],
                "after_to_original_recall": best_metrics["recall"],
                "after_to_original_iou": best_metrics["iou"],
                "hit": True,
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 - preserve one-row failure isolation.
        result["error"] = str(exc)
        return result


def load_latest_actions_for_tasks(
    tasks_df: pd.DataFrame, parallel_results_path: Path
) -> pd.DataFrame:
    """Find the latest parallel result per Final SegID and mapped user_name.

    Rows with no matching hit are preserved and reported with empty after segments.
    """
    results = pd.read_csv(parallel_results_path, low_memory=False)
    if "user_name" not in results.columns:
        raise ValueError(
            f"Expected column 'user_name' in {parallel_results_path}, found: {list(results.columns)}"
        )

    results["Final SegID"] = pd.to_numeric(
        results["Final SegID"], errors="coerce"
    ).astype("Int64")
    results["user_name"] = results["user_name"].fillna("").astype(str).str.strip()
    results["ts_utc"] = pd.to_datetime(results["ts_utc"], errors="coerce", utc=True)
    results["after_segments"] = results["after_root_ids"].apply(parse_after_root_ids)

    tasks = tasks_df.copy()
    tasks["Final SegID"] = pd.to_numeric(tasks["Final SegID"], errors="coerce").astype(
        "Int64"
    )
    tasks["user_name"] = tasks["user_name"].fillna("").astype(str).str.strip()

    merged = tasks.merge(
        results,
        on=["Final SegID", "user_name"],
        how="left",
        suffixes=("", "_parallel"),
    )

    merged = merged.sort_values(
        ["Final SegID", "user_name", "ts_utc", "operation_id"],
        kind="mergesort",
    ).drop_duplicates(subset=["Final SegID", "user_name"], keep="last")

    merged["hit"] = merged["operation_id"].notna()
    merged["after_segments"] = merged["after_segments"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return merged[
        [
            "Final SegID",
            "611 Proofreader",
            "user_name",
            "hit",
            "operation_id",
            "ts_utc",
            "after_segments",
        ]
    ]


seg_ids = load_target_seg_ids(Path("data/ewii_20260414.csv"))
print(len(seg_ids))

# %%
proofreader_table = load_target_proofreaders(
    Path("data/ewii_20260414.csv"),
    Path("data/ewii_20260414_proofreader_to_user_name_prefix_attempts.csv"),
)

print(proofreader_table["611 Proofreader"].nunique())

# %%
latest_action_table = load_latest_actions_for_tasks(
    proofreader_table,
    Path("data/ewii_20260414_latest_per_user_parallel.csv"),
)

print(
    f"Latest actions loaded: {len(latest_action_table):,} rows; "
    f"with hits: {int(latest_action_table['hit'].sum()):,}; "
    f"with after_segments: {int(latest_action_table['after_segments'].map(bool).sum()):,}"
)

# %%
tasks_with_after_segments = latest_action_table[
    latest_action_table["after_segments"].map(bool)
].copy()

scored_rows: list[dict[str, object]] = []
task_records = tasks_with_after_segments.to_dict(orient="records")
scored_columns = [
    "Final SegID",
    "611 Proofreader",
    "user_name",
    "after_segments",
    "after_segment_count",
    "best_after_segment",
    "precision",
    "recall",
    "iou",
    "hit",
    "error",
]

if task_records:
    scored_rows = score_tasks_parallel(
        task_records,
        score_best_after_segment,
        progress_label="Stage 1 (final->best-after)",
    )

scored_table = pd.DataFrame(scored_rows, columns=scored_columns)
if not scored_table.empty:
    scored_table = scored_table.sort_values(
        ["Final SegID", "user_name"], kind="mergesort"
    )

scored_output_path = Path("data/ewii_20260414_latest_after_segment_metrics.csv")
scored_output_path.parent.mkdir(parents=True, exist_ok=True)
scored_table.to_csv(scored_output_path, index=False)
print(f"Wrote: {scored_output_path}")
print(f"Scored rows: {len(scored_table):,}")
print(f"Rows with non-empty after_segments: {len(tasks_with_after_segments):,}")

# %%
# Compare each best-after segment to its original IDs, then keep the best original match.
scored_after_hits = scored_table[scored_table["hit"]].copy()

original_scored_rows: list[dict[str, object]] = []
original_task_records = scored_after_hits.to_dict(orient="records")
original_scored_columns = [
    "Final SegID",
    "611 Proofreader",
    "user_name",
    "best_after_segment",
    "original_ids",
    "original_id_count",
    "best_original_id",
    "after_to_original_precision",
    "after_to_original_recall",
    "after_to_original_iou",
    "hit",
    "error",
]

if original_task_records:
    original_scored_rows = score_tasks_parallel(
        original_task_records,
        score_best_original_segment,
        progress_label="Stage 2 (best-after->best-original)",
    )

original_scored_table = pd.DataFrame(
    original_scored_rows, columns=original_scored_columns
)
if not original_scored_table.empty:
    original_scored_table = original_scored_table.sort_values(
        ["Final SegID", "user_name"], kind="mergesort"
    )

original_scored_output_path = Path(
    "data/ewii_20260414_latest_after_to_original_metrics.csv"
)
original_scored_output_path.parent.mkdir(parents=True, exist_ok=True)
original_scored_table.to_csv(original_scored_output_path, index=False)
print(f"Wrote: {original_scored_output_path}")
print(f"Scored rows (after->original): {len(original_scored_table):,}")
print(f"Rows with best-after hits to evaluate: {len(scored_after_hits):,}")
