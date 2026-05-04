from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_metrics(stage1_csv: Path, stage2_csv: Path) -> pd.DataFrame:
    """Load and join stage-1 and stage-2 metrics with consistent ID dtypes."""
    stage1 = pd.read_csv(stage1_csv, low_memory=False)
    stage2 = pd.read_csv(stage2_csv, low_memory=False)

    id_cols_stage1 = ["Final SegID", "best_after_segment"]
    id_cols_stage2 = ["Final SegID", "best_after_segment", "best_original_id"]

    for col in id_cols_stage1:
        if col in stage1.columns:
            stage1[col] = pd.to_numeric(stage1[col], errors="coerce").astype("Int64")

    for col in id_cols_stage2:
        if col in stage2.columns:
            stage2[col] = pd.to_numeric(stage2[col], errors="coerce").astype("Int64")

    keep_stage1 = [
        "Final SegID",
        "best_after_segment",
        "precision",
        "recall",
        "iou",
    ]
    keep_stage1 = [col for col in keep_stage1 if col in stage1.columns]

    # Merge stage-1 context onto stage-2 so each row has final/after/original IDs.
    merged = stage2.merge(
        stage1[keep_stage1],
        on=["Final SegID", "best_after_segment"],
        how="left",
        suffixes=("_stage2", "_stage1"),
    )

    return merged


def pick_examples(
    df: pd.DataFrame,
    metric_col: str,
    high_q: float,
    low_q: float,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top-N high and low examples for a metric."""
    metric = pd.to_numeric(df[metric_col], errors="coerce")
    valid = df[metric.notna()].copy()
    valid[metric_col] = metric[metric.notna()]

    if valid.empty:
        return valid, valid

    high_cutoff = valid[metric_col].quantile(high_q)
    low_cutoff = valid[metric_col].quantile(low_q)

    high = valid[valid[metric_col] >= high_cutoff].sort_values(
        metric_col, ascending=False
    )
    low = valid[valid[metric_col] <= low_cutoff].sort_values(metric_col, ascending=True)

    return high.head(n), low.head(n)


def print_examples(title: str, examples: pd.DataFrame, metric_col: str) -> None:
    """Print examples with the three IDs and stage metrics."""
    print(f"\n{title}")
    if examples.empty:
        print("No rows matched this category.")
        return

    display_cols = [
        "Final SegID",
        "best_after_segment",
        "best_original_id",
        metric_col,
        "after_to_original_precision",
        "after_to_original_recall",
        "precision",
        "recall",
    ]
    display_cols = [col for col in display_cols if col in examples.columns]

    shown = examples[display_cols].copy()
    print(shown.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print example Final/Pass1/Original seg IDs for high/low precision and recall."
        )
    )
    parser.add_argument(
        "--stage1-csv",
        type=Path,
        default=Path("data/ewii_20260414_latest_after_segment_metrics.csv"),
        help="CSV from stage 1 (Final vs Pass 1)",
    )
    parser.add_argument(
        "--stage2-csv",
        type=Path,
        default=Path("data/ewii_20260414_latest_after_to_original_metrics.csv"),
        help="CSV from stage 2 (Pass 1 vs Original)",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=8,
        help="Number of rows to print for each category",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.9,
        help="Quantile threshold for high metrics",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.1,
        help="Quantile threshold for low metrics",
    )

    args = parser.parse_args()

    merged = load_metrics(args.stage1_csv, args.stage2_csv)

    required_cols = [
        "Final SegID",
        "best_after_segment",
        "best_original_id",
        "after_to_original_precision",
        "after_to_original_recall",
    ]
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    merged = merged.dropna(
        subset=[
            "Final SegID",
            "best_after_segment",
            "best_original_id",
            "after_to_original_precision",
            "after_to_original_recall",
        ]
    ).copy()

    print(f"Rows available with Final/Pass1/Original IDs: {len(merged):,}")

    high_recall, low_recall = pick_examples(
        merged,
        metric_col="after_to_original_recall",
        high_q=args.high_quantile,
        low_q=args.low_quantile,
        n=args.examples_per_category,
    )
    high_precision, low_precision = pick_examples(
        merged,
        metric_col="after_to_original_precision",
        high_q=args.high_quantile,
        low_q=args.low_quantile,
        n=args.examples_per_category,
    )

    print_examples("High recall examples", high_recall, "after_to_original_recall")
    print_examples("Low recall examples", low_recall, "after_to_original_recall")
    print_examples(
        "High precision examples", high_precision, "after_to_original_precision"
    )
    print_examples(
        "Low precision examples", low_precision, "after_to_original_precision"
    )


if __name__ == "__main__":
    main()
