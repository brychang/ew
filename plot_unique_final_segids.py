from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def build_allowed_final_segids(
    sheet_path: Path,
    mapping_path: Path,
    parallel_results_path: Path,
) -> set[int]:
    """Return Final SegIDs where logs only contain the mapped proofreader user_name."""
    sheet = pd.read_csv(
        sheet_path,
        low_memory=False,
        dtype={
            "Final SegID": "string",
            "611 Proofreader": "string",
            "Proofreader 2": "string",
        },
    )
    mapping = pd.read_csv(
        mapping_path,
        low_memory=False,
        dtype="string",
        keep_default_na=False,
    )
    parallel_results = pd.read_csv(
        parallel_results_path,
        low_memory=False,
        dtype={"Final SegID": "string", "user_name": "string"},
    )

    for col in ["611 Proofreader", "Proofreader 2"]:
        sheet[col] = sheet[col].fillna("").astype(str).str.strip()

    sheet = sheet[
        (sheet["611 Proofreader"] != "")
        & (sheet["Proofreader 2"] != "")
        & sheet["Final SegID"].notna()
    ].copy()
    sheet["Final SegID"] = pd.to_numeric(sheet["Final SegID"], errors="coerce").astype(
        "Int64"
    )
    sheet = sheet.dropna(subset=["Final SegID"])
    sheet = sheet.drop_duplicates(subset=["Final SegID"], keep="first")

    mapping["611 Proofreader"] = (
        mapping["611 Proofreader"].fillna("").astype(str).str.strip()
    )
    mapping["matching_user_name"] = (
        mapping["matching_user_name"].fillna("").astype(str).str.strip()
    )
    mapping = mapping[
        (mapping["611 Proofreader"] != "") & (mapping["matching_user_name"] != "")
    ]

    proofreader_to_user = dict(
        zip(mapping["611 Proofreader"], mapping["matching_user_name"])
    )
    sheet["expected_user_name"] = (
        sheet["611 Proofreader"]
        .map(proofreader_to_user)
        .fillna("")
        .astype(str)
        .str.strip()
    )

    expected = sheet[sheet["expected_user_name"] != ""][
        ["Final SegID", "expected_user_name"]
    ].copy()

    parallel_results["Final SegID"] = pd.to_numeric(
        parallel_results["Final SegID"], errors="coerce"
    ).astype("Int64")
    parallel_results["user_name"] = (
        parallel_results["user_name"].fillna("").astype(str).str.strip()
    )
    parallel_results = parallel_results.dropna(subset=["Final SegID"])

    observed_user_sets = (
        parallel_results.groupby("Final SegID")["user_name"]
        .apply(lambda series: {name for name in series if name != ""})
        .rename("observed_users")
        .reset_index()
    )

    expected_vs_observed = expected.merge(
        observed_user_sets,
        on="Final SegID",
        how="left",
    )
    expected_vs_observed["observed_users"] = expected_vs_observed[
        "observed_users"
    ].apply(lambda value: value if isinstance(value, set) else set())

    expected_vs_observed["only_expected"] = expected_vs_observed.apply(
        lambda row: (
            (len(row["observed_users"]) > 0)
            and (row["observed_users"] == {row["expected_user_name"]})
        ),
        axis=1,
    )

    return set(
        expected_vs_observed.loc[expected_vs_observed["only_expected"], "Final SegID"]
        .astype("int64")
        .tolist()
    )


def build_overlay_points(
    scored_table: pd.DataFrame,
    original_scored_table: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    overlay_points = pd.concat(
        [
            pd.DataFrame(
                {
                    "precision": scored_table["precision"],
                    "recall": scored_table["recall"],
                    "stage": "Final vs Pass 1",
                }
            ),
            pd.DataFrame(
                {
                    "precision": original_scored_table["after_to_original_precision"],
                    "recall": original_scored_table["after_to_original_recall"],
                    "stage": "Pass 1 vs Automated",
                }
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["precision", "recall"])

    if random_state is None:
        return overlay_points.sample(frac=1).reset_index(drop=True)
    return overlay_points.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )


def main() -> None:
    scored_path = Path("data/ewii_20260414_latest_after_segment_metrics.csv")
    original_scored_path = Path(
        "data/ewii_20260414_latest_after_to_original_metrics.csv"
    )

    scored_table = pd.read_csv(scored_path)
    original_scored_table = pd.read_csv(original_scored_path)

    stage_colors = {
        "Final vs Pass 1": "#1f77b4",
        "Pass 1 vs Automated": "#ff7f0e",
    }

    # Scatter: final vs best-after.
    plt.figure(figsize=(8, 6))
    plt.scatter(scored_table["precision"], scored_table["recall"], alpha=0.7)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(
        "Precision vs Recall [Final Segments compared to Best Segment after First Proofreading Pass]"
    )
    plt.grid(True)
    plt.show()

    # Scatter: best-after vs best-original.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        original_scored_table["after_to_original_precision"],
        original_scored_table["after_to_original_recall"],
        alpha=0.7,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(
        "Precision vs Recall [Best Segment after First Proofreading Pass compared to Best Original Segment]"
    )
    plt.grid(True)
    plt.show()

    # Overlay of both stages.
    overlay_points = build_overlay_points(scored_table, original_scored_table)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        overlay_points["precision"],
        overlay_points["recall"],
        c=overlay_points["stage"].map(stage_colors),
        alpha=0.7,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision vs Recall for Both Scoring Stages")
    plt.grid(True)
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Final vs Pass 1",
                markerfacecolor=stage_colors["Final vs Pass 1"],
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Pass 1 vs Automated",
                markerfacecolor=stage_colors["Pass 1 vs Automated"],
                markersize=8,
            ),
        ]
    )
    plt.show()

    # Histogram comparisons for precision and recall.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(
        scored_table["precision"].dropna(),
        bins=20,
        alpha=0.7,
        label="Final vs Pass 1",
    )
    plt.hist(
        original_scored_table["after_to_original_precision"].dropna(),
        bins=20,
        alpha=0.7,
        label="Pass 1 vs Automated",
    )
    plt.xlabel("Precision")
    plt.ylabel("Count")
    plt.title("Precision Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(
        scored_table["recall"].dropna(),
        bins=20,
        alpha=0.7,
        label="Final vs Pass 1",
    )
    plt.hist(
        original_scored_table["after_to_original_recall"].dropna(),
        bins=20,
        alpha=0.7,
        label="Pass 1 vs Automated",
    )
    plt.xlabel("Recall")
    plt.ylabel("Count")
    plt.title("Recall Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Overlay with strict expected-proofreader filtering.
    allowed_final_segids = build_allowed_final_segids(
        sheet_path=Path("data/ewii_20260414.csv"),
        mapping_path=Path(
            "data/ewii_20260414_proofreader_to_user_name_prefix_attempts.csv"
        ),
        parallel_results_path=Path("data/ewii_20260414_latest_per_user_parallel.csv"),
    )
    print(
        "Allowed Final SegIDs (only expected proofreader in log): "
        f"{len(allowed_final_segids):,}"
    )

    for col in ["Final SegID", "best_after_segment"]:
        if col in scored_table.columns:
            scored_table[col] = pd.to_numeric(
                scored_table[col], errors="coerce"
            ).astype("Int64")

    for col in ["Final SegID", "best_after_segment", "best_original_id"]:
        if col in original_scored_table.columns:
            original_scored_table[col] = pd.to_numeric(
                original_scored_table[col], errors="coerce"
            ).astype("Int64")

    scored_table_filtered = scored_table[
        scored_table["Final SegID"].isin(allowed_final_segids)
    ].copy()
    original_scored_table_filtered = original_scored_table[
        original_scored_table["Final SegID"].isin(allowed_final_segids)
    ].copy()

    filtered_overlay_points = build_overlay_points(
        scored_table_filtered,
        original_scored_table_filtered,
        random_state=0,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        filtered_overlay_points["precision"],
        filtered_overlay_points["recall"],
        c=filtered_overlay_points["stage"].map(stage_colors),
        alpha=0.7,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision vs Recall (Only Expected Proofreaders in Change Log)")
    plt.grid(True)
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Final vs Pass 1",
                markerfacecolor=stage_colors["Final vs Pass 1"],
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Pass 1 vs Automated",
                markerfacecolor=stage_colors["Pass 1 vs Automated"],
                markersize=8,
            ),
        ]
    )
    plt.show()


if __name__ == "__main__":
    main()
