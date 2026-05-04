from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CATEGORY_ORDER = ["is_synapse", "not_synapse", "unsure"]
CATEGORY_COLORS = {
    "is_synapse": "#1f77b4",
    "not_synapse": "#d62728",
    "unsure": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram of distance_to_skeleton_surface_nm grouped by "
            "bryan_manual_guess."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synapse_tag_props_df.csv"),
        help="Input CSV with bryan_manual_guess and distance_to_skeleton_surface_nm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synapse_tag_distance_hist_by_guess.png"),
        help="Path to save the histogram image.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins to use.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving it.",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_columns = {
        "bryan_manual_guess",
        "distance_to_skeleton_surface_nm",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    df = df.dropna(subset=["distance_to_skeleton_surface_nm"])
    df["bryan_manual_guess"] = pd.Categorical(
        df["bryan_manual_guess"], categories=CATEGORY_ORDER, ordered=True
    )
    return df


def plot_histogram(df: pd.DataFrame, bins: int, output_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = df["distance_to_skeleton_surface_nm"]
    if distances.empty:
        raise ValueError("No distance values available to plot.")

    bin_edges = pd.cut(distances, bins=bins, retbins=True)[1]
    plotted_any = False

    for category in CATEGORY_ORDER:
        category_df = df[df["bryan_manual_guess"] == category]
        if category_df.empty:
            continue

        ax.hist(
            category_df["distance_to_skeleton_surface_nm"],
            bins=bin_edges,
            alpha=0.55,
            label=f"{category} (n={len(category_df)})",
            color=CATEGORY_COLORS.get(category),
            edgecolor="white",
            linewidth=0.5,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No recognized bryan_manual_guess categories were found.")

    ax.set_title("Distance to Skeleton Surface by Manual Guess")
    ax.set_xlabel("Distance to skeleton surface (nm)")
    ax.set_ylabel("Count")
    ax.legend(title="bryan_manual_guess")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved histogram to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = load_data(args.input)
    print(df.groupby("bryan_manual_guess", observed=False).size().to_string())
    plot_histogram(df, args.bins, args.output, args.show)


if __name__ == "__main__":
    main()
