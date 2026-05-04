from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SPREADSHEET = Path("data/ewii_20260414.csv")
DEFAULT_PARALLEL_RESULTS = Path("data/ewii_20260414_latest_per_user_parallel.csv")
DEFAULT_OUTPUT_DIR = Path("data")


def load_unique_proofreaders_with_proofreader2(spreadsheet_path: Path) -> list[str]:
    """Return unique 611 Proofreader values where Proofreader 2 is also non-empty."""
    spreadsheet = pd.read_csv(spreadsheet_path, dtype={"Final SegID": "string"})

    spreadsheet["611 Proofreader"] = (
        spreadsheet["611 Proofreader"].fillna("").astype(str).str.strip()
    )
    spreadsheet["Proofreader 2"] = (
        spreadsheet["Proofreader 2"].fillna("").astype(str).str.strip()
    )

    filtered = spreadsheet[
        (spreadsheet["611 Proofreader"] != "") & (spreadsheet["Proofreader 2"] != "")
    ]

    unique_proofreaders = sorted(filtered["611 Proofreader"].drop_duplicates().tolist())
    return unique_proofreaders


def load_unique_user_names(parallel_results_path: Path) -> list[str]:
    """Return unique user_name entries from parallel results output."""
    results = pd.read_csv(parallel_results_path)

    if "user_name" not in results.columns:
        raise ValueError(
            f"Expected column 'user_name' in {parallel_results_path}, found: {list(results.columns)}"
        )

    unique_user_names = sorted(
        results["user_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )
    return unique_user_names


def best_prefix_match(proofreader: str, user_names: list[str]) -> str:
    """Return first user_name that contains proofreader text, else empty string."""
    """Strip capitalization and whitespace for more forgiving matching."""
    proofreader = proofreader.strip().lower()
    for user_name in user_names:
        if user_name.strip().lower().startswith(proofreader):
            return user_name
        elif proofreader in user_name.strip().lower():
            return user_name
    return ""


def write_list_csv(values: list[str], column_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({column_name: values}).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify unique 611 Proofreader values (where Proofreader 2 exists) and "
            "unique user_name values from parallel results."
        )
    )
    parser.add_argument("--spreadsheet", type=Path, default=DEFAULT_SPREADSHEET)
    parser.add_argument(
        "--parallel-results", type=Path, default=DEFAULT_PARALLEL_RESULTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    unique_proofreaders = load_unique_proofreaders_with_proofreader2(args.spreadsheet)
    unique_user_names = load_unique_user_names(args.parallel_results)

    proofreaders_out = (
        args.output_dir / "ewii_20260414_unique_611_proofreaders_with_proofreader2.csv"
    )
    user_names_out = (
        args.output_dir / "ewii_20260414_unique_user_names_from_parallel.csv"
    )
    match_attempts_out = (
        args.output_dir / "ewii_20260414_proofreader_to_user_name_prefix_attempts.csv"
    )

    write_list_csv(unique_proofreaders, "611 Proofreader", proofreaders_out)
    write_list_csv(unique_user_names, "user_name", user_names_out)

    # Helpful for manual inspection: attempt the same startswith logic from your notebook/script.
    attempts_df = pd.DataFrame({"611 Proofreader": unique_proofreaders})
    attempts_df["matching_user_name"] = attempts_df["611 Proofreader"].apply(
        lambda x: best_prefix_match(x, unique_user_names)
    )
    attempts_df["matched"] = attempts_df["matching_user_name"] != ""
    attempts_df.to_csv(match_attempts_out, index=False)

    print(f"Unique 611 Proofreaders (with Proofreader 2): {len(unique_proofreaders)}")
    print(f"Unique user_name values: {len(unique_user_names)}")
    print(f"Wrote: {proofreaders_out}")
    print(f"Wrote: {user_names_out}")
    print(f"Wrote: {match_attempts_out}")


if __name__ == "__main__":
    main()
