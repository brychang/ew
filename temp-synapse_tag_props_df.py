from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROPS_TO_GUESS = {
    (1, 0, 0): "is_synapse",
    (0, 1, 0): "not_synapse",
    (0, 0, 1): "unsure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract tagged synapse annotations from synapse_tag_state.json and join "
            "distance_to_skeleton_surface_nm from the ribbon CSV."
        )
    )
    parser.add_argument(
        "--state-json",
        type=Path,
        default=Path("data/synapse_tag_state.json"),
        help="Path to synapse_tag_state.json.",
    )
    parser.add_argument(
        "--ribbon-csv",
        type=Path,
        default=Path(
            "data/ribbon_seg_id_runs/run_20260408_185507/ribbon_with_cell_labels.csv"
        ),
        help="Path to ribbon_with_cell_labels.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synapse_tag_props_df.csv"),
        help="Output CSV path for the extracted dataframe.",
    )
    return parser.parse_args()


def load_tagged_annotations(state_json: Path) -> pd.DataFrame:
    state = json.loads(state_json.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for layer in state.get("layers", []):
        for annotation in layer.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            if annotation.get("type") != "point":
                continue

            props = tuple(annotation.get("props", []))
            bryan_manual_guess = PROPS_TO_GUESS.get(props)
            if bryan_manual_guess is None:
                continue

            point = annotation.get("point", [])
            if len(point) != 3:
                continue

            rows.append(
                {
                    "id": int(annotation["id"]),
                    "point_x": int(point[0]),
                    "point_y": int(point[1]),
                    "point_z": int(point[2]),
                    "bryan_manual_guess": bryan_manual_guess,
                }
            )

    if not rows:
        raise ValueError("No tagged point annotations with recognized props were found")

    return pd.DataFrame(rows)


def load_distance_lookup(ribbon_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        ribbon_csv,
        usecols=[
            "source_row",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "distance_to_skeleton_surface_nm",
        ],
    )
    return df.rename(
        columns={
            "source_row": "id",
            "centroid_x": "point_x",
            "centroid_y": "point_y",
            "centroid_z": "point_z",
        }
    )


def build_dataframe(state_json: Path, ribbon_csv: Path) -> pd.DataFrame:
    tags_df = load_tagged_annotations(state_json)
    distance_df = load_distance_lookup(ribbon_csv)

    merged = tags_df.merge(
        distance_df[["id", "distance_to_skeleton_surface_nm"]],
        on="id",
        how="left",
    )

    missing_mask = merged["distance_to_skeleton_surface_nm"].isna()
    if missing_mask.any():
        fallback = tags_df.loc[
            missing_mask, ["id", "point_x", "point_y", "point_z"]
        ].merge(distance_df, on=["point_x", "point_y", "point_z"], how="left")
        merged.loc[missing_mask, "distance_to_skeleton_surface_nm"] = fallback[
            "distance_to_skeleton_surface_nm"
        ].to_numpy()

    return merged[["id", "bryan_manual_guess", "distance_to_skeleton_surface_nm"]].copy()


def main() -> None:
    args = parse_args()

    if not args.state_json.exists():
        raise FileNotFoundError(f"State JSON not found: {args.state_json}")
    if not args.ribbon_csv.exists():
        raise FileNotFoundError(f"Ribbon CSV not found: {args.ribbon_csv}")

    df = build_dataframe(args.state_json, args.ribbon_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()