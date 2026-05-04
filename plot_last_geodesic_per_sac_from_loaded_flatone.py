#!/usr/bin/env python3
"""Create the per-SAC geodesic plot from ribbon_with_cell_labels.csv.

This reproduces the last plot pattern in temp-simone_off_sac_ewi_4d.py, but assumes
flatone skeletons are already loaded into memory.

Expected flatone context format:
    flatone_context_by_target[target_cell_id] = {
        "tree_2d": cKDTree over skeleton xy coordinates (in um),
        "dist_to_root_um": dict[node_index, geodesic_distance_um_from_soma_root],
    }
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

COLORS = {
    "t1": "#188a53",
    "t2": "#e7bd13",
    "t3a": "#1aade8",
    "t3b": "#045684",
    "t4": "#951f92",
    "t5t": "#e41a1c",
    "GluMI": "#ff7f00",
    "RBC": "#6a3d9a",
    "XBC": "#a65628",
    "t5i": "#fdb462",
    "t5o": "#b2df8a",
    "t6": "#33a02c",
    "t7": "#fb9a99",
    "t8": "#1f78b4",
    "t9": "#ff7f00",
}


def build_flatone_context_from_xyz_parents(
    xyz_um: np.ndarray,
    parents: np.ndarray,
    ids: np.ndarray | None = None,
) -> dict:
    """Build one context dict from a flatone skeleton representation.

    Parameters
    ----------
    xyz_um
        Skeleton node coordinates in um, shape (N, 3).
    parents
        Parent node IDs (SWC parent column). Root should be -1.
    ids
        Optional SWC node IDs. If None, assumes ids are 0..N-1 and that parents
        already refer to those indices.
    """
    xyz_um = np.asarray(xyz_um)
    parents = np.asarray(parents)

    if ids is None:
        ids = np.arange(len(xyz_um), dtype=int)
    else:
        ids = np.asarray(ids)

    id_to_idx = {int(node_id): i for i, node_id in enumerate(ids)}
    root_candidates = np.where(parents == -1)[0]
    if len(root_candidates) == 0:
        raise ValueError("No root node found (parent == -1).")
    root_idx = int(root_candidates[0])

    edges: list[tuple[int, int]] = []
    for i, parent_id in enumerate(parents):
        if parent_id == -1:
            continue
        parent_idx = id_to_idx.get(int(parent_id))
        if parent_idx is None:
            continue
        edges.append((i, parent_idx))

    graph = nx.Graph()
    for i, j in edges:
        weight = float(np.linalg.norm(xyz_um[i] - xyz_um[j]))
        graph.add_edge(i, j, weight=weight)

    dist_to_root_um = nx.single_source_dijkstra_path_length(
        graph, root_idx, weight="weight"
    )
    tree_2d = cKDTree(xyz_um[:, :2])

    return {
        "tree_2d": tree_2d,
        "dist_to_root_um": dist_to_root_um,
    }


def compute_geodesic_distance_um_ignore_z(
    point_um: np.ndarray,
    target_cell_id: int,
    flatone_context_by_target: dict[int, dict],
) -> float:
    """Compute 2D geodesic distance using a preloaded flatone context."""
    context = flatone_context_by_target.get(int(target_cell_id))
    if context is None:
        return np.nan

    tree_2d = context["tree_2d"]
    dist_to_root_um = context["dist_to_root_um"]
    _, node_idx = tree_2d.query(np.asarray(point_um)[:2])
    return float(dist_to_root_um.get(int(node_idx), np.nan))


def plot_per_target_cell_id_geodesic_counts(
    csv_path: str | Path,
    flatone_context_by_target: dict[int, dict],
    spacing_nm_xyz: Iterable[float] = (16.0, 16.0, 40.0),
    distance_to_skeleton_threshold_nm: float = 200.0,
    output_dir: str | Path = "data/ribbon_off_sac_dists",
    show_figures: bool = False,
) -> None:
    """Make one saved geodesic-distance count plot per SAC plus a summary plot."""
    spacing_nm_xyz = np.asarray(tuple(spacing_nm_xyz), dtype=float)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["match_found"]]
    df = df[df["distance_to_skeleton_nm"] < distance_to_skeleton_threshold_nm]

    df["centroid_parsed_nm"] = df.apply(
        lambda row: (
            np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
            * spacing_nm_xyz
        ),
        axis=1,
    )
    df["centroid_parsed_um"] = df["centroid_parsed_nm"].apply(
        lambda point: point / 1000.0
    )

    processed_sac_groups: list[pd.DataFrame] = []

    for target_cell_id, sac_group in df.groupby("target_cell_id"):
        target_cell_id = int(target_cell_id)
        if target_cell_id not in flatone_context_by_target:
            print(f"Skipping target_cell_id={target_cell_id}: context not loaded")
            continue

        sac_group = sac_group.copy()
        sac_group["distance_to_soma_geodesic_ignore_z_um"] = sac_group.apply(
            lambda row: compute_geodesic_distance_um_ignore_z(
                row["centroid_parsed_um"], target_cell_id, flatone_context_by_target
            ),
            axis=1,
        )

        max_dist = sac_group["distance_to_soma_geodesic_ignore_z_um"].max()
        if pd.isna(max_dist):
            print(f"Skipping target_cell_id={target_cell_id}: no geodesic distances")
            continue

        processed_sac_groups.append(sac_group)

        bins = np.arange(0, max_dist + 10, 10)
        if len(bins) < 2:
            bins = np.array([0.0, 10.0])
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        plt.figure(figsize=(10, 6))
        for cell_type, cell_group in sac_group.groupby("Cell Type (machine)"):
            counts, _ = np.histogram(
                cell_group["distance_to_soma_geodesic_ignore_z_um"], bins=bins
            )
            plt.plot(
                bin_centers,
                counts,
                marker="o",
                label=cell_type,
                color=COLORS.get(cell_type),
            )

        plt.xlabel("2D geodesic distance from SAC soma (um)")
        plt.ylabel("Ribbon synapse counts [skeliner distance < 200 nm]")
        plt.title(f"target_cell_id={target_cell_id}")
        plt.legend()
        per_sac_path = output_dir / f"sac_{target_cell_id}.png"
        plt.savefig(per_sac_path, dpi=200, bbox_inches="tight")
        if show_figures:
            plt.show()
        plt.close()

    if not processed_sac_groups:
        print("No per-SAC groups were processed. Summary figure not created.")
        return

    combined = pd.concat(processed_sac_groups, ignore_index=True)
    global_max_dist = combined["distance_to_soma_geodesic_ignore_z_um"].max()
    global_bins = np.arange(0, global_max_dist + 10, 10)
    if len(global_bins) < 2:
        global_bins = np.array([0.0, 10.0])
    global_bin_centers = (global_bins[:-1] + global_bins[1:]) / 2.0

    cell_types_in_data = sorted(combined["Cell Type (machine)"].dropna().unique())

    plt.figure(figsize=(10, 6))
    for cell_type in cell_types_in_data:
        per_sac_counts: list[np.ndarray] = []
        for sac_group in processed_sac_groups:
            cell_group = sac_group[sac_group["Cell Type (machine)"] == cell_type]
            counts, _ = np.histogram(
                cell_group["distance_to_soma_geodesic_ignore_z_um"], bins=global_bins
            )
            per_sac_counts.append(counts.astype(float))

        counts_matrix = np.vstack(per_sac_counts)
        mean_counts = counts_matrix.mean(axis=0)
        if counts_matrix.shape[0] > 1:
            sem_counts = counts_matrix.std(axis=0, ddof=1) / np.sqrt(
                counts_matrix.shape[0]
            )
        else:
            sem_counts = np.zeros_like(mean_counts)

        color = COLORS.get(cell_type)
        plt.plot(
            global_bin_centers,
            mean_counts,
            marker="o",
            label=cell_type,
            color=color,
        )
        plt.fill_between(
            global_bin_centers,
            mean_counts - sem_counts,
            mean_counts + sem_counts,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("2D geodesic distance from SAC soma (um)")
    plt.ylabel("Ribbon synapse counts (mean +/- SEM across SACs)")
    plt.title("BC ribbon counts vs geodesic distance (summary across SACs)")
    plt.legend()
    summary_path = output_dir / "summary_bc_mean_sem.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close()
    print(f"Saved per-SAC and summary figures to: {output_dir}")


def read_target_ids(ids_file: Path) -> list[int]:
    target_ids: list[int] = []
    with ids_file.open("r", encoding="utf-8") as f:
        for raw in f:
            value = raw.strip().replace("\r", "")
            if not value:
                continue
            target_ids.append(int(value))
    return sorted(set(target_ids))


def get_target_ids_from_csv(csv_path: Path, threshold_nm: float) -> list[int]:
    df = pd.read_csv(csv_path)
    df = df[df["match_found"]]
    df = df[df["distance_to_skeleton_nm"] < threshold_nm]
    return sorted(df["target_cell_id"].astype(int).unique().tolist())


def load_context_from_swc(swc_path: Path) -> dict:
    data = np.loadtxt(swc_path, comments="#")
    data = np.atleast_2d(data)
    ids = data[:, 0].astype(int)
    xyz = data[:, 2:5]
    parents = data[:, 6].astype(int)
    return build_flatone_context_from_xyz_parents(xyz, parents, ids=ids)


def parse_spacing_arg(spacing_text: str) -> tuple[float, float, float]:
    parts = [x.strip() for x in spacing_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--spacing-nm-xyz must have exactly 3 comma-separated values")
    spacing = tuple(float(x) for x in parts)
    return spacing  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load flatone contexts and plot per-target-cell geodesic count curves."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/ribbon_seg_id_runs/uhd/ribbon_with_cell_labels.csv"),
        help="Path to ribbon_with_cell_labels.csv",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("data/cell_ids/uhd.txt"),
        help=(
            "Optional text file with target_cell_id values to attempt loading (one per line). "
            "Use 'none' to derive IDs directly from csv filtering."
        ),
    )
    parser.add_argument(
        "--flatone-output-root",
        type=Path,
        default=Path("~/flatone/output").expanduser(),
        help="Root directory containing <target_cell_id>/skeleton_warped.swc",
    )
    parser.add_argument(
        "--distance-threshold-nm",
        type=float,
        default=200.0,
        help="Filter threshold for distance_to_skeleton_nm",
    )
    parser.add_argument(
        "--spacing-nm-xyz",
        type=str,
        default="16,16,40",
        help="Voxel spacing in nm as x,y,z",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Exit with non-zero code if any target IDs are missing skeleton_warped.swc",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ribbon_uhd_dists"),
        help="Directory where per-SAC and summary figures will be written.",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 2

    flatone_output_root = args.flatone_output_root.expanduser().resolve()
    if not flatone_output_root.exists():
        print(f"Flatone output root not found: {flatone_output_root}")
        return 2

    if str(args.ids_file).lower() == "none":
        target_ids = get_target_ids_from_csv(csv_path, args.distance_threshold_nm)
    else:
        ids_file = args.ids_file.expanduser().resolve()
        if not ids_file.exists():
            print(f"IDs file not found: {ids_file}")
            return 2
        target_ids = read_target_ids(ids_file)

    if not target_ids:
        print("No target_cell_id values found to process.")
        return 0

    flatone_context_by_target: dict[int, dict] = {}
    missing_ids: list[int] = []
    failed_ids: list[int] = []

    for target_id in target_ids:
        swc_path = flatone_output_root / str(target_id) / "skeleton_warped.swc"
        if not swc_path.exists():
            missing_ids.append(target_id)
            continue
        try:
            flatone_context_by_target[target_id] = load_context_from_swc(swc_path)
        except Exception as exc:
            print(f"Failed to load {swc_path}: {exc}")
            failed_ids.append(target_id)

    print(f"Target IDs requested: {len(target_ids)}")
    print(f"Loaded contexts: {len(flatone_context_by_target)}")
    print(f"Missing skeleton files: {len(missing_ids)}")
    print(f"Load failures: {len(failed_ids)}")

    if missing_ids:
        print("First missing IDs:", ", ".join(str(x) for x in missing_ids[:10]))
    if failed_ids:
        print("First failed IDs:", ", ".join(str(x) for x in failed_ids[:10]))

    if args.strict_missing and (missing_ids or failed_ids):
        return 1

    if not flatone_context_by_target:
        print("No contexts were loaded; nothing to plot.")
        return 1

    spacing_nm_xyz = parse_spacing_arg(args.spacing_nm_xyz)
    plot_per_target_cell_id_geodesic_counts(
        csv_path=csv_path,
        flatone_context_by_target=flatone_context_by_target,
        spacing_nm_xyz=spacing_nm_xyz,
        distance_to_skeleton_threshold_nm=args.distance_threshold_nm,
        output_dir=args.output_dir,
        show_figures=args.show_figures,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
