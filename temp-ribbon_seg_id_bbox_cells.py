from __future__ import annotations

import argparse
import math
import os
import re
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skeliner as sk
from cloudvolume import CloudVolume

DEFAULT_CV_PATH = (
    "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/"
    "stroeh_mouse_retina"
)
REQUIRED_COLUMNS = ("centroid_x", "centroid_y", "centroid_z")
DEFAULT_VOXEL_SIZE = (16.0, 16.0, 40.0)
_VOL: CloudVolume | None = None


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    rows: list[tuple[int, int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel centroid-to-segmentation lookup for ribbon synapses using "
            "a list of target cell IDs."
        )
    )
    parser.add_argument(
        "--cells-file",
        type=Path,
        default=Path("data/off_sac.txt"),
        help="Text file containing target cell seg IDs (comma/newline/space separated).",
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
        "--mesh-dir",
        type=Path,
        default=Path("data/meshes"),
        help="Directory containing {seg_id}.obj meshes (downloaded if missing).",
    )
    parser.add_argument(
        "--max-distance-nm",
        type=float,
        default=500.0,
        help="Only keep ribbons closer than this distance to a cell skeleton.",
    )
    parser.add_argument(
        "--bbox-extra-buffer-nm",
        type=float,
        default=2000.0,
        help=(
            "Extra safety margin added to bbox expansion in nm. "
            "Effective bbox margin is max-distance-nm + bbox-extra-buffer-nm."
        ),
    )
    parser.add_argument(
        "--distance-mode",
        choices=("centerline", "surface"),
        default="surface",
        help="Distance metric for skeleton.distance().",
    )
    parser.add_argument(
        "--size-min",
        type=float,
        default=100.0,
        help="Minimum ribbon size filter.",
    )
    parser.add_argument(
        "--size-max",
        type=float,
        default=1000.0,
        help="Maximum ribbon size filter.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=DEFAULT_VOXEL_SIZE,
        help="Voxel size in nm for x y z used to convert centroids before distance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of process workers for seg-id lookups.",
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
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on number of cell IDs loaded from --cells-file.",
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
    if args.max_cells is not None and args.max_cells < 1:
        raise ValueError("--max-cells must be >= 1 when provided")
    if args.max_distance_nm <= 0:
        raise ValueError("--max-distance-nm must be > 0")
    if args.bbox_extra_buffer_nm < 0:
        raise ValueError("--bbox-extra-buffer-nm must be >= 0")
    if args.size_min >= args.size_max:
        raise ValueError("--size-min must be smaller than --size-max")
    if any(v <= 0 for v in args.voxel_size):
        raise ValueError("--voxel-size values must all be > 0")


def load_cell_ids(path: Path, max_cells: int | None) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Cells file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    parts = re.split(r"[\s,]+", text.strip())

    seen: set[int] = set()
    ordered_ids: list[int] = []
    for part in parts:
        if not part:
            continue
        value = int(part)
        if value in seen:
            continue
        seen.add(value)
        ordered_ids.append(value)

    if max_cells is not None:
        ordered_ids = ordered_ids[:max_cells]

    if not ordered_ids:
        raise ValueError(f"No valid cell seg IDs found in {path}")
    return ordered_ids


def load_input_table(
    path: Path,
    limit: int | None,
    size_min: float,
    size_max: float,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path, header=0, index_col=0)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "size" not in df.columns:
        raise ValueError("Input table must contain a 'size' column for size filtering")

    df = df[(df["size"] > size_min) & (df["size"] < size_max)]

    if limit is not None:
        df = df.head(limit)

    result = df.reset_index(names="ribbon_id")[
        ["ribbon_id", "centroid_x", "centroid_y", "centroid_z", "size"]
    ].copy()
    for col in REQUIRED_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="raise").astype(np.int64)

    result.insert(0, "source_row", np.arange(len(result), dtype=np.int64))
    return result


def resolve_mesh_path(
    mesh_dir: Path,
    skeleton_seg_id: int,
    cv_path: str,
    use_https: bool,
) -> Path:
    mesh_file = mesh_dir / f"{skeleton_seg_id}.obj"
    if mesh_file.exists():
        return mesh_file

    log(f"Mesh not found at {mesh_file}, downloading")
    mesh_dir.mkdir(parents=True, exist_ok=True)

    cv = CloudVolume(cv_path, use_https=use_https)
    meshes = cv.mesh.get([skeleton_seg_id], fuse=False)
    mesh = meshes[skeleton_seg_id]
    obj_data = mesh.to_obj()
    if isinstance(obj_data, str):
        mesh_file.write_text(obj_data, encoding="utf-8")
    else:
        mesh_file.write_bytes(obj_data)
    return mesh_file


def build_skeleton(mesh_path: Path, skeleton_seg_id: int):
    mesh = sk.io.load_mesh(str(mesh_path))
    return sk.skeletonize(
        mesh,
        detect_soma=True,
        collapse_soma=True,
        bridge_gaps=True,
        prune_tiny_neurites=True,
        unit="nm",
        id=skeleton_seg_id,
        verbose=False,
    )


def _as_xyz_array(data: object) -> np.ndarray | None:
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float64, copy=False)

    if arr.ndim == 1 and len(arr) > 0 and hasattr(arr[0], "__dict__"):
        rows: list[tuple[float, float, float]] = []
        for item in arr:
            if hasattr(item, "xyz"):
                xyz = np.asarray(getattr(item, "xyz"), dtype=np.float64)
                if xyz.shape[0] >= 3:
                    rows.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                    continue
            if all(hasattr(item, axis) for axis in ("x", "y", "z")):
                rows.append((float(item.x), float(item.y), float(item.z)))
        if rows:
            return np.asarray(rows, dtype=np.float64)

    return None


def get_skeleton_points_nm(skeleton) -> np.ndarray:
    for attr in ("vertices", "points", "coords", "coordinates", "nodes"):
        if not hasattr(skeleton, attr):
            continue
        coords = _as_xyz_array(getattr(skeleton, attr))
        if coords is not None and len(coords) > 0:
            return coords

    raise RuntimeError(
        "Could not extract skeleton coordinates for bbox prefilter. "
        "Expected one of attributes: vertices, points, coords, coordinates, nodes."
    )


def annotate_near_ribbons_for_cell(
    input_df: pd.DataFrame,
    skeleton,
    voxel_size_nm: tuple[float, float, float],
    max_distance_nm: float,
    bbox_extra_buffer_nm: float,
    distance_mode: str,
) -> pd.DataFrame:
    voxel = np.asarray(voxel_size_nm, dtype=np.float64)
    coords_nm = (
        input_df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=np.float64)
        * voxel
    )

    skeleton_nm = get_skeleton_points_nm(skeleton)
    bbox_margin_nm = float(max_distance_nm) + float(bbox_extra_buffer_nm)
    mins = skeleton_nm.min(axis=0) - bbox_margin_nm
    maxs = skeleton_nm.max(axis=0) + bbox_margin_nm

    in_bbox_mask = np.logical_and(coords_nm >= mins, coords_nm <= maxs).all(axis=1)
    bbox_df = input_df.loc[in_bbox_mask].copy()
    if bbox_df.empty:
        return bbox_df

    soma = getattr(skeleton, "soma", None)
    if soma is None or not hasattr(soma, "distance"):
        raise RuntimeError(
            "Skeleton soma object with a distance() method is required to annotate "
            "distance_to_sac_soma_nm"
        )

    bbox_coords = bbox_df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(
        dtype=np.float64
    )
    centerline_distances = np.empty(len(bbox_df), dtype=np.float64)
    surface_distances = np.empty(len(bbox_df), dtype=np.float64)
    soma_distances = np.empty(len(bbox_df), dtype=np.float64)

    for i, point in enumerate(bbox_coords):
        point_nm = point * voxel
        centerline_distances[i] = float(skeleton.distance(point_nm, mode="centerline"))
        surface_distances[i] = float(skeleton.distance(point_nm, mode="surface"))
        soma_distances[i] = float(soma.distance(point_nm))

    bbox_df["distance_to_skeleton_centerline_nm"] = centerline_distances
    bbox_df["distance_to_skeleton_surface_nm"] = surface_distances
    bbox_df["distance_to_skeleton_nm"] = (
        centerline_distances if distance_mode == "centerline" else surface_distances
    )
    bbox_df["distance_to_sac_soma_nm"] = soma_distances

    near_df = bbox_df[bbox_df["distance_to_skeleton_nm"] < max_distance_nm].copy()
    return near_df


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

    return merged


def prepare_run_dirs(
    base_output_dir: Path,
    output_format: str,
) -> tuple[Path, Path, Path, Path]:
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_output_dir / run_name
    chunk_dir = run_dir / "chunks"
    run_dir.mkdir(parents=True, exist_ok=False)
    chunk_dir.mkdir(parents=True, exist_ok=False)
    ext = "csv" if output_format == "csv" else "parquet"
    final_path = run_dir / f"ribbon_with_seg_ids_cells.{ext}"
    summary_path = run_dir / "cell_summary.csv"
    return run_dir, chunk_dir, final_path, summary_path


def main() -> None:
    started = time.time()
    args = parse_args()
    ensure_valid_args(args)

    cell_ids = load_cell_ids(args.cells_file, args.max_cells)
    log(f"Loaded {len(cell_ids)} cell IDs from {args.cells_file}")

    input_df = load_input_table(args.input, args.limit, args.size_min, args.size_max)
    if input_df.empty:
        raise ValueError("No rows found in input table after filtering/limit")
    log(f"Rows after size filter: {len(input_df)}")

    near_frames: list[pd.DataFrame] = []
    for idx, cell_id in enumerate(cell_ids, start=1):
        log(f"[{idx}/{len(cell_ids)}] Processing cell {cell_id}")
        mesh_path = resolve_mesh_path(
            args.mesh_dir, cell_id, args.cv_path, args.use_https
        )
        skeleton = build_skeleton(mesh_path, cell_id)
        near_df = annotate_near_ribbons_for_cell(
            input_df=input_df,
            skeleton=skeleton,
            voxel_size_nm=tuple(float(v) for v in args.voxel_size),
            max_distance_nm=float(args.max_distance_nm),
            bbox_extra_buffer_nm=float(args.bbox_extra_buffer_nm),
            distance_mode=args.distance_mode,
        )
        if near_df.empty:
            log(f"Cell {cell_id}: 0 rows within distance threshold")
            continue
        near_df.insert(1, "target_cell_id", int(cell_id))
        near_frames.append(near_df)
        log(f"Cell {cell_id}: kept {len(near_df)} rows")

    if not near_frames:
        raise ValueError("No rows remained after processing all cell IDs")

    all_near_df = pd.concat(near_frames, ignore_index=True)
    all_near_df.drop_duplicates(subset=["target_cell_id", "source_row"], inplace=True)
    all_near_df.sort_values(
        ["target_cell_id", "source_row"], inplace=True, kind="mergesort"
    )
    all_near_df.reset_index(drop=True, inplace=True)

    # Lookup seg IDs once per unique ribbon row, then join back to all cell matches.
    lookup_df = all_near_df[
        ["source_row", "ribbon_id", "centroid_x", "centroid_y", "centroid_z"]
    ].drop_duplicates(subset=["source_row"])

    task_count = math.ceil(len(lookup_df) / args.chunk_size)
    log(
        f"Prepared {len(all_near_df)} cell-ribbon matches and {len(lookup_df)} unique "
        f"lookup rows in {task_count} chunks"
    )
    tasks = build_tasks(lookup_df, args.chunk_size)

    run_dir, chunk_dir, final_path, summary_path = prepare_run_dirs(
        args.output_dir, args.format
    )
    log(f"Run directory: {run_dir}")

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
        total_input_rows=len(lookup_df),
        output_format=args.format,
    )

    final_df = all_near_df.merge(
        merged[["source_row", "segmentation_id"]],
        on="source_row",
        how="left",
        validate="many_to_one",
    )
    if final_df["segmentation_id"].isna().any():
        raise RuntimeError("Merge produced missing segmentation_id values")

    final_df.sort_values(
        ["target_cell_id", "source_row"], inplace=True, kind="mergesort"
    )
    final_df.reset_index(drop=True, inplace=True)
    write_table(final_df, final_path, args.format)

    summary = (
        final_df.groupby("target_cell_id", as_index=False)
        .agg(
            ribbon_matches=("source_row", "size"),
            unique_ribbons=("ribbon_id", "nunique"),
            unique_supervoxels=("segmentation_id", "nunique"),
            min_distance_nm=("distance_to_skeleton_nm", "min"),
            median_distance_nm=("distance_to_skeleton_nm", "median"),
            max_distance_nm=("distance_to_skeleton_nm", "max"),
        )
        .sort_values("target_cell_id", kind="mergesort")
        .reset_index(drop=True)
    )
    summary.to_csv(summary_path, index=False)

    elapsed = time.time() - started
    log(
        f"Finished in {elapsed:.1f} seconds; final rows: {len(final_df)}; "
        f"rate: {len(lookup_df) / max(1e-6, elapsed):.1f} lookup rows/s"
    )
    log(f"Final output: {final_path}")
    log(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
