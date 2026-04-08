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
        "--mesh-path",
        type=Path,
        default=None,
        help=(
            "Path to the target cell mesh used to build the skeleton. "
            "If not provided, will check data/meshes/{skeleton-seg-id}.obj and "
            "download if needed."
        ),
    )
    parser.add_argument(
        "--skeleton-seg-id",
        type=int,
        default=None,
        help="Optional seg id to store in skeleton metadata.",
    )
    parser.add_argument(
        "--max-distance-nm",
        type=float,
        default=500.0,
        help="Only query seg ids for ribbons closer than this distance to skeleton.",
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
        help=(
            "Distance metric for skeleton.distance(). "
            "Use centerline to avoid surface-envelope clamping to 0."
        ),
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
    if args.max_distance_nm <= 0:
        raise ValueError("--max-distance-nm must be > 0")
    if args.bbox_extra_buffer_nm < 0:
        raise ValueError("--bbox-extra-buffer-nm must be >= 0")
    if args.size_min >= args.size_max:
        raise ValueError("--size-min must be smaller than --size-max")
    if any(v <= 0 for v in args.voxel_size):
        raise ValueError("--voxel-size values must all be > 0")
    if args.mesh_path is None and args.skeleton_seg_id is None:
        raise ValueError(
            "Either --mesh-path must be provided or --skeleton-seg-id must be set"
        )


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
        result[col] = pd.to_numeric(result[col], errors="raise")

    # CloudVolume indexing expects integer voxel coordinates.
    for col in REQUIRED_COLUMNS:
        result[col] = result[col].astype(np.int64)

    result.insert(0, "source_row", np.arange(len(result), dtype=np.int64))
    return result


def resolve_and_prepare_mesh(
    mesh_path: Path | None,
    skeleton_seg_id: int | None,
    cv_path: str,
    use_https: bool,
) -> Path:
    """Resolve mesh path and download if needed.

    If mesh_path is provided, use it.
    Otherwise, construct path from skeleton_seg_id and download if missing.
    """
    if mesh_path is not None:
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
        return mesh_path

    # Use skeleton_seg_id to resolve path
    if skeleton_seg_id is None:
        raise ValueError(
            "Cannot resolve mesh path without --mesh-path or --skeleton-seg-id"
        )

    mesh_dir = Path("data/meshes")
    mesh_file = mesh_dir / f"{skeleton_seg_id}.obj"

    if mesh_file.exists():
        log(f"Using existing mesh: {mesh_file}")
        return mesh_file

    # Download mesh
    log(f"Mesh not found at {mesh_file}, downloading...")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    cv = CloudVolume(cv_path, use_https=use_https)
    meshes = cv.mesh.get([skeleton_seg_id], fuse=False)
    mesh = meshes[skeleton_seg_id]
    obj_str = mesh.to_obj()
    with open(mesh_file, "wb") as f:
        f.write(obj_str)
    log(f"Downloaded mesh to {mesh_file}")
    return mesh_file


def build_skeleton(mesh_path: Path, skeleton_seg_id: int | None):
    log(f"Loading mesh: {mesh_path}")
    mesh = sk.io.load_mesh(str(mesh_path))
    log("Skeletonizing mesh")
    return sk.skeletonize(
        mesh,
        detect_soma=True,
        collapse_soma=True,
        bridge_gaps=True,
        prune_tiny_neurites=True,
        unit="nm",
        id=skeleton_seg_id,
        verbose=True,
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

    if isinstance(data, dict) and all(k in data for k in ("x", "y", "z")):
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        z = np.asarray(data["z"], dtype=np.float64)
        if len(x) == len(y) == len(z) and len(x) > 0:
            return np.column_stack((x, y, z))

    if all(hasattr(data, axis) for axis in ("x", "y", "z")):
        x = np.asarray(getattr(data, "x"), dtype=np.float64)
        y = np.asarray(getattr(data, "y"), dtype=np.float64)
        z = np.asarray(getattr(data, "z"), dtype=np.float64)
        if len(x) == len(y) == len(z) and len(x) > 0:
            return np.column_stack((x, y, z))

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


def prefilter_with_skeleton_bbox(
    df: pd.DataFrame,
    skeleton,
    voxel_size_nm: tuple[float, float, float],
    max_distance_nm: float,
    bbox_extra_buffer_nm: float,
) -> pd.DataFrame:
    voxel = np.asarray(voxel_size_nm, dtype=np.float64)
    coords_nm = (
        df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=np.float64)
        * voxel
    )

    skeleton_nm = get_skeleton_points_nm(skeleton)
    bbox_margin_nm = float(max_distance_nm) + float(bbox_extra_buffer_nm)
    mins = skeleton_nm.min(axis=0) - bbox_margin_nm
    maxs = skeleton_nm.max(axis=0) + bbox_margin_nm

    mask = np.logical_and(coords_nm >= mins, coords_nm <= maxs).all(axis=1)
    filtered = df.loc[mask].copy()
    log(
        "BBox prefilter kept "
        f"{len(filtered)}/{len(df)} rows "
        f"({(100.0 * len(filtered) / max(1, len(df))):.2f}%) "
        f"with margin={bbox_margin_nm:.1f} nm"
    )
    return filtered


def annotate_distance_to_skeleton(
    df: pd.DataFrame,
    skeleton,
    voxel_size_nm: tuple[float, float, float],
    distance_mode: str,
) -> pd.DataFrame:
    out = df.copy()
    voxel = np.asarray(voxel_size_nm, dtype=np.float64)
    coords = out[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=np.float64)
    soma = getattr(skeleton, "soma", None)
    if soma is None or not hasattr(soma, "distance"):
        raise RuntimeError(
            "Skeleton soma object with a distance() method is required to annotate "
            "distance_to_sac_soma_nm"
        )

    centerline_distances = np.empty(len(out), dtype=np.float64)
    surface_distances = np.empty(len(out), dtype=np.float64)
    soma_distances = np.empty(len(out), dtype=np.float64)
    for i, point in enumerate(coords, start=1):
        point_nm = point * voxel
        centerline_distances[i - 1] = float(
            skeleton.distance(point_nm, mode="centerline")
        )
        surface_distances[i - 1] = float(skeleton.distance(point_nm, mode="surface"))
        soma_distances[i - 1] = float(soma.distance(point_nm))
        if i % 5000 == 0 or i == len(out):
            log(f"Distance progress: {i}/{len(out)}")

    out["distance_to_skeleton_centerline_nm"] = centerline_distances
    out["distance_to_skeleton_surface_nm"] = surface_distances
    out["distance_to_skeleton_nm"] = (
        centerline_distances if distance_mode == "centerline" else surface_distances
    )
    out["distance_to_sac_soma_nm"] = soma_distances
    return out


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
    started = time.time()
    args = parse_args()
    ensure_valid_args(args)

    log("Loading input table")
    input_df = load_input_table(args.input, args.limit, args.size_min, args.size_max)
    total_rows = len(input_df)
    if total_rows == 0:
        raise ValueError("No rows found in input table after filtering/limit")

    log(f"Rows after size filter: {total_rows}")
    mesh_path = resolve_and_prepare_mesh(
        args.mesh_path, args.skeleton_seg_id, args.cv_path, args.use_https
    )
    skeleton = build_skeleton(mesh_path, args.skeleton_seg_id)
    bbox_df = prefilter_with_skeleton_bbox(
        input_df,
        skeleton=skeleton,
        voxel_size_nm=tuple(float(v) for v in args.voxel_size),
        max_distance_nm=float(args.max_distance_nm),
        bbox_extra_buffer_nm=float(args.bbox_extra_buffer_nm),
    )
    if bbox_df.empty:
        raise ValueError(
            "No rows remained after bbox prefilter; check mesh alignment and voxel size"
        )

    bbox_df = annotate_distance_to_skeleton(
        bbox_df,
        skeleton=skeleton,
        voxel_size_nm=tuple(float(v) for v in args.voxel_size),
        distance_mode=args.distance_mode,
    )
    near_df = bbox_df[bbox_df["distance_to_skeleton_nm"] < args.max_distance_nm].copy()
    if near_df.empty:
        raise ValueError(
            f"No rows within {args.max_distance_nm} nm of skeleton after filtering"
        )
    log(
        f"Rows within {args.max_distance_nm:.1f} nm: {len(near_df)}/{len(bbox_df)} "
        f"after bbox prefilter ({len(near_df)}/{len(input_df)} of initial rows)"
    )

    task_count = math.ceil(len(near_df) / args.chunk_size)
    log(
        f"Prepared {len(near_df)} lookup rows, {task_count} chunks, "
        f"workers={args.workers}, format={args.format}"
    )
    tasks = build_tasks(near_df, args.chunk_size)

    run_dir, chunk_dir, final_path = prepare_run_dirs(args.output_dir, args.format)
    log(f"Run directory: {run_dir}")

    print("Looking up supervoxel ids for ribbon centroids")
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
        total_input_rows=len(near_df),
        output_format=args.format,
    )

    final_df = near_df.merge(
        merged[["source_row", "segmentation_id"]],
        on="source_row",
        how="left",
        validate="one_to_one",
    )
    if final_df["segmentation_id"].isna().any():
        raise RuntimeError("Merge produced missing segmentation_id values")
    final_df.sort_values("source_row", inplace=True, kind="mergesort")
    final_df.reset_index(drop=True, inplace=True)
    write_table(final_df, final_path, args.format)

    elapsed = time.time() - started
    log(
        f"Finished in {elapsed:.1f} seconds; "
        f"final rows: {len(final_df)}; "
        f"rate: {len(final_df) / max(1e-6, elapsed):.1f} rows/s"
    )
    log(f"Final output: {final_path}")


if __name__ == "__main__":
    main()
