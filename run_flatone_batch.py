#!/usr/bin/env python3
"""Batch runner for flatone skeleton generation.

Reads a text file containing one segid per line and runs:
    uv run flatone <segid>
for each ID inside a flatone repository directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def read_ids(ids_file: Path) -> list[str]:
    ids: list[str] = []
    with ids_file.open("r", encoding="utf-8") as f:
        for raw in f:
            value = raw.strip().replace("\r", "")
            if not value:
                continue
            ids.append(value)
    return ids


def run_flatone_for_id(
    segid: str, flatone_dir: Path, timeout_sec: int | None
) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "flatone", segid]
    return subprocess.run(
        cmd,
        cwd=flatone_dir,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flatone over a list of segids.")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path(
            "data/ribbon_seg_id_runs/run_20260408_185507/target_cell_ids_missing_flatone_skeleton_warped.txt"
        ),
        help="Path to text file containing segids (one per line).",
    )
    parser.add_argument(
        "--flatone-dir",
        type=Path,
        default=Path("~/flatone").expanduser(),
        help="Path to flatone repo where uv run flatone should be executed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs and success/failure files (defaults to ids file directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="flatone_batch",
        help="Filename prefix for output artifacts.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Per-ID timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip segids that already have ~/flatone/output/<segid>/skeleton_warped.swc (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip IDs that already have skeleton_warped.swc.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after first failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ids_file = args.ids_file.expanduser().resolve()
    flatone_dir = args.flatone_dir.expanduser().resolve()

    if not ids_file.exists():
        log(f"IDs file not found: {ids_file}")
        return 2
    if not flatone_dir.exists():
        log(f"flatone directory not found: {flatone_dir}")
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ids_file.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(ids_file)
    if not ids:
        log(f"No IDs found in {ids_file}")
        return 0

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{args.prefix}_{ts}.log"
    success_file = output_dir / f"{args.prefix}_{ts}_success.txt"
    failed_file = output_dir / f"{args.prefix}_{ts}_failed.txt"
    skipped_file = output_dir / f"{args.prefix}_{ts}_skipped_existing.txt"

    header = (
        f"Start: {dt.datetime.now().isoformat()}\n"
        f"IDs file: {ids_file}\n"
        f"flatone dir: {flatone_dir}\n"
        f"Total IDs: {len(ids)}\n"
        f"Skip existing: {args.skip_existing}\n"
        f"Timeout sec: {args.timeout_sec}\n"
        "=" * 80 + "\n"
    )
    append_text(log_file, header)

    success = 0
    failed = 0
    skipped = 0

    t0 = time.time()
    for idx, segid in enumerate(ids, start=1):
        output_path = flatone_dir / "output" / segid / "skeleton_warped.swc"
        if args.skip_existing and output_path.exists():
            skipped += 1
            msg = f"[{idx}/{len(ids)}] SKIP {segid} (already exists: {output_path})"
            log(msg)
            append_text(log_file, msg + "\n")
            append_text(skipped_file, segid + "\n")
            continue

        msg = f"[{idx}/{len(ids)}] RUN  {segid}"
        log(msg)
        append_text(log_file, msg + "\n")

        try:
            result = run_flatone_for_id(segid, flatone_dir, args.timeout_sec)
        except subprocess.TimeoutExpired:
            failed += 1
            timeout_msg = f"[{idx}/{len(ids)}] FAIL {segid} (timeout)"
            log(timeout_msg)
            append_text(log_file, timeout_msg + "\n")
            append_text(failed_file, segid + "\n")
            if args.stop_on_error:
                break
            continue

        append_text(log_file, result.stdout)
        append_text(log_file, result.stderr)

        if result.returncode == 0:
            success += 1
            ok_msg = f"[{idx}/{len(ids)}] OK   {segid}"
            log(ok_msg)
            append_text(log_file, ok_msg + "\n")
            append_text(success_file, segid + "\n")
        else:
            failed += 1
            bad_msg = f"[{idx}/{len(ids)}] FAIL {segid} (exit={result.returncode})"
            log(bad_msg)
            append_text(log_file, bad_msg + "\n")
            append_text(failed_file, segid + "\n")
            if args.stop_on_error:
                break

    elapsed = time.time() - t0
    summary = (
        "\n"
        + "=" * 80
        + "\n"
        + f"Done in {elapsed:.1f}s\n"
        + f"Success: {success}\n"
        + f"Failed: {failed}\n"
        + f"Skipped existing: {skipped}\n"
        + f"Log: {log_file}\n"
        + f"Success file: {success_file}\n"
        + f"Failed file: {failed_file}\n"
        + f"Skipped file: {skipped_file}\n"
    )
    append_text(log_file, summary)

    log(f"Done in {elapsed:.1f}s")
    log(f"Success: {success}")
    log(f"Failed: {failed}")
    log(f"Skipped existing: {skipped}")
    log(f"Log: {log_file}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
