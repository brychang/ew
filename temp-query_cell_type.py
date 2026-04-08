from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from caveclient import CAVEclient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map ribbon segmentation IDs to EWII cell class/type labels and "
            "write annotated outputs."
        )
    )
    parser.add_argument(
        "--ribbon-csv",
        type=Path,
        default=Path(
            "data/ribbon_seg_id_runs/run_20260328_125747/ribbon_with_seg_ids.csv"
        ),
        help="Path to ribbon_with_seg_ids.csv.",
    )
    parser.add_argument(
        "--ewii-csv",
        type=Path,
        default=Path("data/ewii_bc_20260330.csv"),
        help="Path to ewii_bc_20260330.csv.",
    )
    parser.add_argument(
        "--seg-id-column",
        default="segmentation_id",
        help="Column in ribbon CSV containing supervoxel IDs.",
    )
    parser.add_argument(
        "--dataset",
        default="stroeh_mouse_retina",
        help="Dataset name used by CAVEclient for supervoxel->root mapping.",
    )
    parser.add_argument(
        "--root-batch-size",
        type=int,
        default=2000,
        help="Batch size for chunkedgraph.get_roots calls.",
    )
    parser.add_argument(
        "--output-annotated",
        type=Path,
        default=None,
        help=(
            "Output CSV for row-level ribbon annotations. "
            "Defaults next to ribbon CSV as ribbon_with_cell_labels.csv."
        ),
    )
    parser.add_argument(
        "--output-unique",
        type=Path,
        default=None,
        help=(
            "Output CSV for unique seg_id mapping summary. "
            "Defaults next to ribbon CSV as seg_id_cell_labels.csv."
        ),
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def normalize_header_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def parse_seg_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return None

    if text.isdigit():
        return text

    if text.endswith(".0"):
        base = text[:-2]
        if base.isdigit():
            return base

    return None


def first_nonempty(values: list[str]) -> str:
    for value in values:
        stripped = value.strip()
        if stripped:
            return stripped
    return ""


def choose_first_column(header: list[str], candidates: list[str]) -> int | None:
    normalized_to_index: dict[str, int] = {}
    for i, col in enumerate(header):
        norm = normalize_header_name(col)
        if norm not in normalized_to_index:
            normalized_to_index[norm] = i

    for candidate in candidates:
        idx = normalized_to_index.get(normalize_header_name(candidate))
        if idx is not None:
            return idx
    return None


def detect_ewii_columns(
    header: list[str],
) -> tuple[int, int | None, int | None]:
    latest_seg_id_idx = choose_first_column(header, ["Latest SegID"])
    if latest_seg_id_idx is None:
        raise ValueError("Could not find 'Latest SegID' column in EWII header")

    cell_type_idx = choose_first_column(header, ["Cell Type"])
    cell_type_machine_idx = choose_first_column(
        header, ["Cell Type (machine)", "Cell Type machine"]
    )

    return (
        latest_seg_id_idx,
        cell_type_idx,
        cell_type_machine_idx,
    )


def load_ewii_mapping(
    ewii_csv: Path,
) -> tuple[dict[str, tuple[str, str]], int, int, int]:
    with ewii_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        (
            latest_seg_id_idx,
            cell_type_idx,
            cell_type_machine_idx,
        ) = detect_ewii_columns(header)

        mapping: dict[str, tuple[str, str]] = {}
        skipped_rows = 0
        duplicate_conflicts = 0

        for row in reader:
            if not row:
                continue

            cell_type = (
                row[cell_type_idx].strip()
                if cell_type_idx is not None and cell_type_idx < len(row)
                else ""
            )
            cell_type_machine = (
                row[cell_type_machine_idx].strip()
                if cell_type_machine_idx is not None
                and cell_type_machine_idx < len(row)
                else ""
            )

            if latest_seg_id_idx >= len(row):
                skipped_rows += 1
                continue

            seg_id = parse_seg_id(row[latest_seg_id_idx])
            if seg_id is None:
                skipped_rows += 1
                continue

            label = (
                first_nonempty([cell_type]),
                first_nonempty([cell_type_machine]),
            )

            existing = mapping.get(seg_id)
            if existing is None:
                mapping[seg_id] = label
                continue

            if existing != label and any(existing) and any(label):
                duplicate_conflicts += 1

    return mapping, skipped_rows, 1, duplicate_conflicts


def annotate_ribbon_rows(
    ribbon_csv: Path,
    seg_id_column: str,
    mapping: dict[str, tuple[str, str]],
    sv_to_root: dict[str, str],
    output_annotated: Path,
    output_unique: Path,
) -> tuple[int, int, int, int]:
    with ribbon_csv.open("r", newline="", encoding="utf-8-sig") as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError("Ribbon CSV appears to be empty")
        if seg_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{seg_id_column}' not found in ribbon CSV. "
                f"Columns: {reader.fieldnames}"
            )

        out_fieldnames = [
            *reader.fieldnames,
            "final_seg_id",
            "Cell Type",
            "Cell Type (machine)",
            "match_found",
        ]

        unique_rows: dict[str, dict[str, str]] = {}
        total = 0
        matched = 0
        unmapped_sv = 0

        output_annotated.parent.mkdir(parents=True, exist_ok=True)
        with output_annotated.open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                sv_id = parse_seg_id(row.get(seg_id_column))
                final_seg_id = ""
                cell_type = ""
                cell_type_machine = ""
                match_found = "False"

                if sv_id is not None:
                    final_seg_id = sv_to_root.get(sv_id, "")
                    if not final_seg_id:
                        unmapped_sv += 1

                if final_seg_id and final_seg_id in mapping:
                    cell_type, cell_type_machine = mapping[final_seg_id]
                    match_found = "True"
                    matched += 1

                out_row = {
                    **row,
                    "final_seg_id": final_seg_id,
                    "Cell Type": cell_type,
                    "Cell Type (machine)": cell_type_machine,
                    "match_found": match_found,
                }
                writer.writerow(out_row)

                if sv_id is not None:
                    item = unique_rows.get(sv_id)
                    if item is None:
                        unique_rows[sv_id] = {
                            "supervoxel_id": sv_id,
                            "final_seg_id": final_seg_id,
                            "Cell Type": cell_type,
                            "Cell Type (machine)": cell_type_machine,
                            "match_found": match_found,
                            "ribbon_count": "1",
                        }
                    else:
                        item["ribbon_count"] = str(int(item["ribbon_count"]) + 1)
                        if item["match_found"] != "True" and match_found == "True":
                            item["final_seg_id"] = final_seg_id
                            item["Cell Type"] = cell_type
                            item["Cell Type (machine)"] = cell_type_machine
                            item["match_found"] = "True"

    output_unique.parent.mkdir(parents=True, exist_ok=True)
    with output_unique.open("w", newline="", encoding="utf-8") as out_f:
        fieldnames = [
            "supervoxel_id",
            "final_seg_id",
            "Cell Type",
            "Cell Type (machine)",
            "match_found",
            "ribbon_count",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for sv_id in sorted(unique_rows, key=int):
            writer.writerow(unique_rows[sv_id])

    return total, matched, len(unique_rows), unmapped_sv


def build_supervoxel_to_root_map(
    dataset: str,
    supervoxel_ids: list[str],
    batch_size: int,
) -> dict[str, str]:
    if batch_size < 1:
        raise ValueError("--root-batch-size must be >= 1")

    client = CAVEclient(dataset)
    sv_to_root: dict[str, str] = {}
    total = len(supervoxel_ids)
    log(f"Mapping {total} supervoxels to final roots in dataset '{dataset}'")

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = supervoxel_ids[start:stop]
        batch_ints = [int(x) for x in batch]

        roots = client.chunkedgraph.get_roots(supervoxel_ids=batch_ints)
        for sv_id, root_id in zip(batch, roots, strict=True):
            sv_to_root[sv_id] = str(int(root_id))

        log(f"Root mapping progress: {stop}/{total}")

    return sv_to_root


def load_unique_supervoxel_ids(ribbon_csv: Path, seg_id_column: str) -> list[str]:
    unique: set[str] = set()
    with ribbon_csv.open("r", newline="", encoding="utf-8-sig") as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError("Ribbon CSV appears to be empty")
        if seg_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{seg_id_column}' not found in ribbon CSV. "
                f"Columns: {reader.fieldnames}"
            )

        for row in reader:
            sv_id = parse_seg_id(row.get(seg_id_column))
            if sv_id is not None:
                unique.add(sv_id)

    return sorted(unique, key=int)


def main() -> None:
    args = parse_args()

    ribbon_csv = args.ribbon_csv
    ewii_csv = args.ewii_csv
    if not ribbon_csv.exists():
        raise FileNotFoundError(f"Ribbon CSV not found: {ribbon_csv}")
    if not ewii_csv.exists():
        raise FileNotFoundError(f"EWII CSV not found: {ewii_csv}")

    output_annotated = args.output_annotated
    if output_annotated is None:
        output_annotated = ribbon_csv.with_name("ribbon_with_cell_labels.csv")

    output_unique = args.output_unique
    if output_unique is None:
        output_unique = ribbon_csv.with_name("seg_id_cell_labels.csv")

    log(f"Loading EWII mapping from {ewii_csv}")
    mapping, skipped_rows, seg_col_count, duplicate_conflicts = load_ewii_mapping(
        ewii_csv
    )
    log(
        f"EWII loaded: {len(mapping)} seg IDs mapped from {seg_col_count} Latest SegID column; "
        f"{skipped_rows} rows had no parseable seg ID"
    )
    if duplicate_conflicts > 0:
        log(
            "Warning: conflicting labels found for some seg IDs in EWII. "
            f"Conflicts counted: {duplicate_conflicts}"
        )

    unique_supervoxels = load_unique_supervoxel_ids(ribbon_csv, args.seg_id_column)
    sv_to_root = build_supervoxel_to_root_map(
        dataset=args.dataset,
        supervoxel_ids=unique_supervoxels,
        batch_size=args.root_batch_size,
    )

    log(f"Annotating ribbon rows from {ribbon_csv}")
    total, matched, unique_sv_ids, unmapped_sv = annotate_ribbon_rows(
        ribbon_csv=ribbon_csv,
        seg_id_column=args.seg_id_column,
        mapping=mapping,
        sv_to_root=sv_to_root,
        output_annotated=output_annotated,
        output_unique=output_unique,
    )

    log(
        f"Done. matched_rows={matched}/{total} ({(matched / total * 100.0) if total else 0.0:.2f}%), "
        f"unique_supervoxels={unique_sv_ids}, unmapped_supervoxels={unmapped_sv}"
    )
    log(f"Annotated output: {output_annotated}")
    log(f"Unique seg-id output: {output_unique}")


if __name__ == "__main__":
    main()
