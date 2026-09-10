#!/usr/bin/env python3
"""Generate PARAM v4 execution prefixes for board-level cycle attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


PARAM_MAGIC = 0x544E4945
PARAM_VERSION = 4
PARAM_HEADER_WORDS = 32
PARAM_HEADER_BYTES = PARAM_HEADER_WORDS * 4
EXEC_PLAN_COUNT_WORD = 9
EXEC_PLAN_OFFSET_WORD = 27
EXEC_PLAN_ENTRY_BYTES = 4
EXEC_END = 255

EXEC_KIND_NAMES = {
    1: "CONV",
    2: "POOL",
    6: "BLOCK_AFFINE",
    7: "BLOCK_ADD_AFFINE",
    EXEC_END: "END",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _parse_exec_plan(source: bytes) -> tuple[int, list[tuple[int, int, int, int]]]:
    if len(source) < PARAM_HEADER_BYTES:
        raise ValueError("PARAM blob is shorter than its 128-byte header")

    header = struct.unpack_from("<32I", source, 0)
    if header[0] != PARAM_MAGIC or header[1] != PARAM_VERSION:
        raise ValueError(
            f"expected PARAM v4, got magic=0x{header[0]:08X} version={header[1]}"
        )

    exec_count = int(header[EXEC_PLAN_COUNT_WORD])
    exec_offset = int(header[EXEC_PLAN_OFFSET_WORD])
    exec_end = exec_offset + exec_count * EXEC_PLAN_ENTRY_BYTES
    if exec_count < 2:
        raise ValueError(f"exec_plan_count={exec_count}; expected active entries plus END")
    if exec_offset < PARAM_HEADER_BYTES or exec_end > len(source):
        raise ValueError(
            f"exec plan range [{exec_offset}, {exec_end}) exceeds PARAM size {len(source)}"
        )

    entries = [
        struct.unpack_from("<4B", source, exec_offset + i * EXEC_PLAN_ENTRY_BYTES)
        for i in range(exec_count)
    ]
    end_indices = [i for i, entry in enumerate(entries) if entry[0] == EXEC_END]
    if end_indices != [exec_count - 1]:
        raise ValueError(
            f"source PARAM must have exactly one terminal END at pc={exec_count - 1}; "
            f"found {end_indices}"
        )
    return exec_offset, entries


def build_exec_prefix_blobs(source: bytes) -> tuple[list[bytes], list[dict[str, Any]]]:
    """Return P00..PN where PN is the unmodified full execution plan.

    P00 terminates at exec pc 0 and provides the frame-load/control baseline.
    P01 executes source pc 0, and so on. Only the ``kind`` byte at the new
    terminal pc is modified; descriptors, qparams, layouts, and all later
    bytes remain bit-identical to the source blob.
    """

    exec_offset, entries = _parse_exec_plan(source)
    active_count = len(entries) - 1
    source_hash = _sha256(source)
    prefixes: list[bytes] = []
    records: list[dict[str, Any]] = []

    for prefix_count in range(active_count + 1):
        blob = bytearray(source)
        changed_offset: int | None = None
        if prefix_count < active_count:
            changed_offset = exec_offset + prefix_count * EXEC_PLAN_ENTRY_BYTES
            blob[changed_offset] = EXEC_END
        data = bytes(blob)

        if prefix_count == 0:
            exec_index = None
            kind = None
            desc_id = None
            logical_uop_id = None
            flags = None
        else:
            exec_index = prefix_count - 1
            kind, desc_id, logical_uop_id, flags = entries[exec_index]

        records.append(
            {
                "file": f"P{prefix_count:02d}.BIN",
                "active_exec_count": prefix_count,
                "new_exec_index": exec_index,
                "new_exec_kind": kind,
                "new_exec_kind_name": (
                    "BASELINE" if kind is None else EXEC_KIND_NAMES.get(kind, f"KIND_{kind}")
                ),
                "new_exec_desc_id": desc_id,
                "new_exec_logical_uop_id": logical_uop_id,
                "new_exec_flags": flags,
                "first_end_pc": prefix_count,
                "changed_offset": changed_offset,
                "byte_differences": 0 if changed_offset is None else 1,
                "bytes": len(data),
                "sha256": _sha256(data),
                "is_full_param": prefix_count == active_count,
            }
        )
        prefixes.append(data)

    for record in records:
        record["source_sha256"] = source_hash
    return prefixes, records


def write_exec_prefix_set(param_path: Path, out_dir: Path) -> dict[str, Any]:
    source = param_path.read_bytes()
    prefixes, records = build_exec_prefix_blobs(source)
    out_dir.mkdir(parents=True, exist_ok=True)

    for data, record in zip(prefixes, records):
        (out_dir / record["file"]).write_bytes(data)

    manifest = {
        "format": "ESP_INT8_PARAM_V4_EXEC_PREFIX_V1",
        "source": str(param_path.resolve()),
        "source_bytes": len(source),
        "source_sha256": _sha256(source),
        "exec_plan_count": len(records),
        "active_exec_count": len(records) - 1,
        "prefix_count": len(records),
        "prefixes": records,
    }
    (out_dir / "prefix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "file",
        "active_exec_count",
        "new_exec_index",
        "new_exec_kind",
        "new_exec_kind_name",
        "new_exec_desc_id",
        "new_exec_logical_uop_id",
        "new_exec_flags",
        "first_end_pc",
        "byte_differences",
        "bytes",
        "sha256",
        "is_full_param",
    ]
    with (out_dir / "PXMAP.CSV").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        type=Path,
        default=root / "hw_artifacts" / "binary2_int8_0809" / "PARAM.BIN",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "hw_artifacts" / "binary2_int8_0809" / "exec_prefix",
    )
    args = parser.parse_args()

    manifest = write_exec_prefix_set(args.param, args.out_dir)
    print(
        "EXEC prefix export: "
        f"{manifest['prefix_count']} files, "
        f"active_exec={manifest['active_exec_count']}, "
        f"source_sha256={manifest['source_sha256']}"
    )
    for record in manifest["prefixes"]:
        logical_uop = record["new_exec_logical_uop_id"]
        new_exec = (
            "BASELINE"
            if logical_uop is None
            else f"{record['new_exec_kind_name']}/U{logical_uop}"
        )
        print(
            f"{record['file']}: active={record['active_exec_count']:02d} "
            f"new={new_exec} "
            f"diff={record['byte_differences']} sha256={record['sha256']}"
        )


if __name__ == "__main__":
    main()
