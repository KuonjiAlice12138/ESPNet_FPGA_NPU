#!/usr/bin/env python3
"""Compare a CSim tensor dump (NHWC int8) with golden_sample output_int.npy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_golden(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"{path} shape={arr.shape}; expected NCHW with batch=1")
    arr = np.asarray(arr, dtype=np.int16)
    if arr.min() < -128 or arr.max() > 127:
        raise ValueError(f"{path} values outside int8 range: min={arr.min()} max={arr.max()}")
    return np.ascontiguousarray(np.transpose(arr[0], (1, 2, 0)).astype(np.int8))


def load_dump(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected} for {shape}")
    return data.reshape(shape)


def compare(dump: np.ndarray, golden: np.ndarray) -> dict[str, Any]:
    if dump.shape != golden.shape:
        raise ValueError(f"shape mismatch dump={dump.shape} golden={golden.shape}")
    diff = dump.astype(np.int16) - golden.astype(np.int16)
    abs_diff = np.abs(diff)
    dump_mask = np.argmax(dump, axis=-1) if dump.shape[-1] > 1 else None
    golden_mask = np.argmax(golden, axis=-1) if golden.shape[-1] > 1 else None
    result: dict[str, Any] = {
        "shape": list(dump.shape),
        "bytes": int(diff.size),
        "byte_mismatches": int((diff != 0).sum()),
        "max_abs_diff": int(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "abs_diff_le_1": int((abs_diff <= 1).sum()),
        "abs_diff_le_2": int((abs_diff <= 2).sum()),
        "abs_diff_le_4": int((abs_diff <= 4).sum()),
        "abs_diff_le_8": int((abs_diff <= 8).sum()),
        "dump_min": int(dump.min()),
        "dump_max": int(dump.max()),
        "golden_min": int(golden.min()),
        "golden_max": int(golden.max()),
    }
    if dump_mask is not None and golden_mask is not None:
        result["argmax_mismatches"] = int((dump_mask != golden_mask).sum())
        result["argmax_pixels"] = int(dump_mask.size)
        result["argmax_mismatch_rate"] = float((dump_mask != golden_mask).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    golden = load_golden(args.golden)
    dump = load_dump(args.dump, tuple(int(x) for x in golden.shape))
    result = compare(dump, golden)
    result["dump"] = str(args.dump)
    result["golden"] = str(args.golden)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
