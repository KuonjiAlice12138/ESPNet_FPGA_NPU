#!/usr/bin/env python3
"""Export PARAM-v4 closed-loop replay tensors for P7 precision debugging.

The generated tensors are hardware-ISA goldens: they come from PARAM.BIN,
packed weights, qparams, exec_plan, and row_consumer descriptors, not from
PyTorch fake-quant module hooks. Use these files to align QAT/export behavior
with the actual hardware integer forward path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_param_replay import ParamBlob, compare_i8_arrays, replay_prefix


DEFAULT_ROOT = Path(r"D:\ESP_INT8")
DEFAULT_ARTIFACT_DIR = DEFAULT_ROOT / "hw_artifacts" / "sched_v4_p7_0702"
DEFAULT_QAT_DIR = DEFAULT_ROOT / "quantized_artifacts_hw_constrained_qat_p7_hwconv_0623"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "hw_artifacts" / "p7_param_v4_replay_prefix"


TENSOR_GOLDEN_PATHS = {
    2: ("b1_cat_ff", "output_int.npy"),
    3: ("b1_bn", "output_int.npy"),
    5: ("level2_0_bn", "output_int.npy"),
    9: ("b2_cat_ff", "output_int.npy"),
    10: ("b2_bn", "output_int.npy"),
    16: ("b3_bn", "output_int.npy"),
    17: ("classifier", "output_int.npy"),
}


def load_qat_golden_nhwc(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return np.ascontiguousarray(np.transpose(arr[0], (1, 2, 0))).astype(np.int8)
    if arr.ndim == 3:
        return np.ascontiguousarray(arr).astype(np.int8)
    raise ValueError(f"unsupported golden tensor shape at {path}: {arr.shape}")


def export_prefix(
    artifact_dir: Path,
    qat_dir: Path,
    out_dir: Path,
    stop_logical_uop: int,
    tensor_ids: Iterable[int],
) -> dict:
    blob = ParamBlob(artifact_dir / "PARAM.BIN")
    replay = replay_prefix(blob, artifact_dir / "input_q.bin", stop_logical_uop=stop_logical_uop)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "ESP_INT8_P7_PARAM_REPLAY_PREFIX_V1",
        "artifact_dir": str(artifact_dir),
        "qat_dir": str(qat_dir),
        "stop_logical_uop": int(stop_logical_uop),
        "executed_logical_uops": replay.executed,
        "tensors": {},
    }

    golden_dir = qat_dir / "golden_sample"
    for tid in tensor_ids:
        arr = replay.read_tensor(int(tid))
        npy_name = f"T{int(tid):02d}_replay_nhwc.npy"
        bin_name = f"T{int(tid):02d}_replay_nhwc.bin"
        np.save(out_dir / npy_name, arr)
        np.ascontiguousarray(arr).tofile(out_dir / bin_name)
        record = {
            "shape": list(arr.shape),
            "dtype": "int8",
            "npy": npy_name,
            "bin": bin_name,
        }
        golden_rel = TENSOR_GOLDEN_PATHS.get(int(tid))
        if golden_rel is not None:
            golden_path = golden_dir / golden_rel[0] / golden_rel[1]
            if golden_path.exists():
                golden = load_qat_golden_nhwc(golden_path)
                record["qat_golden"] = str(golden_path)
                record["qat_golden_compare"] = compare_i8_arrays(arr, golden)
        manifest["tensors"][str(int(tid))] = record

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_tensor_ids(text: str) -> list[int]:
    return [int(x, 0) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--qat-dir", type=Path, default=DEFAULT_QAT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stop-logical-uop", type=int, default=20)
    parser.add_argument("--tensor-ids", type=parse_tensor_ids, default=parse_tensor_ids("2,3,5"))
    args = parser.parse_args()

    manifest = export_prefix(
        artifact_dir=args.artifact_dir,
        qat_dir=args.qat_dir,
        out_dir=args.out_dir,
        stop_logical_uop=args.stop_logical_uop,
        tensor_ids=args.tensor_ids,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
