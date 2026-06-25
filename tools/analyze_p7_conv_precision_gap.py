#!/usr/bin/env python3
"""Compare PARAM-v3 integer conv replay against PyTorch golden conv outputs.

This diagnoses whether the mismatch is in HLS schedule execution or in the
model-side fake-quant floating-point conv semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_int8_hw_blob import CONVS, safe_name
from hw_param_replay import (
    FeatureMemory,
    ParamBlob,
    TensorDesc,
    compare_i8_arrays,
    replay_conv,
)


def load_golden_nhwc(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"{path} shape={arr.shape}; expected NCHW batch=1")
    return np.ascontiguousarray(np.transpose(arr[0], (1, 2, 0)).astype(np.int8))


def load_named_input(artifact_dir: Path, name: str) -> np.ndarray:
    if name == "quant":
        return load_golden_nhwc(artifact_dir / "golden_sample" / "quant" / "input_int.npy")
    return load_golden_nhwc(artifact_dir / "golden_sample" / safe_name(name) / "output_int.npy")


def run_one(artifact_dir: Path, blob: ParamBlob, param_id: int) -> dict:
    spec = next(x for x in CONVS if x.param_id == param_id)
    desc = blob.conv_exec[param_id]
    src_arr = load_named_input(artifact_dir, spec.input_scale)
    golden_out = load_golden_nhwc(artifact_dir / "golden_sample" / safe_name(spec.name) / "output_int.npy")

    src_desc = TensorDesc(
        bank_id=0,
        base_offset=0,
        h=src_arr.shape[0],
        w=src_arr.shape[1],
        c=src_arr.shape[2],
        reserved0=src_arr.shape[2],
        reserved1=0,
    )
    dst_desc = TensorDesc(
        bank_id=0,
        base_offset=src_arr.size + 4096,
        h=golden_out.shape[0],
        w=golden_out.shape[1],
        c=golden_out.shape[2],
        reserved0=golden_out.shape[2],
        reserved1=0,
    )
    fmem = FeatureMemory()
    for h in range(src_desc.h):
        for w in range(src_desc.w):
            fmem.store_tile(src_desc, h, w, 0, src_desc.c, src_arr[h, w, :])
    replay_conv(fmem, blob, param_id, src_desc, dst_desc, blob.row_consumer[desc.row_consumer_id])
    replay_out = fmem.dump_tensor(dst_desc)
    cmp_result = compare_i8_arrays(replay_out, golden_out)
    cmp_result.update(
        {
            "param_id": param_id,
            "name": spec.name,
            "input_name": spec.input_scale,
            "output_name": spec.name,
            "shape": list(golden_out.shape),
            "replay_min": int(replay_out.min()),
            "replay_max": int(replay_out.max()),
            "golden_min": int(golden_out.min()),
            "golden_max": int(golden_out.max()),
        }
    )
    if golden_out.shape[-1] > 1:
        replay_argmax = np.argmax(replay_out, axis=-1)
        golden_argmax = np.argmax(golden_out, axis=-1)
        cmp_result["argmax_mismatches"] = int((replay_argmax != golden_argmax).sum())
        cmp_result["argmax_pixels"] = int(replay_argmax.size)
        cmp_result["argmax_mismatch_rate"] = float((replay_argmax != golden_argmax).mean())
    return cmp_result


def parse_layers(value: str) -> Iterable[int]:
    if value == "all":
        return [x.param_id for x in CONVS]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623"))
    parser.add_argument("--param", type=Path, default=Path(r"D:\ESP_INT8\hw_artifacts\sched_v3_single_p7_hwconv_0623\PARAM.BIN"))
    parser.add_argument("--layers", default="0,1,7,13,19,25",
                        help="Comma-separated conv param ids, or 'all'.")
    parser.add_argument("--out-json", type=Path, default=Path(r"D:\ESP_INT8\report_and_workplans\p7_conv_precision_gap_0623.json"))
    args = parser.parse_args()

    blob = ParamBlob(args.param)
    results = [run_one(args.artifact_dir, blob, pid) for pid in parse_layers(args.layers)]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
