#!/usr/bin/env python3
"""Audit P7 precision-contract alignment across QAT/export/PARAM/HLS.

This script is diagnostic, not a model exporter. It answers whether the current
Python QAT hooks, PARAM replay helpers, and HLS arithmetic source are describing
the same INT8 forward path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_param_replay import ParamBlob, compare_i8_arrays, replay_prefix


DEFAULT_ROOT = Path(r"D:\ESP_INT8")
DEFAULT_ARTIFACT_DIR = DEFAULT_ROOT / "hw_artifacts" / "sched_v3_single_p7_hwconv_0623"
DEFAULT_QAT_DIR = DEFAULT_ROOT / "quantized_artifacts_hw_constrained_qat_p7_hwconv_0623"
DEFAULT_LEGACY_DUMP_DIR = (
    DEFAULT_ROOT
    / "ESP_INT8_hls"
    / "hls_work_p7_hwqat_fullres_dump_u39_0623"
    / "hls"
    / "csim"
    / "build"
)


def detect_round_mode(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "((x - bias) + S - 1) >> shift" in text:
        return "round_away_negative"
    if re.search(r"\(x\s*-\s*bias\)\s*>>\s*shift", text):
        return "floor_shift_negative"
    return "unknown"


def inspect_qat_hook_source(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    install_start = text.find("def install_p7_hardware_precision_hooks")
    install_body = text[install_start:] if install_start >= 0 else text
    conv_patch_patterns = (
        "torch.nn.Conv2d",
        "nn.Conv2d",
        "Conv2d",
        "register_forward_pre_hook",
        "module.forward =",
    )
    return {
        "file": str(path),
        "hls_conv2d_i8_helper": "def hls_conv2d_i8_nchw" in text,
        "hls_requant_i32_helper": "def hls_requant_i32_to_i8_tensor" in text,
        "activation_hooks": "register_forward_hook" in install_body
        and "hls_activation_quant_dequant" in install_body,
        "floatfunctional_add_patch": "wrapped_add" in install_body
        and "hls_add_bypass_dequant" in install_body,
        "floatfunctional_cat_patch": "wrapped_cat" in install_body
        and "torch.cat" in install_body,
        "conv2d_forward_patch": "Conv2d" in install_body
        and any(pattern in install_body for pattern in conv_patch_patterns)
        and "weight_fake_quant" in install_body
        and "conv2d" in install_body.lower(),
    }


def load_golden_nhwc(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return np.ascontiguousarray(np.transpose(arr[0], (1, 2, 0))).astype(np.int8)
    if arr.ndim == 3:
        return np.ascontiguousarray(arr).astype(np.int8)
    raise ValueError(f"unsupported golden tensor shape at {path}: {arr.shape}")


def prefix_replay_golden_checks(artifact_dir: Path, qat_dir: Path, stop_logical_uop: int) -> Dict[str, object]:
    blob = ParamBlob(artifact_dir / "PARAM.BIN")
    replay = replay_prefix(blob, artifact_dir / "input_q.bin", stop_logical_uop=stop_logical_uop)
    golden_dir = qat_dir / "golden_sample"
    checks = {}
    nodes = [
        ("U03_B1_CAT", 3, 2, golden_dir / "b1_cat_ff" / "output_int.npy"),
        ("U04_B1_ACT", 4, 3, golden_dir / "b1_bn" / "output_int.npy"),
        ("U20_L20_ACT", 20, 5, golden_dir / "level2_0_bn" / "output_int.npy"),
        ("U72_CLASSIFIER", 72, 17, golden_dir / "classifier" / "output_int.npy"),
    ]
    for name, required_stop, tensor_id, golden_path in nodes:
        if required_stop > stop_logical_uop:
            continue
        got = replay.read_tensor(tensor_id)
        golden = load_golden_nhwc(golden_path)
        record = {
            "tensor_id": tensor_id,
            "golden": str(golden_path),
            **compare_i8_arrays(got, golden),
        }
        if name == "U03_B1_CAT":
            record["channel_slices"] = {
                "level1_conv_relu_c0_15": compare_i8_arrays(got[:, :, :16], golden[:, :, :16]),
                "pool_b1_c16_18": compare_i8_arrays(got[:, :, 16:19], golden[:, :, 16:19]),
            }
        checks[name] = record
    return {
        "stop_logical_uop": stop_logical_uop,
        "executed_logical_uops": replay.executed,
        "checks": checks,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    hls_q = args.root / "ESP_INT8_hls" / "include" / "npu_q.hpp"
    legacy_q = args.legacy_dump_dir / "npu_q.hpp"
    report = {
        "hls_round_mode_current": detect_round_mode(hls_q),
        "hls_round_mode_legacy_dump": detect_round_mode(legacy_q) if legacy_q.exists() else "missing",
        "qat_hook_source": inspect_qat_hook_source(args.root / "tools" / "hw_int8_math.py"),
        "artifact_dir": str(args.artifact_dir),
        "qat_dir": str(args.qat_dir),
    }
    if not args.skip_prefix:
        report["prefix_replay_vs_qat_golden"] = prefix_replay_golden_checks(
            args.artifact_dir,
            args.qat_dir,
            args.stop_logical_uop,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--qat-dir", type=Path, default=DEFAULT_QAT_DIR)
    parser.add_argument("--legacy-dump-dir", type=Path, default=DEFAULT_LEGACY_DUMP_DIR)
    parser.add_argument("--stop-logical-uop", type=int, default=20)
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_audit(args)
    text = json.dumps(report, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
