#!/usr/bin/env python3
"""Compare HLS top-level INT8 output against exported software golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_SHAPE = (1, 64, 128, 2)
OUTPUT_BYTES = int(np.prod(OUTPUT_SHAPE))


def load_i8_logits(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    if data.size != OUTPUT_BYTES:
        raise ValueError(f"{path} has {data.size} bytes, expected {OUTPUT_BYTES}")
    return data.reshape(OUTPUT_SHAPE)


def remap_target(target: np.ndarray) -> np.ndarray:
    mapped = target.astype(np.int64, copy=True)
    uniq = set(np.unique(mapped).tolist())
    if uniq.issubset({0, 1, 255}):
        return mapped
    valid = mapped != 255
    mapped[((mapped < 13) | (mapped > 18)) & valid] = 1
    mapped[(mapped >= 13) & (mapped <= 18)] = 0
    return mapped


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    valid = target != 255
    pred_v = pred[valid]
    target_v = target[valid]
    if target_v.size == 0:
        raise ValueError("target has no valid pixels")

    per_class_acc: list[float] = []
    per_class_iou: list[float] = []
    confusion: list[dict[str, int]] = []
    for cls in (0, 1):
        tp = int(((pred_v == cls) & (target_v == cls)).sum())
        fp = int(((pred_v == cls) & (target_v != cls)).sum())
        fn = int(((pred_v != cls) & (target_v == cls)).sum())
        support = int((target_v == cls).sum())
        per_class_acc.append(float(tp / support) if support else float("nan"))
        denom = tp + fp + fn
        per_class_iou.append(float(tp / denom) if denom else float("nan"))
        confusion.append({"class": cls, "tp": tp, "fp": fp, "fn": fn, "support": support})

    return {
        "valid_pixels": int(target_v.size),
        "pixel_accuracy": float((pred_v == target_v).sum() / target_v.size),
        "mIoU": float(np.nanmean(per_class_iou)),
        "per_class_accuracy": per_class_acc,
        "per_class_IoU": per_class_iou,
        "confusion": confusion,
    }


def logit_diff_stats(hls: np.ndarray, golden: np.ndarray) -> dict[str, Any]:
    diff = hls.astype(np.int16) - golden.astype(np.int16)
    abs_diff = np.abs(diff)
    return {
        "bytes": int(diff.size),
        "byte_mismatches": int((diff != 0).sum()),
        "max_abs_diff": int(abs_diff.max()),
        "min_signed_diff": int(diff.min()),
        "max_signed_diff": int(diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "mse": float((diff.astype(np.float64) ** 2).mean()),
        "abs_diff_le_1": int((abs_diff <= 1).sum()),
        "abs_diff_le_2": int((abs_diff <= 2).sum()),
        "abs_diff_le_4": int((abs_diff <= 4).sum()),
        "abs_diff_le_8": int((abs_diff <= 8).sum()),
        "abs_diff_le_16": int((abs_diff <= 16).sum()),
    }


def mask_diff_stats(hls_mask: np.ndarray, golden_mask: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    diff = hls_mask != golden_mask
    valid = target != 255
    return {
        "all_pixels": int(diff.size),
        "mask_mismatches_all": int(diff.sum()),
        "mask_mismatch_rate_all": float(diff.mean()),
        "valid_pixels": int(valid.sum()),
        "mask_mismatches_valid": int((diff & valid).sum()),
        "mask_mismatch_rate_valid": float((diff & valid).sum() / valid.sum()),
        "golden_correct_hls_wrong": int(((golden_mask == target) & (hls_mask != target) & valid).sum()),
        "golden_wrong_hls_correct": int(((golden_mask != target) & (hls_mask == target) & valid).sum()),
        "both_wrong_diff": int(((golden_mask != target) & (hls_mask != target) & diff & valid).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("D:/ESP_INT8/hw_artifacts/sched_v3_single_p7_hwconv_0623"),
    )
    parser.add_argument(
        "--hls-output",
        type=Path,
        default=Path("D:/ESP_INT8/ESP_INT8_hls/hls_output_q.bin"),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    golden = load_i8_logits(artifact_dir / "golden_output_q.bin")
    hls = load_i8_logits(args.hls_output)
    target = remap_target(np.load(artifact_dir / "target.npy"))

    golden_mask = np.argmax(golden, axis=-1)
    hls_mask = np.argmax(hls, axis=-1)

    result = {
        "artifact_dir": str(artifact_dir),
        "hls_output": str(args.hls_output),
        "logit_diff": logit_diff_stats(hls, golden),
        "mask_diff": mask_diff_stats(hls_mask, golden_mask, target),
        "golden_metrics": segmentation_metrics(golden_mask, target),
        "hls_metrics": segmentation_metrics(hls_mask, target),
    }
    result["metric_delta_hls_minus_golden"] = {
        "pixel_accuracy": result["hls_metrics"]["pixel_accuracy"]
        - result["golden_metrics"]["pixel_accuracy"],
        "mIoU": result["hls_metrics"]["mIoU"] - result["golden_metrics"]["mIoU"],
    }

    out_json = args.out_json or (artifact_dir / "hls_top_compare_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"wrote {out_json}")
    print(
        "logit: "
        f"mismatches={result['logit_diff']['byte_mismatches']}/{result['logit_diff']['bytes']} "
        f"max_abs={result['logit_diff']['max_abs_diff']} "
        f"mean_abs={result['logit_diff']['mean_abs_diff']:.6f}"
    )
    print(
        "mask: "
        f"all={result['mask_diff']['mask_mismatches_all']}/{result['mask_diff']['all_pixels']} "
        f"valid={result['mask_diff']['mask_mismatches_valid']}/{result['mask_diff']['valid_pixels']}"
    )
    print(
        "golden: "
        f"PA={result['golden_metrics']['pixel_accuracy']:.8f} "
        f"mIoU={result['golden_metrics']['mIoU']:.8f}"
    )
    print(
        "hls: "
        f"PA={result['hls_metrics']['pixel_accuracy']:.8f} "
        f"mIoU={result['hls_metrics']['mIoU']:.8f}"
    )
    print(
        "delta_hls_minus_golden: "
        f"PA={result['metric_delta_hls_minus_golden']['pixel_accuracy']:.8f} "
        f"mIoU={result['metric_delta_hls_minus_golden']['mIoU']:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
