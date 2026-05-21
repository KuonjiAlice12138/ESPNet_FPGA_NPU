#!/usr/bin/env python3
"""Evaluate board outputs produced from export_val_hw_dataset.py files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_SHAPE = (64, 128, 2)


def load_logits(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    expected = int(np.prod(OUTPUT_SHAPE))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected}")
    return data.reshape(OUTPUT_SHAPE)


def load_mask(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = 64 * 128
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected}")
    return data.reshape((64, 128))


def read_rows(dataset_dir: Path) -> list[dict[str, str]]:
    csv_path = dataset_dir / "VALMAN.CSV"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_metrics(accum: dict[str, Any], pred: np.ndarray, target: np.ndarray) -> None:
    valid = target != 255
    pred_v = pred[valid]
    target_v = target[valid]
    accum["valid_pixels"] += int(target_v.size)
    accum["correct_pixels"] += int((pred_v == target_v).sum())
    for cls in (0, 1):
        accum["tp"][cls] += int(((pred_v == cls) & (target_v == cls)).sum())
        accum["fp"][cls] += int(((pred_v == cls) & (target_v != cls)).sum())
        accum["fn"][cls] += int(((pred_v != cls) & (target_v == cls)).sum())
        accum["support"][cls] += int((target_v == cls).sum())


def finalize_metrics(accum: dict[str, Any]) -> dict[str, Any]:
    per_class_acc: list[float | None] = []
    per_class_iou: list[float | None] = []
    for cls in (0, 1):
        support = accum["support"][cls]
        denom = accum["tp"][cls] + accum["fp"][cls] + accum["fn"][cls]
        per_class_acc.append(accum["tp"][cls] / support if support else None)
        per_class_iou.append(accum["tp"][cls] / denom if denom else None)
    valid_ious = [x for x in per_class_iou if x is not None]
    return {
        "valid_pixels": accum["valid_pixels"],
        "pixel_accuracy": (
            accum["correct_pixels"] / accum["valid_pixels"]
            if accum["valid_pixels"]
            else None
        ),
        "mIoU": float(sum(valid_ious) / len(valid_ious)) if valid_ious else None,
        "per_class_accuracy": per_class_acc,
        "per_class_IoU": per_class_iou,
        "confusion": [
            {
                "class": cls,
                "tp": accum["tp"][cls],
                "fp": accum["fp"][cls],
                "fn": accum["fn"][cls],
                "support": accum["support"][cls],
            }
            for cls in (0, 1)
        ],
    }


def empty_metrics_accum() -> dict[str, Any]:
    return {
        "valid_pixels": 0,
        "correct_pixels": 0,
        "tp": [0, 0],
        "fp": [0, 0],
        "fn": [0, 0],
        "support": [0, 0],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    rows = read_rows(dataset_dir)
    if args.limit > 0:
        rows = rows[: args.limit]

    metric_accum = empty_metrics_accum()
    ref_metric_accum = empty_metrics_accum()
    missing: list[str] = []
    processed = 0
    logit_mismatches = 0
    logit_total = 0
    max_abs_diff = 0
    sum_abs_diff = 0
    mask_mismatches = 0
    mask_total = 0

    for row in rows:
        idx = int(row["index"])
        out_name = args.output_pattern % idx
        out_path = output_dir / out_name
        if not out_path.exists():
            missing.append(str(out_path))
            if args.allow_missing:
                continue
            raise FileNotFoundError(out_path)

        target = load_mask(dataset_dir / row["target"])
        board_logits = load_logits(out_path)
        board_mask = np.argmax(board_logits, axis=-1).astype(np.uint8)
        update_metrics(metric_accum, board_mask, target)

        if args.compare_ref:
            ref_logits = load_logits(dataset_dir / row["ref"])
            ref_mask = np.argmax(ref_logits, axis=-1).astype(np.uint8)
            update_metrics(ref_metric_accum, ref_mask, target)

            diff = board_logits.astype(np.int16) - ref_logits.astype(np.int16)
            abs_diff = np.abs(diff)
            logit_mismatches += int((diff != 0).sum())
            logit_total += int(diff.size)
            max_abs_diff = max(max_abs_diff, int(abs_diff.max()))
            sum_abs_diff += int(abs_diff.sum())
            mask_mismatches += int((board_mask != ref_mask).sum())
            mask_total += int(board_mask.size)

        processed += 1

    result = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "output_pattern": args.output_pattern,
        "processed": processed,
        "missing": missing,
        "board_metrics": finalize_metrics(metric_accum),
    }
    if args.compare_ref:
        result["reference_metrics"] = finalize_metrics(ref_metric_accum)
        result["reference_diff"] = {
            "logit_bytes": logit_total,
            "logit_mismatches": logit_mismatches,
            "logit_mismatch_rate": logit_mismatches / logit_total if logit_total else None,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": sum_abs_diff / logit_total if logit_total else None,
            "mask_pixels": mask_total,
            "mask_mismatches": mask_mismatches,
            "mask_mismatch_rate": mask_mismatches / mask_total if mask_total else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=r"D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-pattern", default="O%04d.BIN")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-compare-ref", dest="compare_ref", action="store_false")
    parser.set_defaults(compare_ref=True)
    args = parser.parse_args()

    result = evaluate(args)
    out_json = Path(args.out_json) if args.out_json else Path(args.dataset_dir) / "board_val_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = result["board_metrics"]
    print(f"processed={result['processed']}")
    print(f"board PA={metrics['pixel_accuracy']:.8f} mIoU={metrics['mIoU']:.8f}")
    if "reference_diff" in result:
        diff = result["reference_diff"]
        print(
            "ref diff: "
            f"logit_mismatch={diff['logit_mismatches']}/{diff['logit_bytes']} "
            f"max_abs={diff['max_abs_diff']} "
            f"mask_mismatch={diff['mask_mismatches']}/{diff['mask_pixels']}"
        )
    print(f"saved {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
