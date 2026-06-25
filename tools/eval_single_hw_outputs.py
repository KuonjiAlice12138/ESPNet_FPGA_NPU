#!/usr/bin/env python3
"""Evaluate one INT8 NPU sample at low-res logits and full-res mask levels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


LOWRES_SHAPE = (1, 64, 128, 2)
LOWRES_TARGET_SHAPE = (1, 64, 128)
FULLRES_SHAPE = (512, 1024)
UPSAMPLE_SCALE = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_i8_logits(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    expected = int(np.prod(LOWRES_SHAPE))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected}")
    return data.reshape(LOWRES_SHAPE)


def load_lowres_target(path: Path) -> np.ndarray:
    target = np.load(path)
    if tuple(target.shape) != LOWRES_TARGET_SHAPE:
        raise ValueError(f"{path} shape={target.shape}, expected {LOWRES_TARGET_SHAPE}")
    return remap_target_np(target[0])


def remap_target_np(target: np.ndarray) -> np.ndarray:
    mapped = target.astype(np.int64, copy=True)
    valid = mapped != 255
    if set(np.unique(mapped).tolist()).issubset({0, 1, 255}):
        return mapped
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


def diff_metrics(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    diff = a != b
    if valid is None:
        return {
            "pixels": int(diff.size),
            "mismatches": int(diff.sum()),
            "mismatch_rate": float(diff.mean()),
        }
    return {
        "pixels": int(valid.sum()),
        "mismatches": int((diff & valid).sum()),
        "mismatch_rate": float((diff & valid).sum() / max(int(valid.sum()), 1)),
    }


def logit_diff_metrics(hls: np.ndarray, golden: np.ndarray) -> dict[str, Any]:
    diff = hls.astype(np.int16) - golden.astype(np.int16)
    abs_diff = np.abs(diff)
    return {
        "bytes": int(diff.size),
        "byte_mismatches": int((diff != 0).sum()),
        "max_abs_diff": int(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "abs_diff_le_1": int((abs_diff <= 1).sum()),
        "abs_diff_le_2": int((abs_diff <= 2).sum()),
        "abs_diff_le_4": int((abs_diff <= 4).sum()),
        "abs_diff_le_8": int((abs_diff <= 8).sum()),
    }


def bilinear_axis_map(out_idx: int, in_size: int) -> tuple[int, int, int, int]:
    scale = UPSAMPLE_SCALE
    clip_high_start = in_size * scale - (scale // 2)
    if out_idx < scale // 2:
        return 0, 0, 16, 0
    if out_idx >= clip_high_start:
        return in_size - 1, in_size - 1, 16, 0
    numer = 2 * out_idx - (scale - 1)
    idx0 = numer >> 4
    idx1 = idx0 + 1
    w1 = numer & 0x0F
    return idx0, idx1, 16 - w1, w1


def fullres_mask_from_logits(logits: np.ndarray) -> np.ndarray:
    logits2 = logits.reshape(64, 128, 2).astype(np.int16)
    diff = logits2[:, :, 0] - logits2[:, :, 1]
    mask = np.zeros(FULLRES_SHAPE, dtype=np.uint8)
    for y in range(FULLRES_SHAPE[0]):
        y0, y1, wy0, wy1 = bilinear_axis_map(y, 64)
        for x in range(FULLRES_SHAPE[1]):
            x0, x1, wx0, wx1 = bilinear_axis_map(x, 128)
            top = int(diff[y0, x0]) * wx0 + int(diff[y0, x1]) * wx1
            bottom = int(diff[y1, x0]) * wx0 + int(diff[y1, x1]) * wx1
            interp = top * wy0 + bottom * wy1
            mask[y, x] = 0 if interp >= 0 else 1
    return mask


def normalize_sample_name(value: str) -> str:
    return value.replace("\\", "/").replace("//", "/").lstrip("./")


def find_sample_index(artifact_dir: Path, cached_data_file: Path) -> tuple[int, str]:
    sample_info = json.loads((artifact_dir / "sample_info.json").read_text(encoding="utf-8"))
    sample_name = normalize_sample_name(sample_info["sample_name"])
    with cached_data_file.open("rb") as f:
        data = pickle.load(f)
    for idx, image in enumerate(data["valIm"]):
        if normalize_sample_name(image) == sample_name:
            return idx, image
    raise ValueError(f"sample {sample_name} not found in {cached_data_file}")


def load_fullres_target(
    espnet_dir: Path,
    cached_data_file: Path,
    sample_index: int,
) -> np.ndarray:
    sys.path.insert(0, str(espnet_dir))
    os.chdir(espnet_dir)
    import DataSet as myDataLoader  # type: ignore
    import Transforms as myTransforms  # type: ignore

    with cached_data_file.open("rb") as f:
        data = pickle.load(f)
    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(1),
    ])
    dataset = myDataLoader.MyDataset(data["valIm"], data["valAnnot"], transform=val_tf)
    _image, target, _name = dataset[sample_index]
    return remap_target_np(target.numpy())


def check_single_vs_val_smoke(artifact_dir: Path, val_smoke_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {"val_smoke_dir": str(val_smoke_dir), "available": val_smoke_dir.exists()}
    if not val_smoke_dir.exists():
        return checks
    pairs = {
        "input_vs_I0000": (artifact_dir / "input_q.bin", val_smoke_dir / "I0000.BIN"),
        "golden_vs_R0000": (artifact_dir / "golden_output_q.bin", val_smoke_dir / "R0000.BIN"),
    }
    for name, (left, right) in pairs.items():
        if left.exists() and right.exists():
            checks[name] = {
                "left": str(left),
                "right": str(right),
                "same_size": left.stat().st_size == right.stat().st_size,
                "same_sha256": sha256_file(left) == sha256_file(right),
            }
    t0 = val_smoke_dir / "T0000.BIN"
    target_npy = artifact_dir / "target.npy"
    if t0.exists() and target_npy.exists():
        low_target = load_lowres_target(target_npy).astype(np.uint8)
        t_bin = np.fromfile(t0, dtype=np.uint8)
        checks["target_remap_vs_T0000"] = {
            "target_npy": str(target_npy),
            "t0000": str(t0),
            "same_size": int(low_target.size) == int(t_bin.size),
            "mismatches": int((low_target.reshape(-1) != t_bin).sum()) if low_target.size == t_bin.size else None,
        }
    return checks


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    cached_data_file = Path(args.cached_data_file)
    sample_index, sample_name = find_sample_index(artifact_dir, cached_data_file)

    golden_logits = load_i8_logits(artifact_dir / "golden_output_q.bin")
    low_target = load_lowres_target(artifact_dir / "target.npy")
    full_target = load_fullres_target(Path(args.espnet_dir), cached_data_file, sample_index)

    golden_low_mask = np.argmax(golden_logits[0], axis=-1).astype(np.uint8)
    golden_full_mask = fullres_mask_from_logits(golden_logits)

    result: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "sample_index": sample_index,
        "sample_name": sample_name,
        "input_q_sha256": sha256_file(artifact_dir / "input_q.bin"),
        "golden_output_q_sha256": sha256_file(artifact_dir / "golden_output_q.bin"),
        "artifact_consistency": check_single_vs_val_smoke(
            artifact_dir, Path(args.val_smoke_dir)
        ),
        "golden_lowres_metrics": segmentation_metrics(golden_low_mask, low_target),
        "golden_fullres_metrics": segmentation_metrics(golden_full_mask, full_target),
    }

    if args.lowres_logits:
        hls_logits_path = Path(args.lowres_logits)
        hls_logits = load_i8_logits(hls_logits_path)
        hls_low_mask = np.argmax(hls_logits[0], axis=-1).astype(np.uint8)
        result["hls_lowres_logits"] = {
            "path": str(hls_logits_path),
            "sha256": sha256_file(hls_logits_path),
            "logit_diff": logit_diff_metrics(hls_logits, golden_logits),
            "mask_diff_vs_golden": diff_metrics(hls_low_mask, golden_low_mask, low_target != 255),
            "metrics": segmentation_metrics(hls_low_mask, low_target),
        }
        result["hls_lowres_logits"]["metric_delta_vs_golden"] = {
            "pixel_accuracy": result["hls_lowres_logits"]["metrics"]["pixel_accuracy"]
            - result["golden_lowres_metrics"]["pixel_accuracy"],
            "mIoU": result["hls_lowres_logits"]["metrics"]["mIoU"]
            - result["golden_lowres_metrics"]["mIoU"],
        }

    if args.fullres_mask:
        hls_mask_path = Path(args.fullres_mask)
        data = np.fromfile(hls_mask_path, dtype=np.uint8)
        if data.size != int(np.prod(FULLRES_SHAPE)):
            raise ValueError(f"{hls_mask_path} has {data.size} bytes, expected {np.prod(FULLRES_SHAPE)}")
        hls_full_mask = data.reshape(FULLRES_SHAPE)
        result["hls_fullres_mask"] = {
            "path": str(hls_mask_path),
            "sha256": sha256_file(hls_mask_path),
            "invalid_labels": int(((hls_full_mask != 0) & (hls_full_mask != 1)).sum()),
            "mask_diff_vs_golden": diff_metrics(hls_full_mask, golden_full_mask, full_target != 255),
            "metrics": segmentation_metrics(hls_full_mask, full_target),
        }
        result["hls_fullres_mask"]["metric_delta_vs_golden"] = {
            "pixel_accuracy": result["hls_fullres_mask"]["metrics"]["pixel_accuracy"]
            - result["golden_fullres_metrics"]["pixel_accuracy"],
            "mIoU": result["hls_fullres_mask"]["metrics"]["mIoU"]
            - result["golden_fullres_metrics"]["mIoU"],
        }

    return result


def print_summary(result: dict[str, Any]) -> None:
    print(f"sample index={result['sample_index']} name={result['sample_name']}")
    checks = result["artifact_consistency"]
    for key in ("input_vs_I0000", "golden_vs_R0000", "target_remap_vs_T0000"):
        if key in checks:
            print(f"{key}: {checks[key]}")

    gl = result["golden_lowres_metrics"]
    gf = result["golden_fullres_metrics"]
    print(f"golden lowres:  PA={gl['pixel_accuracy']:.8f} mIoU={gl['mIoU']:.8f}")
    print(f"golden fullres: PA={gf['pixel_accuracy']:.8f} mIoU={gf['mIoU']:.8f}")
    if "hls_lowres_logits" in result:
        h = result["hls_lowres_logits"]
        m = h["metrics"]
        d = h["metric_delta_vs_golden"]
        print(
            "hls lowres:    "
            f"PA={m['pixel_accuracy']:.8f} mIoU={m['mIoU']:.8f} "
            f"delta_PA={d['pixel_accuracy']:.8f} delta_mIoU={d['mIoU']:.8f} "
            f"mask_mismatch={h['mask_diff_vs_golden']['mismatches']}"
        )
    if "hls_fullres_mask" in result:
        h = result["hls_fullres_mask"]
        m = h["metrics"]
        d = h["metric_delta_vs_golden"]
        print(
            "hls fullres:   "
            f"PA={m['pixel_accuracy']:.8f} mIoU={m['mIoU']:.8f} "
            f"delta_PA={d['pixel_accuracy']:.8f} delta_mIoU={d['mIoU']:.8f} "
            f"mask_mismatch={h['mask_diff_vs_golden']['mismatches']} "
            f"invalid={h['invalid_labels']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=r"D:\ESP_INT8\hw_artifacts\sched_v3_single")
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument("--val-smoke-dir", default=r"D:\ESP_INT8\hw_artifacts\sched_v3_val_smoke")
    parser.add_argument("--lowres-logits", default="")
    parser.add_argument("--fullres-mask", default="")
    parser.add_argument(
        "--out-json",
        default=r"D:\ESP_INT8\hw_artifacts\sched_v3_single\single_hw_output_metrics.json",
    )
    args = parser.parse_args()

    result = evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
