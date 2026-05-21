#!/usr/bin/env python3
"""Export Cityscapes val samples into the ESP INT8 board-test binary layout.

The existing single-image exporter writes INPUTQ.BIN/golden_output_q.bin for one
sample. This script keeps the same NHWC signed-INT8 contract, but emits the
whole validation set with short 8.3-style names suitable for the SD-card flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hw_constrained_qat import (  # noqa: E402
    add_espnet,
    build_val_loader,
    load_fake_quant_model,
    remap_target,
)


INPUT_SHAPE_NCHW = (3, 512, 1024)
OUTPUT_SHAPE_NCHW = (2, 64, 128)
TARGET_SHAPE = (64, 128)

INPUT_BYTES = 512 * 1024 * 3
OUTPUT_BYTES = 64 * 128 * 2
TARGET_BYTES = 64 * 128


def quantize_activation_tensor(tensor: torch.Tensor, observer: Any) -> torch.Tensor:
    scale = observer.scale.detach().cpu()
    zero_point = observer.zero_point.detach().cpu().to(torch.int32)
    quant_min = int(observer.quant_min)
    quant_max = int(observer.quant_max)

    if scale.numel() != 1 or zero_point.numel() != 1:
        raise ValueError("Only per-tensor activation quantization is supported.")

    q = torch.round(tensor.detach().cpu() / float(scale.item()) + int(zero_point.item()))
    return q.clamp(quant_min, quant_max).to(torch.int32)


def as_i8(array: np.ndarray, name: str) -> np.ndarray:
    if array.min() < -128 or array.max() > 127:
        raise ValueError(f"{name} contains values outside int8 range")
    return array.astype(np.int8, copy=False)


def nchw_to_nhwc_i8(array: np.ndarray, expected_chw: tuple[int, int, int], name: str) -> np.ndarray:
    if array.shape != expected_chw:
        raise ValueError(f"{name} shape={array.shape}, expected={expected_chw}")
    nhwc = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(as_i8(nhwc, name))


def short_bin(prefix: str, index: int) -> str:
    if index < 0 or index > 9999:
        raise ValueError("short SD filenames support indices 0..9999")
    return f"{prefix}{index:04d}.BIN"


def get_nested_module(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module: torch.nn.Module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


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


def compare_file_bytes(left: Path, right: Path) -> dict[str, Any]:
    a = left.read_bytes()
    b = right.read_bytes()
    n = min(len(a), len(b))
    mismatches = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    return {
        "left": str(left),
        "right": str(right),
        "left_bytes": len(a),
        "right_bytes": len(b),
        "mismatches": mismatches,
    }


def export_val_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_param:
        param_src = Path(args.param_src)
        if not param_src.exists():
            raise FileNotFoundError(f"param blob not found: {param_src}")
        shutil.copy2(param_src, out_dir / "PARAM.BIN")

    add_espnet(Path(args.espnet_dir))
    use_cuda = bool(args.cuda and torch.cuda.is_available() and not args.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    loader = build_val_loader(
        Path(args.cached_data_file),
        args.scale_in,
        args.batch_size,
        args.num_workers,
    )
    model = load_fake_quant_model(args, device)
    model.eval()

    input_observer = model.quant.activation_post_process
    classifier_observer = get_nested_module(model, "classifier").activation_post_process

    csv_path = out_dir / "VALMAN.CSV"
    rows: list[dict[str, Any]] = []
    accum = {
        "valid_pixels": 0,
        "correct_pixels": 0,
        "tp": [0, 0],
        "fp": [0, 0],
        "fn": [0, 0],
        "support": [0, 0],
    }

    start = time.time()
    exported = 0
    total = len(loader.dataset)
    max_items = total if args.limit <= 0 else min(args.limit, total)

    with torch.no_grad():
        for batch_idx, (inputs, targets, names) in enumerate(loader):
            if exported >= max_items:
                break

            batch = inputs.shape[0]
            inputs_dev = inputs.to(device, non_blocking=True)
            captured: dict[str, torch.Tensor] = {}

            def capture_classifier(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError("classifier hook received a non-tensor output")
                captured["classifier"] = output.detach()

            handle = get_nested_module(model, "classifier").register_forward_hook(capture_classifier)
            try:
                _ = model(inputs_dev)
            finally:
                handle.remove()
            if "classifier" not in captured:
                raise RuntimeError("failed to capture classifier output")

            q_inputs = quantize_activation_tensor(inputs, input_observer).numpy()
            q_outputs = quantize_activation_tensor(captured["classifier"], classifier_observer).numpy()
            mapped_targets = remap_target(targets).cpu().numpy().astype(np.uint8, copy=False)

            for b in range(batch):
                if exported >= max_items:
                    break

                idx = args.first_index + exported
                input_name = short_bin("I", idx)
                ref_name = short_bin("R", idx)
                target_name = short_bin("T", idx)
                mask_name = short_bin("M", idx)

                input_nhwc = nchw_to_nhwc_i8(q_inputs[b], INPUT_SHAPE_NCHW, input_name)
                ref_nhwc = nchw_to_nhwc_i8(q_outputs[b], OUTPUT_SHAPE_NCHW, ref_name)
                target = np.ascontiguousarray(mapped_targets[b])
                if target.shape != TARGET_SHAPE:
                    raise ValueError(f"{target_name} shape={target.shape}, expected={TARGET_SHAPE}")
                mask = np.ascontiguousarray(np.argmax(ref_nhwc, axis=-1).astype(np.uint8))

                input_nhwc.tofile(out_dir / input_name)
                ref_nhwc.tofile(out_dir / ref_name)
                target.tofile(out_dir / target_name)
                mask.tofile(out_dir / mask_name)

                if args.write_npy:
                    np.save(out_dir / input_name.replace(".BIN", ".npy"), input_nhwc)
                    np.save(out_dir / ref_name.replace(".BIN", ".npy"), ref_nhwc)
                    np.save(out_dir / target_name.replace(".BIN", ".npy"), target)
                    np.save(out_dir / mask_name.replace(".BIN", ".npy"), mask)

                update_metrics(accum, mask, target)
                image_name = names[b] if isinstance(names, (list, tuple)) else names
                rows.append(
                    {
                        "index": idx,
                        "input": input_name,
                        "ref": ref_name,
                        "target": target_name,
                        "mask": mask_name,
                        "image": str(image_name),
                        "input_bytes": INPUT_BYTES,
                        "ref_bytes": OUTPUT_BYTES,
                        "target_bytes": TARGET_BYTES,
                        "mask_bytes": TARGET_BYTES,
                    }
                )
                exported += 1

            if args.progress_every > 0 and (
                exported == max_items or (batch_idx + 1) % args.progress_every == 0
            ):
                print(f"exported {exported}/{max_items} val samples")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "input",
            "ref",
            "target",
            "mask",
            "image",
            "input_bytes",
            "ref_bytes",
            "target_bytes",
            "mask_bytes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "format": "ESP_INT8_VAL_HW_DATASET_V1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "espnet_dir": str(args.espnet_dir),
        "artifact_dir": str(args.artifact_dir),
        "cached_data_file": str(args.cached_data_file),
        "out_dir": str(out_dir),
        "num_exported": exported,
        "num_val_total": total,
        "batch_size": args.batch_size,
        "device": str(device),
        "files": {
            "param": "PARAM.BIN" if args.copy_param else None,
            "manifest_csv": "VALMAN.CSV",
            "input_pattern": "I%04d.BIN",
            "reference_logit_pattern": "R%04d.BIN",
            "target_pattern": "T%04d.BIN",
            "reference_mask_pattern": "M%04d.BIN",
        },
        "shapes": {
            "input_nhwc": [1, 512, 1024, 3],
            "reference_logit_nhwc": [1, 64, 128, 2],
            "target_hw": [64, 128],
            "mask_hw": [64, 128],
        },
        "bytes": {
            "input": INPUT_BYTES,
            "reference_logit": OUTPUT_BYTES,
            "target": TARGET_BYTES,
            "mask": TARGET_BYTES,
        },
        "reference_metrics": finalize_metrics(accum),
        "elapsed_seconds": time.time() - start,
    }

    if args.check_single:
        single_dir = Path(args.single_hw_dir)
        manifest["single_sample_check"] = [
            compare_file_bytes(out_dir / "I0000.BIN", single_dir / "input_q.bin"),
            compare_file_bytes(out_dir / "R0000.BIN", single_dir / "golden_output_q.bin"),
        ]
        failed = [x for x in manifest["single_sample_check"] if x["mismatches"] != 0]
        if failed:
            raise RuntimeError(f"single-sample compatibility check failed: {failed}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument(
        "--artifact-dir",
        default=r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep",
    )
    parser.add_argument(
        "--model-path",
        default=r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth",
    )
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument(
        "--param-src",
        default=r"D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single\param_blob.bin",
    )
    parser.add_argument(
        "--single-hw-dir",
        default=r"D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-in", type=int, default=8)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means full val set")
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available. CPU is the default to match the original golden dump.",
    )
    parser.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--write-npy", action="store_true")
    parser.add_argument("--no-copy-param", dest="copy_param", action="store_false")
    parser.add_argument("--check-single", action="store_true")
    parser.set_defaults(copy_param=True)
    args = parser.parse_args()

    for name in ("espnet_dir", "artifact_dir", "model_path", "cached_data_file", "param_src", "single_hw_dir", "out_dir"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    return args


def main() -> int:
    args = parse_args()
    manifest = export_val_dataset(args)
    print(f"Exported {manifest['num_exported']} samples to: {manifest['out_dir']}")
    print(f"Manifest: {Path(manifest['out_dir']) / 'manifest.json'}")
    metrics = manifest["reference_metrics"]
    print(f"Reference PA={metrics['pixel_accuracy']:.8f} mIoU={metrics['mIoU']:.8f}")
    if "single_sample_check" in manifest:
        print("Single-sample check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
