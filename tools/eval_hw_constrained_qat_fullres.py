#!/usr/bin/env python3
"""Evaluate hardware-constrained QAT ESPNet encoder at low and full resolution.

The existing INT8 baseline evaluates ESPNet_Encoder in its native 1/8 output
resolution. This script keeps that metric for compatibility and adds two
full-resolution metrics:

1. bilinear upsample logits to the original label size, then argmax
2. argmax at low resolution, then nearest-neighbor upsample the mask

The first is the preferred model-side full-resolution metric. The second is a
hardware-friendly reference for a cheap final mask upsampler.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hw_constrained_qat import add_espnet, load_fake_quant_model, remap_target  # noqa: E402


def build_fullres_val_loader(cached_data_file: Path, batch_size: int, num_workers: int):
    import DataSet as myDataLoader
    import Transforms as myTransforms

    with cached_data_file.open("rb") as f:
        data = pickle.load(f)

    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(1),
    ])
    val_dataset = myDataLoader.MyDataset(data["valIm"], data["valAnnot"], transform=val_tf)
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metric_dict(metric: Any) -> dict[str, Any]:
    overall_acc, per_class_acc, per_class_iu, miou = metric.getMetric()
    return {
        "pixel_accuracy": float(overall_acc),
        "mIoU": float(miou),
        "per_class_accuracy": [float(x) for x in per_class_acc],
        "per_class_IoU": [float(x) for x in per_class_iu],
    }


def downsample_target_nearest(target_full: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    target = target_full.unsqueeze(1).float()
    target = F.interpolate(target, size=size_hw, mode="nearest")
    return target.squeeze(1).long()


def upsample_mask_nearest(mask_low: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    mask = mask_low.unsqueeze(1).float()
    mask = F.interpolate(mask, size=size_hw, mode="nearest")
    return mask.squeeze(1).long()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    add_espnet(Path(args.espnet_dir))
    from IOUEval_total import iouEval

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = build_fullres_val_loader(Path(args.cached_data_file), args.batch_size, args.num_workers)
    model = load_fake_quant_model(args, device)

    lowres_metric = iouEval(args.classes)
    fullres_bilinear_metric = iouEval(args.classes)
    fullres_nearest_metric = iouEval(args.classes)

    processed = 0
    input_shape = None
    output_shape = None
    target_shape = None
    start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _names) in enumerate(loader):
            if args.limit > 0 and processed >= args.limit:
                break

            if args.limit > 0:
                remaining = args.limit - processed
                if inputs.shape[0] > remaining:
                    inputs = inputs[:remaining]
                    targets = targets[:remaining]

            inputs = inputs.to(device, non_blocking=True)
            target_full = remap_target(targets).to(device, non_blocking=True)

            outputs = model(inputs)
            low_size = (int(outputs.shape[-2]), int(outputs.shape[-1]))
            full_size = (int(target_full.shape[-2]), int(target_full.shape[-1]))

            low_pred = torch.argmax(outputs, dim=1)
            low_target = downsample_target_nearest(target_full, low_size)

            full_logits = F.interpolate(outputs.float(), size=full_size, mode="bilinear", align_corners=False)
            full_pred_bilinear = torch.argmax(full_logits, dim=1)
            full_pred_nearest = upsample_mask_nearest(low_pred, full_size)

            lowres_metric.addBatch(low_pred.cpu(), low_target.cpu())
            fullres_bilinear_metric.addBatch(full_pred_bilinear.cpu(), target_full.cpu())
            fullres_nearest_metric.addBatch(full_pred_nearest.cpu(), target_full.cpu())

            processed += int(inputs.shape[0])
            input_shape = list(inputs.shape)
            output_shape = list(outputs.shape)
            target_shape = list(target_full.shape)

            if args.progress_every > 0 and (
                (batch_idx + 1) % args.progress_every == 0
                or batch_idx + 1 == len(loader)
                or (args.limit > 0 and processed >= args.limit)
            ):
                low = metric_dict(lowres_metric)
                full = metric_dict(fullres_bilinear_metric)
                print(
                    f"batch {batch_idx + 1}/{len(loader)} processed={processed} "
                    f"low PA={low['pixel_accuracy']:.6f} low mIoU={low['mIoU']:.6f} "
                    f"full PA={full['pixel_accuracy']:.6f} full mIoU={full['mIoU']:.6f}"
                )

    elapsed = time.time() - start
    return {
        "artifact_dir": str(args.artifact_dir),
        "model_path": str(args.model_path),
        "cached_data_file": str(args.cached_data_file),
        "dataset": "city.val",
        "num_images": processed,
        "device": str(device),
        "batch_size": args.batch_size,
        "classes": args.classes,
        "input_shape_last_batch": input_shape,
        "encoder_output_shape_last_batch": output_shape,
        "fullres_target_shape_last_batch": target_shape,
        "metrics": {
            "lowres_argmax": metric_dict(lowres_metric),
            "fullres_bilinear_logits_argmax": metric_dict(fullres_bilinear_metric),
            "fullres_nearest_mask": metric_dict(fullres_nearest_metric),
        },
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument("--artifact-dir", default=r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623")
    parser.add_argument("--model-path", default=r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth")
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument(
        "--out-json",
        default=r"D:\ESP_INT8\hw_artifacts\sched_v4_p7_0702\int8_baseline_metrics_fullres.json",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    result = evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
