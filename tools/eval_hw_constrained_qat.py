#!/usr/bin/env python3
"""Evaluate the hardware-constrained QAT fake-quant ESPNet baseline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.nn.intrinsic.qat import freeze_bn_stats
from torch.quantization import disable_observer


def add_espnet(espnet_dir: Path) -> None:
    sys.path.insert(0, str(espnet_dir))
    os.chdir(espnet_dir)


def build_val_loader(cached_data_file: Path, scale_in: int, batch_size: int, num_workers: int):
    import DataSet as myDataLoader
    import Transforms as myTransforms

    with cached_data_file.open("rb") as f:
        data = pickle.load(f)

    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(scale_in),
    ])
    val_dataset = myDataLoader.MyDataset(data["valIm"], data["valAnnot"], transform=val_tf)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return val_loader


def load_fake_quant_model(args, device: torch.device):
    import Model_flattened_quantized as net
    from test_single_image_for_all import apply_hardware_quant_constraints, get_hardware_qconfig

    model = net.ESPNet_Encoder(args.classes, p=args.p, q=args.q)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.cpu().eval()
    model.qconfig = get_hardware_qconfig(qat_mode=True, symmetric_activations=True)
    model.fuse_model()

    prepared = torch.quantization.prepare_qat(model.train(), inplace=False)
    state = torch.load(Path(args.artifact_dir) / "fake_quant_state_dict.pth", map_location="cpu")
    missing, unexpected = prepared.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"fake-quant state mismatch: missing={missing}, unexpected={unexpected}")

    disable_observer(prepared)
    prepared.apply(freeze_bn_stats)
    apply_hardware_quant_constraints(prepared)
    return prepared.to(device).eval()


def remap_target(target: torch.Tensor) -> torch.Tensor:
    mapped = target.clone().long()
    mapped[((mapped < 13) | (mapped > 18)) & (mapped != 255)] = 1
    mapped[(mapped >= 13) & (mapped <= 18)] = 0
    return mapped


def evaluate(args) -> dict:
    add_espnet(Path(args.espnet_dir))
    from IOUEval_total import iouEval

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = build_val_loader(Path(args.cached_data_file), args.scale_in, args.batch_size, args.num_workers)
    model = load_fake_quant_model(args, device)
    metric = iouEval(args.classes)

    start = time.time()
    processed = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _names) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = remap_target(targets).to(device, non_blocking=True)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            metric.addBatch(preds.cpu(), targets.cpu())
            processed += inputs.shape[0]
            if args.progress_every > 0 and ((batch_idx + 1) % args.progress_every == 0 or batch_idx + 1 == len(loader)):
                overall_acc, _per_class_acc, _per_class_iu, miou = metric.getMetric()
                print(f"batch {batch_idx + 1}/{len(loader)} processed={processed} PA={overall_acc:.6f} mIoU={miou:.6f}")

    overall_acc, per_class_acc, per_class_iu, miou = metric.getMetric()
    elapsed = time.time() - start
    return {
        "artifact_dir": str(args.artifact_dir),
        "dataset": "city.val",
        "num_images": processed,
        "device": str(device),
        "pixel_accuracy": float(overall_acc),
        "mIoU": float(miou),
        "per_class_accuracy": [float(x) for x in per_class_acc],
        "per_class_IoU": [float(x) for x in per_class_iu],
        "elapsed_seconds": elapsed,
        "batch_size": args.batch_size,
        "classes": args.classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument("--artifact-dir", default=r"D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_p7_hwconv_0623")
    parser.add_argument("--model-path", default=r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth")
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument("--out-json", default=r"D:\ESP_INT8\hw_artifacts\sched_v4_p7_0702\int8_baseline_metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-in", type=int, default=8)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    metrics = evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
