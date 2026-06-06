#!/usr/bin/env python3
"""Evaluate full-resolution mask outputs produced by the INT8 NPU.

This script matches the PERF125-UPFULL output contract:

- each board output file is a 512x1024 uint8 mask
- class ids are already argmax results: 0=target, 1=background
- metrics are computed against the original 512x1024 validation labels
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hw_constrained_qat import add_espnet, remap_target  # noqa: E402


MASK_SHAPE = (512, 1024)
MASK_BYTES = MASK_SHAPE[0] * MASK_SHAPE[1]


def metric_dict(metric: Any) -> dict[str, Any]:
    overall_acc, per_class_acc, per_class_iu, miou = metric.getMetric()
    return {
        "pixel_accuracy": float(overall_acc),
        "mIoU": float(miou),
        "per_class_accuracy": [float(x) for x in per_class_acc],
        "per_class_IoU": [float(x) for x in per_class_iu],
    }


def build_fullres_val_dataset(espnet_dir: Path, cached_data_file: Path):
    add_espnet(espnet_dir)
    import DataSet as myDataLoader
    import Transforms as myTransforms

    with cached_data_file.open("rb") as f:
        data = pickle.load(f)

    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(1),
    ])
    return myDataLoader.MyDataset(data["valIm"], data["valAnnot"], transform=val_tf)


def load_mask(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != MASK_BYTES:
        raise ValueError(f"{path} has {data.size} bytes, expected {MASK_BYTES}")
    return data.reshape(MASK_SHAPE)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    add_espnet(Path(args.espnet_dir))
    from IOUEval_total import iouEval

    output_dir = Path(args.output_dir)
    dataset = build_fullres_val_dataset(Path(args.espnet_dir), Path(args.cached_data_file))
    metric = iouEval(args.classes)
    missing: list[str] = []
    processed = 0

    limit = args.limit if args.limit > 0 else len(dataset)
    for index in range(args.start_index, min(args.start_index + limit, len(dataset))):
        out_path = output_dir / (args.output_pattern % index)
        if not out_path.exists():
            missing.append(str(out_path))
            if args.allow_missing:
                continue
            raise FileNotFoundError(out_path)

        _inputs, target, _name = dataset[index]
        target = remap_target(target.unsqueeze(0)).squeeze(0)
        pred = torch.from_numpy(load_mask(out_path)).long()
        metric.addBatch(pred, target)
        processed += 1

        if args.progress_every > 0 and (
            processed % args.progress_every == 0 or processed == limit
        ):
            metrics = metric_dict(metric)
            print(
                f"processed={processed} PA={metrics['pixel_accuracy']:.8f} "
                f"mIoU={metrics['mIoU']:.8f}"
            )

    return {
        "output_dir": str(output_dir),
        "output_pattern": args.output_pattern,
        "start_index": args.start_index,
        "processed": processed,
        "missing": missing,
        "metrics": metric_dict(metric),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument("--output-dir", default=r"E:\\")
    parser.add_argument("--output-pattern", default="O%04d.BIN")
    parser.add_argument("--out-json", default=r"D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val\board_val_fullres_mask_metrics.json")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    result = evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
