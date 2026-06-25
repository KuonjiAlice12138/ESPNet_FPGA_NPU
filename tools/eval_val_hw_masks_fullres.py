#!/usr/bin/env python3
"""Evaluate full-resolution mask outputs produced by the INT8 NPU.

This script matches the PERF125-UPFULL output contract:

- each board output file is a 512x1024 uint8 mask
- class ids are already argmax results: 0=target, 1=background
- metrics are computed against the original 512x1024 validation labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_eval_contract(output_pattern: str, allow_reference_mask_eval: bool = False) -> None:
    if output_pattern == "M%04d.BIN" and not allow_reference_mask_eval:
        raise ValueError(
            "M%04d.BIN is the low-res reference mask pattern, not board output. "
            "Use O%04d.BIN or pass --allow-reference-mask-eval explicitly."
        )
    if output_pattern != "O%04d.BIN" and not allow_reference_mask_eval:
        raise ValueError(
            f"Unexpected board output pattern '{output_pattern}'. "
            "The P7 full-res board contract uses O%04d.BIN by default."
        )


def validate_output_set(
    output_dir: Path,
    pattern: str,
    start: int,
    limit: int,
    expected_bytes: int = MASK_BYTES,
) -> dict[str, Any]:
    missing: list[str] = []
    wrong_size: list[dict[str, Any]] = []
    present: list[str] = []
    for index in range(start, start + limit):
        path = output_dir / (pattern % index)
        if not path.exists():
            missing.append(str(path))
            continue
        size = path.stat().st_size
        if size != expected_bytes:
            wrong_size.append({"file": str(path), "bytes": size, "expected_bytes": expected_bytes})
            continue
        present.append(str(path))
    return {"missing": missing, "wrong_size": wrong_size, "present": present}


def parse_timing_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"samples": None, "total_ticks": None, "avg_ms": None}
    text = path.read_text(encoding="utf-8", errors="ignore")
    samples = None
    total_ticks = None
    avg_ms = None
    m = re.search(r"samples=(\d+)\s+total_cycles=(\d+).*?avg_ms=(\d+)", text, re.S)
    if m:
        samples = int(m.group(1))
        total_ticks = int(m.group(2))
        avg_ms = int(m.group(3))
    return {"samples": samples, "total_ticks": total_ticks, "avg_ms": avg_ms}


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
    wrong_size: list[dict[str, Any]] = []
    processed = 0

    limit = args.limit if args.limit > 0 else len(dataset)
    limit = min(limit, len(dataset) - args.start_index)
    validate_eval_contract(args.output_pattern, args.allow_reference_mask_eval)
    output_set = validate_output_set(output_dir, args.output_pattern, args.start_index, limit)
    missing = output_set["missing"]
    wrong_size = output_set["wrong_size"]
    if (missing or wrong_size) and not args.allow_missing:
        raise FileNotFoundError(
            f"board output set incomplete: missing={len(missing)} wrong_size={len(wrong_size)}"
        )

    for index in range(args.start_index, min(args.start_index + limit, len(dataset))):
        out_path = output_dir / (args.output_pattern % index)
        if not out_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(out_path)
        if out_path.stat().st_size != MASK_BYTES:
            if args.allow_missing:
                continue
            raise ValueError(f"{out_path} has {out_path.stat().st_size} bytes, expected {MASK_BYTES}")

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

    sd_manifest = None
    if args.sd_manifest:
        sd_manifest = json.loads(Path(args.sd_manifest).read_text(encoding="utf-8"))
    param_audit = None
    if args.param_audit:
        param_audit = json.loads(Path(args.param_audit).read_text(encoding="utf-8"))

    return {
        "output_dir": str(output_dir),
        "output_pattern": args.output_pattern,
        "start_index": args.start_index,
        "num_expected": limit,
        "processed": processed,
        "missing": missing,
        "wrong_size": wrong_size,
        "param_blob_sha256": (
            sd_manifest.get("param_blob", {}).get("sha256") if isinstance(sd_manifest, dict) else None
        ),
        "param_version": (
            sd_manifest.get("param_blob", {}).get("version") if isinstance(sd_manifest, dict) else None
        ),
        "param_audit_format": (
            param_audit.get("format") if isinstance(param_audit, dict) else None
        ),
        "timing": parse_timing_log(Path(args.timing_log) if args.timing_log else None),
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
    parser.add_argument("--allow-reference-mask-eval", action="store_true")
    parser.add_argument("--sd-manifest", default="")
    parser.add_argument("--param-audit", default="")
    parser.add_argument("--timing-log", default="")
    args = parser.parse_args()

    result = evaluate(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
