#!/usr/bin/env python3
"""Export Cityscapes val samples into the ESP INT8 board-test binary layout.

The existing single-image exporter writes INPUTQ.BIN/golden_output_q.bin for one
sample. This script keeps the same NHWC signed-INT8 contract, but emits the
whole validation set with short 8.3-style names suitable for the SD-card flow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
)
from geometry_contract import DEFAULT_GEOMETRY, DeploymentGeometry, geometry_from_manifest  # noqa: E402


PARAM_BLOB_MAGIC = 0x544E4945
PARAM_BLOB_VERSION_SCHED = 4


PROFILE_CONTRACTS: dict[str, dict[str, Any]] = {
    "binary2": {
        "classes": 2,
        "param_name": "P2.BIN",
        "input_prefix": "B",
        "reference_prefix": "R",
        "target_prefix": "T",
        "mask_prefix": "M",
        "board_output_prefix": "Q",
        "metric_ignore_class": 255,
    },
    "cityscapes20": {
        "classes": 20,
        "param_name": "P20.BIN",
        "input_prefix": "C",
        "reference_prefix": "S",
        "target_prefix": "U",
        "mask_prefix": "N",
        "board_output_prefix": "D",
        "metric_ignore_class": 19,
    },
}


def profile_contract(profile: str) -> dict[str, Any]:
    try:
        return PROFILE_CONTRACTS[profile]
    except KeyError as exc:
        raise ValueError(f"unknown validation profile: {profile}") from exc


def map_target_for_profile(target: torch.Tensor, profile: str) -> torch.Tensor:
    mapped = target.clone().long()
    if profile == "binary2":
        mapped[((mapped < 13) | (mapped > 18)) & (mapped != 255)] = 1
        mapped[(mapped >= 13) & (mapped <= 18)] = 0
        return mapped
    if profile == "cityscapes20":
        invalid = ((mapped < 0) | (mapped >= 20)) & (mapped != 255)
        if bool(invalid.any()):
            values = torch.unique(mapped[invalid]).detach().cpu().tolist()
            raise ValueError(f"cityscapes20 target contains invalid train IDs: {values}")
        mapped[mapped == 255] = 19
        return mapped
    raise ValueError(f"unknown validation profile: {profile}")


def load_exported_fake_quant_model(
    args: argparse.Namespace, loader: Any, device: torch.device
) -> torch.nn.Module:
    """Rebuild the exact fake-quant graph used by the artifact exporter."""
    import Model_flattened_quantized as net
    from quantization_policy import PrecisionPolicy
    from test_single_image_for_all import (
        apply_hardware_quant_constraints,
        prepare_quantized_model,
    )

    model = net.ESPNet_Encoder(args.classes, p=args.p, q=args.q)
    source_payload = torch.load(
        args.model_path, map_location="cpu", weights_only=False
    )
    source_state = (
        source_payload.get("model", source_payload)
        if isinstance(source_payload, dict)
        else source_payload
    )
    model.load_state_dict(source_state, strict=True)

    model = prepare_quantized_model(
        model.cpu().eval(),
        calibration_loader=loader,
        qat_mode=False,
        calibration_batches=args.calibration_batches,
        symmetric_activations=True,
        convert_model=False,
        hardware_constraints=True,
        precision_policy=PrecisionPolicy.int8(),
    )
    state = torch.load(
        Path(args.artifact_dir) / "fake_quant_state_dict.pth",
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"fake-quant state mismatch: missing={missing}, unexpected={unexpected}"
        )
    apply_hardware_quant_constraints(model)
    return model.to(device).eval()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_param_header_words(path: Path) -> tuple[int, ...]:
    data = path.read_bytes()[:128]
    if len(data) < 128:
        raise ValueError(f"PARAM blob too short: {path}")
    import struct

    return struct.unpack("<32I", data)


def check_param_blob_for_sd(param_path: Path) -> dict[str, Any]:
    header = read_param_header_words(param_path)
    magic, version = header[0], header[1]
    if magic != PARAM_BLOB_MAGIC:
        raise ValueError(f"bad PARAM magic in {param_path}: 0x{magic:08x}")
    if version != PARAM_BLOB_VERSION_SCHED:
        raise ValueError(
            f"{param_path} is PARAM version {version}; P7 SD export requires v4 "
        )
    tensor_offset = int(header[10])
    tensor_count = int(header[2])
    if tensor_count <= 17:
        raise ValueError(f"PARAM tensor descriptor table is incomplete: count={tensor_count}")
    import struct

    def tensor_hw_c(tensor_id: int) -> tuple[int, int, int]:
        offset = tensor_offset + tensor_id * 16
        if offset + 16 > param_path.stat().st_size:
            raise ValueError(f"PARAM tensor descriptor {tensor_id} is out of range")
        _bank, _tag, _phys_c, _base, h, w, c, _c_offset = struct.unpack(
            "<BBHIHHHH", param_path.read_bytes()[offset : offset + 16]
        )
        return int(h), int(w), int(c)

    input_h, input_w, _input_c = tensor_hw_c(0)
    logits_h, logits_w, class_count = tensor_hw_c(17)
    geometry = DeploymentGeometry(input_h, input_w, 8)
    if (logits_h, logits_w) != (geometry.logits_height, geometry.logits_width):
        raise ValueError(
            "PARAM input/logits geometry mismatch: "
            f"input={(input_h, input_w)} logits={(logits_h, logits_w)}"
        )
    return {
        "file": "PARAM.BIN",
        "source": str(param_path),
        "version": int(version),
        "bytes": param_path.stat().st_size,
        "sha256": sha256_file(param_path),
        "geometry": geometry.to_manifest(),
        "class_count": class_count,
    }


def clean_board_outputs(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob("O*.BIN"):
        if path.is_file():
            path.unlink()
            removed += 1
    mask = out_dir / "MASK.BIN"
    if mask.exists():
        mask.unlink()
        removed += 1
    return removed


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


def update_metrics(
    accum: dict[str, Any], pred: np.ndarray, target: np.ndarray, profile: str
) -> None:
    contract = profile_contract(profile)
    valid = target != contract["metric_ignore_class"]
    pred_v = pred[valid]
    target_v = target[valid]
    accum["valid_pixels"] += int(target_v.size)
    accum["correct_pixels"] += int((pred_v == target_v).sum())
    for cls in range(contract["classes"]):
        accum["tp"][cls] += int(((pred_v == cls) & (target_v == cls)).sum())
        accum["fp"][cls] += int(((pred_v == cls) & (target_v != cls)).sum())
        accum["fn"][cls] += int(((pred_v != cls) & (target_v == cls)).sum())
        accum["support"][cls] += int((target_v == cls).sum())


def finalize_metrics(accum: dict[str, Any], profile: str) -> dict[str, Any]:
    contract = profile_contract(profile)
    per_class_acc: list[float | None] = []
    per_class_iou: list[float | None] = []
    for cls in range(contract["classes"]):
        support = accum["support"][cls]
        denom = accum["tp"][cls] + accum["fp"][cls] + accum["fn"][cls]
        per_class_acc.append(accum["tp"][cls] / support if support else None)
        per_class_iou.append(accum["tp"][cls] / denom if denom else None)

    metric_classes = [
        cls for cls in range(contract["classes"])
        if cls != contract["metric_ignore_class"]
    ]
    valid_ious = [per_class_iou[cls] for cls in metric_classes if per_class_iou[cls] is not None]
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
            for cls in range(contract["classes"])
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


def load_scale_table(path: Path) -> dict[str, tuple[float, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["name"]): (float(row["scale"]), int(row["zero_point"])) for row in rows}


def assert_observer_matches_scale_table(observer: Any, table: dict[str, tuple[float, int]], name: str) -> None:
    if name not in table:
        raise KeyError(f"scale_table.json does not contain '{name}'")
    scale = observer.scale.detach().cpu()
    zero_point = observer.zero_point.detach().cpu().to(torch.int32)
    if scale.numel() != 1 or zero_point.numel() != 1:
        raise ValueError(f"{name}: only per-tensor observer is supported")
    got = (float(scale.item()), int(zero_point.item()))
    expected = table[name]
    if abs(got[0] - expected[0]) > max(1e-8, abs(expected[0]) * 1e-5) or got[1] != expected[1]:
        raise ValueError(
            f"{name} observer qparams {got} do not match hardware scale_table {expected}; "
            "rerun D:/ESPNet/export_quantized_artifacts.py and "
            "tools/export_int8_hw_blob.py."
        )


def load_artifact_geometry(artifact_dir: Path) -> DeploymentGeometry:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = artifact_dir / "export_manifest.json"
    if not manifest_path.exists():
        return DEFAULT_GEOMETRY
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return geometry_from_manifest(manifest)


def export_val_dataset(args: argparse.Namespace) -> dict[str, Any]:
    contract = profile_contract(args.profile)
    classes = int(contract["classes"])
    geometry = load_artifact_geometry(Path(args.artifact_dir))
    requested_geometry = DeploymentGeometry(args.height, args.width, args.target_scale)
    if geometry != requested_geometry:
        raise ValueError(
            "CLI geometry disagrees with model artifact: "
            f"requested={requested_geometry.to_manifest()} artifact={geometry.to_manifest()}"
        )
    input_shape_nchw = (3, geometry.input_height, geometry.input_width)
    target_shape = (geometry.logits_height, geometry.logits_width)
    output_shape_nchw = (classes, *target_shape)
    input_bytes = geometry.input_bytes
    target_bytes = geometry.logits_height * geometry.logits_width
    output_bytes = target_bytes * classes
    out_dir = Path(args.out_dir)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_removed = clean_board_outputs(out_dir) if args.clean_board_outputs else 0

    param_info = None
    copied_audit = None
    if args.copy_param:
        param_src = Path(args.param_src)
        if not param_src.exists():
            raise FileNotFoundError(f"param blob not found: {param_src}")
        param_info = check_param_blob_for_sd(param_src)
        if param_info["geometry"] != geometry.to_manifest():
            raise ValueError(
                "PARAM geometry disagrees with model artifact: "
                f"param={param_info['geometry']} artifact={geometry.to_manifest()}"
            )
        if int(param_info["class_count"]) != classes:
            raise ValueError(
                f"PARAM class_count={param_info['class_count']} disagrees with profile={classes}"
            )
        param_info["file"] = args.param_name
        shutil.copy2(param_src, out_dir / args.param_name)
        audit_src = Path(args.param_audit) if args.param_audit else param_src.parent / "param_audit.json"
        if audit_src.exists():
            shutil.copy2(audit_src, out_dir / "param_audit.json")
            copied_audit = "param_audit.json"

    add_espnet(Path(args.espnet_dir))
    use_cuda = bool(args.cuda and torch.cuda.is_available() and not args.cpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    loader = build_val_loader(
        Path(args.cached_data_file),
        args.scale_in,
        args.batch_size,
        args.num_workers,
        geometry,
    )
    # The shared loader reconstructs the same fake-quant graph for either
    # deployment profile; only the classifier width and target mapping differ.
    args.classes = classes
    model = load_exported_fake_quant_model(args, loader, device)
    model.eval()

    input_observer = model.quant.activation_post_process
    classifier_observer = get_nested_module(model, "classifier").activation_post_process
    scale_table_path = Path(args.scale_table) if args.scale_table else Path(args.single_hw_dir) / "scale_table.json"
    if not scale_table_path.exists():
        raise FileNotFoundError(f"scale_table.json not found: {scale_table_path}")
    scale_table = load_scale_table(scale_table_path)
    assert_observer_matches_scale_table(input_observer, scale_table, "quant")
    assert_observer_matches_scale_table(classifier_observer, scale_table, "classifier")

    csv_path = out_dir / "VALMAN.CSV"
    rows: list[dict[str, Any]] = []
    accum = {
        "valid_pixels": 0,
        "correct_pixels": 0,
        "tp": [0] * classes,
        "fp": [0] * classes,
        "fn": [0] * classes,
        "support": [0] * classes,
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
            mapped_targets = map_target_for_profile(
                targets, args.profile
            ).cpu().numpy().astype(np.uint8, copy=False)

            for b in range(batch):
                if exported >= max_items:
                    break

                idx = args.first_index + exported
                input_name = short_bin(contract["input_prefix"], idx)
                ref_name = short_bin(contract["reference_prefix"], idx)
                target_name = short_bin(contract["target_prefix"], idx)
                mask_name = short_bin(contract["mask_prefix"], idx)

                input_nhwc = nchw_to_nhwc_i8(q_inputs[b], input_shape_nchw, input_name)
                ref_nhwc = nchw_to_nhwc_i8(q_outputs[b], output_shape_nchw, ref_name)
                target = np.ascontiguousarray(mapped_targets[b])
                if target.shape != target_shape:
                    raise ValueError(f"{target_name} shape={target.shape}, expected={target_shape}")
                mask = np.ascontiguousarray(np.argmax(ref_nhwc, axis=-1).astype(np.uint8))

                input_nhwc.tofile(out_dir / input_name)
                ref_nhwc.tofile(out_dir / ref_name)
                target.tofile(out_dir / target_name)
                mask.tofile(out_dir / mask_name)
                input_path = out_dir / input_name
                ref_path = out_dir / ref_name
                target_path = out_dir / target_name
                mask_path = out_dir / mask_name

                if args.write_npy:
                    np.save(out_dir / input_name.replace(".BIN", ".npy"), input_nhwc)
                    np.save(out_dir / ref_name.replace(".BIN", ".npy"), ref_nhwc)
                    np.save(out_dir / target_name.replace(".BIN", ".npy"), target)
                    np.save(out_dir / mask_name.replace(".BIN", ".npy"), mask)

                update_metrics(accum, mask, target, args.profile)
                image_name = names[b] if isinstance(names, (list, tuple)) else names
                rows.append(
                    {
                        "index": idx,
                        "input": input_name,
                        "ref": ref_name,
                        "target": target_name,
                        "mask": mask_name,
                        "board_output": short_bin(contract["board_output_prefix"], idx),
                        "image": str(image_name),
                        "input_bytes": input_bytes,
                        "ref_bytes": output_bytes,
                        "target_bytes": target_bytes,
                        "mask_bytes": target_bytes,
                        "input_sha256": sha256_file(input_path),
                        "ref_sha256": sha256_file(ref_path),
                        "target_sha256": sha256_file(target_path),
                        "mask_sha256": sha256_file(mask_path),
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
            "board_output",
            "image",
            "input_bytes",
            "ref_bytes",
            "target_bytes",
            "mask_bytes",
            "input_sha256",
            "target_sha256",
            "ref_sha256",
            "mask_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "format": "ESP_INT8_VAL_HW_DATASET_V3",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "espnet_dir": str(args.espnet_dir),
        "artifact_dir": str(args.artifact_dir),
        "cached_data_file": str(args.cached_data_file),
        "out_dir": str(out_dir),
        "num_exported": exported,
        "num_val_total": total,
        "batch_size": args.batch_size,
        "device": str(device),
        "geometry": geometry.to_manifest(),
        "param_blob": (
            {
                **param_info,
                "audit_file": copied_audit,
            }
            if param_info
            else None
        ),
        "board_contract": {
            "profile": args.profile,
            "input": (
                f"NHWC int8 signed, {geometry.input_height}x{geometry.input_width}x3, "
                f"{contract['input_prefix']}%04d.BIN"
            ),
            "output": (
                f"full-res uint8 mask, {geometry.input_height}x{geometry.input_width}, "
                f"{contract['board_output_prefix']}%04d.BIN"
            ),
            "target": (
                f"low-res uint8 target, {geometry.logits_height}x{geometry.logits_width}, "
                f"{contract['target_prefix']}%04d.BIN"
            ),
            "reference_logit": (
                f"low-res NHWC int8 logits, {geometry.logits_height}x"
                f"{geometry.logits_width}x{classes}, {contract['reference_prefix']}%04d.BIN"
            ),
            "reference_mask": (
                f"low-res uint8 mask, {geometry.logits_height}x{geometry.logits_width}, "
                f"{contract['mask_prefix']}%04d.BIN"
            ),
        },
        "stale_output_cleaned": bool(args.clean_board_outputs),
        "stale_output_removed": stale_removed,
        "files": {
            "param": args.param_name if args.copy_param else None,
            "manifest_csv": "VALMAN.CSV",
            "input_pattern": f"{contract['input_prefix']}%04d.BIN",
            "reference_logit_pattern": f"{contract['reference_prefix']}%04d.BIN",
            "target_pattern": f"{contract['target_prefix']}%04d.BIN",
            "reference_mask_pattern": f"{contract['mask_prefix']}%04d.BIN",
            "board_output_pattern": f"{contract['board_output_prefix']}%04d.BIN",
            "filehash_csv": "FILEHASH.CSV",
        },
        "shapes": {
            "input_nhwc": [1, geometry.input_height, geometry.input_width, 3],
            "reference_logit_nhwc": [1, geometry.logits_height, geometry.logits_width, classes],
            "target_hw": [geometry.logits_height, geometry.logits_width],
            "mask_hw": [geometry.logits_height, geometry.logits_width],
        },
        "bytes": {
            "input": input_bytes,
            "reference_logit": output_bytes,
            "target": target_bytes,
            "mask": target_bytes,
        },
        "reference_metrics": finalize_metrics(accum, args.profile),
        "elapsed_seconds": time.time() - start,
    }

    if args.check_single:
        single_dir = Path(args.single_hw_dir)
        manifest["single_sample_check"] = [
            compare_file_bytes(
                out_dir / short_bin(contract["input_prefix"], 0),
                single_dir / "input_q.bin",
            ),
            compare_file_bytes(
                out_dir / short_bin(contract["reference_prefix"], 0),
                single_dir / "golden_output_q.bin",
            ),
        ]
        failed = [x for x in manifest["single_sample_check"] if x["mismatches"] != 0]
        if failed:
            raise RuntimeError(f"single-sample compatibility check failed: {failed}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "SDMANIFEST.JSON").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "FILEHASH.CSV").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "bytes", "sha256", "role"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if args.copy_param:
            writer.writerow({
                "file": args.param_name,
                "bytes": (out_dir / args.param_name).stat().st_size,
                "sha256": sha256_file(out_dir / args.param_name),
                "role": "param",
            })
        for row in rows:
            writer.writerow({"file": row["input"], "bytes": row["input_bytes"], "sha256": row["input_sha256"], "role": "input"})
            writer.writerow({"file": row["target"], "bytes": row["target_bytes"], "sha256": row["target_sha256"], "role": "target"})
            writer.writerow({"file": row["ref"], "bytes": row["ref_bytes"], "sha256": row["ref_sha256"], "role": "reference_logit"})
            writer.writerow({"file": row["mask"], "bytes": row["mask_bytes"], "sha256": row["mask_sha256"], "role": "reference_mask"})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument(
        "--artifact-dir",
        default=r"D:\ESP_INT8\hw_artifacts\binary2_int8_0809_model",
    )
    parser.add_argument(
        "--model-path",
        default=r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth",
    )
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument(
        "--param-src",
        default=r"D:\ESP_INT8\hw_artifacts\binary2_int8_0809\param_blob.bin",
    )
    parser.add_argument(
        "--single-hw-dir",
        default=r"D:\ESP_INT8\hw_artifacts\binary2_int8_0809",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\ESP_INT8\hw_artifacts\val_binary2_0809",
    )
    parser.add_argument("--param-audit", default="")
    parser.add_argument("--scale-table", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-in", type=int, default=8)
    parser.add_argument("--height", type=int, default=DEFAULT_GEOMETRY.input_height)
    parser.add_argument("--width", type=int, default=DEFAULT_GEOMETRY.input_width)
    parser.add_argument("--target-scale", type=int, default=DEFAULT_GEOMETRY.target_scale)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONTRACTS), default="binary2")
    parser.add_argument("--param-name", default="P2.BIN")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=11)
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
    parser.add_argument("--clean-board-outputs", action="store_true")
    parser.add_argument("--write-npy", action="store_true")
    parser.add_argument("--no-copy-param", dest="copy_param", action="store_false")
    parser.add_argument("--check-single", action="store_true")
    parser.set_defaults(copy_param=True)
    args = parser.parse_args()

    for name in (
        "espnet_dir",
        "artifact_dir",
        "model_path",
        "cached_data_file",
        "param_src",
        "single_hw_dir",
        "out_dir",
        "param_audit",
        "scale_table",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).resolve()))
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
