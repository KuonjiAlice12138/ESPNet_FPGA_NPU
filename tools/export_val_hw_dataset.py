#!/usr/bin/env python3
"""Export Cityscapes val samples into the ESP INT8 board-test binary layout.

The existing single-image exporter writes INPUTQ.BIN/golden_output_q.bin for one
sample. This script keeps the same NHWC signed-INT8 contract, but emits the
whole validation set with short 8.3-style names suitable for the SD-card flow.

Unified export uses --runtime-artifact MODEL.JSN --source-manifest sources.json.
It never reconstructs/calibrates a model or guesses how to generate raw inputs.
The source JSON contract is P7_PREQUANTIZED_VAL_SOURCES_V1, with:
  runtime_model_sha256: SHA256 of the exact MODEL.JSN
  runtime: the canonical MODEL.JSN runtime dictionary
  preprocessing: the unchanged MODEL.JSN preprocessing dictionary
  samples: [{name, input: {file, bytes, sha256}, target: {file, bytes, sha256},
             target_shape_hw: [H, W], target_encoding, center_square_source}]
Files are relative, tightly packed binary arrays: source HWC signed INT8 and
target HW uint8. target_encoding is cityscapes_train_ids (original GT, sampled
with the explicitly registered center_square_source boolean and shared exact
half-pixel/ROI rules) or mapped_output (already on the declared mask grid).
The producer must bind the source bytes and GT transform to this exact model;
raw PNG/JPEG generation or unregistered preprocessing is not supported here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zlib
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
from geometry_contract import (  # noqa: E402
    DEFAULT_GEOMETRY, DeploymentGeometry, geometry_from_manifest,
    runtime_selection_from_manifest, select_runtime_input, runtime_mask_from_logits,
    runtime_target_view,
)
from eval_single_hw_outputs import (  # noqa: E402
    load_runtime_array, load_runtime_contract, map_runtime_target, runtime_artifact_directory,
)


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


def map_target_for_profile(target: torch.Tensor, profile: str, *, already_mapped: bool = False) -> torch.Tensor:
    mapped = target.clone().long()
    if profile == "binary2":
        if already_mapped:
            if bool(((mapped != 0) & (mapped != 1) & (mapped != 255)).any()):
                raise ValueError("mapped binary2 labels must be 0, 1 or 255")
            return mapped
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

    model = net.ESPNet_Encoder(args.classes, p=args.p, q=args.q,
                              input_channels=args.input_channels)
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

    input_h, input_w, input_c = tensor_hw_c(0)
    logits_h, logits_w, class_count = tensor_hw_c(17)
    geometry = DeploymentGeometry(input_h, input_w, 8, input_channels=input_c)
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


def build_artifact_val_loader(
    espnet_dir: Path, cached_data_file: Path, geometry: DeploymentGeometry,
    preprocessing: dict, batch_size: int, workers: int, profile: str,
    target_scale: int = 8,
):
    add_espnet(espnet_dir)
    if geometry.input_channels == 3:
        return build_val_loader(cached_data_file, target_scale, batch_size, workers, geometry)
    if (preprocessing.get("image_mode") != "gray" or
            preprocessing.get("center_square_crop") is not True):
        raise ValueError("C1 export requires explicit gray/center-square-crop preprocessing")
    import pickle
    from train_cityscapes20.main import build_loaders
    from deployment_profiles import get_deployment_profile
    with cached_data_file.open("rb") as handle:
        data = pickle.load(handle)
    expected_hash = preprocessing.get("cache_sha256")
    if expected_hash is not None and sha256_file(cached_data_file) != expected_hash:
        raise ValueError("validation cache hash disagrees with QAT/export manifest")
    loader_args = argparse.Namespace(
        height=geometry.input_height, width=geometry.input_width, target_scale=8,
        image_mode="gray", center_square_crop=True, device="cpu", workers=workers,
        batch_size=batch_size, val_batch_size=batch_size,
    )
    _, loader = build_loaders(loader_args, data, get_deployment_profile(profile))
    return loader


def read_bound_source_file(root: Path, record: dict[str, Any]) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("file"), str):
        raise ValueError("source records require file, bytes and sha256 identities")
    relative = Path(record["file"])
    path = (root / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root)
            or path.suffix.upper() != ".BIN" or not path.is_file()):
        raise ValueError(f"source record must reference an existing relative binary file: {relative}")
    if (type(record.get("bytes")) is not int or record["bytes"] != path.stat().st_size
            or record.get("sha256") != sha256_file(path)):
        raise ValueError(f"prequantized source/GT identity mismatch: {path}")
    if "crc32" in record and record["crc32"] != f"{zlib.crc32(path.read_bytes()) & 0xffffffff:08X}":
        raise ValueError(f"prequantized source/GT CRC mismatch: {path}")
    return path


def load_bound_source_sample(root, sample, selection, deployment):
    if not isinstance(sample, dict) or not isinstance(sample.get("name"), str) or not sample["name"]:
        raise ValueError("each prequantized validation sample requires a nonempty name")
    input_path = read_bound_source_file(root, sample.get("input"))
    target_path = read_bound_source_file(root, sample.get("target"))
    source = load_runtime_array(input_path, np.int8, selection.source_shape)
    shape = sample.get("target_shape_hw")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError(f"{sample['name']}: target_shape_hw must contain two positive integers")
    raw = load_runtime_array(target_path, np.uint8, tuple(shape))
    encoding = sample.get("target_encoding")
    if encoding == "mapped_output":
        if tuple(shape) != selection.mask_shape:
            raise ValueError(f"{sample['name']}: mapped_output target must use registered mask grid {selection.mask_shape}")
        target = map_runtime_target(raw, deployment, encoding=encoding)
    elif encoding == "cityscapes_train_ids":
        center = sample.get("center_square_source")
        if not isinstance(center, bool):
            raise ValueError(f"{sample['name']}: original GT requires explicit center_square_source; unsupported source transform")
        mapped = map_runtime_target(raw, deployment, encoding=encoding)
        target = runtime_target_view(mapped, selection, center_square_source=center)
    else:
        raise ValueError(f"{sample['name']}: unsupported target_encoding {encoding!r}; use cityscapes_train_ids or mapped_output")
    if not np.any(target != deployment.ignore_target):
        raise ValueError(f"{sample['name']}: registered output GT has no valid pixels")
    return source, target


def export_runtime_val_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Export same-model prequantized sources with PARAM integer references only."""
    from hw_param_replay import ParamBlob, replay_prefix

    directory = runtime_artifact_directory(args.runtime_artifact)
    model, selection, deployment = load_runtime_contract(directory)
    source_manifest_path = getattr(args, "source_manifest", "")
    if not source_manifest_path:
        raise ValueError("--runtime-artifact requires --source-manifest with model-bound prequantized inputs; raw source transforms cannot be inferred from encoder preprocessing")
    source_manifest_path = Path(source_manifest_path).resolve()
    sources = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(sources, dict) or sources.get("format") != "P7_PREQUANTIZED_VAL_SOURCES_V1":
        raise ValueError("--source-manifest must be P7_PREQUANTIZED_VAL_SOURCES_V1, not an unregistered raw-image dataset")
    if sources.get("runtime_model_sha256") != sha256_file(directory / "MODEL.JSN"):
        raise ValueError("source manifest is not bound to this MODEL.JSN model/quantization identity")
    if runtime_selection_from_manifest(sources) != selection:
        raise ValueError("source manifest runtime/source/ROI/mask selection differs from MODEL.JSN")
    if "preprocessing" not in sources or sources["preprocessing"] != model.get("preprocessing", {}):
        raise ValueError("source manifest preprocessing differs from the frozen model")
    samples = sources.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source manifest requires a nonempty validation samples list")
    limit = getattr(args, "limit", 0)
    first = getattr(args, "first_index", 0)
    if limit < 0:
        raise ValueError("--limit must be nonnegative (0 means all source records)")
    samples_to_export = samples[:limit] if limit else samples
    short_bin("I", first)
    short_bin("I", first + len(samples_to_export) - 1)
    if getattr(args, "clean", False) or getattr(args, "clean_board_outputs", False):
        raise ValueError("unified export requires a new/empty output directory; clean flags are legacy-only")
    out_dir = Path(args.out_dir).resolve()
    source_root = source_manifest_path.parent
    if (out_dir == directory or out_dir.is_relative_to(directory)
            or directory.is_relative_to(out_dir) or source_manifest_path.is_relative_to(out_dir)):
        raise ValueError("unified output must be separate from immutable runtime/source artifacts")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"unified output must be a new/empty directory: {out_dir}")
    # Preflight every requested identity and GT transform before writing any dataset.
    for sample in samples_to_export:
        load_bound_source_sample(source_root, sample, selection, deployment)

    contract = profile_contract(deployment.profile_name)
    classes = selection.classes
    g = selection.profile.geometry
    shapes = {"input_nhwc": [1, *selection.source_shape],
              "encoder_nhwc": [1, *g.input_shape_nhwc],
              "reference_logit_nhwc": [1, *g.lowres_shape(classes)],
              "target_hw": list(selection.mask_shape), "mask_hw": list(selection.mask_shape)}
    sizes = {"input": selection.input_bytes, "encoder": g.input_bytes,
             "reference_logit": g.logits_height * g.logits_width * classes,
             "target": selection.mask_bytes, "mask": selection.mask_bytes}
    accum = {"valid_pixels": 0, "correct_pixels": 0,
             **{key: [0] * classes for key in ("tp", "fp", "fn", "support")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_param = getattr(args, "copy_param", True)
    if copy_param:
        shutil.copy2(directory / "PARAM.BIN", out_dir / "PARAM.BIN")
    blob = ParamBlob(directory / "PARAM.BIN")
    rows = []
    hashes = []

    def record_file(path, role):
        digest = sha256_file(path)
        hashes.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest,
                       "crc32": f"{zlib.crc32(path.read_bytes()) & 0xffffffff:08X}", "role": role})
        return digest

    if copy_param:
        record_file(out_dir / "PARAM.BIN", "param")
    start = time.time()
    for offset, sample in enumerate(samples_to_export):
        source, target = load_bound_source_sample(source_root, sample, selection, deployment)
        encoder = select_runtime_input(source, selection)
        index = first + offset
        names = {"input": short_bin(contract["input_prefix"], index),
                 "encoder": short_bin("E", index),
                 "ref": short_bin(contract["reference_prefix"], index),
                 "target": short_bin(contract["target_prefix"], index),
                 "mask": short_bin(contract["mask_prefix"], index)}
        source.tofile(out_dir / names["input"])
        encoder.tofile(out_dir / names["encoder"])
        logits = replay_prefix(blob, out_dir / names["encoder"], 74).tensors[17]
        if logits.dtype != np.int8 or logits.shape != g.lowres_shape(classes):
            raise ValueError(f"integer replay logits must be HWC signed INT8 {g.lowres_shape(classes)}, got {logits.shape}/{logits.dtype}")
        mask = runtime_mask_from_logits(logits, selection)
        for role, array in (("ref", logits), ("target", target), ("mask", mask)):
            np.ascontiguousarray(array).tofile(out_dir / names[role])
        if getattr(args, "write_npy", False):
            for role, array in (("input", source), ("encoder", encoder), ("ref", logits), ("target", target), ("mask", mask)):
                np.save(out_dir / Path(names[role]).with_suffix(".npy"), array)
        row = {"index": index, **names, "board_output": short_bin(contract["board_output_prefix"], index),
               "image": sample["name"], "target_encoding": sample["target_encoding"],
               "center_square_source": sample.get("center_square_source"),
               "source_input_sha256": sample["input"]["sha256"],
               "source_target_sha256": sample["target"]["sha256"]}
        for role, filename in names.items():
            path = out_dir / filename
            row[f"{role}_bytes"] = path.stat().st_size
            row[f"{role}_sha256"] = record_file(path, role)
        rows.append(row)
        update_metrics(accum, mask, target, deployment.profile_name)
        every = getattr(args, "progress_every", 0)
        if every > 0 and ((offset + 1) % every == 0 or offset + 1 == len(samples_to_export)):
            print(f"exported {offset + 1}/{len(samples_to_export)} prequantized runtime samples")

    manifest = {
        "format": "P7_UNIFIED_INT8_VAL_DATASET_V1", "out_dir": str(out_dir),
        "num_exported": len(rows), "num_val_total": len(samples),
        "runtime": selection.to_manifest(), "deployment": model["deployment"],
        "geometry": g.to_manifest(), "preprocessing": model.get("preprocessing", {}),
        "runtime_artifact": str(directory), "runtime_model_sha256": sha256_file(directory / "MODEL.JSN"),
        "param_sha256": model["files"]["param"]["sha256"],
        "source_manifest": str(source_manifest_path), "source_manifest_sha256": sha256_file(source_manifest_path),
        "golden_authority": "PARAM_V4_INTEGER_REPLAY", "device": "cpu",
        "board_contract": {"input": "source_HWC_int8_prequantized", "encoder": "encoder_HWC_int8_selected",
                           "reference_logit": "logits_HWC_int8", "reference_mask": "mask_HW_uint8_logits_then_argmax",
                           "target": "mapped_target_HW_uint8_registered_output_view",
                           "output": "board_mask_HW_uint8_registered_output_view"},
        "files": {"param": "PARAM.BIN" if copy_param else None, "manifest_csv": "VALMAN.CSV",
                  "filehash_csv": "FILEHASH.CSV", "encoder_pattern": "E%04d.BIN",
                  **{name: f"{contract[key]}%04d.BIN" for name, key in (
                      ("input_pattern", "input_prefix"), ("reference_logit_pattern", "reference_prefix"),
                      ("target_pattern", "target_prefix"), ("reference_mask_pattern", "mask_prefix"),
                      ("board_output_pattern", "board_output_prefix"))}},
        "shapes": shapes, "bytes": sizes, "samples": rows,
        "reference_metrics": finalize_metrics(accum, deployment.profile_name),
        "accuracy_status": "software integer references only; no board accuracy claim",
        "elapsed_seconds": time.time() - start,
    }
    if getattr(args, "check_single", False):
        originals = {"input": "INPUTQ.BIN", "encoder": "ENCODER.BIN", "ref": "LOGITS.BIN", "mask": "MASK.BIN"}
        manifest["single_sample_check"] = [compare_file_bytes(out_dir / rows[0][role], directory / filename)
                                           for role, filename in originals.items()]
        if any(check["mismatches"] for check in manifest["single_sample_check"]):
            raise ValueError("first validation sample does not match the frozen runtime INPUTQ/ENCODER/LOGITS/MASK")
    for filename, entries in (("VALMAN.CSV", rows), ("FILEHASH.CSV", hashes)):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    for filename in ("manifest.json", "SDMANIFEST.JSON"):
        (out_dir / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def publish_board_validation_suite(spec_path: Path, out_dir: Path) -> dict[str, Any]:
    """Package already exported, model-bound VAL500 sets; never train or change geometry."""
    from geometry_contract import RUNTIME_PROFILES, runtime_selection

    spec_path = Path(spec_path).resolve()
    out_dir = Path(out_dir).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    entries = spec.get("datasets")
    if (spec.get("format") != "P7_UNIFIED_VAL500_SUITE_V1" or not isinstance(entries, list)
            or not 1 <= len(entries) <= 48):
        raise ValueError("suite requires P7_UNIFIED_VAL500_SUITE_V1 and 1..48 dataset entries")
    if out_dir.exists():
        raise ValueError("board suite output must be a new directory; existing results are never overwritten")
    prepared, keys, selections = [], set(), set()
    ordered_images = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each suite entry requires key and dataset")
        key = entry.get("key", "")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z0-9_]{1,8}", key) or key in keys:
            raise ValueError("case keys must be unique uppercase 8.3 directory names")
        keys.add(key)
        if not isinstance(entry.get("dataset"), str) or not entry["dataset"]:
            raise ValueError(f"{key}: dataset manifest path is required")
        dataset_path = (spec_path.parent / entry["dataset"]).resolve()
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        root = dataset_path.parent
        model_root = runtime_artifact_directory(Path(data["runtime_artifact"]))
        model, selection, _ = load_runtime_contract(model_root)
        signature = (selection.profile.profile_id, selection.classes, selection.src_mode, selection.mask_mode)
        if signature in selections:
            raise ValueError(f"{key}: duplicate runtime case in validation suite")
        selections.add(signature)
        if (data.get("format") != "P7_UNIFIED_INT8_VAL_DATASET_V1"
                or runtime_selection_from_manifest(data) != selection
                or data.get("runtime_model_sha256") != sha256_file(model_root / "MODEL.JSN")
                or data.get("param_sha256") != model["files"]["param"]["sha256"]
                or data.get("preprocessing") != model.get("preprocessing", {})
                or data.get("deployment") != model["deployment"]):
            raise ValueError(f"{key}: dataset/model geometry, preprocessing or quantization identity mismatch")
        if (any(path == out_dir or path.is_relative_to(out_dir) or out_dir.is_relative_to(path)
                for path in (root, model_root)) or spec_path.is_relative_to(out_dir)):
            raise ValueError("output must be separate from its immutable source datasets/models/spec")
        samples = data.get("samples")
        if (not isinstance(samples, list) or len(samples) != 500
                or data.get("num_exported") != 500 or data.get("num_val_total") != 500):
            raise ValueError(f"{key}: full validation requires exactly 500 samples, not a prefix or repeated single image")
        names, files, index_rows = [], [], []
        valid_pixels = 0
        for i, row in enumerate(samples):
            if type(row.get("index")) is not int or row["index"] != i or not row.get("image"):
                raise ValueError(f"{key}: sample indices must be exactly 0..499 in dataset order")
            names.append(Path(str(row["image"]).replace("\\", "/")).name)
            role_files, crcs = [], []
            for role, expected in (("input", selection.input_bytes), ("target", selection.mask_bytes)):
                record = dict(file=row.get(role), bytes=row.get(f"{role}_bytes"),
                              sha256=row.get(f"{role}_sha256"))
                path = read_bound_source_file(root, record)
                if record["bytes"] != expected:
                    raise ValueError(f"{key}/{i}: {role} size disagrees with runtime geometry")
                raw = path.read_bytes()
                if role == "target":
                    target = np.frombuffer(raw, dtype=np.uint8)
                    if np.any((target >= selection.classes) & (target != 255)):
                        raise ValueError(f"{key}/{i}: invalid mapped target class ID")
                    ignored = 19 if selection.classes == 20 else 255
                    valid_pixels += int(np.count_nonzero((target != ignored) & (target != 255)))
                crcs.append(f"{zlib.crc32(raw) & 0xffffffff:08X}")
                role_files.append(path)
            files.append(role_files)
            index_rows.append(f"{i},{crcs[0]},{crcs[1]}\n")
        if len(set(names)) != 500 or (ordered_images is not None and names != ordered_images):
            raise ValueError(f"{key}: repeated samples or different validation image order across modes")
        if not valid_pixels:
            raise ValueError(f"{key}: validation targets contain no valid pixels")
        ordered_images = names
        index_bytes = ("index,input_crc,target_crc\n" + "".join(index_rows)).encode("ascii")
        prepared.append((key, model_root, dataset_path, data, selection, files, index_bytes))

    catalog = ["key,profile,classes,source,mask,param,param_crc,samples,samples_crc\n"]
    cases = []
    for key, model_root, dataset_path, data, selection, files, index_bytes in prepared:
        param_crc = zlib.crc32((model_root / "PARAM.BIN").read_bytes()) & 0xffffffff
        index_crc = zlib.crc32(index_bytes) & 0xffffffff
        catalog.append(f"{key},{selection.profile.profile_id},{selection.classes},{selection.src_mode},"
                       f"{selection.mask_mode},{key}/PARAM.BIN,{param_crc:08X},"
                       f"{key}/SAMPLES.CSV,{index_crc:08X}\n")
        cases.append(dict(key=key, runtime=selection.to_manifest(),
                          dataset=str(dataset_path), dataset_sha256=sha256_file(dataset_path),
                          model_sha256=data["runtime_model_sha256"], param_sha256=data["param_sha256"],
                          samples_crc=f"{index_crc:08X}"))
    catalog_bytes = "".join(catalog).encode("ascii")
    if len(catalog_bytes) >= 8192:
        raise ValueError("VALRUNS.CSV exceeds App catalog capacity")
    # Preflight is complete. Never stage old output masks as new board results.
    out_dir.mkdir(parents=True)
    required_bytes = len(catalog_bytes)
    for key, model_root, _, data, _, files, index_bytes in prepared:
        folder = out_dir / key
        folder.mkdir()
        for name in ("PARAM.BIN", "MODEL.JSN"):
            shutil.copyfile(model_root / name, folder / name)
            required_bytes += (folder / name).stat().st_size
        (folder / "SAMPLES.CSV").write_bytes(index_bytes)
        (folder / "VALDATA.JSN").write_text(json.dumps(data, indent=2), encoding="utf-8")
        required_bytes += len(index_bytes) + (folder / "VALDATA.JSN").stat().st_size
        for i, paths in enumerate(files):
            for role, path in zip(("I", "T"), paths):
                shutil.copyfile(path, folder / short_bin(role, i))
                required_bytes += path.stat().st_size
    missing = []
    for profile in RUNTIME_PROFILES:
        sources = (0, 1, 2, 3) if profile.profile_id == 0 else (0, 4, 5, 6) if profile.profile_id == 1 else (0,)
        for classes in (2, 20):
            for source in sources:
                for mask in range(2 if source else 1):
                    if (profile.profile_id, classes, source, mask) not in selections:
                        missing.append(runtime_selection(profile.profile_id, classes, src_mode=source,
                                                         mask_mode=mask).to_manifest())
    result = dict(format="P7_UNIFIED_VAL500_BOARD_V1", case_count=len(cases), samples_per_case=500,
                  cases=cases, not_in_suite=missing, staged_bytes=required_bytes,
                  output_bytes=sum(item[4].mask_bytes*500 for item in prepared),
                  image_order_sha256=hashlib.sha256("\n".join(ordered_images).encode("utf-8")).hexdigest(),
                  accuracy_status="not run on board; missing modes are not validated")
    (out_dir / "SUITE.JSN").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # The App autorun marker is published last, so a failed copy cannot start an incomplete suite.
    (out_dir / "VALRUNS.CSV").write_bytes(catalog_bytes)
    return result


def export_val_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "runtime_artifact", ""):
        return export_runtime_val_dataset(args)
    if getattr(args, "source_manifest", ""):
        raise ValueError("--source-manifest requires --runtime-artifact")
    if (Path(args.artifact_dir) / "MODEL.JSN").exists():
        raise ValueError("unified dataset export requires explicit --runtime-artifact and model-bound --source-manifest")
    contract = profile_contract(args.profile)
    classes = int(contract["classes"])
    geometry = load_artifact_geometry(Path(args.artifact_dir))
    artifact_manifest = json.loads((Path(args.artifact_dir) / "manifest.json").read_text(encoding="utf-8"))
    preprocessing = artifact_manifest.get("preprocessing", {})
    requested_geometry = DeploymentGeometry(args.height, args.width, args.target_scale,
                                            input_channels=geometry.input_channels)
    if geometry != requested_geometry:
        raise ValueError(
            "CLI geometry disagrees with model artifact: "
            f"requested={requested_geometry.to_manifest()} artifact={geometry.to_manifest()}"
        )
    input_shape_nchw = (geometry.input_channels, geometry.input_height, geometry.input_width)
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
    loader = build_artifact_val_loader(
        Path(args.espnet_dir), Path(args.cached_data_file), geometry, preprocessing,
        args.batch_size, args.num_workers, args.profile, args.scale_in,
    )
    # The shared loader reconstructs the same fake-quant graph for either
    # deployment profile; only the classifier width and target mapping differ.
    args.classes = classes
    args.input_channels = geometry.input_channels
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
                targets, args.profile, already_mapped=geometry.input_channels == 1
            ).cpu().numpy().astype(np.uint8, copy=False)
            if geometry.input_channels == 1:
                mapped_targets = mapped_targets[:, ::geometry.target_scale, ::geometry.target_scale]

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
        "preprocessing": preprocessing,
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
                f"NHWC int8 signed, {geometry.input_height}x{geometry.input_width}x{geometry.input_channels}, "
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
    parser.add_argument("--runtime-artifact", default="", help="Unified artifact directory or MODEL.JSN; uses only its frozen PARAM/runtime contract.")
    parser.add_argument("--source-manifest", default="", help="P7_PREQUANTIZED_VAL_SOURCES_V1 JSON of model-bound source INT8 and registered GT records (see module docstring).")
    parser.add_argument("--board-suite", default="", help="P7_UNIFIED_VAL500_SUITE_V1 spec; package existing 500-image unified datasets for App autorun, without replay/training.")
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
        "runtime_artifact",
        "source_manifest",
        "board_suite",
    ):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).resolve()))
    return args


def main() -> int:
    args = parse_args()
    if args.board_suite:
        if args.runtime_artifact or args.source_manifest or args.clean or args.clean_board_outputs:
            raise ValueError("--board-suite is a separate non-destructive packaging operation")
        result = publish_board_validation_suite(Path(args.board_suite), Path(args.out_dir))
        print(f"Prepared VAL500: {result['case_count']} cases; {len(result['not_in_suite'])} hardware modes not in suite")
        print(f"Staged bytes={result['staged_bytes']}; reserve output bytes={result['output_bytes']} plus FAT overhead")
        print(f"Board files: {args.out_dir}; App build and SD copy remain manual")
        return 0
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
