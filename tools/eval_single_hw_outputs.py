#!/usr/bin/env python3
"""Evaluate one INT8 NPU sample at low-res logits and full-res mask levels."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geometry_contract import (
    LEGACY_GEOMETRY, DeploymentGeometry, RuntimeSelection, geometry_from_manifest,
    runtime_selection_from_manifest, select_runtime_input, runtime_mask_from_logits,
    runtime_target_view,
)


LOWRES_TARGET_SHAPE = (1, 64, 128)
FULLRES_SHAPE = (512, 1024)
UPSAMPLE_SCALE = 8


@dataclass(frozen=True)
class DeploymentContract:
    profile_name: str
    class_count: int
    ignore_metric_class: int | None
    geometry: DeploymentGeometry = LEGACY_GEOMETRY
    center_square_crop: bool = False

    @property
    def lowres_shape(self) -> tuple[int, int]:
        return self.geometry.logits_height, self.geometry.logits_width

    @property
    def fullres_shape(self) -> tuple[int, int]:
        return self.geometry.input_height, self.geometry.input_width

    @property
    def ignore_target(self) -> int:
        return 255 if self.ignore_metric_class is None else self.ignore_metric_class


def load_deployment_contract(artifact_dir: Path) -> DeploymentContract:
    if (artifact_dir / "MODEL.JSN").is_file():
        return load_runtime_contract(artifact_dir)[2]
    candidates = (
        artifact_dir / "manifest.json",
        artifact_dir / "export_manifest.json",
        artifact_dir / "single_manifest.json",
        artifact_dir.parent / "manifest.json",
    )
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"deployment manifest not found under {artifact_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deployment = manifest.get("deployment", {})
    class_count = int(deployment.get("class_count", 0))
    if class_count < 2 or class_count > 32:
        raise ValueError(f"invalid deployment class_count={class_count}")
    ignore = deployment.get("ignore_metric_class")
    return DeploymentContract(
        profile_name=str(deployment.get("profile_name", "")),
        class_count=class_count,
        ignore_metric_class=None if ignore is None else int(ignore),
        geometry=geometry_from_manifest(manifest),
        center_square_crop=bool(manifest.get("preprocessing", {}).get("center_square_crop", False)),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_artifact_directory(path: str | Path) -> Path:
    directory = Path(path).resolve()
    if directory.name.upper() == "MODEL.JSN":
        directory = directory.parent
    if not (directory / "MODEL.JSN").is_file():
        raise FileNotFoundError(f"unified artifact requires {directory / 'MODEL.JSN'}")
    return directory


def load_runtime_array(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {expected} for {shape}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_runtime_contract(
    artifact_dir: Path,
) -> tuple[dict[str, Any], RuntimeSelection, DeploymentContract]:
    """Validate MODEL.JSN identities and semantics, never infer a mode from bytes."""
    from export_int8_hw_blob import validate_runtime_artifact

    artifact_dir = runtime_artifact_directory(artifact_dir)
    manifest = json.loads((artifact_dir / "MODEL.JSN").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "P7_UNIFIED_INT8_ARTIFACT_V1":
        raise ValueError("unsupported MODEL.JSN artifact format")
    for key in ("deployment", "preprocessing", "target", "provenance"):
        if not isinstance(manifest.get(key, {}), dict):
            raise ValueError(f"MODEL.JSN {key} metadata must be an object")
    selection = runtime_selection_from_manifest(manifest)
    required = {
        "param": ("PARAM.BIN", "PARAM_v4_unified"),
        "input": ("INPUTQ.BIN", "source_HWC_int8_prequantized"),
        "encoder": ("ENCODER.BIN", "encoder_HWC_int8_selected"),
        "logits": ("LOGITS.BIN", "logits_HWC_int8"),
        "mask": ("MASK.BIN", "mask_HW_uint8_logits_then_argmax"),
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or not required.keys() <= files.keys():
        raise ValueError("MODEL.JSN files must register PARAM, INPUTQ, ENCODER, LOGITS and MASK")
    if set(files) - required.keys() - {"target"}:
        raise ValueError("MODEL.JSN has unsupported file roles")
    if "target" in files:
        required["target"] = ("TARGET.BIN", "mapped_target_HW_uint8")
    for role, (filename, semantics) in required.items():
        entry = files[role]
        if (not isinstance(entry, dict) or entry.get("file") != filename
                or entry.get("semantics") != semantics
                or (artifact_dir / filename).resolve().parent != artifact_dir):
            raise ValueError(f"MODEL.JSN {role} requires {filename} with {semantics} semantics")
    deployment = manifest.get("deployment", {})
    profile = "binary2" if selection.classes == 2 else "cityscapes20"
    ignore = None if selection.classes == 2 else 19
    mapping = ("cityscapes_vehicle_ids_13_to_18_vs_rest_ignore_255" if selection.classes == 2
               else "cityscapes_train_id_0_to_18_void_to_19")
    if (deployment.get("class_count") != selection.classes
            or deployment.get("profile_name") != profile
            or "ignore_metric_class" not in deployment
            or deployment["ignore_metric_class"] != ignore
            or deployment.get("label_mapping", mapping) != mapping):
        raise ValueError("MODEL.JSN deployment class count/profile/label mapping disagrees with runtime")
    g = selection.profile.geometry
    if geometry_from_manifest(manifest, allow_legacy_fallback=False) != g:
        raise ValueError("MODEL.JSN geometry disagrees with runtime encoder profile")
    try:
        validate_runtime_artifact(artifact_dir)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"MODEL.JSN validation metadata is incomplete: {exc}") from exc
    source = load_runtime_array(artifact_dir / "INPUTQ.BIN", np.int8, selection.source_shape)
    encoder = load_runtime_array(artifact_dir / "ENCODER.BIN", np.int8, g.input_shape_nhwc)
    if not np.array_equal(encoder, select_runtime_input(source, selection)):
        raise ValueError("ENCODER.BIN differs from the registered runtime source selection")
    logits = load_runtime_array(artifact_dir / "LOGITS.BIN", np.int8, g.lowres_shape(selection.classes))
    mask = load_runtime_array(artifact_dir / "MASK.BIN", np.uint8, selection.mask_shape)
    if not np.array_equal(mask, runtime_mask_from_logits(logits, selection)):
        raise ValueError("MASK.BIN differs from registered logits interpolation/argmax")
    contract = DeploymentContract(profile, selection.classes, ignore, g)
    if "target" in files:
        target = load_runtime_array(artifact_dir / "TARGET.BIN", np.uint8, selection.mask_shape)
        map_runtime_target(target, contract, encoding="mapped_output")
    return manifest, selection, contract


def map_runtime_target(
    target: np.ndarray, contract: DeploymentContract, *, encoding: str,
) -> np.ndarray:
    """Raw train IDs and mapped binary IDs are distinct, even for an all-0/1 image."""
    target = np.asarray(target)
    if target.ndim != 2 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("runtime target must be a two-dimensional integer label image")
    mapped = target.astype(np.int64, copy=True)
    if encoding == "cityscapes_train_ids":
        if np.any((mapped != 255) & ((mapped < 0) | (mapped > 19))):
            raise ValueError("original Cityscapes target must contain train IDs 0..19 or ignore255")
        if contract.class_count == 2:
            valid = mapped != 255
            mapped[valid] = np.where((mapped[valid] >= 13) & (mapped[valid] <= 18), 0, 1)
    elif encoding != "mapped_output":
        raise ValueError(f"unsupported runtime target encoding: {encoding}")
    if np.any((mapped != 255) & ((mapped < 0) | (mapped >= contract.class_count))):
        raise ValueError(f"mapped {contract.profile_name} target has invalid class IDs")
    if contract.class_count == 20:
        mapped[mapped == 255] = 19
    return mapped.astype(np.uint8)


def load_runtime_target(
    artifact_dir: Path, manifest: dict[str, Any], selection: RuntimeSelection,
    contract: DeploymentContract, original_target: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    if "target" in manifest["files"]:
        path = artifact_dir / "TARGET.BIN"
        target = load_runtime_array(path, np.uint8, selection.mask_shape)
        return map_runtime_target(target, contract, encoding="mapped_output"), {
            "path": str(path), "sha256": sha256_file(path), "encoding": "mapped_output",
            "sampling": manifest.get("target", {}).get("sampling", "registered_output_grid"),
            "mask_shape_hw": list(selection.mask_shape), "roi_yxhw": list(selection.roi),
        }
    info = manifest.get("target", {})
    sampling = "half_pixel_nearest_source_then_center_ROI_then_mask"
    if (info.get("sampling") != sampling or not info.get("original_path")
            or not info.get("original_sha256")):
        raise ValueError("unified evaluation requires registered TARGET.BIN or original GT with an exact registered transform/hash")
    center = info.get("center_square_source")
    if center is None and selection.src_mode == 0:
        center = manifest.get("preprocessing", {}).get("center_square_crop")
    if not isinstance(center, bool):
        raise ValueError("registered original GT requires explicit center_square_source; use TARGET.BIN for unsupported source transforms")
    path = Path(original_target or info["original_path"])
    if not path.is_absolute():
        path = artifact_dir / path
    if sha256_file(path) != info["original_sha256"]:
        raise ValueError(f"registered original GT hash mismatch: {path}")
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        import cv2
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"unreadable registered original GT: {path}")
    # Validate the full raw image before sampling, not just labels that survive the ROI.
    mapped = map_runtime_target(raw, contract, encoding="cityscapes_train_ids")
    target = runtime_target_view(mapped, selection, center_square_source=center)
    return target, {"path": str(path), "sha256": sha256_file(path), "sampling": sampling,
                    "encoding": "cityscapes_train_ids", "center_square_source": center,
                    "mask_shape_hw": list(selection.mask_shape), "roi_yxhw": list(selection.roi)}


def evaluate_runtime(args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = runtime_artifact_directory(artifact_dir)
    manifest, selection, contract = load_runtime_contract(artifact_dir)
    target, target_info = load_runtime_target(
        artifact_dir, manifest, selection, contract, getattr(args, "target", ""),
    )
    shape = selection.profile.geometry.lowres_shape(selection.classes)
    golden_logits = load_runtime_array(artifact_dir / "LOGITS.BIN", np.int8, shape)
    golden_mask = load_runtime_array(artifact_dir / "MASK.BIN", np.uint8, selection.mask_shape)
    golden_metrics = segmentation_metrics(golden_mask, target, selection.classes, contract.ignore_target)
    result = {
        "evaluation": "unified", "artifact_dir": str(artifact_dir),
        "model_manifest_sha256": sha256_file(artifact_dir / "MODEL.JSN"),
        "runtime": selection.to_manifest(), "deployment": manifest["deployment"],
        "geometry": selection.profile.geometry.to_manifest(), "sample": manifest.get("sample", {}),
        "ground_truth": target_info, "param_sha256": manifest["files"]["param"]["sha256"],
        "input_q_sha256": manifest["files"]["input"]["sha256"],
        "golden_output_q_sha256": manifest["files"]["logits"]["sha256"],
        "artifact_consistency": {"file_identities_validated": True, "selected_encoder_mismatches": 0,
                                 "logits_to_mask_mismatches": 0},
        "golden_fullres_metrics": golden_metrics,
    }
    if getattr(args, "lowres_logits", ""):
        path = Path(args.lowres_logits)
        logits = load_runtime_array(path, np.int8, shape)
        mask = runtime_mask_from_logits(logits, selection)
        result["hls_lowres_logits"] = {
            "path": str(path), "sha256": sha256_file(path), "shape_hwc": list(shape),
            "logit_diff": logit_diff_metrics(logits, golden_logits),
            "mask_diff_vs_golden": diff_metrics(mask, golden_mask),
            "metrics": segmentation_metrics(mask, target, selection.classes, contract.ignore_target),
        }
    if getattr(args, "fullres_mask", ""):
        path = Path(args.fullres_mask)
        mask = load_runtime_array(path, np.uint8, selection.mask_shape)
        invalid = int((mask >= selection.classes).sum())
        if invalid:
            raise ValueError(f"board mask {path} has {invalid} invalid class IDs for {selection.classes} classes")
        result["hls_fullres_mask"] = {
            "path": str(path), "sha256": sha256_file(path), "shape_hw": list(selection.mask_shape),
            "invalid_labels": 0, "mask_diff_vs_golden": diff_metrics(mask, golden_mask),
            "metrics": segmentation_metrics(mask, target, selection.classes, contract.ignore_target),
        }
        if "hls_lowres_logits" in result:
            result["hls_fullres_mask"]["mask_diff_vs_board_logits"] = diff_metrics(
                mask, runtime_mask_from_logits(logits, selection),
            )
    for key in ("hls_lowres_logits", "hls_fullres_mask"):
        if key in result:
            result[key]["metric_delta_vs_golden"] = {
                name: result[key]["metrics"][name] - golden_metrics[name]
                for name in ("pixel_accuracy", "mIoU")
            }
    return result


def load_i8_logits(
    path: Path,
    class_count: int,
    geometry: DeploymentGeometry = LEGACY_GEOMETRY,
) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    shape = (1, geometry.logits_height, geometry.logits_width, class_count)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected}")
    return data.reshape(shape)


def load_lowres_target(path: Path, contract: DeploymentContract) -> np.ndarray:
    target = np.load(path)
    expected = (1, *contract.lowres_shape)
    if tuple(target.shape) == (1, *contract.fullres_shape):
        scale = contract.geometry.target_scale
        target = target[:, ::scale, ::scale]
    if tuple(target.shape) != expected:
        raise ValueError(f"{path} shape={target.shape}, expected {expected}")
    return remap_target_np(target[0], contract)


def remap_target_np(target: np.ndarray, contract: DeploymentContract) -> np.ndarray:
    mapped = target.astype(np.int64, copy=True)
    if contract.profile_name == "cityscapes20":
        invalid = ((mapped < 0) | (mapped >= contract.class_count)) & (mapped != 255)
        if np.any(invalid):
            raise ValueError(f"invalid cityscapes20 labels: {np.unique(mapped[invalid]).tolist()}")
        mapped[mapped == 255] = contract.ignore_target
        return mapped

    valid = mapped != 255
    if set(np.unique(mapped).tolist()).issubset({0, 1, 255}):
        return mapped
    mapped[((mapped < 13) | (mapped > 18)) & valid] = 1
    mapped[(mapped >= 13) & (mapped <= 18)] = 0
    return mapped


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_count: int = 2,
    ignore_target: int = 255,
) -> dict[str, Any]:
    valid = target != ignore_target
    pred_v = pred[valid]
    target_v = target[valid]
    if target_v.size == 0:
        raise ValueError("target has no valid pixels")

    per_class_acc: list[float] = []
    per_class_iou: list[float] = []
    confusion: list[dict[str, int]] = []
    metric_classes = [cls for cls in range(class_count) if cls != ignore_target]
    for cls in metric_classes:
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


def bilinear_axis_map(
    out_idx: int,
    in_size: int,
    scale: int = UPSAMPLE_SCALE,
) -> tuple[int, int, int, int]:
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


def fullres_mask_from_logits(
    logits: np.ndarray,
    class_count: int = 2,
    geometry: DeploymentGeometry = LEGACY_GEOMETRY,
) -> np.ndarray:
    logits2 = logits.reshape(geometry.logits_height, geometry.logits_width, class_count).astype(np.int32)
    fullres_shape = (geometry.input_height, geometry.input_width)
    mask = np.zeros(fullres_shape, dtype=np.uint8)
    x_maps = [
        bilinear_axis_map(x, geometry.logits_width, geometry.target_scale)
        for x in range(fullres_shape[1])
    ]
    x0 = np.asarray([item[0] for item in x_maps], dtype=np.intp)
    x1 = np.asarray([item[1] for item in x_maps], dtype=np.intp)
    wx0 = np.asarray([item[2] for item in x_maps], dtype=np.int32)[:, None]
    wx1 = np.asarray([item[3] for item in x_maps], dtype=np.int32)[:, None]
    for y in range(fullres_shape[0]):
        y0, y1, wy0, wy1 = bilinear_axis_map(
            y, geometry.logits_height, geometry.target_scale
        )
        top = logits2[y0, x0, :] * wx0 + logits2[y0, x1, :] * wx1
        bottom = logits2[y1, x0, :] * wx0 + logits2[y1, x1, :] * wx1
        mask[y, :] = np.argmax(top * wy0 + bottom * wy1, axis=1).astype(np.uint8)
    return mask


def normalize_sample_name(value: str) -> str:
    return value.replace("\\", "/").replace("//", "/").lstrip("./")


def find_sample_index(artifact_dir: Path, cached_data_file: Path) -> tuple[int, str]:
    sample_info = json.loads((artifact_dir / "sample_info.json").read_text(encoding="utf-8"))
    sample_name = normalize_sample_name(sample_info["sample_name"])
    with cached_data_file.open("rb") as f:
        data = pickle.load(f)
    for idx, image in enumerate(data["valIm"]):
        normalized = normalize_sample_name(image)
        if (
            normalized == sample_name
            or normalized.endswith(sample_name)
            or sample_name.endswith(normalized)
        ):
            return idx, image
    raise ValueError(f"sample {sample_name} not found in {cached_data_file}")


def resolve_cached_path(
    value: str,
    cached_data_file: Path,
    espnet_dir: Path,
) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path

    candidates = (
        espnet_dir / path,
        cached_data_file.parent / path,
        path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_fullres_target(
    cached_data_file: Path,
    sample_index: int,
    contract: DeploymentContract,
    espnet_dir: Path,
) -> np.ndarray:
    import cv2

    with cached_data_file.open("rb") as f:
        data = pickle.load(f)
    target_path = resolve_cached_path(
        data["valAnnot"][sample_index], cached_data_file, espnet_dir
    )
    target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise FileNotFoundError(target_path)
    if contract.center_square_crop:
        height, width = target.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        target = target[top:top + side, left:left + side]
    if tuple(target.shape[:2]) != contract.fullres_shape:
        target = cv2.resize(
            target,
            (contract.fullres_shape[1], contract.fullres_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return remap_target_np(target, contract)


def check_single_vs_val_smoke(
    artifact_dir: Path,
    val_smoke_dir: Path,
    contract: DeploymentContract,
) -> dict[str, Any]:
    checks: dict[str, Any] = {"val_smoke_dir": str(val_smoke_dir), "available": val_smoke_dir.exists()}
    if not val_smoke_dir.exists():
        return checks
    smoke_manifest_path = val_smoke_dir / "manifest.json"
    files = {}
    if smoke_manifest_path.is_file():
        files = json.loads(smoke_manifest_path.read_text(encoding="utf-8")).get("files", {})
    pairs = {
        "input_vs_I0000": (artifact_dir / "input_q.bin", val_smoke_dir / (files.get("input_pattern", "I%04d.BIN") % 0)),
        "golden_vs_R0000": (artifact_dir / "golden_output_q.bin", val_smoke_dir / (files.get("reference_logit_pattern", "R%04d.BIN") % 0)),
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
        low_target = load_lowres_target(target_npy, contract).astype(np.uint8)
        t_bin = np.fromfile(t0, dtype=np.uint8)
        checks["target_remap_vs_T0000"] = {
            "target_npy": str(target_npy),
            "t0000": str(t0),
            "same_size": int(low_target.size) == int(t_bin.size),
            "mismatches": int((low_target.reshape(-1) != t_bin).sum()) if low_target.size == t_bin.size else None,
        }
    return checks


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    explicit_runtime = getattr(args, "runtime_artifact", "")
    if explicit_runtime:
        return evaluate_runtime(args, runtime_artifact_directory(explicit_runtime))
    artifact_dir = Path(args.artifact_dir)
    if (artifact_dir / "MODEL.JSN").exists():
        return evaluate_runtime(args, artifact_dir)
    cached_data_file = Path(args.cached_data_file)
    espnet_dir = Path(args.espnet_dir)
    contract = load_deployment_contract(artifact_dir)
    sample_index, sample_name = find_sample_index(artifact_dir, cached_data_file)

    golden_logits = load_i8_logits(
        artifact_dir / "golden_output_q.bin",
        contract.class_count,
        contract.geometry,
    )
    low_target = load_lowres_target(artifact_dir / "target.npy", contract)
    full_target = load_fullres_target(
        cached_data_file, sample_index, contract, espnet_dir
    )

    golden_low_mask = np.argmax(golden_logits[0], axis=-1).astype(np.uint8)
    golden_full_mask = fullres_mask_from_logits(
        golden_logits, contract.class_count, contract.geometry
    )
    valid_low = low_target != contract.ignore_target
    valid_full = full_target != contract.ignore_target

    result: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "sample_index": sample_index,
        "sample_name": sample_name,
        "deployment": {
            "profile_name": contract.profile_name,
            "class_count": contract.class_count,
            "ignore_metric_class": contract.ignore_metric_class,
        },
        "geometry": contract.geometry.to_manifest(),
        "input_q_sha256": sha256_file(artifact_dir / "input_q.bin"),
        "golden_output_q_sha256": sha256_file(artifact_dir / "golden_output_q.bin"),
        "artifact_consistency": check_single_vs_val_smoke(
            artifact_dir, Path(args.val_smoke_dir), contract
        ),
        "golden_lowres_metrics": segmentation_metrics(
            golden_low_mask, low_target, contract.class_count, contract.ignore_target
        ),
        "golden_fullres_metrics": segmentation_metrics(
            golden_full_mask, full_target, contract.class_count, contract.ignore_target
        ),
    }

    if args.lowres_logits:
        hls_logits_path = Path(args.lowres_logits)
        hls_logits = load_i8_logits(
            hls_logits_path, contract.class_count, contract.geometry
        )
        hls_low_mask = np.argmax(hls_logits[0], axis=-1).astype(np.uint8)
        result["hls_lowres_logits"] = {
            "path": str(hls_logits_path),
            "sha256": sha256_file(hls_logits_path),
            "logit_diff": logit_diff_metrics(hls_logits, golden_logits),
            "mask_diff_vs_golden": diff_metrics(hls_low_mask, golden_low_mask, valid_low),
            "metrics": segmentation_metrics(
                hls_low_mask, low_target, contract.class_count, contract.ignore_target
            ),
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
        if data.size != int(np.prod(contract.fullres_shape)):
            raise ValueError(
                f"{hls_mask_path} has {data.size} bytes, "
                f"expected {np.prod(contract.fullres_shape)}"
            )
        hls_full_mask = data.reshape(contract.fullres_shape)
        result["hls_fullres_mask"] = {
            "path": str(hls_mask_path),
            "sha256": sha256_file(hls_mask_path),
            "invalid_labels": int((hls_full_mask >= contract.class_count).sum()),
            "mask_diff_vs_golden": diff_metrics(hls_full_mask, golden_full_mask, valid_full),
            "metrics": segmentation_metrics(
                hls_full_mask, full_target, contract.class_count, contract.ignore_target
            ),
        }
        result["hls_fullres_mask"]["metric_delta_vs_golden"] = {
            "pixel_accuracy": result["hls_fullres_mask"]["metrics"]["pixel_accuracy"]
            - result["golden_fullres_metrics"]["pixel_accuracy"],
            "mIoU": result["hls_fullres_mask"]["metrics"]["mIoU"]
            - result["golden_fullres_metrics"]["mIoU"],
        }

    return result


def print_summary(result: dict[str, Any]) -> None:
    if result.get("evaluation") == "unified":
        runtime = result["runtime"]
        print(f"unified model={runtime['model_profile_name']} classes={runtime['class_count']} "
              f"source={runtime['source_shape_nhwc']} ROI={runtime['roi_yxhw']} "
              f"output={runtime['mask_shape_hw']}")
        metrics = result["golden_fullres_metrics"]
        print(f"golden output-view: PA={metrics['pixel_accuracy']:.8f} mIoU={metrics['mIoU']:.8f} "
              f"valid_pixels={metrics['valid_pixels']}")
        for key in ("hls_lowres_logits", "hls_fullres_mask"):
            if key in result:
                entry = result[key]
                metrics = entry["metrics"]
                print(f"{key} output-view: PA={metrics['pixel_accuracy']:.8f} mIoU={metrics['mIoU']:.8f} "
                      f"mask_mismatch={entry['mask_diff_vs_golden']['mismatches']}")
        return
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
    parser.add_argument("--artifact-dir", default=r"D:\ESP_INT8\hw_artifacts\sched_v4_p7_0702")
    parser.add_argument("--runtime-artifact", default="", help="Unified artifact directory or MODEL.JSN; never uses legacy fallback.")
    parser.add_argument("--target", default="", help="Relocate registered original GT with the same SHA256 (TARGET.BIN takes precedence).")
    parser.add_argument("--espnet-dir", default=r"D:\ESPNet")
    parser.add_argument("--cached-data-file", default=r"D:\ESPNet\city.p")
    parser.add_argument("--val-smoke-dir", default=r"D:\ESP_INT8\hw_artifacts\sched_v4_val_smoke")
    parser.add_argument("--lowres-logits", default="")
    parser.add_argument("--fullres-mask", "--board-mask", default="", help="Board uint8 mask on the registered output grid.")
    parser.add_argument(
        "--out-json",
        default="",
        help="Metrics JSON (default: single_hw_output_metrics.json in the selected artifact directory).",
    )
    args = parser.parse_args()

    result = evaluate(args)
    out_path = Path(args.out_json) if args.out_json else Path(result["artifact_dir"]) / "single_hw_output_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
