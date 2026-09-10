#!/usr/bin/env python3
"""Freeze validated CSim outputs as hardware-behavior golden sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


PARAM_MAGIC = 0x544E4945
PARAM_VERSION_V5 = 5


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "export_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frame = manifest.get("frame_data", {})
    if "input_shape_nhwc" not in frame or "output_shape_nhwc" not in frame:
        raise ValueError(f"{path} is missing frame_data shapes")
    return manifest


def finalize_csim_golden(artifact_dir: Path, csim_build_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    csim_build_dir = csim_build_dir.resolve()

    param = (artifact_dir / "PARAM.BIN").read_bytes()
    if len(param) < 8:
        raise ValueError("PARAM.BIN is shorter than its magic/version fields")
    magic, version = struct.unpack_from("<II", param, 0)
    if magic != PARAM_MAGIC or version != PARAM_VERSION_V5:
        raise ValueError(
            f"hardware golden finalization requires PARAM v5, got "
            f"magic=0x{magic:08X} version={version}"
        )

    manifest = _load_manifest(artifact_dir)
    frame = manifest["frame_data"]
    input_shape = tuple(int(value) for value in frame["input_shape_nhwc"])
    output_shape = tuple(int(value) for value in frame["output_shape_nhwc"])
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ValueError("input/output shapes must be four-dimensional NHWC")

    model_logits = (artifact_dir / "golden_output_q.bin").read_bytes()
    hls_logits = (csim_build_dir / "csim_u72_lowres_logits.bin").read_bytes()
    hls_mask = (csim_build_dir / "hls_output_mask.bin").read_bytes()
    expected_logits_bytes = int(np.prod(output_shape))
    expected_mask_bytes = input_shape[1] * input_shape[2]
    if len(model_logits) != expected_logits_bytes:
        raise ValueError(
            f"model golden bytes={len(model_logits)} expected={expected_logits_bytes}"
        )
    if len(hls_logits) != expected_logits_bytes:
        raise ValueError(
            f"HLS logits bytes={len(hls_logits)} expected={expected_logits_bytes}"
        )
    if len(hls_mask) != expected_mask_bytes:
        raise ValueError(
            f"HLS mask bytes={len(hls_mask)} expected={expected_mask_bytes}"
        )

    model = np.frombuffer(model_logits, dtype=np.int8).reshape(output_shape)
    hls = np.frombuffer(hls_logits, dtype=np.int8).reshape(output_shape)
    diff = hls.astype(np.int16) - model.astype(np.int16)
    model_argmax = np.argmax(model, axis=-1)
    hls_argmax = np.argmax(hls, axis=-1)

    (artifact_dir / "hls_golden_output_q.bin").write_bytes(hls_logits)
    (artifact_dir / "hls_golden_mask_fullres.bin").write_bytes(hls_mask)
    np.save(artifact_dir / "hls_golden_output_q_nhwc.npy", hls)

    audit: dict[str, Any] = {
        "format": "ESP_INT8_HLS_CSIM_GOLDEN_V1",
        "param_version": version,
        "param_sha256": _sha256(param),
        "artifact_dir": str(artifact_dir),
        "csim_build_dir": str(csim_build_dir),
        "output_shape_nhwc": list(output_shape),
        "model_golden": {
            "file": "golden_output_q.bin",
            "bytes": len(model_logits),
            "sha256": _sha256(model_logits),
        },
        "hls_golden": {
            "file": "hls_golden_output_q.bin",
            "bytes": len(hls_logits),
            "sha256": _sha256(hls_logits),
        },
        "lowres_model_vs_hls": {
            "mismatches": int(np.count_nonzero(diff)),
            "elements": int(diff.size),
            "max_abs_diff": int(np.max(np.abs(diff))) if diff.size else 0,
            "argmax_mismatches": int(np.count_nonzero(model_argmax != hls_argmax)),
            "pixels": int(model_argmax.size),
        },
        "fullres_mask": {
            "file": "hls_golden_mask_fullres.bin",
            "bytes": len(hls_mask),
            "sha256": _sha256(hls_mask),
        },
    }
    (artifact_dir / "hls_golden_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--csim-build-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = finalize_csim_golden(args.artifact_dir, args.csim_build_dir)
    delta = audit["lowres_model_vs_hls"]
    print(
        "HLS golden finalized: "
        f"param={audit['param_sha256']} "
        f"logits={audit['hls_golden']['sha256']} "
        f"mismatches={delta['mismatches']}/{delta['elements']} "
        f"argmax={delta['argmax_mismatches']}/{delta['pixels']}"
    )


if __name__ == "__main__":
    main()
