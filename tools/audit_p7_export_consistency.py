#!/usr/bin/env python3
"""Audit that P7 model artifacts, PARAM v3 files, and golden dumps share one model state.

The artifact directory stores PyTorch hook outputs in NCHW order. Hardware-facing
BIN files store frame/logit tensors in NHWC order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from torch.nn.intrinsic.qat import freeze_bn_stats
from torch.quantization import disable_observer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_quantized_artifacts import (  # noqa: E402
    HARDWARE_FUSED_RELU_OUTPUTS,
    build_loaders,
    quantize_activation_tensor,
    quantize_per_channel_symmetric,
    quantize_per_tensor,
)
from hw_int8_math import install_p7_hardware_precision_hooks  # noqa: E402
from test_single_image_for_all import (  # noqa: E402
    apply_hardware_quant_constraints,
    get_hardware_qconfig,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    return name.replace(".", "_")


def add_espnet(espnet_dir: Path) -> None:
    if str(espnet_dir) not in sys.path:
        sys.path.append(str(espnet_dir))


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    add_espnet(args.espnet_dir)
    import Model_flattened_quantized as net

    model = net.ESPNet_Encoder(args.classes, p=args.p, q=args.q)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.cpu().eval()
    model.qconfig = get_hardware_qconfig(qat_mode=True, symmetric_activations=True)
    model.fuse_model()

    prepared = torch.quantization.prepare_qat(model.train(), inplace=False)
    state = torch.load(args.artifact_dir / "fake_quant_state_dict.pth", map_location="cpu")
    missing, unexpected = prepared.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"fake-quant state mismatch: missing={missing}, unexpected={unexpected}")
    disable_observer(prepared)
    prepared.apply(freeze_bn_stats)
    apply_hardware_quant_constraints(prepared)
    install_p7_hardware_precision_hooks(prepared, HARDWARE_FUSED_RELU_OUTPUTS)
    return prepared.cpu().eval()


def audit_weights(model: torch.nn.Module, artifact_dir: Path) -> list[str]:
    errors: list[str] = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight_fake_quant"):
            continue
        layer_dir = artifact_dir / "layers" / safe_name(name)
        expected_path = layer_dir / "weight_int8.npy"
        if not expected_path.exists():
            errors.append(f"{name}: missing {expected_path}")
            continue
        weight_fp = module.weight.detach().cpu()
        fq = module.weight_fake_quant
        scale = fq.scale.detach().cpu()
        zp = fq.zero_point.detach().cpu().to(torch.int32)
        qmin = int(fq.quant_min)
        qmax = int(fq.quant_max)
        if fq.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
            q = quantize_per_channel_symmetric(weight_fp, scale, quant_min=qmin, quant_max=qmax)
        else:
            q = quantize_per_tensor(weight_fp, float(scale.item()), int(zp.item()), qmin, qmax)
        got = np.load(expected_path)
        if got.shape != tuple(q.shape) or not np.array_equal(got.astype(np.int32), q.numpy().astype(np.int32)):
            errors.append(f"{name}: exported weight_int8.npy differs from fake_quant_state_dict")
    return errors


def capture_selected_outputs(
    model: torch.nn.Module,
    sample_inputs: torch.Tensor,
    names: Iterable[str],
) -> dict[str, np.ndarray]:
    wanted = set(names)
    captured: dict[str, np.ndarray] = {}
    hooks = []
    patched_methods = []

    def save_output(name: str, module: torch.nn.Module, output: torch.Tensor):
        if name not in wanted or not isinstance(output, torch.Tensor):
            return
        if not hasattr(module, "activation_post_process"):
            return
        value = output.detach().cpu()
        if name in HARDWARE_FUSED_RELU_OUTPUTS:
            value = torch.clamp(value, 0)
        q = quantize_activation_tensor(value, module.activation_post_process)
        captured[name] = q.numpy().astype(np.int8)

    def make_hook(name: str, module: torch.nn.Module):
        def hook(_module, _inputs, output):
            save_output(name, module, output)

        return hook

    for name, module in model.named_modules():
        if name in wanted and hasattr(module, "activation_post_process"):
            hooks.append(module.register_forward_hook(make_hook(name, module)))
            if type(module).__name__ == "FloatFunctional":
                if hasattr(module, "add"):
                    original_add = module.add

                    def wrapped_add(x, y, _orig=original_add, _name=name, _module=module):
                        output = _orig(x, y)
                        save_output(_name, _module, output)
                        return output

                    module.add = wrapped_add
                    patched_methods.append((module, "add", original_add))

                if hasattr(module, "cat"):
                    original_cat = module.cat

                    def wrapped_cat(x, dim=0, _orig=original_cat, _name=name, _module=module):
                        output = _orig(x, dim)
                        save_output(_name, _module, output)
                        return output

                    module.cat = wrapped_cat
                    patched_methods.append((module, "cat", original_cat))

    with torch.no_grad():
        model(sample_inputs.cpu())
    for hook in hooks:
        hook.remove()
    for module, attr_name, original in patched_methods:
        setattr(module, attr_name, original)
    return captured


def audit_golden_hooks(args: argparse.Namespace, model: torch.nn.Module) -> list[str]:
    errors: list[str] = []
    val_loader, _train_loader = build_loaders(
        args.cached_data_file,
        args.scale_in,
        args.batch_size,
        args.num_workers,
    )
    sample_inputs, _sample_targets, _sample_names = next(iter(val_loader))
    names = [
        "level2_0_bn",
        "level2_block_cat_ff.0",
        "level2_blocks.6",
        "b2_cat_ff",
        "b2_bn",
        "level3_0_c1",
        "classifier",
    ]
    captured = capture_selected_outputs(model, sample_inputs[:1], names)
    for name in names:
        path = args.artifact_dir / "golden_sample" / safe_name(name) / "output_int.npy"
        if name not in captured:
            errors.append(f"{name}: hook not captured")
            continue
        if not path.exists():
            errors.append(f"{name}: missing golden {path}")
            continue
        saved = np.load(path).astype(np.int8)
        if saved.shape != captured[name].shape or not np.array_equal(saved, captured[name]):
            errors.append(f"{name}: golden_sample output_int.npy differs from rerun PyTorch hook")
    return errors


def audit_hw_layout(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    manifest_path = args.hw_dir / "export_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    param_bin = args.hw_dir / "PARAM.BIN"
    if manifest.get("param_blob_sha256") != sha256_file(param_bin):
        errors.append("PARAM.BIN sha256 differs from export_manifest.json")

    logits_nchw = np.load(args.artifact_dir / "golden_sample" / "classifier" / "output_int.npy").astype(np.int8)
    if logits_nchw.ndim != 4 or logits_nchw.shape != (1, 2, 64, 128):
        errors.append(f"classifier golden must be NCHW (1,2,64,128), got {logits_nchw.shape}")
    else:
        logits_nhwc = np.ascontiguousarray(np.transpose(logits_nchw, (0, 2, 3, 1)))
        bin_data = np.fromfile(args.hw_dir / "golden_output_q.bin", dtype=np.int8)
        if bin_data.size != logits_nhwc.size or not np.array_equal(bin_data.reshape(logits_nhwc.shape), logits_nhwc):
            errors.append("golden_output_q.bin does not equal classifier output_int.npy transposed NCHW->NHWC")

    input_nhwc = np.load(args.hw_dir / "input_q_nhwc.npy").astype(np.int8)
    input_bin = np.fromfile(args.hw_dir / "input_q.bin", dtype=np.int8)
    if input_bin.size != input_nhwc.size or not np.array_equal(input_bin.reshape(input_nhwc.shape), input_nhwc):
        errors.append("input_q.bin does not equal input_q_nhwc.npy")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espnet-dir", type=Path, default=Path(r"D:\ESPNet"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--hw-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth"))
    parser.add_argument("--cached-data-file", type=Path, default=Path(r"D:\ESPNet\city.p"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-in", type=int, default=8)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    os.chdir(args.espnet_dir)
    model = load_model(args)
    errors = []
    errors.extend(audit_weights(model, args.artifact_dir))
    errors.extend(audit_golden_hooks(args, model))
    errors.extend(audit_hw_layout(args))

    result = {
        "artifact_dir": str(args.artifact_dir),
        "hw_dir": str(args.hw_dir),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    text = json.dumps(result, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
