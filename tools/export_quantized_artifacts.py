import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ESPNET_SRC_DIR = Path(os.environ.get("ESPNET_SRC_DIR", r"D:\ESPNet"))
if str(ESPNET_SRC_DIR) not in sys.path:
    sys.path.append(str(ESPNET_SRC_DIR))

import DataSet as myDataLoader
import Model_flattened_quantized as net
import Transforms as myTransforms
from hw_int8_math import hls_quantize_i8
from test_single_image_for_all import (
    HARDWARE_FUSED_RELU_OUTPUTS,
    HARDWARE_SCALE_GROUPS,
    prepare_quantized_model,
)


def dump_text(array, path, fmt="%s"):
    np.savetxt(path, array, fmt=fmt)


def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(cached_data_file, scale_in, batch_size, num_workers):
    data = pickle.load(open(cached_data_file, "rb"))

    val_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.ToTensor(scale_in),
    ])
    train_tf = myTransforms.Compose([
        myTransforms.Normalize(mean=data["mean"], std=data["std"]),
        myTransforms.Scale(1024, 512),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(scale_in),
    ])

    val_dataset = myDataLoader.MyDataset(data["valIm"], data["valAnnot"], transform=val_tf)
    train_dataset = myDataLoader.MyDataset(data["trainIm"], data["trainAnnot"], transform=train_tf)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max(1, batch_size - 2),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    return val_loader, train_loader


def save_quantized_conv(module_name, module, layer_dir):
    layer_dir.mkdir(parents=True, exist_ok=True)

    weight_q = module.weight()
    weight_i8 = weight_q.int_repr().cpu().numpy()
    np.save(layer_dir / "weight_int8.npy", weight_i8)
    dump_text(weight_i8.reshape(-1), layer_dir / "weight_int8.txt", fmt="%d")

    summary = {
        "name": module_name,
        "weight_qscheme": str(weight_q.qscheme()),
    }

    if weight_q.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine):
        scales = weight_q.q_per_channel_scales().cpu().numpy()
        zero_points = weight_q.q_per_channel_zero_points().cpu().numpy()
        axis = int(weight_q.q_per_channel_axis())
        np.save(layer_dir / "weight_scales.npy", scales)
        np.save(layer_dir / "weight_zero_points.npy", zero_points)
        dump_text(scales, layer_dir / "weight_scales.txt", fmt="%.18e")
        dump_text(zero_points, layer_dir / "weight_zero_points.txt", fmt="%d")
        summary["weight_axis"] = axis
    else:
        summary["weight_scale"] = float(weight_q.q_scale())
        summary["weight_zero_point"] = int(weight_q.q_zero_point())

    bias = module.bias()
    if bias is not None:
        bias_np = bias.detach().cpu().numpy()
        np.save(layer_dir / "bias_fp32.npy", bias_np)
        dump_text(bias_np, layer_dir / "bias_fp32.txt", fmt="%.18e")

    with open(layer_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    return [value]


def quantize_per_channel_symmetric(weight, scales, quant_min=-127, quant_max=127):
    view_shape = [scales.numel()] + [1] * (weight.dim() - 1)
    scaled = weight / scales.view(*view_shape)
    quantized = hls_quantize_i8(scaled, scale=1.0, quant_min=quant_min, quant_max=quant_max).to(torch.int32)
    return quantized


def quantize_per_tensor(weight, scale, zero_point, quant_min, quant_max):
    quantized = hls_quantize_i8(
        weight / scale + zero_point,
        scale=1.0,
        quant_min=quant_min,
        quant_max=quant_max,
    ).to(torch.int32)
    return quantized


def save_fake_quantized_conv(module_name, module, layer_dir):
    layer_dir.mkdir(parents=True, exist_ok=True)

    weight_fp = module.weight.detach().cpu()
    weight_fq = module.weight_fake_quant
    scale = weight_fq.scale.detach().cpu()
    zero_point = weight_fq.zero_point.detach().cpu().to(torch.int32)
    quant_min = int(weight_fq.quant_min)
    quant_max = int(weight_fq.quant_max)

    summary = {
        "name": module_name,
        "weight_qscheme": str(weight_fq.qscheme),
        "weight_quant_min": quant_min,
        "weight_quant_max": quant_max,
    }

    if weight_fq.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
        qweight = quantize_per_channel_symmetric(weight_fp, scale, quant_min=quant_min, quant_max=quant_max)
        np.save(layer_dir / "weight_scales.npy", scale.numpy())
        np.save(layer_dir / "weight_zero_points.npy", zero_point.numpy())
        dump_text(scale.numpy(), layer_dir / "weight_scales.txt", fmt="%.18e")
        dump_text(zero_point.numpy(), layer_dir / "weight_zero_points.txt", fmt="%d")
        summary["weight_axis"] = int(weight_fq.ch_axis)
    else:
        qweight = quantize_per_tensor(weight_fp, float(scale.item()), int(zero_point.item()), quant_min, quant_max)
        summary["weight_scale"] = float(scale.item())
        summary["weight_zero_point"] = int(zero_point.item())

    qweight_np = qweight.numpy()
    np.save(layer_dir / "weight_int8.npy", qweight_np)
    dump_text(qweight_np.reshape(-1), layer_dir / "weight_int8.txt", fmt="%d")

    bias = module.bias
    if bias is not None:
        bias_np = bias.detach().cpu().numpy()
        np.save(layer_dir / "bias_fp32.npy", bias_np)
        dump_text(bias_np, layer_dir / "bias_fp32.txt", fmt="%.18e")

    with open(layer_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def collect_activation_qparams(model):
    qparams = {}
    for name, module in model.named_modules():
        if not hasattr(module, "activation_post_process"):
            continue

        observer = module.activation_post_process
        record = {
            "module_type": type(module).__name__,
            "observer_type": type(observer).__name__,
        }

        if hasattr(observer, "dtype"):
            record["dtype"] = str(observer.dtype)
        if hasattr(observer, "qscheme"):
            record["qscheme"] = str(observer.qscheme)
        if hasattr(observer, "quant_min"):
            record["quant_min"] = int(observer.quant_min)
        if hasattr(observer, "quant_max"):
            record["quant_max"] = int(observer.quant_max)
        if hasattr(observer, "ch_axis"):
            record["ch_axis"] = int(observer.ch_axis)
        if hasattr(observer, "scale"):
            record["scale"] = tensor_to_list(observer.scale)
        if hasattr(observer, "zero_point"):
            record["zero_point"] = tensor_to_list(observer.zero_point)

        qparams[name] = record

    return qparams


def inspect_hardware_quant_constraints(qparams, scale_groups=HARDWARE_SCALE_GROUPS):
    nonzero_zero_points = []
    scale_mismatches = []

    for name, record in qparams.items():
        zps = record.get("zero_point")
        if zps and any(int(zp) != 0 for zp in zps):
            nonzero_zero_points.append({"name": name, "zero_point": zps})

    for group in scale_groups:
        present = []
        for name in group:
            if name not in qparams or "scale" not in qparams[name]:
                continue
            present.append((name, float(qparams[name]["scale"][0])))
        if not present:
            continue

        ref_scale = present[0][1]
        mismatched = [
            {"name": name, "scale": scale}
            for name, scale in present
            if abs(scale - ref_scale) > max(1e-8, 1e-5 * max(abs(scale), abs(ref_scale)))
        ]
        if mismatched:
            scale_mismatches.append({
                "reference": {"name": present[0][0], "scale": ref_scale},
                "members": [{"name": name, "scale": scale} for name, scale in present],
            })

    return {
        "nonzero_zero_points": nonzero_zero_points,
        "scale_mismatches": scale_mismatches,
        "passed": len(nonzero_zero_points) == 0 and len(scale_mismatches) == 0,
    }


def quantize_activation_tensor(tensor, observer):
    scale = observer.scale.detach().cpu()
    zero_point = observer.zero_point.detach().cpu().to(torch.int32)
    quant_min = int(observer.quant_min)
    quant_max = int(observer.quant_max)

    if scale.numel() != 1 or zero_point.numel() != 1:
        raise ValueError("Only per-tensor activation quantization is supported for golden dumps.")

    if int(zero_point.item()) != 0:
        q = hls_quantize_i8(
            tensor.detach().cpu() + float(zero_point.item()) * float(scale.item()),
            float(scale.item()),
            quant_min=quant_min,
            quant_max=quant_max,
        )
    else:
        q = hls_quantize_i8(
            tensor.detach().cpu(),
            float(scale.item()),
            quant_min=quant_min,
            quant_max=quant_max,
        )
    q = q.to(torch.int32)
    return q


def dump_golden_sample(model, sample_inputs, sample_targets, sample_name, output_dir):
    golden_dir = output_dir / "golden_sample"
    golden_dir.mkdir(parents=True, exist_ok=True)

    hooks = []
    patched_methods = []

    def save_module_output(name, module, output):
        if not isinstance(output, torch.Tensor):
            return
        if not hasattr(module, "activation_post_process"):
            return

        observer = module.activation_post_process
        if not hasattr(observer, "scale") or not hasattr(observer, "zero_point"):
            return

        if name in HARDWARE_FUSED_RELU_OUTPUTS:
            output = torch.clamp(output, 0)

        module_dir = golden_dir / name.replace(".", "_")
        module_dir.mkdir(parents=True, exist_ok=True)

        q = quantize_activation_tensor(output, observer)
        np.save(module_dir / "output_int.npy", q.numpy())
        dump_text(q.numpy().reshape(-1), module_dir / "output_int.txt", fmt="%d")

        meta = {
            "module_name": name,
            "module_type": type(module).__name__,
            "scale": tensor_to_list(observer.scale),
            "zero_point": tensor_to_list(observer.zero_point),
            "quant_min": int(observer.quant_min),
            "quant_max": int(observer.quant_max),
        }
        with open(module_dir / "qparams.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def make_hook(name, module):
        def hook(_module, _inputs, output):
            save_module_output(name, module, output)
        return hook

    for name, module in model.named_modules():
        if name and hasattr(module, "activation_post_process"):
            hooks.append(module.register_forward_hook(make_hook(name, module)))
            if type(module).__name__ == "FloatFunctional":
                if hasattr(module, "add"):
                    original_add = module.add

                    def wrapped_add(x, y, _orig=original_add, _name=name, _module=module):
                        output = _orig(x, y)
                        save_module_output(_name, _module, output)
                        return output

                    module.add = wrapped_add
                    patched_methods.append((module, "add", original_add))

                if hasattr(module, "cat"):
                    original_cat = module.cat

                    def wrapped_cat(x, dim=0, _orig=original_cat, _name=name, _module=module):
                        output = _orig(x, dim)
                        save_module_output(_name, _module, output)
                        return output

                    module.cat = wrapped_cat
                    patched_methods.append((module, "cat", original_cat))

    model = model.eval()
    device = next(model.parameters()).device
    sample_inputs = sample_inputs.to(device)
    with torch.no_grad():
        outputs = model(sample_inputs)

    quant_dir = golden_dir / "quant"
    quant_dir.mkdir(parents=True, exist_ok=True)
    quant_observer = model.quant.activation_post_process
    q_input = quantize_activation_tensor(sample_inputs.cpu(), quant_observer)
    np.save(quant_dir / "input_int.npy", q_input.numpy())
    dump_text(q_input.numpy().reshape(-1), quant_dir / "input_int.txt", fmt="%d")

    np.save(golden_dir / "model_output_fp32.npy", outputs.detach().cpu().numpy())
    np.save(golden_dir / "target.npy", sample_targets.cpu().numpy())
    with open(golden_dir / "sample_info.json", "w", encoding="utf-8") as f:
        json.dump({"sample_name": sample_name}, f, indent=2)

    for handle in hooks:
        handle.remove()
    for module, attr_name, original in patched_methods:
        setattr(module, attr_name, original)


def export_quantized_artifacts(args):
    set_deterministic_seed(args.seed)
    os.chdir(ESPNET_SRC_DIR)
    output_dir = Path(args.output_dir)
    layers_dir = output_dir / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    val_loader, train_loader = build_loaders(
        args.cached_data_file,
        args.scale_in,
        args.batch_size,
        args.num_workers,
    )
    if args.qat and args.qat_train_subset > 0:
        train_subset = Subset(train_loader.dataset, list(range(min(args.qat_train_subset, len(train_loader.dataset)))))
        train_loader = DataLoader(
            train_subset,
            batch_size=max(1, args.batch_size - 2),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    model = net.ESPNet_Encoder(args.classes, p=args.p, q=args.q)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.cpu().eval()

    use_fake_quant_eval = args.fake_quant_eval or not args.asymmetric_activations
    quant_device = torch.device(
        "cuda" if torch.cuda.is_available() and (args.qat or use_fake_quant_eval) else "cpu"
    )
    model = model.to(quant_device).eval()

    model_quant = prepare_quantized_model(
        model,
        calibration_loader=val_loader,
        qat_mode=args.qat,
        train_loader=train_loader if args.qat else None,
        calibration_batches=args.calibration_batches,
        qat_epochs=args.qat_epochs,
        qat_lr=args.qat_lr,
        symmetric_activations=not args.asymmetric_activations,
        convert_model=not use_fake_quant_eval,
        hardware_constraints=not args.disable_hardware_constraints,
    )
    model_quant = model_quant.cpu().eval()

    if use_fake_quant_eval:
        torch.save(model_quant.state_dict(), output_dir / "fake_quant_state_dict.pth")
    else:
        torch.save(model_quant.state_dict(), output_dir / "quantized_state_dict.pth")

    activation_qparams = collect_activation_qparams(model_quant)

    all_layers = []
    for name, module in model_quant.named_modules():
        if hasattr(module, "_packed_params"):
            safe_name = name.replace(".", "_")
            save_quantized_conv(name, module, layers_dir / safe_name)
            layer_info = {"name": name}
            if name in activation_qparams:
                layer_info.update(activation_qparams[name])
                with open(layers_dir / safe_name / "activation_qparams.json", "w", encoding="utf-8") as f:
                    json.dump(activation_qparams[name], f, indent=2)
            all_layers.append(layer_info)
        elif hasattr(module, "weight_fake_quant"):
            safe_name = name.replace(".", "_")
            save_fake_quantized_conv(name, module, layers_dir / safe_name)
            layer_info = {"name": name, "export_mode": "fake_quant"}
            if name in activation_qparams:
                layer_info.update(activation_qparams[name])
                with open(layers_dir / safe_name / "activation_qparams.json", "w", encoding="utf-8") as f:
                    json.dump(activation_qparams[name], f, indent=2)
            all_layers.append(layer_info)

    with open(output_dir / "activation_qparams.json", "w", encoding="utf-8") as f:
        json.dump(activation_qparams, f, indent=2)
    hardware_constraint_report = inspect_hardware_quant_constraints(activation_qparams)

    if use_fake_quant_eval and not args.skip_golden_dump:
        sample_inputs, sample_targets, sample_names = next(iter(val_loader))
        dump_golden_sample(
            model_quant,
            sample_inputs[:1].cpu(),
            sample_targets[:1].cpu(),
            sample_names[0],
            output_dir,
        )

    manifest = {
        "model_path": args.model_path,
        "qat": bool(args.qat),
        "fake_quant": bool(use_fake_quant_eval),
        "hardware_constraints": bool(not args.disable_hardware_constraints),
        "hardware_constraint_report": hardware_constraint_report,
        "num_quantized_convs": len(all_layers),
        "layers": all_layers,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Export finished. Artifacts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=r"D:\ESPNet\results_vehicle(model_flattened)_enc__enc_1_1\model_64.pth")
    parser.add_argument("--cached_data_file", default=r"D:\ESPNet\city.p")
    parser.add_argument("--output_dir", default=r"D:\ESPNet\quantized_artifacts")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--scale_in", type=int, default=8)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--calibration_batches", type=int, default=11)
    parser.add_argument("--qat_epochs", type=int, default=3)
    parser.add_argument("--qat_lr", type=float, default=1e-4)
    parser.add_argument("--qat_train_subset", type=int, default=0)
    parser.add_argument("--asymmetric_activations", action="store_true")
    parser.add_argument("--fake_quant_eval", action="store_true")
    parser.add_argument("--skip_golden_dump", action="store_true")
    parser.add_argument("--disable_hardware_constraints", action="store_true")
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()
    export_quantized_artifacts(args)


if __name__ == "__main__":
    main()
