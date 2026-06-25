"""P7 INT8 hardware precision helpers used by QAT/export scripts.

The functions in this file mirror the HLS arithmetic in
``ESP_INT8_hls/include/npu_q.hpp`` at tensor boundaries. They are intentionally
small and dependency-light so QAT/export can share one definition of round,
clamp and bypass-add behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


INT8_MIN = -128
INT8_MAX = 127


def _as_tensor(value, device=None, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def hls_round_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    """Round like the HLS fixed-point path: halves go away from zero."""
    return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))


def round_away_float_scalar(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def choose_multiplier(real_multiplier: float, max_shift: int = 31) -> tuple[int, int]:
    """Choose the same fixed-point multiplier/shift pair as export_int8_hw_blob."""
    if not math.isfinite(real_multiplier):
        raise ValueError(f"non-finite multiplier: {real_multiplier}")
    if real_multiplier == 0.0:
        return 0, 0

    sign = -1 if real_multiplier < 0 else 1
    value = abs(real_multiplier)
    for shift in range(max_shift, -1, -1):
        mult = round_away_float_scalar(value * (1 << shift))
        if 0 < mult <= 0x7FFFFFFF:
            return sign * mult, shift
    return sign * min(round_away_float_scalar(value), 0x7FFFFFFF), 0


def hls_round_shift_int(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Match npu_q.hpp round_shift() for signed integer tensors."""
    value = value.to(torch.int64)
    if int(shift) == 0:
        return value
    bias = 1 << (int(shift) - 1)
    scale = 1 << int(shift)
    # HLS uses round-away-from-zero. For negative integers this is a ceil after
    # subtracting the half-LSB bias, not an arithmetic floor shift.
    return torch.where(
        value >= 0,
        (value + bias) >> int(shift),
        ((value - bias) + scale - 1) >> int(shift),
    )


def clamp_i8(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(INT8_MIN, INT8_MAX)


def ste_replace(forward_value: torch.Tensor, gradient_value: torch.Tensor) -> torch.Tensor:
    """Use hard hardware value in forward while keeping gradient through input."""
    return gradient_value + (forward_value - gradient_value).detach()


def hls_quantize_i8(
    value: torch.Tensor,
    scale,
    relu: bool = False,
    quant_min: int = INT8_MIN,
    quant_max: int = INT8_MAX,
) -> torch.Tensor:
    """Quantize a dequantized activation tensor to signed INT8."""
    scale_t = _as_tensor(scale, device=value.device, dtype=value.dtype)
    q = hls_round_away_from_zero(value / scale_t)
    q = q.clamp(int(quant_min), int(quant_max))
    if relu:
        q = torch.clamp(q, min=0)
    return q


def hls_activation_quant_dequant(value: torch.Tensor, scale, relu: bool = False, ste: bool = False) -> torch.Tensor:
    """Signed INT8 activation quant/dequant using HLS rounding and clamp."""
    scale_t = _as_tensor(scale, device=value.device, dtype=value.dtype)
    q = hls_quantize_i8(value, scale_t, relu=relu)
    dequant = q * scale_t
    return ste_replace(dequant, value) if ste else dequant


def hls_add_bypass_dequant(a: torch.Tensor, b: torch.Tensor, scale, relu: bool = False, ste: bool = False) -> torch.Tensor:
    """Hardware ADD bypass: quantize operands, add in INT domain, clamp, dequant."""
    scale_t = _as_tensor(scale, device=a.device, dtype=a.dtype)
    qa = hls_quantize_i8(a, scale_t, relu=False).to(torch.int16)
    qb = hls_quantize_i8(b, scale_t, relu=False).to(torch.int16)
    qsum = clamp_i8(qa + qb).to(a.dtype)
    if relu:
        qsum = torch.clamp(qsum, min=0)
    dequant = qsum * scale_t
    return ste_replace(dequant, a + b) if ste else dequant


def hls_requant_i32_to_i8_tensor(
    acc_i64: torch.Tensor,
    bias_i32: torch.Tensor,
    mult_i32: torch.Tensor,
    shift_u8: torch.Tensor,
    relu: bool = False,
) -> torch.Tensor:
    """Per-output-channel HLS conv requant for NCHW integer accumulators."""
    if acc_i64.dim() != 4:
        raise ValueError(f"expected NCHW accumulator, got shape={tuple(acc_i64.shape)}")
    out_c = acc_i64.shape[1]
    if int(bias_i32.numel()) < out_c or int(mult_i32.numel()) < out_c or int(shift_u8.numel()) < out_c:
        raise ValueError("qparam vectors are shorter than output channels")

    acc_i64 = acc_i64.to(torch.int64)
    bias_i32 = bias_i32.to(device=acc_i64.device, dtype=torch.int64).reshape(-1)
    mult_i32 = mult_i32.to(device=acc_i64.device, dtype=torch.int64).reshape(-1)
    shift_u8 = shift_u8.to(device=acc_i64.device, dtype=torch.int64).reshape(-1)

    out_channels = []
    for ch in range(out_c):
        scaled = (acc_i64[:, ch, :, :] + bias_i32[ch]) * mult_i32[ch]
        rounded = hls_round_shift_int(scaled, int(shift_u8[ch].item()))
        out_channels.append(rounded.clamp(INT8_MIN, INT8_MAX).to(torch.int16))
    out = torch.stack(out_channels, dim=1)
    if relu:
        out = torch.clamp(out, min=0)
    return out.to(torch.int8)


def hls_conv2d_i8_nchw(
    input_i8: torch.Tensor,
    weight_i8: torch.Tensor,
    bias_i32: torch.Tensor,
    mult_i32: torch.Tensor,
    shift_u8: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    relu: bool = False,
) -> torch.Tensor:
    """Diagnostic HLS-equivalent INT8 conv forward.

    Inputs and weights are signed INT8 tensors in PyTorch NCHW/OIHW layout.
    The multiply-accumulate is evaluated with float64 conv2d because all INT8
    products and sums are exactly representable in fp64; the result is converted
    back to int64 before applying the same fixed-point requant path as HLS.
    This helper is intended for correctness/QAT-contract checks, not for fast
    training throughput.
    """
    if input_i8.dim() != 4 or weight_i8.dim() != 4:
        raise ValueError(f"expected input NCHW and weight OIHW, got {tuple(input_i8.shape)} / {tuple(weight_i8.shape)}")
    acc_f64 = F.conv2d(
        input_i8.to(torch.float64),
        weight_i8.to(torch.float64),
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    acc_i64 = torch.round(acc_f64).to(torch.int64)
    return hls_requant_i32_to_i8_tensor(acc_i64, bias_i32, mult_i32, shift_u8, relu=relu)


def quantize_weight_i8_from_fake_quant(module) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a QAT Conv2d weight tensor using the module's fake-quant state."""
    weight_fq = getattr(module, "weight_fake_quant", None)
    if weight_fq is None or not hasattr(weight_fq, "scale"):
        raise ValueError(f"{type(module).__name__} has no weight_fake_quant scale")

    weight = module.weight
    scale = weight_fq.scale.detach().to(device=weight.device, dtype=weight.dtype).reshape(-1)
    zero_point = weight_fq.zero_point.detach().to(device=weight.device, dtype=weight.dtype).reshape(-1)
    quant_min = int(weight_fq.quant_min)
    quant_max = int(weight_fq.quant_max)

    if torch.any(zero_point != 0):
        raise ValueError("P7 hardware Conv2d expects symmetric weight zero_point=0")
    if scale.numel() == 1:
        qweight = hls_quantize_i8(weight / scale.reshape(()), 1.0, quant_min=quant_min, quant_max=quant_max)
    else:
        if scale.numel() != weight.shape[0]:
            raise ValueError(f"weight scale length {scale.numel()} does not match out_channels {weight.shape[0]}")
        view_shape = [scale.numel()] + [1] * (weight.dim() - 1)
        qweight = hls_quantize_i8(weight / scale.view(*view_shape), 1.0, quant_min=quant_min, quant_max=quant_max)
    return qweight.to(torch.int8), scale


def hls_conv2d_hw_dequant(
    input_float: torch.Tensor,
    module,
    input_scale,
    output_scale,
    relu: bool = False,
    gradient_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a QAT Conv2d with the same integer arithmetic used by P7 HLS.

    The forward value is hardware-equivalent INT8 conv -> fixed-point requant
    -> dequant. ``gradient_value`` may be the original QAT conv output to keep a
    straight-through gradient path during QAT.
    """
    input_scale_t = _as_tensor(input_scale, device=input_float.device, dtype=input_float.dtype).reshape(())
    output_scale_t = _as_tensor(output_scale, device=input_float.device, dtype=input_float.dtype).reshape(())
    if float(input_scale_t.detach().cpu()) <= 0.0 or float(output_scale_t.detach().cpu()) <= 0.0:
        raise ValueError("Conv2d input/output scales must be positive")

    qinput = hls_quantize_i8(input_float, input_scale_t).to(torch.int8)
    qweight, weight_scale = quantize_weight_i8_from_fake_quant(module)
    weight_scale = weight_scale.to(device=input_float.device, dtype=torch.float64).reshape(-1)

    out_c = int(module.weight.shape[0])
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.repeat(out_c)
    if weight_scale.numel() != out_c:
        raise ValueError(f"weight_scale length {weight_scale.numel()} != out_c {out_c}")

    bias_fp = getattr(module, "bias", None)
    if bias_fp is None:
        bias_fp64 = torch.zeros(out_c, device=input_float.device, dtype=torch.float64)
    else:
        bias_fp64 = bias_fp.detach().to(device=input_float.device, dtype=torch.float64).reshape(-1)
        if bias_fp64.numel() != out_c:
            raise ValueError(f"bias length {bias_fp64.numel()} != out_c {out_c}")

    input_scale_f = float(input_scale_t.detach().cpu())
    output_scale_f = float(output_scale_t.detach().cpu())
    mult = []
    shift = []
    bias_i32 = []
    for ch in range(out_c):
        scale_acc = input_scale_f * float(weight_scale[ch].detach().cpu())
        real_multiplier = scale_acc / output_scale_f
        m, s = choose_multiplier(real_multiplier)
        mult.append(m)
        shift.append(s)
        bias_i32.append(round_away_float_scalar(float(bias_fp64[ch].detach().cpu()) / scale_acc))

    qout = hls_conv2d_i8_nchw(
        qinput,
        qweight,
        torch.tensor(bias_i32, device=input_float.device, dtype=torch.int32),
        torch.tensor(mult, device=input_float.device, dtype=torch.int32),
        torch.tensor(shift, device=input_float.device, dtype=torch.uint8),
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        relu=relu,
    )
    dequant = qout.to(dtype=input_float.dtype) * output_scale_t
    if gradient_value is not None and gradient_value.requires_grad:
        return ste_replace(dequant, gradient_value)
    return dequant


def observer_scale(observer) -> torch.Tensor | None:
    if observer is None or not hasattr(observer, "scale"):
        return None
    scale = observer.scale.detach()
    if scale.numel() != 1:
        return None
    return scale.reshape(()).clone()


@dataclass
class HardwareHookReport:
    activation_hooks: int = 0
    conv_patches: int = 0
    add_patches: int = 0
    cat_patches: int = 0
    fused_relu_outputs: tuple[str, ...] = ()
    conv_relu_outputs: tuple[str, ...] = ()


class P7HardwarePrecisionHandle:
    def __init__(self):
        self.hooks = []
        self.patches = []
        self.report = HardwareHookReport()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        for module, attr_name, original in self.patches:
            setattr(module, attr_name, original)
        self.hooks.clear()
        self.patches.clear()


def _get_nested_module(model, dotted_name: str):
    module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def install_p7_hardware_precision_hooks(
    model,
    fused_relu_outputs: Iterable[str] = (),
    conv_input_scale_map: Mapping[str, str] | None = None,
    conv_relu_outputs: Iterable[str] = (),
):
    """Install P7 hardware-precision hooks on a prepared QAT model.

    This does not replace the network topology. It clamps and quantizes module
    boundary tensors with the same signed INT8 rounding/clamp contract used by
    the HLS NPU. When ``conv_input_scale_map`` is supplied, Conv2d forward also
    uses the HLS integer MAC + fixed-point requant path.
    """
    if getattr(model, "_p7_hw_precision_handle", None) is not None:
        return model._p7_hw_precision_handle

    fused_relu_set = set(fused_relu_outputs)
    conv_input_scale_map = dict(conv_input_scale_map or {})
    conv_relu_set = set(conv_relu_outputs)
    handle = P7HardwarePrecisionHandle()
    handle.report.fused_relu_outputs = tuple(sorted(fused_relu_set))
    handle.report.conv_relu_outputs = tuple(sorted(conv_relu_set))

    def make_activation_hook(name):
        def hook(module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return output
            scale = observer_scale(getattr(module, "activation_post_process", None))
            if scale is None:
                return output
            relu = name in fused_relu_set
            return hls_activation_quant_dequant(output, scale.to(output.device), relu=relu, ste=output.requires_grad)
        return hook

    def maybe_patch_conv(name, module):
        if name not in conv_input_scale_map:
            return
        if not isinstance(module, torch.nn.Conv2d) or not hasattr(module, "weight_fake_quant"):
            raise TypeError(f"{name}: expected QAT Conv2d with weight_fake_quant, got {type(module).__name__}")

        input_scale_module = _get_nested_module(model, conv_input_scale_map[name])
        original_forward = module.forward

        def wrapped_forward(
            x,
            _orig=original_forward,
            _module=module,
            _input_scale_module=input_scale_module,
            _name=name,
        ):
            del _orig
            gradient_out = F.conv2d(
                x,
                _module.weight,
                _module.bias,
                stride=_module.stride,
                padding=_module.padding,
                dilation=_module.dilation,
                groups=_module.groups,
            )
            input_scale = observer_scale(getattr(_input_scale_module, "activation_post_process", None))
            output_scale = observer_scale(getattr(_module, "activation_post_process", None))
            if input_scale is None or output_scale is None:
                return gradient_out
            return hls_conv2d_hw_dequant(
                x,
                _module,
                input_scale.to(x.device),
                output_scale.to(x.device),
                relu=_name in conv_relu_set,
                gradient_value=gradient_out,
            )

        module.forward = wrapped_forward
        handle.patches.append((module, "forward", original_forward))
        handle.report.conv_patches += 1

    for name, module in model.named_modules():
        observer = getattr(module, "activation_post_process", None)
        if observer is None:
            continue

        maybe_patch_conv(name, module)

        handle.hooks.append(module.register_forward_hook(make_activation_hook(name)))
        handle.report.activation_hooks += 1

        if type(module).__name__ == "FloatFunctional":
            if hasattr(module, "add"):
                original_add = module.add

                def wrapped_add(x, y, _orig=original_add, _module=module):
                    out = _orig(x, y)
                    scale = observer_scale(getattr(_module, "activation_post_process", None))
                    if scale is None:
                        return out
                    return hls_add_bypass_dequant(x, y, scale.to(out.device), ste=out.requires_grad)

                module.add = wrapped_add
                handle.patches.append((module, "add", original_add))
                handle.report.add_patches += 1

            if hasattr(module, "cat"):
                original_cat = module.cat

                def wrapped_cat(x, dim=0, _orig=original_cat, _module=module):
                    out = _orig(x, dim)
                    scale = observer_scale(getattr(_module, "activation_post_process", None))
                    if scale is None:
                        return out
                    q_inputs = [
                        hls_activation_quant_dequant(t, scale.to(t.device), ste=t.requires_grad)
                        if isinstance(t, torch.Tensor) else t
                        for t in x
                    ]
                    return hls_activation_quant_dequant(torch.cat(q_inputs, dim), scale.to(out.device), ste=out.requires_grad)

                module.cat = wrapped_cat
                handle.patches.append((module, "cat", original_cat))
                handle.report.cat_patches += 1

    model._p7_hw_precision_handle = handle
    return handle
