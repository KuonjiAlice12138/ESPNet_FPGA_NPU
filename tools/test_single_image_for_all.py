# 2025.09.02 单张图片测试脚本
import loadData as ld
import os
import torch
import pickle
import Model_flattened_quantized as net
import copy
from torch.autograd import Variable
import VisualizeGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval_total import iouEval
import torch.optim.lr_scheduler
import torch.quantization
import torch.nn as nn
from torch.quantization import enable_observer, disable_observer
from torch.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize, FusedMovingAvgObsFakeQuantize
from torch.nn.intrinsic.qat import freeze_bn_stats
import numpy as np
import functools # 确保导入
from torch.utils.data import Subset
from hw_int8_math import install_p7_hardware_precision_hooks


HARDWARE_SCALE_GROUPS = [
    ["level1_conv", "b1_cat_ff"],
    [
        "level2_0_d1", "level2_0_d2", "level2_0_d4", "level2_0_d8",
        "level2_0_d16", "level2_0_add2_ff", "level2_0_add3_ff",
        "level2_0_add4_ff", "level2_0_cat_ff",
    ],
    [
        "level2_0_bn", "level2_blocks.1", "level2_blocks.2",
        "level2_blocks.3", "level2_blocks.4", "level2_blocks.5",
        "level2_block_add2_ff.0", "level2_block_add3_ff.0",
        "level2_block_add4_ff.0", "level2_block_cat_ff.0",
        "level2_block_res_ff.0", "level2_blocks.6", "b2_cat_ff",
    ],
    [
        "level3_0_d1", "level3_0_d2", "level3_0_d4", "level3_0_d8",
        "level3_0_d16", "level3_0_add2_ff", "level3_0_add3_ff",
        "level3_0_add4_ff", "level3_0_cat_ff",
    ],
    [
        "level3_0_bn", "level3_blocks.1", "level3_blocks.2",
        "level3_blocks.3", "level3_blocks.4", "level3_blocks.5",
        "level3_block_add2_ff.0", "level3_block_add3_ff.0",
        "level3_block_add4_ff.0", "level3_block_cat_ff.0",
        "level3_block_res_ff.0", "level3_blocks.6", "b3_cat_ff",
    ],
]

HARDWARE_FUSED_RELU_OUTPUTS = {
    "b1_bn",
    "level2_0_bn",
    "level2_blocks.6",
    "b2_bn",
    "level3_0_bn",
    "level3_blocks.6",
    "b3_bn",
}

HARDWARE_CONV_INPUT_SCALE_MAP = {
    "level1_conv": "quant",
    "level2_0_c1": "b1_bn",
    "level2_0_d1": "level2_0_c1",
    "level2_0_d2": "level2_0_c1",
    "level2_0_d4": "level2_0_c1",
    "level2_0_d8": "level2_0_c1",
    "level2_0_d16": "level2_0_c1",
    "level2_blocks.0": "level2_0_bn",
    "level2_blocks.1": "level2_blocks.0",
    "level2_blocks.2": "level2_blocks.0",
    "level2_blocks.3": "level2_blocks.0",
    "level2_blocks.4": "level2_blocks.0",
    "level2_blocks.5": "level2_blocks.0",
    "level3_0_c1": "b2_bn",
    "level3_0_d1": "level3_0_c1",
    "level3_0_d2": "level3_0_c1",
    "level3_0_d4": "level3_0_c1",
    "level3_0_d8": "level3_0_c1",
    "level3_0_d16": "level3_0_c1",
    "level3_blocks.0": "level3_0_bn",
    "level3_blocks.1": "level3_blocks.0",
    "level3_blocks.2": "level3_blocks.0",
    "level3_blocks.3": "level3_blocks.0",
    "level3_blocks.4": "level3_blocks.0",
    "level3_blocks.5": "level3_blocks.0",
    "classifier": "b3_bn",
}

HARDWARE_CONV_RELU_OUTPUTS = {
    "level1_conv",
}


def get_hardware_qconfig(qat_mode=True, symmetric_activations=True):
    if qat_mode:
        if symmetric_activations:
            activation = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                quant_min=-128,
                quant_max=127
            )
        else:
            activation = FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                quant_min=0,
                quant_max=255
            )

        weight = FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            quant_min=-127,
            quant_max=127
        )
        return QConfig(activation=activation, weight=weight)

    if symmetric_activations:
        activation = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            quant_min=-128,
            quant_max=127
        )
    else:
        activation = MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=255
        )

    weight = MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-127,
        quant_max=127
    )

    return QConfig(activation=activation, weight=weight)


def _get_nested_module(model, dotted_name):
    module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_fake_quant_qparams(fake_quant, scale=None, zero_point=None):
    if fake_quant is None:
        return
    if hasattr(fake_quant, "observer_enabled"):
        fake_quant.observer_enabled.detach().fill_(0)
    if hasattr(fake_quant, "fake_quant_enabled"):
        fake_quant.fake_quant_enabled.detach().fill_(1)
    if scale is not None and hasattr(fake_quant, "scale"):
        fake_quant.scale.detach().fill_(float(scale))
    if zero_point is not None and hasattr(fake_quant, "zero_point"):
        fake_quant.zero_point.detach().fill_(int(zero_point))


def apply_hardware_quant_constraints(model, scale_groups=HARDWARE_SCALE_GROUPS):
    """
    Enforce the INT8 NPU v1 quantization contract on a prepared fake-quant model.

    Hardware assumptions:
    - activation/weight zero-points are fixed to 0
    - concat-copy and ADD-bypass domains use a shared activation scale
    """
    for module in model.modules():
        _set_fake_quant_qparams(getattr(module, "activation_post_process", None), zero_point=0)
        _set_fake_quant_qparams(getattr(module, "weight_fake_quant", None), zero_point=0)

    reports = []
    for group in scale_groups:
        scales = []
        for name in group:
            fake_quant = getattr(_get_nested_module(model, name), "activation_post_process", None)
            if fake_quant is None or not hasattr(fake_quant, "scale"):
                raise RuntimeError(f"Missing activation fake-quant for hardware scale group member: {name}")
            scales.append(float(fake_quant.scale.detach().cpu().reshape(-1)[0].item()))

        target_scale = max(scales)
        for name in group:
            fake_quant = getattr(_get_nested_module(model, name), "activation_post_process", None)
            _set_fake_quant_qparams(fake_quant, scale=target_scale, zero_point=0)

        reports.append({
            "target_scale": target_scale,
            "members": list(group),
            "old_scales": scales,
        })

    return reports


def fuse_conv_bn(conv, bn, start_idx=None, end_idx=None):
    """
    Fuse a Conv2d layer with a subset or all of BatchNorm2d parameters.
    """
    gamma = bn.weight[start_idx:end_idx] if start_idx is not None else bn.weight
    beta = bn.bias[start_idx:end_idx] if start_idx is not None else bn.bias
    mu = bn.running_mean[start_idx:end_idx] if start_idx is not None else bn.running_mean
    var = bn.running_var[start_idx:end_idx] if start_idx is not None else bn.running_var
    eps = bn.eps
    
    scale = gamma / torch.sqrt(var + eps)
    shift = beta - mu * scale
    
    fused_weight = conv.weight * scale.view(-1, 1, 1, 1)
    fused_bias = shift if conv.bias is None else conv.bias * scale + shift
    
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        bias=True
    )
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    fused_conv.eval()  # 显式设置为评估模式
    return fused_conv

def qat_train(model_prepared, train_loader, criterion, optimizer, epochs=1, hardware_constraints=True):
    """
    执行量化感知训练
    :param model_prepared: 已经准备好的QAT模型
    :param train_loader: 训练数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epochs: 训练轮数
    """
    model_prepared.train()
    device = next(model_prepared.parameters()).device
    if hardware_constraints:
        hook_handle = install_p7_hardware_precision_hooks(
            model_prepared,
            HARDWARE_FUSED_RELU_OUTPUTS,
            HARDWARE_CONV_INPUT_SCALE_MAP,
            HARDWARE_CONV_RELU_OUTPUTS,
        )
        print(
            "P7 hardware precision hooks active: "
            f"activation={hook_handle.report.activation_hooks}, "
            f"conv={hook_handle.report.conv_patches}, "
            f"add={hook_handle.report.add_patches}, "
            f"cat={hook_handle.report.cat_patches}"
        )
    for epoch in range(epochs):
        print(f"\n[QAT Epoch {epoch+1}/{epochs}]")
        
        # 硬件约束版 QAT 使用校准后固定的共享 scale 训练权重，避免 ADD/concat 域在训练中漂移。
        if hardware_constraints:
            disable_observer(model_prepared)
            model_prepared.apply(freeze_bn_stats)
            apply_hardware_quant_constraints(model_prepared)
        elif epoch < max(1, epochs // 2):
            enable_observer(model_prepared)
        else:
            disable_observer(model_prepared)
            model_prepared.apply(freeze_bn_stats)
        
        total_loss = 0.0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            target_var = torch.autograd.Variable(targets)
            target_var[((target_var < 13) | (target_var > 18)) & (target_var != 255)] = 1
            target_var[(target_var >= 13)&(target_var <= 18)] = 0

            optimizer.zero_grad()
            
            # 前向传播
            outputs = model_prepared(inputs)
            loss = criterion(outputs, target_var)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            if hardware_constraints:
                apply_hardware_quant_constraints(model_prepared)
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item()}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

def prepare_quantized_model(model, calibration_loader, qat_mode=False, train_loader=None,
                            calibration_batches=11, qat_epochs=3, qat_lr=1e-4,
                            symmetric_activations=True, convert_model=True,
                            hardware_constraints=True):
    try:
        print("Starting quantization process...")
        model.eval()
        device = next(model.parameters()).device
        
        use_fake_quant_emulation = not convert_model

        # 配置量化参数
        model.qconfig = get_hardware_qconfig(
            qat_mode=qat_mode or use_fake_quant_emulation,
            symmetric_activations=symmetric_activations
        )
        
        # 融合模型
        try:
            model.fuse_model()
        except Exception as e:
            print(f"Warning: Model fusion failed: {str(e)}")
        
        # 准备量化
        if qat_mode or use_fake_quant_emulation:
            model_prepared = torch.quantization.prepare_qat(model.train(), inplace=False)
            if not qat_mode:
                model_prepared = model_prepared.eval()
        else:
            model_prepared = torch.quantization.prepare(model.eval(), inplace=False)
        
        # 校准阶段
        print("Running calibration...")
        with torch.no_grad():
            for batch_idx, data in enumerate(calibration_loader):
                inputs, targets, _ = data
                model_prepared(inputs.to(device, non_blocking=True))
                if batch_idx + 1 >= calibration_batches:
                    break

        if hardware_constraints and symmetric_activations:
            print("Applying hardware quantization constraints after calibration...")
            apply_hardware_quant_constraints(model_prepared)
            if qat_mode or use_fake_quant_emulation:
                hook_handle = install_p7_hardware_precision_hooks(
                    model_prepared,
                    HARDWARE_FUSED_RELU_OUTPUTS,
                    HARDWARE_CONV_INPUT_SCALE_MAP,
                    HARDWARE_CONV_RELU_OUTPUTS,
                )
                print(
                    "P7 hardware precision hooks installed after calibration: "
                    f"activation={hook_handle.report.activation_hooks}, "
                    f"conv={hook_handle.report.conv_patches}, "
                    f"add={hook_handle.report.add_patches}, "
                    f"cat={hook_handle.report.cat_patches}"
                )

        # 量化感知训练阶段
        if qat_mode and train_loader:
            print("\nStarting Quantization-Aware Training...")
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            optimizer = torch.optim.Adam(model_prepared.parameters(), lr=qat_lr)
            
            qat_train(
                model_prepared,
                train_loader,
                criterion,
                optimizer,
                epochs=qat_epochs,
                hardware_constraints=hardware_constraints and symmetric_activations,
            )
        
        # 转换为量化模型
        if convert_model:
            print("Converting to quantized model...")
            model_quantized = torch.quantization.convert(model_prepared.cpu().eval(), inplace=False)
            print("Quantization process completed successfully")
            return model_quantized

        print("Freezing fake-quant parameters for evaluation...")
        disable_observer(model_prepared)
        model_prepared.apply(freeze_bn_stats)
        if hardware_constraints and symmetric_activations:
            apply_hardware_quant_constraints(model_prepared)
        model_prepared = model_prepared.eval().to(device)
        print("Fake-quantized model is ready for evaluation")
        return model_prepared
        
    except Exception as e:
        print(f"Critical error in prepare_quantized_model: {str(e)}")
        raise

# --- 工具函数：保存张量到文件 ---
def dump_tensor_to_file(tensor, filename, is_quantized=False):
    """
    将 PyTorch 张量数据保存到文本文件。
    """
    try:
        if is_quantized:
            # 对于量化张量，获取其底层整数表示
            data = tensor.int_repr().cpu().detach().flatten().tolist()
            with open(filename, 'w') as f:
                f.write(' '.join(map(str, data)))
        else:
            # 对于浮点张量，将其转换为 NumPy 数组并保存
            data = tensor.cpu().detach().numpy().flatten()
            np.savetxt(filename, data, fmt='%.6f')
        print(f"数据已成功保存到 {filename}")
    except Exception as e:
        print(f"保存数据失败: {e}")

# --- 功能集成函数：捕获并保存指定卷积层的量化信息 ---
def dump_all_quant_tensors(model, target_conv_name):
    """
    统一保存目标卷积层 (由参数指定) 的量化输入、量化权重、Scale、Zero-Point 和量化输出。
    :param model: 量化后的模型
    :param target_conv_name: 目标卷积层的名称字符串 (例如 'level2_0_c1' 或 'level2_blocks.3')
    """
    # 将名称中的'.'替换为'_'以便创建文件夹
    output_dir_name = target_conv_name.replace('.', '_')
    output_dir = f'./model_quantization_logs/{output_dir_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- 正在为 '{target_conv_name}' 保存量化数据 ---")

    conv_module = None

    # 1. 遍历模型，找到目标量化卷积模块 (使用完全匹配来精确定位)
    for name, module in model.named_modules():
        if name == target_conv_name and hasattr(module, '_packed_params'):
            conv_module = module
            break
    
    if conv_module is None:
        print(f"警告: 未在模型中找到名为 '{target_conv_name}' 的量化卷积模块，无法保存数据。")
        return

    # 2. 保存量化权重、Scale 和 Zero-Point
    print(f"保存 {target_conv_name} 的量化权重、Scale和Zero-Point...")
    weight_quant = conv_module.weight() 
    
    dump_tensor_to_file(weight_quant, os.path.join(output_dir, 'quant_weights.txt'), is_quantized=True)
    
    if weight_quant.qscheme() == torch.per_channel_symmetric or weight_quant.qscheme() == torch.per_channel_affine:
        scales = weight_quant.q_per_channel_scales()
        scales_path = os.path.join(output_dir, 'pytorch_weight_scales.txt')
        dump_tensor_to_file(scales, scales_path, is_quantized=False)
        
        zero_points = weight_quant.q_per_channel_zero_points()
        zps_path = os.path.join(output_dir, 'pytorch_zero_points.txt')
        dump_tensor_to_file(zero_points, zps_path, is_quantized=True)
    else:
        print(f"权重为 Per-Tensor 量化。Scale: {weight_quant.q_scale()}, Zero-Point: {weight_quant.q_zero_point()}")

    # 3. 注册前向钩子来捕获输入和输出特征图
    conv_hook_handle = None
    def dump_conv_output_hook(module, input, output):
        input_tensor = input[0]

        # --- 获取并保存输入特征图的量化参数 ---
        if input_tensor.is_quantized:
            input_scale = input_tensor.q_scale()
            input_zp = input_tensor.q_zero_point()
            print(f"捕获到 '{target_conv_name}' 的输入量化参数: Scale={input_scale}, Zero-Point={input_zp}")
            quant_params_path = os.path.join(output_dir, 'input_quant_params.txt')
            with open(quant_params_path, 'w') as f:
                f.write(f"Scale: {input_scale}\n")
                f.write(f"Zero-Point: {input_zp}\n")
            print(f"输入量化参数已保存到: {quant_params_path}")
        else:
            print("警告: 捕获到的输入张量不是一个量化张量。")
        
        # --- 获取并保存输出特征图的量化参数 ---
        if output.is_quantized:
            output_scale = output.q_scale()
            output_zp = output.q_zero_point()
            print(f"捕获到 '{target_conv_name}' 的输出量化参数: Scale={output_scale}, Zero-Point={output_zp}")
            output_params_path = os.path.join(output_dir, 'output_quant_params.txt')
            with open(output_params_path, 'w') as f:
                f.write(f"Scale: {output_scale}\n")
                f.write(f"Zero-Point: {output_zp}\n")
            print(f"输出量化参数已保存到: {output_params_path}")
        else:
            print("警告: 捕获到的输出张量不是一个量化张量。")
        
        print("捕获并保存量化后的输入特征图...")
        dump_tensor_to_file(input_tensor, os.path.join(output_dir, 'quant_input.txt'), is_quantized=True)
        
        print(f"捕获并保存 '{target_conv_name}' 的量化输出特征图...")
        dump_tensor_to_file(output, os.path.join(output_dir, 'quant_output.txt'), is_quantized=True)
        
        if conv_hook_handle:
            conv_hook_handle.remove()

    conv_hook_handle = conv_module.register_forward_hook(dump_conv_output_hook)

    print(f"成功为模块 '{conv_module._get_name()}' 注册钩子。模型进行前向推理后将自动保存数据。")

def test_model(args, model, testLoader, iou_eval, model_type="Original"):
    """
    Function to test the model and evaluate its performance.
    """
    model.eval()

    total_infer_time = 0.0
    total_images = 0

    with torch.no_grad():
        for i, data in enumerate(testLoader):
            inputs, targets, filenames = data
            target_var = torch.autograd.Variable(targets)
            target_var[((target_var < 13) | (target_var > 18)) & (target_var != 255)] = 1
            target_var[(target_var >= 13)&(target_var <= 18)] = 0

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            if args.onGPU:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            iou_eval.addBatch(preds, target_var.data)

            end_event.record()
            torch.cuda.synchronize()

            infer_time = start_event.elapsed_time(end_event) / 1000
            total_infer_time += infer_time
            total_images += inputs.size(0)

    avg_infer_time = total_infer_time / total_images
    print(f"[{model_type}] Final average inference time per image: {avg_infer_time} seconds")

    overall_acc, per_class_acc, per_class_iu, mIoU = iou_eval.getMetric()

    print(f'[{model_type}] Final Results after Testing:')
    print(f'Overall Accuracy: {overall_acc}')
    print(f'Mean IoU: {mIoU}')
    print(f'Per-Class Accuracy: {per_class_acc}')
    print(f'Per-Class IoU: {per_class_iu}')

def test_model_on_cpu(args, model, testLoader, iou_eval, model_type="Original"):
    """
    执行模型测试的完整流程（兼容CPU/量化模型）
    """
    device = torch.device("cpu")
    model = model.to(device).eval()
    
    iou_eval.reset()
    total_inference_time = 0.0
    processed_images = 0
    total_batches = len(testLoader)
    
    print(f"\n[{model_type}] Start testing, total batches: {total_batches} ...")
    
    # --- MODIFIED LOGIC: 动态设置钩子 ---
    dump_enabled = args.dump_layer_data
    
    if dump_enabled and args.probe_layer:
        print(f"\n[INFO] Setting up data dump hook for layer: {args.probe_layer}")
        dump_all_quant_tensors(model, args.probe_layer)
    elif dump_enabled and not args.probe_layer:
        print("\n[WARNING] Data dumping was enabled, but --probe_layer was not specified. No hook-based data will be saved.")
    # --- END MODIFIED LOGIC ---

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(testLoader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_var = torch.autograd.Variable(targets)
            target_var[((target_var < 13) | (target_var > 18)) & (target_var != 255)] = 1
            target_var[(target_var >= 13)&(target_var <= 18)] = 0
            
            batch_start = time.time()
            outputs = model(inputs)
            batch_time = time.time() - batch_start
            
            _, preds = torch.max(outputs, 1)
            iou_eval.addBatch(preds, target_var.data)
            
            batch_size = inputs.size(0)
            processed_images += batch_size
            total_inference_time += batch_time
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                overall_acc, per_class_acc, per_class_iu, mIoU = iou_eval.getMetric()
                avg_time_per_image = total_inference_time / processed_images
                images_per_sec = processed_images / total_inference_time
                
                print(
                    f"[{model_type}] Batch: {batch_idx+1}/{total_batches} | "
                    f"mIoU: {mIoU} | "
                    f"Average Inference Time per Image: {avg_time_per_image}s | "
                    f"Throughput: {images_per_sec:.2f} img/s"
                )

    overall_acc, per_class_acc, per_class_iu, mIoU = iou_eval.getMetric()
    avg_inference_time = total_inference_time / processed_images
    
    print(f"[{model_type}] Test finished:")
    print(f"Total images: {processed_images}")
    print(f"Average Inference Time per Image: {avg_inference_time}s")
    print(f"mIoU: {mIoU}")
    print(f"Overall Acc.: {overall_acc:.2%}")
    print(f"Per-Class mIoU: {per_class_iu}")
    print(f"Per-Class Acc.: {per_class_acc}")
    
    return {
        "mIoU": mIoU,
        "accuracy": overall_acc,
        "inference_time": avg_inference_time
    }


def test_model_on_device(args, model, testLoader, iou_eval, model_type="FakeQuantized", device=None):
    if device is None:
        if args.onGPU and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = model.to(device).eval()
    iou_eval.reset()
    total_inference_time = 0.0
    processed_images = 0
    total_batches = len(testLoader)

    print(f"\n[{model_type}] Start testing on {device}, total batches: {total_batches} ...")

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(testLoader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_var = torch.autograd.Variable(targets)
            target_var[((target_var < 13) | (target_var > 18)) & (target_var != 255)] = 1
            target_var[(target_var >= 13)&(target_var <= 18)] = 0

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs = model(inputs)
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                batch_start = time.time()
                outputs = model(inputs)
                batch_time = time.time() - batch_start

            _, preds = torch.max(outputs, 1)
            iou_eval.addBatch(preds, target_var.data)

            batch_size = inputs.size(0)
            processed_images += batch_size
            total_inference_time += batch_time

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                overall_acc, per_class_acc, per_class_iu, mIoU = iou_eval.getMetric()
                avg_time_per_image = total_inference_time / processed_images
                images_per_sec = processed_images / total_inference_time
                print(
                    f"[{model_type}] Batch: {batch_idx+1}/{total_batches} | "
                    f"mIoU: {mIoU} | "
                    f"Average Inference Time per Image: {avg_time_per_image}s | "
                    f"Throughput: {images_per_sec:.2f} img/s"
                )

    overall_acc, per_class_acc, per_class_iu, mIoU = iou_eval.getMetric()
    avg_inference_time = total_inference_time / processed_images

    print(f"[{model_type}] Test finished:")
    print(f"Total images: {processed_images}")
    print(f"Average Inference Time per Image: {avg_inference_time}s")
    print(f"mIoU: {mIoU}")
    print(f"Overall Acc.: {overall_acc:.2%}")
    print(f"Per-Class mIoU: {per_class_iu}")
    print(f"Per-Class Acc.: {per_class_acc}")

    return {
        "mIoU": mIoU,
        "accuracy": overall_acc,
        "inference_time": avg_inference_time
    }

def trainValidateSegmentation(args):
    # --- HELPER FUNCTION to get nested attributes (e.g., 'level2_blocks.0') ---
    def rgetattr(obj, attr, *default):
        def _getattr(obj, attr):
            return getattr(obj, attr, *default)
        return functools.reduce(_getattr, [obj] + attr.split('.'))
    # ---

    if args.mode == 'test':
        if not os.path.isfile(args.cached_data_file):
            dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
            data = dataLoad.processData()
            if data is None:
                print('Error while pickling data. Please check.')
                exit(-1)
        else:
            data = pickle.load(open(args.cached_data_file, "rb"))

        q, p = args.q, args.p
        model = net.ESPNet_Encoder(args.classes, p=p, q=q)
        if args.onGPU:
            model = model.cuda()
        model_path = './results_vehicle(model_flattened)_enc__enc_1_1/model_64.pth'
        
        testDataset = myTransforms.Compose([
            myTransforms.Normalize(mean=data['mean'], std=data['std']),
            myTransforms.Scale(1024, 512),
            myTransforms.ToTensor(args.scaleIn),
        ])
        testLoader = torch.utils.data.DataLoader(
            myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=testDataset),
            batch_size=args.batch_size + 4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        single_test_image = "./city//leftImg8bit/val/lindau/lindau_000034_000019_leftImg8bit.png"
        single_image_index = data['valIm'].index(single_test_image)
        single_image_dataset = myDataLoader.MyDataset([data['valIm'][single_image_index]],[data['valAnnot'][single_image_index]], transform=testDataset)
        single_image_loader = torch.utils.data.DataLoader(single_image_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        trainDataset_main = myTransforms.Compose([
            myTransforms.Normalize(mean=data['mean'], std=data['std']),
            myTransforms.Scale(1024, 512),
            myTransforms.RandomCropResize(32),
            myTransforms.RandomFlip(),
            myTransforms.ToTensor(args.scaleIn),
        ])
        train_dataset = myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main)
        if args.qat_train_subset > 0:
            subset_len = min(args.qat_train_subset, len(train_dataset))
            train_dataset = Subset(train_dataset, list(range(subset_len)))
        trainLoader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size + 2, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        iou_eval = iouEval(nClasses=args.classes)

        if args.test_original:
            if os.path.isfile(model_path):
                print(f"=> loading model from '{model_path}'")
                model.load_state_dict(torch.load(model_path))
                print(f"=> model loaded from '{model_path}'")
            else:
                print(f"=> no model found at '{model_path}'"); exit(-1)
            print("\n[1/4] Testing Original Model...")
            test_model(args, model, testLoader, iou_eval, model_type="Original")

        if args.quantize:
            dump_enabled = args.dump_layer_data
            use_fake_quant_eval = args.fake_quant_eval or not args.asymmetric_activations
            
            if dump_enabled and args.probe_layer:
                target_layer_name = args.probe_layer
                log_dir_name = target_layer_name.replace('.', '_')
                log_dir = f'./model_quantization_logs/{log_dir_name}'
                os.makedirs(log_dir, exist_ok=True)
                
                # ======================= 保存原始 FP32 权重 =======================
                try:
                    print(f"\n[INFO] Saving original FP32 weights for '{target_layer_name}'...")
                    original_conv_layer = rgetattr(model, target_layer_name)
                    original_weight_tensor = original_conv_layer.weight
                    
                    fp32_weight_path = os.path.join(log_dir, 'original_weights_fp32.txt')
                    dump_tensor_to_file(original_weight_tensor, fp32_weight_path, is_quantized=False)
                    print("[INFO] Original FP32 weights saved successfully.")
                except AttributeError:
                     print(f"[ERROR] Could not find layer '{target_layer_name}' on the FP32 model.")
                except Exception as e:
                    print(f"[ERROR] Failed to save original FP32 weights: {e}")
                # ===========================================================================

            model.load_state_dict(torch.load(model_path))
            quant_device = torch.device(
                "cuda" if args.onGPU and torch.cuda.is_available() and (args.qat or use_fake_quant_eval) else "cpu"
            )
            model = model.to(quant_device).eval()
            print("Quantizing Model...")
            model_quant = prepare_quantized_model(
                model, 
                calibration_loader=testLoader,
                qat_mode=args.qat,
                train_loader=trainLoader,
                calibration_batches=args.calibration_batches,
                qat_epochs=args.qat_epochs,
                qat_lr=args.qat_lr,
                symmetric_activations=not args.asymmetric_activations,
                convert_model=not use_fake_quant_eval
            )

            if args.single_image:
                print("\n[INFO] Testing on Single Image...")

                if dump_enabled and args.probe_layer:
                    target_layer_name = args.probe_layer
                    log_dir_name = target_layer_name.replace('.', '_')
                    log_dir = f'./model_quantization_logs/{log_dir_name}'
                    
                    print(f"\n[INFO] Checking bias for quantized layer '{target_layer_name}'...")
                    try:
                        conv_layer = rgetattr(model_quant, target_layer_name)
                        if conv_layer.bias() is None:
                            print(f"量化后 {target_layer_name} bias 为 None")
                        else:
                            bias_tensor = conv_layer.bias()
                            print(f"量化后 {target_layer_name} bias 存在，形状:", bias_tensor.shape)
                            if torch.all(bias_tensor == 0):
                                print(f"量化后 {target_layer_name} bias 全为零值")
                            else:
                                print(f"量化后 {target_layer_name} bias 非零值:", bias_tensor)
                                bias_data = bias_tensor.cpu().detach().numpy().flatten()
                                bias_path = os.path.join(log_dir, 'quant_bias.txt')
                                np.savetxt(bias_path, bias_data, fmt='%.18e')
                                print(f"量化偏置已成功保存到 {bias_path}")
                    except AttributeError:
                        print(f"[ERROR] Could not find layer '{target_layer_name}' on the quantized model.")
                
                test_model_on_cpu(args, model_quant, single_image_loader, iou_eval, "Quantized_Single_Image")
            else:
                print("\n[INFO] Testing on full validation set...")
                if use_fake_quant_eval:
                    device = torch.device("cuda" if args.onGPU and torch.cuda.is_available() else "cpu")
                    test_model_on_device(args, model_quant, testLoader, iou_eval, "FakeQuantized", device=device)
                else:
                    test_model_on_cpu(args, model_quant, testLoader, iou_eval, "Quantized")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./city", help='Data directory')
    parser.add_argument('--scaleIn', type=int, default=8, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. Change as per the GPU memory')
    parser.add_argument('--classes', type=int, default=2, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=False,help='True if ESPNet. False for ESPNet-C')
    parser.add_argument('--pretrained', default='../pretrained/encoder/espnet_p_2_q_8.pth', help='Pretrained ESPNet-C weights. Only used when training ESPNet')
    parser.add_argument('--p', default=1, type=int, help='depth multiplier')
    parser.add_argument('--q', default=1, type=int, help='depth multiplier')

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'validate', 'test'], help='Mode of operation: train, validate, or test')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model')
    parser.add_argument('--qat', action='store_true', help='Enable Quantization-Aware Training')
    parser.add_argument('--test_original', action='store_true', help='Test the original (non-quantized) model')
    parser.add_argument('--output_layer_info', action='store_true', help='Print layer information')
    parser.add_argument('--single_image', action='store_true', help='Test on a single image')
    
    # --- 新增和修改的参数 ---
    parser.add_argument('--dump_layer_data', action='store_true', help='Enable dumping data for a specific layer specified by --probe_layer.')
    parser.add_argument('--probe_layer', type=str, default=None, help='The name of the layer to probe (e.g., "level2_0_c1", "level2_blocks.3"). Requires --dump_layer_data.')
    parser.add_argument('--calibration_batches', type=int, default=11, help='Number of batches used for PTQ/QAT calibration.')
    parser.add_argument('--qat_epochs', type=int, default=3, help='Number of QAT fine-tuning epochs.')
    parser.add_argument('--qat_lr', type=float, default=1e-4, help='Learning rate used during QAT.')
    parser.add_argument('--qat_train_subset', type=int, default=0, help='Use only the first N training samples for QAT debugging. 0 means full training set.')
    parser.add_argument('--asymmetric_activations', action='store_true', help='Use asymmetric activation quantization. Default is symmetric for hardware compatibility.')
    parser.add_argument('--fake_quant_eval', action='store_true', help='Keep the prepared model in fake-quant form for evaluation instead of converting to backend quantized kernels.')

    trainValidateSegmentation(parser.parse_args())
