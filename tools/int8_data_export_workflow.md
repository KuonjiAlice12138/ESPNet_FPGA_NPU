# ESPNet INT8 Data Export Workflow

本文档说明从“带硬件约束的 QAT”到生成 FPGA/NPU 可用 `hw_artifacts` 的完整数据处理流程。

## 1. 脚本角色

`D:\ESPNet\export_quantized_artifacts.py`

主入口。负责从 FP32 checkpoint 开始执行带硬件约束的 QAT，并导出模型侧 artifact：

- `fake_quant_state_dict.pth`
- `activation_qparams.json`
- `layers/*/weight_int8.npy`
- `layers/*/weight_scales.npy`
- `layers/*/weight_zero_points.npy`
- `golden_sample/*/output_int.npy`
- `manifest.json`

该脚本当前默认启用硬件约束：

- activation/weight zero-point 固定为 `0`
- ADD bypass 域共享 activation scale
- concat/store-copy 域共享 activation scale
- QAT 训练中持续冻结并重新施加这些约束

`D:\ESP_INT8\tools\export_int8_hw_blob.py`

硬件二进制导出器。读取模型侧 artifact，生成 HLS/PS/SD 卡使用的最终数据：

- `param_blob.bin`
- `input_q.bin`
- `golden_output_q.bin`
- `uop_table.bin`
- `tensor_desc_table.bin`
- `scale_table.json`
- `export_manifest.json`

运行时必须使用 `--strict-zp`。若出现 warning，说明当前 artifact 不能作为 bit-level 硬件比对基准。

当前导出器已经适配最新 HLS memory contract：

- 共享物理 `FMBUF`，不再按 `FMEM0/1/2` 导出三份独立大 buffer
- `tensor_desc.reserved0 = physical_c_stride`
- `tensor_desc.reserved1 = channel_offset`
- `T_L20_*` 是 `B2` 中间 64 通道 view，`phys_c=131, channel_offset=64`
- `T_L2B0_*` 是 `B2` 前 64 通道 view，`phys_c=131`
- `T_L3B0_*` 是 `B3` 后 128 通道 view，`phys_c=256, channel_offset=128`
- `T_POOL2` 使用独立 `BANK_BRAM_SCR1`
- `POOL_TMP -> POOL2` 两级 pool uop 提前到 B1 affine 之后，避免后续 `L20/L30` 覆盖共享物理区后再读取 pool 临时数据

`D:\ESP_INT8\tools\eval_hw_constrained_qat.py`

INT8 software baseline 评估脚本。加载带硬件约束 QAT 后的 `fake_quant_state_dict.pth`，在完整 val set 上计算：

- PA / pixel accuracy
- mIoU
- per-class accuracy
- per-class IoU

结果保存为 `int8_baseline_metrics.json`。

`D:\ESP_INT8\tools\compare_hls_top_output.py`

Top 级 HLS C model 输出对比脚本。读取 `golden_output_q.bin`、`target.npy` 和 HLS TB 写出的 `hls_output_q.bin`，统计：

- logits 逐 byte 差异
- `argmax` 后的 mask mismatch
- golden/HLS mask 各自相对 target 的 PA 和 mIoU
- HLS 相对 golden 的指标下降

`D:\ESPNet` 下脚本的边界

`D:\ESPNet\export_quantized_artifacts.py` 和 `test_single_image_for_all.py` 只负责硬件约束 QAT、fake-quant 推理和模型侧 golden dump。它们不编码片上 memory 地址或 uop schedule；硬件布局变化后通常不需要重跑 QAT，只需要重新执行 `export_int8_hw_blob.py` 生成新的 `param_blob/uop/tensor_desc`。

## 2. 推荐输出目录

模型侧 QAT artifact：

```bat
D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep
```

硬件侧 artifact：

```bat
D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single
```

当前硬件侧目录应至少包含：

- `param_blob.bin`
- `input_q.bin`
- `golden_output_q.bin`
- `scale_table.json`
- `export_manifest.json`
- `int8_baseline_metrics.json`

其中 `input_q.bin` 和 `golden_output_q.bin` 均为 NHWC INT8 layout。

## 3. 手动执行流程

### Step 1: 带约束 QAT 并导出模型侧 artifact

```bat
set PYTHONDONTWRITEBYTECODE=1
D:\ESPNet\.venv\Scripts\python.exe D:\ESPNet\export_quantized_artifacts.py ^
  --output_dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep ^
  --qat ^
  --fake_quant_eval ^
  --qat_epochs 3 ^
  --qat_lr 1e-4 ^
  --batch_size 4 ^
  --num_workers 0 ^
  --calibration_batches 11
```

成功后检查：

```bat
D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep\manifest.json
```

必须满足：

```json
"hardware_constraint_report": {
  "nonzero_zero_points": [],
  "scale_mismatches": [],
  "passed": true
}
```

### Step 2: 导出硬件可用 hw_artifacts

```bat
D:\ESPNet\.venv\Scripts\python.exe D:\ESP_INT8\tools\export_int8_hw_blob.py ^
  --artifact-dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep ^
  --out-dir D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single ^
  --strict-zp
```

成功条件：

```text
Warnings: 0
UOP count: 75
```

当前版本还应满足：

```text
uop[5]  = POOL  T_INPUT    -> T_POOL_TMP
uop[6]  = POOL  T_POOL_TMP -> T_POOL2
uop[36] = STORE T_L2B0_ACT -> T_B2_CAT, c_offset=0   # view/no-op check
uop[37] = STORE T_L20_ACT  -> T_B2_CAT, c_offset=64
uop[38] = STORE T_POOL2    -> T_B2_CAT, c_offset=128
uop[70] = STORE T_L3B0_ACT -> T_B3_CAT, c_offset=128 # view/no-op check
```

### Step 3: 评估 INT8 software baseline

```bat
D:\ESPNet\.venv\Scripts\python.exe D:\ESP_INT8\tools\eval_hw_constrained_qat.py ^
  --artifact-dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep ^
  --out-json D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single\int8_baseline_metrics.json ^
  --batch-size 4 ^
  --num-workers 0 ^
  --progress-every 25
```

当前已跑通版本的 baseline：

```text
PA   = 0.9818317506
mIoU = 0.8882841695
```

如需同时得到 full-resolution 指标，使用模型侧 full-res 评估脚本：

```bat
D:\ESPNet\.venv\Scripts\python.exe D:\ESP_INT8\tools\eval_hw_constrained_qat_fullres.py ^
  --artifact-dir D:\ESP_INT8\quantized_artifacts_hw_constrained_qat_3ep ^
  --out-json D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single\int8_baseline_metrics_fullres.json ^
  --batch-size 4 ^
  --num-workers 0 ^
  --progress-every 25
```

该脚本同时输出三套指标：

```text
lowres_argmax                  # 旧口径：64x128 logits argmax vs 64x128 target
fullres_bilinear_logits_argmax # 主 full-res 口径：logits 双线性上采样到 512x1024 后 argmax
fullres_nearest_mask           # 硬件友好口径：64x128 mask 最近邻上采样到 512x1024
```

若使用 `PERF125-UPFULL` 这类板端直接输出 `512x1024` mask 的硬件版本，板端验证集输出不再适用于旧的 logits 评估脚本，应使用：

```bat
D:\ESPNet\.venv\Scripts\python.exe D:\ESP_INT8\tools\eval_val_hw_masks_fullres.py ^
  --output-dir E:\ ^
  --output-pattern O%%04d.BIN ^
  --out-json D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val\board_val_fullres_mask_metrics.json ^
  --progress-every 25
```

### Step 4: 验证 param_blob 可被 HLS 控制域解析

如果 `blob_file_tb.exe` 已存在：

```bat
D:\ESP_INT8\ESP_INT8_hls\hls_work\blob_file_tb.exe ^
  D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single\param_blob.bin
```

成功条件：

```text
blob_file_tb passed: bytes=129856, uops=75, scales=55
```

`export_int8_hw_blob.py` 内置了同等的 tensor/uop contract 检查；如果 tensor 地址、`phys_c/channel_offset` 或关键 uop 序号不匹配，脚本会直接失败。

### Step 5: Top 级 HLS C model 输出对比

完整 top TB 会在 `D:\ESP_INT8\ESP_INT8_hls` 下写出：

```text
hls_output_q.bin
```

随后执行：

```bat
python D:\ESP_INT8\tools\compare_hls_top_output.py
```

当前单图结果：

```text
logit mismatches = 15590 / 16384
max_abs_diff    = 32
mean_abs_diff   = 6.564758
mask mismatch   = 72 / 8192
valid mismatch  = 65 / 3842
golden PA/mIoU  = 0.97188964 / 0.92504997
HLS PA/mIoU     = 0.96954711 / 0.91880367
delta           = -0.00234253 / -0.00624630
```

这里的逐 byte 不一致不直接等价于功能错误。PC fake-quant golden 与 HLS 整数实现存在舍入级差异时，优先看 `argmax mask` 和 PA/mIoU；当前 top TB 只把 mask mismatch 超过 `128/8192` 作为粗粒度失败条件。板端输出则应优先与 HLS C model 输出做 bit-level 对比。

## 4. 一键脚本

可以使用：

```bat
D:\ESP_INT8\tools\run_hw_constrained_qat_export.bat
```

默认执行：

1. 带约束 QAT
2. 模型侧 artifact 导出
3. 硬件 `param_blob/input/golden` 导出
4. INT8 software baseline 评估
5. 如果已有 `blob_file_tb.exe`，执行 blob 解析验证

默认不清理旧输出。如果需要重跑前删除旧目录，请编辑 `.bat`：

```bat
set "CLEAN_OUTPUT=1"
```

## 5. 验收标准

最终交给硬件侧的数据必须满足：

- `export_manifest.json` 中 `warnings` 数量为 `0`
- `scale_table.json` 中所有 `zero_point` 为 `0`
- `manifest.json` 中 `hardware_constraint_report.passed=true`
- `param_blob.bin` 可通过 `blob_file_tb`
- top 级 HLS C model 已完成 `compare_hls_top_output.py` 统计
- `input_q.bin` 大小为 `1572864` bytes
- `golden_output_q.bin` 大小为 `16384` bytes
- `uop_count=75`

只有满足以上条件的数据，才可作为后续 FPGA/NPU 推理比对基准。若用于板端 bit-level 比对，优先使用 HLS C model 导出的 `hls_output_q.bin` 作为整数硬件语义参考。

## 6. 全验证集板端测试数据导出

`D:\ESPNet\generate_val_dataset.py` 是 FP32 版验证集导出脚本，不能直接用于当前 INT8 NPU。INT8 全验证集应使用：

```bat
D:\ESP_INT8\tools\export_val_hw_dataset.bat ^
  --out-dir D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val ^
  --batch-size 4
```

该 `.bat` 会调用 `D:\ESPNet\.venv\Scripts\python.exe`。输出目录采用 SD 友好的短文件名：

- `PARAM.BIN`：共享参数 blob
- `I0000.BIN`：第 0 张验证图的 NHWC INT8 输入，大小 `1572864` bytes
- `R0000.BIN`：fake-quant classifier 参考 logits，NHWC INT8，大小 `16384` bytes
- `T0000.BIN`：remap 后 target mask，大小 `8192` bytes
- `M0000.BIN`：参考 logits 的 argmax mask，大小 `8192` bytes
- `VALMAN.CSV` / `manifest.json`：文件索引、原始图片路径和参考 PA/mIoU

轻量自检命令：

```bat
D:\ESP_INT8\tools\export_val_hw_dataset.bat ^
  --out-dir D:\ESP_INT8\hw_artifacts\val_export_smoke ^
  --limit 1 ^
  --check-single
```

`--check-single` 会检查 `I0000.BIN/R0000.BIN` 是否与当前单图 `input_q.bin/golden_output_q.bin` bit-exact 一致。完整 val set 约 500 张，单输入文件约 1.5 MB，总数据量约 0.8 GB；后续板端 app 可按 `VALMAN.CSV` 顺序读取 `Ixxxx.BIN` 并输出 `Oxxxx.BIN`，再由主机离线计算全验证集 PA/mIoU。

脚本默认使用 CPU 生成参考 logits，以保持和旧单图 golden dump 完全一致；如只需要快速生成近似软件参考，可显式加 `--cuda`，但不建议与 `--check-single` 混用。

板端批量跑完后，可用下面的脚本离线评估输出：

```bat
D:\ESP_INT8\tools\eval_val_hw_outputs.bat ^
  --dataset-dir D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val ^
  --output-pattern O%%04d.BIN
```

该脚本读取 `VALMAN.CSV`、`Txxxx.BIN` 和板端输出 `Oxxxx.BIN`，计算整体验证集 PA/mIoU，并可统计相对 `Rxxxx.BIN` 的 logits/mask 差异。

当前 val set 版 app 的 SD 卡根目录应放置：

- `PARAM.BIN`
- 可选 `INPUTQ.BIN`：用于上板后先跑一次单图 profiling；若不存在，app 自动使用 `I0000.BIN`
- `I0000.BIN ... I0499.BIN`

运行流程为：先执行一次单图 `MODE_RUN` 并打印 `NPU: prof SINGLE ...` 计数器，再依次执行验证集 `Ixxxx.BIN -> Oxxxx.BIN`。验证集循环在遇到首个缺失输入文件时停止，并在末尾打印 sample 数、总时间、平均/最小/最大单图延迟。
