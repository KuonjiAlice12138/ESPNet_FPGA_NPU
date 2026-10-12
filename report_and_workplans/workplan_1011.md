# Workplan 1011 - H240W240 灰度二分类 INT8 导出与上板

> **执行要求：** 后续 Agent 按本文 checkbox 顺序执行。模型侧、编译器、HLS、App 和
> SD 文件必须来自同一份 manifest；每个 gate 未通过时停止，不得通过修改 golden、放宽
> 比对或恢复旧 fallback 绕过问题。

**目标：** 将 `workplan_0922.md` 中精度最好的 `240x240` 单通道灰度 binary2 模型做
硬件约束 INT8 QAT，导出 PARAM v4 artifact，完成对应 HLS/App 平台修改并上板验证。
首版已上板，延时为 **64.855 ms**；后续目标是在 **100 MHz、相同模型与输出** 下压至
**50 ms 以内**，具体执行第 10 节，不改回已失败的跨行常驻流水架构。

**架构：** 从已签核的 H256W512 Round 4 提交 `a9c5a43` 建立独立 H240W240/C1
分支，保留 R4 的单套 32x32 SA、WinGen、Conv POST、PPU 和 RTL counter。新版本采用
编译期固定几何，不在硬件中同时保留 RGB/H256 与灰度/H240 两套路径。

**技术栈：** PyTorch QAT、Python PARAM v4 compiler/replay、Vitis HLS 2025.1、Vivado
2025.1、Vitis bare-metal App、Zynq UltraScale+ `xczu15eg-ffvb1156-2-i`。

---

## 0. 冻结输入与最终接口

### 0.1 模型基线

| 项目 | 固定值 |
|---|---|
| FP32 checkpoint | `D:/ESPNet/resolution_h240w240_gray/fp32_binary2_dice025_ft20/model_best.pth` |
| 训练结果 | epoch 8，PA `0.97027910`，mIoU `0.80554192` |
| 两类 IoU | 车辆 `0.64248172`，背景 `0.96860213` |
| 模型 | `Model_flattened.ESPNet_Encoder(p=1,q=1,input_channels=1,classes=2)` |
| 预处理 | 保持纵横比的中心正方形裁剪，随后 resize 到 `240x240`，真实灰度单通道 |
| 标签 | Cityscapes trainId `13..18 -> 0`，其他有效像素 `-> 1`，void `255` |
| 推理输出 | `2x30x30` logits，`bilinear(align_corners=False)` 到 `240x240` 后 argmax |

本轮不从头重新训练 FP32，不增加 decoder，不把灰度复制成 RGB，不改卷积拓扑。用户已确认
使用 ReLU：保留原 PReLU checkpoint，显式适配并保存到
`resolution_h240w240_gray/fp32_binary2_relu_adapt/model_best.pth`，对同一 500 张验证集重新
测试 PReLU/ReLU 精度，再以 ReLU checkpoint 做 INT8 QAT，继续使用
`weighted CE + 0.25 * vehicle soft Dice`。所有精度 gate 仍以原始最佳 FP32 为基准。

### 0.2 硬件接口 contract

| 项目 | H240W240 灰度版本 |
|---|---:|
| 输入张量 | NHWC INT8，`240x240x1` |
| 输入字节 / 256-bit words | `57,600 / 1,800` |
| 低分辨率 logits | HWC INT8，`30x30x2`，`1,800` bytes |
| 输出 mask | uint8，`240x240` |
| 输出字节 / 256-bit words | `57,600 / 1,800` |
| PARAM | v4，预期仍为 75 UOP、16 EXEC、26 CONV、7 AFFINE、14 ADD、3 POOL |
| 时钟 | 100 MHz，10 ns |
| SD 单图文件名 | `P2G.BIN`、`I2G.BIN`、`O2G.BIN` |

PARAM v4 只表示二进制 record ABI 不变。新 artifact 必须额外声明
`hardware_profile=p7_h240w240_gray_c1_v1`；HLS geometry gate 必须拒绝旧 H256W512
PARAM，因此同为 v4 也不能混用。

### 0.3 绝对约束

- [x] 从 `a9c5a43` 建立 `feature/h240w240-gray-int8`，禁止带入已证实会卡死的 Round 5A
  多行常驻流水代码。
- [x] 保持单套 32x32 SA、WinGen、weight buffer、Conv POST、PPU、AvgPool 和 memory
  owner；不得为 C1 新增第二套完整 datapath。
- [x] HLS 只执行 PARAM 已编译的 mode/schedule，不按层名或运行时 shape 推断网络结构。
- [x] 不增加 DDR activation spill、完整特征图副本、runtime task array 或 Tcl
  `ALLOCATION` 依赖。
- [x] 量化、ReLU、add、concat、pool、双线性插值和 argmax 与 replay bit-exact。
- [x] 保留 stage counter 与 Conv 内部 WIN/SA/POST counter，接口宽度不变。
- [x] 新版本完成上板前保留现有两个 H256W512 release artifact，不覆盖、不删除。

## 1. 文件边界

| 责任 | 文件 |
|---|---|
| 灰度量化模型 | `D:/ESPNet/Model_flattened_quantized.py` |
| 部署 contract | `D:/ESPNet/deployment_profiles.py` |
| 硬件约束 QAT | `D:/ESPNet/qat_cityscapes20.py` |
| 模型 artifact 导出 | `D:/ESPNet/export_quantized_artifacts.py` |
| 模型侧回归 | `D:/ESPNet/test_h240w240_gray_deployment.py` |
| 几何与通道 contract | `tools/geometry_contract.py` |
| PARAM v4 compiler | `tools/export_int8_hw_blob.py` |
| PARAM integer replay | `tools/hw_param_replay.py` |
| 验证集导出/评估 | `tools/export_val_hw_dataset.py`、`tools/eval_single_hw_outputs.py`、`tools/eval_val_hw_masks_fullres.py` |
| 工具回归 | `tools/test_h240w240_gray_contract.py`、`tools/test_p7_contracts.py` |
| HLS 结构检查 | `tools/check_p6_hls_structure.py` |
| HLS 固定配置 | `ESP_INT8_hls/include/npu_config.hpp`、`ESP_INT8_hls/include/npu_schedule.hpp` |
| HLS C1 数据流 | `ESP_INT8_hls/src/param_dma.cpp`、`win_gen.cpp`、`avgpool_unit.cpp`、`upsample_unit.cpp` |
| HLS 存储/顶层 | `ESP_INT8_hls/src/memory.cpp`、`frame_dma.cpp`、`int8_core.cpp` |
| HLS 验证 | `ESP_INT8_hls/tb/top_golden_sample_tb.cpp`、`blob_v3_tb.cpp`、`memory_tile_tb.cpp`、`frame_dma_tb.cpp` |
| App | `ESP_INT8_app/src/app_config.h`、`ESP_INT8_app/src/main.c`、`ESP_INT8_app/src/hal/int8_npu.c` |

不得新增独立 C1 SA、第二个 WinGen CPP 或第二个 AvgPool engine。C1/C17/C129 逻辑在现有
owner 内替换 H256 专用 C3/C19/C131 实现；H256 版本由原分支保存。

## 2. Round 0 - QAT 与模型 artifact

### 2.1 先锁定部署 contract

- [x] 给两份 `DeploymentGeometry` 增加 `input_channels`，H240 profile 固定为 1；读取旧
  manifest 时缺省为 3，保证旧 artifact 仍能被离线工具识别。
- [x] manifest 增加 `image_mode=gray`、`center_square_crop=true`、灰度 mean/std、cache
  SHA-256、checkpoint SHA-256 和 `input_shape_nchw=[1,1,240,240]`。
- [x] `Model_flattened_quantized.ESPNet_Encoder` 增加 `input_channels`，仅允许 1 或 3。
  `level1_conv.in_channels=1`，B1/B2 拼接与 BN/下采样输入分别为 17 和 129。
- [x] 最佳 FP32 checkpoint 必须与量化模型激活行为一致后才能严格加载。实际核查发现
  FP32 是 **PReLU（109315 参数）**，现有量化模型是 **ReLU（108513 参数）**，不能
  直接严格加载。按已批准的显式转换仅移除 8 个激活斜率；其他 state 项严格检查且不改变。
  ReLU checkpoint 保存后必须重新严格加载。C1 ReLU 模型前向形状已验证为 `[1,2,30,30]`。
- [x] 由训练目录的 `build_loaders()` 作为 QAT 和 exporter 的唯一预处理入口；禁止
  `export_quantized_artifacts.py` 继续使用根目录旧 RGB `DataSet/Transforms` 重建样本。

运行：

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' -m unittest discover `
  -s 'D:/ESPNet' -p 'test_h240w240_gray_deployment.py'
```

预期：所有 contract 测试 PASS；同一验证样本在训练、QAT、export 三条入口产生完全相同
的 C1 tensor、中心裁剪坐标和标签。

### 2.2 硬件约束 INT8 QAT

- [x] `qat_cityscapes20.py` 增加并记录 `--image-mode gray`、
  `--center-square-crop`、`--vehicle-dice-weight 0.25`。
- [x] QAT 模型按 `image_mode` 构造 C1；训练 loss 调用训练侧
  `compute_segmentation_loss()`，不得退回纯 CE。
- [x] 保持对称 INT8：activation zero-point=0、weight zero-point=0、现有 P7 shared-scale
  group、HLS rounding/ReLU 约束不变。
- [x] 先校准 32 batches，再训练 5 epoch；observer 和 BN 按现有硬件约束流程冻结，按
  full-resolution mIoU 保存最佳 prepared state。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' 'D:/ESPNet/qat_cityscapes20.py' `
  --profile binary2 --precision-policy int8 --class-weights `
  --source-checkpoint 'D:/ESPNet/resolution_h240w240_gray/fp32_binary2_relu_adapt/model_best.pth' `
  --cache-file 'D:/ESPNet/resolution_h240w240_gray/cache/binary2_gray_center_stats_v2.pkl' `
  --output-dir 'D:/ESPNet/resolution_h240w240_gray/qat_binary2_int8_hw' `
  --height 240 --width 240 --target-scale 8 `
  --image-mode gray --center-square-crop --vehicle-dice-weight 0.25 `
  --loss-resolution logits --calibration-batches 32 `
  --epochs 5 --lr 1e-5 --weight-decay 5e-4 `
  --batch-size 8 --val-batch-size 1 --workers 4 `
  --device cuda --seed 20261011
```

QAT gate：PA 相对 FP32 下降不超过 `0.5` 个百分点，mIoU 不超过 `1.5` 个百分点，车辆
IoU 不超过 `2.0` 个百分点。若首次不通过，只允许从最佳 prepared state 以 `2e-6`
续训最多 3 epoch；仍不通过则停止，不进入 HLS，也不修改 FP32/golden 掩盖退化。

### 2.3 导出模型侧 artifact

- [x] `export_quantized_artifacts.py` 使用与 QAT 相同的 loader、C1 模型和 prepared state。
- [x] 导出前核对 26 个卷积的实际 weight shape；重点为 C1 首层、C17 L2 下采样和 C129
  L3 下采样。
- [x] golden 必须同时保存 NCHW 模型语义和硬件所需 NHWC 语义；manifest 明确两者轴序。
- [x] 输出目录固定为
  `D:/ESPNet/resolution_h240w240_gray/quantized_artifacts_binary2_int8_hw`。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' 'D:/ESPNet/export_quantized_artifacts.py' `
  --profile binary2 --precision-policy int8 `
  --model-path 'D:/ESPNet/resolution_h240w240_gray/fp32_binary2_relu_adapt/model_best.pth' `
  --prepared-state 'D:/ESPNet/resolution_h240w240_gray/qat_binary2_int8_hw/model_best.pth' `
  --cache-file 'D:/ESPNet/resolution_h240w240_gray/cache/binary2_gray_center_stats_v2.pkl' `
  --output-dir 'D:/ESPNet/resolution_h240w240_gray/quantized_artifacts_binary2_int8_hw' `
  --height 240 --width 240 --target-scale 8 `
  --image-mode gray --center-square-crop --fake-quant-eval `
  --calibration-batches 32 --skip-golden-text
```

Round 0 gate：manifest、checkpoint、cache hash 齐全；输入 `[1,1,240,240]`、输出
`[1,2,30,30]`；classifier zero-point=0；26 个卷积全部为 W8A8；500 张验证集指标通过
QAT gate。

## 3. Round 1 - PARAM compiler、replay 与 artifact

### 3.1 通道化编译 contract

- [x] `tools/geometry_contract.py` 将 `input_channels` 纳入 input bytes、NHWC shape、manifest
  解析和相等性检查；H240 不能依赖默认 C3。
- [x] `ArtifactDeploymentContract` 读取 C1 与预处理字段；CLI、模型 manifest、PARAM tensor
  desc 三者任一不一致时导出失败。
- [x] `build_tensor_plan()` 与 affine/pool plan 使用 C1/C17/C129：`T_INPUT/T_POOL1/
  T_POOL2/T_POOL_TMP=1`，`T_B1_*=17`，`T_B2_*=129`。
- [x] PARAM record 布局保持 v4，但不得复用旧 mode 数值。新增
  `STAGED_C1=12`、`STAGED_C17=13`、`STAGED_C129=14`；旧 C3/C19/C131 数值只用于识别
  历史 artifact，不得在新 HLS hierarchy 中保留对应 datapath。
- [x] 第一层 C1 source row 为 240 bytes，不满足旧 C3 KEEP1 的 32-byte row 对齐条件。
  首版明确编译 `ROW_REUSE_NONE`，禁止错误复用旧 48-word contract；后续性能优化另立轮次。

### 3.2 固定片上 layout

编译器和 `npu_config.hpp` 必须使用同一组常量：

| 区域 | 十进制 bytes | 十六进制/说明 |
|---|---:|---|
| B2 primary payload | 464,400 | `60*60*129 = 0x71610` |
| BLOCK5 base / L2 bytes | 230,400 | `0x38400` |
| residual base | 491,520 | `0x78000` |
| residual high-water | 245,760 | `120*align32(120*17)`；逻辑 payload 为 244,800 |
| C1 scratch base | 753,664 | `0xB8000` |
| L2 scratch slot | 57,600 | `0xE100` |
| L3 scratch slot | 23,040 | `30*align32(30*25) = 0x5A00` |
| URAM prefix | 819,200 | `0xC8000` |
| POOL1 base / bytes | `819,200 / 14,400` | `0xC8000 / 0x3840` |
| POOL_TMP base | 833,600 | `0xCB840` |
| POOL2 logical / reserved bytes | `3,600 / 3,616` | `0xE10 / 0xE20`，尾字按 32-byte 对齐 |
| FMBUF total / BRAM tail | `848,000 / 28,800` | `0xCF080 / 0x7080` |

- [x] exporter 生成 lifetime/overlap audit，并逐项证明 BLOCK5、residual、C1 scratch、POOL1、
  POOL2 alias 和 POOL_TMP 不重叠。
- [x] HLS 以相同表达式和 static_assert 固化这些值；禁止手工维护另一套不一致地址。
- [x] `MAX_K_TILE_COUNT` 仍为 40，因为最大 3x3 输入 C129 需要 37 K tiles；不得为新几何
  扩大 schedule record。

**240 列的必要对齐补充：** 不仅 Pool/mask，C17、C129、C12、C25、C28 compact 行也存在
非 32-byte 对齐。沿用 16-byte tensor record，在 `reserved0` 的 bit15 显式编码
`ROW_ALIGN32`，低 15 bit 仍为 physical C；memory owner 按该 flag 计算 row stride。
Frame/Pool 的 C1 仍紧凑连续；中间 compact 行独立 zero-pad，禁止跨行覆盖或热路径 RMW。
B2 allocation 为 `60*7744=464640`，C17 行为 2048 bytes；L3 C25 scratch 行为 768 bytes。
上述补充不改变 PARAM record 大小、卷积算术或主存容量，compiler/replay/HLS 必须共同验收。

### 3.3 Replay 与硬件 artifact

- [x] `hw_param_replay.py` 支持 C1/C17/C129 mode 和 C1 AvgPool；移除 `src.c==3` 限制。
- [x] `export_val_hw_dataset.py` 使用 manifest 的 C1、gray、center crop，不再写死
  `(3,H,W)`；`eval_val_hw_masks_fullres.py` 使用相同 preprocessing contract。
- [x] compiler 在 artifact 中生成 PARAM replay lowres golden，以及按 HLS 定点双线性规则
  生成的 `golden_fullres_mask.bin`；不能直接把 PyTorch fake-quant logits 当硬件 golden。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' 'tools/export_int8_hw_blob.py' `
  --artifact-dir 'D:/ESPNet/resolution_h240w240_gray/quantized_artifacts_binary2_int8_hw' `
  --out-dir 'D:/ESP_INT8/hw_artifacts/binary2_int8_h240w240_gray_v4' `
  --param-version 4 --height 240 --width 240 --target-scale 8 `
  --strict-zp --strict-schedule

& 'D:/ESPNet/.venv/Scripts/python.exe' 'tools/hw_param_replay.py' `
  --param 'hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN' `
  --input 'hw_artifacts/binary2_int8_h240w240_gray_v4/INPUTQ.BIN' `
  --output 'hw_artifacts/binary2_int8_h240w240_gray_v4/replay_check.bin'

& 'D:/ESPNet/.venv/Scripts/python.exe' -m unittest `
  tools.test_h240w240_gray_contract tools.test_eval_single_hw_outputs

& 'D:/ESPNet/.venv/Scripts/python.exe' 'tools/test_p7_contracts.py'
```

Round 1 gate：`parse_and_check_blob_v4()` PASS；PARAM init contract 为 H240/W240/C1；
replay output 与发布 lowres golden 逐字节一致；fullres mask 为 57,600 bytes；audit 中无
hot RMW、无 overlap、无 warning。

## 4. Round 2 - HLS H240/C1 数据流

### 4.1 配置、接口和 geometry gate

- [x] `npu_config.hpp` 固定 `MAX_FM_H/W=240/240`、`INPUT_FRAME_C=1`、
  `ENCODER_OUT_H/W=30/30`、input/output AXI words=1,800，并更新第 3.2 节 layout。
- [x] `int8_core.cpp` 的 m_axi depth、static_assert 和 CSim dump shape 同步修改；
  `frame_dma_load()` 只搬运 1,800 words。
- [x] `param_dma.cpp::validate_deployment_geometry()` 接受且只接受 H240/W240/C1、
  logits H30/W30、class count 2；旧 H256 PARAM 必须返回错误。

### 4.2 WinGen 静态专用路径

- [x] 在现有 `win_gen.cpp` owner 内把 active C3/C19/C131 staged builder 改为
  C1/C17/C129；保留 C12/C25/C28/C64/C128 等仍由网络实际使用的 mode。
- [x] C1 首层每个 3x3 window 只有 9 个有效 activation，其余 TK lanes 显式补零；
  last-K mask 由 schedule 入口锁存，热循环不做 shape 推断。
- [x] C17 为 5 K tiles，C129 为 37 K tiles；pack command、cache chunk 和静态 word
  composer 必须由 compiler contract 驱动并与 replay 对齐。
- [x] 不启用旧 C3 row reuse，不新增 runtime generic fallback，不复制 WinGen 或 SA。

### 4.3 C1 AvgPool 与非 32-byte 行宽

H240 的两级池化为 `240x240x1 -> 120x120x1 -> 60x60x1`。输出行分别为 120 和
60 bytes，均不是 32-byte 对齐；旧 C3 `group32*3` writer 不能复用。

- [x] 在现有 `avgpool_unit.cpp` 中以一个 C1 engine 替换 C3 engine，保持 3x3、stride 2、
  padding 1 和 signed round-div9 行为。
- [x] 使用顺序 256-bit pack accumulator 跨行延续：每得到一个输出像素追加 1 byte，满
  32 bytes 写一次；层结束时只写一个 zero-padded 尾字。禁止逐像素 RMW。
- [x] POOL1 共 14,400 bytes，恰好 450 words；POOL2 逻辑数据 3,600 bytes，为 112 words
  加 16-byte 尾字。`BRAM_SCR1_BYTES/FMBUF_POOL2_ALIAS_BYTES` 必须向上对齐为 3,616
  bytes，使 zero-padded 尾字合法且不覆盖下一个 live tensor。

### 4.4 30x30 到 240x240 输出

当前 `upsample_unit.cpp` 按 `FULLRES_MASK_W/32` 每行写字，会丢失每行最后 16 像素。

- [x] 保持现有 bilinear axis map、tie-breaking 和 argmax；只重构输出 pack。
- [x] 使用跨行顺序 mask accumulator，按全帧线性顺序追加标签并每 32 bytes 写一次；
  240x240 总计恰好 1,800 words，无尾字、无 RMW。
- [x] `upsample_fused_begin()` 同时复位 previous-logits-row 和 output pack 状态；最后一行
  必须断言 fill=0、word_index=1,800。

### 4.5 HLS 验证与清理

- [x] 更新 `blob_v3_tb.cpp`、`memory_tile_tb.cpp`、`frame_dma_tb.cpp` 和
  `top_golden_sample_tb.cpp` 到新 artifact；测试旧 H256 PARAM 被 geometry gate 拒绝。
- [x] 更新 `tools/check_p6_hls_structure.py` 与 `tools/test_p7_contracts.py` 的当前 release
  contract，检查新 mode ID、H240/C1 常量和旧 staged builder 零引用。
- [x] top TB 预填充输出为 `0xAA`，比较全部 57,600 bytes，并检查最后一个 AXI word，
  防止 240 列尾部静默漏写。
- [x] 扫描 C3/C19/C131 active mode、H256 常量、旧 AvgPool helper、旧每行输出 writer 和
  未引用 static helper；仅保留必要的历史枚举注释，不允许旧逻辑进入 hierarchy。
- [x] 运行结构检查和一次 fullres CSim；不跑 cosim，不使用旧 work directory。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' 'tools/check_p6_hls_structure.py'

& 'D:/Xilinx/2025.1/Vitis/bin/vitis-run.bat' --mode hls `
  --config hls_config.cfg --work_dir hls_work_h240w240_gray --csim
```

Round 2 gate：PARAM INIT 成功；75 UOP/16 EXEC 完成；ERROR=0；lowres logits 与 replay
0 mismatch；fullres mask 与 `golden_fullres_mask.bin` 0 mismatch；无越界写、死锁、旧逻辑
双份常驻或 owner clone。实际 RTL clone、II 和资源占用仍须通过 Round 3 报告确认；CSim 不能替代
该检查，也不能证明有限深度 FIFO 在 RTL 中不会阻塞。

## 5. Round 3 - csynth、实现与平台

- [x] 用户手动使用正式 `hls_config.cfg` 和全新 work directory 跑 top csynth；禁止引用
  R5A 生成 RTL 或旧 package cache。
- [x] 检查 singleton hierarchy：WinGen、SA、Conv POST、PPU、AvgPool、Upsample 各 1 份；
  SA K-loop II=1、POST II=1。本轮综合 11 分 18 秒，未失控；未取得 R4 同阶段报告，
  不声称已经验证严格的两倍同比耗时门槛。
- [x] 检查接口：frame input/output depth 均为 1,800 words，PARAM depth 仍为 8,192 words，
  stage/Conv profiling ports 保持原宽度。
- [x] 资源 gate：DSP 不高于 R4 的 1,391，URAM 不高于 56，BRAM tile 不高于 246，逻辑
  LUT 不高于 R4 的 242,069；H240 layout 应显著降低存储资源，若反而增长则先查旧数组或
  duplicated path。
- [x] 重新 package IP，在 Vivado 替换 NPU 后验证 address map、AXI master、stage counter
  和时钟复位连接；implementation 必须 routing error=0、WNS>=0、WHS>=0。
- [ ] 导出全新 XSA，记录 HLS source commit、PARAM/artifact hash、XSA hash、bitstream hash
  和实现资源，不以旧 XSA 时间戳或 GUI cache 判断版本。

未达到时序/路由 gate 时只修 H240 代码；不得降低 SA/TM/TK，不得恢复 C3/H256 双路径，
不得通过 Vivado 忽略 DRC 强行出 bitstream。

## 6. Round 4 - App、SD 与上板

### 6.1 单图 App

- [x] `app_config.h` 改为 binary2 单图模式：H/W/C=`240/240/1`，输入/输出均 57,600
  bytes，classes=2，PARAM v4，75 UOP、16 EXEC。
- [x] build tag 固定为 `INT8-BOARD-20261011-P7-H240W240-GRAY-B2`，关闭 dual model、
  exec-prefix 和 valset；保留 timer self-test、stage counter、Conv counter 和 timeout。
- [x] SD 根目录文件只放当前 `BOOT.BIN`、`P2G.BIN`、`I2G.BIN`；先删除同名旧输出
  `O2G.BIN`。App 打印的三个 DDR buffer 地址必须 64-byte 对齐且互不重叠。
- [x] 上板前核对 XSA/bitstream、App ELF、PARAM、INPUTQ 的 SHA-256 与发布记录。

单图 board gate：timer self-test PASS；PARAM INIT 显示 v4/75/16；MODE_RUN 完成且
ERROR=0；输出 57,600 bytes；`O2G.BIN` 与 CSim/replay fullres mask 逐字节一致；类别只含
0/1 且直方图与 golden 一致。

首次 bring-up 的门槛为 100 MHz 下目标 `<=6.0M PL cycles`（60 ms），允许上限
`<=7.0M cycles`（70 ms）。本次 6.485468M cycles 通过首版上限、未达到原 60 ms 目标；
后续正式性能 gate 改为 `<5.0M cycles`（<50 ms），见第 10 节，不能把 bring-up 通过
写成性能目标已经完成。

### 6.2 500 张验证集

- [ ] 单图通过后，使用同一 QAT prepared state 和 PARAM 导出 500 张验证集输入；目录
  固定为 `I/I0000.BIN..I/I0499.BIN`，输出为 `O/O0000.BIN..O/O0499.BIN`，避免 FAT 根目录
  文件项耗尽。
- [ ] 每个输入必须为 57,600 bytes，manifest 记录原图名、中心裁剪、input hash 和
  reference hash；第 0 张必须与单图 `I2G.BIN` 完全相同。
- [ ] App 只在首张打印 stage/Conv counters，统计 500 张 min/avg/max MODE_RUN；SD 读写
  不计入 MODE_RUN。
- [ ] 主机按 240x240 中心裁剪后的 full-resolution target 统计 PA、mIoU、车辆 IoU 和
  背景 IoU。

最终精度 gate：板级 500 张结果与 PARAM replay mask 完全一致；板级指标与已通过的 INT8
QAT 指标仅允许由既定 HLS 定点语义造成的已记录差异，不允许相对 replay 再下降。

## 7. 停止规则与最终交付

| 失败点 | 动作 |
|---|---|
| QAT 未过精度 gate | 停止，不修改 HLS |
| model artifact 与 replay 不一致 | 修 exporter/量化 contract，不改 golden |
| PARAM audit 有 overlap/hot RMW/warning | 停止，不进入 CSim |
| CSim logits 或 mask mismatch | 定位首个发散 tensor，不进入 csynth |
| csynth clone、II 回退或资源增长 | 清除旧路径/修 call graph，不靠 directive 掩盖 |
| implementation 负 slack或非法 route | 不导出平台 |
| board ERROR、卡死或 mask mismatch | 回退到 H240 上一 gate；R4 H256 分支保持不动 |

最终保留：

- [x] `hw_artifacts/binary2_int8_h240w240_gray_v4`，包含 PARAM、单图输入、模型 golden、
  replay golden、fullres mask、audit、manifest 和全部 hash。
- [ ] H240 HLS/App source commit 与 tag，完整 csynth/implementation 报告摘要。
- [ ] 单图和 500 张板测日志、stage/Conv cycle 分解、PA/mIoU/两类 IoU。
- [ ] 更新 `tools/README.md` 和架构文档，明确 H240 C1 分支与 H256 R4 artifact 不兼容。

**执行顺序固定为：Round 0 QAT -> Round 1 compiler/replay -> Round 2 HLS/CSim ->
Round 3 csynth/implementation -> Round 4 单图上板 -> 500 张验证。任何步骤不得跳过前一
gate，也不得在同一次提交中混入额外性能架构升级。**

## 8. 1012 验收结果与下一步

**Round 0/1/2 功能、Round 3 综合/实现及 Round 4 单图上板均已通过；
当前延时 64.855 ms，50 ms 性能目标与板级 500 张验证尚未完成。**
当前分支为 `feature/h240w240-gray-int8`，从 R4 `a9c5a43` 分出；原 PReLU
checkpoint、R4 原提交及两组 H256W512 release artifact 均保留。App 已选择
`platform_h240_1012` 并切为 H240/C1 单图模式；已重新 build，未复用旧 ELF/BOOT。

### 精度 gate

原 FP32 的 8 个逐通道 PReLU 包含 802 个斜率参数。经用户批准仅移除这些 key，
其他 state 严格加载；原 checkpoint 不覆盖。直接 ReLU 精度未过车辆 IoU gate，但
5 epoch 硬件约束 INT8 QAT 的最佳 epoch 5 checkpoint 严格重载后通过全部 gate，
无需额外续训。下表为同一 **500 张验证集、240x240 中心裁剪输出**：

| 模型 | PA | mIoU | 车辆 IoU | 背景 IoU |
|---|---:|---:|---:|---:|
| 原 PReLU FP32 复测 | 97.0277% | 80.5534% | 64.2467% | 96.8600% |
| 直接 ReLU FP32 | 96.7534% | 79.0790% | 61.5827% | 96.5752% |
| ReLU 硬件约束 INT8 QAT | 96.9113% | 79.9633% | 63.1882% | 96.7384% |

QAT 相对第 0.1 节原最佳 FP32 的 PA/mIoU/车辆 IoU 分别下降
**0.117/0.591/1.060 个百分点**，均在 0.5/1.5/2.0 的允许范围内。

### Artifact 与 HLS 功能 gate

- 新产物 `binary2_int8_h240w240_gray_v4`：PARAM v4、143040 bytes、
  75 UOP/16 EXEC/26 CONV；严格 audit 为 0 warning、无 hot RMW、无 lifetime overlap。
- 模型 INT8 logits、独立 PARAM replay 和发布 golden 三者 **1800 bytes 全一致**。
  验证集 loader smoke 的首图输入/参考 logits 与单图逐字节一致，目标映射也一致。
- 中间 compact tensor/scratch 采用显式 row-padding flag，POOL2 reserve=3616 bytes；
  BLOCK5 first4 L2/L3 allocation 为 190080/95040 bytes，当前第五分支仍只留在 row buffer。
- HLS 已替换 C1/C17/C129 静态窗口、C1 AvgPool、跨行 mask pack，并移除旧 C3 reuse、
  C3/C19/C131 builder、握手 probe、过时中间 dump hooks 和未使用 QBUF 常量。
  保留 INIT 的旧 ABI 拒绝逻辑、现有 profiling 接口和被 synthesis guard 排除的 lowres dump。
- 最终 fullres CSim 使用 `hls_work_h240w240_gray`，2026-10-12 完成，约 2 分钟，
  UOP=74/ERROR=0；logits **0/1800 mismatch**，mask **0/57600 mismatch**，
  invalid=0，末尾 AXI word 全覆盖。4 种非法/退役 schedule 和旧 H256 PARAM 均被 INIT 拒绝，
  随后新 PARAM 可重新 INIT 成功。
- 当前样本 CSim 与 replay 的 fullres PA/mIoU 为 **97.8722%/87.2564%**，
  车辆/背景 IoU 为 **76.8019%/97.7109%**；仅为首张单图结果，不替代上述 500 张指标。
  明细位于 artifact 的 `csim_single_metrics.json`。
- 24 项灰度/评估测试、12 组 P7 contract 以及 source structure check 通过。
  14 个 CPP 静态引用扫描无未引用 local static helper，single call-site/owner 结构保持；
  实际 hierarchy clone count 已由下述本轮 csynth/RTL 审查确认。
- CSim 最大 stream depth=1110 是无界 CSim 的观测值，不是 RTL FIFO 深度要求或
  RTL 无死锁证明；不据此扩大 FIFO 或恢复 R5A 多行常驻结构。

用户已手动完成综合、package、Vivado 实现、平台/App build 与单图板测；正式 cfg 仍为
10ns、原顶层与 14 个 CPP。后续进入第 10 节性能优化，本次记录不修改 HLS 或启动综合。

### 1012 顶层综合审查

本轮使用新生成的 `ESP_INT8_hls/ESP_INT8_hls/hls`，11:36 开始、11:47 完成，
总耗时 **11 分 18 秒**，peak memory=3.489 GB。报告与当前 H240 源码对应，综合期间
源码没有后续修改；无 HLS error，全部 loop constraints 满足，未出现搜索时间失控。

| HLS 估计项 | 本轮结果 |
|---|---:|
| LUT | 359869（详细 top report；summary 为 359871） |
| FF | 198951 |
| DSP | 1277，其中 SA 为 1024 |
| BRAM18K | 660，约 330 个 BRAM36 容量当量，非实际 placed tile 数 |
| URAM | 28 |
| 目标 / 估计周期 | 10ns / 7.300ns，uncertainty=2.70ns |

RTL 实例链确认 Conv、scheduled WinGen、SA、Conv POST、PPU row consumer、BLOCK5
finalizer、C1 AvgPool 和 Upsample 各一份；无旧 C3/C19/C131 active builder、旧 reuse 或
R5A resident engine。SA K-loop、psum emit、POST 主循环均为 II=1。生成接口确认三路
256-bit AXI master、input/output=57600 bytes、PARAM capacity=262144 bytes；原 profiling
端口齐全且宽度不变。report audit blocker=0，source/report structure check PASS。

**结论：具备进入 Vivado 综合的条件，但不能宣称全部物理资源 gate 已通过。** HLS LUT
估计为器件容量的 105.45%；第 5 节的 R4 LUT/BRAM 门槛来自实现报告，不能直接与 HLS
估计比较，也不能预先假定 Vivado 一定优化回来。本次未取得可比的 R4 完整 csynth 报告，
不宣称已证明资源同比下降或耗时小于其两倍；URAM 的 28 对 56 是明确的容量减少。

剩余风险为 task builder 控制 fanout=8495、SA enable fanout=8192、PPU consumer
fanout=7083，以及带 profiling 输出的 DATAFLOW throughput warning。日志还有一条
PPU compact writer 调度时 7.6115ns 的 clock warning，最终该循环 II=1、报告 slack=0.19ns；
这些不是综合失败，但须用物理报告确认。先检查 Vivado 映射后的逻辑 LUT/总 LUT、BRAM
tile 和控制压力；资源可接受后再以 100MHz 跑实现，并严格要求合法 route、WNS/WHS>=0。
不根据通用 shape 上界的 HLS max latency 预测整网延时，也不以估计 Fmax 保证实现时序。

### 1012 实现验收与单图上板交付

新平台为 `platform_h240_1012`。本次实现 14:57 开始，16:06 完成 bitstream；全部
421785 个 routable nets 合法布通，routing error=0，DRC error=0。最终 100MHz 下
WNS=**+0.252ns**、WHS=**+0.010ns**、TNS/THS=0；post-route phys-opt 未修改 netlist。
资源取 placed utilization，不能与 HLS 估计混用：

| 项目 | H240 本轮 | H256 R4 门槛 |
|---|---:|---:|
| 逻辑 LUT / 总 LUT | 219686 / 236675 | 242069 / 259800 |
| FF | 192017 | 206388（参考） |
| BRAM tile | 203.5 | 246 |
| DSP / URAM | 1298 / 28 | 1391 / 56 |

最差路径是 PARAM AXI 读缓存到 INIT 参数寄存器，route delay=8.589ns；现有物理 gate
已经通过，本轮不额外改动 datapath。XSA、平台内部/导出副本逐字节 hash 一致；内嵌 bit
与实现输出及平台 SDT bit 相同；Vivado 使用的 299 个 Verilog 文件与当前 packaged IP
全部 hash 一致，无旧 IP 混入。NPU/counter 地址仍为 `0xA0000000/0xA0010000`，
stage-valid、active、pc、issue-kind 与三组 Conv state 连线完整、同钟同复位。

App 已关闭 dual/prefix/valset，保留 timer self-test、Conv counter 和 timeout。单图复用
`load_param_and_init()`，补回全网 END 检查及 v4/75/16 打印；不再重复旧单图 INIT 逻辑。
修正了遗留 R5A debug launch 的 bit/FSBL 路径及缓存 bit，3 个 C 文件通过新 BSP 下 ARM
编译器 syntax check，10 项 App contract 通过；未代替用户 build 或执行板测。

发布 hash 已记录到 App README：XSA=`67eb53a8...ac9ec56`，bit=`b3f1ddbf...65201f3`，
PARAM=`bb92bafa...4244fa0`，input=`c1f3b032...2b5dd87`。用户于 17:05 手动 build 新 App；
ELF=`2f440a0d...90ae56b`，随后用 `boot_h240.bif` 打包 BOOT=`f1d82983...a08a6f2`。
`E:\` 为 FAT32，根目录三个文件均完成逐文件 hash 核对，未预置 `O2G.BIN`；旧 H256
根目录测试文件已移除，B/C/D/Q 验证集目录保留。ELF 中 input/output/param 地址为
`0x28440/0x36540/0x44640`，尺寸为 `57600/57600/262144`，64-byte 对齐且互不重叠。
源文件仍未提交；上述版本已完成单图功能验收，板测结果如下。500 张板级验证仍待完成。

## 9. 1012 首版上板结果与归因

### 9.1 功能与计时

App 为 `INT8-BOARD-20261011-P7-H240W240-GRAY-B2`，平台为 `platform_h240_1012`。
PARAM INIT 正确显示 v4/75 UOP/16 EXEC，MODE_RUN 正常结束，ERROR=0，Conv capability=1。
FSBL 仍打印 PMU-FW 未运行提示；本次没有因此中断，不能把该提示作为性能瓶颈证据。

实读 `E:/O2G.BIN`：**57600 bytes，0/57600 mismatch，invalid=0，c0=4105、c1=53495**。
SHA-256 为 `88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce`，
与发布 golden、replay/CSim mask 完全一致，不只是直方图相同。因此该图 fullres PA/mIoU
与第 8 节一致，为 **97.8722%/87.2564%**；车辆/背景 IoU 为 **76.8019%/97.7109%**。
这不是 500 张板级精度结论，也不代表板上已单独验证低分辨率 logits。

ARM 计数 **2161813 ticks / 33333000 Hz = 64.855039 ms**；RTL 计数
**6485468 cycles / 100 MHz = 64.854680 ms**。两者相差约 **0.36 us**，约 5.5 ppm；
终端显示 65 ms 是整数取整。timer self-test 为 PASS，但真正的交叉证据是 ARM 与独立
PL counter 一致，不能仅用 `usleep()` 自检证明时钟频率。计时范围仍为完整 MODE_RUN，
包含硬件输入/输出搬运，不包含 SD、INIT、cache maintenance 和 UART counter 打印。

### 9.2 Stage 与 Conv 内部状态

| Stage | cycles | ms @100 MHz | 总时间占比 |
|---|---:|---:|---:|
| IDLE / MAIN_CTRL | 258 / 1001 | 0.00258 / 0.01001 | 0.02% 合计 |
| FRAME_LOAD | 14025 | 0.14025 | 0.22% |
| CONV_WEIGHT_LOAD | 34416 | 0.34416 | 0.53% |
| CONV_ROW_DATAPATH | 4129114 | 41.29114 | 63.67% |
| PPU_ROW_CONSUME | 528634 | 5.28634 | 8.15% |
| PPU_BLOCK5_FINAL | 893550 | 8.93550 | 13.78% |
| VEC_FIXED | 631133 | 6.31133 | 9.73% |
| AVGPOOL | 253337 | 2.53337 | 3.91% |
| PARAM_INIT / UPSAMPLE_OUT / FRAME_STORE / ERROR | 均为 0 | 0 | 0 |
| **TOTAL** | **6485468** | **64.85468** | **100%** |

UPSAMPLE_OUT/FRAME_STORE 为 0 不表示没有上采样/输出 DMA；本版 U72 的融合输出由
`ppu_consume_upsample_row()` 执行，计入 PPU_ROW_CONSUME。该 Stage 也包含其他 Conv
的 store/cat-affine，不能把全部 5.286 ms 都归因于上采样。

| 并行 owner | IDLE_OR_DONE | 状态 1 | 状态 2 | 合计 |
|---|---:|---:|---:|---:|
| WIN | 1288410 | ACTIVE=2840704 | 无 | 4129114 |
| SA | 1281090 | COMPUTE_OR_WAIT_ACT=269993 | PSUM_EMIT_OR_WAIT=2578031 | 4129114 |
| POST | 1433393 | STAGING_WORD_OR_WAIT_PSUM=546341 | COMMIT_WORD_OR_WAIT_PSUM=2149380 | 4129114 |

三套 owner 在同一 Conv 窗口内并行，**不能相加**。SA emit 包括阻塞/等待，POST staging/
commit 是 psum word 分类，包含等待、量化、pack/write；不能据此宣称写 BRAM 独占 52.05%，
或宣称 SA 只有 6.53% 时间在真实计算。约 1.28M 的共同 idle 区间也不能直接当作可全部消除
的气泡：需用逐 EXEC 数据及 generated RTL/Schedule Viewer 核对入口、退出和调用开销。

### 9.3 当前工作点与优先级

本版 26 个 Conv 的标准 MAC 工作量为 **142517700**；按已编译的 paired/K tile 计算，
静态算术填充率为 **68.124%**，activation K-tile issue 共 **204300**。32x32 INT8 阵列在
100 MHz 下峰值为 **102.4 GMAC/s**；当前 Conv/整网有效吞吐为 **3.452/2.197 GMAC/s**，
对应峰值利用率 **3.37%/2.15%**。静态填充尚可，但时间利用率低，优先改善供数/交接和
短循环服务开销，而不是扩大阵列。未测量可持续片上带宽，不能声称已定量求出 roofline ridge
或严格证明某个单独 RAM port 是瓶颈；当前证据支持片上数据服务/控制受限，而非 DDR 受限。

源码/10 月 12 日综合报告的直接证据：

- `win_gen.cpp:791/828/910`：窄通道 loader 串行更新行缓存，再向 assembler 发每像素
  9 个 256-bit spatial token；paired 为 18 个。C1 每个位置只用 1 byte，却仍走此协议；
  所有当前 schedule 的 row_reuse_mode 均为 0。C12/d1 已有 incremental producer，不能回退它。
- `conv_engine.cpp:217` 的 POST 主循环已被 HLS flatten，实际 II=1、depth=4；SA K-loop
  II=1/depth=3，emit II=1/depth=2。不能靠继续增加 POST lanes 或重复加 II=1 获得确定收益。
- `ppu.cpp:504/881/936`：compact writer 按 4-byte group 顺序拼字；BLOCK5 composer 后
  每个输出 tile 依次 read residual、add/affine、write。`vec_alu_engine.cpp` 的对齐 tile
  路径也串行 read/transform/write；现有 qparam 已驻留，重复加参数缓存不是新优化。
- `avgpool_unit.cpp:133`：C1 group cache 已存在，但像素算术 loop 为 PIPELINE off。
  Pool 总量仅 2.533 ms，即使消除全部 Pool 也不能单独达到 50 ms；DDR/frame 与 weight load
  合计也不足 0.5 ms。主线仍应是 Conv，其次是 PPU/Vec。

## 10. 后续性能计划：100 MHz 下 <50 ms

### 10.1 冻结约束与周期预算

从本次基线至少节省 **1485468 cycles（14.855 ms、22.905%）**，才到 5.0M 边界；
正式目标必须严格小于它，建议按 **4.83M / 48.3 ms** 预留余量。下表是工作预算，不是
已经证明的收益；每轮用板级 cycle gate 更新，不能用 HLS max latency 代替。

| 区域 | 当前 M cycles | 目标预算 M cycles |
|---|---:|---:|
| Conv | 4.129114 | 3.10 |
| PPU row（含上采样） | 0.528634 | 0.40 |
| BLOCK5 final | 0.893550 | 0.62 |
| Vec | 0.631133 | 0.46 |
| Pool | 0.253337 | 0.20 |
| 其他 | 0.049700 | 0.05 |
| **合计** | **6.485468** | **4.83** |

继承第 0.3 节约束：固定模型、240x240/C1、W8A8、100 MHz、PARAM v4 record ABI，
不重新 QAT、不加 decoder、不恢复 U71 融合、不扩/减 SA。仍为单套 Conv/SA/POST/PPU，
不新增 CPP、完整 FM 副本、DDR spill、hot RMW 或 ALLOCATION/Tcl 兜底。
**不再尝试 R5A 跨行常驻 DATAFLOW、双 owner 共享同一行 buffer、task array/反馈流。**
后续改变的是原模块内部传输/局部循环，不移动 MainCtrl/Conv/PPU 的模块边界。

### 10.2 Round 5：一次逐 EXEC 归因，冻结修改对象

**范围：** `tools/export_exec_prefix_params.py`、`tools/analyze_conv_cycle_profile.py`、
`ESP_INT8_app/src/app_config.h`、`tools/test_app_h240w240_gray_contract.py`，必要时修
`main.c` 中 profiling 元数据；本轮不改 HLS。

- [ ] 从当前 143040-byte PARAM 导出 P00..P15，保持 geometry/weight/qparam/descriptors
  不变，只修改提前 END；P15 必须与完整 PARAM hash 一致。App 保持 H240/C1，切 prefix
  单次启动跑全套，input=I2G.BIN，最终 output=O2G.BIN，reference=6485468，target 暂为 0。
  App contract 同步允许本轮明确的 prefix 模式，其他 dual/valset 模式仍关闭；最终单图
  验收前恢复原配置与对应测试，不能为通过测试静默关闭 prefix 实验。
- [ ] 取得每个 EXEC 的 stage/WIN/SA/POST delta。P15 与本次 full MODE_RUN 允许测量差
  `<=0.1%`，输出仍为 0 mismatch；三组 Conv owner sum 各自等于 Conv delta。重点区分
  U2/C1、U7/C17、U40/C129、四个 BLOCK5、U39/U71 与 U72，不根据 MAC 数排序代替实测。
- [ ] 用结果确定 Round 6 的热点顺序，记录 C1/narrow/wide/1x1 各组可节省的服务开销；
  若 C1 占比不高，不把 C1 优化假定为足够节省 1.03M Conv cycles。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/export_exec_prefix_params.py `
  --param hw_artifacts/binary2_int8_h240w240_gray_v4/PARAM.BIN `
  --out-dir hw_artifacts/binary2_int8_h240w240_gray_v4/exec_prefix
```

### 10.3 Round 6：WinGen 减少无效搬运，保持行级 Conv DATAFLOW

**范围：** `win_gen.cpp`、`npu_schedule.hpp`、`param_dma.cpp`；配套修改
`export_int8_hw_blob.py`、`hw_param_replay.py`、`test_h240w240_gray_contract.py`。
改动顺序固定为 C1 packed-row，再由 Round 5 的热点决定窄通道增量传输；不能一次扩至
所有 mode 或重新写 generic WinGen。原 `scheduled_window_generator_row()` 仍是唯一主入口。

- [ ] C1 用编译器明确编码的 packed-row/stride2 KEEP1 协议替换当前 9 spatial-token
  路径。用三份小行缓存，每份 **8 个 256-bit words**；读入有效源行后直接组成每像素
  9 个有效 TK lanes，输出到原 act_stream0/1，其余 lanes 为 0。每个新输出行保留一行，
  只更新两行；整层最多 240x8=1920 个 aligned source-word 读取，不含 act FIFO token。
  这只是 WinGen 内部复用，不是 Conv/PPU 跨行 resident pipeline。
- [ ] 严格处理 240-byte 行起点的 0/16-byte 相位：每行只取覆盖真实行的 8 个源字，局部
  拼成标准行；末尾 16 bytes 清零，不改变全帧 NHWC、输入 DMA 或 FM row stride。
  最后一行不得预读第 1800 个输入字；边界行/列补零，out_row=0 无条件清缓存有效位，
  换 source/非连续 row 也失效。重复 MODE_RUN、prefix 后重新 INIT 都不能带入旧行。
- [ ] 在 PARAM v4 现有 reserved schedule 字段内显式编码协议与 packed row count=8；
  不复用旧 C3 对齐假设，不按 cfg.shape 临时推断协议。INIT/compiler/replay 共用合法性
  约束，拒绝不支持的 flags/count；候选 artifact 单独命名并记录新 hardware_profile，
  保留本次 baseline。只重新编译 schedule，不改变 QAT、weights、qparams 或数学 golden。
- [ ] 对确认为热点的 C12/C25 mode，改为只传编译 update columns 的增量 token，assembler
  单独拥有所需 slot cache，用现有固定段 composer 生成 4/8 个 K words；不再每像素重传
  9 个 spatial words。保留已有效的 C12/d1 incremental 路径；d2/4/8/16、paired phase、
  slot wrap 的 FIFO 生产/消费数必须由 exporter 审计。禁止宽 2304-bit window token、
  dynamic lane scatter、完整 runtime segment interpreter 或 loader/assembler 双写缓存。
- [ ] 全面移除已替换 mode 的旧入口/分支，不让新旧 producer 同时进入 hierarchy。
  运行轻量窗口边界/重入测试、structure check 和一次 fullres CSim：logits 0/1800、mask
  0/57600 mismatch。然后交用户手动 csynth/package/实现，必要时同套 prefix 复测。

**Gate：** 原 act/psum FIFO 和单套 SA/POST 不变，K-loop/emit/POST II=1，clone count=1。
Conv 预算为 <=3.10M；若实测未达，不假装结束本轮，也不盲目增大 cache/强制大 loop II=1。
先用 delta 定位仍占主导的 wide/1x1 或共同调用开销，再做同一 owner 内的局部修补。

### 10.4 Round 7：PPU/Vec 局部字级流水，Pool 只做小修

**范围：** `ppu.cpp`、`vec_alu_engine.cpp`、`conv_store.cpp`、`avgpool_unit.cpp`；保持
MainCtrl、Conv 调用图和 memory layout 不变，默认无需修改 PARAM 数学语义。

- [ ] compact writer 从 4-byte group 改为 8-byte pair 加 4/1-byte tail；使用固定切片、
  小 carry、顺序字拼接，满 256-bit 再写，行 padding 保持 zero。覆盖 C12/C16/C17/C25/C28，
  不加动态 256-bit lane scatter 或逐像素 RMW；通过后删除旧整组实现。
- [ ] BLOCK5 与 Vec 的对齐 tile 路径缩短 read/transform/write 的调用/循环边界：先明确
  实际 RAM 端口能力，再将可流水的局部固定数量 tile 服务收敛为单个有界 word loop。
  source/residual 读取、一次 add/affine、一次提交按数据依赖推进；是否能 II=1 必须由
  schedule/端口证据判断，禁止对共享 URAM 的整行调度强制 II=1。保持一套底层 add/affine/
  store kernel，不能按 L2/L3、affine/add-affine 分别克隆。现有参数驻留保持，不重复加缓存。
- [ ] Vec compact 的逆行遍历及读完一行再提交顺序不得打乱；BLOCK5 scratch/residual
  先读后写语义不变，不能通过重排 tile 悄悄覆盖后续 source。局部寄存器只缓存在途 word，
  不引入 current Conv output 中间 FM，不把 BLOCK5 finalizer 恢复成 Vec standalone。
- [ ] AvgPool 仅将 group cache 中的纯算术/pack loop 局部流水；word write 若限制 II，
  拆出该调用而不是展开 32 个像素或强行流水整个内存 loop。保持 signed round_div9、
  两级 pool padding、跨行 pack 和 POOL2 尾字；该小修不能占用主线资源/综合预算。
- [ ] 修改全部完成后做 static/deadlogic/singleton 扫描及一次 fullres CSim，确认同一
  golden 全一致，再由用户手动综合/实现。PPU row/BLOCK5/Vec/Pool 分别以
  <=0.40/0.62/0.46/0.20M 为预算；任一路出现回退先定位，不用 Conv 的收益掩盖回退。

### 10.5 Round 8：最终验收与交付

- [ ] 每次 csynth 对同级 baseline（LUT359869、FF198951、DSP1277、BRAM18K660、URAM28）
  比较，重点查 SA=1024 DSP 和 singleton、新增 port conflict/II violation、高扇出及
  clone。超过 baseline 5% 必须解释来源后再决定实现；不能与 placed 数值交叉比较。
  综合超过首版 11m18s 的两倍时检查日志/CPU/报告进度，不按 warning 单独判死锁。
- [ ] 100 MHz 物理验收必须 route error/overlap=0、WNS/WHS>=0；资源以上板 baseline 的
  logic LUT219686、BRAM tile203.5 为参照。追求不回退，新增量必须证明必要；不能预设
  HLS 超资源会自动被 Vivado 拉回，也不能用升频代替 cycle 优化。
- [ ] 回到 binary2 单图 full MODE_RUN，保留全部 counter，至少连续测 3 次；**每次
  RTL cycles <5000000，ARM 未取整延时也 <50 ms**，两种时钟换算一致，ERROR=0，
  输出 0 mismatch，才宣称
  <50 ms。保持 SD/INIT/cache/UART 不计入，不删 counter 获得不可比的数字。
- [ ] 最后完成第 6.2 节的 500 张板级精度/延时验证，首张与本次输入相同；确认无旧行
  缓存污染、无跨图状态错误，输出保持 replay 一致。未过 gate 保留已上板基线，并把失败
  原因记录在本 workplan；普通实施细节不再逐条追加日志。

**下一步先执行 Round 5 归因，再动 Round 6；本次仅更新计划，未修改硬件。**
