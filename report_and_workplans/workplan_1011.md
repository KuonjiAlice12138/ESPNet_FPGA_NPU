# Workplan 1011 - H240W240 灰度二分类 INT8 导出与上板

> **执行要求：** 后续 Agent 按本文 checkbox 顺序执行。模型侧、编译器、HLS、App 和
> SD 文件必须来自同一份 manifest；每个 gate 未通过时停止，不得通过修改 golden、放宽
> 比对或恢复旧 fallback 绕过问题。

**目标：** 将 `workplan_0922.md` 中精度最好的 `240x240` 单通道灰度 binary2 模型做
硬件约束 INT8 QAT，导出 PARAM v4 artifact，完成对应 HLS/App 平台修改并上板验证。
首版已上板，延时为 **64.855 ms**；Round 6 为 **58.452 ms**，Round 7 已降至
**53.34364 ms（较 R6 -8.74%，较首版 -17.75%）**，单图 mask 仍与 golden 完全一致。
后续目标是在 **100 MHz、相同模型与输出** 下压至 **50 ms 以内**，先执行第 10.4.3 节的
Vec/BLOCK5 定点修补，不改回已失败的跨行常驻流水架构。

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

PARAM v4 只表示二进制 record ABI 不变。首版 artifact 声明
`hardware_profile=p7_h240w240_gray_c1_v1`；Round 6 已上板版本为
`p7_h240w240_gray_c1_r6_delta`。HLS geometry/transfer gate 必须拒绝旧 H256W512
及已替换的窄通道传输协议，因此同为 v4 也不能混用。

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

Round 7 已上板，当前为 **5.334364M cycles / 53.34364 ms**，相对 R6 节省 510787 cycles
（8.74%），相对首版降低 17.75%。严格小于 50 ms 至少还需节省 **334365 cycles**，约为
当前整网的 6.27%。PPU row/Pool 原预算已经通过，BLOCK5/Vec 未通过；Vec 还比 R6 增加
36401 cycles，不能因整网变快而忽略这一回退，也不能沿用旧局部报告推断该路径已提速。

后续只做 **Round 7R：Vec 回退修复 + BLOCK5 局部服务收敛**，见 10.4.3。以下为工作预算，
合计 **4.943639M / 49.43639 ms**，留约 0.56 ms 余量；不是已经证明可达的预测。
预算须先通过真实 EXEC 数量加权的局部服务分析，不再盲目追求旧 0.45M/0.36M 极限目标。
Conv/PPU row/Pool 保留 R7 收益；**原 <=3.10M Conv gate 仍未通过**。

| 区域 | 当前 M cycles | 目标预算 M cycles |
|---|---:|---:|
| Conv（冻结 R7） | 3.418908 | 3.418908 |
| PPU row（含上采样，冻结） | 0.289114 | 0.289114 |
| BLOCK5 final | 0.773550 | 0.60 |
| Vec | 0.667534 | 0.45 |
| Pool（冻结） | 0.135617 | 0.135617 |
| 其他 | 0.049641 | 0.05 |
| **合计** | **5.334364** | **4.943639** |

继承第 0.3 节约束：固定模型、240x240/C1、W8A8、100 MHz、PARAM v4 record ABI，
不重新 QAT、不加 decoder、不恢复 U71 融合、不扩/减 SA。仍为单套 Conv/SA/POST/PPU，
不新增 CPP、完整 FM 副本、DDR spill、hot RMW 或 ALLOCATION/Tcl 兜底。
**不再尝试 R5A 跨行常驻 DATAFLOW、双 owner 共享同一行 buffer、task array/反馈流。**
后续改变的是原模块内部传输/局部循环，不移动 MainCtrl/Conv/PPU 的模块边界。

### 10.2 Round 5：一次逐 EXEC 归因，冻结修改对象

**执行决定：** 按用户要求跳过本轮实验，不重新设为后续修复的必跑门槛。当前已有 R6/R7
整网 Stage 数据，但没有逐 EXEC 实测，不能把全部收益分配给某一层。只有局部服务分析仍
不能确定修补对象时才补做；使用 R7 已验证的 R6 artifact，不导出旧 transfer 协议的 PARAM。

**范围：** `tools/export_exec_prefix_params.py`、`tools/analyze_conv_cycle_profile.py`、
`ESP_INT8_app/src/app_config.h`、`tools/test_app_h240w240_gray_contract.py`，必要时修
`main.c` 中 profiling 元数据；本轮不改 HLS。

- [ ] 从当前 143040-byte PARAM 导出 P00..P15，保持 geometry/weight/qparam/descriptors
  不变，只修改提前 END；P15 必须与完整 PARAM hash 一致。App 保持 H240/C1，切 prefix
  单次启动跑全套，input=I2G.BIN，最终 output=O2G.BIN，reference=5334364，target 暂为 0。
  App contract 同步允许本轮明确的 prefix 模式，其他 dual/valset 模式仍关闭；最终单图
  验收前恢复原配置与对应测试，不能为通过测试静默关闭 prefix 实验。
- [ ] 取得每个 EXEC 的 stage/WIN/SA/POST delta。P15 与本次 full MODE_RUN 允许测量差
  `<=0.1%`，输出仍为 0 mismatch；三组 Conv owner sum 各自等于 Conv delta。重点区分
  U2/C1、U7/C17、U40/C129、四个 BLOCK5、U39/U71 与 U72，不根据 MAC 数排序代替实测。
- [ ] 若需补修 Conv，用结果定位 R6 剩余 C1/narrow/wide/1x1 与调用开销；不把已经取得
  的 640246 cycles 收益全部归给 C1，也不把约 1.28M idle 当作可免费消除的空间。

```powershell
& 'D:/ESPNet/.venv/Scripts/python.exe' tools/export_exec_prefix_params.py `
  --param hw_artifacts/binary2_int8_h240w240_gray_r6_v4/PARAM.BIN `
  --out-dir hw_artifacts/binary2_int8_h240w240_gray_r6_v4/exec_prefix
```

### 10.3 Round 6：WinGen 减少无效搬运，保持行级 Conv DATAFLOW

**范围：** `win_gen.cpp`、`npu_schedule.hpp`、`param_dma.cpp`；配套修改
`export_int8_hw_blob.py`、`hw_param_replay.py`、`test_h240w240_gray_contract.py`。
改动顺序固定为 C1 packed-row，再做 C12/C25 窄通道增量传输；C17 同步归入同一个 assembler
缓存 owner，以移除 loader 的旧 full-window 缓存，不新增独立 datapath。wide/1x1 保持不变，
不重写 generic WinGen。原 `scheduled_window_generator_row()` 仍是唯一主入口。

- [x] C1 用编译器明确编码的 packed-row/stride2 KEEP1 协议替换当前 9 spatial-token
  路径。用三份小行缓存，每份 **8 个 256-bit words**；读入有效源行后直接组成每像素
  9 个有效 TK lanes，输出到原 act_stream0/1，其余 lanes 为 0。每个新输出行保留一行，
  只更新两行；整层最多 240x8=1920 个 aligned source-word 读取，不含 act FIFO token。
  这只是 WinGen 内部复用，不是 Conv/PPU 跨行 resident pipeline。
- [x] 严格处理 240-byte 行起点的 0/16-byte 相位：每行只取覆盖真实行的 8 个源字，局部
  拼成标准行；末尾 16 bytes 清零，不改变全帧 NHWC、输入 DMA 或 FM row stride。
  最后一行不得预读第 1800 个输入字；边界行/列补零，out_row=0 无条件清缓存有效位，
  换 source/非连续 row 也失效。重复 MODE_RUN、prefix 后重新 INIT 都不能带入旧行。
- [x] 在 PARAM v4 现有 reserved schedule 字段内显式编码协议与 packed row count=8；
  不复用旧 C3 对齐假设，不按 cfg.shape 临时推断协议。INIT/compiler/replay 共用合法性
  约束，拒绝不支持的 flags/count；候选 artifact 单独命名并记录新 hardware_profile，
  保留本次 baseline。只重新编译 schedule，不改变 QAT、weights、qparams 或数学 golden。
- [x] C12/C17/C25 mode 改为只传编译 update columns 的增量 token，assembler
  单独拥有所需 slot cache，用现有固定段 composer 生成 4/8 个 K words；不再每像素重传
  9 个 spatial words。保留已有效的 C12/d1 incremental 路径；d2/4/8/16、paired phase、
  slot wrap 的 FIFO 生产/消费数必须由 exporter 审计。禁止宽 2304-bit window token、
  dynamic lane scatter、完整 runtime segment interpreter 或 loader/assembler 双写缓存。
- [x] 全面移除已替换 mode 的旧入口/分支，不让新旧 producer 同时进入 hierarchy。
  运行轻量窗口边界/重入测试、structure check 和一次 fullres CSim：logits 0/1800、mask
  0/57600 mismatch，last_error=0；C1 连续两帧分别为 1920 次源字读取，非连续行重入与
  narrow paired/odd-tail/slot-wrap 测试通过。结构和 Python contract 检查通过。
- [x] 用户已手动完成 csynth、package、实现、平台/App build 与单图上板；clone/II、
  物理时序及单图功能通过。原 <=3.10M Conv 性能 gate 尚未通过，见下一节实测。

已上板目录为 `hw_artifacts/binary2_int8_h240w240_gray_r6_v4`，只改变 schedule word 9
的 12 个字节，权重、qparams、输入、logits/mask golden 与首版逐字节一致。reserved[9]
的 bits1:0 为 reuse、bits8:2 为 packed words、bits10:9 为 transfer；INIT 拒绝旧 narrow
协议。首版 artifact 保留用于历史对比；当前 App/SD 已切到 R6，不与旧 IP/PARAM 混用。
主 `hls_config.cfg` 不变。

**2026-10-13 首次综合失败：** 55.505 秒即停在 source analysis，未进入 scheduling；
`HLS 207-3325` 报出 `param_dma.cpp` 四处 `ap_uint<8> == enum` 歧义，不是搜索爆炸。
已统一改成 `.to_uint() == static_cast<unsigned>(enum)`，不改变校验值、协议或数据流。
新增回归测试先复现四处问题再通过；全部 14 个生产 CPP 已通过真实 HLS synthesis headers
下的 frontend 检查，35 项 H240/App 测试通过。后续用户手动综合成功；本修复无需重新导出
artifact。普通代码实施细节不再追加记录。

**Gate：** 原 act/psum FIFO 和单套 SA/POST 不变，K-loop/emit/POST II=1，clone count=1。
原 Conv 预算为 <=3.10M；实测未达，不宣称该性能 gate 通过，也不盲目增大 cache/强制
大 loop II=1。现在优先按修订 Round 7 压非 Conv 开销；若需进一步改 Conv，先用 delta
定位 wide/1x1 或共同调用开销，再做同一 owner 内的局部修补。

#### 10.3.1 1014 Round 6 上板验收与归因

App=`INT8-BOARD-20261014-P7-H240W240-GRAY-R6-B2`，平台=`platform_1014`。
PARAM v4/75 UOP/16 EXEC 完整，Conv capability=1，MODE_RUN 正常结束，ERROR=0。
ARM 为 **1948375/33333000=58.4518345 ms**，PL 为 **5845151/100MHz=58.451510 ms**，
相差 0.325 us（5.55 ppm），计时可信；终端的 58 ms 是整数显示，不是精确延时。
输入/输出/参数地址为 `0x2F440/0x3D540/0x4B640`，均 64-byte 对齐且不重叠。

实读 SD 的 P2G/I2G 与 R6 artifact hash 一致；`E:/O2G.BIN` 为 57600 bytes，
**0/57600 mismatch、invalid=0、c0=4105、c1=53495**。mask SHA256 仍为
`88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce`，与首版及 R6 golden
完全一致。该单图 PA/mIoU 仍为 97.8722%/87.2564%；不代表已完成 500 张板级验证，也不
代表本次已单独读取板级 logits。

| Stage | R6 cycles | ms @100MHz | 占比 | 相对首版 cycles |
|---|---:|---:|---:|---:|
| Conv row datapath | 3488868 | 34.88868 | 59.69% | -640246 |
| PPU row（含上采样） | 528634 | 5.28634 | 9.04% | 0 |
| BLOCK5 final | 893550 | 8.93550 | 15.29% | 0 |
| Vec | 631133 | 6.31133 | 10.80% | 0 |
| Pool | 253337 | 2.53337 | 4.33% | 0 |
| 其他 | 49629 | 0.49629 | 0.85% | -71 |
| **TOTAL** | **5845151** | **58.45151** | **100%** | **-640317（-9.87%）** |

其他包括 IDLE=262、FRAME_LOAD=14028、MAIN_CTRL=923、WEIGHT_LOAD=34416；PARAM_INIT、
UPSAMPLE_OUT、FRAME_STORE、ERROR 均为 0。Conv 减少 **15.51%**，其余四个计算 Stage
完全未变，说明收益来自本轮 Conv 服务优化；不能把改善归给未修改的 PPU/Vec/Pool。

| 并行 owner | 首版状态 cycles | R6 状态 cycles | 差值 |
|---|---:|---:|---:|
| WIN active | 2840704 | 2194248 | -646456 |
| SA compute-or-wait | 269993 | 260640 | -9353 |
| SA emit-or-wait | 2578031 | 1948368 | -629663 |
| POST staging-or-wait | 546341 | 572277 | +25936 |
| POST commit-or-wait | 2149380 | 1493781 | -655599 |

R6 WIN/SA/POST idle 分别为 1294620/1279860/1422810，各 owner 状态总和都为 3488868。
上述减少发生在同一并行窗口，不能相加；POST staging 增加也不能单独认定为量化回退。
约 1.28M idle 仍存在，但未取得逐 EXEC 或启动/等待拆分，不能承诺全部消除。当前有效 Conv
吞吐约 4.085 GMAC/s，仅占 102.4 GMAC/s 理论峰值约 3.99%；仍以片上服务/控制开销为
优先方向，不据此声称已到 compute-bound 或已经测得 roofline ridge。

综合用时 10m7s，LUT/FF/DSP/BRAM18K/URAM=361303/194376/1275/654/28，主循环 II=1、
singleton 通过。实现 413601 个 routable nets 全部合法，100MHz 下 WNS=+0.876ns、
WHS=+0.009ns、TNS/THS=0。placed logic/total LUT=213085/230632、FF=187186、
BRAM tile=200.5、DSP=1296、URAM=28，较首版下降；但 CLB 使用率仍为 96.78%，不能因
LUT 仅用约 68% 就假设可以任意复制 datapath。298 个 packaged/integrated RTL 文件 hash
一致，平台 XSA/内嵌 bit 与实现输出一致。

**结论：R6 架构、单图功能和物理实现通过，有确定性能收益；原 Conv 3.10M 与整网 <50ms
均未达到。保留 R6 为新的安全回退点，进入调整后的 Round 7，不回退 WinGen，也不恢复 R5A。**

### 10.4 Round 7：收敛 BLOCK5/Vec 服务开销，补上融合上采样

**范围：** `ppu.cpp`、`vec_alu_engine.cpp`、`conv_store.cpp`、`upsample_unit.cpp`、
`avgpool_unit.cpp`。MainCtrl/Conv 调用图、WinGen/SA/POST、memory layout 和 PARAM v4
数学语义保持不变，沿用 R6 artifact/golden，不重新 QAT 或导出权重。执行顺序为 A/B/C，
只在最后 D 做一次 fullres CSim；每项都必须删除被替换的旧路径。

**A. 优先压 BLOCK5/Vec，不重复做已存在的 8-byte Vec pack。** R6 两个 Stage 合计
1.524683M（26.08%）；源码与本轮局部报告显示 BLOCK5 emit_word 有独立 residual read、
add/affine、store 调用，服务为 3..27 cycles；Vec inner tile 为 19..31 cycles。这些是
具体局部服务证据，不把 generic 整网 max latency 换算成推理时间。

- [x] `ppu_finalize_block5_emit_tiles/emit_word` 收敛为单个共享 word-service 入口，L2/L3
  静态 composer 不变，只选当前 tile/qparam；每 word 先取 residual，按原顺序完成
  **INT8 add 饱和，再 affine/ReLU**，最后通过唯一 conv_store tail 提交。当前
  `ppu_apply_block_add_affine_word()` 已是逐 lane 融合，不能再把“融合 add+affine”当作
  新改进；应缩短现有 4-lane group 的服务和 emit 调用边界，默认保持 unroll factor=4，
  不全展开 32 lanes。BLOCK5 前置 branch-chain add 和 scratch compose 也计入 Stage 7，
  必须保留前置 add 的 INT8 饱和及先读后写语义，不能为了省一遍处理直接删除。
  无 residual 路径不增加空 add pass；保持一套 affine 算术，不按 pattern/mode 克隆。
- [x] `run_fixed_affine_common()` 已有 resident qparam 和 8-byte pair pack，保留它们；
  对 aligned 与 compact 的内部服务明确地址递推和实际有效 tile 上界，缩短多重 loop 的
  进入/退出。保持同一个 affine call-site。compact 仍逆行处理，读完一行后再提交原
  `s_shared_row_contig_words`，禁止边读边覆盖；aligned 的 cblock-major 顺序也不得随意换。
  `memory.cpp` 的 RAM owner 不迁移；若确需新增小型地址 gateway，必须替换旧热点调用，
  不能同时保留两个底层 datapath 或增加不明 underlying-object 的 context pointer。
- [x] 先为实际 BRAM/URAM 数组列出每 word 的 read/write 占用；两个逻辑 bank 可能仍
  共享物理 RAM，不据逻辑名字假定无冲突。有证据才流水有界算术/word loop，接受端口要求
  的 II=2/3，不强制整行 II=1、不展开 2/4 个 tile 同时访存，不引入新跨行 DATAFLOW。
  用局部真实 II、服务周期和单套 kernel 证明改进，而不是仅补 PIPELINE pragma；源码
  端口约束已落实，实际 II/服务周期及物理 clone 数仍由本轮 csynth 核验。

端口核对：FMEM0/1/2 共享 `memory.cpp` 的同一 URAM/BRAM 地址空间，SCR0 映射 pool
临时区，SCR1 映射 pool1 的 BRAM alias；逻辑 bank 名不同不代表独立端口。保持串行访存：

| 服务 | 物理共享存储访问上界/占用 | 局部流水边界 |
|---|---|---|
| BLOCK5 final word | residual 存在时读 1 字，最终写 1 字；无 residual 不读 | 仅 4-lane 算术的 8 个 group；系数为 4-bank 小 LUTRAM |
| BLOCK5 四路 scratch | L2/L3 分别消费 52/103 bytes 每像素；四个滚动 cursor 顺序补字，每 cursor 每次最多读 1 字，另有行首预取 | 不并发发出四路读，不对整个 pixel/tile 强制 II=1 |
| Vec source tile | 最多读 2 字处理未对齐 span；aligned 再写 1 字；compact 先完整缓存一行，再按字提交 | 16-lane 算术和已有 pair pack；共享 RAM 的大循环不流水 |
| C1 Pool group | 三行 cache 最多读 9 字，每 32 个有效输出最多提交 1 字 | 纯缓存算术与 pack，提交调用隔离在 loop 外 |
| 上采样 | corner 只读原有前/当前 logits 行；每 32 个输出 label 写 1 个 AXI 字 | 8 像素插值/pack，AXI 写在小 loop 外 |

**B. PPU compact pack + 融合上采样，目标 PPU row <=0.35M。** Stage 6 同时包括普通
compact store/cat-affine 与 U72 上采样，不能把 528634 cycles 全归给其中任一项。

- [x] `ppu_consume_conv_stream()` 的 4-byte group 改为 8-byte pair 加 4/1-byte tail，
  可借鉴 Vec 已有 pair append；固定切片、小 carry、顺序拼字，允许 pair 跨 32-byte
  word 边界。覆盖 C12/C16/C17/C25/C28 和行尾 zero padding；保留错误时 drain stream，
  不引入动态 lane scatter、hot RMW、第二套旧 packer 或共享 buffer 双写。
- [x] `upsample_unit.cpp:interpolate_argmax_label/emit_fullres_rows` 当前逐输出像素重复
  读取四个低分辨率 corner。对固定 8x 插值用少量 corner 寄存器，在 x0/x1 改变时才更新；
  复用一组 corner 计算连续位置，不新增完整输出行或 logits 副本，不移出 PPU owner。
  保持现有首/末 4 行处理、align_corners=False 坐标、跨行 256-bit 输出 pack 和 1800 字提交。
  二分类可在证明精确等价后用 `(logit1-logit0)` 的插值符号做 argmax，差值/乘加仍用足够
  signed 位宽且 0 必须选 class0；禁止先 argmax 再插值或省略中间精度语义。局部 loop
  流水须证明 corner 端口与 AXI write 不冲突，不能恢复 Conv/PPU resident feedback。

**C. Pool 只做有限算术/pack 小修，目标 <=0.20M。**

- [x] 保留 C1 group cache，仅局部流水纯算术/pack，word write 若限制 II 则隔离调用；
  不展开 32 像素、不给整个共享 memory loop 强制 II=1。signed round_div9、requant、
  padding、跨行 pack 和 POOL2 尾字必须与现有 golden 一致。若资源/搜索代价不划算，
  停止该小修，明确记录仍缺的 cycles，不把 2.533ms 的整个 Pool 当作可免费消除。

**D. 一次统一验证与性能门槛。**

- [x] A/B/C 完成后静态扫描 dead/legacy 和 singleton，检查 source single call-site、
  buffer owner、读写依赖及 FIFO 数量；仅跑一次 H240 fullres CSim，logits 0/1800、mask
  0/57600 mismatch、last_error=0。用户手动 csynth/实现；不能自动运行或改主 cfg。
- [x] 1019 已完成单图板测：PPU row/Pool/Conv 不回退门槛通过；BLOCK5/Vec 原预算未通过，
  整网 <50ms 未通过。保留已证明有效的小修，按 10.4.2/10.4.3 修订预算和后续任务，
  不以降低模型、移除 counter、升频或复活 U71/R5A 绕过性能 gate。

#### 10.4.1 1015 Round 7 综合检查：资源通过，compact pack 时序未通过

检查本轮 `ESP_INT8_hls/ESP_INT8_hls/hls/syn/report`，报告时间为 2026-10-15
15:46；源码修改早于本轮综合启动。综合完成，用时 **10m17s**，峰值内存 3.149GB，
未出现搜索时间失控。H240 fullres CSim 已通过 logits 0/1800、mask 0/57600，ERROR=0。

| 同级 HLS 资源 | R6 | Round 7 | 变化 |
|---|---:|---:|---:|
| LUT | 361303 | 353273 | -8030（-2.22%） |
| FF | 194376 | 186761 | -7615（-3.92%） |
| DSP | 1275 | 1266 | -9（-0.71%） |
| BRAM18K | 654 | 661 | +7（+1.07%） |
| URAM | 28 | 28 | 0 |

资源增长门槛通过，但 HLS LUT 仍为器件容量的约 103.51%，不能据此保证物理实现。
RTL 实例和 hierarchy 确认 Conv row engine、WinGen、SA、Conv POST、BLOCK5 final
word service、Vec affine owner 各一套；SA 仍为 **1024 DSP**，profiling 输出完整。
BLOCK5 `emit_word` 局部服务从 3..27 收敛到 **4..18 cycles**，4-lane 算术 8 个 group
达到 II=1。Vec 新混合 tile/commit loop 报告 **11..29 cycles/iteration**，旧 tile loop
为 19..31；新下限包含 commit 分支，不能当作每个 affine tile 的周期。这些仅是局部
调度证据，不等于对应 Stage 按相同比例下降，更不能直接宣称已达到 <50ms。

**阻塞项：** `ppu_consume_conv_stream()` 的 8-byte pair loop（当前 `ppu.cpp:540`）
产生 **14.281ns** 估计周期，顶层估计 Fmax 降至 70.02MHz；报告 timing slack=-6.98ns。
日志 `hls_compile.log:3124` 的 HLS 200-887/200-871/200-1016 明确定位到
`remaining/pair_bytes -> carry/group_count -> 256-bit word mux -> ppu_write_compact_word`
及 loop 回授，写 helper 在该组合路径上占约 5.498ns。虽报告 Final II=1、全部 loop
constraints 满足，**100MHz 时序门槛仍未通过**。不能用总资源下降或 R6 实现曾收敛来
忽略这一新增长路径；本版暂不推进实现。

次要风险：HLS 214-366 将 `ppu_apply_add_word` 生成为两套（preadd 与 compact consumer
两个调用路径），每套约 LUT5124/DSP8。主要 engine singleton 检查通过不代表所有 helper
均只有一份；现有 checker 未将该 helper 纳入 clone gate。另有 task builder/SA/PPU 控制
fanout 约 6.5k..8.5k，须继续在实现中核验，不因主循环 II=1 宣称 routing 风险消失。

**下一轮最小修复：**

1. 保留 8-byte pack 和现有数学行为，将完整字提交移出 pair loop。当前 compact layout
   每像素最多 28 bytes，加上已有不足 32 bytes 的残留，每像素最多形成 1 个完整字；用
   一个 pending word/valid 承接，pair loop 只更新小 carry/group 状态，结束后唯一 write
   tail 提交。不新增整行缓存、DATAFLOW feedback 或第二套旧 packer。
2. 缩短 `pair_bytes/group_count` 的控制回授；访存结果不得参与 pack loop 的同拍状态链。
   若纯算术仍无法收敛，再做局部寄存/有界 II 调整并核算实际服务周期，不盲目强制 II=1，
   也不把整个 pixel/memory loop 流水化。保留现有 BLOCK5/Vec/Pool/上采样收益。
3. 检查上述 ADD 两个调用接口的 specialization 来源；只有能保持调用图简单和性能时才
   合并，不借修 pack 时序恢复跨行大调度或用 ALLOCATION/Tcl 掩盖复制。复核 C12/16/17/
   25/28、跨字 carry、尾字 padding、错误 drain 与完整 golden，再由用户手动综合。
   验收：该 pack 的 timing violation 消失，按 10ns/2.7ns uncertainty 尽量回到约 7.3ns
   的局部预算；资源不显著回退、单套 SA/engine 保持，局部服务与 FIFO 不引入新瓶颈。

#### 10.4.2 1019 Round 7 上板结果：53.34364 ms，仍未达到 50 ms

App=`INT8-BOARD-20261019-P7-H240W240-GRAY-R7-B2`，平台=`platform_h240_1019`。
PARAM v4/75 UOP/16 EXEC，Conv capability=1；MODE_RUN 正常结束，ERROR=0。
ARM 为 **1778112/33333000=53.3438934 ms**，PL 为 **5334364/100MHz=53.3436400 ms**，
相差 0.253 us（4.75 ppm）。终端 53 ms 为整数显示，不能据此认为已接近到仅差 3 ms。

实读 `E:/P2G.BIN`、`I2G.BIN` 与 R6 artifact 逐字节一致；`E:/O2G.BIN` 为 57600 bytes，
**0/57600 mismatch、invalid=0、c0=4105、c1=53495**，SHA256 为
`88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce`。
该图 PA/mIoU 仍为 97.8722%/87.2564%；不代表已完成 500 张板测或已读取板级 logits。

| Stage | R6 cycles | R7 cycles | R7 ms | R7 占比 | R7-R6 cycles |
|---|---:|---:|---:|---:|---:|
| Conv row datapath | 3488868 | 3418908 | 34.18908 | 64.09% | -69960 |
| PPU row（含上采样） | 528634 | 289114 | 2.89114 | 5.42% | -239520 |
| BLOCK5 final | 893550 | 773550 | 7.73550 | 14.50% | -120000 |
| Vec | 631133 | 667534 | 6.67534 | 12.51% | **+36401** |
| Pool | 253337 | 135617 | 1.35617 | 2.54% | -117720 |
| 其他 | 49629 | 49641 | 0.49641 | 0.93% | +12 |
| **TOTAL** | **5845151** | **5334364** | **53.34364** | **100%** | **-510787** |

其他为 IDLE=258、FRAME_LOAD=14044、MAIN_CTRL=923、WEIGHT_LOAD=34416；其余 Stage=0。
上采样和输出写入融合在 PPU row，Stage 10/11 为 0 不是遗漏这些工作。

**Conv 不能按状态名误归因。** WIN active=2194248，SA compute-or-wait=260640、
emit-or-wait=1948368，POST staging-or-wait=572277、commit-or-wait=1493781，全部与 R6
相同；WIN/SA/POST idle 分别为 1224660/1209900/1352850，均比 R6 减少 69960。
三个 owner 的状态总和各自等于 3418908，互相重叠，不能相加。此次 Conv 改善出现在共同
idle/调用间隔，而不是更多 MAC 吞吐；尚不能细分到哪个 EXEC/控制边界。POST 两类仍含
等待/量化/pack，不是独立的“纯写 RAM 时间”。没有带宽测量，不能宣称已达到 compute-bound。

**本轮有效，但并未全面收尾：** PPU row 降低 45.31%、Pool 降低 46.47%，均达到原预算；
BLOCK5 降低 13.43%，但未达到 0.45M；Vec 增加 5.77%，未达到 0.36M。HLS 新混合 loop
11..29 cycles 含 commit 分支，不能与 R6 的纯 tile 19..31 cycles 直接比较后宣称提速。
旧目标未完成，不整轮回退；R7 成为新的单图功能/物理实现安全基线。

最新修复版 csynth 用时 9m38s，LUT/FF/DSP/BRAM18K/URAM=
352400/186464/1266/661/28；100MHz 实现 WNS=+0.476ns、WHS=+0.007ns、TNS/THS=0，
405935 个 routable nets 全部合法。placed logic/total LUT=212172/229889、FF=180278、
BRAM tile=204、DSP=1291、URAM=28，CLB=41068/42660（96.27%）。资源不是任意增加并行
datapath 的空间；后续应优先缩短串行服务和控制，不通过增加整行副本/核复制换性能。
证据见 `D:/ESP_INT8_vp/ESP_INT8_vp.runs/impl_1/` 下的
`ESP_INT8_wrapper_timing_summary_postroute_physopted.rpt`、`ESP_INT8_wrapper_route_status.rpt`
及 `ESP_INT8_wrapper_utilization_placed.rpt`。

#### 10.4.3 Round 7R：先消除 Vec 回退，再压 BLOCK5 服务

沿用 10.1 的绝对约束。冻结已有效的 R6 WinGen、SA/POST、PPU compact pack、上采样和
Pool；不重做 QAT/export，不改 PARAM v4、MainCtrl 边界或 counter。只改原模块内部，
**不新建 resident DATAFLOW、反馈 FIFO、任务数组或 mode-specific affine datapath。**

1. **Vec 先定位服务回退，再改控制。** 用当前 PARAM 的 U39/U71 descriptor，逐项列出
   aligned/compact 的有效 tile 数、跨字 source read 数、pack 数、commit 字数、系数加载
   次数，结合现有局部 schedule/RTL 状态计算加权服务开销；不能再用混合 loop 的 min/max
   代替两个 EXEC 的实际成本。`run_fixed_affine_common()` 当前把 tile 和 commit 合到一个
   service loop，每 tile 还携带多源选择、column/cblock 和 row 地址更新；这是需排查的
   控制开销，不是已证明的唯一根因。优先将 aligned cblock 内恒定的 source bank/base/
   count 选择、stride/边界校验提前到外层，收窄计数器，减少每次服务的重复决策，保留
   一个 affine call-site 和共同 store tail。compact 仍逆行、完整读完一行再写，不删除
   alias 保护。R6/R7 的 source reader 底层都依赖 `read_tile_packed_word()`，不能仅凭
   gateway 改名断言多了一次物理读取；若改 aligned gateway，须证明端口/实际服务收益，
   并复用唯一 RAM owner，不能恢复整套旧 Vec 路径。第一门槛是 Vec <=631133，后续预算
   <=450000；仅恢复 R6 的 36401 cycles 还不足 50ms，严格达标仍缺 297964 cycles。
2. **BLOCK5 按实际工作量压开销，不删必要 pass。** Stage 7 包括 branch-chain preadd、
   scratch cursor/compose、qparam 装载、final residual/add/affine/store，先把四个 block
   的调用/word/group 数分别加权，找出 773550 cycles 中哪个服务真正占主导。当前
   `ppu_finalize_block5_static_row()` 每行重新校验布局、初始化 cursor、加载系数，再对
   每像素顺序读四个 scratch source、组 2/4 个 tile、调用唯一 `emit_word`。优先去掉热
   路径重复 descriptor/地址运算和小循环启动；能上提的布局校验与系数加载可在现有 PPU
   owner 中使用一份小型上下文复用，必须有明确的 INIT/descriptor/行连续性失效条件，
   不增加完整 FM/行副本。保留静态 L2/L3 composer、4-lane/8-group 算术和共用写尾；
   scratch/残差/写回仍按真实共享 RAM 端口串行，禁止无依据地把四路读同时展开。
   branch preadd 的 INT8 饱和及最终 residual-add→affine/ReLU 顺序不变，不将 preadd
   静默吸收到 final affine，也不将 finalizer 迁回 Vec。目标 <=600000，须由局部加权
   服务证明可行，不能用 `emit_word` max latency 的下降比例推算整个 Stage。
3. **一次统一验证，保留退路。** 上述两个局部改动完成后清理替代路径和 deadlogic，
   检查 singleton/端口/缓存重入；仅最后跑一次 H240 fullres CSim，mask 0/57600、logits
   0/1800 mismatch、last_error=0，再交给用户手动 csynth。与 R7 同级资源及关键 II/局部
   时序比较，不能以 HLS 总 max latency 决定性能；实现必须 100MHz 合法布线、setup/hold
   均通过。上板按 10.5 验收，PPU row/Pool 不回退。若局部加权证据表明仍不足节省
   334365 cycles，先明确缺口，再有针对性地补做 10.2 的 EXEC delta，定位同一 Conv
   owner 内的剩余开销；不重启跨行 overlap 大改，不无条件要求先重复整套 prefix 实验。

当前 PARAM 的固定工作量：U39 为 18000 个 affine tile、18000 次源字读取、14520 个提交字；
U71 为 7200 个 tile/源字/提交字，两个 EXEC 均无跨字源读取。BLOCK5 final Stage 合计
9000 个最后分支 preadd word、21600 个 final word（其中 10800 次 residual 读取）、
36000 次 scratch cursor 服务；其他分支的链式 add 留在 PPU row Stage，不能重复计入。
连续执行下，单份系数缓存将 BLOCK5 qparam block 装载从 480 次减为 12 次；这只是工作量
变化，不代表已证明达到 0.60M Stage 预算或整网 <50ms。

#### 10.4.4 Round 7R 综合检查与补修结果

2026-10-19 前两轮报告已核对源码。16:48 版用时 9m33s，compact 有效提交从 R7 的
11 cycles/word 回退为 12，因而拆开 tile/commit 两个循环。23:21 完成的短提交版本用时
11m44s，仍无搜索失控。同级顶层比较如下，不与 Vivado placed 资源混比：

| 指标 | 已上板 R7 | 16:48 R7R | 23:21 短提交版 |
| --- | ---: | ---: | ---: |
| LUT | 352400 | 353011 | 351652 |
| FF | 186464 | 181800 | 180392 |
| DSP / BRAM18K / URAM | 1266 / 661 / 28 | 1266 / 661 / 28 | 1266 / 661 / 28 |
| HLS 估计周期 | 7.300ns | 7.300ns | 7.589ns |

23:21 版 LUT/FF 相对 R7 降低 0.21%/3.26%。生成 RTL 实例树中的 Conv、WinGen、SA、POST、
PPU consumer、BLOCK5 finalizer、Vec affine 及 commit pipeline 均各一份；SA 仍为
1024 DSP，Vec 原行 BRAM 仍为 8 个 BRAM18K。未发现主计算核复制或新增行存储。

**23:21 版方向有效，但当时暂缓实现。** `VEC_COMMIT_WORDS` 实际 II=2、目标 II=1，iteration
latency=2，已显著缩短原每字串行提交，但没有通过门槛。顶层摘要虽写 all loop constraints
satisfied，具体子报告及日志 `HLS 200-1470` 仍明确记录 II=2，不能据摘要宣称 II=1。
同时新增 `HLS 200-871`：Vec 的 7.589ns 超过 10ns-2.7ns=7.300ns 有效预算 0.289ns；
这不等于物理 100MHz 必然失败，但不应忽略新风险直接进入高代价 routing。

根因证据在 `.autopilot/db/run_fixed_affine_common_Pipeline_VEC_COMMIT_WORDS.verbose.sched.rpt`
及对应 RTL：第 1 拍读行 BRAM，第 2 拍完成 checked write 后，word/offset 才在
write-success 条件下更新，形成跨两拍控制反馈。局部关键链为 BRAM Q（1.237ns）→
写 gateway（5.498ns）→状态/游标寄存（0.427ns），父模块 commit 调用被报为 7.589ns。
不是 SA、WinGen 或 BLOCK5 新增热点。

下一补修只针对该链：在写前一次性验证行/通道块的目标 bank、首址、步长、末址及计数界限，
保留无效请求拒绝与错误报告；解除正常 word/offset 迭代对每字 checked-write 返回值的
反馈依赖，并对完成/错误状态形成明确寄存边界。仍只允许一个源读、一个 affine、一个
共同全字写尾及原逆行 alias 保护，复用原行 BRAM，不放宽时钟、不删检查或改被冻结模块。
下一报告须确认 II=1、无新增时钟预算 warning、singleton 与资源不回退。U39/U71 共
21720 次提交，II=2→1 的纯吞吐差仅约 0.2172ms，不能将它宣传成整网 <50ms；仍须按
tile、aligned staging、行启动及 BLOCK5 实际工作量加权，最终上板验收。

BLOCK5 preadd 18..20 cycles、final tile 16..18 cycles 及系数缓存保持不变；不以局部
iteration 或 HLS 总 latency 外推 Stage 7 <=600000 / Vec <=631133 已通过。

**2026-10-20 最新检查：源码补修已通过综合门槛，可以进入实现。** 顶层报告生成于
10-19 23:59:23，全部报告于 10-20 00:00 完成，综合约 10m01s、峰值内存 3.517GB。
预处理源码包含末址验证与解除提交状态反馈的补修；`VEC_COMMIT_WORDS` 实际 II=1、
depth=2、局部估计周期 6.735ns，Vec 为 7.093ns，顶层恢复 7.300ns；本轮无
`HLS 200-871`、`HLS 200-880`，日志中的 target/final II 均无违反。

最新顶层 LUT=352151、FF=180503、DSP=1269、BRAM18K=661、URAM=28。相对已上板 R7，
LUT -249（-0.07%）、FF -5961（-3.20%）、DSP +3，存储不变；新增 DSP 位于 Vec 地址/
跨度校验相关计算，SA 仍为 1024 DSP。RTL 实例树确认 Conv、WinGen、SA、POST、PPU
consumer、BLOCK5 finalizer、Vec affine、commit pipeline 各一份，结构检查与报告审计通过。
跨度验证仅执行前调用一次，延时 3 cycles、386 LUT、1 DSP，不在每字提交热循环中。

**下一步先验证物理结果，不继续盲改。** HLS LUT 仍约 103%，已有 SA 控制高扇出及
DATAFLOW 提示仍需关注；结合 R7 的 CLB96.27%，不能保证本轮 routing/setup/hold 必过。
若合法布线且 100MHz setup/hold 通过，再上板核验 Vec/BLOCK5 与整网周期；II=1 仅证明
提交吞吐修复，未证明 Vec <=631133、BLOCK5 <=600000 或整网 <50ms，不能用总 max
latency 或局部 II 比例替代实测。

**2026-10-20 实现与导出核查通过。** 全部 395696 条可绕线网络合法布线，错误为 0；
100MHz 下 WNS=+0.163ns、WHS=+0.010ns，TNS/THS=0。Placed logic/total LUT 为
208173/225886，FF174321、BRAM tile204、DSP1292、URAM28，CLB96.17%；相对 R7
资源略降，但 setup 裕度从 +0.476ns 收窄，不能称时序改善。`platform_h240_1020` 的
XSA/bitstream 与该实现 hash 一致，关键 Vec/BLOCK5 RTL 与本轮综合一致。App 已对齐
R7R 单图 profiling，复用 R6 artifact；SD 参数/输入已核验，旧 BOOT/输出已移除。
用户随后完成 App build 和本轮单图上板，结果见 10.4.5；该实现本身不证明 <50ms。

#### 10.4.5 Round 7R 上板结果：50.97039 ms，Vec 回退已修复

App=`INT8-BOARD-20261020-P7-H240W240-GRAY-R7R-B2`，平台=`platform_h240_1020`。
PARAM v4/75 UOP/16 EXEC、240x240 灰度输入不变，Conv capability=1，ERROR=0。
PL 为 **5097039 cycles = 50.9703900ms**；ARM 为
**1699004/33333000 = 50.9706297ms**，两者相差 0.240us（4.70ppm）。终端的
51ms 为整数显示，不能当作精确值，也不能四舍五入后宣称 <50ms。

实读 SD 的 `P2G.BIN` / `I2G.BIN` 与 R6 artifact 逐字节一致；`O2G.BIN` 为 57600 bytes，
与 frozen golden **0/57600 mismatch**，invalid=0、c0=4105、c1=53495，SHA256 为
`88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce`。
该同图输出与 R7 完全一致，沿用其 PA/mIoU=97.8722%/87.2564%；本轮未读取板级 logits，
也未完成 500 张验证集或至少三次连续时延测试。

| Stage | R7 cycles | R7R cycles | R7R ms | R7R 占比 | R7R-R7 cycles |
|---|---:|---:|---:|---:|---:|
| Conv row datapath | 3418908 | 3418908 | 34.18908 | 67.08% | 0 |
| PPU row（含上采样） | 289114 | 289114 | 2.89114 | 5.67% | 0 |
| BLOCK5 final | 773550 | 679846 | 6.79846 | 13.34% | -93704 |
| Vec | 667534 | 523904 | 5.23904 | 10.28% | -143630 |
| Pool | 135617 | 135617 | 1.35617 | 2.66% | 0 |
| 其他 | 49641 | 49650 | 0.49650 | 0.97% | +9 |
| **TOTAL** | **5334364** | **5097039** | **50.97039** | **100%** | **-237325** |

其他为 IDLE=267、FRAME_LOAD=14044、MAIN_CTRL=923、WEIGHT_LOAD=34416，其余 Stage=0。
Conv 内 WIN/SA/POST 各状态与 R7 逐项相同，三个 owner 总和各为 3418908，不能相加。
本轮整网节省 **237325 cycles（2.37325ms，4.45%）**，可归因于 Vec 减少 21.52%、
BLOCK5 减少 12.11%，未损失被冻结路径的性能。Vec 不仅消除了 R7 的回退，相比 R6
631133 cycles 也减少 107229（16.99%）；第一门槛 Vec<=631133 已通过。
但 Vec<=450000 与 BLOCK5<=600000 的工作预算尚差 73904/79846 cycles，不能整轮标记完成。
现有 counter 仍不能把 BLOCK5 收益精确拆成系数缓存、preadd 和 final tail 的独立贡献，
也不能把 POST wait 分类当纯 RAM 服务时间；本轮不据此宣称达到 compute-bound。

**下一步沿用局部优化方向，不回退本轮、不重启跨行 overlap。** PL 严格 <50ms 至少还需
节省 **97040 cycles（0.97040ms，当前总周期的 1.904%）**；ARM 也须独立满足 <50ms，
因此不能只卡 PL 等号边界，应预留测量和调用开销裕度。保留当前单图功能/实现可用点，
先按 10.4.3 分解 U39/U71 的剩余 Vec 服务与 BLOCK5 preadd/compose/final 服务，优先去掉
重复控制/地址/小循环启动开销，维持 singleton、原 RAM owner、整数饱和顺序及 alias 保护。
现有两个预算若均兑现且其他 Stage 不变，总周期为 4943289（49.43289ms）；这是条件预算，
不是性能预测或已验收结果。继续优化仍须通过同级资源、100MHz 物理实现及完整输出门槛。

### 10.5 Round 8：最终验收与交付

- [ ] 每次 csynth 优先对同级 R7（LUT352400、FF186464、DSP1266、BRAM18K661、URAM28）
  比较，R6 保留为历史参考；重点查 SA=1024 DSP 和 singleton、新增 port conflict/II violation、高扇出及
  clone。超过 baseline 5% 必须解释来源后再决定实现；不能与 placed 数值交叉比较。
  综合超过首版 11m18s 的两倍时检查日志/CPU/报告进度，不按 warning 单独判死锁。
- [x] 本轮 100 MHz 物理验收通过：route error=0、WNS=+0.163ns、WHS=+0.010ns；资源以上板 R7 的
  logic LUT212172、BRAM tile204、CLB96.27% 为参照。追求不回退，新增量必须证明必要；不能预设
  HLS 超资源会自动被 Vivado 拉回，也不能用升频代替 cycle 优化。
- [ ] 回到 binary2 单图 full MODE_RUN，保留全部 counter，至少连续测 3 次；**每次
  RTL cycles <5000000，ARM 未取整延时也 <50 ms**，两种时钟换算一致，ERROR=0，
  输出 0 mismatch，才宣称
  <50 ms。保持 SD/INIT/cache/UART 不计入，不删 counter 获得不可比的数字。
- [ ] 最后完成第 6.2 节的 500 张板级精度/延时验证，首张与本次输入相同；确认无旧行
  缓存污染、无跨图状态错误，输出保持 replay 一致。未过 gate 保留已上板基线，并把失败
  原因记录在本 workplan；普通实施细节不再逐条追加日志。

**当前：Round 7R 已通过 csynth、100MHz 实现及单图上板：50.97039ms、ERROR=0、板级 mask
0/57600 mismatch，较 R7 节省 4.45%；R7 保留为历史退路。Vec 回退修复门槛通过，但 Vec/
BLOCK5 后续预算、严格 <50ms、至少三次连续测量及 500 张板测仍未完成。此前 fullres CSim
logits 0/1800、mask 0/57600，资源与 singleton 门槛通过；本轮记录未修改 HLS/App/脚本，
PARAM/artifact、counter 及被冻结计算路径保持不变。下一步继续按 10.4.3/10.4.5 做局部收敛。**
