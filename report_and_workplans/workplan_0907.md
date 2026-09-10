# Workplan 0907 - H256W512 双模型 INT8 NPU

日期：2026-09-07
当前基线：`bdb83c6 / P7-CLASS20-BOARD-0806`，PARAM v4，100 MHz，U71/U72 独立执行。

当前状态（2026-09-09）：H256W512 的模型、编译器、artifact、replay、双模型
fullres CSim 和 HLS 综合均已完成；首轮 Vivado implementation 在 route 阶段因
全局拥塞失败，尚未导出平台、尚未上板。第 9 节已据此升级为当前阻断项。

本文中的分辨率统一写成 `H x W`。用户提出的 `512 x 256` 解释为图像宽 512、高 256，即：

- 输入：`H=256, W=512, C=3`；
- 编码器 logits：`H=32, W=64, C=2/20`；
- NPU 输出：`H=256, W=512` 的 `uint8` mask；
- 下采样/上采样倍率仍为 8，宽高比仍为 2:1。

## 0. 结论与路线

**不需要从头重新训练。** `Model_flattened.ESPNet_Encoder` 只有卷积、池化、逐元素运算和归一化，没有绑定空间尺寸的全连接层；权重形状不会随输入分辨率改变。目标宽高仍可被 8 整除，网络拓扑、通道数和 32x32 MAC 阵列均可保持不变。旧二分类训练流程中还曾使用 `Scale(512, 256)` 做多尺度训练，因此二分类 checkpoint 对该尺寸已有一定适应基础。

但不能直接把现有 INT8 参数原封不动作为发布版本。输入缩小后，激活分布、小目标占比和边界细节都会变化；应重新做目标分辨率的校准和低学习率 INT8 QAT。20 类模型原训练流程主要使用 1024x512，分辨率下降对小类别更敏感，因此先测 FP32，再决定是否加一轮短程 FP32 适配。

确定采用以下决策树：

1. 用现有最佳 FP32 checkpoint 在 H256W512 完整验证集上评估。
2. 若 FP32 mIoU 相对原分辨率下降不超过二分类 `1.5` 个百分点、20 类 `2.0` 个百分点，直接进入重新校准和 INT8 QAT。
3. 若超过门槛，从现有最佳 checkpoint 做 `5~15` epoch、低学习率的 H256W512 FP32 微调，再进入 QAT；仍不从头训练。
4. 每个模型重新建立 observer 统计。可以继承已训练权重，但不能直接复用旧分辨率 prepared-QAT 文件中的 observer 状态。

Cityscapes 标签定义、二分类映射、20 类 `255 -> 19` 规则均不变，不重新生成标签。缓存必须继续区分 binary2 与 cityscapes20，但图像尺寸由 transform/manifest 显式记录。

## 1. 本轮目标

1. 同一套固定 H256W512 HLS 平台运行 binary2 与 cityscapes20，类别数继续由 PARAM 描述。
2. 保持 PARAM v4、75 UOP、16 EXEC（15 个有效执行项加 END），U71/U72 分离，不恢复 0906 的融合和 PARAM v5。
3. 重新生成两套 H256W512 INT8 QAT、量化产物、PARAM、golden、replay 和验证集输入。
4. HLS CSim 对两模型均完成全网，`last_error=0`；HLS mask 与同一 artifact 的 replay mask 一致。
5. 100 MHz 实板完整 MODE_RUN 目标：`<=15.0M` PL cycles，即 `<=150 ms`；期望区间为 `12.8M~14.5M cycles`。
6. 首轮已完成“只改变有效几何”的隔离实验；其实现结果证明旧物理存储规模无法继续沿用，片上存储压缩现为上板前阻断项。

### 长期 Roofline 目标

H256W512 NPU 的长期目标不是仅靠降低分辨率缩短延时，而是把工作点从明显的
memory-bound 区域推到 ridge point，进一步争取进入 compute-bound 区域。这里的
主要带宽约束是 PL 内部 feature-map/window/row-buffer 的读写与重排带宽，而不是只看
DDR frame DMA。后续性能改进必须优先提高卷积数据复用、减少重复片上访问，并扩大
WinGen、SA 和后处理之间的有效 overlap。

每轮优化用三类证据判断是否真正向 ridge point 移动：整网实际 MAC/cycle 与 SA
有效计算占比上升；WinGen/存储等待及 psum 输出等待占比下降；总 PL cycles 下降且不
依赖复制 SA/WinGen/PPU 或增加存储端口。若只把周期从一个 memory stage 转移到另一个
stage，或只增加理论峰值而没有提高持续吞吐，则不视为达到该目标的有效改进。

## 2. 性能预估

旧 H512W1024 平台的有效单图基线为 binary2 `51,319,833 cycles / 513.198 ms`、cityscapes20 `51,393,493 cycles / 513.935 ms`。新图像面积是旧版的四分之一，卷积、PPU、BLOCK5、Vec、AvgPool、输入 DMA 和输出 mask 工作量都主要按像素数缩放；权重装载和顶层控制只占很小的固定开销。

| 项目 | 旧版 H512W1024 | 四分之一面积理论值 | 新版验收值 |
|---|---:|---:|---:|
| binary2 | 51,319,833 cycles | 12,829,958 cycles | <=15,000,000 cycles |
| cityscapes20 | 51,393,493 cycles | 12,848,373 cycles | <=15,000,000 cycles |
| 100 MHz 延时 | 约 513 ms | 约 128.5 ms | <=150 ms |

这不是“算力提升四倍”，而是处理像素数减少到四分之一。边界行、启动/排空和固定控制会使实际加速比略低于 4 倍。若完整推理明显超过 15M cycles，必须用 RTL stage counter 判断是哪一阶段没有随面积缩放，不能仅接受总时间。

## 3. 全轮次约束

- HLS 构建固定为 H256W512，不加入运行时双分辨率选择器；旧尺寸由 Git/旧 bitstream 保留。
- 不改变 `TM=32`、`TK=32`、INT8 MAC、量化舍入、饱和、激活及 bilinear-logits-then-argmax 语义。
- 不新增第二套 WinGen、SA、PPU 或 memory owner；不通过复制 datapath 获得性能。
- PARAM 继续负责 tensor/conv/window/consumer/exec 描述；HLS 不根据网络名称或通道组合重新推断调度。
- 第一版保留当前 FMBUF 基址和容量，先只缩小有效 tensor 几何及 frame/logits/output 缓冲；避免把地址重排错误误判为分辨率错误。
- 保留 RTL stage counter，并保留已恢复的 3 个 conv-internal state 输出；它们只向邻近 RTL counter 输出 24 bit 状态，不参与数据通路。
- 二分类与 20 类必须分别使用自己的 checkpoint、qparams、PARAM、INPUTQ 和 golden，禁止只替换 classifier 文件。
- 所有 manifest 都同时记录 `height=256`、`width=512`；文件或文档不得只写歧义性的 `512x256`。

## 4. 固定几何合同

| 层级 | H | W | 说明 |
|---|---:|---:|---|
| 输入 | 256 | 512 | NHWC signed INT8，393,216 B |
| Level 1 / Pool 1 | 128 | 256 | 第一次 2x 下采样 |
| Level 2 | 64 | 128 | 第二次 2x 下采样 |
| Level 3 / logits | 32 | 64 | 第三次 2x 下采样 |
| 输出 mask | 256 | 512 | uint8，131,072 B |

对应 AXI 256-bit 深度：输入 `12,288 words`，输出 `4,096 words`。上采样倍率仍为 8，二分类与 20 类 logits 都只占一个 32 通道输出 tile。

## 5. Round 0：模型分辨率决策

### 修改范围

- `D:/ESPNet/train_cityscapes20/main.py`
- `D:/ESPNet/qat_cityscapes20.py`
- `D:/ESPNet/export_quantized_artifacts.py`
- `D:/ESPNet/test_qat_cityscapes20.py`
- `D:/ESPNet/test_export_multiclass_artifacts.py`

### 执行步骤

1. 先为 loader、logits 尺寸和 manifest 几何编写测试；`--height 256 --width 512 --target-scale 8` 必须得到 `32x64` logits 和 H256W512 metric target。
2. 修改 `export_quantized_artifacts.py` 中硬编码的 `Scale(1024, 512)`，改为显式 CLI 参数；训练、验证、校准、golden 必须共用同一 geometry 对象。
3. 分别用现有最佳 FP32 checkpoint 跑 500 张 H256W512 验证集，记录 PA、mIoU、逐类 recall/IoU 和原分辨率差值。
4. 按第 0 节门槛决定直接 QAT 或短程 FP32 微调。微调仅改变模型权重，不改变 p=q=1、类别定义和 ReLU/量化约束。
5. 为两模型重新校准至少 32 个 batch，再做 `3~5` epoch QAT，初始学习率 `1e-5`；若最后一轮仍稳定改善，可追加 `1e-6` 的短程微调。

### 精度 gate

- binary2 INT8：H256W512 mIoU `>=0.83`，并且相对同尺寸 FP32 的 mIoU 差不超过 `2.0` 个百分点。
- cityscapes20 INT8：H256W512 mIoU `>=0.42`，并且相对同尺寸 FP32 的 mIoU 差不超过 `2.0` 个百分点；必须同时列出 0..18 类结果，不能只看总体 PA。
- observer、activation zero point 和 weight zero point 仍满足硬件对称 INT8 合同。

若 FP32 本身未达到绝对门槛，不继续用更多 QAT epoch 掩盖分辨率损失，应先完成 FP32 适配；若 FP32 达标而 INT8 不达标，问题才归入 QAT/量化行为。

### Round 0 实测记录（2026-09-07）

已建立统一 `DeploymentGeometry`，训练、验证、QAT 和量化导出均显式使用 `H=256, W=512, scale=8`，并在 manifest 中记录 `32x64` logits。相关 geometry、导出和 QAT 合同测试共 `19 + 13` 项通过。

| 模型 | 原 H512W1024 FP32 mIoU | 直接 H256W512 FP32 | 目标尺寸适配后 FP32 | H256W512 INT8 QAT | 结论 |
|---|---:|---:|---:|---:|---|
| binary2 | 0.82078 | 0.80169 | **0.83027** | **0.83054** | FP32/INT8 gate 均通过 |
| cityscapes20 | 0.44861 | 0.32507 | **0.42445** | **0.41713** | FP32 gate 通过；INT8 绝对 mIoU gate 未通过 |

执行细节：

- binary2 先后完成 `5 + 5` epoch FP32 适配；最优 checkpoint 为 `D:/ESPNet/resolution_h256w512/fp32_binary2_ft2/model_best.pth`，PA `0.97143`、mIoU `0.83027`。
- binary2 从上述 checkpoint 重新校准 32 batch，并完成 3 epoch 全 INT8 对称 QAT；最优为 epoch 1，PA `0.97126`、mIoU `0.83054`，相对同尺寸 FP32 无显著量化退化。产物位于 `D:/ESPNet/resolution_h256w512/qat_binary2_int8`。
- cityscapes20 先完成 `5 + 10 = 15` epoch 的原 logits-resolution loss 适配，mIoU 仅从 `0.32507` 恢复到 `0.38950`。单独提高学习率到 `1e-4` 连续 3 epoch 反而降至 `0.37579`，排除“仅学习率不足”。
- 标签审计发现，先将 H256W512 标签最近邻降到 H32W64 会使 class 6/9/12/14/17 在约 `15%~20%` 的原本出现图像中完全消失；审计记录位于 `report_and_workplans/round0_lowres_target_audit_0908.json`。
- 保持网络推理图不变，训练时改为 `32x64 logits -> bilinear H256W512 -> full-resolution CE`。首段 3 epoch、`lr=2.5e-5` 依次达到 `0.41780 / 0.41978 / 0.42133` mIoU；再以 `1e-5` 收尾 3 epoch，最优提升到 PA `0.86180`、mIoU `0.42445`。
- cityscapes20 FP32 发布候选为 `D:/ESPNet/resolution_h256w512/fp32_cityscapes20_fullres_loss_ft2/model_best.pth`；SHA256 为 `9b36ee8c7787d217010f004040f2a62cd054bff58319c35fc01041ca3e9a3d05`。独立重载后完整 500 张复验得到相同指标。
- 原始双分辨率 500 张评估记录位于 `report_and_workplans/round0_fp32_resolution_metrics_0907.json`；两段适配的逐 epoch 日志和逐类 IoU 保存在各自 `summary.json` / `train_log.csv`。
- 20 类 INT8 QAT 使用上述 FP32 checkpoint 重新校准 32 batch，并以 `fullres` CE 完成 3 epoch、`lr=1e-5`；mIoU 为 `0.410777 / 0.399700 / 0.417135`。按“最后一轮改善”追加 2 epoch、`lr=1e-6`，mIoU 为 `0.415950 / 0.409094`，best 仍为 `0.4171349811`。
- 为补齐 Round 0 允许的 5 个主学习率 epoch，从同一 best prepared checkpoint 再执行 2 个 `lr=1e-5` epoch，mIoU 为 `0.415435 / 0.411209`，没有超过 best；该尝试保存在 `D:/ESPNet/resolution_h256w512/qat_cityscapes20_int8_fullres_loss_ft5`。
- 独立重载 `qat_cityscapes20_int8_fullres_loss/model_best.pth` 后结果完全一致：PA `0.8482446330`、mIoU `0.4171349811`，相对同尺寸 FP32 `0.4244532283` 下降 `0.7318` 个百分点，满足相对差值门槛但未达到绝对 `0.42` 门槛。逐类 IoU (0..18)：`[0.872315, 0.531151, 0.786974, 0.240479, 0.253089, 0.237780, 0.182687, 0.318598, 0.814472, 0.462786, 0.834203, 0.429911, 0.129840, 0.761936, 0.214400, 0.277550, 0.068635, 0.119921, 0.388838]`。
- QAT manifest 已记录 `loss_resolution=fullres`、H256W512/32x64 geometry、activation/weight zero-point 均为 0；验证目录为 `D:/ESPNet/resolution_h256w512/qat_cityscapes20_int8_fullres_loss_final_verify`。该 INT8 产物暂不作为 Round 1 发布输入。

**Round 0 状态：两模型 FP32 gate 均通过；binary2 INT8 gate 通过；cityscapes20 INT8 QAT 已完成但绝对 mIoU gate 未通过。** 在 cityscapes20 INT8 达到 `mIoU>=0.42` 前，不进入双模型 Round 1，也不导出其发布 artifact。

## 6. Round 1：编译器、artifact 与 replay

### 修改范围

- `tools/export_int8_hw_blob.py`
- `tools/hw_param_replay.py`
- `tools/export_val_hw_dataset.py`
- `tools/eval_single_hw_outputs.py`
- `tools/eval_val_hw_masks_fullres.py`
- 对应 `tools/test_*.py`

### 执行步骤

1. 在导出器中建立唯一的编译期 geometry：`input_h=256`、`input_w=512`、`logits_h=32`、`logits_w=64`、`scale=8`。`TENSORS`、75 条 UOP 的空间尺寸、window schedule 的 `out_w`、BLOCK5 行数和 frame metadata 都从该对象生成，禁止散落数字替换。
2. PARAM 仍为 v4。保持权重、qparam、EXEC 和描述符 ABI；tensor descriptor 和 UOP 字段写入新尺寸。导出时拒绝 checkpoint/class profile/geometry 不一致的组合。
3. 第一版沿用当前物理 FMBUF 基址与上限，只缩小有效 tensor extent 和 scratch 使用范围。运行 memory-lifetime audit，证明每个新 tensor 范围不重叠且不越界。
4. replay 从 PARAM tensor descriptor/manifest 获取 H/W，不再硬编码 `512/1024/64/128`；输入 reshape、logits reshape、上采样和 mask 大小全部由 geometry 驱动。
5. 验证集导出与评估脚本同样改为 manifest 驱动：INPUTQ 为 393,216 B，low-res target/logits 为 32x64，板端 mask 为 131,072 B。
6. 输出目录固定为：
   - `hw_artifacts/binary2_int8_h256w512_v4`
   - `hw_artifacts/cityscapes20_int8_h256w512_v4`
   旧 `*_int8_0809` 目录保持只读，不覆盖。

### Round 1 gate

- 两份 PARAM：magic 正确、version=4、75 UOP、16 EXEC、仅最后一个 END、U71/U72 独立。
- 所有 tensor/UOP 空间尺寸符合第 4 节，scale 恒为 8；类数仅为 2/20 的差异。
- replay 完整结束、无越界、输出尺寸正确；golden、INPUTQ、PARAM 的 SHA256 写入 manifest。
- 搜索发布路径中裸写的旧几何数字；除兼容测试数据外，不允许主路径继续依赖 H512W1024 常量。

### Round 1 执行记录（2026-09-08）

- `geometry_contract.py` 已成为 tools 侧唯一的 H256/W512 几何入口；编译器的 tensor、75 条 UOP、window schedule、BLOCK5 行数和 frame metadata 均由该 geometry 派生。
- `export_int8_hw_blob.py` 已支持 2/20 类动态 classifier，严格校验 checkpoint、类别数、golden 形状和 geometry；WBUF 改为 `124 KiB`，可容纳 20-class 的 `123424 B` packed weights。
- 二分类 PARAM v4 导出成功：`143424 B`，INPUTQ `393216 B`，logits/golden `4096 B`；严格 blob parse 通过，replay 与 hardware golden 为 `0 mismatch`。
- 20-class PARAM v4 导出成功：`148032 B`，INPUTQ `393216 B`，logits/golden `40960 B`。原 `24494/40960` 差异来自 PyTorch fake-quant hook 与 PARAM 定点 affine 的数值语义不同，不是 layout 或 replay 错误。导出器现保留 `model_golden_output_q.*` 作为模型参考，并以最终 PARAM v4 整数 replay 发布 `golden_output_q.*`；独立 replay 与发布 golden 的 SHA256 均为 `a36349f7...02332`。
- 两模型验证集导出脚本均以 H256/W512 完成单样本 smoke，`Single-sample check: PASS`；manifest 已记录 INPUTQ、golden、PARAM 的 SHA256。
- 新尺寸契约测试 `test_round1_geometry_contract.py`、`test_multiclass_deployment_contract.py`、`test_eval_single_hw_outputs.py` 合计 `15/15` 通过；新增回归保证 hardware golden 不覆盖 model golden。
- Round1 仅完成模型编译、artifact、replay 和评估工具迁移，未修改 HLS；二分类候选可进入后续 HLS Round2，20-class 候选仍受 Round0 mIoU 绝对 gate 约束。

## 7. Round 2：HLS 固定尺寸迁移

### 修改范围

- `ESP_INT8_hls/include/npu_config.hpp`
- `ESP_INT8_hls/src/int8_core.cpp`
- `ESP_INT8_hls/src/frame_dma.cpp`
- `ESP_INT8_hls/src/upsample_unit.cpp`
- 仅在静态检查证明必要时修改 `memory.cpp`、`avgpool_unit.cpp`、`ppu.cpp`
- `ESP_INT8_hls/tb/top_golden_sample_tb.cpp` 及相关 TB 路径/尺寸检查

### 执行步骤

1. 固定 `INPUT_FRAME_H/W=256/512`、`ENCODER_OUT_H/W=32/64`、`FULLRES_MASK_H/W=256/512`；`UPSAMPLE_SCALE=8` 不变。
2. 将顶层 AXI 深度改为输入 12,288、输出 4,096 words，并用 static_assert 将 pragma 数值和派生常量绑定，防止再次出现端口深度漂移。
3. 缩小 frame DMA 循环、upsample logits 行缓存和输出写循环。保持当前定点 bilinear 映射、4-class lane argmax 与 256-bit 顺序打包，不引入新的上采样实现。
4. PARAM INIT 后验证输入 tensor 为 H256W512、classifier/logits 为 H32W64；旧 H512W1024 PARAM 必须明确进入 ERROR，不能带错尺寸继续运行。
5. 保持当前 FMBUF/URAM/BRAM owner、SA/WinGen/PPU 调用图和地址基址；只清理本轮改动后确实无引用的尺寸专用 dead logic。

### Round 2 gate

- 结构检查确认单一 memory、WinGen、SA、PPU owner，TM/TK=32，PARAM v4，无 U71 融合/v5 路径。
- binary2 与 cityscapes20 各跑一次 fullres CSim；均 `last_error=0`、mask 尺寸 131,072 B、类别范围正确。
- HLS mask 与同一 artifact replay mask 0 mismatch；若保留已知定点差异，必须先定位并写明差异来源，不能只凭 PA 放行。
- 旧 H512W1024 PARAM 的负向测试必须被拒绝。

### Round 2 执行记录（2026-09-08）

- HLS 已固定为输入 H256W512、logits H32W64、输出 mask H256W512；顶层输入/输出 AXI depth 分别收窄到 `12288/4096` words，并由 static assertion 锁定。
- frame DMA 与 upsample 的循环和行缓存均继续由上述编译期常量派生；定点双线性、argmax、TM/TK=32、PARAM v4、FMBUF/scratch 物理基址和单 WinGen/SA/PPU 调用图未改变。
- PARAM DMA 新增部署几何门禁，只接受 H256W512 输入及 H32W64、2/20 类输出；旧 H512W1024 PARAM 的负向初始化测试已被拒绝。
- binary2 顶层 CSim：`last_error=0`，U72 logits `0/4096` mismatch，mask `0/131072` mismatch，上采样 `0/131072` mismatch。
- cityscapes20 顶层 CSim：`last_error=0`，U72 logits `0/40960` mismatch，mask `0/131072` mismatch，上采样 `0/131072` mismatch。
- Round 2 专用 HLS geometry/ABI 合同 `4/4` 通过；两次 fullres CSim 均约 `5m12s`，stream 最大深度由旧尺寸的 4736 降为 2368。

## 8. Round 3：综合、实现、App 与上板

1. 用主 `hls_config.cfg` 做一次完整 csynth；不在同轮加入 FMBUF 重排或新性能优化。
2. 核对三个 AXI master、AXI-Lite、四个 stage 端口和 100 MHz 约束；资源不得高于回退基线，关键循环 II 不得因缩尺寸退化。
3. 完成 Vivado implementation 后导出独立平台，建议命名 `platform_p7_h256w512_0907`。
4. App 修改：
   - `INT8_INPUT_BYTES=393216`；
   - `INT8_OUTPUT_BYTES=131072`；
   - PL 仍为 100 MHz，ARM timer 仍读 `CNTFRQ_EL0`；
   - 保留 binary2/cityscapes20 各自 MODE_INIT、ERROR 拒绝和 stage counter；
   - 使用独立 8.3 文件名，避免和旧尺寸文件混用：`P2H.BIN/I2H.BIN/O2H.BIN`、`P20H.BIN/I20H.BIN/O20H.BIN`。
5. 先做两模型单图测试。要求 ARM 与 RTL 时间一致、stage ERROR=0、输出尺寸/类别范围/主机指标通过；之后再导出并运行 500 张双验证集。

### 上板性能 gate

- 两模型完整 RTL total 均 `<=15,000,000 cycles`；stretch goal 为 `<=13,500,000 cycles`。
- 各主要 stage 相对 H512W1024 基线应接近四分之一；单项超过原周期的 35% 必须解释。
- binary2 与 cityscapes20 的末层仍各占一个 TM=32 tile，两者总周期差应小于 1%。
- MODE_RUN 计时不包含 SD、PARAM INIT、cache maintenance 和 UART；stage total 与 ARM 时间换算误差维持微秒级。

### Round 3 实现记录（2026-09-09）

HLS csynth 通过，估计时钟为 `7.300 ns`；Top 估计资源为 LUT `378888`、FF
`282039`、DSP `1313`、BRAM18 `1625`、URAM `112`。Vivado placement/phys-opt
也完成，post-place `WNS=+0.344 ns`，但这不能替代合法 route。

| 项目 | 本轮结果 |
|---|---:|
| placed CLB LUT / FF | `238177 / 206677` |
| placed BRAM tile | `688.5 / 744 = 92.54%` |
| placed URAM | `112 / 112 = 100%` |
| placed DSP | `1320 / 3528 = 37.41%` |
| unique control sets | `1895` |
| initial global/short/timing congestion | `level 6` |
| route elapsed | 约 `4 h 58 min` |
| failed signals / node overlaps | `45109 / 48640` |
| route 状态 | 非法，implementation 失败 |

路由中的 `WNS=-3.471 ns` 是大量网络未布通时的中间估计，不能作为最终时序数据。
router top-10 overlap 主要落在 `avgpool_c3_group32_pack_write` 驱动的
`s_fmbuf_uram_address1/d1`；其余冲突涉及 fixed affine、compact PPU、PARAM
读取和 WinGen。布局诊断还在 3x3 WinGen row assembler 附近报告 West/Short
`level 5`，该窗口 RAMB 密度达到 `85%`。完整证据保存在
`report_and_workplans/route_diag_0909/` 和 `D:/ESP_INT8_vp/ESP_INT8_vp.runs/impl_1/runme.log`。

根因不是分辨率迁移的计算逻辑，也不是新增的 3 个状态端口，而是**有效 tensor
缩小后物理存储没有同步缩小**：当前仍保留 `FMBUF=0x598000`、URAM 区
`0x380000` 和旧版 L2/L3/Pool/BLOCK5 scratch。H256 lifetime audit 中最大单个
有效 tensor 仅 `1073152 B`，部分对象却仍被固定到 `0x418000` 以上地址；巨大单体
FMBUF 使 Pool、WinGen、PPU 和 Vec 的地址、写数据及控制网络跨越全部 URAM/BRAM
列。恢复的计数输出仅增加 24 bit 低频状态线，无法解释数万条冲突网络。

## 9. Round 3R：物理存储与路由恢复

本轮只恢复可实现性，不改变 32x32 SA、WinGen schedule、PPU 算术、PARAM v4
字段、量化语义和理论处理周期；禁止增加外部访存或复制 datapath。

### 3R-A：H256 生命周期驱动的存储压缩

1. 扩展 exporter 的 memory audit，使其除 19 个 tensor 外还覆盖 L2/L3 branch
   scratch、BLOCK5 row-group、B2 backup、两级 Pool scratch 及所有 alias；生命周期
   必须由 75 UOP/16 EXEC schedule 推导，不能仅检查静态地址上限。
2. 所有容量从 H256W512 geometry 派生并 32-byte 对齐。重点重算当前仍采用旧尺寸的
   `FMBUF_L2_SCRATCH_SLOT_BYTES`、`FMBUF_L30_SCRATCH_SLOT_BYTES`、
   `FMBUF_L3B0_SCRATCH_SLOT_BYTES`、`B2_SRC1_BACKUP_*`、`BRAM_SCR0/1_BYTES`
   和 BLOCK5 row-group 占用；禁止只移动 base 而不缩短物理数组深度。
3. 由完整 lifetime 做确定性 interval allocation，输出新的 tensor/scratch base、物理
   high-water mark 和冲突证明。同步更新 `export_int8_hw_blob.py`、
   `hw_param_replay.py`、`npu_config.hpp` 及 contract test；PARAM ABI 仍为 v4，
   binary2/20-class 仅 classifier 数据不同。
4. 保持 `memory.cpp` 为唯一存储 owner，但按实际物理区缩短 URAM/BRAM 数组。目标是
   placed URAM `<=96`、BRAM tile `<=620`；硬门槛至少要求 URAM `<112`、BRAM
   明显低于本轮 `688.5`，否则不进入长时间 route。

### 3R-B：热路径访问局部化

1. 在 `memory.cpp` 内提供预解析的 URAM/BRAM word 接口；在 op/row 边界一次确定
   物理区和 row base，内层循环只递增 local word index，不再逐 word 经过
   `bank_base -> resolve_phys_addr -> URAM/BRAM mux`。
2. Pool tensor 全部放入经 lifetime 证明安全的 BRAM 区。C3 fast AvgPool 保持现有
   pack/write 次序和 II，其写回只能连接 BRAM writer，禁止继续扇出到全部 URAM。
3. 以相同方式局部化 fixed affine、compact PPU 和 WinGen 已知连续行的地址 context；
   不新增数组副本，不把 memory helper 移入各 engine，不引入第二 owner。
4. 3R-A 后若初始拥塞已降到 `<=4`，3R-B 只做 Pool 的确定性 BRAM 写回；若仍为
   `>=5`，再依照 congestion report 逐一处理 Vec/PPU/WinGen，禁止同时重写计算内核。

### 验证与停止门槛

1. 先运行 layout/ABI/结构检查，确认所有 tensor、scratch 和 alias 不重叠、不越界，
   单一 memory/WinGen/SA/PPU owner 不变，旧地址常量无主路径引用。
2. binary2 与 cityscapes20 各跑一次 fullres CSim：`last_error=0`，logits/mask 与各自
   replay `0 mismatch`；旧 H512W1024 PARAM 仍必须被拒绝。
3. csynth 必须保持 `7.300 ns` 量级，所有 WinGen/SA/PPU/AvgPool 关键 II 和 latency
   不得高于当前 H256 报告；资源必须达到 3R-A 门槛，且不得出现 engine clone。
4. implementation 先检查 placement：若 URAM 仍满占、BRAM 未明显下降或初始 route
   仍为 congestion level 6，立即停止并保存 DCP，不再等待数小时。最终 gate 为
   `failed nets=0`、`node overlaps=0`、`WNS>=0`、`WHS>=0` @ 100 MHz。
5. 只有 legal route 后才导出 `platform_p7_h256w512_0907` 并执行第 8 节双模型单图与
   500 张验证集；不得用 post-place 正 WNS 宣称实现通过。

### Round 3R completion record (2026-09-09)

- The H256 physical memory is now compacted to `0x1F0000` bytes: a
  `0x1C0000` URAM prefix plus a `0x30000` BRAM-only Pool tail. The estimated
  URAM footprint is 56 256-bit blocks, down from the previous 112-block
  full-device allocation. Pool traffic uses direct BRAM gateways, while the
  main tensor path uses direct URAM gateways; `memory.cpp` remains the sole
  owner of both arrays.
- Full-network CSim initially exposed a real lifetime bug at U68. The U54
  level-3 C1 result and the U55-U68 BLOCK5 branch workspace both started at
  `0x080000`, so later branches overwrote a still-live C1 tensor. The
  level-3 C1 scratch is now fixed at `0x1A0000` in both the exporter and HLS.
  The memory audit models this allocation separately and hard-fails unless
  `l3b0_c1_disjoint_from_block5=true`.
- Regenerated PARAM v4 hashes are
  `DED712B46D1FEA3DA0D3315BD45927380D85CD39F910F24B3342F072E9BD6ED4`
  for binary2 and
  `ECF2A7C2F07063C1BE359FCC2FDAB899596A54459C9390B7E12B67E2AE29BC01`
  for cityscapes20. Both lifetime audits pass all bounds, alias, overlap,
  backup-retirement, and L3B0 scratch-separation gates.
- Binary2 fullres CSim passes with `last_error=0`, logits `0/4096` mismatch,
  mask `0/131072` mismatch, and legacy H512 geometry rejected. Cityscapes20
  passes with `last_error=0`, logits `0/40960` mismatch, mask `0/131072`
  mismatch, and the same negative geometry check. Maximum stream depth is
  2368 in both runs.
- The final contract suite passes `26/26`; the architecture checker confirms
  one MainCtrl execute entry and no forbidden P6/PARAM-v5 path. A source-wide
  static-helper scan reports no unreferenced helper, and retired H512 memory,
  B2 backup, cold-RMW, and old FMBUF constants have no active HLS references.
- Functional repair and dead/legacy cleanup are complete. The next action is
  a manual top-level `csynth` with the production `hls_config.cfg`; no
  implementation should start unless resource and II gates in this section
  pass.
- After the final interface cleanup, the production-config binary2 CSim was
  rerun on the exact handoff source and passed in `4m52s`: `last_error=0`,
  logits `0/4096`, mask `0/131072`, upsample `0/131072`, legacy geometry
  rejected, and maximum stream depth 2368. No `csynth` was run by the agent.

## 10. 最终交付标准

- 模型：无需从头训练；有明确的 H256W512 FP32、INT8 QAT 和逐类精度报告。
- Python：训练/QAT、量化导出、PARAM 编译、replay、验证集导出和评估共享同一 geometry contract。
- HLS：固定 H256W512、H32W64 logits、PARAM v4、U71/U72 分离、单一 32x32 SA 数据通路。
- 功能：两模型 CSim/replay 对齐，单图和 500 张板测均无 ERROR，输出为 H256W512 mask。
- 性能：100 MHz 下完整单图不超过 150 ms，并保留 stage 级证据解释总周期。
- 实现：完成 Round 3R 后 BRAM/URAM 留出明确余量，100 MHz placement、route 和 timing 全部合法。
- 版本隔离：旧 H512W1024 源码、平台和 `*_int8_0809` artifact 不覆盖；新文件名和 manifest 必须明确 `h256w512`。
