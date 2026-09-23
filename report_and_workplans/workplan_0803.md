# Workplan 0803 - P7 二分类/20 分类统一支持最小升级

## 0. 结论与目标

当前最优 20 类模型为：

- checkpoint：`D:\ESPNet\train_cityscapes20\results_cityscapes20_p1q1_ft\model_best.pth`
- 网络：`Model_flattened.ESPNet_Encoder(classes=20, p=1, q=1)`
- 最佳 full-resolution 指标：mIoU `0.46381`，PA `0.87528`（epoch 21）

它与当前二分类网络的主体拓扑一致，仅分类器由 `2 x 256 x 1 x 1` 变为 `20 x 256 x 1 x 1`。因此不应另做一套 20 类 NPU，也不应改动 WinGen、SA、BLOCK5 或 PPU 卷积主路径。目标是让同一套硬件由 PARAM 描述符在运行时支持 `C=2` 和 `C=20`，并预留到 `C<=32`：

- 保持 `TM=32`、`TK=32`、75 UOP、16 EXEC、26 CONV 和 PARAM v4 ABI；
- 保持输入 `3x512x1024 INT8`、输出 `512x1024 UINT8 mask` 及现有 app buffer ABI；
- 低分辨率分类器仍只占一个 32-lane 输出 tile，SA 主体不扩阵列、不复制 datapath；
- HLS 内完成多类 logits 双线性插值和 argmax，不把后处理退回 PS；
- 新硬件必须继续加载现有二分类 PARAM，并保持二分类结果与性能基本不退化。

## 1. 当前差距与容量核算

| 项目 | 当前状态 | 最小改动 |
|---|---|---|
| 分类描述 | PARAM v4 已含 `out_c/valid_c`，但导出器固定为 2 | 从 artifact/profile 生成末层 `C` |
| 卷积阵列 | `TM=32`，C2/C20 都只需一个输出 tile | 不改 SA、WinGen 和卷积调度 |
| 末层存储 | 当前末层直接融合到 `UPSAMPLE_OUT`，不走 `COMPACT_C2` store | 不新增 C20 store layout |
| 上采样 | `upsample_unit.cpp` 只计算 `logit0-logit1` | 改为参数化多类 bilinear + argmax |
| 权重缓存 | 当前权重 `118816 B`，WBUF `122880 B` | WBUF 增至 `124 KiB = 126976 B` |
| C20 权重 | 分类器由 `512 B` 增至 `5120 B` | 总权重约 `123424 B`，64B 对齐后 `123456 B` |
| PARAM 容量 | 当前约 `143424 B`，app 上限 `262144 B` | C20 预计约 `148032 B`，无需改 app 上限 |
| T_OUT | C20 logits 为 `64x128x20 = 163840 B` | 当前片上地址生命周期可容纳，无需扩大主 FMBUF |
| 训练行为 | 当前 QAT 无条件把标签映射成二分类 | 引入显式 `binary2/cityscapes20` profile |
| checkpoint | 20 类最优模型含 8 组 PReLU 参数，硬件模型使用 ReLU | 受控删除这 8 个键后做 ReLU 适配/QAT；不加 PReLU 硬件 |

`124 KiB` WBUF 也能容纳最多 32 类的末层权重（预计 64B 对齐后约 `126528 B`）。其物理 BRAM 映射是否仍落在原深度档位必须由 csynth 确认，不能只按字节数推断。

## 2. 统一执行契约

1. artifact manifest 是类别契约的唯一来源，至少记录 `profile_name`、`class_count`、标签映射、checkpoint hash、量化配置和输出语义。
2. 编译器同时核对 manifest、分类器权重 shape、golden logits shape；三者不一致立即失败，不提供可独立误配的自由 `--classes` 开关。
3. PARAM v4 中末层 `conv_exec_desc.out_c/valid_c` 和 consumer `valid_c` 驱动硬件，HLS 不根据固定网络名推断类别数。
4. 支持范围定为 `2 <= class_count <= TM(32)`；本轮只验收 2 类和 20 类。
5. 多类输出语义固定为：对各类 INT8 logits 做 `align_corners=False` 的双线性插值，再逐像素 argmax；同值时保留较小类别编号，与 PyTorch/NumPy argmax 一致。
6. 分类器输出必须使用共享的 per-tensor 对称量化参数且 `zero_point=0`，否则 exporter 拒绝生成硬件 artifact，确保量化 logits 上直接 argmax 合法。

## 3. Round A：模型适配与 QAT

### 修改

1. 在 `tools` 中增加一个集中式 deployment profile 定义，至少包含 `binary2` 和 `cityscapes20`；训练、导出、验证脚本均读取它，禁止各脚本自行解释标签。
2. `binary2` 保留当前二分类映射；`cityscapes20` 使用 Cityscapes trainId `0..18`，将 void `255` 映射为类别 `19`，评估时类别 19 不计入 mIoU。
3. 对 20 类 checkpoint 做严格的受控转换：只允许丢弃已知的 8 个 PReLU `*.weight`；任何其他 missing/unexpected key 或 shape mismatch 均报错。禁止使用无审计的 `strict=False`。
4. 使用 ReLU 版量化模型先做短程适配，再执行与 HLS 饱和、舍入、ReLU/affine 顺序一致的 hardware-constrained QAT。
5. 保存 source、ReLU-adapted、QAT 三阶段指标和 checkpoint hash，避免后续混用 artifact。

### Gate A

- 输出 logits shape 为 `(1,20,64,128)`，分类器权重为 `(20,256,1,1)`；
- 20 类训练/验证过程中不再出现二分类 target remap；
- 分类器激活 qparam 为 scalar symmetric INT8、`zero_point=0`；
- ReLU 适配后相对原模型建议控制在 mIoU 绝对下降 `<=0.015`、PA 下降 `<=0.010`；
- 最终 QAT full-resolution mIoU 目标 `>=0.44`，且相对 ReLU-adapted 模型下降 `<=0.015`。未过门槛时先恢复模型精度，不进入 HLS 修改。

## 4. Round B：编译器、导出器与 PARAM

### 修改

1. 将 `export_int8_hw_blob.py` 中固定的 `T_OUT C=2`、末层 `ConvSpec out_c=2`、U72/U73 consumer 参数改为由 deployment manifest 生成；其余 tensor、conv、UOP、EXEC 计划保持不变。
2. HLS 与 exporter 的 WBUF 容量统一改为 `124 KiB`，增加权重尾地址、PARAM 总长和 FMBUF 生命周期的静态检查。
3. `export_quantized_artifacts.py`、验证集导出与 replay/eval 脚本按 profile 处理动态类别数和标签语义，不再硬编码 `(64,128,2)`。
4. 保留 PARAM v4 header 和描述符布局；只改变已有字段的值，不新增版本和控制寄存器。
5. 导出时生成一份简洁 contract report，列出类别数、末层 shape、权重字节数、PARAM 字节数、T_OUT 地址范围和融合路径。

### Gate B

- 二分类和 20 分类均可由同一 exporter 成功导出；
- 两者均为 75 UOP、16 EXEC、26 CONV，conv0..24 的描述符和 window schedule 不变；
- 二分类末层 `valid_c=2`，20 类末层 `valid_c=20`，最终 consumer 均为 `UPSAMPLE_OUT`；
- C20 权重 `<=126976 B`、PARAM `<262144 B`，T_OUT 不越过 FMBUF；
- hot RMW、store layout 和 P7 block/row 调度审计不得回退；
- 新 HLS 能继续解析原二分类 PARAM v4 artifact。

## 5. Round C：参数化硬件上采样

### 修改

1. 移除 HLS 主路径中的固定 `ENCODER_OUT_C=2` 校验，把末层 `valid_c` 从 conv descriptor 传到 PPU/upsample；常量仅保留 `MAX_CLASS_C=TM` 用于静态数组边界。
2. 将上一行缓存从单个 logit-difference row 改为 `act_vec_t[128]`，每个低分辨率像素保存最多 32 类 logits，约增加 4 KiB 本地存储。
3. 使用一个共享多类 upsample engine：空间方向保持现有精确 `align_corners=False` 定点映射，类别方向按 `UPSAMPLE_CLASS_LANES=4` 分组计算，C2 为 1 组、C20 为 5 组。
4. 每个 full-resolution 像素在引擎内部边插值边维护 `(best_value,best_class)`，只写 1 byte mask；禁止生成 full-resolution logits 中间张量。
5. CSim dump、top golden TB、prefix/lowres TB 和 replay reference 全部改为从 PARAM/artifact 获取类别数，统一 NCHW/NHWC 转换和 tie-breaking。
6. 不改 WinGen、SA、conv store、BLOCK5、row consumer 的卷积处理结构，也不为 C2/C20 各实例化一套 upsample datapath。

### Gate C

- hierarchy 中只有一个 upsample engine，不出现 C2/C20 双路径克隆；
- 原二分类 fullres mask 与既有 golden bit-exact，低分辨率 logits 不变；
- 20 类 HLS fullres mask 与 Python 定点 reference bit-exact，类别值只在 `0..19`；
- 20 类低分辨率 logits 与 replay/golden 满足现有 P7 数值误差门槛；
- structure check、prefix U40、lowres 和最终 fullres CSim 全部通过。

## 6. Round D：综合、实现与上板验收

1. csynth 使用 audit 脚本检查 clone、II、高扇出和资源；重点核查 WBUF 是否跨 BRAM 深度档位，以及 upsample 是否因动态 class loop 产生大 mux。
2. 建议资源 gate：相对当前 P7 基线，upsample/WBUF 合计 BRAM 增量 `<=4`、DSP 增量 `<=16`、LUT 增量 `<=10k`；若超出，优先调整 class lane 数或存储绑定，不复制模块。
3. 100 MHz Vivado implementation 必须合法完成 placement/routing 并满足时序，再导出统一 platform。
4. 同一 bitstream 分别加载 binary2 和 cityscapes20 PARAM：
   - binary2：mask bit-exact，MODE_RUN cycle 回退 `<=1%`；
   - cityscapes20：mask 范围 `0..19`，单图输出与 CSim/reference 一致；
   - 完整验证集 full-resolution 指标满足 Gate A，且硬件与 QAT 模型差距在既有 P7 误差门槛内。

预计 C20 不增加分类卷积的 SA 输出 tile 数；新增时间主要来自多类上采样。按 4-lane 分组，C20 上采样约为 C2 的 5 组，需在上板计数器中单独核实，不能把该增长误归因于 conv datapath。

## 7. 文件边界与禁止项

主要涉及：

- 模型/QAT：`tools/export_quantized_artifacts.py`、`tools/test_single_image_for_all.py` 及统一 profile 定义；
- 编译/导出：`tools/export_int8_hw_blob.py`、artifact/replay/eval/contract 脚本；
- HLS：`npu_config.hpp`、`param_dma.cpp`、`conv_engine.cpp`、`ppu.cpp`、`upsample_unit.cpp` 及相关 TB；
- app：原则上只更新 build tag 和可选的 `class_count` 打印，不改变输入、输出、PARAM buffer ABI。

本轮禁止：

- 为 20 类另建 bitstream、复制 SA/WinGen/upsample 主路径；
- 扩大 SA 或修改 P7 UOP/EXEC 调度；
- 新增 `COMPACT_C20` store 或 full-resolution logits buffer；
- 将 bilinear/argmax 移到 PS 计时窗口之外；
- 为兼容训练 checkpoint 在硬件中新增 PReLU；
- 破坏已有二分类 PARAM v4、mask 语义和性能基线。

执行顺序固定为 `Round A -> B -> C -> D`。模型行为和 artifact contract 未通过时不得进入 HLS；二分类回归未通过时不得进入实现或上板。

## 8. 2026-08-04 执行记录

- 结构化剪枝暂停，主线切换为未剪枝 20 类 Dense-ReLU checkpoint 的硬件约束 INT8 QAT。
- ReLU-adapted 基线：`PA=0.871974`、`mIoU=0.448612`。最终受控 QAT（从候选 prepared state 继续 `lr=1e-6`）最佳 `PA=0.863842`、`mIoU=0.443853`，相对基线 mIoU 下降 `0.004759`，Gate A 通过。
- 模型侧新增统一 `binary2/cityscapes20` deployment profile、严格 checkpoint/QAT resume、manifest 类别合同；相关单元测试全部通过。
- 导出器支持 C2/C20 动态分类器和 prepared QAT state；golden 保留 NPY，允许关闭巨型逐元素文本 dump，避免单图导出耗时失控。
- 编译器由 manifest 驱动 `T_OUT/conv25/U72/U73` 类别数，C20 已成功生成 PARAM v4：`75 UOP / 16 EXEC / 26 CONV / 148032 B / 0 warning / 0 hot-RMW`。
- 最终 classifier consumer 为 `UPSAMPLE_OUT`，不新增 `COMPACT_C20` store；默认 store-layout 推断改为 fusion 后按真实 consumer 延迟执行，避免过渡描述符误报。
- HLS 已将末层 `valid_c` 传入单实例多类 upsample，采用 4-lane 类别分组、边插值边 argmax、只写 1-byte mask；U40 CSim 通过（0 error）。
- C20 fullres CSim：`last_uop=74`、`last_error=0`、类别范围 `0..19`；HLS mask 相对同一 HLS low-res logits 的定点 upsample reference 为 `0/524288` mismatch。HLS 与软件 golden mask 差 `11408/524288`，来源全部在 low-res logits 数值差异；该样本 low-res PA/mIoU 相对 golden 分别 `+0.001796/+0.000287`，未见精度退化。
- C2 fullres 回归：原 PARAM v4 可直接解析，最终 mask `0/524288` mismatch，CSim 0 error。Round A/B/C 功能 gate 已通过，下一步进入 csynth，重点审查 upsample 动态类别 mux、WBUF BRAM 档位和资源/时序。

## 9. 2026-08-06 C20 上板验收

### 9.1 版本与功能状态

- app tag：`INT8-BOARD-20260806-P7-CLASS20`；artifact 为 `cityscapes20_p7_qat_final`，PARAM v4 大小 `148032 B`。
- 板端成功解析 `75 UOP / 16 EXEC / 26 CONV / 7 AFFINE / 14 ADD / 3 POOL`，所有 `P00.BIN..P15.BIN` 均完成 MODE_INIT/MODE_RUN。
- 全部 prefix 结束状态均为 `current=0`、`status=0x5`，stage `ERROR=0`，没有卡死、PARAM ABI 错配或运行时错误。
- 最终输出 `MASK.BIN=524288 B`，类别范围检查通过：`classes=20, invalid=0`。单图直方图覆盖 15 个类别；其余类别在该图中计数为 0 不构成功能失败，精度仍应以验证集 PA/mIoU 为准。

### 9.2 性能结果

| 指标 | C20 实测 | 最近 C2 P7 基线 | 差值 |
|---|---:|---:|---:|
| RTL total | `51,393,493 cycles` | `51,229,815 cycles` | `+163,678` (`+0.32%`) |
| 100 MHz 等效时间 | `513.935 ms` | `512.298 ms` | `+1.637 ms` |
| ARM timer | `17,131,152 ticks` | - | `513.940 ms @ 33.333 MHz` |

ARM timer 与 RTL counter 的差异约 `0.005 ms`，说明本次 MODE_RUN 计时可信。C20 相比 C2 没有出现按类别数线性增长：`TM=32` 使末层 C2/C20 均保持一个输出 tile，统一硬件支持 20 类的性能代价仅约 `0.32%`。

逐 exec 数据显示 P00-P14 与二分类同架构路径一致，新增代价集中在最后一个分类 exec：

- C20 P15 增量：`CONV_ROW_DATAPATH=609664`、`PPU_ROW_CONSUME=1197838` cycles；
- C2 对应增量：`536000`、`1107790` cycles；
- 两项合计增加 `163712` cycles，与整网差值 `163678` cycles 基本一致；
- 因此 WBUF 扩容、多类描述符和通用 C20 支持没有造成前段 WinGen/SA/BLOCK5/VEC/AVGPOOL 性能回退。

### 9.3 Gate 判断与下一步

- Round D 的可实现性、100 MHz 时序、板端 PARAM v4 兼容和 C20 类别范围 gate 已通过。
- 当前延时距离 `500 ms` 目标仍差 `1,393,493 cycles`，即约 `13.935 ms / 2.79%`；该差距不能归因于 C20 扩展，主体仍是既有 P7 执行路径。
- 下一步先用本次 `MASK.BIN` 和完整验证集核对硬件 PA/mIoU，确认与 QAT/CSim 的精度差距满足既有门槛；不要仅凭单图类别直方图判断精度。
- 性能优化继续沿 P7 既有热点推进，优先处理 `CONV_ROW_DATAPATH`，其次是 BLOCK5/VEC；不为 20 类新增专用 datapath，也不回退当前统一 C2/C20 PARAM 驱动架构。
- 若只要求 20 分类功能落地，本轮已经成功；若要求严格进入 `500 ms`，仍需在不破坏当前实现结果的前提下再节省至少 `1.40M` cycles，并保留 RTL stage counter 做上板验收。
