# Workplan 0805 - P7 INT4 混合精度与双 MAC/PE 升级计划

## 0. 文档目标

本计划指导当前 P7 NPU 从对称 INT8 扩展到 INT4，并同时支持：

- 二分类 `binary2` 模型；
- 20 分类 `cityscapes20` 模型；
- 原有 INT8 PARAM/artifact 继续运行；
- 内部卷积逐步启用 W4A4，不为两个模型或两种精度复制整套 NPU；
- 在不扩大 256-bit activation/weight 物理流宽度的前提下，使一个物理 INT8 K lane
  完成两个 INT4 MAC，目标有效 K 并行度由 32 提升到 64。

INT4 的目标不是只减小 BIN 文件，而是同时证明：

1. 模型在硬件约束 W4A4 QAT 下精度可接受；
2. 双 INT4 MAC/PE 能达到 `II=1`，且不会造成 DSP/LUT/routing 失控；
3. packed activation/weight 能让 WinGen 和 SA 的实际 K tile 数下降；
4. 整网板级延迟相对 P7 INT8 有可重复的改善。

执行顺序固定为：

```text
Round 0 -> Round 1 -> Round 2 -> Round 3 -> Round 4 -> Round 5
模型 gate   SA 原型    ABI/权重   activation   整网集成   实现/上板
```

任何一轮未过 gate，不得把过渡实现并入 P7 主路径。

---

## 1. 当前基线与性能上限

### 1.1 当前硬件

- 物理阵列：单实例 `32 output lanes x 32 K lanes`；
- `act_vec_t/wgt_vec_t = ap_uint<256>`，每个 word 为 `32 x INT8`；
- SA 的 K tile 循环 `II=1`，每周期最多发射 `1024 INT8 MAC`；
- 当前 SA 综合约使用 `992 DSP`，不是可由 HLS 自动拆分的双 INT4 乘法阵列；
- 小通道 pixel-parallel 路径把 32 个输出 lane 分给两个 16-channel 像素，未增加 K
  并行度；
- tensor descriptor、memory、WinGen、PPU、Vec 和 Pool 均按“一通道一字节”解释数据。

### 1.2 两个模型的计算量

| 模型 | CONV | 有效 MAC | 原始 INT8 tiling fill |
|---|---:|---:|---:|
| binary2 | 26 | 1.346G | 47.60% |
| cityscapes20 | 26 | 1.383G | 48.94% |

20 分类只扩大末层 `256 -> 20` 分类器，主体卷积完全相同。INT4 设计必须由 PARAM
format/precision descriptor 驱动，禁止按模型名或类别数选择不同硬件主路径。

### 1.3 端到端收益边界

以二分类 P7 板级约 `51.23M cycles @100 MHz` 为参考：

```text
Conv 约 32.23M cycles
非 Conv 约 19.00M cycles
```

即使 Conv 完美减半，整网下限仍约为：

```text
32.23M / 2 + 19.00M = 35.12M cycles ~= 351 ms
```

因此本项目的合理目标是：

- SA 算术吞吐：接近 `2x`；
- Conv stage：`1.6x~1.9x`；
- 整网：`1.25x~1.45x`，即约 `350~410 ms @100 MHz`；
- 只做 W4A8 时主要收益是权重存储，不能宣称计算 2x；
- 20 分类延迟预计只比二分类高几个百分点，full-resolution upsample 不受 SA INT4
  加速。

---

## 2. 全轮次绝对约束

1. 保持单实例 MainCtrl、Conv engine、WinGen、SA、PPU、Vec、Pool 和 Upsample owner；
   不为 INT8/INT4、C2/C20 各复制 datapath。
2. 保持物理 AXI/stream word 为 256 bit；INT4 通过每字节两个 nibble 实现，不扩大总线。
3. INT8 是兼容基线。精度格式必须来自 PARAM descriptor，不允许 runtime shape
   `if (in_c == 12...)` 或模型名判断。
4. 不全局把现有 `TK=32` 直接改成 64。必须区分：

   ```text
   PHYSICAL_WORD_BITS = 256
   INT8_K_LANES       = 32
   INT4_K_LANES       = 64
   ```

5. 禁止使用 `elem_bytes=0.5`。tensor/weight descriptor 必须新增 format enum，并由
   format-aware helper 计算 row bytes、channel offset 和 tile count。
6. 第一版 INT4 不改变 accumulator、bias、requant multiplier 的宽度：psum/bias 为
   INT32，乘法中间值维持当前安全宽度。
7. 不通过新增 current-conv-output 中间 FMBUF 往返换取实现简单；保持 P7 row/block
   数据流边界。
8. 不假设两个 `ap_int<4>` 乘法会自动合并到一个 DSP。最终是否为双 MAC/PE 必须由
   SA 独立 csynth hierarchy、operator report 和 RTL 仿真证明。
9. 每轮修改后扫描 legacy/dead logic；旧 INT8 兼容实现可以保留，但不得与 INT4 主
   datapath 形成两套长期常驻 engine。
10. 所有精度结论必须使用同一 checkpoint、artifact、PARAM、INPUT 和 golden；所有
    性能结论最终以 RTL stage counter 和 ARM timer 为准，不使用 HLS max latency。

---

## 3. 目标数据格式与硬件形态

### 3.1 格式定义

建议新增：

```cpp
enum tensor_format_t {
    TENSOR_FMT_INT8 = 0,
    TENSOR_FMT_INT4_PACKED = 1,
};

enum weight_format_t {
    WEIGHT_FMT_INT8 = 0,
    WEIGHT_FMT_INT4_PACKED = 1,
};
```

INT4 packed contract 固定为：

- 一个 byte 保存相邻两个 logical channel/K value；
- low nibble 保存偶数 logical index，high nibble 保存奇数 logical index；
- nibble 按 4-bit two's complement 解释，读取时显式 sign extend；
- K/channel 尾部不足 2 或 64 时补零；
- tensor row stride 按 `ceil(phys_c * bits / 8)` 计算并保持现有物理总线对齐；
- manifest 和 PARAM 同时记录 format，二者不一致时 exporter/compiler 立即失败。

### 3.2 目标 SA

保持：

```text
32 output lanes x 32 physical K lanes
```

INT4 模式下每个 physical K lane 接收：

```text
(act0, act1) x (wgt0, wgt1)
```

并向同一个 output-lane accumulator 提交：

```text
act0*wgt0 + act1*wgt1
```

因此：

```text
INT8 effective K lanes = 32
INT4 effective K lanes = 64
```

INT4 不需要输出两个独立乘积，只需得到精确的两项 dot-product 和。实现优先级为：

1. 显式 DSP48E2 packed dot-product/intrinsic；
2. 若 DSP packing 不可稳定实现，再评估局部 LUT dual multiplier；
3. 禁止直接把 DSP 数翻倍作为最终方案。

### 3.3 初始混合精度策略

第一版不追求所有 tensor 全 INT4。建议初始配置：

| 区域 | 初始格式 |
|---|---|
| 网络输入、第一层 | INT8 |
| ESP block 内部重卷积 | W4A4 |
| residual/add/concat 边界 | INT8，后续按敏感度收窄 |
| classifier weight/activation | INT8 |
| low-res logits、upsample、mask | INT8/UINT8 |

模型 gate 通过后再扩大 INT4 layer set。每层格式由编译器写入 PARAM，不在 HLS 中
硬编码层号。

---

## 4. Round 0 - 模型侧 W4A4 敏感度与 QAT Gate

### 4.1 修改范围

主要在 `D:\ESPNet`：

- `test_single_image_for_all.py`：将 hardware qconfig 参数化为 activation/weight bits；
- `qat_cityscapes20.py`：支持 W8A8、W4A8、W4A4 和 per-layer precision policy；
- `export_quantized_artifacts.py`：导出实际 bit width、qmin/qmax 和 precision manifest；
- binary2 对应 QAT/eval 入口：复用同一 qconfig/policy 实现，禁止复制量化数学。

### 4.2 量化契约

默认候选：

```text
activation INT4: per-tensor symmetric, zero_point=0, qmin=-8, qmax=7
weight INT4: per-output-channel symmetric, qmin=-7, qmax=7
```

同时比较 activation `[-8,7]` 与 `[-7,7]`，但最终只能选一套进入硬件 ABI。QAT 必须
继续模拟当前硬件的 rounding、saturation、ReLU、affine、add 和 concat 顺序。

### 4.3 实验顺序

1. 对两个模型运行 W4A8 PTQ/QAT smoke，确认权重收窄本身的精度敏感度。
2. 运行内部卷积 W4A4、首尾层/合并边界 INT8 的 mixed policy。
3. 生成逐层 sensitivity 表：每次只把一个 layer family 从 INT8 切到 INT4。
4. 根据 sensitivity 自动生成 precision policy manifest，不在 exporter 中写死层名。
5. 对候选 policy 做完整验证集 QAT/eval，并记录 checkpoint hash、PA、mIoU、各层格式。

### 4.4 Gate 0

- binary2：PA/mIoU 相对当前 INT8 QAT 绝对下降建议 `<=0.01`；
- cityscapes20：以当前约 `mIoU=0.44385` 为硬件约束 INT8 基线，首轮要求
  `mIoU>=0.42` 且绝对下降 `<=0.02`；
- 所有 activation zero point 为 0；
- add/concat 两侧若格式或 scale 不兼容，必须显式插入编译期 requant descriptor；
- precision manifest 能唯一复现 QAT eval，不依赖脚本默认值；
- 未通过 Gate 0 时停止 HLS 主线修改，只继续模型侧 mixed-precision 搜索。

---

## 5. Round 1 - 双 INT4 MAC/PE 独立 SA 原型

### 5.1 原则

本轮不修改 top、memory、WinGen 或 PPU，只建立可独立 csim/csynth 的 SA microbenchmark。
INT8 SA 作为同输入 golden，INT4 输入由软件 reference 解包计算。

### 5.2 两个候选内核

1. `dual_i4_dot_dsp`：显式 packed signed dot-product，优先使用 DSP intrinsic/受控
   arithmetic expression；
2. `dual_i4_dot_lut`：两个明确 4x4 乘法加和，用作正确性和资源对照，不默认进入主线。

每个原型均需覆盖：

- 所有 `[-8,7]` activation 与 `[-7,7]` weight 组合；
- 随机 64-lane dot product；
- K tail 1..63；
- paired/unpaired output issue；
- 多 K tile INT32 accumulation；
- 正负极值和溢出边界。

### 5.3 Gate 1

- CSim/RTL reference bit-exact；
- INT4 K tile loop 达到 `II=1`；
- effective K lanes 为 64，不能仍发射两个串行 32-lane cycle；
- SA DSP 不超过当前 INT8 SA 的 `1.10x`；
- SA LUT/FF 建议不超过当前的 `1.25x`；
- 目标时钟 10 ns 下有明确裕度，独立 csynth 时间可控；
- operator/hierarchy 报告证明每个 physical lane 未实例化两个普通 DSP multiplier；
- 两种候选都不过 gate 时，不进入 Round 2，先调整 packed arithmetic/RTL boundary。

---

## 6. Round 2 - PARAM ABI、编译器与 INT4 权重

### 6.1 ABI 修改

在 PARAM descriptor 的 reserved/versioned 字段中加入 tensor/weight format，升级 manifest
contract；保持 AXI-Lite、app buffer 地址和 top 函数参数不变。旧 PARAM v4 INT8 必须继续
解析为默认 INT8 format。

统一增加 helper：

```text
format_bits(format)
packed_elements_per_byte(format)
logical_values_per_axi_word(format)
packed_row_bytes(desc)
k_lanes_per_word(weight_format)
k_tile_count(K, weight_format)
```

禁止在 exporter、memory、WinGen 中分别维护不同公式。

### 6.2 工具修改

- `export_quantized_artifacts.py` 保存 `weight_int4.npy` 或通用 `weight_q.npy`，并记录
  bit width/qrange；文件名不能成为格式真值，manifest 才是唯一来源；
- `export_int8_hw_blob.py` 扩展为 precision-aware compiler，INT4 每 256-bit word 打包
  64 个 weight；
- contract checker 校验 nibble 顺序、sign extension、K tail、offset/size 和 PARAM hash；
- replay 增加 INT4 packed weight 解包路径，与 QAT fake-quant 输出逐层对齐；
- 离线利用率脚本同时报告 INT8 `ceil(K/32)` 和 INT4 `ceil(K/64)`。

### 6.3 HLS 修改

- param loader 解析 weight format；
- WBUF 物理宽度维持 256 bit；
- weight loader 根据 format 解释每 word 的 logical K lanes；
- SA INT8/INT4 在同一个 engine 入口按 descriptor 选择算术 mode；不得形成两套 top-level
  SA clone；
- 本轮 activation 暂可保持 INT8，只验证权重格式、ABI 和资源收益，不宣称整网 2x。

### 6.4 Gate 2

- 同一编译器可导出 binary2/cityscapes20 的 INT8、W4A8 artifact；
- 原 INT8 fullres CSim bit-exact；
- W4 权重 pack/unpack 与 QAT/replay bit-exact；
- 权重 payload 和对应 WBUF 有效容量约减半；
- 75 UOP、16 EXEC、26 CONV 及 P7 schedule 拓扑不变；
- hierarchy 中仍只有一个 SA/weight loader；
- csynth 不出现新的搜索爆炸、DSP 翻倍或明显 routing-risk clone。

---

## 7. Round 3 - Packed INT4 Activation、Memory 与 WinGen

### 7.1 Memory contract

新增 signed nibble helper，但保持 owner 集中在 memory 模块：

```text
read_packed_i4
write_packed_i4
read_tensor_word_by_format
write_tensor_word_by_format
```

要求：

- 不允许上层模块直接重复实现 nibble addressing；
- aligned 256-bit word 保持整字读写；
- narrow write 若不可避免，必须在编译期标记并统计，禁止恢复大范围 runtime RMW；
- INT4 tensor row/plane offset 必须由 descriptor helper 生成。

### 7.2 WinGen

- `scheduled_window_generator_row()` 仍是唯一主入口；
- schedule descriptor 增加 logical K lanes/format，不恢复 cfg shape 推断；
- INT4 每个 act word 发射 64 个 logical K value；
- first-C3、1x1、3x3 segment、boundary mode 均使用 schedule 中的 format；
- row cache/narrow cache 存储格式必须统一，禁止同时保留 byte cache 和 nibble cache 两套
  大型 datapath；
- padding、dilation、stride 和 K tail 由 lightweight CSim 覆盖。

### 7.3 PPU/Vec/Pool

本轮只实现 precision policy 实际需要的转换：

- Conv postprocess 可按 destination format clamp/pack INT4；
- add/affine/concat 边界按 policy 执行 INT4 或显式 INT4->INT8 requant；
- 不为每个 consumer mode复制一套 INT4 store tail；
- avgpool、classifier、upsample 首版可保持 INT8；若其输入来自 INT4，转换必须在编译好的
  producer/consumer 边界完成。

### 7.4 Gate 3（中间 fullres 节点）

这是本计划第一次整网 fullres CSim：

- binary2 与 cityscapes20 各跑一张固定样本；
- HLS low-res logits/fullres mask 与同一 precision policy replay 对齐；
- mask PA/mIoU 不低于 Gate 0；
- INT8 artifact 仍 bit-exact；
- INT4 Conv 的实际 K stream word 数按 `ceil(K/64)` 下降；
- no hot-RMW、no duplicate SA/WinGen/PPU main path；
- 通过后才进入一次完整 csynth，检查 BRAM/LUT/DSP/Fmax 和综合时间。

---

## 8. Round 4 - 整网 INT4 扩面与静态利用率优化

根据 Gate 0 sensitivity 和 Round 3 资源结果扩大 INT4 layer set，不一次性全开。

优先顺序：

1. `K>=64` 且计算量大的 3x3/1x1 Conv；
2. level3 的 C25/C28 宽层；
3. level2 的 C12/C16 层；
4. residual/add/concat 边界；
5. 第一层和 classifier 最后评估，默认维持 INT8。

对每个新增 family 同时核查：

- `ceil(K/32) -> ceil(K/64)` 是否真实减少；
- TM fill 是否成为新瓶颈；
- 当前 pixel-parallel 是否应继续使用；
- 双 INT4 sub-lane 应用于 K 并行还是双像素，不允许 runtime 大 mux 同时保留两种复杂
  调度；模式必须由编译器选择并写入 schedule；
- WinGen 物理读、SA issue、psum drain 和 PPU store 的理论/实测工作量。

### Gate 4

- Conv model cycles 相对 INT8 至少下降 `35%`，否则不得认为 dual MAC 已落地；
- 整网 csynth 只有一个 SA/WinGen/Conv/PPU owner；
- 资源和时序至少达到可进入 implementation 的水平；
- 不依靠 Tcl allocation directive 掩盖源码级 datapath clone；
- 第二次也是最终一次 fullres CSim：两个模型的 INT8 回归和 INT4 输出全部通过。

> 2026-08-11 执行修订：`35%` 只适用于扩大 A4 覆盖后仍通过 Gate 0 的候选策略。
> 现有 accuracy-safe release policy 未达到该覆盖率，因此不得将它标记为 Gate 4
> 性能通过，也不得继续训练新候选来强行追数值。本轮冻结现有策略，先完成双模型
> INT4 功能、综合、实现和板级实测；性能结论留到 RTL stage/exec counter 后判断。

---

## 9. Round 5 - 综合、实现与板级验收

### 9.1 Csynth audit

必须检查：

- SA instance/operator 数量；
- INT4 K loop II；
- DSP packed-dot-product 是否按原型映射；
- WinGen/SA/PPU clone count；
- BRAM/URAM/LUT/FF/DSP 与当前 P7 INT8 基线差异；
- high-fanout precision/qparam 控制是否被复制到 lane；
- 综合时间是否恢复到当前 P7 可控量级。

### 9.2 Implementation gate

- 100 MHz placement/routing 合法完成；
- WNS/WHS >= 0；
- 无 LUT/BRAM/URAM/CLB packing DRC；
- 无 global congestion level 6、node overlap 或大量 failed route；
- 若实现压力过大，优先缩小 INT4 layer set或修 packed datapath fanout，不降低 SA/TM/TK
  基线、不复制模块、不恢复旧路径。

### 9.3 Board gate

同一 bitstream 依次测试：

1. binary2 INT8；
2. binary2 mixed INT4；
3. cityscapes20 INT8；
4. cityscapes20 mixed INT4。

记录 ARM timer、RTL stage cycles、逐 exec prefix 和输出 mask。目标：

- INT8 相对当前 P7 cycle 回退 `<=1%`；
- mixed INT4 整网目标 `<=41M cycles`，stretch goal `<=37M cycles`；
- Conv stage 相对 INT8 下降 `>=35%`；
- binary2/cityscapes20 指标满足 Gate 0；
- PARAM、INPUT、输出 hash 和 build tag 全部可追溯。

---

## 10. 文件边界

### 模型/QAT（`D:\ESPNet`）

- `test_single_image_for_all.py`
- `qat_cityscapes20.py` 及 binary2 QAT 入口
- `deployment_profiles.py`
- `export_quantized_artifacts.py`
- hardware constrained quant math/replay/eval 脚本

### 编译器/工具（`D:\ESP_INT8\tools`）

- `export_int8_hw_blob.py`（后续可重命名为 precision-neutral 名称，但本轮不为重命名
  打断主线）
- PARAM/artifact contract checker
- replay、利用率和性能归因脚本

### HLS

- `npu_types.hpp`、`npu_schedule.hpp`、PARAM descriptor
- `param_dma.cpp`、`memory.cpp`
- `win_gen.cpp`、`sa_core.cpp`、`conv_engine.cpp`
- `ppu.cpp`、`vec_alu_engine.cpp`、`avgpool_unit.cpp` 中实际需要的格式边界
- SA microbenchmark 和必要 TB/cfg

### App

正常情况下只更新 build tag、PARAM format/version 检查和可选 precision 打印。输入/输出
地址、MASK.BIN 语义和计时窗口不得因 INT4 改动。

---

## 11. 禁止项与失败回退

禁止：

- 仅把 typedef 改成 `ap_int<4>` 后宣称双 MAC；
- 只压权重却按 2x 计算吞吐宣传；
- 为 INT4 新建第二套完整 NPU、SA、WinGen 或 PPU；
- 同时保留 byte/nibble 两套大型 cache 与 consumer；
- 使用 runtime layer shape 推断 precision mode；
- 未经过 QAT gate 就把所有 activation 改成 INT4；
- 用 HLS estimated max latency 替代板级 stage cycles；
- 为过资源而降低当前 INT8 SA 维度或破坏 P7 性能基线。

失败回退顺序：

1. 精度失败：缩小 INT4 layer set，保留敏感边界 INT8；
2. DSP packed 原型失败：尝试受控 RTL/DSP intrinsic，不直接污染 top；
3. LUT/routing 失败：放弃全阵列 LUT dual multiplier，回到 DSP packed 方案；
4. WinGen 未加速：检查 activation 是否真实 packed、K tile 是否仍按 32 计算；
5. 整网收益不足：用 RTL stage/exec counter定位非 Conv 瓶颈，不继续盲目扩 INT4；
6. 任一阶段无法达到性能/资源 gate：保持当前 P7 INT8 为发布版本，INT4 留在独立分支。

---

## 12. 最终完成状态

只有同时满足以下条件，才能称为 P7 INT4 完成：

- QAT、artifact、PARAM、HLS 对 INT4 的 qrange、rounding、zero point 和 nibble layout
  完全一致；
- 一个物理 PE 每周期精确完成两个 INT4 MAC contribution；
- INT4 effective K tile 为 64 lanes，且板级 Conv 周期显著下降；
- binary2/cityscapes20、INT8/INT4 共用同一 bitstream 和模块主路径；
- INT8 兼容性能不退化；
- fullres 精度通过、实现合法、100 MHz 时序通过；
- 所有性能结论来自可复现的板级 cycle 计数。

在上述状态前，文档中统一使用“INT4 原型”“W4A8”或“mixed W4A4”描述，不提前使用
“2x INT4 NPU”作为结论。

---

## 13. 2026-08-11 执行记录与顶层综合交接

### 13.1 冻结模型与产物

按当前决定停止新增模型训练和 precision policy 扩面，仅保留以下 release：

| 模型 | A4 producer | 使用 packed A4 输入的后继 CONV | 完整验证集结果 |
|---|---|---|---|
| binary2 | `level2_blocks.0` | `level2_blocks.1~5` | PA `0.9742648341`，mIoU `0.8460953762` |
| cityscapes20 | `level3_blocks.0` | `level3_blocks.1~5` | PA `0.8602459371`，mIoU `0.4303388803` |

- binary2 INT8 QAT 基线 mIoU 为 `0.8419225485`，现有 mixed policy 未退化；
- city20 INT8 QAT 基线 mIoU 为 `0.4438527763`，当前绝对下降约 `0.01351`，通过
  `mIoU >= 0.42` 和下降 `<= 0.02` 的 Gate 0；
- city20 的 `level2_blocks.0 + level3_blocks.0` 扩面候选在 5 epoch 后仅
  `mIoU=0.3622526257`，明确拒绝；binary2 扩面训练已终止，不进入 artifact；
- `hw_artifacts` 只保留 `binary2_p7_int4_release_{model,hw}` 与
  `cityscapes20_p7_int4_final_{model,hw}` 四个目录。

### 13.2 Round 1/2 硬件与 ABI 结果

- INT4 算术保留在唯一 `sa_core.cpp`；没有第二套 INT4 SA 源文件或 top-level engine；
- SA isolated CSim 通过；isolated csynth 的 K issue loop `II=1`；
- SA isolated csynth：DSP `1024`、LUT `70,877`、FF `4,145`、BRAM/URAM `0`，
  10 ns target 下 estimated period `7.197 ns`；
- 一个 256-bit physical word 在 A4/W4 模式承载 64 个 logical K value；W4A8
  仍按 32 个 activation K lane 执行，不把权重压缩误报为计算 2x；
- 两个 artifact 都是 PARAM v4、`75 UOP / 16 EXEC / 26 CONV / 21 schedule`；
- binary2 PARAM 为 `85,568 B`，SHA256
  `3cfce9dcc89d959c9cebefb503388744268b5f0355212b39b56542214bd514d7`；
- city20 PARAM 为 `90,176 B`，SHA256
  `fca7129d10c0ee6ad5de0afb9522cfb152d0639d02276cc7630e7f500a5e8e6e`；
- 每个模型均为 `1 x W4A4 + 23 x W4A8 + 2 x W8A8`；两个 RMW audit 的
  `problem_count=0`。

### 13.3 Round 3 fullres 功能状态

binary2 使用当前源码完成 fullres CSim：

- low-res logits 对 replay/golden：`0 / 16,384 B` mismatch；
- fullres mask 对 golden：`0 / 248,277` valid-pixel mismatch，invalid label `0`；
- low-res：PA `0.95601249`，mIoU `0.88477547`；
- fullres：PA `0.95621020`，mIoU `0.88380400`；
- 结果保存在 `binary2_p7_int4_release_hw/csim_round5_metrics.json`。

city20 也已使用当前源码完成 fullres CSim：

- low-res logits 对 PARAM-v4 integer replay：`0 / 163,840 B` mismatch，max diff `0`；
- HLS logits 经硬件固定点上采样后与 HLS fullres mask：`0 / 524,288 B` mismatch；
- low-res：PA `0.84765625`，mIoU `0.42857725`；
- fullres：PA `0.83604787`，mIoU `0.35328581`，invalid label `0`；
- 结果保存在 `cityscapes20_p7_int4_final_hw/csim_round5_metrics.json`。

PyTorch fake-quant hook golden 与 replay/HLS 的逐字节差异属于已确认的执行语义差异；当前
HLS 功能验收以同 PARAM 的 integer replay、HLS 固定点上采样一致性、label 合法性和
PA/mIoU 为准。至此 binary2/city20 的 Round 3 当前策略功能验证均已关闭。

### 13.4 Round 4 静态收益边界

现有精度策略的静态 row-dataflow 下界如下：

| 模型 | INT8 下界 | mixed INT4 下界 | 下降 | K issue 下降 |
|---|---:|---:|---:|---:|
| binary2 | 2,676,288 | 2,509,760 | 6.222% | 8.734% |
| cityscapes20 | 2,717,248 | 2,550,720 | 6.129% | 8.584% |

这说明 dual INT4 MAC 已在 A4 覆盖层真实减少 K tile，但 accuracy-safe 覆盖率不足以产生
原计划的 `35%` Conv 降幅。当前不再训练新模型，Round 4 以“冻结可用策略、诚实保留性能
gate 未决”收尾；最终收益必须由顶层综合后的硬件和板级 counter 给出。

### 13.5 进入手动顶层综合前的状态

已确认：

- `check_p6_hls_structure.py` 通过，MainCtrl/Conv/WinGen/SA/PPU 保持 single-owner；
- P7 contract tests `31/31`、工具单元测试 `9/9`、模型侧 contract tests `29/29` 通过；
- 单图评估工具已兼容 artifact 绝对样本路径与 `city.p` 相对样本路径，并有回归测试；
- HLS `src` 静态函数引用扫描未发现只定义一次且从未调用的 helper；
- legacy 扫描只命中 WinGen 的布局说明注释，无活动旧 datapath；
- binary2/city20 的 `PARAM.BIN == param_blob.bin`、`INPUTQ.BIN == input_q.bin`；
- 主 `hls_config.cfg` 未改，SHA256 为
  `05271865a11ddd24482eef45c235787b5b77b13bc119d7577f9fd6ea4468db38`。

尚未执行且不能提前宣称通过：顶层 csynth、implementation、100 MHz timing/routing 和
板级性能 Gate。现有 binary2/city20 INT4 功能链路复核完成后，交由用户使用冻结的主
`hls_config.cfg` 手动运行顶层综合。

---

## 14. 2026-08-14 顶层实现失败审计

### 14.1 结论

本轮 INT4 顶层 csynth 通过，但 100 MHz implementation 在 `route_design` 失败，Round 5
可实现性 Gate **不通过**，当前版本不能 package、不能导出 platform，也不能用更激进的
Vivado strategy 或更换 seed 代替源码修复。

这不是单一的“LUT 超过 100%”问题。HLS 估计 LUT 为 `426,827 / 341,280 = 125%`，但
Vivado 放置后实际 CLB LUT 为 `269,581 / 341,280 = 78.99%`；真正致命的是：

- 已占用 `42,442 / 42,660 = 99.49%` CLB，LUT/FF 因控制集、宽 mux 和局部硬宏约束无法
  高效打包；
- BRAM tile 为 `696 / 744 = 93.55%`，URAM 为 `112 / 112 = 100%`，WinGen、SA、PPU 和
  avgpool 的宽数据通路被迫穿越拥挤的 BRAM/URAM/DSP 列；
- 路由结束时仍有 `36,125` 条 net routing error 和 `38,606` 个 node overlap；这不是
  少量残余错误，而是结构性全局拥塞；
- QoR assessment 为 `Score 1 - Implementation will not complete`，最终
  `WNS=-3.144 ns`、`WHS=-0.034 ns`、setup failing endpoints `5,506`。

诊断报告冻结在 `route_diag_0814/`，对应失败 run 的时间戳为
`2026-08-14 18:32:50`。后续比较必须使用这些报告，不再只看 HLS resource estimate。

### 14.2 失败证据与热点

| 项目 | 本轮结果 | 判断 |
|---|---:|---|
| route 初始 global/short congestion | level 6 | 不可通过 |
| route 最终 failed nets | `36,125` | 严重结构性拥塞 |
| route 最终 node overlaps | `38,606` | 严重结构性拥塞 |
| route elapsed | `5:15:54` | router 已充分搜索，不是过早终止 |
| placed CLB occupancy | `99.49%` | 打包空间耗尽 |
| placed LUT | `78.99%` | 原始 LUT 百分比不是唯一根因 |
| BRAM / URAM | `93.55% / 100%` | 硬存储列成为物理瓶颈 |
| route WNS / TNS | `-3.144 ns / -5025.807 ns` | 100 MHz 完全未收敛 |

hierarchy 中仍是单套 MainCtrl/Conv/WinGen/SA/PPU，没有发现第二套 SA clone；主要逻辑
集中在以下单实例热点：

- `run_conv_rows_task`：`177,390 LUT / 101,148 FF / 1,260 DSP`；
- `shared_conv_row_engine`：`88,853 LUT / 37,569 FF / 1,093 DSP`；
- `ppu_consume_block5_final_row`：`33,423 LUT / 38,537 FF`；
- `consume_compact_conv_row`：`25,195 LUT / 16,901 FF`；
- `avgpool_unit_c3_fast`：`17,381 LUT / 13,908 FF`；
- 全局 FMBUF：`477 RAMB36 + 112 URAM`，且被多个热 consumer 共享。

最差 setup 路径由 FMBUF URAM 到 PPU BLOCK5 compact cursor，数据路径约
`12.804 ns`，其中约 `77%` 是 route delay。其余前列路径包括：

- URAM/BRAM 到 BLOCK5 finalizer cursor；
- BRAM 到 WinGen row loader/assembler；
- BLOCK5 schedule/qparam 到 PPU arithmetic；
- 深层 stage 状态到外部 `npu_stage_counter`；
- WinGen/SA 窄通道 pack 与 K pipeline 周边连接。

高扇出报告进一步定位到：

- SA `dual_i4` 控制：fanout `32,633`；
- SA `paired` 控制：fanout `7,104`；
- `conv_issue_kind`：fanout `7,818`，局部 net delay 约 `8.65 ns`；
- weight-load FSM：fanout `9,372`；
- PPU compact/BLOCK5 FSM 和 qparam：约 `2,560~4,733`；
- profiling `latched_stage`：fanout `1,104`，不是总资源根因，但已进入次关键路径。

### 14.3 源码级根因

1. **INT4 控制扇出大于其当前性能收益。** 当前精度策略只将 row-dataflow 理论下界降低
   `6.1%~6.2%`，但 `dual_i4/paired/issue_kind` 仍进入完全展开的 SA/WinGen 数据通路；
   一个运行时控制位驱动数千到数万个 mux/PE 输入。
2. **WinGen schedule 化尚未等于物理静态化。** `window_mode` 虽来自 PARAM，但
   `emit_narrow_words_for_mode()` 和 `emit_narrow_word_pair_for_mode()` 仍在窄 pack 内部
   根据 `mode/act_i4` 分支，形成大 mode mux 和跨模块控制网。
3. **BLOCK5 finalizer 仍有动态 cursor。** `ppu_read_block5_storage_cursor()` 以运行时
   `byte_count` 做 512-bit 拼接/位移，`ppu_finalize_block5_emit_tiles()` 又动态选择四组
   complete-partition qparam；它与 URAM 全局读口共同构成最差路径。
4. **FMBUF 是单体物理 owner，但热路径选择仍过于通用。** `memory.cpp` 中一个
   `s_fmbuf_bram` 和一个 `s_fmbuf_uram` 由 `read_phys_word()` 动态选择，WinGen、PPU、
   avgpool 等热 consumer 汇聚到同一组地址/数据 mux。单 owner 正确，但“每次访问都通用
   dispatch”不利于物理局部性。
5. **C3 avgpool 是既有大组合热点。** 32 pixel group 同时使用三组 complete-partition
   cache、动态 cache-word 选择和 32-way pack switch。它不是 INT4 新增根因，但在当前
   99.49% CLB 占用下成为压垮 route 的共同因素。
6. **profiling 边界跨层过深。** stage 状态从 engine 内部直接传播到 HLS 顶层输出，再进
   RTL counter，造成不必要的跨层长网。counter 本体只有约 `758 LUT / 1,159 FF`，因此
   不能靠简单删除 counter 解决拥塞，但必须把接口寄存化、局部化。

### 14.4 排除项

以下做法不作为下一轮主方案：

- 不降低 `TM/TK`、不缩小 SA、不放宽关键 loop 的 `II=1`；
- 不复制 INT8/INT4 WinGen、SA 或 PPU，不新增第二套 datatype engine；
- 不恢复 current-conv-output 的 FMBUF 中间往返；
- 不用 Tcl allocation/placement 指令掩盖源码 call graph 问题；
- 不先做未经地址生命周期证明的通用 4-bank BRAM；过去的通用 banking 尝试不能证明
  当前 PARAM v4 地址图可安全切分；
- 不把 `AggressiveExplore`、更换 seed 或关闭 DRC 当作修复；本轮 router 已运行五小时仍
  保留数万冲突；
- 不为省资源接受超过 `1%` 的 P7 INT8/C20 cycle 回退。当前 mixed INT4 理论收益有限，
  任何明显基线退化都会使 INT4 失去意义。

---

## 15. Round 5R 可实现性恢复计划

本轮只安排两个 implementation 检查点。第一检查点一次完成 5R-A/B/C；只有它仍失败，
才执行 5R-D。避免每个小改动都付出数小时实现代价。

### 15.1 5R-0：冻结基线与自动审计

1. 冻结当前 csynth、失败 routed checkpoint 和 `route_diag_0814/`，记录源码 commit/diff、
   HLS report 时间戳及 binary2/city20 PARAM SHA256。
2. 扩展现有 audit，统一输出：singleton clone count、关键 loop II、hierarchy LUT/FF、top
   fanout、CLB occupancy、BRAM/URAM、congestion level、failed nets 和 WNS/WHS。
3. 每轮先跑 structure/deadlogic check 和轻量 INT8/INT4 contract；5R-A/B/C 全部完成后只跑
   一次 binary2 与 city20 fullres CSim，避免反复长仿真。

### 15.2 5R-A：控制局部化与 profiling 寄存边界

目标：不复制计算 datapath，先消除横跨整个 Conv/SA/PPU 的高扇出控制网。

1. 在 Conv task/row 入口把 PARAM precision、paired 和 issue kind 解码成一个紧凑的
   row-local 配置；禁止在完全展开 PE/lane loop 内重复解析 descriptor。
2. SA 保持唯一 `systolic_array_core_row` call site，将 `dual_i4/paired` 在入口打一拍，并按
   `4 x 8` physical K group 建立局部控制副本。只允许复制控制寄存器，不允许复制 MAC、
   psum 或 weight buffer。
3. WinGen 在 row 入口完成 `window_mode + activation_format` 分发；内层 pack helper 接收已
   确定的局部模式，不再让一个 mode net直接驱动所有 byte/lane mux。
4. `prof_stage_id` 改为 MainCtrl 拥有的寄存状态。engine 只提交小型 stage event，顶层只
   输出寄存后的 stage id；禁止 engine 内部经引用直接驱动 AXI/top port。
5. 保持现有 stage 编号和 RTL counter ABI，先用 RTL TB 验证 stage bin 总和、start/done
   边界和 IDLE 计数不变，app 不因本轮改接口。

5R-A gate：

- WinGen/SA/PPU/Vec/Pool clone count 均为 `1`，SA K loop `II=1`；
- SA `dual_i4` fanout 从 `32,633` 降至 `<8,192`，`conv_issue_kind` 降至 `<4,096`；
- implementation top-50 setup path 中不再出现 HLS stage output 到 RTL counter 的跨层路径；
- INT8 和 mixed INT4 静态 issue/row cycle 不回退。

### 15.3 5R-B：BLOCK5 静态 gather + 单一 arithmetic tail

目标：移除当前最差的 URAM -> dynamic cursor/shift -> qparam mux 路径，同时保持一个
BLOCK5 finalizer owner。

1. 在 row 入口仅按 PARAM `pattern` 和 storage format 分发一次，分别进入固定的 L2/L3、
   INT8/A4 gather helper；禁止在 tile/lane loop 中运行时判断层 shape。
2. 用模板常量固定 source segment、byte count、destination lane 和位移，替换
   `ppu_read_block5_storage_cursor(unsigned byte_count, ...)` 的动态 512-bit shift。
3. gather 分支只生成一个标准 `act_vec_t` 和 tile qparam index；所有分支汇合后只调用一次
   shared affine/add/write tail，禁止因 L2/L3/A4/W8 建立多套 arithmetic datapath。
4. qparam 在 row 入口读入并按 tile 产生局部 8-lane/word slice；禁止把四个 complete
   qparam 数组广播到全部 finalizer lane。
5. 保留 PPU row-level fusion、BLOCK5 scratch 语义和 PARAM 驱动，不恢复 Vec standalone
   finalizer，也不增加当前 conv output 的中间写回。

5R-B gate：

- `ppu_read_block5_storage_cursor` 从 csynth hierarchy 消失；
- BLOCK5 finalizer LUT/FF 相对本轮分别下降至少 `15%/10%`，DSP 不增长；
- PPU qparam/tile 控制 fanout 降至 `<1,024`，或至少相对本轮下降 `60%`；
- fullres CSim 的 low-res logits、fixed upsample mask 和 error/status gate 不变。

### 15.4 5R-C：WinGen/SA narrow path 物理静态化

目标：保留 schedule 驱动和唯一 WinGen/SA，同时去掉窄通道 hot loop 内的运行时大 mux。

1. exporter/PARAM 继续提供 `window_mode`、activation/weight format 和 paired policy；HLS
   不从 `in_c/kernel/dilation` 推断模式。
2. 将 C3/C12/C19、generic narrow 的 byte-map 编译成小型静态 pack micro-kernel；outer
   row dispatcher 只选一次 micro-kernel，公共 loader、assembler、stream 和 SA 不复制。
3. 拆除 `emit_narrow_word_pair_for_mode()` 内逐 lane 的 `mode + act_i4` 级联判断；每个
   micro-kernel 使用固定 byte-map，统一输出同一种 packed stream token。
4. SA 使用 5R-A 的 group-local mode，保留同一 DSP packed PE。INT8、W4A8、W4A4 只改变
   operand encoding 和有效 K lane，不出现三套 PE。
5. 对照 PARAM 统计每种 micro-kernel 的调用层、issue 数和预期 K tile，防止降低扇出时
   重新引入 P6 shape-special-case 或错误 padding。

5R-C gate：

- `window_row_assembler`、SA、postprocess 仍各只有一个实例；
- 所有 K/issue hot loop `II=1`，WinGen 预期 word/issue 数与 compiler audit 完全一致；
- hierarchy 中 WinGen + SA LUT/FF 不增长，L5/L6 hotspot 数量明显下降；
- binary2/city20 fullres CSim 与各自 integer replay 维持当前验收结果。

### 15.5 第一实现检查点

完成 5R-A/B/C 后统一跑一次顶层 csynth 和 implementation。进入实现的 HLS gate：

- top LUT estimate 相对当前下降至少 `8%`，FF 不增长，DSP `<=1360+2%`；
- BRAM/URAM 不增长，singleton 与所有关键 `II=1` 通过；
- 综合时间可控，不出现新 clone 或 scheduling 搜索爆炸。

implementation pass gate：

- placed CLB occupancy `<95%`，目标 `<93%`；
- initial global/short/long congestion 不得出现 level 6；
- `failed nets=0`、`node overlaps=0`；
- `WNS>=0`、`WHS>=0` @ 10 ns；
- 只有同时满足以上条件才允许 package/export/platform。

### 15.6 5R-D：仅在第一检查点仍失败时执行的 memory/avgpool pass

1. 编译器对 binary2/city20 两套 release artifact 输出统一的 physical address/lifetime
   审计：每个 BRAM/URAM region 的最高访问 word、峰值同时存活范围、producer/consumer。
2. 只有审计证明存在未访问尾部时，才同步缩小 `FMBUF_*_AXI_WORDS` 和 memory map；禁止
   猜测式改地址、覆盖 live tensor 或无依据引入 4-bank。
3. 把 `read_phys_word()` 的 BRAM/URAM 类型选择提升到 task/row 入口，热 inner loop 使用
   已确定的 local reader；保留一个物理 memory owner，不为每个 consumer 复制 RAM。
4. C3 avgpool 保留 32-pixel 外层和三次 256-bit 写回，但把 32-way byte pack 改为顺序
   3-byte append/full-word emit；cache 按 8-pixel microgroup 读取或分段选择，禁止 32-way
   switch 和三组全广播 cache crossbar。
5. avgpool 改动前后按真实 read/compute/write 次数建立 cycle 公式；新实现不得增加
   `AVGPOOL` 板级理论 cycles，也不得增加 FMBUF 访问次数超过 `2%`。

5R-D 后再进行第二次且最后一次 implementation 检查；gate 与 15.5 完全相同。

---

## 16. 决策门槛

1. 若 5R-A/B/C 后 implementation 通过，立即停止结构修改，进入 binary2/city20 双模型
   上板性能与精度验收，不为追求更低 LUT 继续扰动 datapath。
2. 若 5R-D 后仍存在 level 6 或不能合法 route，当前 mixed INT4 作为功能原型冻结，不再
   占用 P7 release；发布版本回到已实现、已上板的 `P7-CLASS20-BOARD-0806`。
3. 即使实现通过，若 INT8/C20 MODE_RUN 回退超过 `1%`，或 mixed INT4 Conv/总周期收益
   不足以覆盖新增控制代价，也判定 INT4 主路径失败；不得仅凭“支持 INT4”合入发布版。
4. 最终 INT4 性能结论只允许使用 RTL stage/exec cycle counter 和 ARM timer 自检后的板级
   数据，不再使用 HLS max latency。
