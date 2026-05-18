# INT8 NPU System Architecture for ESPNet_Encoder

## 1. 文档目的
本文档将现有 INT8 方案从“架构设想”收敛为可直接指导 HLS 编码的实现规格。目标不是继续讨论概念，而是冻结以下 6 件事：

1. 顶层 IP 形态
2. 内部子模块划分
3. 片上参数/特征图存储方式
4. 指令和参数描述格式
5. 与现有 PS 软件的接口变化
6. 支撑 `ESPNet_Encoder` INT8 推理达到 `50 ms` 级别的关键约束

## 2. 现有 FP32 工程对 INT8 架构的约束

### 2.1 现有系统瓶颈
根据 `D:\ESP_FP32\encoder_fp32\src` 与 `report_and_workplans`：

- 当前 FP32 路线由 PS 端反复调用 `npu_dispatch_conv`、`npu_dispatch_add`、`npu_dispatch_br`
- `avgpool`、`concat`、张量复用、DMA 轮询等待主要在 CPU 上完成
- 当前单帧端到端基线约为 `102.1188 s / image`
- 当前 FP32 正确性已经闭环，因此 INT8 阶段的主要矛盾不是功能正确性，而是吞吐率和调度方式

### 2.2 对 INT8 设计的直接结论
为实现 `50 ms` 级时延，INT8 版本必须满足以下约束：

1. 不再沿用“每个算子一次 DMA 往返 + CPU 轮询”的 FP32 调度模型
2. `conv / add / affine(BR) / relu / avgpool / concat` 必须尽量在单 IP 内完成
3. 中间特征图不得按层写回 DDR，只允许“输入一次读 DDR，输出一次写 DDR”
4. 控制流必须由 PL 内部微指令控制器串行执行，CPU 只负责 `INIT` 和 `RUN`

### 2.3 当前板级实现对频率目标的修正
根据已导出的 FP32 平台，当前板卡 `PL0` 实际工作频率为 `100 MHz`。因此 INT8 `v1` 版本冻结如下实现策略：

1. **首版实现目标频率**：`100 MHz`
2. **第二阶段优化目标**：`150 MHz`
3. **`200 MHz` 仅作为后续冲刺目标**，不作为 `v1` 默认验收频率

原因：

- 当前 INT8 top 已从 FP32 的多独立 IP 扩展为单一 HLS 顶层 IP，内部子模块和片上互连显著更重
- 功能未收敛前直接把实现验收绑死在 `200 MHz`，工程风险过高
- `100 MHz` 仍具备接近 `50 ms` 的可行性，前提是中间特征图驻留片上且 CPU 不再逐层 dispatch

## 3. 冻结后的顶层架构

### 3.1 顶层选择
`v1` 版本冻结为：

- **单一 HLS 核心**：`espnet_encoder_int8_core`
- **单一软件可见包装 IP**：`espnet_encoder_int8_top`
- **外部接口**：
  - `AXI4-M_AXI`：DDR 输入/输出数据
  - `AXI4-M_AXI`：参数 blob / 指令表加载
  - `AXI4-Lite`：控制寄存器、状态寄存器、性能计数器
- **内部通过 HLS dataflow 连接多个子模块**

不采用 FP32 时期的多独立 IP + 多 DMA 并发模式。原因很直接：FP32 方案的 CPU 调度成本过高，且中间特征图跨 IP 往返是 `50 ms` 目标的主要阻碍。

补充说明：

- 当前板级平台虽然保留了 `4` 个 AXI DMA IP，但本 `v1` top **不直接使用这 4 个外部 DMA**
- `frame_dma / param_dma` 在本规格中表示 top 内部的 HLS memory mover 逻辑，而不是板级独立 DMA 外设
- `m_axi` master 接口用于把整帧输入、整帧输出和参数 blob 统一收敛到单 IP 内部
- 软件侧固定对接的是包装后的 `espnet_encoder_int8_top`，固定寄存器表以 `06_DMA_and_AXI_Interface.md` 为准

### 3.2 内部子模块
`espnet_encoder_int8_core` 内部必须拆成以下逻辑子模块：

1. `instruction_fetch_decode`
2. `frame_dma`
3. `param_dma`
4. `on_chip_memory`
5. `window_generator`
6. `systolic_array_core`
7. `post_process_unit`
8. `avgpool_unit`
9. `concat_writer`
10. `perf_counter_and_irq`

补充说明：

- 上述是 **最终冻结的逻辑子模块边界**
- 当前 `02~07` 号文档采用的是**按功能域归档**，而不是“一个文档严格对应一个 HLS 源文件/硬件模块”
- 因此阅读时应以“模块职责是否覆盖完整、接口是否唯一冻结”为准，而不应以文档文件名是否与模块名一一对应作为判断标准

### 3.2.1 子模块与 Spec 文档映射

| 逻辑子模块 | 当前实现文件名 | 对应 spec 文档 | 说明 |
|---|---|---|---|
| `instruction_fetch_decode` | `src/if_dec.cpp` | `06_DMA_and_AXI_Interface.md` | `uop` / `tensor_desc` / `param_blob` 解析，控制状态机入口 |
| `frame_dma` | `src/frame_dma.cpp` | `06_DMA_and_AXI_Interface.md` | `gmem_frame_in / gmem_frame_out` 的 DDR 访存搬运 |
| `param_dma` | `src/param_dma.cpp` | `06_DMA_and_AXI_Interface.md` | `gmem_param` 的 DDR 访存搬运与参数预加载 |
| `on_chip_memory` | `src/memory.cpp` | `04_On_Chip_BRAM_Controller.md` | `FMEM / WBUF / QBUF / BRAM scratch` 组织与读写仲裁 |
| `window_generator` | `src/win_gen.cpp` | `04_On_Chip_BRAM_Controller.md`, `02_Systolic_Array_Engine.md` | 前者定义缓存与窗口生成机制，后者定义输出给阵列的激活向量格式 |
| `systolic_array_core` | `src/sa_core.cpp` | `02_Systolic_Array_Engine.md` | `TM=32, TK=32` 阵列、权重布局、循环顺序 |
| `post_process_unit` | `src/ppu.cpp` | `03_Data_Formatter_and_Quantization.md` | `CONV_POST / ADD_POST / AFFINE_POST / requant / relu` |
| `avgpool_unit` | `src/avgpool_unit.cpp` | `05_Hardware_Pooling_and_Activation.md` | `3x3 s2 avgpool` 数据路径与 3 个固定 pool 语义 |
| `concat_writer` | `src/concat_unit.cpp` | `05_Hardware_Pooling_and_Activation.md` | `concat` 目标地址、通道切片写入规则 |
| `perf_counter_and_irq` | `src/perf_irq.cpp` | `06_DMA_and_AXI_Interface.md` | 性能计数器、完成中断、状态寄存器 |

备注：

1. `05_Hardware_Pooling_and_Activation.md` 中还包含 `activation_unit` 的定义，但它在实现上属于 `post_process_unit` 末端的轻量子路径，不单独作为顶层逻辑子模块列出。
2. `07_ESPNet_Encoder_Uop_Schedule.md` 不单独对应某一个模块；它是所有逻辑子模块共享的**整网执行约束文档**。

### 3.3 数据流原则

- **数据流类型**：Weight-stationary
- **权重驻留位置**：片上 WBUF
- **特征图驻留位置**：片上 FMBUF
- **卷积输出位宽**：INT32
- **层间流转位宽**：INT8
- **残差/分支融合位置**：FMBUF + post process，不回 DDR

## 4. 模型范围与层级执行图
本规格仅要求支持当前 FP32 工程里已经落地的 `ESPNet_Encoder` 路径：

1. `Level1 Conv3x3 s2 3->16`
2. `B1 = concat(Level1, AvgPool(input)) -> affine/relu`
3. `Level2_0`
4. `Level2_Block0`
5. `B2 = concat(Level2_Block0, Level2_0, AvgPool(AvgPool(input))) -> affine/relu`
6. `Level3_0`
7. `Level3_Block0`
8. `B3 = concat(Level3_0, Level3_Block0) -> affine/relu`
9. `Classifier Conv1x1 256->2`

### 4.1 来自 FP32 软件调度的精确算子清单

- 卷积：`26` 个
- 累积加法：`14` 个
- 独立 affine/BR：`8` 个
- `avgpool 3x3 s2`：`3` 个
- concat 写入：`7` 个

### 4.2 各阶段张量形状

| Stage | Tensor | Shape |
|---|---|---|
| Input | `input` | `1 x 512 x 1024 x 3` |
| L1 | `level1_act` | `1 x 256 x 512 x 16` |
| B1 | `b1_act` | `1 x 256 x 512 x 19` |
| L2_0 | `level2_0_act` | `1 x 128 x 256 x 64` |
| L2_B0 | `level2_block0_act` | `1 x 128 x 256 x 64` |
| B2 | `b2_act` | `1 x 128 x 256 x 131` |
| L3_0 | `level3_0_act` | `1 x 64 x 128 x 128` |
| L3_B0 | `level3_block0_act` | `1 x 64 x 128 x 128` |
| B3 | `b3_act` | `1 x 64 x 128 x 256` |
| Output | `classifier` | `1 x 64 x 128 x 2` |

### 4.3 卷积模式全集
本 IP 只需支持以下卷积模式即可完整覆盖当前 Encoder：

- Kernel：`1x1`、`3x3`
- Stride：`1`、`2`
- Dilation：`1`、`2`、`4`、`8`、`16`
- Padding：`same`
- Bias：支持 `int32` bias，但当前大多数层可配置为 `bias_en = 0`

## 5. 参数规模与片上缓存策略

### 5.1 静态参数规模
依据现有 FP32 `TOTAL_WEIGHTS_SIZE = 431452 bytes`：

- 全模型 FP32 权重：约 `421 KB`
- 全模型 INT8 权重：约 `108 KB`

这意味着**整个 Encoder 的 INT8 权重可以一次性预加载到片上**。`v1` 版本采用：

- `WBUF`：一次性加载全部 INT8 权重
- `BQBUF`：bias / requant / affine 参数一次性加载
- 运行阶段不再为每层单独搬权重

### 5.2 特征图规模
INT8 下的关键特征图大小约为：

- Input：`512 x 1024 x 3 = 1.50 MB`
- B1：`256 x 512 x 19 = 2.38 MB`
- B2：`128 x 256 x 131 = 4.09 MB`
- B3：`64 x 128 x 256 = 2.00 MB`

因此片上设计必须支持“大 bank + 生命周期复用”，但**不应再按原始设想静态预留 `11.5 MB` 级全 URAM 空间**。当前综合修正后的实现约束为：

- 物理 `FMBUF` 固定为 `0x598000` bytes
- `FMBUF_URAM = 0x380000` bytes
- `FMBUF_BRAM = 0x218000` bytes
- `FMEM0/1/2` 仅保留逻辑 bank/view 语义，不再对应三块独立物理大 RAM
- 依靠 tensor 生命周期复用和 channel-slice view，而不是为所有阶段静态保留独立物理大 bank

原因：

- Vitis HLS 对当前目标器件给出的 URAM/BRAM 原始容量不足以容纳 `0x940000` byte 的三大静态 FMEM
- 若直接把大 FMEM 写成静态数组，综合会把它们实现成超量 BRAM/URAM
- 这会给 line buffer、FIFO、bank 复制和布线时序留下过小余量

## 6. 量化策略冻结
为降低硬件复杂度并保证 HLS 可落地，`v1` 版本冻结以下量化约束：

1. **Activation**：有符号 `INT8`，按 tensor 共享 scale
2. **Weight**：有符号 `INT8`，按 output channel 共享 scale
3. **Bias**：`INT32`
4. **Zero-point**：`v1` 强制使用对称量化，`zp = 0`
5. **Add / Concat 输入约束**：
   - 进入同一个 `add` 节点的两个输入必须共享同一 activation scale
   - 进入同一个 `concat` 目标 tensor 的所有分支必须先被 requant 到同一目标 tensor scale

这 5 条是硬件 spec，不再作为开放问题。

## 7. 50ms 级时延预算

### 7.1 计算预算
`32 x 32` 阵列在 `100 MHz` 下：

- 若按保守实现计算，每周期 `1024` 次 INT8 MAC
- 峰值约 `102.4 GMAC/s`

对约 `2 GMAC` 级 Encoder：

- 理论纯计算下界约 `19.5 ms`
- 若阵列有效利用率达到 `40%` 左右，纯计算约 `48.8 ms`

因此 `100 MHz` 首版下，“接近或达到 `50 ms`”并非不可能，但前提比 `200 MHz` 更严格：

1. 中间特征图不写回 DDR
2. 参数一次性预加载
3. `avgpool/add/affine/concat` 不再由 CPU 发起独立 DMA
4. `FMBUF` 生命周期复用必须有效，不能退化成频繁 DDR spill

### 7.2 端到端预算建议

| Item | Budget |
|---|---|
| 输入帧 DDR -> FMBUF | `< 3 ms` |
| 全部卷积与后处理 | `40 ~ 45 ms` |
| 输出写回 DDR | `< 1 ms` |
| 控制器开销 | `< 1 ms` |
| 余量 | `0 ~ 5 ms` |

## 8. HLS 代码组织建议
建议以当前 component 目录 `ESP_INT8_hls/` 组织如下文件：

```text
ESP_INT8_hls/
├── include/
│   ├── npu_types.hpp
│   ├── npu_config.hpp
│   ├── npu_uop.hpp
│   └── npu_q.hpp
├── src/
│   ├── int8_core.cpp
│   ├── if_dec.cpp
│   ├── frame_dma.cpp
│   ├── param_dma.cpp
│   ├── memory.cpp
│   ├── win_gen.cpp
│   ├── sa_core.cpp
│   ├── ppu.cpp
│   ├── avgpool_unit.cpp
│   ├── concat_unit.cpp
│   └── perf_irq.cpp
└── tb/
    ├── top_tb.cpp
    └── golden/
```

当前说明：

1. `int8_core.cpp / if_dec.cpp / memory.cpp / ppu.cpp / concat_unit.cpp / win_gen.cpp / sa_core.cpp / perf_irq.cpp / npu_q.hpp / top_tb.cpp` 是当前工程中已采用的实现文件名。
2. 逻辑模块名仍按 spec 中的全名表述，不因为实现文件名缩写而改变架构边界。

## 9. 编码前必须准备的离线产物
硬件写码前，导出脚本必须能生成以下文件：

1. `param_blob.bin`
2. `tensor_manifest.json`
3. `uop_queue.bin`
4. `input_q.bin`
5. `golden_output_q.bin`

其中：

- `param_blob.bin` 是硬件 `MODE_INIT` 唯一读取的静态参数镜像
- `uop_queue.bin` 仍保留为离线调试/比对产物，但运行时内容必须与 `param_blob.bin` 内的 `uop_table` 完全一致
- 字段定义以 `06_DMA_and_AXI_Interface.md` 为准

## 10. 子文档导航

### 10.1 模块一一对应版 spec

- `module_specs/00_Module_Spec_Index.md`
- `module_specs/01_int8_core.md`
- `module_specs/02_if_dec.md`
- `module_specs/03_frame_dma.md`
- `module_specs/04_param_dma.md`
- `module_specs/05_memory.md`
- `module_specs/06_win_gen.md`
- `module_specs/07_sa_core.md`
- `module_specs/08_ppu.md`
- `module_specs/09_avgpool_unit.md`
- `module_specs/10_concat_unit.md`
- `module_specs/11_perf_irq.md`

这套文档按最终硬件模块边界一一对应组织，用于日常实现和代码对照。

### 10.2 原功能域文档

- `02_Systolic_Array_Engine.md`：阵列维度、tile、流水
- `03_Data_Formatter_and_Quantization.md`：INT32 -> INT8 规则、add/affine 规则
- `04_On_Chip_BRAM_Controller.md`：bank 划分、地址生成、生命周期
- `05_Hardware_Pooling_and_Activation.md`：avgpool、relu、concat 写入
- `06_DMA_and_AXI_Interface.md`：寄存器、微指令、PS 驱动接口
- `07_ESPNet_Encoder_Uop_Schedule.md`：整网 tensor id、local scratch 和执行顺序

这组原始文档继续保留，作为共享约束、系统信息和来源追溯文档。
