# ESPNet Encoder INT8 NPU 架构说明与组会汇报材料

> 更新时间：2026-05-18  
> 适用版本：已上板验证基线 `INT8-BOARD-20260515-P2B-WGTFIFO-FLWIN`；最新待综合源码 `INT8-BOARD-20260518-P2D-4OPT-PROFV3`

## 1. 汇报主线

本项目当前 INT8 NPU 的核心目标是：将经过硬件约束 QAT 和导出器对齐的 ESPNet Encoder INT8 模型，映射为一组可由 FPGA PL 端顺序执行的微指令，在 Zynq MPSoC 平台上完成片上推理。

当前汇报应强调三点：

| 主题 | 重点 |
|---|---|
| 系统闭环 | PS 从 SD 卡加载输入和参数，PL NPU 完成 INT8 推理，结果再写回 SD 卡离线比对 |
| 架构可信性 | 板端 `D72OUT.BIN` 已与 HLS reference bit-exact，说明当前硬件数据流和量化行为可闭环验证 |
| 性能收敛 | 已从 817ms 优化到 468ms，下一步继续围绕 window generation、写回/RMW 和小通道 PE 利用率优化 |

当前项目已经不再停留在“模块能否跑通”的阶段，而是进入“可信上板结果 + profiling 归因 + 定向性能收敛”的阶段。

## 2. 顶层系统结构

系统由 PS 端软件、DDR/SD 卡数据、PL 端 INT8 NPU IP 和 Vivado block design 共同构成。

```text
SD Card
  ├─ PARAM.BIN     INT8 参数、UOP、tensor desc、scale/qparam
  ├─ INPUTQ.BIN    单张输入图像的量化 INT8 NHWC 数据
  └─ HLSREF.BIN    可选：HLS/CSim reference 输出

PS App
  ├─ 挂载 SD 卡
  ├─ 将 PARAM.BIN / INPUTQ.BIN 搬到 DDR buffer
  ├─ 通过 AXI-Lite 配置 NPU 寄存器
  ├─ 启动 MODE_INIT / MODE_RUN
  ├─ 读取 profiling/debug 寄存器
  └─ 将 D72OUT.BIN 写回 SD 卡，PC 端离线比对

PL INT8 NPU IP
  ├─ M_AXI 读取 input frame
  ├─ M_AXI 读取 param blob
  ├─ M_AXI 写回 output frame
  └─ AXI-Lite 控制、状态、debug、profiling 寄存器
```

当前 PS/PL 交互是显式寄存器控制模式。PS 不参与中间层计算，只负责准备输入、参数和输出 buffer，并在推理后读取计数器和保存结果。

## 3. 软件与硬件数据流

### 3.1 板端运行流程

| 阶段 | 作用 |
|---|---|
| `MODE_INIT` | PL 端解析 `PARAM.BIN`，加载 header、tensor desc、UOP table、qparams、weights 等片上参数结构 |
| `MODE_RUN` | PL 端加载输入 feature map，按 UOP 顺序执行整网 Encoder，最后写回输出 |
| debug stop-after | 可在指定 UOP 后 dump 中间 tensor，用于逐层定位 |
| profiling readout | PS 端读取 NPU 内部事件和周期计数，用于性能归因 |

当前完整模型 UOP 表规模如下：

| 类型 | 数量 |
|---|---:|
| `LOAD_FM` | 1 |
| `CONV` | 26 |
| `POOL` | 3 |
| `ADD` | 14 |
| `AFFINE` | 7 |
| `STORE/CONCAT` | 23 |
| `END` | 1 |
| 总计 | 75 |

其中 `CONV` 是主要计算负载；`ADD/AFFINE/STORE/POOL` 对应 ESPNet Encoder 中残差、BN/激活、concat/store 和下采样池化等结构。

### 3.2 数据格式

当前硬件侧采用 INT8 NHWC 语义：

| 数据 | 格式 |
|---|---|
| activation | INT8，NHWC，片上以 256-bit word 存储和访问 |
| weight | INT8，按 `param_id / output channel / k-tile` 索引 |
| accumulation | INT32 |
| output | INT8，经过 bias、mult、shift、clamp、ReLU 等后处理 |
| tensor descriptor | 描述 bank、base、H/W/C、physical channel 和 channel offset |

量化行为与导出器对齐：Conv/Affine 使用 per-channel 参数，Add 使用导出的 scale 对齐参数，Pool 支持 same-scale 或 requant 路径。当前验收标准不是追求 PyTorch fake-quant 的逐算子 bit-exact，而是硬件、HLS CSim 和导出 reference 的 bit-exact。

## 4. NPU 内部模块划分

当前 HLS 组件保持单一顶层 NPU IP，内部按功能域拆成多个逻辑模块。

| 模块 | 功能定位 |
|---|---|
| `espnet_encoder_int8_core` | 顶层控制状态机，处理 `MODE_INIT/MODE_RUN/debug/profiling`，调度各类 UOP |
| `param_dma` | 解析参数 blob，提供 tensor desc、UOP、conv/pool/add/affine qparam 和 weight 访问 |
| `instruction_fetch_decode` | 检查 UOP 数量、opcode、参数索引和基本合法性 |
| `frame_dma` | 在 DDR frame buffer 与片上 feature memory 之间搬运输入/输出 |
| `on_chip_memory` | 管理 URAM/BRAM feature buffer，提供 tile、packed word、byte 和 aligned row 访问 |
| `window_generator` | 根据 conv 配置生成 activation window stream |
| `systolic_array_core` | 执行 INT8×INT8 到 INT32 的 MAC 计算 |
| `post_process_unit` | 对 psum 做 bias、requant、activation，输出 INT8 |
| `avgpool_unit` | 实现 3x3 stride2 avgpool 及其 C3 fast path |
| `concat_writer` | 实现 concat/store 类 channel range 写回 |
| `add/affine datapath` | 实现 residual add 和 BN/scale/activation 类逐元素算子 |
| `perf_counter_and_irq` | 提供计数器、状态和可扩展中断/性能观测接口 |

这些模块不是并列的固定流水 stage，而是由顶层 UOP 调度器按操作类型调用。卷积路径内部有 row-level dataflow；不同 UOP 之间当前仍是顺序执行。

## 5. 片上存储组织

片上 feature memory 是当前架构能否跑通整网的关键。

| 存储 | 用途 |
|---|---|
| URAM feature buffer | 容纳主要大 tensor，例如输入、中间大特征图和最终输出 |
| BRAM feature/scratch buffer | 容纳局部 scratch、pool 临时结果、小 tensor |
| 256-bit word layout | 以 32 个 INT8 lane 为一个基本 packed word |
| physical channel padding | 对部分 tensor 使用 `phys_c` 和 `c_offset` 支持 channel view / concat |
| local scratch IDs | `LS_C1/LS_A/LS_B/LS_TMP` 用于 block 内部多分支临时结果 |

当前 memory map 的设计目标是：在 ZU15EG 的 URAM/BRAM 限制下放下 ESPNet Encoder 的关键中间特征，同时尽量避免大规模外部 DDR 中间层访问。除输入和最终输出外，中间 tensor 主要在片上复用。

## 6. 卷积主路径与调度方式

卷积是 NPU 的主计算路径。当前默认采用 row-level stream datapath：

```text
每个 CONV UOP:
  1. 读取 src/dst tensor desc 和 conv qparam
  2. 将本层权重按 k-tile / output channel 缓存到本地 buffer
  3. 对每个输出行执行 row-level DATAFLOW:
       window generation
         -> activation stream
       cached weight stream
         -> systolic array
       INT32 psum stream
         -> post process
       INT8 row buffer
  4. 当前输出行计算结束后，再写回片上 memory
```

这里最重要的设计选择是：DATAFLOW 区域内只读源 feature memory，不直接写目标 feature memory。输出先进入 row buffer，随后在 DATAFLOW 外写回。这是为了解决 HLS 曾经把 window generator、SA 和 writeback 合并成单个进程导致 RTL stream 死锁的问题。

当前卷积 UOP 满足 `out_c <= TM` 的硬件约束。若某层逻辑输出通道较多，由导出器拆分为多个 UOP 或配合 store/concat 写回。这避免了 activation stream 需要被多个 output-channel tile 重复消费的问题。

## 7. 脉动阵列与数据精度

当前计算阵列参数：

| 参数 | 含义 |
|---|---|
| `TM = 32` | 一次最多处理 32 个输出通道 lane |
| `TK = 32` | 一次读取 32 个 reduction/k-channel lane |
| activation word | 256-bit，包含 32 个 INT8 activation |
| weight word | 256-bit，包含 32 个 INT8 weight |
| psum word | 32 lane INT32 partial sum |

计算过程是 INT8 activation 与 INT8 weight 相乘，在 INT32 中累加。后处理使用导出的 bias、mult、shift 和 activation 类型，最终输出 INT8。

当前硬件阵列的主要问题不是 MAC 单元数量不足，而是有效利用率偏低。原因包括：

| 原因 | 说明 |
|---|---|
| 小输出通道层 | 很多层 `out_c=12/16/25/28`，无法填满 32 个输出 lane |
| 小输入通道 3x3 | `in_c=12/19/25` 时，window packing 与 k-tile 填充效率有限 |
| 每行重喂权重 | 当前 row dataflow 为每个输出行重新向 SA stream 喂 cached weights |
| post/writeback 开销 | 对小通道输出，后处理和 partial-word 写回占比明显 |

离线模型估算的 weighted arithmetic PE fill 约为 `47.6%`，但端到端有效利用率远低于这个数，因为 window、post、writeback 和调度开销占比较大。

## 8. 非卷积算子支持

### AvgPool

AvgPool 主要用于前端和多尺度路径下采样。当前已实现 C3 fast path：对输入 3 通道图像/特征，使用 packed row 读取减少 3x3 邻域访问次数。

### Add

Add 对应 residual connection。硬件按 tile 读取两路 INT8 输入，使用导出的 add qparam 做 scale 对齐、加法、requant 和 activation，再写回目标 tensor。

### Affine

Affine 对应 BN/scale/activation 类逐元素变换。硬件按 channel block 读取 per-channel mul/bias/shift，对输入 tile 做 INT8 到 INT8 的仿射变换。

### Concat / Store

Concat/Store 用于将多分支结果写入目标 tensor 的指定 channel range。当前通过 `c_offset/valid_c` 与 tensor physical channel 描述支持 channel 拼接和局部写回。

## 9. 已实现与待验证的计算加速手段

当前性能优化可分为四类。表中“已上板验证”指已经进入 P2B bitstream 并完成 full `MODE_RUN`；“源码已实现”指 P2D 当前源码已通过整网 CSim，但还需要综合和上板验证真实收益。

| 类别 | 内容 | 状态 | 作用 |
|---|---|---|---|
| 专用 window path | 第一层 C3 3x3 stride2 行段读取、small-C stride1 reuse | 已上板验证 | 降低 3x3 window generation 和 packed read 成本 |
| 专用 window path | C19 stride2、C131 stride2 | 源码已实现，待综合/上板 | 针对 level2 early conv 和 U40 大通道降采样卷积减少动态分段调度 |
| packed data access | 256-bit packed tile read/write，aligned full-tile read，连续 row segment 读取 | 已上板验证 | 减少 byte/tile 级访存和动态 lane 拼接 |
| row-level dataflow | 每个输出行内 overlap window、weight stream、SA、post | 已上板验证 | 避免整层 dataflow feedback 死锁，同时保留局部流水 |
| writeback 优化 | padded direct write、减少部分 RMW | 已上板验证 | 降低 partial-word read-modify-write 成本 |
| writeback 优化 | C12/C16 compact row aligned write | 源码已实现，待综合/上板 | 将部分 scratch 输出从逐像素 RMW 改为整字写回 |

这些优化都遵循一个边界：不改变 UOP 协议，不改变 QAT/export 的 INT8 语义，不引入大规模中间 DDR 交换。

## 10. 性能与正确性结果

### 10.1 正确性

当前可信上板版本 `P2B-WGTFIFO-FLWIN` 已完成完整 `MODE_RUN`，输出与 HLS reference bit-exact。

| 比对项 | 结果 |
|---|---:|
| output bytes | `16384` |
| mismatch | `0 / 16384` |
| max abs diff | `0` |
| SHA256 | `7CFDE9B332E7426B8F4CEA5EAD6FC3C8D878C8F83ADF2B333405B89EE41D9364` |

这说明：参数导出、UOP 调度、片上 memory layout、INT8 量化后处理和最终输出路径已经形成可信闭环。

### 10.2 实测性能演进

| 版本 | 主要变化 | Full `MODE_RUN` |
|---|---|---:|
| profiling v1 | 初始 profiling 硬件 | `817 ms` |
| `PROFWG1` | window/packed path 初步优化 | `526 ms` |
| `P2A-PROFV2` | profiling v2 与进一步 packed path | `511 ms` |
| `P2B-WGTFIFO-FLWIN` | weight FIFO 收窄、first-layer/window/writeback 优化 | `468 ms` |
| `P2D-4OPT-PROFV3` | C19/C131 专用 path、compact row write | CSim 通过，待综合/上板 |

P2B 相比 profiling v1 约 `1.74x` 加速，相比 `PROFWG1` 继续降低约 `11%`。但距离 50ms 目标仍有约一个数量级差距。

### 10.3 P2B profiling 数据

| 计数器 | 数值 |
|---|---:|
| full cycles | `46,893,423` |
| full latency @100MHz | `468 ms` |
| executed UOP / CONV | `74 / 26` |
| window read ops | `4,419,328` |
| window words | `2,760,704` |
| weight words | `468,992` |
| SA steps | `2,760,704` |
| psum words | `630,784` |
| output tiles | `630,784` |
| RMW ops | `1,179,648` |
| model cycles | `8,363,584` |

`model_cycles` 远低于实测 cycles，说明当前仍存在大量 HLS schedule、访存仲裁、stream back-pressure 或非卷积路径开销。profiling v3 的目的就是进一步拆分这些 residual。

## 11. 当前瓶颈判断

基于上板 profiling 和离线性能归因，当前主要瓶颈不是单纯 MAC 计算，而是以下几类：

| 瓶颈 | 现象 | 优化方向 |
|---|---|---|
| Window generation | level2/level3 大量 3x3 小通道层由 window read/pack 主导 | 专用 stride/dilation path、减少动态 lane pack、复用行/列窗口 |
| Writeback/RMW | 小通道输出触发 partial-word RMW | compact row write、aligned full-word write、调整 tensor layout |
| Post/output drain | 小 `out_c` 层仍承担 32-lane 结构开销 | 小通道专用 post path 或多空间点并行 |
| PE 利用率低 | weighted PE fill 约 47.6%，端到端利用率更低 | 同时计算多个空间点，或重排 small-Cout 层映射 |
| Row-level weight feed | 每个输出行都要重新向 SA stream 喂 cached weights | 评估跨行 weight reuse 或更深层 pipeline |

当前短期目标是先稳定进入 `300-400 ms` 区间；50ms 目标需要更激进的结构并行化，而不是只做局部 HLS pragma 或小路径修补。

## 12. 验证与工程流程

当前推荐验证链：

```text
HLS 修改
  -> 整网 CSim，确认 top_tb 0 errors
  -> C synthesis
  -> audit 综合报告
  -> package IP
  -> Vivado system implementation / bitstream
  -> export platform
  -> Vitis clean rebuild platform/app
  -> 上板 full MODE_RUN
  -> 保存 D72OUT.BIN
  -> PC 端离线 bit-exact 比对
```

上板前必须重点检查：

| 检查项 | 不能接受的情况 |
|---|---|
| dataflow | `HLS 214-475` process merging / feedback |
| stream | `HLS 200-975` 同一 stream 同函数读写 |
| 接口 | scalar M_AXI address computation error |
| 资源 | Vivado BRAM/URAM/FIFO over-utilized |
| 版本 | app/platform/bitstream tag 与最新源码不一致 |

完整真实尺寸 C/RTL co-sim 已不再作为主验证路径，因为 AXI VIP 和真实数据规模带来的时间成本远高于收益。当前以 CSim + 上板 bit-exact + profiling 作为主验收手段。

## 13. 下一步优化路线

### P2D：当前待验证版本

当前 `P2D-4OPT-PROFV3` 已完成源码修改并通过整网 CSim，等待综合和上板：

| 修改 | 预期收益 |
|---|---|
| C19 stride2 专用 window path | 优化 level2 early conv |
| C131 stride2 专用 window path | 优化 U40 大通道降采样卷积 |
| C12/C16 compact row write | 减少 scratch 输出 RMW |
| profiling v3 phase model | 更清楚地区分 window/SA/post/writeback 贡献 |

若综合报告无 blocker，建议导出 `platform_p2d_0518` 并进行 full run。若 bit-exact 且延迟低于 468ms，则将 P2D 作为新基线。

### 后续结构性优化

| 优先级 | 方向 | 目标 |
|---|---|---|
| P3.1 | 小通道多空间点并行 | 用同一组 PE 同时计算多个 `ow`，提高 `out_c<32` 层 PE 利用率 |
| P3.2 | post/writeback 融合 | 减少 row buffer 和写回之间的额外调度开销 |
| P3.3 | dilation 3x3 专用 path | 对 `dilation=2/4/8/16` 的 ESPNet 分支做更直接的窗口访问 |
| P3.4 | tensor layout co-design | 让中间 tensor 更适合 aligned full-word write，减少 RMW |
| P3.5 | INT6/INT4 reconfigurable 扩展 | 在 INT8 稳定后，扩展可选量化模式和硬件打包方式 |

## 14. Slides 建议结构

建议组会 slides 按以下顺序组织：

| Slide | 内容 |
|---|---|
| 1 | 任务目标：ESPNet Encoder INT8 片上推理闭环 |
| 2 | PS/PL/SD/DDR 顶层系统图 |
| 3 | UOP schedule 与模型算子映射 |
| 4 | NPU 内部模块划分 |
| 5 | 片上 memory layout 与数据格式 |
| 6 | Conv row-level dataflow |
| 7 | INT8 量化与后处理路径 |
| 8 | 已实现的加速手段 |
| 9 | 正确性验证：bit-exact 结果 |
| 10 | 性能演进：817ms 到 468ms |
| 11 | Profiling 暴露的瓶颈 |
| 12 | P2D 当前工作与下一步优化路线 |

汇报时建议明确一句话结论：当前硬件已经实现真实板端整网 INT8 推理并达到 bit-exact，性能已进入可归因优化阶段；后续的关键不再是功能跑通，而是围绕 window、writeback 和小通道并行做结构性性能收敛。
