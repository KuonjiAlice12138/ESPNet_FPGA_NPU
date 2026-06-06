# P6 Static-Fused INT8 NPU 设计工作计划 (2026-06-03)

## 1. 目标与硬约束

下一版不再沿当前 `full100` 的 UOP 顺序解释架构继续做局部 patch，而是切到固定 ESPNet Encoder graph 的 static-fused NPU。目标是在保持 `512x1024` 输入分辨率和 full-resolution mask 输出的前提下，以 `100 MHz` 作为实现频率，追求 `100 ms` 级端到端 `MODE_RUN` 延迟。

硬目标：

| 项目 | 目标 |
|---|---|
| 输入分辨率 | 固定 `512x1024x3` |
| 输出 | 固定 `512x1024` uint8 二分类 mask |
| PL 频率 | `100 MHz` timing-clean，先不追 125/150 MHz |
| 目标延迟 | `<= 10M PL cycles`，即 `<= 100 ms @100MHz` |
| 第一验收线 | `<= 15M PL cycles`，即 `<= 150 ms @100MHz` |
| 精度 | full100 baseline 附近，PA/mIoU 不出现显著下降 |
| 正确性 | CSim 通过；板端 mask 可评估；不要求与旧 UOP 硬件 bit-exact |

当前 full100 的可信基线：

| 项目 | 数据 |
|---|---:|
| 平均 `MODE_RUN` | `41,788,175` A53 timer ticks |
| 真实 wall-time | 约 `1.254 s` |
| 等效 100 MHz PL cycles | 约 `125.37M cycles` |
| PA / mIoU | `0.97800421 / 0.86356491` |

因此 P6 需要约 `8x-12.5x` 的总周期下降。这个目标不能靠 HLS pragma 微调实现，必须改数据流和调度粒度。

## 2. 总体架构冻结

P6 固定采用如下设计：

```text
PS / DDR / SD
  -> frame_dma_load
  -> static_espnet_scheduler
       -> stem stage
       -> level2 fused blocks
       -> level3 fused blocks
       -> classifier
  -> fullres upsample + argmax
  -> frame_dma_store
```

保留内容：

| 保留项 | 说明 |
|---|---|
| 顶层 IP 名称和 AXI 接口 | 继续使用 `espnet_encoder_int8_core`，避免重新搭系统平台 |
| `PARAM.BIN` 权重/qparam 格式 | `MODE_INIT` 仍解析并加载参数，尽量减少导出器重写 |
| INT8 量化语义 | 保持当前硬件约束 QAT / export 量化行为 |
| full-resolution upsample 输出 | 继续由 PL 输出 `512x1024` mask |
| full100 baseline | 作为功能、精度和延迟对照 |

废弃内容：

| 废弃项 | 原因 |
|---|---|
| `MODE_RUN` 逐 UOP 动态解释执行 | 控制和 memory 往返开销过大，无法到 100ms |
| 每个 UOP 都写回全局 feature memory | 中间 tensor 往返是主要延迟来源 |
| pair2/dual wrapper 架构 | 曾上板验证但端到端退化，不能继续叠在旧数据流上 |
| 为追 125/150MHz 做的周期退化型补丁 | 当前目标先回到 100MHz，优先压总 cycles |
| profiling/debug 端口作为主线依赖 | 后续 profiling 可单独开版本，不进入性能验收版 |

## 3. 模块划分

P6 允许新增 HLS 源文件，因为这是架构级重构。模块边界固定如下：

| 模块文件 | 功能 |
|---|---|
| `int8_core.cpp` | 顶层接口、`MODE_INIT/MODE_RUN` 分发、调用 static scheduler |
| `param_dma.cpp` | 保持参数加载和 qparam/weight 查询接口 |
| `frame_dma.cpp` | 输入 frame load；输出 full-resolution mask store |
| `memory.cpp` | 仅保留必要的 block boundary feature memory 和 ping-pong buffer 访问 |
| `static_sched.cpp` | 固定 ESPNet Encoder stage 调度，不再逐 UOP 解释 |
| `linebuf_win.cpp` | streaming line buffer 和 dilation-aware window 生成 |
| `sa_core.cpp` | 改为原生 `2x16x32` spatial-SIMD MAC 阵列 |
| `block_l2.cpp` | level2 fused ESP block |
| `block_l3.cpp` | level3 fused ESP block |
| `nonconv_fused.cpp` | fused affine/add/concat/store 辅助路径 |
| `upsample_unit.cpp` | full-resolution bilinear upsample + argmax |

若为了控制风险，第一版可以先把 `static_sched/block_l2/block_l3/nonconv_fused` 写在 `int8_core.cpp` 中，但文档和函数边界必须按上表组织，后续再拆文件。

## 4. Static Graph Scheduler

P6 的 `MODE_RUN` 不再执行：

```text
for uop in uop_table:
    decode opcode
    execute one operator
    write global tensor
```

改为固定 stage 调度：

```text
MODE_RUN:
  load input frame
  run_stem()
  run_level2()
  run_level3()
  run_classifier()
  upsample_logits_bilinear_argmax_store()
```

每个 stage 内部按固定 ESPNet graph 调用参数 ID。`PARAM.BIN` 仍提供权重和 qparam，但 UOP table 只作为兼容/校验，不作为性能路径的主循环。

调度规则：

1. 每个 stage 输入输出只在 stage 边界写入 feature memory。
2. block 内的 `Conv -> Affine/Requant -> Add -> Concat` 尽量在 stream/line buffer 中完成。
3. 只有跨 block 需要复用的 feature 才落到全局片上 memory。
4. `STORE/CONCAT` 不再作为独立 memory pass；在 producer 生成 row 时直接写目标 channel slice。
5. `ADD` 不再独立扫描整张 tensor；在 branch 输出可用时完成 residual accumulation。

## 5. Block-Level Fusion

P6 最大改动是 ESP block 内融合。以 level2 为例，当前旧路径近似为：

```text
input tensor
  -> branch conv d=1 -> scratch
  -> branch conv d=2 -> scratch
  -> branch conv d=4 -> scratch
  -> branch conv d=8 -> scratch
  -> branch conv d=16 -> scratch
  -> store/concat
  -> add/affine
  -> output tensor
```

P6 固定改为：

```text
input stripe
  -> shared line buffer
  -> five dilation branch windows
  -> shared/native SA execution
  -> branch requant
  -> concat/add fusion
  -> write final block output once
```

实现原则：

| 项目 | 设计 |
|---|---|
| 数据粒度 | row stripe，优先 1-2 输出行粒度 |
| branch 输入 | 共享 line buffer，不重复从全局 memory 读同一 input tensor |
| branch 输出 | 优先写局部 row buffer，不写 scratch tensor |
| concat | producer 直接写目标 channel slice |
| add | branch 输出与 residual 在 row buffer 内完成 |
| block 输出 | 只在 block 末尾写一次 feature memory |

该改动的目标是把 `STORE/CONCAT=23`、`ADD=14`、`AFFINE=7` 中的大部分 memory pass 融入 block 内。

## 6. Line-Buffer Window Generator

旧 `win_gen` 的问题是围绕全局 feature memory 做 window read/pack。P6 固定采用 streaming line buffer：

```text
feature row stream
  -> line buffer / ring buffer
  -> dilation taps
  -> 3x3 or 1x1 window stream
  -> SA
```

设计要求：

| 项目 | 要求 |
|---|---|
| 第一层 | 输入图像 C3 3x3 stride2 走专用 line buffer |
| level2 | `128x256` 小通道 dilation branches 共享输入 line buffer |
| level3 | `64x128` 小通道 dilation branches 共享输入 line buffer |
| dilation | 固定支持 `1/2/4/8/16`，不要保留复杂动态 generic path |
| 1x1 | 走直接 channel stream，不进入 3x3 line buffer |
| 边界 | 边界补零在 line buffer tap 阶段处理 |

不要再写复杂动态 `kernel/stride/dilation/channel` 通用窗口生成器。P6 是固定 ESPNet graph 专用硬件，允许 shape-specific 实现。

## 7. 原生 `2x16x32` SA

当前 `TM=32/TK=32` 对小通道卷积利用率低。P6 固定将 SA 改为原生 spatial-SIMD：

```text
2 pixels x 16 output channels x 32 K lanes
```

映射规则：

| 卷积形状 | 映射 |
|---|---|
| `out_c <= 16` | 同时计算两个相邻输出像素，每个像素占 16 output lanes |
| `16 < out_c <= 32` | 两个 16-lane group 合并计算一个像素 |
| `out_c > 32` | 按 output-channel tile 顺序执行 |
| `K > 32` | 按 K tile 顺序累加 |

理论依据：

| 阵列 | 理想 SA steps | 加权填充率 |
|---|---:|---:|
| 旧 `32x32` | `2.76M` | `47.6%` |
| 原生 `2x16x32` | `1.88M` | `70.1%` |

注意：这不是恢复旧 pair2 wrapper。P6 只能有一套 SA 数据流，不能同时综合 scalar 和 pair2 两套 window/SA/post。

## 8. Memory 与 Buffer 方案

P6 不复制大规模全局 feature memory。全局 memory 只用于 stage/block 边界。

固定策略：

1. 保留必要的 URAM/BRAM feature banks，作为 `stem -> level2 -> level3 -> classifier` 的边界存储。
2. Block 内使用局部 row buffer、branch buffer 和 line buffer。
3. Row buffer 尽量用 BRAM/LUTRAM，不使用 URAM。
4. 不新增完整 feature-map 级别的 branch scratch。
5. `512x1024` 输入只在 front/stem 读取；后续不再反复访问 DDR。
6. Full-resolution mask 只写 DDR，不回读。

资源上限：

| 资源 | 约束 |
|---|---|
| URAM | 不超过 full100 当前 `100%`，不能新增 URAM 需求 |
| BRAM | 尽量保持在 `<= 90%` routed 可实现区间 |
| LUT | 不超过 full100 + 新 scheduler 的合理增量，避免 pair2 时的 LUT 爆炸 |
| DSP | 当前 DSP 余量较大，可接受增加小通道 engine，但第一版只做单套 `2x16x32` |

## 9. Full-Resolution 输出

P6 继续保留 PL 侧 upsample：

```text
64x128x2 logits
  -> bilinear upsample
  -> argmax
  -> 512x1024 mask
  -> M_AXI write
```

实现要求：

1. 不把 upsample 放回 PS。
2. 不保留 full-resolution logits。
3. 输出仍为 `512*1024` bytes。
4. 若 cycles 超预算，允许把 upsample 改成多像素并行写出，但不能牺牲输出格式。

## 10. 性能预算

P6 的 100MHz 目标预算如下：

| 部分 | 目标 cycles |
|---|---:|
| Stem + frame load | `<= 1.5M` |
| Level2 fused blocks | `<= 4.0M` |
| Level3 fused blocks | `<= 3.0M` |
| Classifier | `<= 0.5M` |
| Full-resolution upsample/store | `<= 1.0M` |
| Control/stall margin | `<= 1.0M` |
| 总计 | `<= 10M-12M` |

第一版若无法直接到 `10M`，至少要证明结构性下降：

| 阶段 | 验收线 |
|---|---:|
| P6A static scheduler | 不慢于 full100；输出正确 |
| P6B line buffer + stem/level2 | 总 cycles 降到 `<= 60M` |
| P6C block fusion | 总 cycles 降到 `<= 30M` |
| P6D native `2x16x32` SA | 总 cycles 降到 `<= 15M` |
| P6E final tuning | 冲 `<= 10M-12M` |

## 11. 实现顺序

### P6A: Static Scheduler Skeleton

目标：替换 UOP 解释循环，但暂时允许内部调用旧 operator 实现，先把固定 graph 调度跑通。

交付：

1. `static_espnet_scheduler()`。
2. 固定 param-id/tensor-id/stage 调用表。
3. CSim 输出与 full100 参考一致或可解释。

### P6B: Line Buffer Front + Level2

目标：优先重写高分辨率阶段，减少最大的 memory/window 开销。

交付：

1. 第一层 C3 line-buffer path。
2. level2 block 输入 line buffer。
3. dilation `1/2/4/8/16` taps。
4. level2 输出仍可先落 memory。

### P6C: Level2/Level3 Block Fusion

目标：取消 block 内 branch scratch、STORE/CONCAT 独立 pass、ADD 独立 pass。

交付：

1. level2 fused block。
2. level3 fused block。
3. row buffer concat/add fusion。
4. stage boundary 写回。

### P6D: Native `2x16x32` SA

目标：提高小通道卷积阵列利用率，不能复制两套 SA。

交付：

1. 单套 `2x16x32` SA。
2. `out_c<=16` 双像素并行。
3. `out_c>16` 合并 lane 模式。
4. 全网 CSim 正确。

### P6E: Final Timing/Throughput Tuning

目标：在 100MHz 下进入 `100ms` 级。

交付：

1. 100MHz C synthesis 健康。
2. Vivado routed timing clean。
3. 上板单图 `MODE_RUN <= 10M-12M PL cycles` 为目标，`<=15M` 为最低有意义成果。
4. full-resolution mask PA/mIoU 不显著退化。

## 12. 验证标准

每个 P6 子版本必须按以下顺序验收：

1. Full top CSim 通过。
2. 输出 mask 与软件/旧 full100 baseline 对比，至少 PA/mIoU 可接受。
3. HLS synthesis 无 dataflow blocker：
   - 无 `HLS 214-475`
   - 无 `HLS 200-975`
   - 无 scalar M_AXI 地址错误
   - 无异常 FIFO/BRAM/LUT 爆炸
4. Vivado implementation 至少 100MHz timing clean。
5. App 使用 `CNTFRQ_EL0` 自检后的真实时间口径。
6. 上板保存 `MASK.BIN` 或验证集 `Oxxxx.BIN`，主机离线评估。

## 13. 风险与处理

| 风险 | 处理 |
|---|---|
| Static graph 与导出器不一致 | 保留 UOP table 校验；param-id/tensor-id 固定表从 artifact 自动生成 |
| Block fusion 引入数值差异 | 允许不 bit-exact，但必须 PA/mIoU 接近 full100；关键中间节点用 CSim dump 调试 |
| Line buffer 边界错误 | 先单层/单 block CSim，再整网 CSim |
| `2x16x32` SA 资源或时序恶化 | 只保留单套 SA；不恢复 pair2 wrapper；必要时先单独综合 SA |
| URAM 无余量 | 不新增 full-map scratch；局部 buffer 用 BRAM/LUTRAM |
| HLS 综合时间爆炸 | 禁止复杂动态 generic path；所有 path shape-specific；每个子模块先单独 CSim/synthesis |

## 14. 当前工作入口

下一步从 P6A 开始，不直接改现有 full100 性能路径：

1. 建立 `static_espnet_scheduler()` 和固定 stage 表。
2. 保留当前 UOP 路径作为 fallback/reference，但默认 `MODE_RUN` 切到 static scheduler。
3. 先不动 SA 和 line buffer，确认 static graph 调度输出正确。
4. P6A 通过后再进入 P6B 的 line-buffer 重写。

一句话路线：**先把调度粒度从 UOP 提升到 ESPNet stage/block，再把 block 内数据流改成 line-buffer + fused row stream，最后用原生 `2x16x32` SA 提升小通道利用率。**

