# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-15)

## 1. 当前进展

今天的工作重点从 profiling 版硬件转向 P2 阶段性能收敛。当前验证链仍保持为：每轮 HLS 修改后先跑整网 CSim，再综合、audit 报告，最后进入硬件平台实现和上板测试。

| 项目 | 当前状态 |
|---|---|
| HLS CSim | 最新代码整网通过，`top_tb` 0 errors |
| HLS synthesis | 最新版通过，audit 无 blocker/high-risk warning |
| 已上板版本 | `INT8-BOARD-20260515-P2A-PROFV2` |
| 最新待上板版本 | HLS 已综合通过，硬件平台正在 Vivado 实现中 |
| 下一步 | 等待实现完成后导出 platform，build app，上板验证延迟和 `D72OUT.BIN` |

已上板的 `P2A-PROFV2` 单次整网 `MODE_RUN` 结果：

| 指标 | 数值 |
|---|---:|
| full cycles | `51,120,885` |
| full latency @100MHz | `511 ms` |
| window read ops | `5,204,224` |
| window words | `2,760,704` |
| weight words | `468,992` |
| psum words | `9,601,024` |
| RMW ops | `1,212,416` |
| model cycles | `14,048,832` |

相比 `PROFWG1` 的 `526 ms`，P2A 实测降低到 `511 ms`，说明当前优化方向有收益，但增幅有限，后续必须继续针对 window generation、post/output drain 和写回路径做结构性收敛。

## 2. 今日 HLS 修改

今天完成的主要 HLS 修改如下：

| 方向 | 修改 | 当前结论 |
|---|---|---|
| `wgt_stream` FIFO | depth 从 `2305` 收窄到 `1200` | 已复核实际最大 `wgt_count=1184`，当前模型安全；综合中 `wgt_stream_U` 降到 `29 BRAM` |
| 第一层 3x3 window | 改为连续 9 字节行段读取，并拆出边界路径 | 主循环 II 从 `99` 降到 `33`，max latency 从约 `101k` 降到约 `34k cycles` |
| profiling 模型 | 同步更新第一层 window read 估计 | 模型可继续用于趋势判断，但与真实上板延迟仍存在较大残差 |

最新综合报告时间戳为 `2026-05-15 11:29`。关键资源和风险：

| 项目 | 最新综合估计 |
|---|---:|
| BRAM | `1452 / 1488 = 97%` |
| URAM | `112 / 112 = 100%` |
| DSP | `1314 / 3528 = 37%` |
| LUT | `481319 / 341280 = 141%` |

综合报告中未出现 dataflow 合并、stream 读写冲突或死锁类 warning。资源仍然偏紧，尤其 URAM 已满、BRAM 接近上限，后续不能再通过大 FIFO 或重复 buffer 换性能。

## 3. 当前判断

当前最新版比 P2A 更适合进入实现和上板测试，因为它保留了 `wgt_stream` BRAM 收窄，同时显著改善了第一层 window row 的调度。但是从综合报告看，整网主瓶颈已经不再是第一层，而是 `window_generator_row` 中的大通道 `emit_window_row_3x3_fast`、small-C 路径以及后处理/写回阶段。

因此，当前正在实现的硬件平台应作为一次“验证 P2B/P2C 改动真实收益”的中间版本，而不是最终性能版本。

## 4. 下一步计划

1. 等待当前 Vivado 实现完成，导出新的 hardware platform。
2. 在 Vitis 中重新 build platform 和 app，确认 app 指向最新 platform，避免缓存引用旧 BSP。
3. 上板运行单次 full `MODE_RUN`，记录 cycles、profiling/prof2 counters，并保存 `D72OUT.BIN`。
4. 离线比对 `D72OUT.BIN` 与 HLS reference，要求 bit-exact 后再记录性能数据。
5. 若最新版延迟明显低于 `511 ms`，继续沿当前路径优化；若收益很小，下一轮重点转向 `emit_window_row_3x3_fast` 和 post/writeback 的周期级瓶颈。

短期目标仍是先稳定压到 `300-400 ms` 区间并保持 bit-exact；50ms 目标需要后续更激进的并行化或任务级重构，不能只依赖当前局部优化。
