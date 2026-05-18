# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-18)

## 1. 当前进展

今天继续推进 P2 阶段性能收敛，目标是在保持 bit-exact 和 stream 主路径不变的前提下，一次性合并前四项局部优化，然后进入综合验证。

| 项目 | 当前状态 |
|---|---|
| 已验证上板基线 | `INT8-BOARD-20260515-P2B-WGTFIFO-FLWIN` |
| 基线延迟 | full `MODE_RUN = 46,893,423 cycles = 468 ms` |
| 基线正确性 | `D72OUT.BIN` 与 HLS reference bit-exact |
| 最新源码版本 | `INT8-BOARD-20260518-P2D-4OPT-PROFV3` |
| 最新 HLS CSim | 整网 `top_tb` 通过，0 errors |
| 当前阶段 | 等待 C synthesis 和 audit 报告 |

P2B 的 profiling 数据显示，当前主要瓶颈仍集中在 window generation、写回/RMW、post/output drain 和低 PE 利用率的小通道卷积上。因此今天的修改不再只加计数器，而是直接做一轮更激进的硬件路径收敛。

## 2. 今日修改

| 方向 | 修改 | 预期作用 |
|---|---|---|
| AvgPool C3 | 保留已实现的 C3 packed row fast path | 继续压缩 `U01_POOL1` 的前端开销 |
| Small-C 3x3 | 将 C12/C19/C25 pack 函数改为内联，并新增 `C19 stride2` 行段读取路径 | 减少 level2 early conv 的 window read/pack 开销 |
| Large-C 3x3 | 新增 `C131 stride2` 专用路径 | 针对 `U40` 这类大通道降采样卷积减少动态分段调度开销 |
| Writeback/RMW | 增加 C12/C16 compact row aligned write | 将部分 scratch 输出从逐像素 RMW 改为整字写回 |
| Profiling 模型 | 同步更新硬件 prof3 和离线 attribution 估计 | 便于后续解释优化收益来源 |

硬件接口和 uop 协议未改变；本轮改动集中在 `win_gen.cpp`、`memory.cpp`、`int8_core.cpp` 和性能分析脚本。

## 3. 当前判断

本轮源码已经通过整网 CSim，说明功能路径与当前 golden reference 对齐。但这些修改对综合调度有一定风险，尤其是新增专用 window path 和 compact row write 后，必须重新检查：

| 风险项 | 检查重点 |
|---|---|
| Dataflow 正确性 | 不能再出现 `HLS 214-475`、`HLS 200-975` 或 stream 读写冲突 |
| Pipeline II | 关注 `emit_window_row_3x3_*`、`store_compact_*`、`on_chip_memory_*` 的 II 是否异常放大 |
| 资源 | URAM 已满、BRAM 接近上限，不能引入新的大 buffer/FIFO |
| Timing/LUT | HLS LUT 估计长期偏高，但若明显恶化，需要先收敛再上板 |

本轮离线模型的 lower-bound 从约 `83.6 ms` 降到约 `79.7 ms`，只能说明局部结构确有收益；真实收益仍以后续综合和上板 full `MODE_RUN` 为准。

## 4. 下一步计划

1. 手动运行当前版本 C synthesis。
2. 使用 audit 脚本检查最新综合报告，确认无 dataflow/deadlock 类 blocker。
3. 若综合通过且资源风险可接受，package IP 并导出新硬件平台，建议命名为 `platform_p2d_0518`。
4. 在 Vitis 中 clean rebuild platform/app，确认 app tag 为 `INT8-BOARD-20260518-P2D-4OPT-PROFV3`。
5. 上板运行单次 full `MODE_RUN`，记录 cycles、prof/prof2/prof3，并保存 `D72OUT.BIN`。
6. 离线比对 `D72OUT.BIN` 与 HLS reference，bit-exact 后再把该版作为新的性能基线。

若 P2D 延迟能明显低于 `468 ms`，下一轮继续沿 window/writeback 路线收敛；若收益有限，则需要转向更结构性的并行化，例如 small-Cout 多空间点并行或 post/writeback 与 row dataflow 的更深融合。
