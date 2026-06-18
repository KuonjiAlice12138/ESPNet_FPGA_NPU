# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-19)

## 1. 当前基线

当前可信功能基线仍是 `P2F/P2E` 稳定 stream 路径。该版本已完成 500 张验证集闭环，输出文件完整，板端精度与软件参考基本一致。

| 项目 | 结果 |
|---|---|
| full-val app tag | `INT8-BOARD-20260519-P2F-VALSET` |
| val set | `I0000.BIN` 到 `I0499.BIN`，共 500 张 |
| 平均单图延迟 | `39,942,512` A53 timer ticks，真实约 `1198 ms` |
| 500 张总延迟 | `19,971,256,489` A53 timer ticks，真实约 `599.1 s` |
| 板端 PA / mIoU | `0.98156198 / 0.88452899` |
| 软件参考 PA / mIoU | `0.98183763 / 0.88832441` |

2026-05-20 完成一版去除 profiling/debug 端口的 125 MHz 性能实验版：

| 项目 | 结果 |
|---|---|
| app tag | `INT8-BOARD-20260520-PERF125-NOPROF` |
| PL clock | `124.998749 MHz` |
| 单图 full `MODE_RUN` | `32,987,959` A53 timer ticks，真实约 `990 ms` |
| 单图输出比对 | `D72OUT.BIN` vs CSim golden，`0 / 16384` mismatch |
| routed timing | setup 未收敛，WNS `-1.075 ns`，TNS `-6112.018 ns` |

结论：功能正确性和精度链路已经闭合。`PERF125-NOPROF` 证明去除 profiling/debug 后可完成整网并得到 bit-exact 单图结果，真实墙钟延迟相比 `P2F/P2E` 约 `1198 ms` 降到约 `990 ms`。但该版本不是 timing-clean 验收版本，当前只能作为 125 MHz exploratory performance point。

2026-05-22/24 完成一版小卷积并行化后的 100 MHz timing-clean 上板实验，对外命名为 `P3A`：

| 项目 | 结果 |
|---|---|
| app tag | `INT8-BOARD-20260522-P3A` |
| PL clock | `100 MHz` |
| routed timing | WNS `0.019 ns`，TNS `0`，setup/hold/pulse width 均无失败端点 |
| 单图 full `MODE_RUN` | `46,939,949` A53 timer ticks，真实约 `1408 ms` |
| 单图输出比对 | `D72OUT.BIN` vs CSim golden，`0 / 16384` mismatch |

结论：`P3A` 证明 pair2 小卷积并行路径可以在 100 MHz 下完成实现、上板运行并保持 bit-exact，但端到端性能慢于 `P2F/P2E` 的真实约 `1198 ms`。因此 `P3A` 暂作为“小卷积并行化功能与时序闭合检查点”，不替代当前性能基线。

## 2. Profiling 观察

P2F/P2E 关键计数器：

```text
win_read=4077946, win_words=2760704
wgt=468992, sa_steps=2760704
psum=630784, out_tiles=630784
rmw_ops=347136, rmw_reads=0
model_cycles=7457344
wgt_cyc=468992, win_cyc=6123642, sa_cyc=2760704
post_cyc=2465792, write_cyc=347136, row_region_cyc=7104512
```

判断：

1. conv datapath 内仍以 window path 为主瓶颈，`win_cyc=6.12M` 高于 `sa_cyc=2.76M`。
2. RMW 读已基本消除，继续优化 output writeback 的边际收益下降。
3. `MODE_RUN=39.94M` A53 timer ticks，而 analytical `model_cycles=7.46M` 只能解释约 19% 计数规模；剩余时间需要真实 per-phase counter 进一步拆解。

## 3. P2G 实验与回退

P2G 的目标是保留稳定 stream 数据流，同时增加 `profiling v5`，覆盖 `IF/uop fetch/frame load-store/non-conv tiles/non-conv mem ops/conv model/total work`。

单图上板结果：

```text
FULL_MODE_RUN = 49199988 A53 timer ticks = 约 1476 ms
prof5 if_words=75 frame_ld=49152 frame_st=512 uop_fetch=75
prof5 pool_tiles=294912 affine_tiles=557056 add_tiles=344064 store_tiles=688128
prof5 nonconv_mem_ops=4702208 conv_model=7457344 total_work=12209366
```

该结果证明 app/platform/header 已对齐 P2G，因为 `prof5` 寄存器可正常读出；但性能从真实约 `1198 ms` 回退到约 `1476 ms`。综合报告中已确认 P2G 对 `avgpool/add/affine/concat` 强行打开的 `PIPELINE II=1` 没有达到目标 II，反而增加时序和资源压力。

已完成回退：

1. 保留 `prof5` 计数器和 P2G 单图 app 配置。
2. 回退 `avgpool_pixel_c3_inner_fast` 的 inline 改动。
3. 回退 `avgpool/concat/affine/add` 上失败的 tile 级强制 pipeline。
4. 顶层 `top_tb` CSim 已通过，`0 errors`。

后续路线已从 `P2G-prof5` 转向 no-prof 性能验收版。`prof5` 的主要价值是证明非卷积工作量仍不可忽略，但该版本本身不再作为性能基线继续推进。

## 4. 仍可推进的性能空间

在不做“小卷积并行计算/阵列数据流重构”这类大改的前提下，仍有几类可尝试优化：

1. 先做 timing closure：125 MHz 版本最差路径集中在 `s_bram -> window_generator_row/read_tile_packed_word`，route delay 占比高，说明继续优化必须同时考虑时序和布局压力。
2. 区分 APU timer tick 与 PL cycle：当前 app 表中的 `cycles` 是 PS/APU 计时 tick，不是 NPU 125 MHz cycle。后续报告必须同时记录墙钟时间和归一化后的 PL cycle 估计，避免把频率收益误判为数据流 cycle 收敛。
3. 增加真实周期级 profiling：当前 `prof5` 是工作量计数，不是周期计数。下一版 profiling 应记录 `frame load/store`、`uop dispatch`、`pool/add/affine/store`、`conv row-region` 的真实 cycle，优先解释约 `32M cycles` 残差。
4. 做 shape-specific 非卷积路径：不要再对 generic loop 盲目 `II=1`，而是针对固定网络中常见的 `ADD/AFFINE/STORE/POOL` shape 写专用 word-aligned 路径，减少动态分支和无效 tile 遍历。
5. 继续收敛 window path：优先针对现有网络的固定 `kernel/stride/channel` 组合扩展专用 window generator，而不是扩大 FIFO 或复制 buffer。验收指标必须是 `win_read/win_cyc/row_region_cyc` 下降，同时不能恶化 125 MHz 时序。
6. UOP/schedule 层轻量融合：对相邻 `AFFINE+ADD`、`STORE/CONCAT` 等固定模式考虑生成 fused uop 或专用执行分支，减少片上 memory 往返；这属于中等改动，但不需要重写 conv stream 数据流。

## 5. 下一步计划

1. 固化 `P3A` 实验结果：保留 100 MHz timing-clean、bit-exact 和真实约 `1408 ms` 单图数据，明确其功能价值高于性能价值。
2. 不继续盲目扩大 pair2 命中范围；下一轮先审查 `window_generator_row_dual`、`post_process_row_dual` 和 row-region 控制逻辑带来的额外开销，解释相对 `P2F/P2E` 的约 `210 ms` 真实回退。
3. 若 dual path 开销主要来自动态分支和统一 wrapper，可尝试收窄 pair2 启用 shape、拆出更轻的专用小卷积路径，或回退大部分层到 `P2F/P2E` stable stream。
4. 若 pair2 无法带来端到端收益，则停止该方向，把优化重心转回 `P2F/P2E` 路线上的 window path、非卷积专用路径和轻量 uop fusion。
5. 任何后续性能版本必须同时满足：`top_tb` 整网 CSim 通过、单图 `D72OUT.BIN` bit-exact、100 MHz routed timing clean，且端到端延迟不差于 P2F/P2E 的真实约 `1198 ms` 基线。

## 6. P3A/P3B/P3C/P3D 小卷积并行化实验

P3A 目标是利用 `out_c <= 16` 小卷积在 `TM=32` 阵列上的空闲 lane，将同一行相邻两个输出像素并行映射到低/高 16 个输出通道 lane。实现上采用双 activation stream、共享 weight stream、双 psum stream；不复制完整 `32x32` MAC 阵列，避免资源翻倍。

当前启用边界：

1. `out_c <= 16` 且 `out_w` 为偶数。
2. `kernel=1` 小卷积。
3. `kernel=3, in_c=3, stride=2, dilation=1` 的第一层卷积。
4. `kernel=3, in_c=19, stride=2, dilation=1` 的 level2 入口卷积。
5. `kernel=3, in_c=12, stride=1` 的 ESPNet 小通道卷积。
6. 不满足条件的卷积自动回退到 `PERF125-NOPROF` 原 stream row path。

已完成：

1. 新增 `ENABLE_SMALLCONV_PAIR2` 编译期/配置开关，必要时可一键回退。
2. `win_gen` 增加 pair2 window 输出路径；`sa_core` 增加 split-lane pair2 row core；顶层 conv row region 增加受控分支。
3. P3B 已把 pair2 命中范围扩展到 U02/U07/U08/U09/U11/U14/U17/U21/U22/U23/U25/U28/U31/U72；`out_c=25/28` 的 level3 卷积无法在 `TM=32` 中容纳两个像素的完整输出通道，暂不进入 pair2。
4. `top_tb` 整网 CSim 已通过，`0 errors`。

下一步验收：

1. 先跑 HLS C synthesis，重点检查是否出现 `HLS 214-475/200-975`、DATAFLOW 合并、FIFO 深度异常、II 退化和 DSP/LUT/BRAM 激增。
2. 若综合报告存在资源或 II 明显退化，优先收窄启用 shape 或关闭 `ENABLE_SMALLCONV_PAIR2`，不得继续推进到实现/上板。
3. 若综合报告健康，再进行实现和上板单图测试；验收下界是不慢于 `PERF125-NOPROF` 的真实约 `990 ms`，并保持 `D72OUT.BIN` bit-exact。

P3B 综合观察：功能 CSim 正确，但 HLS 将 scalar row-region 和 pair2 row-region 作为两套独立 DATAFLOW 硬件同时综合，导致 `systolic_array_core_row` 与 `systolic_array_core_row_pair2` 资源复制。报告中 top 估计 `BRAM_18K=1730/1488`、`LUT=701154/341280`、`Estimated Fmax=72.55 MHz`，不满足继续实现/上板条件。

P3C 修补：已合并为单一 row-region 数据流，当前只保留 `execute_conv_stream_row_region -> window_generator_row_dual -> systolic_array_core_row_dual -> post_process_row_dual`。旧 `execute_conv_stream_row_region_pair2`、旧 row-only/pair2-only SA 与 post-process 函数已移除，pair2 仅作为统一 SA 内部运行模式。`top_tb` 整网 CSim 已通过，`0 errors`。P3C C synthesis 不再出现独立 pair2 row-region/SA，但 `window_generator_row_dual` 内部仍同时综合 scalar/pair2 window wrapper，top 估计 `BRAM_18K=1629/1488`、`LUT=636080/341280`，仍不可推进实现。

P3D 修补：继续合并 window 生成逻辑，`window_generator_row_dual` 不再调用旧 `window_generator_row` / `window_generator_row_pair2` wrapper，改为直接调度 `first_layer/c19_stride2/smallc/1x1` 四类 dual emitter；旧 pair2 window wrapper 和旧 scalar row wrapper 已清理。`top_tb` 整网 CSim 已通过，`0 errors`。下一步由 GUI 重新跑 C synthesis，重点确认报告中不再出现 `window_generator_row` 与 `window_generator_row_pair2` 两套子模块，并检查 `window_generator_row_dual` 的 LUT/BRAM 是否明显回落。

P3A 上板验收：在上述代码收敛后，最终对外命名为 `P3A` 并完成 100 MHz 降频实现。Vivado routed timing clean，`clk_pl_0=10 ns`，WNS `0.019 ns`，methodology checks `0`。上板单图 `FULL_MODE_RUN=46939949` A53 timer ticks，真实约 `1408 ms`；`D72OUT.BIN` 与 CSim golden 完全一致，`0 / 16384` mismatch。该结果说明 pair2 路径功能可用且可实现，但当前 dual path 额外调度/选择/window/post 开销抵消了小卷积并行收益，性能仍慢于 `P2F/P2E`。

P3E 修补方向：回到 `PERF125-NOPROF` 的单路 stream 性能路径，移除 top 路径中的 `pair2/dual` window、SA 和 post-process 逻辑，只保留 `window_generator_row -> systolic_array_core_row -> post_process_row_to_buffer`。本轮不再推进小卷积并行化，目标是先恢复不差于真实约 `990 ms` 的 125 MHz no-prof 基线，再基于 timing-clean 报告继续做局部 window/non-conv 优化。当前 `top_tb` 整网 CSim 已通过，`0 errors`；下一步由 GUI 跑 8 ns C synthesis，并重点审查 II、FIFO/BRAM、LUT 和 estimated Fmax 是否回到 `PERF125-NOPROF` 量级。

P4B/P4C/P4D 当前进展：基于已测试过的单路 stream 基线继续做保守性能收敛，不恢复 pair2/dual path。P4B 已把 `AFFINE` 量化参数读取移到通道块外层，避免逐像素重复取 qparam；`ADD` 改为通道块优先遍历；`STORE/concat` 增加 32-byte 对齐整字复制路径，仅在源/目标物理通道跨度和目标 offset 均对齐时启用。P4C 保持单路 window fast dispatch，覆盖 first-layer、`c19 stride2`、small-channel stride1 reuse、`c131 stride2` 和 aligned `1x1` 路径，并确认源码中已无 `pair2/dual` window/SA/post 残留。P4D 已加入 `ADD+STORE` 轻量融合：ADD 结果同时写 scratch 和最终 concat slice，然后跳过紧随其后的 STORE，减少一次 scratch 读，同时保留残差链后续 ADD 所需的 scratch 数据。

验证：`top_tb` 整网 CSim 已通过，`0 errors`。下一步由 GUI 跑 8 ns C synthesis，重点检查 `HLS 214-475/200-975`、DATAFLOW 合并、FIFO/BRAM 异常、关键循环 II、LUT/BRAM 估计和 estimated Fmax；若报告健康，再推进实现和上板，性能目标是不差于 `PERF125-NOPROF` 的真实约 `990 ms` 探索基线，并保持 `D72OUT.BIN` bit-exact。
