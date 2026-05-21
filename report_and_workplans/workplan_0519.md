# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-19)

## 1. 当前基线

当前可信功能基线仍是 `P2F/P2E` 稳定 stream 路径。该版本已完成 500 张验证集闭环，输出文件完整，板端精度与软件参考基本一致。

| 项目 | 结果 |
|---|---|
| full-val app tag | `INT8-BOARD-20260519-P2F-VALSET` |
| val set | `I0000.BIN` 到 `I0499.BIN`，共 500 张 |
| 平均单图延迟 | `39942512 cycles`，约 `399 ms` |
| 500 张总延迟 | `19971256489 cycles`，约 `199.7 s` |
| 板端 PA / mIoU | `0.98156198 / 0.88452899` |
| 软件参考 PA / mIoU | `0.98183763 / 0.88832441` |

2026-05-20 完成一版去除 profiling/debug 端口的 125 MHz 性能实验版：

| 项目 | 结果 |
|---|---|
| app tag | `INT8-BOARD-20260520-PERF125-NOPROF` |
| PL clock | `124.998749 MHz` |
| 单图 full `MODE_RUN` | `32987959` APU timer ticks，约 `329 ms` |
| 单图输出比对 | `D72OUT.BIN` vs CSim golden，`0 / 16384` mismatch |
| routed timing | setup 未收敛，WNS `-1.075 ns`，TNS `-6112.018 ns` |

结论：功能正确性和精度链路已经闭合。`PERF125-NOPROF` 证明去除 profiling/debug 后可完成整网并得到 bit-exact 单图结果，墙钟延迟相比 `P2F/P2E` 约 `399 ms` 降到约 `329 ms`。但该版本不是 timing-clean 验收版本，当前只能作为 125 MHz exploratory performance point。

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
3. `MODE_RUN=39.94M cycles`，而 analytical `model_cycles=7.46M cycles` 只能解释约 19% 延迟；剩余时间需要真实 per-phase cycle counter 进一步拆解。

## 3. P2G 实验与回退

P2G 的目标是保留稳定 stream 数据流，同时增加 `profiling v5`，覆盖 `IF/uop fetch/frame load-store/non-conv tiles/non-conv mem ops/conv model/total work`。

单图上板结果：

```text
FULL_MODE_RUN = 49199988 cycles = 491 ms
prof5 if_words=75 frame_ld=49152 frame_st=512 uop_fetch=75
prof5 pool_tiles=294912 affine_tiles=557056 add_tiles=344064 store_tiles=688128
prof5 nonconv_mem_ops=4702208 conv_model=7457344 total_work=12209366
```

该结果证明 app/platform/header 已对齐 P2G，因为 `prof5` 寄存器可正常读出；但性能从 `399 ms` 回退到 `491 ms`。综合报告中已确认 P2G 对 `avgpool/add/affine/concat` 强行打开的 `PIPELINE II=1` 没有达到目标 II，反而增加时序和资源压力。

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

1. 固化 `PERF125-NOPROF` 实验结果：保留 `329 ms` 单图数据和 `D72OUT.BIN` bit-exact 记录，但明确标注该版本 125 MHz timing 未收敛。
2. 优先做 125 MHz timing closure：检查 `s_bram` 级联读、`read_tile_packed_word`、window small-channel fast path 的寄存器切分、bank 选择逻辑和布局压力；目标是 routed WNS >= 0。
3. 并行保留 100 MHz/110 MHz clean build 作为安全性能对照：若 125 MHz 长期无法收敛，至少保证无 profiling 版本在 timing-clean 频点下 cycle 不差于 `P2F/P2E`。
4. timing-clean 后再继续结构优化：先恢复少量真实 cycle counter 做 residual 归因，再推进 shape-specific 非卷积路径、window 专用路径和轻量 uop fusion。
5. 暂不进入小卷积并行化/阵列重构，除非现有 stream 路径在 timing-clean 后仍无法接近阶段性目标。
