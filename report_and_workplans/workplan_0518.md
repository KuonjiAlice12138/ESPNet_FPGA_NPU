# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-18)

## 1. 当前进展

今天完成 P2D 和 P2E 两版硬件的综合、平台导出和上板 full `MODE_RUN` 测试。P2D 证明 compact write/RMW 缩减方向有效；P2E 在此基础上继续推进 window read 和写回路径收敛，并加入/使用 profiling v3 计数器观察阶段开销。

| 项目 | 结果 |
|---|---|
| 当前最新版本 | `INT8-BOARD-20260518-P2E-WINPACK-PROFV3` |
| full `MODE_RUN` | `39,941,854` A53 timer ticks，真实约 `1198 ms` |
| 相比 P2D | 减少 `794,139` ticks，约 `23.8 ms / 1.95%` |
| 相比 P2B | 减少 `6,951,569` ticks，约 `208.5 ms / 14.8%` |
| 相比 PROF01 | 约 `51.2%` 总体降幅，真实延迟从约 `2454 ms` 降到约 `1198 ms` |
| 输出正确性 | `D72OUT.BIN` 与 HLS reference bit-exact，`0 / 16384` mismatch |

P2E 可以作为当前新的上板性能基线。相比 P2D，本轮收益较小，但它把剩余 RMW read 完全清零，说明输出写回路径已经基本收敛到当前数据布局下的较优状态。旧文档中的 `399 ms` 是错误地按 `100MHz` 换算 A53 timer tick 得到的旧口径。

## 2. Profiling 对比

| 计数项 | P2D | P2E | 变化 |
|---|---:|---:|---:|
| `win_read` | `4,419,328` | `4,077,946` | 减少 `341,382` |
| `win_words` | `2,760,704` | `2,760,704` | 无变化 |
| `wgt` | `468,992` | `468,992` | 无变化 |
| `sa_steps` | `2,760,704` | `2,760,704` | 无变化 |
| `psum` | `630,784` | `630,784` | 无变化 |
| `out_tiles` | `630,784` | `630,784` | 无变化 |
| `rmw_ops` | `860,160` | `347,136` | 减少 `513,024` |
| `rmw_reads` | `352,256` | `0` | 清零 |
| `direct_words` | `155,648` | `347,136` | 增加 `191,488` |
| `model_cycles` | `8,044,096` | `7,457,344` | 减少 `586,752` |

P2E 的主要结构性收益仍来自写回路径：`rmw_reads=0`，`direct_words` 增加到 `347,136`。Window read 也有下降，但 `win_words` 和 `sa_steps` 不变，说明核心计算映射和 k-tile 数没有改变。

## 3. Profiling v3 观察

| 阶段计数 | cycles | 占实测比例 |
|---|---:|---:|
| `wgt_cyc` | `468,992` | `1.17%` |
| `win_cyc` | `6,123,642` | `15.33%` |
| `sa_cyc` | `2,760,704` | `6.91%` |
| `post_cyc` | `2,465,792` | `6.17%` |
| `write_cyc` | `347,136` | `0.87%` |
| `row_region_cyc` | `7,104,512` | `17.79%` |

P2E 的 `model_cycles = 7.46M`，但板端计时为 `39.94M` A53 timer ticks，仍有约 `32.48M` tick residual。也就是说，当前显式计数器解释了方向性收益，但还不能解释大部分真实延迟。后续不能再只围绕单个算子局部循环盲改，必须把 row-level 调度、memory back-pressure 和函数边界控制开销纳入模型。

## 4. 当前判断

| 方向 | 判断 |
|---|---|
| Output/RMW | 当前基本收敛，继续优化空间有限 |
| Window generation | `win_read` 有下降但仍是显著阶段开销，需要继续做 row/window 级 profiling |
| SA 阵列 | `sa_steps` 不变，短期不是主要收益点 |
| Weight path | `wgt_cyc` 很低，不是瓶颈 |
| Residual cycles | 最大未解释项，下一轮必须优先定位 |
| 阵列形状 | 暂不建议改，单改阵列无法解决 `32M+` residual cycles |

## 5. 下一步计划

1. 保留 P2E 作为当前 bit-exact 上板基线，记录 `39,941,854` A53 timer ticks，真实约 `1198 ms`。
2. 做 profiling v4：增加每层/每 row 的实际 cycle 计数，而不仅是全网累计阶段估计。
3. 在 HLS 中定位 row-region 外的额外等待来源，重点检查 `execute_conv_stream_datapath` 外层循环、store 后同步、uop 调度和 on-chip memory 访问仲裁。
4. 继续优化 window path，但必须以 `win_read/win_cyc/row_region_cyc` 的上板下降作为验收标准。
5. 暂停大规模阵列形状调整；如后续评估 `64x16`，必须同步设计 activation-window replay/broadcast，否则会放大 window 生成压力。
