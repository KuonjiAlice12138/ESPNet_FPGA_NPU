# Workplan 0618 - P6J 延时归因与 P7 性能收敛计划

## 当前基线

当前冻结基线为 `platform_full100_0615 / P6J-TC100`。该版本已完成 100MHz timing-clean 实现、单图上板和 500 张验证集 full-resolution mask 评估。

| 项目 | 数据 |
|---|---:|
| 单图 `MODE_RUN` | `33,434,687` A53 timer ticks |
| A53 timer | `33,333,000 Hz` |
| 真实延迟 | 约 `1003 ms` |
| 等效 100MHz PL cycles | 约 `100.3M cycles` |
| 输出分辨率 | `512x1024` uint8 mask |
| 验证集 PA / mIoU | `0.97800421 / 0.86356491` |

结论：P6J 相比 `fullres100_0527` 约 `1254 ms` 有约 `20%` 下降，但距离 `100 ms` 级目标仍约 `10x`。后续不能继续依赖 pragma 微调，必须减少全网 memory pass 和中间张量往返。

## 延时归因结论

综合/实现报告和代码共同指向：**当前主要时间代价不是 MAC 阵列乘加本身，而是静态图顺序执行下的全局 feature memory 扫描、写回和调度同步。**

证据：

- P6J 端到端约 `100.3M` PL cycles。
- `sa_utilization.json` 中卷积计算下限 `sa_compute_cycles_lower_bound=23.4M`，只占端到端约 `23%`。
- 旧 profiling 口径下：`win_cyc=6.12M`、`sa_cyc=2.76M`、`row_region_cyc=7.10M`。WinGen 是卷积核心内瓶颈，但仍不足以解释 1s 级端到端延时。
- `frame_dma_load` 只有约 `49K cycles`，full-resolution upsample 估计约 `1M cycles` 量级，均不是主因。
- HLS 报告显示 `run_p6_store_op / run_p6_affine_op / run_p6_add_op / run_p6_pool_op` 都是大范围非流水三重扫描结构。其 max latency 是保守上限，不等于真实每次运行，但足以说明这些路径结构上昂贵。
- Vivado post-route 关键路径主要落在 `wgt_stream -> weight_buf`、`row_buf -> fmbuf URAM/BRAM`、`store/add/affine/pool` 相关写回路径，route 占比常在 `83%-88%`。这说明物理瓶颈同样是 memory/writeback 网络，而不是 DSP/MAC 链。

静态 UOP 规模：

| 类型 | 数量 | 主要问题 |
|---|---:|---|
| `CONV` | 26 | WinGen 和 row-region 同步重，但不是全部瓶颈 |
| `STORE` | 23 | concat / slice copy 造成独立 full-tensor pass |
| `ADD` | 14 | residual add 单独读写中间 tensor |
| `AFFINE` | 7 | requant/affine 后处理仍单独扫描 |
| `POOL` | 3 | 初始高分辨率 C3 pooling 扫描量大 |

因此，后续主线应从“让 MAC 阵列更忙”转为“减少 memory pass、减少中间张量落地、减少 block 内重复读”。

## P7 优化计划

### P7D：先补轻量 cycle counter

P6J 已移除 profiling，当前只能用旧 profiler 和 HLS 上限推断。下一版性能探索应先加入可开关轻量计数器，不进入最终 no-prof bitstream：

- `conv_cycles`
- `pool_cycles`
- `add_cycles`
- `affine_cycles`
- `store_cycles`
- `upsample_cycles`
- `scheduler_cycles`

目标是把 `100.3M` cycles 按 UOP 类型精确拆开，避免继续盲改。

### P7B：block-level fusion 与 shared line-buffer WinGen

这是最重要方向。将 level2/level3 ESP block 从“多个 UOP 顺序执行、每步读写全局 tensor”改成 block 内局部流：

```text
block input
  -> shared line buffer / row cache
  -> d=1/2/4/8/16 window lanes
  -> SA
  -> local concat / residual add / affine
  -> block output writeback once
```

目标：

- 多 dilation 分支共享 input row cache，减少重复 window read。
- concat/store 不再作为独立 UOP 扫描整张 tensor。
- residual add 和 affine 尽量融合到 block 输出写回前。
- level2 先做一个 block 原型，通过后再扩展 level2 全部 block，再扩展 level3。

### P7A：小通道 two-pixel mode 作为配套优化

`out_c <= 16` 的层可以把两个相邻输出像素映射到 `TM=32`：

```text
lane 0..15   -> pixel0 output channels
lane 16..31  -> pixel1 output channels
```

可覆盖 `U02,U07,U08,U09,U11,U14,U17,U21,U22,U23,U25,U28,U31,U72`，理论可使全网 `sa_steps` 下降约 `32%`。但单独折算到 P6J 端到端只有约 `1%` 收益，因此必须与 P7B 的 line-buffer/block fusion 配套推进。

### P7C：classifier 专用路径

`U72 C256 -> C2 1x1` 在当前 `TM=32` 阵列上填充率只有 `6.25%`。后续可增加轻量 classifier engine，对多个空间像素并行计算 2 个输出通道，并直接进入 full-resolution upsample/argmax。

## 执行顺序

1. 冻结 P6J 作为功能/精度 demo 基线。
2. 建立 P7 profiling 分支，加入 per-UOP-type cycle counters，上板确认真实延时构成。
3. 实现 P7B level2 单 block 原型：shared line buffer + branch window lanes + concat/add/affine fusion。
4. 在 P7B 原型中接入 P7A two-pixel 小通道模式，但只保留单套主数据流。
5. 扩展到 level2 全部 block，再扩展到 level3。
6. 最后实现 P7C classifier 专用路径。

## 明确禁止

- 不复制多套完整 `win_gen + SA + post`。
- 不恢复旧 pair2/dual wrapper。
- 不在复杂 `window_generator_row` 外层盲目强行 `PIPELINE II=1`。
- 不靠扩大 FIFO 深度假装解决吞吐问题。
- 不只改阵列形状而不改 activation replay/broadcast。
- 不让 P7 与旧 UOP 通用 fallback 混用，避免综合时间再次失控。

## 验收标准

- CSim 输出 mask PA/mIoU 不明显低于 P6J。
- HLS 代码中只有单套主数据流，无旧 datapath、debug/profiling 残留和 dead logic。
- HLS synthesis 时间可控，不再出现 static scheduler 或 memory helper 长时间卡死。
- 资源估计不出现双倍 SA、双倍 WinGen 或异常 LUTRAM/SRL FIFO。
- 实现阶段至少保持 100MHz routed timing clean。

目标路径：

| 阶段 | 目标 |
|---|---:|
| P7D counters | 得到可信 per-op cycle breakdown |
| P7B level2 fusion | 端到端先压到 `60M-70M` PL cycles |
| P7B + level3 fusion | 继续压到 `30M-40M` PL cycles |
| P7C + final tuning | 冲 `10M-15M` PL cycles，即 `100 ms` 级 |

当前最重要判断：P7 的主线不是孤立提高 MAC 阵列利用率，而是减少 WinGen/memory/non-conv pass，并让小通道层在新 block-level 数据流下自然提高阵列填充率。
