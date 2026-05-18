# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-13/14)

## 1. 当前进展

本轮工作已从“硬件能否跑通”推进到“结果可信 + 性能归因 + 定向优化”。当前可信版本链如下：

| 项目 | 当前状态 |
|---|---|
| HLS CSim | 整网通过，生成 `hls_output_q.bin` |
| HLS synthesis/package | 最新版通过，audit 无 dataflow/blocker |
| Vivado bitstream | `platform_0514_prof_wg1` 已导出 |
| Vitis app | 已对齐 `platform_0514_prof_wg1` |
| 上板输出 | `D72OUT.BIN` 与 HLS reference bit-exact |

最新 `platform_0514_prof_wg1` 对应 app tag：

`INT8-BOARD-20260514-PROFWG1`

上板后 `D72OUT.BIN` 与 `E:\HLSREF.BIN`、`D:\ESP_INT8\ESP_INT8_hls\hls_output_q.bin` 均完全一致：

| 比对项 | 结果 |
|---|---:|
| byte mismatch | `0 / 16384` |
| max abs diff | `0` |
| SHA256 | `7CFDE9B332E7426B8F4CEA5EAD6FC3C8D878C8F83ADF2B333405B89EE41D9364` |

## 2. 性能结果

当前完整整网单次 `MODE_RUN` 实测：

| 版本 | cycles | 延迟 |
|---|---:|---:|
| profiling v1 | `81,787,640` | `817 ms` |
| `PROFWG1` | `52,600,953` | `526 ms` |

`PROFWG1` 相比 profiling v1 约 `1.56x` 加速，延迟降低约 `35.7%`。这说明 window/packed-path 方向有效，但距离 50ms 目标仍有数量级差距。

`PROFWG1` profiling counter：

| 计数器 | 数值 |
|---|---:|
| executed uops / conv uops | `74 / 26` |
| window read ops / words | `5,693,440 / 2,760,704` |
| weight words | `468,992` |
| SA steps | `2,760,704` |
| psum words | `9,601,024` |
| output tiles / RMW ops | `630,784 / 1,261,568` |
| model cycles | `14,097,984` |

注意：在 `PROFWG1` 上，`model_cycles` 已明显低于实测 cycles，说明当前 profiling v1 的模型不再能完整解释优化后路径的真实延迟。后续需要补充更细粒度周期计数器，而不能只看事件数。

## 3. 当前风险

1. HLS 资源仍紧：BRAM 约 95%，URAM 100%，LUT 仍超过器件估计容量；Vivado 实现虽然可能继续通过，但后续优化不能随意增加大 FIFO 或重复 buffer。
2. 当前 526ms 仍远高于 50ms 目标，后续必须做结构性性能收敛，不能只依赖 pragma 微调。
3. profiling v1 已能记录事件规模，但缺少 per-phase cycle、FIFO stall、SA wait 等周期级信息；优化后模型残差增大，下一版应补这类计数器。
4. 每轮切换 platform 后必须 clean/reconfigure app，避免 BSP/include/lib 仍引用旧平台。

## 4. 下一步计划

### P0：保持可信验证链

每轮 HLS 修改后固定执行：

1. 跑整网 top CSim。
2. 确认 `hls_output_q.bin` 与当前 reference 对齐。
3. 综合前运行 HLS report audit。
4. 上板后比对 `D72OUT.BIN` vs `HLSREF.BIN`，要求 bit-exact。

### P1：补 profiling v2

下一版硬件优先增加轻量周期级计数器，而不是继续盲改数据通路：

| 计数器方向 | 目的 |
|---|---|
| conv/window/SA/post/writeback phase cycles | 拆分 526ms 的真实来源 |
| stream empty/full 或 wait counter | 判断是否存在 back-pressure |
| SA valid step / drain step | 估算有效 PE 利用率和无效 lane 代价 |
| RMW/writeback cycles | 量化 partial-word 写回瓶颈 |

计数器必须走 AXI-Lite 只读寄存器，不引入大存储，不改变主数据流功能边界。

### P2：按 profiling 结果继续优化

当前优先级暂定如下：

| 优先级 | 方向 | 目标 |
|---|---|---|
| P2.1 | `win_gen.cpp` / packed tile read | 降低 3x3 window generation 与 packed read 开销 |
| P2.2 | post/output drain | 减少 `psum_words` 中无效 lane drain |
| P2.3 | writeback/RMW | 降低 `rmw_ops=1,261,568` 的 partial-word 写回代价 |
| P2.4 | small-Cout 多空间点并行 | 在确认瓶颈后提高 level2 小通道层 SA 利用率 |

短期目标先把 full `MODE_RUN` 从 `526ms` 压到 `300-400ms` 区间，并保持 D72 bit-exact；50ms 目标需要更激进的结构性并行化，不能用当前事件模型直接保证。
