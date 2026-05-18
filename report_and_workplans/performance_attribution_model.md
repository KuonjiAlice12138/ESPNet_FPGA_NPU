# INT8 NPU 性能归因模型

## 目标

后续性能优化不再只根据单个 HLS warning 或局部直觉推进，而是先建立按层/按阶段的 cycle 归因表，用于判断当前瓶颈属于计算受限、window generation 受限，还是 memory/writeback back-pressure 受限。

## 当前工具

脚本：`tools/analyze_perf_attribution.py`

默认输入：

- `hw_artifacts/hw_constrained_qat_3ep_single/uop_table.bin`
- 脚本内置的最新上板 prefix profiling 数据

默认输出：

- `hw_artifacts/hw_constrained_qat_3ep_single/perf_attribution.json`
- `hw_artifacts/hw_constrained_qat_3ep_single/perf_attribution_conv.csv`
- `hw_artifacts/hw_constrained_qat_3ep_single/perf_attribution_stage.csv`

运行方式：

```powershell
python D:\ESP_INT8\tools\analyze_perf_attribution.py
```

如后续保存了新的串口 profiling log，可用：

```powershell
python D:\ESP_INT8\tools\analyze_perf_attribution.py --profile-log path\to\board_perf.log
```

## 表中字段含义

每个 CONV uop 至少拆分以下估算项：

| 字段 | 含义 |
|---|---|
| `input_feature_read_cycles_est` | 按当前 window path 估算的片上 activation tile read 次数 |
| `weight_cache_cycles_est` | 每层开始时从 param blob/cache 取权重向量的估算开销 |
| `weight_stream_cycles_est` | 当前 row dataflow 中每个输出行重新向 SA stream 喂权重的估算开销 |
| `window_gen_cycles_est` | 结合当前 `win_gen.cpp` 路径估算的窗口生成/打包开销 |
| `sa_mac_cycles_est` | 理想情况下 SA 每个输出像素执行 `k_tiles` 个 MAC step 的开销 |
| `psum_output_drain_cycles_est` | 当前 SA/post 仍按 `TM=32` 循环 drain 输出的开销 |
| `output_write_cycles_est` | 输出 tile 写回片上 memory 的估算开销 |
| `read_modify_write` | 输出写回是否会触发 partial-word RMW |
| `pe_fill` | 仅由 `Cout/K` tiling 决定的有效 PE 填充率 |

stage 表会把实测 prefix cycles 与模型 lower bound 对齐，给出 `unattributed_or_stall_cycles`。该项不能直接等价于 FIFO stall，它还包含真实 HLS schedule II、函数调用、memory arbitration、stream empty/full 等未被离线模型精确拆分的开销。

## 当前结论

基于 2026-05-13 的上板 profiling：

| 阶段 | 主要估算瓶颈 | 说明 |
|---|---|---|
| U00-U04 | `psum_output_drain` | U02 首层较大，当前每像素仍承担 `TM=32` 输出 drain |
| U05-U20 | `window_gen` | level2 小通道 3x3 卷积主导，窗口打包和 RMW 开销明显 |
| U21-U39 | `window_gen` | 与 level2_0 类似，小通道层 PE fill 低且 window path 成本高 |
| U40-U53 | `window_gen` | U40 输入通道 131，activation segment read 压力高 |
| U54-U69 | `window_gen` | measured/model ratio 已接近，后续优化收益相对更可预测 |
| U70-U72 | `psum_output_drain` | classifier `Cout=2`，PE fill 很低但总计算量小 |

这说明下一步不应只盯着 SA MAC 本身。level2 的主问题是 window generation/packing 与 partial-word writeback；U02 和尾部 classifier 则更像 output drain 结构效率问题。

## Profiling v1

当前 profiling 版硬件已加入一组轻量 AXI-Lite 计数器，app 会在 full `MODE_RUN` 后打印：

- `uop`、`conv`
- `win_read`、`win_words`
- `wgt`
- `sa_steps`、`psum`
- `out_tiles`、`rmw_ops`
- `model_cycles`

这些计数器用于校验硬件实际执行序列和离线模型的一致性，并把 board full-run cycles 拆成 `model_cycles + residual`。它们不是周期级 stall 计数器，因此不能直接给出 FIFO empty/full 或 SA wait 的精确周期数。

如果 profiling v1 的 residual 仍过大，再实现 profiling v2：增加 phase/uop cycle、win_gen active/idle、SA wait-act/wait-weight、post/writeback active、stream empty/full 等周期级计数器。v2 需要更谨慎评估资源和时序风险。
