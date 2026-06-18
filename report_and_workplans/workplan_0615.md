# Workplan 0615 - P6 Timing Closure Follow-up

## 当前状态

本轮 P6 实现经历两阶段推进：早期版本已经能够完成合法布线但 post-route physopt 后仍未满足 100MHz 时序；随后按 P6J 方向继续拆短写回、window staging 和结构检查后，最新实现已在 100MHz 下完成 route 并 timing clean。问题性质已经从前几轮的“不可布线/严重拥塞”推进到“可布通且时序刚好通过”，说明 P6 主路径结构方向可以保留，但 BRAM/URAM 周边布线仍是后续性能与稳定性收敛的核心风险。

关键报告结论：

- 最新实现 `route_status` 显示布线完成：`routable nets=454050`，`fully routed nets=454050`，`routing errors=0`。
- `timing_summary_postroute_physopted` 显示：`WNS=+0.139ns`，`TNS=0`，setup failing endpoints `0`；hold 也 clean (`WHS=+0.011ns`, `THS=0`)。
- route 阶段耗时约 `1h19m`，峰值内存约 `10.9GB`；route finalize 后 `Failed Nets=0`、`Unrouted Nets=0`、`Node Overlaps=0`。
- 资源实现后可接受但余量集中在逻辑，存储仍很紧：`CLB LUTs=56.72%`，`CLB=91.79%`，`BRAM Tile=97.58%`，`URAM=100%`，`DSP=35.74%`。
- DRC 无 error/critical warning，methodology checks 为 `0`；剩余 DRC warning 主要是 DSP input/output pipeline 建议，不阻断上板。
- 当前可以推进 bitstream、导出硬件平台和单图上板验证；但 `WNS=+0.139ns` 裕度很窄，仍不适合继续冲频或立即视为最终收敛版本。

## 关键路径分析

早期未过时序版本的最坏路径 1 是卷积输出写回路径：

- Source：`store_conv_output_row -> on_chip_memory_write_packed_tile -> lanes_reg_533`
- Destination：`s_fmbuf_uram/ram_reg_uram_144/DIN_A[34]`
- Data path delay `10.487ns`，其中 logic `2.564ns`，route `7.923ns`，route 占 `75.55%`。
- 该路径说明 packed lane 拼接、跨 word 写、bank/address/mask 选择与 URAM DIN 之间仍存在过长的组合链和过远的物理连线。

早期未过时序版本的最坏路径 2 是 window 生成到 act stream 的写入路径：

- Source：`window_generator_row -> emit_window_row_3x3_c19_stride2_fast -> on_chip_memory_read_packed_contiguous`
- Destination：`act_stream` BRAM FIFO `DINBDIN[28]`
- Data path delay `10.839ns`，其中 logic `2.314ns`，route `8.525ns`，route 占 `78.65%`。
- 该路径说明从 feature buffer 读出、边界/stride/pixel 选择到 stream 写入之间仍过于直接，缺少足够的局部 staging。

早期未过时序版本的最坏路径 3 仍与 `s_fmbuf_uram` 相关：

- 涉及 `s_fmbuf_uram` 内部或周边的 URAM 写入、frame load alias、pool/add/store 写回 fanout。
- 报告中还出现 `avgpool_unit_c3_fast`、`avgpool_c3_group32_inner_fast_pack_write`、`write_packed_cross_word`、`store_compact_c*_row` 等路径。
- 这说明问题不是单个函数 bug，而是“多个算子共享一套过于通用的 packed memory writer”，导致 HLS 生成大量跨模块选择逻辑和长布线路径。

最新 timing-clean 版本的最坏路径已经发生变化：

- 最坏 setup path slack `+0.139ns`，source 为 `wgt_stream` FIFO BRAM，destination 为 `systolic_array_core_row` 内部 `weight_buf` BRAM，data path delay `9.328ns`，其中 route `8.193ns`，route 占 `87.8%`。
- 后续关键路径多为 `row_buf` BRAM 到 `s_fmbuf_uram` DIN，route 占比约 `83%~85%`。
- 这说明 P6J 修补有效降低了原先 packed writer/act stream 的直接阻断风险，但最终瓶颈转移为 BRAM/FIFO/URAM 之间的物理距离与存储周边布线压力。

## 根因判断

当前 100MHz 不收敛的主因不是 SA/MAC 计算路径。SA 本体 HLS 估计较短，真正限制来自片上 feature buffer 周边的数据搬运和写回网络。

主要根因：

- `on_chip_memory_write_packed_tile` 和 `write_packed_cross_word` 仍被 conv/store/pool/add 等热路径复用，功能过通用，导致 bank 选择、word 对齐、lane mask、cross-word 写入在内层循环中形成宽组合逻辑。
- `s_fmbuf_uram` 是全局热点存储，conv、pool、add、frame load/store 等多个路径都会驱动其写口，综合后形成高扇出、高拥塞的写回网络。
- `window_generator_row` 仍从 packed contiguous read 结果直接构造 act stream word，导致 read-side 地址/像素选择逻辑与 FIFO DIN 之间跨区域连接过长。
- 当前把部分中间缓冲转为 BRAM 后降低了 LUTRAM/SRL 风险，但 BRAM/URAM 资源与放置压力仍高，单靠实现阶段优化已经无法消除 `-1ns` 量级 setup slack。
- HLS 估计 Fmax 与实现后 timing 差距仍然明显，说明下一轮判断标准必须以 post-route 关键路径为准，而不是只看 HLS csynth timing。

最新实现通过后，根因判断需要更新为：

- P6 结构已经能够 100MHz timing-clean，但余量仅 `0.139ns`，实际限制仍是 route-dominated 的存储互连。
- SA/MAC 本身不是当前最差逻辑链；`wgt_stream -> weight_buf` 和 `row_buf -> fmbuf_uram` 是下一步更值得处理的物理热点。
- 继续优化应以减少跨 BRAM/URAM 宽总线、缩短 weight/row staging 的物理路径为主，而不是大规模改变 P6 调度或恢复旧 datapath。

## 下一轮 HLS 改进计划

### P6J-1：拆分热路径 packed writer

将当前通用 packed writer 拆成更窄、更确定的写回路径：

- 为卷积输出行写回提供 aligned row-word writer，内层循环只接收已经确定的 `bank/base/word_idx/word_data`。
- 为 `pool2` 或其它固定布局 tensor 提供专用 writer，避免在热路径中传播完整 tensor descriptor。
- `write_packed_cross_word` 只保留为低频 fallback，不允许出现在 conv output、avgpool c3 fast、compact store 的内层循环。
- 对 `store_c16_into_c19_row`、`store_compact_c*_row` 等已知布局路径，预先在外层计算 address/bank，内层只做连续 word 写。

目标：切断 `lanes_reg -> CARRY/LUT -> URAM DIN_A` 的长组合链，降低 `s_fmbuf_uram` 写口附近的 route delay。

### P6J-2：写回路径增加显式 staging

在不改变 P6 调度边界的前提下，将写回拆成两级：

- Stage A：完成 lane packing、mask、目标 bank、目标 word address 计算。
- Stage B：只执行最终 BRAM/URAM write。
- 对跨 word 写入，优先转化为两个明确 word write，而不是在同一组合路径中完成 read-modify-write 和 bank 选择。

允许为了 timing 增加少量 cycle；当前瓶颈已经是无法稳定 100MHz，牺牲少量 cycle 换取 timing closure 是合理的。

### P6J-3：window 读出到 act stream 增加局部寄存

针对 `emit_window_row_3x3_c19_stride2_fast` 和 `on_chip_memory_read_packed_contiguous`：

- 在 packed read 结果和 `act_stream.write()` 之间增加局部 pixel/window word staging。
- 将边界判断、stride2 地址选择、pixel unpack 与 FIFO 写入拆开，避免直接驱动 BRAM FIFO DIN。
- 对 c19 stride2 fast path 保持专用路径，不回退到通用 window generator。

目标：降低 `window_generator_row -> act_stream BRAM FIFO` 的 `8.5ns` route-dominated path。

### P6J-4：继续控制 SA 相关存储压力

当前 SA 本体不是最坏路径，但 `weight_buf` BRAM 化后整体 BRAM 压力较高，需要避免反向制造新的放置拥塞：

- 禁止将 SA weight/window stream buffer 回退为 LUTRAM/SRL。
- 优先采用较少 bank 的 BRAM 分组或 load-and-compute tile staging，而不是 32 路完全复制。
- 不恢复旧的 group-local stream/FIFO 多 datapath 方案，避免重新引入高 fanout 和综合时间爆炸。

### P6J-5：检查 pool/add/store 的写回实现

报告中 `avgpool_unit_c3_fast` 仍出现在 timing path 中，下一轮需要同步处理：

- `avgpool_c3_group32_inner_fast_pack_write` 改成 aligned word writer。
- add/store 输出尽量使用相同的窄接口 writer，但不能共用过度通用的 cross-word 动态逻辑。
- 保持模块边界清晰：算子模块负责产生 word-level 输出，memory/scratch 层负责确定物理 bank 写入。

### P6J-6：强化结构检查脚本

更新 `tools/check_p6_hls_structure.py`，将以下情况列为 high risk：

- 热路径中继续调用 `write_packed_cross_word`。
- conv/pool/add/store 内层循环继续直接调用通用 `on_chip_memory_write_packed_tile`。
- SA/window 相关 FIFO 或 buffer 被绑定到 LUTRAM/SRL。
- 顶层重新出现旧 datapath、debug/profiling 残留或未调用 dead logic。

## 验证与推进门槛

下一轮代码修改后按以下顺序推进：

- 先跑结构检查脚本，确认没有旧 datapath、热路径通用 writer、LUTRAM/SRL 回退。
- 跑轻量 csim 或 prefix/top golden csim，确认功能没有破坏。
- 跑 HLS csynth，要求综合时间可控，不能重新出现 static scheduler 或 memory helper 长时间卡死。
- 用 audit 脚本检查 HLS 报告，重点看是否新增 II 爆炸、函数克隆、资源异常。
- 只有在 HLS 报告无 blocker 后再推进 Vivado 实现。

实现阶段判断标准：

- route 必须保持合法完成。
- 首轮目标是把 post-route WNS 从 `-1.220ns` 改善到 `>-0.5ns`，再继续追求 timing clean。
- 重点观察 top timing path 是否仍指向 `on_chip_memory_write_packed_tile/write_packed_cross_word` 和 `window_generator_row -> act_stream`；如果仍然相同，说明 HLS staging 不足。
- 资源可以小幅超估，但不能通过 LUTRAM/SRL 换 BRAM，也不能显著增加全局控制 fanout。

## 明确禁止的方向

- 不继续仅靠 Vivado strategy、physopt 或更多实现轮次碰运气。
- 不把 SA/weight/window buffer 推到 LUTRAM/SRL 来降低 BRAM 数字。
- 不恢复旧版多套 datapath、debug/profiling 端口或 group-local stream 过渡逻辑。
- 不在低收益控制循环上强制 `II=1`，避免再次造成综合搜索时间爆炸。
- 不只看 HLS estimated Fmax 决策，必须以后续 post-route timing path 为最终依据。

## 下一步优先级

1. 先基于最新 timing-clean bitstream 做单图上板验证，确认 fullres mask 输出与 host 评估没有明显退化。
2. 导出新 platform 后同步更新 app 的 platform 路径和 launch bitstream；当前 app 已将 build tag 更新为 `INT8-BOARD-20260615-P6J-TC100-SINGLE`，PL 目标频率改为 `100MHz/10ns`。
3. 若上板正确，记录真实单图 A53 timer tick/ms，与 fullres100 和上一版 P6 对齐比较。
4. 下一轮 HLS 优化不再优先改大结构，而是针对最新最差路径：收敛 `wgt_stream FIFO -> weight_buf BRAM` 与 `row_buf BRAM -> fmbuf URAM` 的物理连线。
5. 继续保留结构检查脚本，禁止恢复旧 datapath、profiling/debug 残留、LUTRAM/SRL 回退和通用 packed writer 热路径。

## 单图上板结果与验证集测试

`platform_full100_0615` 已导入 Vitis，并完成 platform/app build。配置核查结果：

- app、CMake cache、launch bitstream、FSBL 均指向 `platform_full100_0615`。
- ELF 内含 `INT8-BOARD-20260615-P6J-TC100-SINGLE` 和 `platform_full100_0615` 字符串，无旧平台路径残留。
- app 使用 A53 `CNTPCT_EL0` timer，`cntfrq=33333000Hz`，自检误差约 `9982ppm`，计时口径可信。

单图上板结果：

```text
FULL_MODE_RUN = 33,434,687 A53 ticks
真实延迟约 1003 ms
PL target = 100 MHz / 10 ns
输出 MASK.BIN = 512*1024 bytes
```

与 `fullres100_0527` 基线相比：

- `fullres100_0527` 平均 `41,788,175` A53 ticks，约 `1254 ms`。
- 当前 P6J 单图 `33,434,687` ticks，约 `1003 ms`。
- 单图延迟下降约 `20.0%`，说明 P6J 已带来端到端周期压缩，但距离 `100 ms` 级目标仍约 `10x`。

下一步切换 app 到全验证集测试模式：

- `INT8_APP_BUILD_TAG` 改为 `INT8-BOARD-20260615-P6J-TC100-VAL`。
- `INT8_APP_ENABLE_VAL_SET_TEST=1`。
- `INT8_APP_ENABLE_SINGLE_PERF_BEFORE_VAL=0`，避免验证集前重复跑单图。
- SD 卡已有 `PARAM.BIN`、`I0000.BIN` 起始验证集输入和 `Txxxx.BIN` 标签；当前 `O*.BIN=0`，不会混淆新输出。

全验证集跑完后需记录：

- `VALSET SUMMARY` 的 sample count、total ticks、avg ticks/ms。
- `Oxxxx.BIN` 是否完整生成。
- 用 host 侧 full-resolution mask 评估脚本重新计算 PA/mIoU。

## 全验证集输出与精度结果

SD 卡输出检查：

- `O0000.BIN` 到 `O0499.BIN` 共 `500` 个 full-resolution mask。
- 每个输出文件大小均为 `524288 bytes`。
- `I0000.BIN` 到 `I0499.BIN` 输入和 `T0000.BIN` 到 `T0499.BIN` 标签均完整，尺寸检查无异常。

使用 `tools/eval_val_hw_masks_fullres.py` 对 `E:\O%04d.BIN` 重新评估，结果如下：

```text
processed = 500
PA        = 0.97800421
mIoU      = 0.86356491
acc       = [0.81670988, 0.99223832]
IoU       = [0.75068569, 0.97644412]
```

结果已保存到：

```text
D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_val\board_val_fullres_mask_metrics.json
```

判断：

- P6J 输出的 full-resolution mask 精度与 `full100_0527` 板端 fullres baseline 一致，没有出现可观测精度退化。
- P6J 单图真实延迟约 `1003 ms`，相比 `full100_0527` 的约 `1254 ms` 有约 `20%` 改善；但仍远未达到 `100 ms` 级目标。
- 当前缺少本次验证集串口 `VALSET SUMMARY`，因此验证集平均延迟待补。后续若需要完整表格，应补记录 `samples/total_ticks/avg_ticks/min/max`。
