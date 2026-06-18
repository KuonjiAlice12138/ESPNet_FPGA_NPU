# P6 实现阶段时序与拥塞收敛计划 (2026-06-08)

## 1. 当前结论

本轮 P6 版本已经完成 HLS 综合、package、Vivado synthesis、place 和 phys_opt，目前实现正在 `route_design` 阶段推进。已有部分报告可以判断下一步优化方向，但还不能替代最终 routed timing report：现在能确定的是**拥塞主风险集中在 conv row-region 附近，HLS 逻辑时序主风险集中在 avgpool C3 fast path**。

当前不建议中断实现。Route overlap 节点数曾从 `411765 -> 131742 -> 55214` 下降，说明 router 有实质收敛；但随后进入 Global Iteration 2 后 overlap 又回到 `421992`，且 Vivado 已提示会优先完成布线而非优化时序。因此本轮最终可能出现两种结果：能 routed 但 WNS 较差，或因 global congestion 失败。

## 2. 当前进展

| 项目 | 状态 |
|---|---|
| HLS blocker | 无 `HLS 214-475`、无 `HLS 200-975` |
| P6 结构检查 | `check_p6_hls_structure.py` 通过 |
| HLS 资源 | BRAM `1441/1488`，DSP `1202/3528`，LUT 估计 `469840/341280`，URAM `112/112` |
| Vivado placed 资源 | CLB LUT `61.47%`，CLB `94.53%`，BRAM Tile `81.85%`，URAM `100%`，DSP `34.27%` |
| Route 状态 | `Phase 5 Rip-up And Reroute`，已进入 `Global Iteration 2` |
| Route 拥塞 | Global/Short congestion level `6`，Timing congestion level `7` |
| Route 中间时序 | `WNS=-3.722 ns`，`TNS=-20968 ns` |

## 3. 部分报告能说明什么

### 3.1 可以判断的内容

HLS 报告显示 top timing slack 为 `-1.46`，对应有效时钟预算下已经有逻辑级违例。具体层级上：

| 模块 | HLS 现象 | 判断 |
|---|---|---|
| `run_p6_pool_op -> avgpool_unit_c3_fast -> avgpool_pixel_c3_inner_fast` | timing slack `-1.46`，`write_c3_avg_pixel` 占 `19265 LUT`，内部走 `on_chip_memory_write_packed_tile` | 当前最明确的 HLS critical path 候选 |
| `run_p6_conv_op -> execute_conv_stream_datapath` | timing slack `-0.25` | 次级逻辑时序风险 |
| `execute_conv_stream_row_region` | HLS timing 接近过线，但资源集中：DSP `1036`，LUT `195863`，BRAM FIFO `101` | 不是 HLS 逻辑最差路径，但很可能是物理拥塞中心 |
| `store_conv_output_row` | timing slack `-0.25`，多条 compact/write path | 次级写回路径风险 |
| `on_chip_memory_write_packed_tile / write_packed_cross_word` | II 从目标 `1` 退到 `3`，LUT 较高 | memory write/RMW 仍是局部风险 |

Vivado route log 显示拥塞区域与 `BRAM/URAM/DSP/CLE` 列附近相关，且 `execute_conv_stream_row_region` 在 HLS 中占用了大量 DSP、LUT、FF 和宽 FIFO。因此如果本轮 route 失败，优先按“物理拥塞”处理，而不是先重写调度。

### 3.2 不能判断的内容

当前还没有最终 routed timing summary，因此不能精确判断最终 critical path 是 avgpool、conv row-region、AXI/SmartConnect 还是跨 SLR/跨列连线。现在的 `WNS=-3.722 ns` 只是 route 中间结果，且 Vivado 明确说明拥塞会影响 timing optimization。

因此本轮实现结束前，不应根据中间 WNS 做大规模回退；应等最终报告确认失败类型。

## 4. 下一步收敛计划

### P6F-1: 等待本轮 route 结束并归档结果

验收项：

- 若实现成功，读取最终 `timing_summary_routed`、`route_status`、`utilization_routed`。
- 若实现失败，保留 `runme.log`、placed utilization、route congestion 信息。
- 记录最终结论：是 timing-only、global congestion，还是 unroutable。

处理原则：

- 如果只是不满足时序但能 routed，优先修 avgpool 和 store/memory write critical path。
- 如果 route 失败或 congestion level 仍高，优先修 conv row-region 物理拥塞。

### P6F-2: 修 avgpool C3 fast path 的逻辑关键路径

目标是降低 `avgpool_pixel_c3_inner_fast` 的组合深度和 LUT 扇入。

计划：

- 拆分 `read_c3_fast_rows -> sum_c3_fast_rows -> write_c3_avg_pixel` 的一拍内组合链。
- 对 `write_c3_avg_pixel` 避免直接走通用 `on_chip_memory_write_packed_tile` 的 cross-word/RMW 路径。
- 针对 C3 avgpool 输出写入建立专用 aligned write 或 row-local pack，减少 `write_packed_cross_word` 调用。
- 不对 reset/控制类循环强制 `II=1`，避免再次引入综合搜索爆炸。

验收：

- `avgpool_pixel_c3_inner_fast` 不再是 top 的 `-1.46` critical path。
- `write_c3_avg_pixel` LUT 明显下降。
- csynth 时间仍在可接受范围内。

### P6F-3: 修 conv row-region 物理拥塞

目标是缓解 SA/window/FIFO 周围的局部 CLB 与 routing 压力。

计划：

- 保留单套 SA、单套 window、单套 weight 数据流，不回到模板克隆或多 datapath。
- 继续保持 `act_stream/wgt_stream/psum_stream` 为 BRAM FIFO，不允许回退到 LUTRAM。
- 检查 `systolic_array_core_row` 内部 `weight_buf`、局部数组、SRL/LUTRAM 是否导致集中拥塞；必要时改成更明确的 BRAM/register 分层。
- 避免扩大 `window_generator_row` 的 shape dispatch，不再新增大块 generic fallback。

验收：

- Vivado placed 后 CLB 利用率和 LUTRAM/SRL 局部占用不继续上升。
- route congestion level 低于当前 level 6/7，或至少不再出现 global unroutable。

### P6F-4: 修 memory write/RMW 路径

目标是降低 `on_chip_memory_write_packed_tile`、`write_packed_cross_word` 对时序和 II 的影响。

计划：

- 对已知 aligned row/compact row 写入继续走专用写路径。
- 对 unavoidable cross-word path 降低位选择和条件写的组合深度。
- 尽量把 read-modify-write 限定在 stage boundary，不在高频 inner loop 内反复触发。

验收：

- `write_packed_cross_word` Final II 不再成为高频路径瓶颈。
- `store_conv_output_row` HLS timing slack 改善。

## 5. 禁止回退的方向

以下方向已经被前几轮验证为低效或高风险，不应在下一轮重复：

| 禁止方向 | 原因 |
|---|---|
| 回到模板化 `run_p6_static_uop<N>` | 会导致硬件克隆和资源爆炸 |
| 恢复旧 `if_dec` / 动态 UOP fetch-decode | 与当前 P6 静态调度目标相反 |
| 新增第二套 SA/dual datapath | 当前问题是拥塞和时序，不是简单算力不足 |
| 扩大 generic window fallback | 会增加 HLS 搜索空间和 LUT 资源 |
| 对低收益控制循环强制 `II=1` | 已多次导致综合时间不可控 |

## 6. 下一次工作入口

本轮 route 完成后，先按最终结果二选一：

1. 如果 routed 成功但 WNS 仍差：先实施 `P6F-2 avgpool timing`，再处理 `P6F-4 memory write`。
2. 如果 route 失败或 congestion 仍不可接受：先实施 `P6F-3 conv row-region congestion`，重点看 SA/window/FIFO/weight buffer 周边物理资源。

无论哪条路线，修改后都必须先跑：

- `python tools/check_p6_hls_structure.py`
- prefix CSim 或轻量 CSim
- top csynth
- `python tools/audit_hls_reports.py`

 只有在 HLS blocker 为 0、无硬件克隆、宽 FIFO 没有回退到 LUTRAM 后，才推进 Vivado 实现。

## 7. 2026-06-08 route_design 失败归档

本轮实现于 13:40 进入 `route_design`，18:26 以 `route_design failed` 退出，历经 3 个 Global Iteration：

| Iteration | overlaps 起点 | overlaps 终点 | 状态 |
|---|---|---|---|
| 1 | 463,649 failed nets (353,455 unrouted) | 55,214 | 收敛中 |
| 2 | 421,992 | 220,516 | 反弹 |
| 3 | — | 35,330 node overlaps | 所有 nets 已布线但物理冲突残存 |

最终错误：

```
CRITICAL WARNING: [Route 35-162] 26,609 signals failed to route due to routing congestion.
ERROR: [Route 35-2] Design is not legally routed. There are 35,330 node overlaps.
```

### 7.1 拥塞区域

TOP 10 冲突节点集中在 `INT_X37/X38` 区域（即 X37Y84 ~ X38Y136），类型为 `NODE_VLONG12`（极长线）。这对应芯片布局上 SA 核 + win_gen + FIFO 所在列的纵向长距离布线。

### 7.2 放置后资源

| 资源 | 值 |
|---|---|
| CLB LUT | 209,783 / 341,280 = **61.47%** |
| CLB Registers | 160,175 / 682,560 = 23.47% |
| CLB Tile | ~**94.53%** |
| BRAM Tile | ~81.85% |
| URAM | 112 / 112 = 100% |
| DSP | 1,209 / 3,528 = 34.27% |
| HLS avgpool slack | **-1.46ns** |
| HLS conv row slack | **-0.25ns** |

### 7.3 根因归类

**物理拥塞主因：CLB Tile 94.53%**。不是 LUT 总量不够，而是几乎所有 CLB 被占用后路由器缺少可用的过孔（routing switch）来走线。CLB 高利用率来自三部分：

1. **SA 核区域 LUTRAM/SRL 密集**：`weight_buf` 被 HLS 映射为 RAM_AUTO，Vivado 放置后落入 LUTRAM/SRL。`systolic_array_core_row_Pipeline_VITIS_LOOP_208_4` 含大量局部寄存器和 SRL，在 SA 周围形成高密度 CLB 地带。
2. **URAM 100% 削弱布线通路**：URAM 列占满后，其所在 SLR 区域的通用布线 switch box 容量下降，影响相邻逻辑的 cannel 分配。
3. **avgpool C3 区域**：`write_c3_avg_pixel` 调用通用 `write_packed_cross_word`，19265 LUT 集中在 avgpool 附近，进一步挤占 channel。

## 8. P6F-2: avgpool C3 关键路径修复（细化方案）

基于 route 失败结果，avgpool 区域不仅是时序热点也是物理拥塞贡献者。当前 `avgpool_pixel_c3_inner_fast` 的数据流为：

```
read_c3_fast_rows (3次 packed URAM read)
  → sum_c3_fast_rows (27路 INT32 加法, 完全 UNROLL)
  → write_c3_avg_pixel → on_chip_memory_write_packed_tile
    → write_packed_one_word / write_packed_cross_word (RMW: read→mask→shift→write)
```

### 8.1 修复方案

1. **C3 专用 aligned write**：C3 avgpool 输出始终为 3 字节，目标地址相对于 256-bit word 的偏移固定且可预判。新建 `write_c3_pool_aligned()`，不经过通用 `write_packed_tile`：
   - 如果字节偏移 + 3 ≤ 32：一次 `read_bank_word` → 修改低 24 位 → `write_bank_word`
   - 如果跨 32 字节边界（仅出现在极少数非对齐 layout）：退化为跨字 RMW
   - 省去通用路径的 `make_low_byte_mask` / dynamic shift / 跨字判断开销

2. **拆 `sum_c3_fast_rows` 为两拍 pipeline**：
   - Stage 1：3 行各自 intra-sum（3 列 × 3 行 = 9 个 INT32 加法）
   - Stage 2：3 行 inter-sum（3 个 INT32 加法）
   - 目标把单拍 27 路加法降为 2 拍 9+3 路，降低组合深度

3. **显式 BRAM 绑定**：`read_c3_fast_rows` 内部的临时 `act_vec_t row0/1/2` 如果落入寄存器，加 `BIND_STORAGE bram` 或确保只作为 pipeline stage。

### 8.2 验收

- HLS csynth: `avgpool_pixel_c3_inner_fast` slack ≥ 0
- `write_c3_avg_pixel` LUT 从 19K 显著下降
- Vivado placed: avgpool 区域 CLB 局部密度不进一步上升

## 9. P6F-3: conv row-region 物理拥塞修复（增强方案）

### 9.1 weight_buf 显式 BRAM 绑定

```
当前: i8_t weight_buf[MAX_SA_K_TILES][TM][TK];
      ARRAY_PARTITION complete dim=2,3 → RAM_AUTO_1R1W

修复: 加 BIND_STORAGE type=ram_2p impl=bram
```

SA 核的 `weight_buf` 是 72×32×32 的数组。虽然 HLS 报告显示 `RAM_AUTO_1R1W`，但 Vivado 放置后可能部分落入 LUTRAM/SRL（特别是完整分区的 dim=2 和 dim=3 生成大量独立寄存器读写口）。显式 BRAM 绑定能强制使用 BRAM36 模块，释放 ~5000-8000 LUT。代价是增加约 16-20 BRAM36，但当前 BRAM 82% 有空间。

### 9.2 SA 核内 SRL 控制

`systolic_array_core_row_Pipeline_VITIS_LOOP_208_4` 中的 SRL 和大扇出控制寄存器是 CLB 局部拥塞的来源。在不改动 SA MAC 阵列的前提下，对这些控制逻辑：
- 生成 `ap_CS_fsm` 相关的寄存器如果需要大扇出，考虑用 BUFG 复制
- 但不在 HLS 层做——这是 Vivado PhysOpt 的工作，HLS 层只需确保不把关键控制信号做成 SRL

### 9.3 win_gen shape dispatch 确认

`window_generator_row()` 的专用路径（c19 stride2 / c19 stride1 / 1x1 aligned / small-c stride2 等）均以 `INLINE off` 实现为独立模块，结构正确。不需要再加 `ALLOCATION` 或 PBLOCK——place 阶段的区域约束留给 Vivado Strategy，当前 HLS 代码结构已足够。

## 10. CLB 利用率收敛硬性指标

为解除 94.53% CLB 导致的布线拥塞，下一版 csynth 必须满足：

| 指标 | 当前值 | 目标值 | 手段 |
|---|---|---|---|
| HLS LUT 估算 | 469,840 (138%) | < 450,000 (132%) | weight_buf BRAM 绑定 + avgpool C3 LUT 削减 |
| Vivado placed CLB LUT | 209,783 (61.5%) | < 195,000 (57%) | LUTRAM→BRAM 迁移 |
| Vivado placed CLB Tile | ~94.5% | < 90% | CLB LUT 降 15% 以上 |
| avgpool HLS slack | -1.46ns | ≥ 0 | P6F-2 |
| conv row HLS slack | -0.25ns | ≥ 0 | P6F-3 weight_buf BRAM |

## 11. 禁止方向补充

基于 route 失败的直接经验：

| 禁止方向 | 原因 |
|---|---|
| 恢复 LUTRAM FIFO (act/wgt/psum_stream) | 已验证导致 4.5k LUT 消耗和 CLB 拥塞恶化 |
| 对 `clear_tables` / invalidation 类 reset 循环加 `II=1` | 已导致综合 hang（workplan_0606 §9），且对性能无贡献 |
| 将多个 shape-specific win_gen 路径合并回一个函数 | 会增加单点 LUT 密度，恶化拥塞 |
| 新增第二套 SA 或 dual datapath | 当前问题不是算力，是布线空间 |
| 在 HLS 层做 `ALLOCATION` / PBLOCK 物理约束 | 留给 Vivado Strategy 阶段，HLS 层不做物理干预 |
| 恢复旧 UOP dispatch / generic win_gen fallback | P6 方向不可退 |

## 12. 下一版入口

优先同时实施 P6F-2（avgpool C3）和 P6F-3（weight_buf BRAM），因为两者：
- 不冲突：avgpool 在 `avgpool_unit.cpp`，weight_buf 在 `sa_core.cpp`
- 都低风险：单函数改动，不出新的 shape dispatch 或控制流分支
- 直接命中两个最大问题：avgpool 是唯一 -1.46ns 时序违例，weight_buf 是 CLB 拥塞主因

实施后验收顺序：
1. `python tools/check_p6_hls_structure.py` ✅ 必须
2. prefix CSim（`ESP_INT8_CSIM_MAX_UOP=6` 覆盖 stem stage）
3. top csynth（目标 < 45min）
4. `python tools/audit_hls_reports.py`
5. HLS 报告 avgpool slack ≥ 0 + HLS LUT < 450K → 推进 Vivado 实现
