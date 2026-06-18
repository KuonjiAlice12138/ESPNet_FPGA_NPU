# P6 时序收敛与 HLS 返修计划 (2026-06-11)

## 1. 当前结论

本轮 P6 版本已经完成 HLS 综合，并推进到 Vivado 实现的 route 阶段。实现过程说明：**当前结构已经从“不可综合/不可布线”推进到“可进入 route”，但仍无法靠实现策略自然收敛时序**。上一轮 route 能做到 `Failed Nets=0 / Unrouted=0 / Node Overlaps=0`，但 routed 后 WNS 仍约 `-1.8ns`，TNS 仍在 `-2.3e4ns` 量级。

最新一轮 post-phys-opt 后 timing 仍无实质改善：`WNS=-2.066ns / TNS=-13922ns`。进入 route 后，router 初期即报告 Global/Short congestion level `6`、Timing congestion level `7`，并提示会优先完成布线而不是优化时序。该证据说明下一轮 HLS 返修的重点必须转为**降低 memory/store 写回路径的局部拥塞和 URAM 周边扇出**，而不是继续单纯挤 SA 阵列或调整 Vivado strategy。

因此下一步不应继续等待长时间 route 或反复调 Vivado strategy，而应回到 HLS 代码侧，针对 `URAM packed write/read`、`store_compact`、`avgpool C3 write` 和 `fmbuf_uram` 高扇出路径做结构收敛。

## 2. 今天进展

| 项目 | 结果 |
|---|---|
| Top HLS csynth | 通过，耗时约 `9m20s` |
| HLS blocker | 无 `HLS 214-475`、无 `HLS 200-975`、无 fatal error |
| HLS top timing | target `10ns`，estimated `7.733ns` |
| HLS top resources | BRAM `1441/1488`，DSP `1215/3528`，LUT `449750/341280`，URAM `112/112` |
| CSim 正确性 | full-res mask 差异可接受：`3549/524288`，单图 mIoU 下降约 `0.46pp` |
| Vivado placed 资源 | CLB `94.95%`，CLB LUT `62.91%`，BRAM Tile `81.85%`，URAM `100%` |
| Route 拥塞 | 初始 Global/Short congestion level `6`，Timing congestion level `6` |
| Route 结果 | nets 可布通，但 timing 差：route 后 WNS 约 `-1.8ns` |
| 最新 post-phys-opt | WNS `-2.066ns`，TNS `-13922ns`，route 初期 congestion `6/7` |

与 0608 版相比，本轮 BRAM FIFO 和结构清理让设计从“route 拥塞失败”推进到了“可以 legal route”；但时序没有进入可验收区间。这个结果有价值：它把问题从“不可布线”缩小到“可布线但关键路径/局部拥塞过重”。

## 2.1 0612 HLS 返修进展

已按 P6G-1 的低风险切入点完成第一轮 HLS 代码修补：

- `avgpool C3` 输出不再走 per-pixel packed/RMW 写回，改为行内 256-bit word 顺序拼接，满 word 后 aligned write。
- `conv_store` compact row 写回提前固定 `row_base/bank`，避免每个 256-bit word 写回都重新传播完整 tensor descriptor。
- `memory.cpp` 删除已弃用的 C3 byte/RMW 专用写回入口，P6 路径只保留 aligned absolute word 写入和仍需保留的通用 packed tile 写入。
- `tools/check_p6_hls_structure.py` 已加入检查，禁止旧 `write_c3_avg_pixel / on_chip_memory_write_c3_abs` 路径重新混入。

验证状态：

- `python tools/check_p6_hls_structure.py`：通过。
- `git diff --check`：无实质错误，仅剩 CRLF 提示。
- `hls_config_csim_prefix_lite.cfg` 轻量 CSim：`top_prefix_lite_tb` 通过，`CSim done with 0 errors`。

下一步可以进入 top csynth；综合后重点观察 `write_c3_avg_pixel` 是否已从报告热点消失，以及 `store_compact_c25_row / fmbuf_uram` 相关路径是否仍是 route 主瓶颈。

## 2.2 0612 Vivado route failure 复盘

本轮 HLS 修补后 top csynth 通过，且实现推进到 `route_design`，但最终 route 失败：

| 项目 | 结果 |
|---|---|
| Route fatal | `[Route 35-162] 3002 signals failed to route due to routing congestion` |
| Legal route | `[Route 35-2] Design is not legally routed` |
| Route status | `Failed Nets=0`，`Unrouted Nets=0`，`Partially Routed Nets=0`，但 `Node Overlaps=2119` |
| Route timing | 中途 WNS 从约 `-0.627ns` 恶化到约 `-5.683ns` |
| Congestion | Global/Short congestion level `6`，Timing congestion level `7` |
| Placed utilization | CLB `93.69%`，URAM `100%`，BRAM Tile `80.91%`，LUT as Memory `27977`，SRL `8867` |

判断：这不是“线没连上”或“总 LUT 数量不够”，而是局部物理资源争用。router 已经把 nets 全部尝试路由完，但多个 net 抢占同一批 routing node，导致 final route 不合法。此类问题不能靠继续等待或简单换 strategy 解决，必须回到 HLS 结构降低局部拥塞。

关键证据：

- top overlap 直接落在 `systolic_array_core_row_group_0/1` 的 DSP MAC 区域，包含 `VITIS_LOOP_155_3` 下的 `bits_*_srl28`、`icmp_ln97_*`、DSP A/B/P 输入路径。
- `act_stream_g0_U/empty_n_reg_0` 曾被 BUFG 驱动 `18256` loads，说明当前 SA 内部 dataflow/FIFO 控制信号扇出过大。
- `store_compact_c25_row`、`on_chip_memory_write_packed_tile`、`write_packed_cross_word`、`s_fmbuf_uram_d1`、`s_pool2_bram_d1` 仍反复出现在 phys_opt 和 overlap nets 中。
- `param_dma` 也出现在少量 overlap，但只属于 `MODE_INIT`，当前优先级低于 `MODE_RUN` 的 SA 与写回热区。

结论：上一轮修补消除了旧 C3 per-pixel 写回热点，但新的实现失败证明主瓶颈已转移为 **SA 内部 SRL/DSP 局部拥塞 + 通用 packed memory writer 克隆/扇出**。

## 3. 当前瓶颈判断

### 3.1 不是继续调频率或 Vivado 策略的问题

HLS 报告虽然 estimated clock 能进入 `8ns` 以内，但实现后 route 不合法。说明当前瓶颈不是单纯逻辑级 Fmax，而是 HLS 结构映射到器件后的物理路径：URAM 列、BRAM/FIFO、DSP/CLB 区域之间存在高扇出和局部 routing node 争用。

### 3.2 主要风险路径

| 路径 | 现象 | 判断 |
|---|---|---|
| `systolic_array_core_row_group_* -> VITIS_LOOP_155_3` | top overlap 位于 DSP MAC 区域，包含 `bits_*_srl28` 和 DSP A/B/P nets | SA 内部 SRL/FIFO/深流水寄存器与 DSP 区域局部拥塞，是下一轮最高优先级 |
| `act_stream_g*_U/empty_n_reg` 等 FIFO 控制 | 曾被 BUFG 驱动 `18256` loads | dataflow FIFO 控制信号扇出过大，需要降低 SRL FIFO 和 group 间广播控制压力 |
| `store_conv_output_row -> store_compact_c25_row -> fmbuf_uram` | route log 多次出现该路径 | compact row 写回与 256-bit URAM/BRAM 写口仍耦合过深 |
| `on_chip_memory_write_packed_tile -> write_packed_cross_word` | conv/add/affine 多条路径反复出现 | 通用 packed write 的动态 byte/lane/cross-word 逻辑在多个算子中复制并扇到全局存储 |
| `fmbuf_uram` | URAM `112/112`，多条 net 优化无改善 | 全局 feature buffer 高扇出与 URAM 列布线压力仍是根因 |
| `param_dma_init` | 少量 overlap 出现在 `MODE_INIT` 参数加载路径 | 非推理主路径，暂不作为第一优先级 |

## 4. 下一步 HLS 返修计划

### P6H-1: 先收敛 SA 内部局部拥塞

目标是在不改变 P6 `2x16x32` 计算语义的前提下，降低 `systolic_array_core_row_group_*` 附近的 SRL/DSP/控制扇出压力。

实施方向：

- 取消 `sa_core.cpp` 内部小 FIFO 强制 `impl=srl`，优先改为更可控的 BRAM FIFO 或寄存器级局部缓冲，减少 `bits_*_srl28` 进入 DSP 热区。
- 对 `systolic_array_core_row_group` 内 16-lane 计算做更明确的物理分组，降低单个 pipelined loop 的局部 fanout；保持外部接口仍是两组 16 输出 lane。
- 降低 `broadcast_act_stream` 和 group FIFO 控制信号扇出，避免 `empty_n_reg` 一类控制 net 被全阵列高扇出使用。
- 不增加第三套 SA、不改变 `TM=32/TK=32` 对外语义，避免偏离 P6 主架构。

验收：

- top overlap 不再集中在 `systolic_array_core_row_group_* / VITIS_LOOP_155_3`。
- Vivado route 不再出现 FIFO control net 超高 fanout 记录。
- HLS 综合时间保持可控，不引入新的 dataflow deadlock 或 `HLS 200-975`。

### P6H-2: 继续拆短 packed memory write 路径

目标是让高频写回路径不再直接经过通用 `on_chip_memory_write_packed_tile / write_packed_cross_word` 的完整组合逻辑。

实施方向：

- 保留 C3 avgpool 的 aligned word 写回，不回退到 per-pixel/RMW。
- 为 conv/add/affine/store 的常见 aligned full-word 写入提供更窄专用入口，减少通用 `write_packed_cross_word` 克隆。
- 对 `store_compact_c25_row`、`store_c16_into_c19_row` 这类 compact 写回继续固定 row-local 地址和 bank，避免完整 tensor descriptor 扇到每个 256-bit word 写入点。
- 必须 RMW 的路径拆成 read/modify/write 局部阶段，不在一个函数里形成 `URAM read -> mask/shift -> URAM write` 的长组合链。

验收：

- `on_chip_memory_write_packed_tile / write_packed_cross_word` 在 route log 中出现频率下降。
- HLS 中 memory writer LUT 不继续膨胀。
- 不新增 URAM，BRAM 增量可控。

### P6H-3: 控制 CLB Tile 与 LUTRAM/SRL 占用

目标是把 placed CLB 从当前约 `95%` 压到更可布线的区间。

实施方向：

- 继续保留 BRAM FIFO，不回退到 LUTRAM FIFO。
- 审查 `sa_core`、`win_gen`、`avgpool_unit`、`upsample_unit` 中的大局部数组，明确绑定到 BRAM/register，避免工具自动映射成分散 LUTRAM/SRL。
- 删除或隔离未被 P6 top 调用的 dead logic，避免综合前端和实现层继续携带无用控制网络。

验收：

- Vivado placed CLB 目标低于 `92%`，理想低于 `90%`；SRL 数量显著低于当前 `8867`。
- route congestion level 低于当前 level `6`，或至少 timing optimization 不再被 congestion 阻断。

## 5. 下一轮验证顺序

每轮 HLS 修改后按以下顺序推进：

1. `python tools/check_p6_hls_structure.py`
2. 轻量 CSim 或 prefix CSim，确认 P6 stage 没有功能退化
3. Top csynth，目标时间仍控制在 `<= 45min`
4. `python tools/audit_hls_reports.py --root D:\ESP_INT8 --max-matches 20`
5. 只有满足以下条件才进入 Vivado 实现：
   - 无 HLS blocker
   - 无硬件克隆
   - 无宽 FIFO 回退到 LUTRAM
   - HLS LUT/BRAM/URAM 没有明显恶化

## 6. 禁止重复的方向

| 禁止方向 | 原因 |
|---|---|
| 继续等待当前实现试图自然修到过线 | route 已证明可布通但 WNS 差距过大，继续等待收益低 |
| 依赖 HLS estimated Fmax 判断上板可行 | 当前已证明 HLS 7.733ns 不能代表 routed timing |
| 恢复旧 UOP dispatch / generic win_gen | 会偏离 P6 结构并放大综合搜索空间 |
| 增加第二套 SA 或 dual datapath | 当前瓶颈是写回/URAM/拥塞，不是单纯算力不足 |
| 对 reset、invalidation、低收益控制循环强行 `II=1` | 已多次导致综合时间不可控 |

## 7. 下一次工作入口

下一次直接从 `memory.cpp`、`avgpool_unit.cpp`、`conv_store.cpp` 三处进入：

1. 先重构 C3 avgpool 和 compact row 写回，拆短 URAM RMW 组合链。
2. 再检查 `fmbuf_uram` 访问入口是否可收敛，避免多个模块重复生成宽位选逻辑。
3. 最后审查 `sa_core/win_gen` 的局部 buffer 映射，目标是降低 CLB Tile 和 route congestion，而不是追求更多并行硬件。

路线判断：本轮已经证明 P6 主路径可综合、可布通；下一阶段核心目标是**把可布通版本变成 timing-clean 版本**，而不是继续做大规模架构跳变。

## 8. 2026-06-14 实现中间结果复盘

本轮在上一版基础上将 SA 内部 `weight_buf` 和 group-local FIFO 从 BRAM/SRL 进一步压向 LUTRAM/分布式实现，以降低 HLS BRAM 压力。HLS 与 Vivado OOC 综合均通过，且 NPU IP/DCP 时间戳确认实现使用的是最新 package 版本。但 Vivado implementation 的 route 中间结果显示：**该方向没有改善时序，反而把瓶颈从 BRAM 容量压力转移为 CLB Tile 与局部 routing 压力**。

补充：后续复查时 Vivado 进程已退出，`impl_1` 目录仅剩 `.vivado.begin.rst`，未保留最终 routed timing/route status 报告。因此本节结论基于实现过程中已读取到的 `runme.log`、placed utilization、`clockInfo.txt` 和 congested nets 文本，不等同于最终 routed report。

| 项目 | 当前结果 | 对比判断 |
|---|---|---|
| HLS blocker | 无 `HLS 214-475`、无 `HLS 200-975` | 可综合性仍可接受 |
| HLS top resource | BRAM `1441/1488`，LUT `452545/341280`，URAM `112/112` | BRAM 可控，但 LUT 估计仍高 |
| Vivado OOC synth | CLB LUT `68.03%`，BRAM Tile `80.91%`，URAM `100%` | BRAM 压力下降，URAM 仍满 |
| Vivado placed | CLB Tile `95.09%`，CLB LUT `66.94%`，LUT as Memory `28649`，SRL `7987` | CLB Tile 比 0612 版更高，布线余量更差 |
| Route 初始状态 | Failed Nets `1361`，Unrouted `245`，Timing congestion level `6` | 能进入 route，但拥塞仍阻断 timing optimization |
| Route 中间时序 | `WNS=-2.280ns` 后到 `-2.052ns`，TNS 约 `-1.9e4ns` | 比上一版 routed 约 `-1.8ns` 更差 |
| Route 进展 | overlap 可降到 `0` 后又进入下一轮反复迭代 | 不像完全不可布线，但 timing 收敛概率低 |

关键证据：

- `ESP_INT8_wrapper_utilization_placed.rpt` 显示 CLB Tile 已到 `40565/42660 = 95.09%`，这比 0612 失败版的 `93.69%` 更紧张；即使 LUT 总量没有爆满，几乎所有 CLB tile 被占用会显著减少 routing switch 余量。
- `clockInfo.txt` 中出现多条由 HLS 生成控制/valid 信号驱动的 BUFGCE，例如 `act_stream_g0/g1_U/dout_vld_reg_0`、`systolic_array_core_row_group_* / CEB1`、`window_generator_row` 内部状态信号。这说明 SA/window/dataflow 控制网络仍有高扇出，且被 Vivado 当成 clock-like net 处理。
- `iter_120/140_CongestedCLBsAndNets.txt` 中仍出现 `param_dma_init` 和 `s_fmbuf_uram_U/ram_reg_uram_*`，说明 MODE_INIT 不是主矛盾但仍贡献少量拥塞；MODE_RUN 的 `fmbuf_uram` 周边仍是根因之一。
- 当前 route 日志显示热点仍围绕 `PSS_ALTO -> DSP`、`DSP_X31/X39`、`CLEL/CLEM` 区域，符合“SA + URAM/fmbuf + window/control”形成局部拥塞带的判断。

结论：本轮 SA 内部 LUTRAM 化不是正确方向。它节省了 BRAM，但在我们当前器件和 P6 数据流下增加了 CLB 占用、LUTRAM/SRL 分散放置和控制网扇出，导致 route timing 比上一版更差。后续不应继续沿“用 LUTRAM 换 BRAM”的方向推进。

## 9. 下一步修补计划更新

下一轮目标仍是保持 P6 架构语义不变：单套 `win_gen -> 2x16x32 SA -> post/store` 主路径、静态 P6 调度、full-res 输出。优化重点从“压 BRAM”切换为“降低 CLB Tile 占用和控制扇出”。

### P6I-1: 回收 SA 内部 LUTRAM 化改动

目标：降低 SA 周边 CLB/LUTRAM/SRL 密度，让 route 有足够通道完成 timing optimization。

- 将 `sa_core.cpp` 内部 group-local FIFO 从 LUTRAM 优先改回 BRAM FIFO 或寄存器级浅 FIFO，避免分散 LUTRAM 占用 DSP 周边 CLB。
- 重新评估 `weight_buf` 绑定策略：不再简单 `impl=lutram`；优先使用 BRAM 或分层寄存器缓存，只接受 BRAM 增量可控的方案。
- 保留 `2x16` group 划分和 `TM=32/TK=32` 对外语义，不恢复多 datapath 或第二套 SA。

验收：

- Vivado placed CLB Tile 必须低于当前 `95.09%`，目标先回到 `<=93%`，再追求 `<90%`。
- `clockInfo.txt` 中不应继续出现大量由 `act_stream_g*_U/dout_vld` 或 group `CEB1` 驱动的高扇出 clock-like 控制网。

### P6I-2: 降低 dataflow control fanout

目标：减少 HLS stream/FIFO 控制信号在 SA/window 边界处的全局传播。

- 审查 `broadcast_act_stream`、`split_weight_stream`、`systolic_array_core_row_group_*` 的 stream 读写粒度，必要时用显式小数组/局部寄存器替代 group 内部 stream。
- 避免 group 内部再产生多级 dataflow 子图；顶层保留 dataflow，group 内部尽量顺序化和局部化。
- 不强行 `II=1` 于控制循环，避免再次触发综合时间暴增。

验收：

- HLS 综合时间仍控制在可接受范围。
- 不出现新的 `HLS 200-975` 或 dataflow read/write 同函数问题。
- Vivado route log 中 FIFO control net 不再是拥塞热点。

### P6I-3: 继续收敛 fmbuf/URAM 周边访问

目标：降低 `s_fmbuf_uram` 周边高扇出和长线穿越。

- 保持 C3 avgpool aligned word 写回，不回退到 per-pixel RMW。
- 对 `fmbuf_uram` 的读写入口继续做 row-local 地址预计算，减少完整 descriptor 与 byte-lane 控制跨模块传播。
- 如果 BRAM 有余量，优先把小型 staging buffer 放到 BRAM，而不是 LUTRAM/SRL。

验收：

- `iter_*_CongestedCLBsAndNets.txt` 不再反复出现 `s_fmbuf_uram_U/ram_reg_uram_*`。
- Route 初始 failed/unrouted nets 低于当前 `1361/245`，Timing congestion level 低于 `6` 或至少 WNS 不劣于上一版。

## 10. 当前禁止方向补充

| 禁止方向 | 原因 |
|---|---|
| 继续把 SA/FIFO/weight buffer 推向 LUTRAM | 已实测 CLB Tile 升到 `95.09%`，route timing 更差 |
| 只看 HLS BRAM 降低就推进实现 | Vivado placed/route 证明 BRAM 降低不等于可实现性改善 |
| 在 group 内部继续嵌套 dataflow stream | 容易生成高扇出 FIFO 控制网和 clock-like BUFGCE |
| 为了降低 BRAM 牺牲 CLB routing 余量 | 当前主瓶颈是 routing/timing，不是 BRAM hard overuse |

## 11. Encoder vs Fullres/P6 资源与 routing 差异复盘

从 0513 到 0527 的记录看，encoder 输出版本与 full-resolution 输出版本的**资源总量并没有呈数量级增长**，但实现难度明显变化：

| 阶段 | 代表版本 | 资源/时序观察 | 结论 |
|---|---|---|---|
| Encoder profiling | `PROFWG1 / P2E / P2F` | P2E/P2F 上板约 `39.94M` A53 timer ticks，真实约 `1198ms`，输出 `64x128`，bit-exact；早期资源紧但可实现 | 主问题是周期数，routing 尚可被工具收敛 |
| Encoder no-prof 125MHz | `PERF125-NOPROF` | 单图约 `32.99M` A53 timer ticks，真实约 `990ms`，但 routed timing 未 clean，WNS `-1.075ns` | 125MHz 已暴露 route-delay 主导问题 |
| Pair2 encoder | `P3A` | 100MHz timing clean，WNS `+0.019ns`，但延迟退到约 `46.94M` A53 timer ticks，真实约 `1408ms` | 小卷积并行增加控制/调度开销，算力提升未转化为端到端收益 |
| Fullres100 | `platform_full100_0527` | 平均约 `41.79M` A53 timer ticks，真实约 `1254ms`；LUT `73.35%`，FF `25.39%`，BRAM Tile `86.02%`，URAM `100%`，DSP `39.77%`，WNS `+0.073ns` | full-resolution 输出可 timing-clean，但余量很小 |
| Fullres 125/150MHz 尝试 | `P5` 系列 | 125MHz routed WNS 约 `-1.368ns`，失败端点约 `45011`，route delay 占比很高 | 频率失败不是 upsample 算法本身，而是既有 memory/window/URAM 路径被压到极限 |
| P6 fused/static | `P6F-P6I` | CLB Tile `94%~95%`，URAM `100%`，routing congestion level `6/7`，WNS 约 `-1.8~-2.1ns` | 结构融合减少 memory pass 的同时显著提高局部物理拥塞 |

关键判断：**资源总量相近不代表物理实现压力相近**。Vivado 的 routing/timing 更敏感于逻辑摆放位置、跨列连线、clock-enable/control fanout、URAM/DSP/BRAM 列附近局部密度，而不是只看 LUT/BRAM/DSP 百分比。

### 11.1 为什么 fullres/P6 routing 压力显著上升

1. **URAM 从 encoder 阶段就已 100%，fullres/P6 没有新的 URAM 余量。** 后续任何围绕 `s_fmbuf_uram` 的读写、row-local 地址、packed lane 选择、store/concat/fusion 控制，都只能挤在 URAM 列周边已有布线通道上。fullres 并不一定多用很多 LUT，但它把更多逻辑连接到同一个满占 URAM feature buffer。

2. **fullres 输出本身不是最差路径，但它延长了 frame/store 控制范围。** 0527 记录中已经判断：新增 upsample 不是 critical path，最紧路径仍在 `s_uram/window/memory`，route delay 占比接近 90%。因此不能把 fullres 时序恶化简单归因给 bilinear/argmax；真正问题是已有主数据流在 fullres 输出后没有物理余量继续提频。

3. **P6 static/fused 让更多逻辑同时存在于 MODE_RUN 主路径。** 旧 encoder 逐 UOP 解释虽然慢，但每次只执行一个较窄 operator datapath；P6 为了减少 memory pass，把 static stage、conv row-region、store/concat/add/avgpool 写回更紧地耦合在一起，HLS 会生成更大的同周期控制图和更多跨模块 enable/valid 信号。

4. **HLS stream/dataflow 控制网比算术资源更伤 routing。** 当前 `clockInfo.txt` 中出现 `act_stream_g0/g1_U/dout_vld_reg_0`、SA group `CEB1`、window generator 状态信号等 BUFGCE/clock-like nets，说明部分 FIFO/valid/block 控制被提升为高扇出控制网络。这类 net 即使 LUT 数不大，也会横跨多个 clock region，直接拉低 WNS。

5. **CLB Tile 接近满占是比 LUT 百分比更直接的拥塞指标。** fullres100 可 clean 时资源表给出 LUT `73.35%`、BRAM `86.02%`，但 P6 route 失败/退化时 placed CLB Tile 已到 `94%~95%`。这意味着剩余 CLB 虽然还有 LUT 逻辑容量，但 routing switch、局部连线和可放置空隙已经不足。

6. **用 LUTRAM/SRL 换 BRAM 的方向会恶化物理局部性。** 0614 实测证明 SA 内部 `weight_buf/FIFO` LUTRAM 化后，BRAM 降低但 CLB Tile 升到 `95.09%`，route WNS 比上一版更差。对当前器件而言，BRAM 约 `80%~86%` 不是第一瓶颈；CLB/routing 通道才是第一瓶颈。

### 11.2 下一步 HLS 代码改进原则

下一轮不应再把目标写成“资源百分比下降”，而应写成“降低 routing 压力”。具体代码约束如下：

| 方向 | 要做 | 不做 |
|---|---|---|
| SA 内部 buffer | 回收 LUTRAM 化，改成 BRAM 或显式小寄存器 staging；控制 `weight_buf` 与 group FIFO 的物理密度 | 不再用 LUTRAM 换 BRAM，不再在 group 内嵌套 dataflow |
| Stream/control | 将 group 内 stream 改为局部数组/寄存器握手，减少 `empty/full/dout_vld` 高扇出 | 不让 HLS 自动生成多级 FIFO control network |
| URAM/fmbuf | 入口前做 row-local 地址预计算，store/concat/add 尽量在局部 row buffer 完成后再一次性写边界 tensor | 不在高频内层传播完整 tensor descriptor 和 dynamic byte-lane |
| Window path | 保留 shape-specific emitter，减少跨 URAM/BRAM 到 window 的长组合 path；必要时牺牲少量 II 换 shorter route | 不合并回 generic win_gen，不扩大动态 shape dispatch |
| Fullres output | 保留 fullres100 已验证的 upsample/store 结构，只做轻量并行或局部缓存 | 不把 fullres path 继续和 conv/window 主路径耦合 |
| 验收指标 | 看 placed CLB Tile、LUT as Memory/SRL、clockInfo 高扇出 net、route congestion level、最终 WNS | 不只看 HLS estimated Fmax 或 BRAM/LUT 总百分比 |

### 11.3 下一版建议切入顺序

1. **先回退/重写 SA 内部 LUTRAM buffer 策略。** 目标是把 placed CLB Tile 从 `95.09%` 拉回 `<=93%`，同时确认 BRAM Tile 仍不超过 fullres100 的 `86%` 太多。
2. **去掉 SA group 内部多级 stream 化。** 将 `broadcast_act_stream/split_weight_stream/group FIFO` 的高扇出 valid/control 改为 row-region 局部寄存器/小数组传递，减少 clock-like BUFGCE。
3. **收敛 `s_fmbuf_uram` 访问入口。** 对高频 writer/read path 做 row-local `bank/base/offset` 固化，避免 `s_fmbuf_uram_U/ram_reg_uram_*` 再成为 congested net。
4. **再跑 csynth 与 implementation。** 进入 Vivado 实现的门槛改为：HLS blocker 为 0、无硬件克隆、placed CLB 预计不继续升高、结构检查通过；如果 HLS 资源仅 BRAM 小幅升高但 CLB/LUTRAM/SRL下降，应优先接受。

本节结论会覆盖前文中“压 BRAM”的旧优先级：当前首要目标不是最小化 BRAM，而是恢复 fullres100/encoder 版本更健康的物理局部性和 routing 余量。
