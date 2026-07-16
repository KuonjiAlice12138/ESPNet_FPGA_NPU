# Workplan 0713 - P7 Round 3 上板结论与后续性能收敛

## 0. 本轮结论

2026-07-13 上板版本 `INT8-BOARD-20260713-P7-PROFILING` 已完成单图整网推理。ARM 计时器自检通过，RTL stage counter 总周期与 ARM 计时一致，因此本轮性能数据可信。

Round 3 的 WinGen 服务率恢复和小输出通道双像素映射已经取得决定性收益：P7 总周期从 `170.39M` 降到 `92.66M`，并首次超过 P6J 的 `100.3M` 基线。但当前仅领先 P6J 约 `7.6%`，尚未达到 P7 `<80M` 的必达目标，后续必须转向 PPU、BLOCK5 final 和 Vec fixed，停止继续扩展 Conv 专用路径。

本轮 SD 卡 `MASK.BIN` 已使用同组 `sched_v4_p7_0702` artifact 和原始 `512x1024` 标签完成离线核验。板端 PA/mIoU 为 `0.96537738 / 0.90720853`，相对 golden fullres 的 `0.96542974 / 0.90729244` 仅下降 `0.00005236 / 0.00008392`，无非法标签。因此 Round 3 的性能与板级功能 gate 均通过。

---

## 1. 可信实测数据

### 1.1 整网结果

| 项目 | 实测值 |
|---|---:|
| PL 时钟 | `100 MHz` |
| ARM timer | `33.333 MHz`，自检 PASS |
| RTL `MODE_RUN` window | `92,656,373 cycles` |
| ARM timer | `30,885,446 ticks` |
| 实测延时 | `927 ms` |
| PARAM | `143,424 bytes`，75 UOP / 16 exec / 26 conv |
| `MASK.BIN` | `524,288 bytes`，SHA256 `ADFFB9CD...DC63F84` |

RTL 的 `92,656,373 cycles @ 100MHz` 对应 `926.56 ms`，与 ARM 输出的 `927 ms` 一致，计时误差小于 1 ms。

### 1.2 板端 mask 精度

| 指标 | Golden fullres | Board `MASK.BIN` | Board - Golden |
|---|---:|---:|---:|
| PA | `0.96542974` | `0.96537738` | `-0.00005236` |
| mIoU | `0.90729244` | `0.90720853` | `-0.00008392` |

- 板端 mask 无非法标签，取值仅为 0/1。
- 在 `248,277` 个有效标注像素中，相对 golden 有 `1,985` 个像素不同，差异率约 `0.800%`；这些边界像素差异没有造成显著指标下降。
- 评估记录保存在 `hw_artifacts/sched_v4_p7_0702/board_20260713_mask_metrics.json`。

### 1.3 Stage 构成

| Stage | Cycles | 占比 |
|---|---:|---:|
| `CONV_ROW_DATAPATH` | `33,187,072` | `35.81%` |
| `PPU_ROW_CONSUME` | `34,999,712` | `37.77%` |
| `PPU_BLOCK5_FINAL` | `10,867,630` | `11.72%` |
| `VEC_FIXED` | `10,126,598` | `10.92%` |
| `AVGPOOL` | `3,048,809` | `3.29%` |
| `FRAME_LOAD` | `382,757` | `0.41%` |
| 其余控制/权重阶段 | `43,795` | `<0.1%` |

当前最大单项瓶颈已由 Conv 转为 `PPU_ROW_CONSUME`。PPU row、BLOCK5 final 和 Vec fixed 合计 `55,993,940 cycles`，占整网 `60.43%`；再加 AvgPool 后，非 Conv 核心路径占 `63.72%`。

### 1.4 与历史基线比较

| 版本 | Total cycles | 延时 | 相对结论 |
|---|---:|---:|---|
| P7 Round 1 | `170,387,445` | `1704 ms` | 路线 2 起点 |
| P6J | 约 `100.3M` | `1003 ms` | 旧版必须超过的基线 |
| P7 Round 3 | `92,656,373` | `927 ms` | 当前版本 |

Round 1 到 Round 3：

- 总周期减少 `77,731,072`，下降 `45.62%`，吞吐提升约 `1.84x`。
- Conv 从 `110,917,888` 降至 `33,187,072`，下降 `70.08%`，提升约 `3.34x`。
- PPU row、BLOCK5 final、Vec fixed 和 AvgPool 周期与 Round 1 基本不变。

这说明本轮全部主要收益来自 Conv，Round 2/3 的 WinGen cache、scheduled issue 和 pixel-parallel 路径有效；后处理尚未分享该收益，已经成为新的 Amdahl 瓶颈。

与 P6J 相比，当前总周期约减少 `7.6%`，延时降低约 `76 ms`。这证明 P7 架构不再是负优化，但仍不足以宣告性能目标完成。

---

## 2. Round 3 验收决定

`workplan_0707.md` 的 Round 3 门槛如下：

| Gate | 要求 | 当前结果 | 状态 |
|---|---:|---:|---|
| Conv cycles | `<=55M` | `33.19M` | PASS |
| Total cycles | `<=110M` | `92.66M` | PASS |
| 单实例 SA/WinGen | clone count = 1 | 最新 csynth 符合 | PASS |
| DSP | 不高于 Round 2 | `1251`，变化可控 | PASS |
| 板级功能 | mask 精度不明显下降 | PA/mIoU 仅下降 `5.24e-5 / 8.39e-5` | PASS |

决定：冻结 Round 2/3 的 Conv 数据流，不再增加 WinGen mode、第二套 cache、第二套 SA 或新的像素并行分支。后续修改必须满足：

```text
CONV_ROW_DATAPATH <= 34.2M cycles（相对当前最多回退 3%）
scheduled WinGen = 1
SA = 1
weight buffer owner = 1
DSP/URAM 不增加
```

最新综合中 `C131` 路径仍可能出现 `II=2`，但 Conv 已远低于 Round 3 gate。除非后续板级 PC 数据证明该路径重新成为最大热点，否则不再为追求局部 `II=1` 承担复制、BRAM 端口或 routing 风险。

---

## 3. 对 Round 4-6 的调整

原 Round 4 的总周期 gate 为 `85M~95M`，当前版本已经自然落入该区间，无法再约束真实优化。因此 Round 4 必须拆分为低风险的 word-stream 收敛和有条件的 overlap 两步，并收紧总周期门槛。

### Round 4A：PPU packed-word 主路径

本轮先不引入 DATAFLOW overlap，只消除 PPU 内部的 bytewise repack、重复地址计算和不必要 RMW。

代码与 csynth 已给出直接证据：

```text
ppu_store_compact_row             max 102,431 cycles/row
ppu_store_block5_scratch_row      max 102,431 cycles/row
ppu_add_other_row_to_buffer       max 24,602 cycles/row
ppu_consume_upsample_out          max 23,982 cycles/row
ppu_apply_row_affine              max 14,361 cycles/row
```

`ppu.cpp` 的 add/cat/affine/store 热循环仍存在 `PIPELINE off`；`ppu_store_compact_row_core()` 仍按像素和通道逐 byte 拼接，再调用 compact-word writer。该结构与 `PPU_ROW_CONSUME=35.0M` 的板级结果一致，Round 4A 应优先修 store 服务率，而不是先改算术并行度。

实施顺序：

1. 以 PPU row buffer 中已有的 `act_vec_t` 为输入，按静态 layout 直接生成完整 256-bit destination word。
2. 为 C3/C12/C19/C25/C28 使用固定 pixel/lane 映射，禁止动态 shift、动态 lane destination 和逐 byte word accumulation。
3. add、affine、add-affine、cat-affine 只保留算术前端，统一进入一个 compact-word store tail。
4. qparam 在 row/tile 入口加载到局部窄结构，禁止在 lane hot loop 重复访问全局 descriptor。
5. 完整覆盖的 destination word 直接写；仅 PARAM 明确标记的边界 partial word允许 mask merge。
6. BLOCK5 scratch store 复用同一 packed writer，但不得恢复 standalone finalizer 或 generic concat writer。

Round 4A gate：

```text
PPU_ROW_CONSUME <= 20M cycles，最终目标 <=15M
FULL_MODE_RUN <= 78M cycles
CONV_ROW_DATAPATH <= 34.2M cycles
不增加新的 PARAM runtime shape 推断
无 store/affine/add datapath clone
100MHz implementation 可布线
```

若 PPU 达到原定 `15M`，在其他 stage 不变时整网理论周期约为 `72.66M`，已满足 `<80M` 必达目标。

### Round 4A-R：routing 恢复轮（Round 3 后的强制实现门槛）

2026-07-14 的 Round 4A 实现未通过 routing：`1062` 条信号无法布通、最终仍有 `670` 个 node overlap。该失败不能通过继续等待 router 或单纯切换 directive 解决，必须先回到 HLS 源码降低局部连接密度。

#### 失败证据与根因判断

| 证据 | 结果 | 判断 |
|---|---:|---|
| HLS 资源相对 Round 3 | LUT `+6,706`、FF `+6,435`、DSP `+3` | Round 4A packer 是本轮新增压力 |
| placed LUT | logic `239,872`，memory `17,496`，总计 `75.41%` | 不是全芯片 LUT 容量不足 |
| placed memory | BRAM `702/744=94.35%`，URAM `112/112=100%` | 存储列附近可用布线通道非常紧张 |
| control sets | `2,026` | 高局部密度下进一步限制 CLB packing |
| route congestion | global/short level `6`，timing level `7` | 已超过稳定可布线范围 |
| timing | post-place WNS `+0.438ns`，route 中间 WNS `-2.732ns` | 负时序主要由拥塞绕行产生，不是 HLS 组合路径先失控 |

route top overlap 直接包含以下网络：

1. `scheduled_narrow_3x3` 的三行 cache 与 FMBUF BRAM/URAM；
2. 新增 `ppu_pack_compact_word_28` 的 256-bit `out_word` 寄存器；
3. `run_b2_c131_row_contiguous_backup_op` 的 Vec 状态/数据；
4. shared weight buffer 与 SA/WinGen 控制网络。

因此根因是：原本已接近 routing 边界的 WinGen/cache、FMBUF、Vec 区域，被 Round 4A 新增的五套 layout-specialized packer 和跨模块 256-bit 返回总线进一步挤压。当前 `ppu_store_compact_row_core` 同时常驻 C12/C16/C19/C25/C28 五个 packer，共约 `10,534 LUT / 7,617 FF / 7 DSP`；其中 C28 packer已出现在最严重 overlap 列表。与此同时，`ppu_add_other_row_to_buffer` 在普通 row consumer 与 BLOCK5 final 下各实例化一次，每套约 `12,810 LUT / 2,794 FF / 11 DSP`。这两类重复逻辑是优先修复对象。

#### 修改步骤

1. **冻结已验证的 Conv 路径。** 不修改 scheduled narrow WinGen、SA 维度、weight buffer、pixel-parallel 映射和 cache 容量；保持 `CONV_ROW_DATAPATH<=34.2M` 的回归门槛。
2. **将 row add 提升为单一物理 owner。** 在 `conv_engine.cpp` 的普通 consumer/BLOCK5 final 分支之前统一选择 add tensor、row、qparam 和 enable，只调用一次 PPU pre-add gateway；删除 `ppu_consume_conv_row()` 与 `ppu_consume_block5_final_row()` 内部各自的 add call site。普通 add 与 BLOCK5 chain add 在时间上互斥，因此该共享不增加执行周期，也不得引入覆盖完整 PPU 的 runtime mega-switch。
3. **把五套 packer 收敛为一个 4-byte/group 的 compact pack owner。** layout 只在 word 入口选择一次静态 phase table；hot loop 保持 8 个 group、每组 4 byte 并行，禁止运行时除法/取模、动态 destination lane 和逐 byte RMW。C12/C16/C19/C25/C28 只保留映射常量，不再形成五个独立 HLS module。
4. **消除 256-bit 跨模块返回总线。** pack、word-valid 处理和 compact write 放在同一 store owner 内；`out_word` 只作为本地寄存器，不再从 `ppu_pack_compact_word_*` 跨层返回到 common writer。保持完整 destination word 直接写，边界 partial word规则不变。
5. **统一 pool2 absolute read gateway。** 规范 `on_chip_memory_read_pool2_abs_word()` 的唯一类型签名和唯一 PPU wrapper，消除 HLS 对不同调用上下文的 helper duplication；不合并 FMBUF/POOL2 物理存储，不改变 memory map。
6. **局部化控制。** layout、row base、row bytes、valid/phys channel 在 row 入口转成窄本地值；禁止把完整 descriptor、五路 packer enable 或 256-bit word 广播到多个 mode-specific engine。
7. **清理和验证。** 删除被替代的模板 packer、重复 cursor helper、旧 add call site及不可达 wrapper；依次执行 structure check、prefix-lite、U40 和 lowres CSim，再进入 csynth。

#### 综合与实现 gate

```text
ppu_add_other_row_to_buffer physical instance = 1
compact pack/store physical owner = 1
旧 ppu_pack_compact_word_12/16/19/25/28 module = 0
HLS helper duplication warning = 0
ppu_store_compact_layout_row max latency <= 26,659 cycles/row（允许 +3% 工具波动）
ppu_add_other_row_to_buffer max latency <= 24,602 cycles/row
top LUT <= Round 3 的 431,280，至少必须完全收回 Round 4A 的 +6,706 LUT
top FF <= Round 3 的 264,200，BRAM/URAM/DSP 不增加
estimated timing <= 7.30ns
implementation global congestion <= level 4
route overlap = 0，100MHz timing pass
```

不得采用以下方式换取表面通过：降低 SA/TM/TK、关闭 WinGen cache、恢复 bytewise store、增加中间 frame-memory round trip、使用 allocation/Tcl directive 掩盖源码多 call site，或用更激进的 Vivado directive 代替源码修复。

#### 是否在上板前继续完成 Round 5

Round 5 的 BLOCK5 final/Vec common packed tail从方向上能够同时降低周期和重复逻辑，理论上有助于 routing；但不能在未审计的情况下直接叠加到当前失败 netlist。执行策略如下：

1. 先完成 Round 4A-R，并单独得到一份通过上述 gate 的 csynth 报告，不立即运行 Vivado implementation。
2. 若 Round 5 保持单一 add/affine/store owner，且其 csynth 相对 4A-R **LUT、FF、control-set proxy 不增长，clone 数不增长，BLOCK5/Vec latency下降**，则合并 Round 5 后只跑一次 implementation 和上板。这是优先方案，可避免重复导出平台。
3. 若 Round 5 引入新的 mode-specific engine、runtime task mux、BRAM 端口复制，或 LUT/FF 任一增长超过 `3%`，立即停止叠加，先对 4A-R 做 implementation。不能以“后续可能优化回来”为理由再次把不可布线风险带入实现。

因此，**做完资源负增长版本的 Round 5 再上板是合理的，但必须以 4A-R 独立 csynth checkpoint 为前提**。Round 5 不是本次 routing 失败的替代修复，也不能绕过 4A-R 的物理单 owner gate。

### Round 4B：受控 row ping-pong overlap

仅当 Round 4A 后 `PPU_ROW_CONSUME>15M` 或总周期仍高于 `75M` 时进入本轮。

1. 使用两个固定 BRAM row buffer；Conv producer 和 PPU consumer必须具有明确、互斥的 buffer owner。
2. 固定执行 `produce(row r+1)` 与 `consume(row r)`，不得形成 runtime task mux。
3. 初版只覆盖普通 Conv row；BLOCK5 final、fused upsample 和 fixed Vec 不进入该 DATAFLOW region。
4. producer/consumer 若需要同时访问同一 URAM owner，则保持串行，不用 pragma 强迫 overlap。
5. overlap 后 stage 分桶可能不再具有简单可加性，性能验收以 `RTL_STAGE_TOTAL` 和 Conv regression 为准。

出现 `HLS 214-475`、新的 `HLS 200-1449` 扩散、FIFO 深度异常、函数 clone、综合搜索爆炸或 routing 资源显著增加时，立即撤销 Round 4B，保留 Round 4A。

Round 4B gate：

```text
FULL_MODE_RUN <= 75M cycles
CONV_ROW_DATAPATH 不回退超过 3%
单实例 Conv/WinGen/SA/PPU owner
无共享 URAM feedback DATAFLOW
```

### Round 5：BLOCK5 final 与 Vec fixed

Round 5 仍有必要，但排在 PPU packed-word 收敛之后。当前两项合计 `20,994,228 cycles`，只优化这两项到 `10M` 时，总周期约为 `81.66M`，单独不足以达到 `<80M`；与 Round 4A 同时达标时，理论总周期约为 `61.66M`。

1. 保留静态 L2/L3 composer，清除 runtime segment loop 和动态 destination lane。
2. BLOCK5 add/affine/final store 汇入唯一 common tail；源码级修复多个 final emit call site，不使用 allocation directive 掩盖复制。
3. Vec fixed 按 packed word 读写，affine-only 与 add-affine 共享 lane arithmetic 和 store owner。
4. qparam 在 row/block 入口缓存，避免 descriptor 到所有 lane 的高扇出。
5. 不新增 U69 专用 full-pass，除非 RTL PC 数据证明单一 fixed family 占 Vec 大多数周期。

Round 5 gate：

```text
PPU_BLOCK5_FINAL + VEC_FIXED <= 10M cycles
FULL_MODE_RUN <= 65M cycles
LUT/BRAM/control-set 不恶化到无法 placement/routing
```

### Round 6：最终闭环

Round 6 不再做架构扩展，只进行最终功能、综合、实现和上板闭环：

1. exporter/replay/contracts 与 PARAM ABI hash 一致。
2. dead/legacy scan；无旧 PPU repack、旧 BLOCK5 composer 和重复 owner。
3. prefix/U40/lowres 后执行最终 fullres CSim。
4. csynth audit 检查 clone、II、资源和高扇出。
5. 100MHz implementation 必须合法布线并满足时序。
6. 上板核对 RTL total、stage cycles 和 `MASK.BIN` PA/mIoU。

最终 gate 保持 `<80M` 必达、`<50M` 冲刺。根据当前 Amdahl 构成，`50M` 不能仅依靠 Conv 继续优化，必须同时实现 PPU word-stream、BLOCK5/Vec common tail，并在资源允许时获得安全的 row overlap。

---

## 4. 验证频率与停止规则

为避免重复付出 fullres、综合和实现时间：

| 节点 | 验证 |
|---|---|
| Round 4A | contracts、结构检查、prefix/U40/lowres、csynth、实现与上板 stage counter |
| Round 4A-R | structure check、prefix/U40/lowres、独立 csynth；通过资源 gate 后决定合并 Round 5 或单独实现 |
| Round 4B | 仅在触发条件满足时执行；prefix/U40/lowres、csynth、实现与上板 |
| Round 5 | contracts、结构检查、prefix/U40/lowres、csynth |
| Round 6 | 最终 fullres CSim、csynth、implementation、上板精度与性能 |

任一轮出现下列情况立即停止扩展并回退本轮：

1. Conv/WinGen/SA/PPU/store owner 被复制。
2. 热路径重新出现 runtime shape 推断、dynamic lane mux 或 runtime segment loop。
3. 综合长时间停在动态数组、巨大 switch 或共享存储 DATAFLOW 调度。
4. LUT/BRAM/control-set 增量无法由板级 cycle 收益解释。
5. 目标 stage 下降但 `RTL_STAGE_TOTAL` 无下降，或 Conv 回退超过 3%。
6. implementation 发生 placement/routing failure 或 100MHz 时序不收敛。

---

## 5. 下一步执行顺序

1. 冻结当前 WinGen/SA/pixel-parallel 路径和 `33.19M` Conv regression baseline。
2. 执行 Round 4A-R，优先收敛 add、compact pack/store 和 pool2 read gateway 的物理 owner，不引入 ping-pong。
3. 完成 prefix/U40/lowres 与独立 csynth，核对 clone、latency、LUT/FF/BRAM/URAM/DSP gate。
4. 若 4A-R 通过且 Round 5 能继续实现资源负增长，则完成 Round 5 后统一实现、导出和上板；否则先实现 4A-R，不能继续叠加风险。
5. 仅在 packed store与 Round 5 实测后总周期仍高于 `75M` 时考虑 Round 4B overlap。

当前最重要的判断是：P7 scheduled WinGen 与双像素映射已经把 Conv 从绝对瓶颈降为第二大阶段。下一步若仍围绕 Conv 做专用化，会增加实现风险却无法达到 `<80M`；只有把 PPU row 的逐 byte 存储转换为稳定 packed-word 服务率，P7 的 block/row-level 架构收益才能真正反映到整网延时。

---

## 6. 2026-07-16 Round 5 上板验收

上板版本为 `INT8-BOARD-20260716-P7-R5`，对应 `platform_p7_0716`。ARM 计时器自检通过；RTL stage counter 为 `71,427,285 cycles`，ARM 计时为 `23,809,083 ticks / 714 ms`，两者在 100 MHz PL 与 33.333 MHz ARM timer 下相互吻合。`MODE_INIT`、整网 `MODE_RUN` 和 `MASK.BIN` 写回均正常完成。

| Stage | Cycles | 占比 | 相对 Round 3 |
|---|---:|---:|---:|
| `CONV_ROW_DATAPATH` | `33,187,072` | `46.46%` | 基本不变 |
| `PPU_ROW_CONSUME` | `11,118,816` | `15.56%` | `-68.23%` |
| `PPU_BLOCK5_FINAL` | `12,330,030` | `17.26%` | `+13.46%` |
| `VEC_FIXED` | `11,316,007` | `15.84%` | `+11.75%` |
| `AVGPOOL` | `3,048,809` | `4.26%` | 基本不变 |
| `FRAME_LOAD` | `382,744` | `0.53%` | 基本不变 |

与 Round 3 的 `92,656,373 cycles / 927 ms` 相比，Round 5 减少 `21,229,088 cycles`，总延时下降约 `22.9%`；与 P6J 的约 `100.3M cycles / 1003 ms` 相比，延时下降约 `28.8%`。本版已经通过 P7 `<80M cycles` 必达 gate，但尚未达到 `<50M cycles` 冲刺目标。

本轮收益明确来自 compact packed-word PPU row 主路径，说明 Round 4A-R/Round 5 的物理 owner 收敛没有抵消数据流性能。下一步冻结 `33.19M` Conv 和 `11.12M` PPU row 基线，不再修改已验证的 WinGen、SA、pixel-parallel 和 compact store 路径。后续性能工作只针对合计 `23.65M cycles` 的 BLOCK5 final 与 Vec fixed，优先共享 packed arithmetic/store tail 和局部化 qparam；任何修改均不得使 Conv 或 PPU row 回退超过 `3%`，也不得重新引入多 owner、动态 lane mux 或不可布线结构。

本次串口仅确认 `MASK.BIN` 成功生成，精度结论仍沿用前次已验证数据；若该文件用于论文最终结果，应另做一次离线 PA/mIoU 核验并记录哈希。

提交前源码结构检查通过，确认 MainCtrl 单一 `execute_issue` 调用点、单一 Vec arithmetic/PPU emit tail，且未恢复额外 logits memory pass。`tools/test_p7_contracts.py` 的 `prefix_replay_contract` 当前仍有 `441,181 / 2,097,152` mismatch（最大绝对差 `19`）；该项反映当前 fixture 中 PARAM/input 与 QAT golden 的回放对齐缺口，不作为本次已实测板级性能结论的通过项，后续重新生成统一 artifact/golden 后必须复核。
