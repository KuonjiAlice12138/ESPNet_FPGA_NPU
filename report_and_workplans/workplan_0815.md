# Workplan 0815 - P7 mixed-precision routability recovery

## 0. 本轮目标

本轮只解决 2026-08-15 最新实现的全局路由拥塞，不改变 P7 的计算架构和性能主路径：

- 保持唯一 WinGen、唯一 `32 x 32` SA、唯一 PPU/Vec/Pool owner；
- 不降低 `TM/TK`，不放宽关键 `II=1`，不新增第二套 INT8/INT4 datapath；
- 不恢复 current-conv-output 的 FMBUF 往返；
- 只依据 PARAM v4 exec plan 和真实地址生命周期缩减片上 FMBUF；
- 本轮结束后交由 GUI 手动运行顶层 csynth，再根据报告决定是否进入实现。

## 1. 最新实现结果

失败 run 为 `D:\ESP_INT8_vp\ESP_INT8_vp.runs\impl_1`，`runme.log` 更新时间为
2026-08-15 17:41。诊断报告保存在 `route_diag_0815/`。

| 项目 | 最新结果 | 判断 |
|---|---:|---|
| post-place WNS | `+0.507 ns` | 布局后逻辑时序可满足 10 ns |
| post-physopt WNS | `+0.277 ns` | route 前仍为正裕量 |
| global/short congestion | level 6 | 不可绕线 |
| timing congestion | level 7 | 局部硬列压力严重 |
| route 状态 | `484,783` routable nets 尚未路由 | router 在初始阶段终止 |
| CLB LUT | `253,188 / 341,280 = 74.19%` | 原始 LUT 比例不是唯一根因 |
| CLB used | `41,877 / 42,660 = 98.16%` | 打包与局部放置空间接近耗尽 |
| control sets | `2,248` | 加剧 CLB 打包压力 |
| BRAM tile | `716 / 744 = 96.24%` | 当前最直接的硬资源瓶颈 |
| URAM | `112 / 112 = 100%` | 无任何硬列余量 |
| DSP | `1,369 / 3,528 = 38.80%` | 不是容量瓶颈 |

相对 `workplan_0805` 第 14 节记录的上一轮失败，placed LUT 已减少约 `16,393`
（约 `6.1%`），CLB occupancy 从 `99.49%` 降至 `98.16%`；但 BRAM 从
`696` 增至 `716`。因此上一轮源码修复有效，但收益被更高的 BRAM 硬列占用抵消。

`report_design_analysis -congestion` 的 level-6 窗口同时达到 `RAMB=100%`、
`URAM=100%`，热点集中在 WinGen assembler/loader、FMBUF BRAM、PPU BLOCK5 finalizer
以及 SA 周边。该证据说明当前失败首先是硬存储列与其邻近路由资源耗尽，不应继续盲目拆
计算 datapath。

## 2. HLS 存储根因

当前 HLS FMBUF 为：

```text
FMBUF_BYTES            = 0x598000
FMBUF_URAM_BYTES       = 0x380000
FMBUF_BRAM_BYTES       = 0x218000
FMBUF_BRAM_AXI_WORDS   = 68608 x 256 bit
```

综合/实现中该数组对应约 `477 RAMB36`，占整个设计 BRAM 的大部分。高地址尾部的
`0x54A000..0x598000` 并非 persistent tensor，而是由 L2 BLOCK5 的 64-row 暂存上界
保留下来的容量。

PARAM v4 release exec plan 已把四个 BLOCK5 固化为 row-group issue。活动路径只直接使用
`LS_C1`：

- L2 `LS_C1`：`0x418000..0x498000`；
- L2 BLOCK5 compact row scratch（64 rows）：`0x498000..0x598000`；
- L30 persistent compact tensor：`0x418000..0x518000`；
- L30 `LS_C1`：`0x518000..0x54A000`。

旧逻辑 uop 中的 `LS_A/LS_B/LS_TMP` 已由 BLOCK5 exec entry 吸收，不应再作为 release
exec 的直接 tensor operand。`scratch_mgr.cpp` 仍为四个旧 scratch slot 保留数组、索引和
动态地址生成，这既阻止 FMBUF 缩小，也留下不再属于 P7 主路径的通用 mux。

## 3. 本轮方案：32-row BLOCK5 + 单 C1 scratch

### 3.1 新内存图

将 `BLOCK5_ROW_BLOCK_ROWS` 从 `64` 改为 `32`：

```text
0x418000..0x498000  L2 LS_C1 / B2 backup（按阶段复用）
0x498000..0x518000  L2 BLOCK5 32-row compact scratch
0x418000..0x518000  L30 persistent compact tensor（不同阶段）
0x518000..0x54A000  L30 LS_C1
0x54A000             新 FMBUF 结束地址
```

新容量为：

```text
FMBUF_BYTES            = 0x54A000
FMBUF_BRAM_BYTES       = 0x1CA000
FMBUF_BRAM_AXI_WORDS   = 58624 x 256 bit
```

相对当前减少 `9,984` 个 256-bit word，即 `319,488 B`，理论释放约 `69` 个
RAMB36 tile。若其他模块不变，placed BRAM 预计从 `716` 降至约 `647`，即约 `87%`；
HLS BRAM18 预计从 `1416` 降至约 `1278`。

### 3.2 性能边界

row-group 变小不改变卷积 row、WinGen word、SA step、PPU arithmetic 或 FMBUF 有效数据量。
它只增加 BLOCK5 group 边界和权重重载：

- 两个 L2 block：每个由 2 groups 增至 4 groups；
- 两个 L3 block：每个由 1 group 增至 2 groups；
- 总计增加 30 次 branch issue；
- 依据板级 `CONV_WEIGHT_LOAD` 计数，新增权重/控制开销保守估计 `<33k cycles`，小于
  当前约 51M cycles 的 `0.07%`。

本轮不得以降低 SA 吞吐或增加 FMBUF 数据搬运换取 BRAM。

## 4. 代码改进步骤

### 4.1 先建立失败 contract

在 `tools/test_p7_contracts.py` 中增加以下门槛，并先确认旧代码会失败：

1. exporter 与 HLS 常量必须一致：`row_group_h=32`、`FMBUF_BYTES=0x54A000`；
2. release exec plan 中，非 BLOCK5 活动 entry 不得直接引用 `LS_A/LS_B/LS_TMP`；
3. 四个 BLOCK5 schedule 均为 32-row，并验证各 pattern 的最大 scratch end；
4. 所有 persistent/transient region 的最高地址不得超过新 FMBUF end；
5. `scratch_mgr.cpp` 不得再保留四槽 `s_scratch_desc[4]` 或 LS_A/B/TMP case。

### 4.2 同步 compiler/exporter ABI

1. 在 `export_int8_hw_blob.py` 定义单一 `BLOCK5_ROW_GROUP_H=32`，禁止散落 magic number；
2. 输出 BLOCK5 schedule 时使用该常量；
3. 扩展 memory lifetime audit，记录 FMBUF 容量、BRAM words、BLOCK5 scratch end、L30 C1
   end以及 active scratch operand gate；
4. PARAM v4 ABI struct 大小和 section layout 不变，只有 schedule 字段值变化；
5. 后续重新导出 binary2/city20 artifact，旧 `row_group_h=64` 的 PARAM 不得与新 IP 上板混用。

### 4.3 收缩 HLS memory owner

1. `npu_config.hpp` 同步改为 32-row 和 `FMBUF_BYTES=0x54A000`；
2. 删除“L2 三个完整 legacy slot 必须同时存在”的旧 static assert，改成 release P7 的精确
   C1/BLOCK5/L30 边界断言；
3. `memory.cpp` 仍只保留一套 URAM owner 和一套 BRAM owner；仅通过数组深度自然减少
   RAMB，不做 4-bank，不新增 memory port；
4. `main_ctrl.cpp` 继续严格检查 PARAM 的 `row_group_h` 与 HLS 常量一致，防止旧 artifact
   静默运行。

### 4.4 清理 scratch manager

1. 保留四个 logical scratch ID 的 ISA 定义，但 synthesized P7 resolver 只接受 `LS_C1`；
2. 将四元素 scratch desc/valid 数组收敛为单一 C1 owner；
3. 删除未启用的 channel-view 状态、LS_A/B/TMP slot index 和 ADD/STORE legacy region decode；
4. BLOCK5 branch scratch 仍由 `resolve_block5_scratch_descs()` 的 compact descriptors 管理，
   不与 C1 owner 混合；
5. 非法/旧 PARAM 最终返回 range/bank error，不允许访问缩小后的数组边界。

## 5. 验证与进入综合门槛

本轮修改后按以下顺序验证：

1. targeted P7 memory contract；
2. `tools/check_p6_hls_structure.py` 与 dead/legacy symbol scan；
3. exporter PARAM v4 parse/strict schedule contract；
4. C++ 前端/轻量 CSim 仅在现有 artifact 已重导出时运行；旧 64-row PARAM 预期被拒绝，
   不能把这种拒绝误判为计算错误；
5. 人工检查 diff 后交给 GUI 手动 csynth。

csynth pass gate：

- `s_fmbuf_bram` 深度为 `58624 x 256 bit`；
- top BRAM18 目标 `<=1300`，且必须低于当前 `1416`；
- LUT/FF/DSP 不增长超过 `1%`，URAM 保持 `112`；
- SA/WinGen/PPU/Vec/Pool clone count 仍为 1；
- 关键 K/row loop II 不回退，目标时钟仍为 10 ns；
- 不出现新的 HLS 搜索时间爆炸或 DATAFLOW warning。

implementation pass gate 沿用 `workplan_0805`：不得出现 congestion level 6，必须
`failed nets=0`、`node overlaps=0`、`WNS>=0`、`WHS>=0`。若 BRAM 已按预期下降但仍不可
绕线，下一轮才处理 WinGen local-cache/控制集打包，不再继续猜测式缩存储。

## 6. 本轮实施结果（2026-08-15）

已完成：

- HLS `FMBUF_BYTES=0x54A000`，`s_fmbuf_bram` 目标深度为 `58624 x 256 bit`；
- BLOCK5 schedule 统一改为 32-row group，PARAM v4 section ABI/大小不变；
- `scratch_mgr.cpp` 已收敛为单一 `LS_C1` scratch owner，四槽 desc/valid、动态 slot 状态、
  LS_A/B/TMP resolver 和旧 ADD/STORE region decode 已退出综合路径；
- binary2 与 cityscapes20 release artifact 已重新导出，二者 memory lifetime audit 均满足：
  `required_end == capacity == 0x54A000`、active scratch 仅为 `LS_C1`、所有 region 均不越界；
- exporter 增加严格 row-group、active scratch operand 和物理生命周期检查，旧 64-row PARAM
  将被 HLS/解析器拒绝，不能与本轮 IP 混用。
- replay 不再复制 FMBUF 容量常量，直接复用 exporter 的内存图定义，避免离线行为继续按
  `0x598000` 接受硬件已越界的地址。

验证结果：

| 检查 | 结果 |
|---|---|
| targeted FMBUF compaction contract | 旧常量下按预期失败；修复后通过 |
| P7 Python contracts | 所有断言均打印 `[PASS]`；runner 在最终成功打印后的退出清理阶段触发外层超时 |
| P7 structure checker | PASS；`execute_issue` 单调用点、PPU/Vec 单 arithmetic tail 保持 |
| strict exporter + PARAM v4 parse | binary2/city20 均通过，`0 warnings` |
| U40 prefix CSim | `0 errors`，总耗时 `30m03s` |
| U40 dumps | U39 `4292608 B`；U40 rowbuf/C1 各 `204800 B` |
| max HLS stream depth | `4736`，与既有 P7 TB 记录一致 |

当前已满足进入手动顶层 csynth 的源码门槛。综合后必须先核对 `s_fmbuf_bram=58624 x 256`
和 top BRAM18 是否降至 `<=1300`；若深度或 BRAM 未下降，说明新源码/新 IP 未被本次综合实际
采用，不应继续 package/implementation。

## 7. 2026-08-16 routed-error DCP 复盘与计划调整

### 7.1 本轮实现结果

最新失败存档为：

```text
D:\ESP_INT8_vp\ESP_INT8_vp.runs\impl_1\ESP_INT8_wrapper_routed_error.dcp
```

基于该 DCP 重新生成的报告位于 `route_diag_0816/`。本轮不是初始 route 立即退出，router
持续约 4 小时后仍有 `17,671` 个冲突 net 和 `19,156` 个 node overlap，最终 WNS 为
`-4.693 ns`。与 8 月 15 日失败相比：

| 项目 | 0815 | 0816 | 结论 |
|---|---:|---:|---|
| BRAM tile | `716/744 = 96.24%` | `652/744 = 87.63%` | 32-row/FMBUF 压缩完全生效，释放 64 tile |
| URAM | `112/112` | `112/112` | 仍无硬列余量 |
| CLB LUT | `74.19%` | `74.25%` | 逻辑总量基本不变 |
| CLB used | `98.16%` | `98.33%` | 打包空间仍接近耗尽 |
| control sets | `2248` | `2228` | 略降，但不足以恢复可绕线性 |
| route errors | 初始退出 | `17671` nets | 存储修复让 route 明显向后推进，但未收敛 |

因此第 3 至 6 节的存储方案不回退；它解决了已确认的 BRAM 根因，但不是充分条件。后续不再
继续猜测式缩小 FMBUF，而应进入原计划注明的 WinGen/控制局部化修复。

### 7.2 新的物理根因

`report_design_analysis -congestion` 显示 North/South long congestion 仍为 level 6/7；热点
窗口通常同时达到 `RAMB=99~100%`、`URAM=100%`、`LUT=77~82%`。层级和冲突 net 指向：

1. `window_row_assembler` 及 C28/C64/C128/C131 四套 wide-emitter pipeline；四套控制器围绕
   唯一 `s_wide_cache` 形成大范围 enable、mode mux 和 cache-read 网络。
2. `scheduled_1x1_window_row/c_begin_reg` 到 `s_fmbuf_uram ADDR_A` 是最差建立时间路径之一；
   典型路径约 `13.9 ns`，其中 route delay 超过 `92%`，不是算术组合逻辑过深。
3. SA 的 `dual_i4_group` 与 logical-lane 控制仍有约 `8.9k~9.3k` fanout；现有 4 组 x 8 lane
   的复制粒度不够局部。
4. 失败并非 SA 算力不足：DSP 仅 `38.58%`，关键 K/row loop 的综合 II 仍为 1。降 TM/TK
   或放宽 II 会直接牺牲性能，禁止作为本轮修复。

### 7.3 调整后的 0816A 修复

本轮只改变控制和地址生成拓扑，不改变任何 work/token 数：

1. **单 wide-emitter loop**：`emit_wide_words_for_mode()` 只保留一个上界 37、`II=1` 的
   发射循环；mode 只选择静态 C28/C64/C128/C131 word builder，不再各自生成一套 loop FSM。
   每个 mode 的输出仍严格为 `8/18/36/37` words。
2. **1x1 顺序物理地址**：每个 row 只求一次 row base，每个 pixel 只求一次 pixel base；
   K tile 地址固定每次增加 32 bytes。`c_begin` 不再驱动 FMBUF/URAM 物理地址，只保留
   `remaining_c` 计算尾 tile mask。aligned INT8 仍走直接单字快路径；非对齐/INT4 使用两字
   cursor 复用跨 256-bit 边界的相邻 word，不增加读数。
3. **SA 控制局部化**：mode control 从 4 组 x 8 lanes 调整为 8 组 x 4 lanes；32x32 PE、
   1024 DSP MAC、累加顺序和 `II=1` 均保持不变。
4. 保持 PPU/Vec/Pool、weight loader、FMBUF/BRAM/URAM split、PARAM v4 ABI 和 artifact 不变；
   不新增 cache、memory port 或 datapath clone。

### 7.4 性能与验证门槛

源码门槛：

- wide emitter 的源代码和 csynth hierarchy 中只能有一个发射 loop controller；
- 1x1 hot loop 不得调用 channel-derived INT8/INT4 tile reader，不得由 `c_begin` 生成物理地址；
- SA mode control 必须为 8 组 x 4 lanes；
- P7 structure/contracts、轻量 CSim 通过，且无新增 legacy/dead helper。

csynth 门槛：

- C28/C64/C128/C131 的发射 word 数和关键 II 不回退，所有关键 pipelined loops 仍为 `II=1`；
- SA 仍为单实例 32x32，DSP 不增长；top LUT/FF/BRAM/URAM 相对本轮基线不得增长超过 `1%`；
- 综合时间保持可控，不允许 single-loop mode mux 引发搜索爆炸；若出现该问题，回退实现方式，
  不能以关闭 pipeline 规避。

implementation 门槛：

- placer/router long/short congestion 不得再出现 level 6/7；
- SA mode net fanout 目标 `<5k`，WinGen 不再出现四套 emitter FSM/cache enable 热点；
- `failed nets=0`、`node overlaps=0`、`WNS>=0`、`WHS>=0`。

若 0816A 的 csynth 资源/II 合格但实现仍失败，下一轮才评估小幅 URAM/BRAM 边界再平衡；
不得在同一轮混入存储迁移，以免无法区分 WinGen/SA 局部化的真实效果。

### 7.5 0816A 源码实施与综合前验证

已完成第 7.3 节三项源码修复，并把对应约束加入 `check_p6_hls_structure.py` 与
`test_p7_contracts.py`，防止后续恢复旧调用图。验证结果：

| 检查 | 结果 |
|---|---|
| 新 route-locality structure gate | 旧源码按预期失败；修复后 PASS |
| WinGen 独立 CSim | `win_gen_tb passed`，`0 errors`，max stream depth `4608` |
| SA 独立 CSim | `sa_core_tb passed`，`0 errors`，max stream depth `16` |
| P7 Python contracts | 36 项全部 PASS |
| 顶层 prefix-lite CSim (`MAX_UOP=4`) | PASS，`0 errors`，max stream depth `4608` |
| legacy/dead scan | 三个旧 1x1 reader 及无用前置声明已删除；其余单次命中均为实际实例化的模板 helper |
| `git diff --check` | PASS |

本轮尚未声称物理问题已解决。下一步必须先运行完整 top csynth，核对 single wide-emitter
hierarchy、关键 II、SA 单实例及资源门槛；只有 csynth 符合第 7.4 节要求，才值得重新 package
和 implementation。
