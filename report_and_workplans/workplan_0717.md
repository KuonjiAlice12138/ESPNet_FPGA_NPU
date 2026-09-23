# Workplan 0717 - P7 向 500 ms 收敛的性能改进计划

## 0. 当前结论

2026-07-21，build tag 为 `20260720-P7-R2` 的硬件已在 100 MHz 下完成单图整网上板测试：

| 项目 | 实测值 |
|---|---:|
| RTL `MODE_RUN` window | `60,219,515 cycles` |
| ARM timer | `20,073,160 ticks` |
| 实测延时 | `602 ms` |
| 相对 P7-R5 | `-11,207,770 cycles`，下降 `15.69%` |
| 相对 P6J | 约下降 `40.0%` |

RTL 计数 `602.195 ms` 与 ARM timer `602.201 ms` 基本一致，计时结果可信。Round 1/2 已证明 BLOCK5 source cache、C131 4-byte packetizer 与 C256 cblock-major 路线有效；距离 500 ms 仍需减少 `10,219,515 cycles`，距离 48M 工程目标仍需减少 `12,219,515 cycles`。

当前若完全隐藏 `PPU_ROW_CONSUME=10.01M`，总周期仍约为 `50.21M`。因此 Round 3 不能只做 overlap，还必须保留约 `0.7M~1.2M` 的串行尾部收敛余量，并先关闭板上末尾短暂进入 `ERROR` stage 的异常状态。

---

## 1. Round 5 可信基线与周期预算

### 1.1 Stage 构成

| Stage | Cycles | 占比 |
|---|---:|---:|
| `CONV_ROW_DATAPATH` | `33,187,072` | `46.46%` |
| `PPU_ROW_CONSUME` | `11,118,816` | `15.56%` |
| `PPU_BLOCK5_FINAL` | `12,330,030` | `17.26%` |
| `VEC_FIXED` | `11,316,007` | `15.84%` |
| `AVGPOOL` | `3,048,809` | `4.26%` |
| Frame/control/weight/idle | `426,551` | `0.60%` |

当前 Conv 已稳定在 `33.19M`，而 BLOCK5、Vec 和普通 PPU 合计 `34.76M`，成为后续唯一值得投入的主路径。

### 1.2 更新后的 500 ms 预算

最终按板级 cycle验收，不使用单个 HLS `max latency` 替代：

| 部分 | Round 5 | Round 2 | Round 3 gate |
|---|---:|---:|---:|
| Conv row datapath | `33.19M` | `32.68M` | `<=34.2M` |
| PPU row 可见周期 | `11.12M` | `10.01M` | `<=0.8M` |
| BLOCK5 final | `12.33M` | `8.26M` | `<=7.5M` |
| Vec fixed | `11.32M` | `5.79M` | `<=5.5M` |
| AvgPool | `3.05M` | `3.05M` | `<=3.05M` |
| 其余 | `0.43M` | `0.42M` | `<=0.5M` |
| **Total** | **`71.43M`** | **`60.22M`** | **`<50M`** |

Round 1/2 已完成大部分确定性串行工作量削减。Round 3 必须同时隐藏至少 `9.2M` 普通 PPU 周期，并再削减约 `1M` BLOCK5/Vec串行尾部，才能给 segment fill/drain留出实现余量。

---

## 2. 当前代码证据与根因

### 2.1 Conv/PPU 仍严格串行

`ESP_INT8_hls/src/conv_engine.cpp::run_conv_rows_task()` 对每一行固定执行：

```text
load weights once
for each row:
    shared_conv_row_engine(..., s_shared_conv_row_buf)
    optional ppu_preadd_row(...)
    ppu_consume_conv_row(...) or ppu_consume_block5_final_row(...)
```

当前只有一个 BRAM row buffer，Conv 生产完成后 PPU 才开始消费，板上 stage 周期因此完全相加。该结构是 Round 4B 的 overlap 入口，但不能直接在外层添加 `DATAFLOW`。

### 2.2 直接 DATAFLOW 会重新形成共享存储 feedback

`memory.cpp` 中 `BANK_FMEM0/1/2` 最终共享：

```text
s_fmbuf_uram[]
s_fmbuf_bram[]
```

逻辑 bank 只通过 descriptor/base offset 区分。Conv/WinGen 读输入、PPU 读 add/cat/residual 并写输出时仍访问同一物理 owner。HLS 无法从 runtime descriptor 证明不别名，直接并行会带来：

- `HLS 214-475` dataflow process merging/shared feedback；
- memory owner 或 read/write gateway clone；
- 深 FIFO、运行时仲裁和综合搜索爆炸；
- 当前已接近上限的 BRAM/URAM 与 routing 再次失控。

因此 Round 4B 必须先把 Conv 行拆为“全局存储预取”和“仅访问局部 cache 的计算”两个阶段，再让局部计算与前一行 PPU overlap。禁止 producer/consumer 在同一 DATAFLOW region 内直接访问共享 FMBUF owner。

### 2.3 BLOCK5 重复读取相同 compact source

`ppu_finalize_block5_static_row()` 已移除 runtime segment loop，但每个 tile composer 仍独立调用 compact segment reader：

- L2 每像素组合 2 个 tile，执行 5 次 compact segment read；实际只有 `s0/s1/s2/s3` 4 个唯一 source word；
- L3 每像素组合 4 个 tile，执行 7 次 compact segment read；实际也只有 4 个唯一 source word；
- 每个 composer 都构造 complete-partition lane array，再 pack 成 256-bit word；
- emit tail 对 tile 做 runtime 选择，随后执行 affine/add-affine 和写回。

板级 `PPU_BLOCK5_FINAL=12.33M` 与该重复读、重复 lane compose 结构一致。

### 2.4 Vec fixed 仍按 byte 拼接 C131

`vec_alu_engine.cpp::run_fixed_affine_common()` 当前存在两项确定开销：

1. `h -> w -> cblock` 热循环内每个 tile 都重新读取 affine qparam；
2. C131 row-contiguous 路径对每个 tile 的 32 lane 执行 `PIPELINE II=1` 的逐 byte append。

C131 的 tile 数为：

```text
128 * 256 * ceil(131/32) = 163,840 tiles
```

仅 32-lane byte pack 的理论下限就约为 `5.24M cycles`。这解释了 `VEC_FIXED=11.32M` 的主要来源。C256 aligned 路径也因 pixel-major loop 反复读取同一 cblock qparam。

---

## 3. 全轮次绝对约束

1. 保持 100 MHz、512x1024 full-resolution 输出和现有量化行为，不用降分辨率或 PS 后处理换取延时。
2. 冻结 Round 2/3 Conv 主路径：只允许一个 WinGen、一个 SA、一个 weight buffer owner、一个 Conv row engine。
3. 不降低 `TM/TK`，不新增第二套 SA/WinGen/cache engine，不恢复 P6 cfg/shape-based dispatch。
4. 所有执行选择来自 PARAM schedule/flag；HLS hot loop 禁止根据 `in_c/kernel/dilation/valid_c` 理解网络形状。
5. 不新增 current Conv output 的 frame-memory round trip，不恢复 bytewise generic store 或 narrow RMW 主路径。
6. 不使用 allocation/Tcl directive 掩盖源码多 call site；single owner 必须首先由 call graph 保证。
7. 不新增 `.cpp/.hpp`。修改限制在现有 `ppu.cpp`、`vec_alu_engine.cpp`、`conv_engine.cpp`、`win_gen.cpp`、`npu_schedule.hpp` 和配套 Python 工具。
8. 当前实现资源已接近物理边界：URAM `112/112`、BRAM 约 `700.5/744`，100 MHz post-route 裕度约 `+0.017ns`。任何轮次不得增加 URAM；BRAM 增量必须有明确上限并由 csynth/implementation 复核。
9. Round 5 的 `MASK.BIN` 尚需补做离线 PA/mIoU 记录；后续性能版本必须继续维持无明显精度下降。

---

## 4. Round 1：BLOCK5 source cache 与直接 tile compose

### 4.1 修改范围

主要文件：

- `ESP_INT8_hls/src/ppu.cpp`
- `tools/check_p6_hls_structure.py`

### 4.2 实施步骤

1. 在每个 `(row, ow)` 入口只读取一次 `s0/s1/s2/s3` compact source word，形成四个局部 `act_vec_t`。
2. L2/L3 tile 直接用编译期固定 `.range()` 从四个 source word 和 `row_buf[ow]` 拼接，不再调用多次 `ppu_read_compact_segment_to_lanes<>`。
3. 删除六个“读存储+lane array+pack”式 composer，保留一个局部 compose section；L2/L3 pattern 只在 row 入口分发一次。
4. 保留唯一 `ppu_finalize_block5_emit_word` common tail；禁止 L2/L3 各自复制 affine/add-affine/store。
5. 先维持当前 lane arithmetic factor，不在同一轮同时扩大算术并行度。只有 csynth 证明 LUT/时序不恶化时，才单独评估 factor 4 -> 8。
6. 更新 checker：compact source 每像素最多 4 个物理 read call，旧 segment-to-lanes composer 不得被顶层引用。

### 4.3 预期收益与 gate

```text
PPU_BLOCK5_FINAL 初步目标 <= 8.0M cycles
冲刺目标 <= 6.5M~7.0M cycles
BLOCK5 compact source read 数：L2/L3 均为 4 reads/pixel
BLOCK5 affine/add/store physical owner = 1
top LUT/FF 不增长，BRAM/URAM 不增长
CONV_ROW_DATAPATH 回退 <= 3%
```

若直接 source slicing 形成新的 256-bit 大 mux、高扇出或多个 emit engine，立即回退到 Round 5 版本，只保留已证明能减少物理 read 的部分。

---

## 5. Round 2：Vec fixed 4-byte packetizer 与 loop reorder

### 5.1 修改范围

- `ESP_INT8_hls/src/vec_alu_engine.cpp`
- `ESP_INT8_hls/include/npu_schedule.hpp`
- `tools/export_int8_hw_blob.py`
- `tools/hw_param_replay.py`
- `tools/test_p7_contracts.py`

### 5.2 C131 row-contiguous 路径

1. 保留 compact C131 layout 和现有 B2 reverse-row/backup 语义，不改为 padded C160。
2. 将逐 byte lane append 改成固定 4-byte group packetizer：每个 256-bit affine output 切为 8 个固定 32-bit group。
3. 使用最多 8 项的小型 `u32` group buffer，满 8 group 后生成一个 256-bit word；最后 C131 tile 只提交有效的 3-byte group。
4. 禁止动态 256-bit barrel shift、动态 `.range()` destination 和 32 次 byte RMW。
5. 继续在覆盖源 tensor 前完成 B2 backup；不得为消除 backup 改变 tensor lifetime 或增加新的整行副本。

### 5.3 C256 aligned 路径

1. exporter 为该 schedule 生成明确的 compiled loop-mode flag。
2. aligned 路径改为 `cblock -> h -> w`，每个 cblock 只加载一次 qparam，再连续完成 tile read/affine/write。
3. row-contiguous C131 仍保持 pixel-major，以维持顺序 pack；禁止在 HLS 中用 `valid_c==131/256` 推断模式。
4. qparam cache 只缓存实际 cblock 数；若 complete partition 导致 mux/高扇出，则保持窄局部 qparam 并撤销全量 cache。

### 5.4 中间检查点

Round 1/2 完成后才执行第一次昂贵闭环：

```text
structure/contracts -> prefix/U40/lowres -> fullres CSim
-> csynth audit -> 100MHz implementation -> 单图上板 stage counter
```

中间 gate：

```text
VEC_FIXED <= 6.0M cycles，冲刺 <= 5.0M
PPU_BLOCK5_FINAL <= 8.0M cycles
RTL_STAGE_TOTAL <= 62M cycles
CONV_ROW_DATAPATH <= 34.2M cycles
implementation 合法布线且 100MHz timing pass
PA/mIoU 相对当前版本无明显下降
```

若中间总周期仍高于 `62M`，不得直接进入 overlap；先用 stage counter 判断 BLOCK5/Vec 哪一项未达到预算。

### 5.5 2026-07-21 上板结果与 Round 2 gate

| Stage | P7-R5 | P7-R2 | 变化 |
|---|---:|---:|---:|
| `CONV_ROW_DATAPATH` | `33,187,072` | `32,683,712` | `-1.52%` |
| `PPU_ROW_CONSUME` | `11,118,816` | `10,010,706` | `-9.97%` |
| `PPU_BLOCK5_FINAL` | `12,330,030` | `8,256,942` | `-33.03%` |
| `VEC_FIXED` | `11,316,007` | `5,794,399` | `-48.79%` |
| `AVGPOOL` | `3,048,809` | `3,048,809` | 不变 |
| `RTL_STAGE_TOTAL` | `71,427,285` | `60,219,515` | `-15.69%` |

Round 2 判定为**条件通过**：

- `VEC_FIXED <= 6.0M`、`RTL_STAGE_TOTAL <= 62M`、`CONV_ROW_DATAPATH <= 34.2M` 均通过；
- `PPU_BLOCK5_FINAL=8.257M` 比 `8.0M` 初步目标高 `0.257M`，偏差 `3.21%`，不阻止进入 Round 3，但仍是串行尾部优化对象；
- Vec 距离 `5.0M` 冲刺目标还差 `0.794M`；
- 实现、合法布线与 100 MHz 上板运行已通过；full-resolution mask 的离线精度结果仍需补录；
- counter 记录 `ERROR=99 cycles`，结束状态为 `current=12/status=0x00000C05`。该开销可忽略，但它表示顶层结束时确实进入过 `PROF_STAGE_ERROR`，不能作为正常尾状态长期保留。

当前 PARAM audit 已记录 `exec_entry_count=16`，exporter也强制验证最后一项为 `EXEC_END`；综合前同组 fullres CSim通过时 `csim_last_error=0`。因此不能直接归因为 artifact 缺少 END，需区分板级 MainCtrl返回错误和 RTL counter采到短暂/残留状态。

进入 overlap 代码前必须定位该板级末尾错误。优先复用现有 `prof_pc/prof_issue_kind` 输出记录失败 PC、issue kind 与 error code，不新增独立调试总线。正常 `MODE_RUN` 完成后必须稳定落在 `IDLE`，且 `ERROR=0`。

---

## 6. Round 3：串行尾部收敛与受控 segment overlap

### 6.1 更新后的周期预算

当前完全去除普通 PPU 可见周期后的串行下限为：

```text
60,219,515 - 10,010,706 = 50,208,809 cycles
```

实际 overlap 必然存在 segment fill/drain，因此只做 overlap 无法稳定进入 500 ms。Round 3 的最低可实现预算调整为：

```text
PPU_BLOCK5_FINAL <= 7.5M
VEC_FIXED <= 5.5M
PPU_ROW_CONSUME visible <= 0.8M
overlap both_busy >= 9.2M
RTL_STAGE_TOTAL < 50M，优先目标 <=49.5M
```

达到上述预算约可把串行部分降至 `49.16M`，再保留不超过 `0.8M` 的 PPU fill/drain。`48M` 保留为后续冲刺目标，不作为本轮首个实现 gate。

### 6.2 Round 3A：结束状态与 overlap 可行性 gate

1. 使用当前同组 `PARAM.BIN/INPUTQ.BIN` 重跑 top CSim，确认 `csim_last_error=0`、最后一项 exec plan 为 `EXEC_END`。
2. 复用 `prof_pc/prof_issue_kind`：发生错误时由 `prof_issue_kind` 编码 error code，RTL counter 锁存最后 PC/kind/error；正常结束必须回到 `IDLE`。
3. profiling build 将 Conv 顺序阶段拆为 `cache_fill`、`local_emit+SA+post`，同时统计 `ppu_busy` 与未来的 `both_busy`。并发后不得继续把互斥 stage bin 简单相加解释总延时。
4. 只有 `local_emit+SA+post >= 9.2M cycles`，且可 overlap schedule 覆盖的 PPU 工作量不低于 `9.2M`，才进入 Round 3B；否则停止结构重构，转而继续压 BLOCK5/Vec 串行尾部。
5. 补做本轮 `MASK.BIN` 的 PA/mIoU；精度未通过时不得把性能 gate 判为完成。

### 6.3 Round 3B：无共享 FMBUF feedback 的 segment overlap

当前 WinGen cache 是滚动列缓存而不是整行缓存：窄通道 cache 为 64 个 column slot，宽通道 cache 为 `3x4x5` 个 256-bit word。把完整三行 source 预取到本地最坏需要约 100 KiB，会消耗约 23 个 BRAM36，并显著加剧当前 `94% BRAM + 100% URAM` 的实现压力。

因此 Round 3 不采用 full-row source prefetch，而调整为 PARAM 编译的 segment-level 三段调度：

```text
Phase A: prefetch segment n+1 所需的 scheduled WinGen columns 到局部 cache
Phase B: 从局部 cache 计算 Conv segment n+1 -> ping/pong segment buffer
         与 PPU consume segment n 并行
Phase C: drain 最后一个 output segment
```

Phase B 内 Conv 只能读取本地 WinGen cache、weight buffer并写 segment buffer；不得访问 `s_fmbuf_uram/s_fmbuf_bram`。这样 PPU 才能在同一时间独占全局 FMBUF read/write owner，避免共享 URAM feedback。总周期按 `prefetch + max(local_conv_compute, ppu_consume)` 形成，不能假设 WinGen 的全部周期都能被 overlap。

### 6.4 实施步骤

1. exporter 为每个可 overlap schedule 编译 `segment_out_w`、输入列 footprint、首尾 padding、compact-word 对齐信息和显式 overlap enable；不满足约束的 schedule 固定走 serial。
2. 在 `win_gen.cpp` 将 scheduled row 行为拆为同一 owner 下的 `cache_fill_segment` 和 `emit_cached_segment`，共享唯一 cache，不创建第二套 WinGen/cache engine。
3. 在进入并发区前，把 descriptor 与当前 segment 所需 qparam 固化到局部窄结构；并发 producer/consumer 内禁止调用共享 PARAM/memory gateway。
4. `shared_conv_row_engine` 增加编译好的 `[ow_begin, ow_count]` 范围，SA/postprocess 只生成当前 segment，仍由唯一 Conv engine call site调用。
5. `ppu_consume_conv_row` 改为同一 owner 下的 segment consumer；segment 边界按 256-bit destination word 对齐，禁止跨 segment 保存动态 byte cursor或恢复 narrow RMW。
6. `conv_engine.cpp` 只允许两个固定 BRAM segment buffer，禁止 full-row副本、runtime task queue和动态多 owner。若动态 ping/pong 导致大 mux或 engine clone，改用固定 trip-count、深度不超过一个 segment 的 stream channel，不叠加第三套 buffer。
7. 首版只覆盖普通 Conv consumer；BLOCK5 final、upsample、fixed Vec 保持串行。producer/consumer 各只有一个 call site，token 只携带 row、segment、buffer id与 valid/error。
8. FIFO depth固定为 `2~4` 个控制 token；数据 FIFO 深度不得超过单 segment。所有 producer/consumer word count必须由 contract脚本静态相等检查，防止死锁。
9. 普通 consumer 的 add/cat/residual read与输出 write由 PPU 独占全局 memory owner；Conv local compute阶段不得回读 FMBUF。
10. profiling build记录 `cache_fill_busy/conv_compute_busy/ppu_busy/both_busy/stall`；最终性能只按 RTL total与 ARM timer验收。

### 6.5 Round 3C：串行尾部定点收敛

1. overlap 结构功能和资源通过后，依据新的 busy counter只处理仍可见的串行尾部；不得在同一轮继续改 Conv/WinGen/SA。
2. BLOCK5 只允许优化现有 common emit tail、固定 tile compose与 qparam复用，目标 `<=7.5M`；不得恢复多 finalizer或增加 scratch副本。
3. Vec 只允许在现有 4-byte packetizer和 cblock-major路径上减少 flush/control/qparam开销，目标 `<=5.5M`；不得恢复 bytewise pack或增加全量 qparam partition。
4. 若 overlap 后总周期已经 `<50M`，立即冻结，不为追求 `48M` 引入新的 datapath、memory owner或 routing风险。

### 6.6 资源约束

segment 初始上限取 64 个 output position；两个 `64 x 256-bit` output buffer合计仅 4 KiB，约 1 个 BRAM36。宽通道 mode可以由 exporter选择更小 segment，不允许为了统一宽度扩成 full-row cache。实现时要求：

```text
新增 URAM = 0
新增 BRAM <= 2 BRAM36（且总 BRAM必须保留实现余量）
SA/WinGen/weight owner clone = 0
memory read/write owner clone = 0
HLS 214-475 = 0
异常 FIFO/deadlock warning = 0
```

若 cache fill/emit 拆分需要复制整套 cache、引入 full-row source buffer或让 BRAM超过预算，则停止 Round 3，不通过增加存储硬做 overlap。

### 6.7 最终性能 gate

```text
MODE_RUN 正常结束：ERROR=0，final stage=IDLE
PPU_ROW_CONSUME 可见周期 <= 0.8M
overlap 同时忙周期 >= 9.2M
CONV_ROW_DATAPATH <= 34.2M
PPU_BLOCK5_FINAL <= 7.5M
VEC_FIXED <= 5.5M
RTL_STAGE_TOTAL < 50M，优先目标 <=49.5M
```

`48M` 仅作为下一轮冲刺目标。若普通 PPU overlap 后总周期停在 `50M~52M`，先按 Round 3C处理可见 BLOCK5/Vec尾部；不得重新扩展 SA/WinGen、复制 cache或增加 current-output 中间访存。

### 6.8 2026-07-21 Round 3 综合恢复修复

上一轮 csynth 已实现 single WinGen/single SA，但暴露两项结构问题：普通 overlap segment 与串行整行路径分别综合出两套 compact PPU，且 compact packetizer 的动态位移成为约 `11.38 ns` 关键路径。该报告资源为 `BRAM 104% / LUT 108%`，不能直接进入实现。

本轮已完成：

- 删除旧 `ppu_consume_conv_row` 整行入口，所有普通 Conv/BLOCK5 branch 统一调用唯一 `ppu_consume_conv_segment`；BLOCK5 final与 upsample仍保留明确串行尾部。
- compact packetizer改为固定 32-bit group抽取和固定 shift merge，移除 `byte_base*8`、动态 carry shift；跨 segment只保存一个未满 256-bit word的有限状态。
- exporter为所有非 upsample direct Conv编译显式 `segment_out_w`，contracts要求这些 descriptor均进入编译好的 segment路径，不再因 C25非整字边界回落到第二套 consumer。
- Conv DATAFLOW core收敛为无控制流的 canonical region，控制判断留在 owner外层；checker新增单 consumer、单 DATAFLOW和动态 shift禁用 gate。

验证结果：structure checker与 PARAM v4 contracts通过；prefix-lite、U40和 fullres CSim均为 `0 errors`。fullres得到 `last_uop=74 / last_error=0`，最终 mask为 `0/524288 mismatch`，stream最大深度为64。

下一轮 csynth必须确认：`HLS 200-471/214-113/214-114=0`，compact PPU只保留一个RTL实例，WinGen/SA仍为 singleton，估计时钟恢复到 `<10 ns`，且 BRAM/LUT不高于上一轮。未取得新报告前，不把源码修复视为综合门槛已通过。

---

## 7. 验证顺序与停止规则

### 7.1 每轮轻量检查

1. `python tools/check_p6_hls_structure.py`
2. dead/legacy/reference scan；确认旧 composer、byte packer 和重复 owner已退出主路径。
3. `python tools/test_p7_contracts.py`；当前 prefix replay fixture若仍与 artifact 不同组，必须先重建 fixture，不能把已知错配当成硬件回归。
4. prefix-lite/U40 或对应局部 CSim。

### 7.2 两个昂贵检查点

仅在以下节点跑 fullres、综合、实现和上板：

- Round 1+2 全部完成后；
- Round 3 overlap 完成后。

fullres 必须使用同组 `PARAM.BIN/INPUTQ.BIN/golden`，核对 lowres logits、fullres mask、PA/mIoU 和非法标签。

### 7.3 立即停止/回退条件

出现以下任一情况，本轮不得继续叠加优化：

- Conv stage 回退超过 3%；
- HLS 214-475、dataflow process merging、SA/WinGen/PPU clone；
- csynth runtime 相比当前版本明显失控；
- URAM 增长，BRAM 增量超过 2 个 BRAM36，或 LUT/FF 增长超过 3%；
- implementation global congestion level >=5、route overlap非零或 100MHz timing fail；
- fullres mask 精度明显下降。

---

## 8. 最终交付状态

完成版本必须同时满足：

```text
功能：fullres CSim/board mask 精度通过
架构：single WinGen + single SA + single Conv/PPU/memory owner
性能：100MHz RTL total <=50M cycles，目标 <=48M
实现：合法 placement/routing，100MHz timing pass
代码：无 P6 cfg dispatch、旧 BLOCK5 composer、bytewise Vec pack、dead/legacy path
工具：PARAM schedule、replay、contracts 与 HLS mode完全一致
```

本计划的优先级是“先减少确定的串行工作量，再以局部 cache 隔离全局存储后做 overlap”。不接受直接在共享 URAM/BRAM owner两侧添加 DATAFLOW，也不接受为了达到 500 ms 重新引入多套 Conv/SA 或不可实现的存储复制。
