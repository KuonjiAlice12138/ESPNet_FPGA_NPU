# Workplan 0727 - P7 WinGen/SA 利用率与 500 ms 收敛计划

## 0. 当前结论

2026-08-03 Round 4A/4B 已完成 csynth、legal implementation 和板级逐 exec
验收：

- RTL 整网为 `51,229,815 cycles / 512.298 ms`；
- ARM 为 `17,076,592 ticks @ 33.333 MHz = 512.303 ms`，与 RTL 只差
  `0.0047 ms / 9.24 ppm`，计时口径可信；
- 相对 Round 3 `54,200,473 cycles` 减少
  `2,970,658 cycles / 5.481%`；
- 相对可信 P7-R2 `63,603,157 cycles` 累计减少
  `12,373,342 cycles / 19.454%`；
- 所有 prefix 均为 `status=0x5`、`ERROR=0`，最终回到 `IDLE`；
- `E:\MASK.BIN` 为 `524288 bytes`，SHA256
  `655C433884531D4789A48F1C063C416BA6AF67BBED2D0718B1DE211E28FDE2AD`，
  与 golden/fullres CSim 完全一致。

本轮距离 `<50,000,000 cycles` 仍差 `1,229,815 cycles / 2.40%`；若按工程
目标 `49.5M`，还需减少 `1,729,815 cycles / 3.38%`。

综合代码、报告和板级 prefix 数据后，结论更新为：

1. 当前物理阵列仍为单套 `32x32` MAC，MAC 循环 `II=1`，不存在 SA 被缩成
   `16x32` 或多套 SA 并存的问题。
2. Round 4A 明确有效：`VEC_FIXED 7.532M -> 6.156M`，减少
   `1.376M / 18.27%`；U71 已通过目标，U39 仍差约 `0.379M`。
3. Round 4B 是混合结果：C3/s2 的 U2 Conv 减少 `2.822M / 31.14%`，但
   C12/s1 的 U20/U35 Conv 各回退 `0.797M / 12.11%`。当前 row-bank 直发路径
   对每个窗口重新读取 3x3 tile，丢失原 narrow horizontal cache 的列复用，
   因而不适合 C12 stride1。
4. 暂不执行原 Round 4C。先做 **Round 4B-R selective rollback**：保留 C3
   row reuse，关闭 C12 row reuse，使 C12 回到 Round 3 稳定路径。仅恢复这两项
   回退，预测整网约 `49,636,795 cycles`，已低于 50M。
5. BLOCK5 final 和 AvgPool 周期仍完全不变，只在 4B-R 实测仍未过 50M 时选择
   一个补差项，不与 WinGen 修复同轮叠加。

---

## 1. 证据基线

### 1.1 板级热点

本轮最终 stage 构成为：

| Stage | 当前 cycles | 占比 | 相对 0723 |
|---|---:|---:|---:|
| `CONV_ROW_DATAPATH` | `33,823,468` | `62.40%` | `-10,219,348` |
| `PPU_ROW_CONSUME` | `1,112,352` | `2.05%` | `0` |
| `PPU_BLOCK5_FINAL` | `8,256,942` | `15.23%` | `0` |
| `VEC_FIXED` | `7,532,167` | `13.90%` | `-32,768` |
| `AVGPOOL` | `3,048,809` | `5.62%` | `0` |
| Frame/control/weight/idle | `426,735` | `0.79%` | 计数噪声 |
| **Total** | **`54,200,473`** | **`100%`** | **`-10,251,932`** |

逐 exec 的当前绝对热点是：

| Exec | Total delta | 主要构成 | 占整网 |
|---|---:|---|---:|
| U35 | `9,880,512` | `6.578M Conv + 3.293M B5` | `18.23%` |
| U20 | `9,225,186` | `6.578M Conv + 2.638M B5` | `17.02%` |
| U2 | `9,061,504` | `9.061M` 首层 C3 Conv | `16.72%` |
| U39 | `5,761,600` | fixed affine | `10.63%` |
| U68 | `4,137,152` | `2.802M Conv + 1.327M B5` | `7.63%` |
| U53 | `3,809,472` | `2.802M Conv + 0.999M B5` | `7.03%` |
| U40 | `2,081,378` | `2.074M` C131 Conv | `3.84%` |
| U71 | `1,770,560` | fixed affine | `3.27%` |
| U72 | `1,645,410` | `0.536M Conv + 1.108M PPU` | `3.04%` |

与 0728 回退版相比，Conv 的收益只集中在窄 paired schedule：

| Exec | 0728 Conv | 当前 Conv | 减少 |
|---|---:|---:|---:|
| U2 | `12,681,888` | `9,060,704` | `3,621,184 / 28.55%` |
| U7 | `2,718,416` | `1,911,088` | `807,328 / 29.70%` |
| U20 | `10,014,504` | `6,578,146` | `3,436,358 / 34.31%` |
| U35 | `10,014,504` | `6,578,146` | `3,436,358 / 34.31%` |
| U40/U53/U68 | 基线 | 基本不变 | `<0.02%` 噪声 |

这证明 Round 2R 的 compiled run 与窄 loader 已经命中真实瓶颈；继续压 Conv
必须减少跨输出行的重复物理字读取，不能再改已经稳定的 wide/unpaired 路径。

### 1.2 最新 csynth/implementation

当前报告与本轮 bitstream 日期均为 2026-07-30：

| 项目 | 结果 |
|---|---:|
| HLS estimated clock | `7.300 ns` |
| HLS total | BRAM18 `1620`、DSP `1199`、FF `266414`、LUT `346637`、URAM `112` |
| `scheduled_window_generator_row` | BRAM18 `18`、DSP `23`、FF `23802`、LUT `28849` |
| `systolic_array_core_row` | 单实例、`32x32`、MAC loop `II=1` |
| csynth elapsed | `13m10s` |
| implementation WNS/WHS | `+0.211 ns / +0.010 ns` |
| physical CLB LUT / register | `218059/341280 = 63.89%` / `29.11%` |
| physical BRAM tile | `685/744 = 92.07%` |
| physical URAM | `112/112 = 100%` |
| physical DSP | `1206/3528 = 34.18%` |
| route status | `0` routing errors |

实现能够合法布线，物理 LUT/FF 和 DSP 仍有余量，但 BRAM 已达 `92.07%`、
URAM 已满。后续可用少量 DSP 换取算术吞吐；任何 WinGen 行缓存都必须替换
现有 `18 BRAM18` 窄 cache，而不是与其并存。

当前最差实现路径不在 SA，而在 FMBUF BRAM cascade 到
BLOCK5/FMBUF 读取，仍经过 7 级 RAMB36 cascade；最差 setup slack 为
`+0.211 ns`。因此 BLOCK5 的 lane 并行只能作为后续独立实验，不能与新增缓存
同轮叠加。

最新 HLS audit 为 `blocker=0, high_risk=22`，其中仍有两项
`HLS 200-1449`：

- `systolic_array_core_row` 同时有 predecessor 且读取 caller input；
- `ppu_consume_conv_stream` 同样存在 caller input。

本计划不通过复制 weight/qparam 大数组处理该 warning；只有在数据流 viewer
证明它形成真实吞吐阻塞时，才传递窄控制副本。

---

## 2. SA 利用率模型

### 2.1 三种利用率必须分开

1. **静态填充率**

   ```text
   useful MAC / 已发射的 32x32 MAC slot
   ```

   它描述 Cout 和 K tail 的空槽，不包含等待 WinGen、输出 drain 和写回。

2. **SA issue 时间占用**

   ```text
   MAC issue cycles / Conv+PPU board window
   ```

3. **端到端有效 PE 利用率**

   ```text
   useful MAC / (1024 * Conv+PPU board cycles)
   ```

按当前 PARAM v4 和 paired schedule：

```text
useful MAC                         = 1,345,740,800
issued MAC slots                   = 1,920,991,232
weighted static fill              = 70.05%
actual MAC issue cycles lower bound = 1,875,968
psum drain lower bound              =   729,088
SA issue + drain lower bound         = 2,605,056
current Conv+PPU board window         =34,935,820
issue time occupancy                 = 5.37%
issue + drain occupancy              = 7.46%
end-to-end useful PE utilization     = 3.76%
```

`70.05%` 的静态填充并不差；Round 2R 已把端到端有效利用率从约 `2.97%`
提高到 `3.76%`，但 Conv+PPU 仍是 issue+drain 理论下限的 `13.41x`。

### 2.2 主要卷积形状的静态填充

| 形状类别 | 当前 pixel parallel | 静态填充率 |
|---|---:|---:|
| C3 -> C16, 3x3 | 2 | `84.38%` |
| C19 -> C12, 3x3 | 2 | `66.80%` |
| C12 -> C16, 3x3 | 2 | `84.38%` |
| C12 -> C12, 3x3 | 2 | `63.28%` |
| C64 -> C12, 1x1 | 2 | `75.00%` |
| C131 -> C25, 3x3 | 1 | `77.80%` |
| C25 -> C28, 3x3 | 1 | `76.90%` |
| C25 -> C25, 3x3 | 1 | `68.66%` |
| C128 -> C25, 1x1 | 1 | `78.13%` |
| C256 -> C2 classifier | 2 | `12.50%` |

低填充的 classifier 计算量很小，U72 的主要周期来自 full-resolution output，
不是这部分 MAC。C12/C25 的跨像素 ragged packing 理论可继续提高填充，但需要
partial-pixel psum carry、输出重排和更复杂的 activation 选择；其全网理论收益
上限仍被 `0.562M cycles` 限制，不应作为当前主线。

### 2.3 阵列形状结论

当前不修改 `TM=32/TK=32`：

- 改成更宽 TM 会扩大 weight/psum 广播和 placement 压力；
- 改成更宽 TK 不能消除 WinGen 等待；
- 复制 SA 会直接增加约 992 DSP 及大规模广播网络；
- 当前真正需要的是让现有 SA 持续获得 activation，而不是增加峰值 MAC 数。

---

## 3. 500 ms 工程预算

在 100 MHz 下，验收线为 `<50,000,000 cycles`，工程目标取
`49.5M cycles` 留出约 5 ms 裕量：

| 部分 | 当前可信值 | 目标 | 需减少 |
|---|---:|---:|---:|
| Conv + visible PPU | `34.936M` | `<=31.5M` | `>=3.436M` |
| BLOCK5 final | `8.257M` | 首个检查点冻结 | `0` |
| Vec fixed | `7.532M` | `<=5.8M` | `>=1.732M` |
| AvgPool | `3.049M` | 首个检查点冻结 | `0` |
| 其余 | `0.427M` | `<=0.5M` | 不回退 |
| **Total** | **`54.200M`** | **`<=49.1M`** | **`>=5.1M`** |

上表给主线保留约 `0.9M cycles` 的余量。若 Conv+PPU 和 Vec 只刚好达到
`31.5M / 5.8M`，总周期约为 `49.0M`；因此第一次板级检查前不需要同时修改
BLOCK5 和 AvgPool。若主线未完全达到预算，再按实际剩余缺口只选择一个补差项。

---

## 4. 全轮次绝对约束

1. 保持 100 MHz、512x1024 full-resolution 输出、PARAM v4 ABI 和当前量化行为。
2. 只保留一个 WinGen、一个 `32x32` SA、一个 weight buffer、一个 postprocess
   和一个 FMBUF owner。
3. 禁止恢复 activation capture/replay、segment cache restart、output copy 或
   current Conv output 的 frame-memory round trip。
4. 禁止第二套 line cache/SA/PPU；新增缓存只能是 bounded word cursor、FIFO 或
   单套窄 line buffer。
5. 所有模式来自 PARAM schedule。hot loop 不允许根据 `in_c/kernel/dilation`
   重新理解网络 shape。
6. 禁止在 complete-partition 大数组上使用动态 lane 写；禁止用 allocation/Tcl
   directive 掩盖源码多 call site。
7. 不盲目打开所有 `PIPELINE off`。每个 II 修改必须有固定访存端口和循环依赖
   证明，并单独核查综合时间。
8. 不增加 URAM。单轮 BRAM 增量不超过 `4 BRAM36`，最终 physical BRAM
   `<=93%`；HLS LUT/FF 增量 `<=3%`，DSP 增量 `<=64`。
9. csynth 应维持约 10 分钟量级；超过 25 分钟或同一模块 45 分钟无推进即停止，
   不接受用综合搜索爆炸换取理论 II。
10. 100 MHz implementation 必须合法布线，WNS `>=0`；目标保留
    `>=0.10 ns` 裕量。

---

## 5. 准备步骤：修正性能归因工具

修改 `tools/analyze_sa_utilization.py`，不改 HLS：

1. 从 PARAM v4 的 `conv_exec_desc + window_sched_desc` 读取实际
   `pixel_parallel/k_tiles/valid_c`，不再仅用旧 UOP shape。
2. 删除 2026-05 的过时默认 prefix 数据，改为读取 0723 的
   `P00...P15` CSV/日志。
3. 输出每个 conv descriptor 的：
   - useful MAC、issued slot、static fill；
   - 理论 issue/drain cycles；
   - 实测 exec/stage cycles；
   - `measured / SA lower-bound`；
   - compact physical word 读数估计。
4. 为 C3/C12/C19/C25/C131 生成“当前逐 tile 读取”和“物理字只读一次”的
   traffic 对比表，作为 Round 1 gate。

该工具只建立预算，不以估算值替代板级 RTL counter。

---

## 6. Round 1：packed-word 复用与低风险供数修复

### 6.1 目标

不改变现有整行 WinGen/SA/PPU 边界，先消除 compact layout 下相邻像素重复读取
同一 256-bit 物理字，并解除低 k-tile 卷积的 postprocess 反压。

Round 1C 与本轮合并；当前 Round 1C 不单独上板。

### 6.2 HLS/工具修改

主要文件：

- `ESP_INT8_hls/src/win_gen.cpp`
- `ESP_INT8_hls/src/memory.cpp`
- `ESP_INT8_hls/src/conv_engine.cpp`
- `ESP_INT8_hls/include/npu_schedule.hpp`
- `tools/export_int8_hw_blob.py`
- `tools/analyze_sa_utilization.py`

步骤：

1. 为每个 kernel row 增加两个 256-bit 的 aligned-word cursor/cache：
   - 仅缓存最近物理 word 和 tag；
   - compact column 跨 word 时最多组合两个缓存 word；
   - 相邻 column 命中时不再访问 FMBUF；
   - cache 每个 output row 初始化一次。
2. cursor 由单一 WinGen memory-reader owner 使用；不得复制
   `read_phys_word`、不得新增完整 feature row 副本。
3. narrow paired issue 一次计算 pixel0/pixel1 的 column union，统一更新
   column cache，再分别组装两个 window；删除第二次重复 tag 扫描和重复 stage
   控制，但保持实际必要物理列只加载一次。
4. mode/reuse class 在 row 入口从 PARAM schedule 分发一次；如需新增字段，使用
   `window_sched_desc.reserved[]`，不改变 PARAM section 大小。
5. `post_process_conv_row_to_buffer` 从 4 lane/cycle 试升至 8 lane/cycle：
   - 保持单一 postprocess owner；
   - 预期额外约 16 DSP；
   - 若 LUT/route 或关键路径明显回退，独立撤销该项，不回退 packed-word cache。
6. `act_stream0/1` 只允许扩到能够吸收一个窄 issue 的深度；本轮不添加大 FIFO，
   不做新的 DATAFLOW region。
7. 保留 full-row cache lifetime、single SA 和 compact PPU；删除本轮产生的旧
   helper 后再进入综合。

### 6.3 Gate

代码/综合 gate：

```text
single scheduled_window_generator_row / systolic_array_core_row /
post_process_conv_row_to_buffer
无 HLS 214-475、200-975、datapath clone
WinGen/SA/post DATAFLOW 定长生产消费一致
HLS DSP 增量 <=64，LUT/FF 增量 <=3%，URAM不变
csynth <=25 min
```

### 6.4 2026-07-27 实施状态

- 已加入每个 kernel row 的 aligned-word cursor，旧 row packed reader 已退出；
- narrow paired issue 统一准备列集合，stride-1 走双窗口 stage，C3 stride-2 保留
  4-slot 安全顺序；
- Conv postprocess 已由 4 提升到 8 lane/cycle；
- structure checker、WinGen 专项和 prefix-lite CSim 通过；fullres CSim 为
  `0/524288 mismatch`、`last_error=0`；
- 7 月 27 日第二次 csynth：`7.30ns`、`9m45s`，BRAM18 `1620`、DSP `1199`、
  FF `275430`、LUT `339573`、URAM `112`；BRAM/LUT/DSP/URAM/耗时及 clone
  gate 通过，但 FF 相对基线仍 `+6.60%`，未达到 `+3%` gate；
- FF 增量定位为 `read_packed_tile_from_row_cached` 独立 `ap_ctrl_hs` 边界为
  narrow/wide cursor 生成的大量输入/返回寄存器；该无循环 helper 已内联到
  唯一 loader owner；
- 第三次 csynth：`7.30ns`、`9m57s`，BRAM18 `1620`、DSP `1199`、
  FF `268851`、LUT `336976`、URAM `112`。helper 内联使 FF 再降 `6579`，
  但相对基线仍 `+4.05%`，超出 `+3%` gate `2719 FF`，Round 1 尚未收尾；
- 单项 MRU 收窄方案已在静态审查中否决：dilation 分支会形成
  `col 0 -> 4 -> 1 -> 5` 一类交错访问，两项 cursor 能命中较早物理字，单项
  会增加访存。当前保留两项 cursor，不用性能回退换取剩余 `2719 FF`；
  Round 1 尚未通过严格 FF gate，不能进入 Round 2；
- 最新 hierarchy 中 `load_narrow_cache_column` / `load_wide_cache_column`
  分别占 `1956/2025 FF`，且均只有一个调用点。两者已改为内联薄 wrapper，
  保留两项 cursor、循环次序和物理读请求，仅消除宽参数/状态的
  `ap_ctrl_hs` 边界；
- 第四次 csynth：`7.30ns`、`9m56s`，BRAM18 `1620`、DSP `1199`、
  FF `265500`、LUT `336559`、URAM `112`。相对 Round 1C 基线分别为
  BRAM `+0`、DSP `+16`、FF `+2.76%`、LUT `+0.76%`、URAM `+0`；
  两个 loader 已从 hierarchy 消失，WinGen latency 保持 `739331`，audit
  blocker/主 datapath clone 均为 0。Round 1 的 csynth gate 正式通过。

首次板级候选同时包含 Round 1C 和 Round 1，使用 `P00...P15`：

```text
U20 <= 11.930M cycles
U35 <= 12.585M cycles
U2/U20/U35 合计至少再下降 4M cycles
Conv+PPU <=39M cycles
RTL total <=58.5M cycles
ERROR=0，MASK bit-exact
```

若 packed-word 读数显著下降但板级 Conv 不降，说明瓶颈已转到 assembler/post；
进入 Round 2。若物理读数和周期均不降，则撤销 cursor 实现，不叠加更深流水。

---

## 7. Round 2：两级 WinGen 与 SA 连续供数

### 7.1 目标架构

只在 Round 1 证明 compact physical read 是有效方向后实施：

```text
single packed-row loader
    -> bounded {word, row, chunk, slot} stream
single window assembler / existing column cache
    -> act_stream0/1
single SA
    -> single postprocess
```

loader 是唯一 FMBUF reader，assembler 是唯一 local cache writer，不允许两者
共同读写同一 cache，因此不会形成旧版 DATAFLOW feedback。

### 7.2 实施步骤

1. 将“物理 word 读取”和“column cache/window 组装”拆为两个固定 token-count
   进程；metadata 使用窄固定字段，不传递超宽 column bundle。
2. 每个 schedule 在 PARAM 中给出编译后的 loader class、每 issue 新增 column
   数和 k-tile 数；禁止在 hot loop 用 shape 推断。
3. FIFO 只吸收 memory burst 抖动：
   - metadata/word FIFO 深度 `<=16`；
   - `act_stream0` 最大允许覆盖一个 C131 issue，`act_stream1` 只覆盖窄 paired
     issue；
   - 总新增 BRAM 不超过 `4 BRAM36`。
4. 保持 cache 在整行内有效；不得重新引入 segment 或每 issue cache reset。
5. 只有 dilation=1 的窄模式仍明显受垂直重复读限制时，才评估一套共享 3-row
   rolling line buffer：
   - 只覆盖 C3/C12/C19/C25；
   - 不覆盖 C131 和 dilation>1；
   - 只允许一个物理实例，预算 `<=8 BRAM36`，实施前必须先释放等量 BRAM。
6. 处理 `HLS 200-1449` 时只复制窄 cfg/flag stream；禁止复制 weight buffer、
   qparam 或 FMBUF descriptor bank。

### 7.3 中间闭环与 gate

本轮执行一次完整闭环：

```text
structure/contracts -> WinGen/U40 -> fullres CSim
-> csynth audit -> 100MHz implementation
-> P00...P15 board prefix
```

板级 gate：

```text
Conv+PPU <=34.5M cycles
U2 Conv <=9.8M
U20/U35 各自 Conv <=7.8M
RTL total（非卷积未改时）<=54M
100MHz legal route，WNS >=0.10ns
```

若 Conv+PPU 仍高于 `36M`，暂不进入尾部优化；先按 mode/descriptor counter
确认是 C3、narrow C12/C19、wide C131/C25 还是 1x1 未达标。禁止再次做全网
segment overlap。

### 7.4 2026-07-27 实施状态

- PARAM v4 在原 `128B window_sched_desc` 的 reserved 区加入 loader class、
  request 数、warmup/steady update mask 和 words/column；section 大小及量化数据
  不变，artifact 已重导出。
- 3x3 主路径已统一为单一 `configure -> window_row_loader ->
  window_row_assembler` 层级 DATAFLOW：loader 独占 FMBUF 读，assembler 独占
  narrow/wide cache 写；旧 narrow/wide 调度入口、runtime column tag 和
  load/assemble 混合 helper 已删除。
- metadata/word FIFO 均为 depth 16；`act_stream0` 为 depth 40 LUTRAM FIFO，
  `act_stream1` 为 depth 8 LUTRAM FIFO，未增加 SA、cache 或 current-output
  中间访存。
- structure checker、全部 Python contract、WinGen 专项和 U40 prefix CSim
  通过；fullres CSim `last_error=0`、mask `0/524288 mismatch`。
- 首轮 csynth 的 loader/assembler、SA、postprocess 均为单实例，综合耗时
  `10m06s`，预计时钟 `7.300ns`；相对 Round1，LUT `-5233`、FF `-5948`、
  DSP `-10`，但 BRAM18 `+15`。增量全部来自 `40x256-bit act_stream0`
  被强制映射为 BRAM，已改为等深 LUTRAM，不改变 token/吞吐语义。
- LUTRAM 修复后的复综合仍为 `7.300ns / 10m06s`，BRAM18 已从 `1635`
  回落到 `1620`，仅变化 `LUT +48 / FF -54`。相对 Round1 最终基线为
  BRAM `+0`、DSP `-10`、FF `-6002`、LUT `-5185`、URAM `+0`；
  blocker、主 datapath clone 均为 0，Round2 csynth gate 通过。
- 允许推进 100MHz implementation。WinGen loader 的 HLS 最坏上界仍为
  `819201 cycles`，不能据此宣称性能通过；实现成功后必须按 7.3 执行
  P00...P15 prefix 上板验收。

### 7.5 2026-07-28 板级验收

最终 stage 构成为：

| Stage | Cycles | 占比 | 相对 P7-R2 |
|---|---:|---:|---:|
| `CONV_ROW_DATAPATH` | `45,123,992` | `68.86%` | stage 口径已变化 |
| `PPU_ROW_CONSUME` | `1,112,352` | `1.70%` | stage 口径已变化 |
| **`Conv+PPU`** | **`46,236,344`** | **`70.55%`** | **`+1,930,456`** |
| `PPU_BLOCK5_FINAL` | `8,256,942` | `12.60%` | `0` |
| `VEC_FIXED` | `7,564,935` | `11.54%` | `0` |
| `AVGPOOL` | `3,048,809` | `4.65%` | `0` |
| Frame/control/weight/idle | `426,561` | `0.65%` | 计数噪声 |
| **Total** | **`65,533,591`** | **`100%`** | **`+1,930,434`** |

单独比较 `Conv` 或 `PPU` 会被 stage 边界移动误导；`Conv+PPU` 才是跨版本
可信口径。整网回退与 `Conv+PPU` 增量只差 `22 cycles`，证明 BLOCK5、
Vec、AvgPool 和 frame/control 均不是本轮问题来源。

逐 exec 的合并 delta 如下：

| Exec/模式 | 当前 | P7-R2 | 相对 R2 | 结论 |
|---|---:|---:|---:|---|
| U2，C3 paired | `12,682,412` | `11,963,148` | `+719,264` | 回退 |
| U7，C19 paired | `2,718,684` | `2,325,772` | `+392,912` | 回退 |
| U20，5x C12 paired | `12,653,419` | `11,921,859` | `+731,560` | 回退 |
| U21，C64 1x1 | `1,009,804` | `854,412` | `+155,392` | 小幅回退 |
| U35，5x C12 paired | `13,308,779` | `12,577,219` | `+731,560` | 回退 |
| U39，fixed vec | `5,794,375` | `5,794,375` | `0` | 不变 |
| U40，C131 wide | `2,074,100` | `2,650,956` | `-576,856` | 明显收益 |
| U53，5x C25 unpaired | `3,801,213` | `3,941,477` | `-140,264` | 收益 |
| U54，C128 1x1 | `472,204` | `447,372` | `+24,832` | 基本不变 |
| U68，5x C25 unpaired | `4,128,893` | `4,269,157` | `-140,264` | 收益 |
| U71，fixed vec | `1,770,566` | `1,770,566` | `0` | 不变 |
| U72，classifier/output | `1,643,790` | `1,611,470` | `+32,320` | 基本不变 |

证据表明两级架构适合 `C131 wide` 和 `C25 unpaired`，但当前通用 loader
不适合 `C3/C19/C12 paired`：

1. `window_row_loader` 每个 issue 固定扫描最多 6 个 request，并为每个 request
   发送 metadata；steady-state 实际只更新 2 或 4 列。
2. 每个更新列都进入 `window_loader_emit_column` 的 `3 x
   MAX_3X3_CACHE_CHUNKS` 非流水循环；窄模式实际每行只有一个 chunk。
3. paired assembler 再消费 6 个 metadata，并两次调用
   `stage_narrow_3x3_window_select`。对于 `k_tiles=1/4/6` 的层，控制、stream
   handshake 和 cache staging 已超过 SA 有效工作量。
4. csynth 中 `on_chip_memory_read_aligned_tensor_word` 的有效读延迟仅 1 cycle；
   真正膨胀来自重复 helper/cursor/address/control 调度，而不是片上 RAM 本体。
5. 当前每 issue 约为 U2 `193.5 cycles`、U7 `165.9 cycles`、U20/U35
   `122.2 cycles`，远高于对应 SA 的 `1/6/4` 个 k-tile issue，确认 SA 长时间
   等待 WinGen。

Round 2 的原始板级 gate 全部失败：

```text
Conv+PPU       46.236M > 34.5M
U2 Conv+PPU    12.682M > 9.8M
U20 Conv+PPU   10.016M > 7.8M
U35 Conv+PPU   10.016M > 7.8M
RTL total      65.534M > 54.0M
```

### 7.6 Round 2R：窄 paired steady-run 修复

本轮只修改窄 paired 3x3 供数，冻结 wide C131、unpaired C25、1x1、SA、
PPU、BLOCK5、Vec 和 AvgPool。不得用 `in_c/kernel/dilation` 在 HLS
运行时重新理解网络形状。

1. 在 `tools/export_int8_hw_blob.py` 和
   `ESP_INT8_hls/include/npu_schedule.hpp` 中使用现有
   `window_sched_desc_t.reserved[10]` 编译 warmup/steady 各最多 3 个
   physical-column run。固定编码为：

   ```text
   reserved[0] = warmup_run_count
   reserved[1..3] = {signed input_col_delta[7:0], count[15:8]}
   reserved[4] = steady_run_count
   reserved[5..7] = {signed input_col_delta[7:0], count[15:8]}
   reserved[8] = {steady_phase0_cols[15:8], warmup_phase0_cols[7:0]}
   reserved[9] = 0
   ```

   delta 相对当前 paired issue 的第一个输入窗口基准列。C3/C19 steady
   编译为一段连续 4 列，C12 steady 编译为一段连续 2 列；dilation warmup
   最多拆为 3 段、每段 2 列。`phase0_cols` 仅用于 C3/C19 stride-2：
   5 个物理列会复用 4-slot cache，assembler 必须在后两列覆盖 alias 前快照
   第一个窗口；其余 schedule 编译为 0。PARAM 仍为 v4、descriptor 仍为
   128 bytes。
2. 同步更新 `tools/hw_param_replay.py`、`tools/test_p7_contracts.py` 和
   `ESP_INT8_hls/src/param_dma.cpp`。contract 必须证明 run 展开后与现有
   warmup/steady mask 完全相同，并拒绝越过 6-request 窗口的 descriptor。
3. 在 `ESP_INT8_hls/src/win_gen.cpp` 内新增唯一的 narrow-paired
   steady-run loader。steady issue 只遍历已编译 run，不再执行
   `request=0..5` 扫描，也不再为未更新列发送 metadata。
4. narrow-paired token 改为每 issue 一条窄控制 token，加上实际更新列的
   3-row data token。slot0/slot1 由 compiled schedule 与 issue cursor 直接
   递增得到；assembler 不再为两个 window 各执行一次通用 metadata phase。
5. 将 `window_loader_emit_column` 留给 wide/unpaired 路径。窄 helper 固定读取
   3 行、每行一个 compact chunk，禁止保留
   `MAX_3X3_CACHE_CHUNKS` runtime loop。
6. 在窄 helper 内按 row 计算一次起始 offset，并在一次调用中完成 2/4 个连续
   column 的 aligned-word 复用。先在 WinGen 内保持唯一 cursor owner；只有
   csynth 仍显示 scalar memory call handshake 为主瓶颈时，才在
   `memory.cpp` 增加单一 bounded run gateway。禁止全局 inline
   `read_phys_word`，避免复现综合搜索爆炸。
7. wide/unpaired 保持当前 `configure -> loader -> assembler` 层级和 FIFO；
   不新增第二套 WinGen、SA、cache、FMBUF reader 或 current-output
   中间访存。

验证顺序：

```text
exporter/replay/contracts
-> singleton/deadlogic structure scan
-> WinGen CSim: C3, C19, C12 d1/d16, C131, C25
-> U40 prefix
-> fullres CSim: 0 mismatch
-> csynth audit
-> 100MHz implementation
-> P00...P15 board prefix
```

综合/实现 gate：

```text
narrow steady hierarchy 不含 6-request 扫描
narrow hierarchy 不调用 generic MAX_3X3_CACHE_CHUNKS emit loop
WinGen/SA/PPU/memory owner clone = 0
LUT/FF 增长 <=3%
BRAM36 增长 <=4，URAM 不增长
csynth <=25 min
100MHz legal route，WNS >=0
```

第一阶段板级恢复 gate：

```text
U2 combined       <=11.95M
U7 combined       <= 2.35M
U20 combined      <=11.95M
U35 combined      <=12.60M
U40 combined      <= 2.10M
U53 combined      <= 3.85M
U68 combined      <= 4.18M
Conv+PPU          <=43.7M
RTL total         <=63.0M
ERROR=0，MASK bit-exact
```

通过恢复 gate 后，再以相同结构继续压缩 contiguous run：

```text
U2 Conv+PPU       <= 9.8M
U20/U35 Conv+PPU  <= 7.8M each
Conv+PPU          <=36.0M
RTL total         <=55.0M
```

只有 `Conv+PPU <=36M` 才允许进入 Round 3。若 narrow paired 已恢复但仍高于
该门槛，下一步只评估 bounded aligned-word run gateway；不得通过尾部优化掩盖
WinGen gate 失败。

### 7.7 2026-07-29 Round 2R 实施状态

1. exporter、replay、PARAM DMA 和 HLS ABI 已统一使用上述 run 编码；新
   `PARAM.BIN` 仍为 v4/143424 bytes，量化参数与 exec/uop section 未变化。
2. narrow paired 路径现在由唯一 `window_row_loader_narrow_paired` 直接遍历
   warmup/steady run；每个 row-run 一次完成连续 2/4 列的地址生成和 aligned-word
   cursor 复用，并按实际更新列发送一条 issue token 与 3-row data。不再扫描
   6 个 request、逐列调用 helper，也不再进入 generic chunk loop。
3. assembler 使用两阶段单端口快照。已禁止双窗口同周期 cache read，避免为了
   paired staging 复制 BRAM；odd tail 仍保持固定 token count，第二路输出清零。
4. wide/unpaired C25/C131、1x1、SA、PPU、BLOCK5、Vec、AvgPool 均未修改。
5. 验证结果：
   - structure checker 与全部 Python contracts 通过；
   - WinGen CSim 覆盖 C3/C19、C12 d1/d16、C25、C131，`0 errors`；
   - U40 prefix CSim 通过，`last_error=0`；
   - fullres CSim：`last_uop=74`、`last_error=0`、mask
     `0/524288 mismatch`，输出 SHA256 与修改前一致：
     `655C433884531D4789A48F1C063C416BA6AF67BBED2D0718B1DE211E28FDE2AD`。

6. 首轮 csynth（16:06）未通过资源 gate：
   - estimated clock `7.634ns`，综合 `10m45s`；
   - BRAM18 `1620`、DSP `1191`、FF `276619`、LUT `347983`、URAM `112`；
   - 独立 narrow loader 为 FF `5719` / LUT `12642`，独立 narrow assembler
     为 FF `4533` / LUT `1094`。三行模板读和第二套窗口状态是主要新增逻辑。
7. 第一轮结构修复将三行模板读改为单一顺序行循环，并内联 narrow assembler。
   复综合（18:10）恢复到 `7.300ns / 10m33s`，资源降至 BRAM18 `1620`、
   DSP `1191`、FF `272676`、LUT `341663`、URAM `112`；独立 narrow
   assembler 已从 hierarchy 消失，scheduled WinGen 降至 FF `29231` /
   LUT `29287`。但相对 Round 2 基线仍为 FF `+5.08%`、LUT `+3.10%`，
   尚未满足 `<=3%` gate。
8. 第二轮结构修复进一步把 narrow/generic assembler 合并为单一 issue loop、
   单一 `window0/window1` 状态 owner 和 common emit tail。structure checker、
   全部 Python contracts、WinGen CSim 与 fullres CSim 已通过；fullres 仍为
   `last_uop=74`、`last_error=0`、mask `0/524288 mismatch`，stream 最大深度
   `4736`。
9. 第二轮结构修复复综合（18:47）通过：`7.300ns`，BRAM18 `1620`、DSP
   `1191`、FF `267247`、LUT `341225`、URAM `112`。相对 Round 2 基线，
   FF `+2.986%`、LUT `+2.973%`，均进入 `<=3%` gate；相比上一轮报告再减少
   FF `5429`、LUT `438`。`window_row_assembler_narrow_paired` 已内联为
   common assembler 的填充 helper，未形成第二套 assembler 状态；SA/PPU
   heavy datapath 各保持单实例。报告未出现 `HLS 214-475`、`HLS 200-975`
   或新增 II violation。

截至 2026-07-29，Round 3 仍冻结；当时仅通过 csynth gate，需 implementation
成功且板级 prefix 证明 `Conv+PPU <=36M` 后才允许进入。该条件随后已由
8.6 节的 2026-07-30 板级结果满足。

---

## 8. Round 3：串行尾部收敛

只有 `Conv+PPU <=35M` 后才进入本轮。

### 8.1 BLOCK5 final

目标：`8.257M -> <=6.8M cycles`。

1. 在 row 入口固定四个 source 的 bank/base/stride，pixel loop 只递增物理 word
   cursor，避免每 tile 重走 bank/address mux。
2. 每个像素的 `s0/s1/s2/s3` 只读取一次，继续使用静态 L2/L3 composer 和唯一
   common affine/add/store tail。
3. qparam 在 row/tile-group 外加载；禁止增加第二套 finalizer。
4. 重点检查实现关键路径不再穿过 7 级 BRAM cascade；WNS 不得低于本计划 gate。

### 8.2 Vec fixed

目标：`7.565M -> <=5.2M cycles`。

1. U39 保留 4-byte packetizer，减少 group flush 和重复地址解析；不得恢复
   bytewise pack。
2. U71 保留 cblock-major，qparam 每 cblock 加载一次，连续完成 read/apply/write。
3. 只使用一个 affine/add-affine arithmetic owner；不新增 mode-specific engine。

### 8.3 AvgPool

目标：`3.049M -> <=2.5M cycles`。

1. 保留唯一 C3 inner-fast datapath。
2. 复用 Round 1 的 aligned-word cursor 思路，顺序读取 compact C3 row；
   不新增第二套 pool engine。
3. 输出 writer 维持顺序 packed word 提交，禁止 narrow RMW 回到主路径。

### 8.4 最终 gate

```text
Conv+PPU        <=34.5M
BLOCK5 final    <= 6.8M
Vec fixed       <= 5.2M
AvgPool         <= 2.5M
other           <= 0.5M
RTL total       <=49.5M cycles
ARM timer       <500 ms
```

完成后执行第二次 fullres CSim、csynth、implementation、单图和验证集闭环。

### 8.5 2026-07-30 实施状态

按本轮上板前完成 Round 3 的决定，三项尾部修补已落地：

1. BLOCK5 scratch 在 row 入口验证并固定为 `BANK_FMEM0`，pixel loop 只维护四个
   rolling word cursor；四源初始化/读取不再使用短路布尔链。静态 L2/L3
   composer、factor=4 可布线算术基线和唯一 common affine/add/store tail 保持
   不变。
2. Vec 仍只有 `run_fixed_affine_common` 和一个 affine arithmetic owner。U39
   保留 4-byte packing 语义，连续两个完整 group 合并为一次 8-byte packetizer
   更新，尾块继续使用原 partial-group 路径；U71 的 cblock-major 与 qparam
   block 复用不变。
3. AvgPool 仍只有一个 C3 fast engine。三行 aligned-word cache 改为顺序读取，
   相邻两个输出像素共享 5-column window；旧 32-way pixel store switch 已由
   小型 4-byte group packetizer 替代，输出仍只提交三个完整 256-bit word。

进入综合前 gate：

- `check_p6_hls_structure.py`：PASS；
- `test_p7_contracts.py`：全部 PASS；
- 本轮新增/替代 helper 的静态引用扫描：无未引用 local static；
- fullres CSim：`last_error=0`，`mismatches=0/524288`，
  `invalid_labels=0`，最大 stream depth `4736`。

这是进入综合前的状态记录；三个子补丁的最终板级结论见 8.6 节，不能再用
该阶段的 CSim 结果代替周期判断。

---

### 8.6 2026-07-30 板级验收

本轮功能、计时和实现 gate 均通过，但原 Round 3 性能 gate 只有 Conv 接近：

| Gate | 目标 | 实测 | 结论 |
|---|---:|---:|---|
| Conv + PPU | `<=34.5M` | `34.936M` | 差 `0.436M` |
| BLOCK5 final | `<=6.8M` | `8.257M` | 未改善 |
| Vec fixed | `<=5.2M` | `7.532M` | 仅改善 `0.033M` |
| AvgPool | `<=2.5M` | `3.049M` | 未改善 |
| RTL total | `<=49.5M` | `54.200M` | 差 `4.700M` |

各 prefix 的累计总周期单调增加，stage delta 与对应 exec 类型一致；ARM/RTL
计时一致、`ERROR=0`、mask bit-exact。因此不存在 counter 分桶、PARAM prefix
或功能错误导致的假回退。

Round 3 的代码不整体回退：它已通过 CSim、csynth 和 legal route，且没有使
任一板级 stage 明显回退。但其 BLOCK5/AvgPool 修改只视为结构清理，不再计入
性能收益；若后续资源或时序变差，允许把这两个独立子补丁回退到 0723 等价结构。

---

## 9. Round 4：板级驱动的 50M 收尾

本轮不再尝试全网 segment overlap，也不新增第二套 datapath。执行顺序固定为
Round 4A + Round 4B 后进行第一次 fullres/综合/实现/板级检查；只有实测仍高于
`50M` 才进入 Round 4C。

### 9.1 Round 4A：Fixed Vec resident qparam 与 16-lane 算术

板级证据：

```text
U39: 128*256*ceil(131/32) = 163840 tiles
     5761607 / 163840 = 35.17 cycles/tile
U71: 64*128*8 = 65536 tiles
     1770566 / 65536 = 27.02 cycles/tile
```

U39 比 U71 多出的 `8.15 cycles/tile` 是可优化上限，并不全等于 qparam
开销；其中还包含三源读取和 compact packetizer。当前
`loaded_qparam_block` 只记住上一 block，而 U39 的 `h->w->cblock` 每个像素
都会循环 `0..4`，所以不会形成跨像素 qparam 命中。

实施步骤：

1. 保持唯一 `run_fixed_affine_common`、唯一
   `vec_alu_apply_affine_block` 和现有 U39/U71 loop order，不恢复 bytewise
   store，也不拆 mode-specific engine。
2. exec 入口把实际 `c_blocks<=8` 的 affine qparam 装入一个 bounded resident
   bank。按 lane 完全分 bank、按 block 保持深度 8，不允许
   `aff_q_t qcache[8]` 全维 complete partition 后形成巨型动态 mux。
3. hot loop 只按 `c_blk` 读取当前 8-lane group 的 mul/bias/shift；不再每 tile
   调用全局 `param_dma_get_affine_qparam`。PARAM v4 和 qparam 内容不变。
4. 将 4 个 8-lane group 以 factor=2 执行，使单个 affine block 从 8-lane
   提升到 16-lane 并行。仍是一个算术 owner，预计 DSP 增量约 `16`；禁止
   factor=4 一步扩到 32 lane。
5. HLS 局部门槛：

   ```text
   vec_alu_apply_affine_block latency <=5 cycles
   run_fixed_affine_common DSP increase <=20
   top LUT/FF increase <=2%
   no second affine arithmetic instance
   csynth <=25 min
   ```

板级门槛：

```text
U39 VEC_FIXED <=4.4M
U71 VEC_FIXED <=1.4M
total VEC_FIXED <=5.8M
all Conv/PPU/B5/Pool deltas <=1%
```

若 resident bank 导致高扇出、clone 或 LUT/FF 超门槛，只撤销 resident bank；
可单独保留 16-lane arithmetic。若 arithmetic 不能把局部 latency 降至 5
cycles，则两项均停止，不继续提高 unroll。

### 9.2 Round 4B：编译期窄 packed-row 复用

当前 narrow cache 是 `3 x 64 x 256-bit` 的水平 column ring，共
`18 BRAM18`。每次 `scheduled_window_generator_row` 仍重置 packed-word
cursor，跨输出行不复用输入物理字。可获益的 schedule 只有：

- C3、stride 2、dilation 1：相邻输出行复用 1/3 输入行；
- C12、stride 1、dilation 1：相邻输出行复用 2/3 输入行；
- C19 行宽需要约 304 个 256-bit word，禁止纳入本轮；
- dilation 2/4/8/16 相邻输出行没有直接行重叠，固定走现路径。

实施步骤：

1. 在 `window_sched_desc_t.reserved[9]` 编译 row-reuse word，保持 PARAM v4
   结构和 128-byte descriptor 不变：

   ```text
   bits[1:0]  reuse_mode: 0=none, 1=stride1_keep2, 2=stride2_keep1
   bits[8:2]  packed_words_per_source_row，当前 C3/C12 均为 96
   bits[15:9] 保留为 0
   ```

   exporter、replay、contract 和 PARAM DMA 必须共同检查；HLS 不得根据
   `in_c/stride/dilation` 自行推断模式。
2. **替换** `s_narrow_cache_row0/1/2[64]`，不得并存第二套 cache。新 owner
   为 3 个深度 96 的 256-bit packed-row bank，附带 3 个 source-row tag 和
   valid bit；首次 row 加载 3 行，后续按 compiled mode 只加载 1 或 2 行。
3. narrow loader 直接从 packed-row bank 提取连续 C3/C12 字节并生成现有
   issue/data token。边界 padding 由 row tag 无效时填零；wide/unpaired、
   1x1、SA、PPU 和 weight path 完全冻结。
4. cache 在 conv task/branch 切换、out_row 不连续或 descriptor identity
   变化时显式失效。BLOCK5 每个 branch 内 row 顺序推进，禁止跨 branch 错误
   复用。
5. 存储门槛：

   ```text
   old 18 BRAM18 narrow cache must disappear
   net BRAM increase <=6 BRAM18 (<=3 BRAM36)
   physical BRAM <=93%
   no new URAM
   no dynamic write to complete-partition row array
   ```

板级门槛：

```text
U2 Conv       <=7.38M
U20 Conv      <=5.70M
U35 Conv      <=5.70M
U7/U40/U53/U68 regression <=1%
Conv+PPU      <=31.5M
```

### 9.3 第一次检查点

Round 4A/4B 均完成后只进行一次完整闭环：

```text
exporter/replay/contracts
-> singleton/deadlogic scan
-> focused Vec/WinGen CSim
-> fullres CSim, 0 mismatch
-> csynth audit
-> 100 MHz implementation
-> P00...P15 board prefix
```

进入板级前将 APP 的 `reference_cycles` 更新为 `54,200,473`，将 pass gate
从临时 `54,000,000` 改为真实 `50,000,000`。第一次检查点的目标为：

```text
Conv+PPU <=31.5M
Vec      <= 5.8M
RTL total <50.0M，工程目标 <=49.5M
```

### 9.4 Round 4C：只选一个补差项

第一次检查点已 `<50M` 时立即停止。若仍有缺口：

1. 缺口 `<=0.8M`：优先 AvgPool 双像素。保留唯一 C3 engine，在现有
   32-pixel group 内用两个独立三通道 accumulator 同时处理偶/奇像素，
   仍按顺序拼成三个 256-bit word。目标 `3.049M -> <=2.3M`，DSP 增量
   `<=28`。
2. 缺口 `>0.8M`：只评估 BLOCK5 common tail 的 factor=8 lane arithmetic。
   保持静态 L2/L3 composer、四 source cursor 和唯一 emit owner，不做
   DATAFLOW/ping-pong，不新增 scratch。目标 `8.257M -> <=7.2M`，DSP 增量
   `<=40`。
3. 两项不得同轮实施。任何一项若使 7 级 BRAM cascade 路径 WNS 变负、route
   失败或独立板级 delta 非负，立即回退。

### 9.5 Round 5：可选静态填充

该轮只在最终差距 `<=0.6M cycles` 时评估。候选仍限 C25/C28 ragged
output-lane packing 和 classifier 低 Cout 多像素 packing；理论全网上限约
`0.562M cycles`。禁止第三路 activation stream、第二套 SA、扩大 TM/TK、
大型 reorder buffer或额外写回 pass。

---

## 10. 停止规则与最终验收

任一情况出现即回退当前独立子补丁：

- WinGen/SA/post/memory owner clone；
- 恢复 segment materialization、cache restart 或 output copy；
- 新增 URAM，physical BRAM 超过 93%，CLB 超过 95.5%；
- `HLS 214-475`、`HLS 200-975`、综合超过 25 分钟或搜索停滞；
- 100 MHz routing 失败或 WNS < 0；
- fullres `MASK.BIN` 不再 bit-exact、`ERROR!=0` 或最终不回到 IDLE；
- 板级优化项 `delta>=0`。

最终成功标准：

```text
100 MHz
single WinGen / 32x32 SA / postprocess / memory owner
RTL MODE_RUN <50,000,000 cycles
ARM timer <500 ms
full-resolution mask无精度退化
legal route and timing pass
```

Round 3 的 `54,200,473 cycles` 保留为 Round 4 对照基线。Round 4A/4B 已按
单一检查点完成验证，其板级结果见 10.2。

### 10.1 2026-07-30 Round 4A/4B 实施状态

- Round 4A：fixed Vec 在 exec 入口一次装载 lane-partitioned、深度 8 的
  resident qparam；hot loop 不再访问 PARAM，全网仍只有一个 affine block
  owner；4 个 8-lane group 改为每拍并行 2 组，即 16-lane 算术。
- Round 4B：PARAM v4 `window_sched_desc.reserved[9]` 已由 exporter 编译并由
  HLS/replay/contracts 共同校验。仅 C3/s2/d1 与 C12/s1/d1 启用
  `96-word` 三行复用；C19、dilation 路径保持原实现。
- 原 `3x64` narrow ring 已被 `3x96` 单 owner bank 替换；复用 loader 直接从
  packed row 提取窗口，descriptor/branch/非连续行会显式失效。旧
  narrow stage/run helper 已清除，assembler 不访问 row bank。
- 验证：structure checker、全部 Python P7 contracts、WinGen 新旧路径专项、
  U40 prefix 与轻量 prefix 均通过；fullres CSim 为
  `last_error=0`、`0/524288 mismatch`。
- 下一门槛为 csynth：重点核对 affine block latency/DSP、narrow cache BRAM
  净增量、singleton hierarchy、综合耗时与 10 ns 时序；通过后再做实现和
  `P00...P15` 板级检查。

### 10.2 2026-08-03 Round 4A/4B 板级验收

本轮 app/platform/SD 三方已对齐 `P7-R4`。`P15.BIN` 与完整 `PARAM.BIN`
bit-exact，ARM/RTL 计时一致，所有 prefix 均 `ERROR=0` 且最终回到 IDLE。
最终 `MASK.BIN` 与既有 fullres golden SHA256 完全一致。

| Stage | Round 3 | Round 4 | 变化 |
|---|---:|---:|---:|
| Conv row datapath | `33,823,468` | `32,229,027` | `-1,594,441 / -4.71%` |
| PPU row consume | `1,112,352` | `1,112,352` | `0` |
| BLOCK5 final | `8,256,942` | `8,256,942` | `0` |
| Vec fixed | `7,532,167` | `6,155,931` | `-1,376,236 / -18.27%` |
| AvgPool | `3,048,809` | `3,048,809` | `0` |
| Frame/control/weight/idle | `426,735` | `426,754` | `+19` |
| **RTL total** | **`54,200,473`** | **`51,229,815`** | **`-2,970,658 / -5.48%`** |

原 gate 的验收情况：

| Gate | 目标 | 实测 | 结论 |
|---|---:|---:|---|
| Conv + PPU | `<=31.5M` | `33.341M` | 差 `1.841M` |
| Vec total | `<=5.8M` | `6.156M` | 差 `0.356M` |
| RTL total | `<50.0M` | `51.230M` | 差 `1.230M` |
| U39 Vec | `<=4.4M` | `4.779M` | 有收益，未过 gate |
| U71 Vec | `<=1.4M` | `1.377M` | 通过 |

逐 exec 证明 Round 4B 不能作为统一 C3/C12 row-reuse 方案保留：

| Exec | Round 3 stage | Round 4 stage | 变化 | 判断 |
|---|---:|---:|---:|---|
| U2 C3/s2 Conv | `9,060,704` | `6,238,975` | `-2,821,729 / -31.14%` | 明确有效 |
| U7 C19 Conv | `1,911,088` | `1,958,624` | `+47,536 / +2.49%` | 小幅回退 |
| U20 C12/s1 Conv | `6,578,146` | `7,374,656` | `+796,510 / +12.11%` | 明确负收益 |
| U35 C12/s1 Conv | `6,578,146` | `7,374,656` | `+796,510 / +12.11%` | 明确负收益 |
| U40 C131 Conv | `2,074,024` | `1,821,776` | `-252,248 / -12.16%` | 共享结构带来收益 |
| U53/U68 C25 Conv | `2,801,880` | `2,721,370` | 各 `-80,510` | 小幅收益 |
| U39 fixed Vec | `5,761,607` | `4,778,578` | `-983,029 / -17.06%` | Round 4A 有效 |
| U71 fixed Vec | `1,770,566` | `1,377,359` | `-393,207 / -22.21%` | Round 4A 通过 |

代码证据与板级结果一致：`window_row_loader_narrow_reuse()` 对每个像素调用
`window_loader_emit_narrow_reuse_window()`，后者重新遍历 3x3 并读取 9 个 tile；
它绕过了旧 narrow loader 的 `window_loader_update_narrow_run()` 水平列缓存。
C3/s2 从减少整行物理读取中获得净收益，但 C12/s1 丢失相邻窗口的水平列复用，
整行缓存收益不足以抵消重复 3x3 提取。

### 10.3 Round 4B-R：C12 selective rollback

在进入 Round 4C 前先执行低风险修复：

1. exporter 仅为 C3/s2/d1 编译 `STRIDE2_KEEP1`；C12/s1/d1 恢复
   `WINDOW_ROW_REUSE_NONE`，继续走已经板级验证的 narrow paired steady-run。
2. 不回退 Round 4A，不修改 C3 row bank、SA、PPU、BLOCK5、AvgPool、PARAM v4
   ABI 或物理阵列维度；C19/dilation/wide 路径继续冻结。
3. contracts 必须断言 C3 开启 reuse、所有 C12 schedule 关闭 reuse。重新导出
   PARAM 与 P00..P15，跑 C3/C12 focused test 和一次 fullres CSim。
4. csynth/implementation gate 继续要求 singleton、100 MHz legal route 和
   WNS>=0；随后只做一次 prefix 板级检查。

按 Round 3 的 C12 实测恢复估算：

```text
51,229,815 - 2 * (7,374,656 - 6,578,146)
= 49,636,795 cycles
```

Round 4B-R 板级 gate：

```text
U2 Conv      <=6.30M
U20 Conv     <=6.65M
U35 Conv     <=6.65M
U39 Vec      <=4.80M
U71 Vec      <=1.40M
RTL total    <50.0M（工程目标 <=49.8M）
ERROR=0，MASK bit-exact
```

若 selective rollback 后仍高于 50M，再按 9.4 只选择一个 Round 4C 补差项；
不得同时修改 BLOCK5 与 AvgPool，也不得把 C12 direct row-reuse 再次接回主路径。
