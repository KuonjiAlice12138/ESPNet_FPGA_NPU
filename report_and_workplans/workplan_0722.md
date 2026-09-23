# Workplan 0722 - P7 R3 overlap 回归分析与恢复计划

## 0. 结论

**2026-07-23同artifact A/B更正：** 下表R2数据来自一次在exec 12后进入
ERROR的截断运行，不是完整整网结果。使用同一组P00-P15后，R2完整P15为
`63,603,157 cycles`，当前恢复版为 `64,452,405 cycles`，真实净回退为
`849,248 cycles / 1.34%`。旧表用于记录R3失败现象，但不得再用于当前性能gate；
有效归因与后续计划以7.7为准。

2026-07-22，`INT8-BOARD-20260722-P7-R3` 在 100 MHz 下完成单图整网上板：

| 项目 | R2 | R3 | 变化 |
|---|---:|---:|---:|
| RTL `MODE_RUN` | `60,219,515 cycles` | `84,246,007 cycles` | `+24,026,492`，`+39.90%` |
| ARM 实测延时 | `602 ms` | `842 ms` | `+240 ms` |
| `CONV_ROW_DATAPATH` | `32,683,712` | `44,406,656` | `+35.87%` |
| `PPU_ROW_CONSUME` | `10,010,706` | `20,524,280` | `+105.02%` |
| `PPU_BLOCK5_FINAL` | `8,256,942` | `8,273,190` | `+0.20%` |
| `VEC_FIXED` | `5,794,399` | `7,564,935` | `+30.56%` |
| `AVGPOOL` | `3,048,809` | `3,048,809` | 不变 |

RTL total 与 ARM timer 完全同口径，性能回退可信。R3 的 `MASK.BIN` 与同组 golden 为 `0/524288 mismatch`，PA/mIoU 完全一致，`ERROR=0`；PARAM、输入、bitstream、platform 和 app 版本也已核对一致。因此 R3 是**功能正确、实现通过，但 overlap 架构负优化**，不是数据或平台版本错误。

本轮决定：**撤销 R3 overlap 执行路径，但不整版回退到 R2。** 保留 R1/R2 的 BLOCK5/Vec 优化、R3 的正确结束状态、PARAM v4、单 WinGen/SA/PPU owner及已验证的资源/时序修复。

---

## 1. R3 回退的根因

### 1.1 实际执行不是预期的三阶段 overlap

当前 `conv_engine.cpp` 每个 segment 实际执行：

```text
prefetch_window_segment
  = WinGen -> act buffer materialization

overlap_compute_and_consume_segment
  = max(act buffer replay -> SA -> post, previous segment PPU)

stage_segment_for_consumer
  = current output buffer -> previous-output buffer copy
```

因此总周期是：

```text
sum(prefetch) + sum(max(compute, consume)) + sum(segment copy)
```

WinGen/cache fill 没有与 SA 或 PPU 重叠；激活还额外经历一次 buffer 写入和一次 replay。当前始终由 `s_segment_out_buf0` 计算，再复制到 `s_segment_out_buf1`，并不是真正的无拷贝 ping/pong。

### 1.2 segment 破坏了 WinGen 的行内 cache 生命周期

`scheduled_narrow_3x3_window_row()` 和 `scheduled_wide_3x3_window_row()` 的 cache tag 在每次函数调用时重新初始化。PARAM v4 将 26 个 Conv 描述符的 `2624` 个输出行拆为 `33,472` 次 segment 调用，调用数扩大 `12.76x`：

| WinGen mode | 行数 | segment 调用 | 调用倍率 |
|---|---:|---:|---:|
| first C3 | `256` | `2,048` | `8x` |
| narrow C12 | `1,280` | `10,240` | `8x` |
| wide C25 | `640` | `10,240` | `16x` |
| staged C19 | `128` | `1,664` | `13x` |
| staged C131 | `64` | `8,192` | `128x` |

C131 的 `k_tiles=37`，受 `CONV_SEGMENT_ACT_WORD_CAP=64` 限制后得到 `segment_out_w=1`。它每个输出像素都重新建立 wide 3x3 cache，直接违背 Round 3“cache 在整行内持续有效”的前提。这是 `CONV_ROW_DATAPATH` 增加 `11.72M cycles` 的首要来源。

### 1.3 当前 stage 数据不能把 PPU 增量理解为纯 PPU

进入 `overlap_compute_and_consume_segment()` 前，只要上一 segment 可消费，`prof_stage_id` 就被设为 `PPU_ROW_CONSUME`。该 bin 实际覆盖 `max(current compute, previous consume)`，不是纯 PPU 时间；R2 与 R3 的 `PPU_ROW_CONSUME` 语义已经不同。

因此可信结论是：

- R3 total 比 R2 多 `24.03M cycles`；
- `44.41M` 主要是串行 prefetch/materialization；
- `20.52M` 是 Conv compute 与 PPU 的组合区间；
- 不能用 `R3 PPU - R2 PPU` 单独归因，但可以确认当前 overlap 没有抵消新增工作量。

### 1.4 综合/实现不是本轮失败点

R3 csynth 为约 `7.30 ns`，single WinGen/SA/compact PPU结构成立；实现后合法布线，100 MHz post-route `WNS=+0.111 ns`、`WHS=+0.010 ns`。实际资源约为 LUT `59.68%`、BRAM `92.07%`、URAM `100%`、DSP `33.76%`。

现有高风险项仍包括 C131 WinGen `II=2`、compact PPU packetizer `II=2` 和两项 `HLS 200-1449`，但它们不能解释本次最主要的 `24.03M` 系统级回退。优先级应先放在错误的 segment/caching 粒度，而不是继续局部压 II。

---

## 2. 回退边界

### 2.1 必须撤销

1. 删除生产主路径中的 `prefetch_window_segment -> capture -> replay` 激活物化链路。
2. 删除 `stage_segment_for_consumer` 输出复制；不得保留“固定写 buf0、再复制到 buf1”的伪 ping/pong。
3. exporter停止为全部 Conv 编译 `segment_out_w`/overlap flag；contracts同步禁止当前全覆盖策略。
4. WinGen恢复每个输出行只初始化一次 cache，并直接通过 stream连接唯一 SA。
5. 当前 R3 overlap代码由 Git历史保留，不在生产源码中保留宏关闭的 legacy 分支。

### 2.2 必须保留

1. P7 PARAM v4 ABI、schedule驱动 WinGen和现有量化/布局行为。
2. R1/R2 的 BLOCK5 source cache、Vec 4-byte packetizer与 cblock-major优化。
3. R3 已验证的 `ERROR=0`、最终回到 `IDLE` 的控制修复。
4. single WinGen、single SA、single compact PPU/memory owner；不得恢复双 consumer或依赖 allocation directive限制 clone。
5. 当前 full-resolution输出和 bit-exact mask行为。

这不是 `git checkout` 整版 R2。应以函数级改动恢复串行 row datapath，避免丢失 R2之后已经验证有效的正确性、资源和时序修复。

---

## 3. 下一步执行计划

### Round 1：恢复无额外物化的串行可信基线

1. 先保存 R3 源码、artifact、综合/实现报告和板级 counter，建立可复现实验锚点。
2. `conv_engine.cpp` 恢复单一 row 调用：`scheduled WinGen -> SA -> postprocess -> shared row buffer`；WinGen直接流向SA，不经过 segment act buffer。
3. 普通 Conv 行完成后，只调用一次现有 compact PPU owner：`ow_begin=0`、`ow_count=out_w`、`row_begin=row_end=true`。不得恢复第二套 full-row PPU函数。
4. BLOCK5 final、upsample、Vec、AvgPool保持当前实现，不在本轮顺手修改。
5. `win_gen.cpp` 保留 PARAM mode dispatch，但入口按整行执行，cache/tag每行初始化一次。
6. exporter清零 overlap reserved/flag；replay、contracts和structure checker同步到单一路径。
7. 清除 R3-only segment act/output buffer、capture/replay/copy helper和无引用逻辑。

验证顺序：

```text
structure/contracts -> U40 prefix -> fullres CSim
-> csynth audit -> 100MHz implementation -> 单图上板
```

Round 1 gate：

```text
ERROR=0，最终 stage=IDLE
fullres mask mismatch=0
single WinGen/SA/compact PPU owner
CONV_ROW_DATAPATH <=34.2M
PPU_ROW_CONSUME <=10.5M
PPU_BLOCK5_FINAL <=8.5M
VEC_FIXED <=6.0M
RTL_STAGE_TOTAL <=62M cycles
100MHz 合法布线且 timing pass
```

若恢复后仍高于 `62M`，先核对 stage语义和是否仍存在 act materialization；不得直接开始第二轮 overlap。

### Round 2：校准 overlap 的可观测性与收益模型

1. profiling接口将 `prefetch_busy`、`conv_compute_busy`、`ppu_busy` 和 `both_busy` 分开统计；单值 `prof_stage_id` 不再用于解释并发区内部归因。
2. exporter按 descriptor输出静态成本表：`out_h/out_w/k_tiles/segment_w/segment_count`、预计 act materialization words、cache重启次数和可隐藏 PPU cycles。
3. 明确禁止 `segment_out_w=1` 的 C131进入 overlap；初始候选要求 `segment_w>=16`、`segment_count<=16`，且预测隐藏收益大于 capture/replay/fill/drain成本。
4. 用单 descriptor白名单做板级 A/B测试，每次只打开一类 schedule；只有 RTL total 相对串行基线下降才允许扩大覆盖面。

Round 2 只建立可信模型，不同时改 BLOCK5/Vec/AvgPool。

### Round 3：重新实现受控 overlap

1. cache/tag生命周期提升到整行：segment之间保留 WinGen cache状态，禁止每段重新初始化。
2. 实现真正的交替 ping/pong output buffer：Conv直接写当前 bank，PPU直接读上一 bank，不允许中间复制。
3. act buffer只在确有 overlap收益的 descriptor中使用；不得为统一调度让所有 Conv都经历 capture/replay。
4. C131默认维持串行整行流；只有专门解决 `k_tiles=37` 导致的容量问题并通过单层A/B后才可开放。
5. PARAM只编码编译后的 overlap mode/segment，不在HLS运行时根据shape推断。
6. 保持新增 URAM为0；BRAM增量不超过2个BRAM36，且实现后BRAM、routing和100MHz时序必须通过。

Round 3 gate：

```text
每个启用 overlap 的 descriptor：board delta < 0
cache restart = 1次/row，而非1次/segment
segment output copy = 0
both_busy有实测值，且visible PPU显著下降
CONV_ROW_DATAPATH不高于串行基线3%
RTL_STAGE_TOTAL <50M cycles
```

若受资源约束无法满足持久 cache和无拷贝 ping/pong，则停止 overlap路线，冻结 Round 1 基线，转向 BLOCK5/Vec/AvgPool 的确定性串行优化；不得再次以全网 segment化换取理论 overlap。

---

## 4. 停止规则

出现以下任一情况立即回退当前轮次，不叠加补丁：

- WinGen调用次数再次从 `2624 rows` 膨胀为数量级更高的 segment调用；
- C131使用 `segment_out_w<16`；
- activation或output新增“写buffer后再读/复制”的串行阶段，却没有板级隐藏收益；
- HLS出现 WinGen/SA/PPU/memory owner clone、`HLS 214-475` 或综合搜索失控；
- URAM增加、BRAM超过实现余量、routing congestion或100MHz时序失败；
- fullres mask精度下降或结束状态非 `IDLE/ERROR=0`。

本轮优先恢复 R2级性能和R3级正确性。500 ms目标继续保留，但下一次 overlap必须由逐层板级A/B数据证明，而不能仅凭内层 DATAFLOW interval或HLS latency估计判定成功。

---

## 5. Round 3 实施结果（2026-07-22）

本轮采用比“输出 ping/pong + copy”更直接的无拷贝实现：WinGen 先把一个编译期受限的 segment 写入本地 activation BRAM，随后在同一 DATAFLOW region 内执行：

```text
activation replay -> single SA -> requant stream -> single PPU/sink
```

关键落实项：

- WinGen cache/tag 生命周期提升到整行，segment 间不再清空；只在 `ow_begin=0` 时复位 tag。
- postprocess 对每个有效输出像素恰好产生一个 `act_vec_t`，PPU 即使遇到状态错误也先按 `ow_count` 排空 stream，避免定长生产/消费失配。
- compact Conv 由 PPU 直接完成 CAT/affine/pack/store；BLOCK5 final 与 classifier/upsample 仍使用唯一共享 row buffer，不新增 current-conv 中间访存。
- 删除旧 `s_segment_out_buf*`、output staging/copy、旧 row consumer 与旧 WinGen row 入口；single WinGen/SA/PPU owner 保持不变。
- PARAM v4 编译 segment gate：`segment_w>=16`、每行最多16段；C131 为 `17 pixels/segment`、`8 segments/row`。全网编译 segment 调用从旧 R3 的 `33,472` 降至 `10,496`，cache restart 为每行一次。
- activation buffer 为 `640x256b + 128x256b`，均显式绑定 BRAM并关闭冗余 reset；综合后必须重点核查 BRAM增量、clone count与 routing。

验证结果：

- structure checker：通过，旧 overlap/copy/row API未进入源码主路径。
- Python contracts：全部通过；严格 exporter 为 PARAM v4、75 UOP、16 exec entry、0 warning。
- PARAM replay：U2 fused B1与U20输出均对QAT golden达到 `0 mismatch`。
- 唯一一次 fullres CSim：`last_uop=74`、`last_error=0`、mask `0/524288 mismatch`、非法标签0、Vitis `0 errors`；最大 stream 深度629。

当前尚未完成的是 csynth/implementation/board 性能验收。下一步由 GUI 手动综合，必须核查：

```text
single scheduled_window_generator_segment / systolic_array_core_row /
post_process_conv_stream / ppu_consume_conv_stream
无 HLS 214-475、HLS 200-975、datapath clone
BRAM/URAM/LUT/DSP可实现，100 MHz timing可收敛
```

本版 overlap 的实际边界是“WinGen segment预取串行 + SA/post/PPU像素流并行”，并未声称 WinGen 与 SA 同时运行。是否获得净收益必须以综合后上板 RTL total 为准；若总周期未低于 R2，则直接依据 stage counter 判断 activation materialization 是否抵消了 PPU隐藏收益，不继续叠加补丁。

---

## 6. 综合后恢复轮次（2026-07-22）

上一版 csynth 已触发停止规则：BRAM18 `1579/1488`，其中两个 segment activation buffer 占 `23 BRAM18`；同时仍有 DATAFLOW form warning、C131 WinGen `II=2` 和 compact PPU packetizer `II=2`。因此本轮不再修补全网 activation materialization，而是恢复 Round 1 的可信串行基线：

1. Conv 每行只调用一次 `scheduled WinGen -> single SA -> requant -> shared row buffer` DATAFLOW；删除 segment act buffer、prefetch/capture/replay、segment cfg/scheduler 和相关主路径。
2. compact PPU 仍保持唯一 stream owner，但改为每行结束后只调用一次；row buffer 以 II=1 顺序送入，不恢复第二套 generic row consumer。
3. compact packetizer 的跨 segment static 状态改为单行局部状态，并以 8 个固定 32-bit group 组装 256-bit word，避免宽字逐组移位。
4. C131 wide cache 按 channel chunk 分 bank，尝试消除多 chunk 读取造成的 II=2；不改 SA/TM/TK，不新增 BRAM/URAM。
5. exporter 清零 PARAM v4 schedule/conv descriptor 的 overlap flag 与 reserved；ABI、量化、布局和 exec plan 均保持不变。

验证结果：

```text
structure checker: PASS，single WinGen/SA/compact PPU owner
Python contracts: 11/11 PASS
PARAM v4 export: 75 UOP，16 exec，0 warning，segment calls=0
U72 PARAM replay: U02/U20/U72 均 0 mismatch
fullres CSim: last_uop=74，last_error=0
fullres mask: 0/524288 mismatch，invalid_labels=0，Vitis 0 errors
```

下一步只进入 csynth audit，不继续改代码。必须确认：

```text
无 HLS 214-113/200-471/214-475/200-975
segment activation buffer 从 memory/resource table 消失
shared_conv_row_engine / scheduled_window_generator_segment /
systolic_array_core_row / consume_compact_conv_row 均无 clone
C131 WinGen 与 compact PPU 的 final II 不劣于上一版
BRAM、LUT、DSP及估计时钟具备100 MHz实现余量
```

通过后再进入100 MHz implementation与板级 counter验收；当时采用的
`<=62M cycles / R2=60.22M`为临时gate，已被7.7同artifact A/B结果取代。

---

## 7. 2026-07-23 恢复版上板结果与后续路线

### 7.1 实测结果

`INT8-BOARD-20260723-P7-R3` 使用同组 PARAM v4 artifact，在100 MHz下完成单图整网：

> 更正：下表的R2列来自7月20日旧artifact运行，结束状态为
> `ERROR/current_exec=12`，实际只执行到P13边界，不能作为完整整网基线。
> 同一组P00-P15 artifact的有效A/B结果见7.7；旧表仅保留为问题发现记录。

| Stage | R2 | 失败 overlap R3 | 当前恢复版 | 当前相对R2 |
|---|---:|---:|---:|---:|
| `CONV_ROW_DATAPATH` | 32,683,712 | 44,406,656 | 44,042,816 | +11,359,104 |
| `PPU_ROW_CONSUME` | 10,010,706 | 20,524,280 | 1,112,352 | -8,898,354 |
| `PPU_BLOCK5_FINAL` | 8,256,942 | 8,273,190 | 8,256,942 | 0 |
| `VEC_FIXED` | 5,794,399 | 7,564,935 | 7,564,935 | +1,770,536 |
| `AVGPOOL` | 3,048,809 | 3,048,809 | 3,048,809 | 0 |
| `RTL_STAGE_TOTAL` | 60,219,515 | 84,246,007 | 64,452,437 | +4,232,922 |

ARM timer为 `21,484,134 ticks / 33.333 MHz = 645 ms`，与RTL的 `64,452,437 cycles / 100 MHz` 同口径。最终状态为 `IDLE`，`ERROR=0`。实现合法布线，post-route `WNS=+0.145 ns`、`WHS=+0.010 ns`。

`E:\MASK.BIN` 与同组 golden：

```text
524288 / 524288 bytes checked
mask mismatch = 0
invalid labels = 0
PA   = 0.97019055
mIoU = 0.92065819
```

评估记录：`hw_artifacts/sched_v4_p7_0702/board_20260723_mask_metrics.json`。

### 7.2 结论

恢复版相对失败 overlap R3 减少 `19.79M cycles`（`-23.50%`），证明移除
segment activation materialization 与 cache 重启是正确的。原先由错误截断的
R2结果推导出的“回退4.23M、Vec回退1.77M”结论无效。

同artifact A/B表明：当前版为 `64.452M`，R2为 `63.603M`，真实净回退仅
`0.849M cycles / 1.34%`；`VEC_FIXED`、BLOCK5 final、pool均未回退。
当前 `CONV` 增加 `10.856M`、可见 `PPU` 减少 `10.006M`，必须合并评价，
不能按stage名把工作迁移误判为卷积退化。

旧Round 2/3的全网segment overlap计划仍不恢复。后续不得重新引入activation
capture/replay、segment cache重启或output copy，只允许按7.7定位到的正回退
descriptor做定点修改。

### 7.3 Round 1B：逐 exec 归因，不改HLS数据流

1. 冻结当前bitstream、PARAM、app和本节counter作为可信基线。
2. 工具侧由同一 `PARAM.BIN` 生成短文件名 `P00.BIN...P15.BIN`：`P00` 在pc0终止，作为零exec/frame-load基线；`P01`执行pc0，依此类推，`P15`为原始完整PARAM。每个截断文件只改一个exec `kind`字节为 `EXEC_END`，不改descriptor、qparam或布局。
3. app增加临时prefix profiling模式：一次启动依次对16个PARAM执行 `MODE_INIT -> MODE_RUN -> RTL counter snapshot`，逐项即时输出 `PREFIX_CUM` 和相邻前缀的 `PREFIX_DELTA`；`P15`必须复现 `64,452,437`，误差不超过0.1%，并保存最终 `MASK.BIN`。
4. 分别用保留的R2 bitstream和当前bitstream运行同一组prefix PARAM，形成逐exec A/B表；该项已完成，原先根据ERROR截断结果推导的 `CONV+PPU +2.461M` 和 `VEC +1.771M` 已被7.7纠正。
5. 本轮不改HLS、不跑fullres CSim；prefix结束状态必须均为 `IDLE/ERROR=0`。

只有完成逐exec归因后才允许修改热点代码，禁止继续依赖HLS max latency猜测整网瓶颈。

### 7.4 Round 1C：收敛同artifact A/B正回退

按逐exec结果只修出现回退的descriptor：

1. 冻结U2和U40路径：相对R2分别改善约180k和252k cycles，不得被统一回退覆盖。
2. 第一优先级只检查U20/U35两个BLOCK5 descriptor。两者合计正回退
   `814,718 cycles`，已解释全网净回退的95.9%；其BLOCK5-final周期完全一致，
   问题限定在五个branch conv的 `CONV+PPU` 行级调度。
3. 第二优先级检查U7/U21的compact C12 store；二者各回退约158k cycles。
   U53/U68各43k、U54约26k、U72约38k只作为第二批，不与首批同时修改。
4. `VEC_FIXED`、pool和BLOCK5 final在A/B中逐周期一致，全部冻结；不得继续修改
   Vec、AvgPool或静态tile composer。
5. 保持整行一次cache初始化、single WinGen/SA/weight/PPU owner；禁止segment
   materialization、第二套cache、扩大SA或全网统一新路径。
6. stage gate使用 `CONV_ROW_DATAPATH + PPU_ROW_CONSUME`，不再单独约束可见PPU，
   因为两版stage边界不同且当前实现已把大量consumer工作移入conv区间。

阶段gate：

```text
U20 delta <=11,929,888
U35 delta <=12,585,250
CONV_ROW_DATAPATH + PPU_ROW_CONSUME <=44,305,888
RTL_STAGE_TOTAL <=63,603,157
ERROR=0，fullres mask mismatch=0
```

先恢复同artifact R2总周期，再进入确定性热点优化；不以旧的60.219M截断值作为gate。

### 7.5 Round 2：确定性热点收敛

在同artifact R2基线恢复后按逐exec占比依次处理：

1. WinGen：1x1 aligned路径尝试顺序读/emit `II=1`；C25/C131只减少已定位的cache read或mode mux开销，不复制cache/emit engine。
2. BLOCK5 final：保持静态tile composer和唯一common tail，只减少重复scratch读、qparam加载及固定tile间控制空泡。
3. Vec：在现有4-byte packetizer/cblock-major路径上减少flush和循环控制，不恢复bytewise pack。
4. AvgPool：保留C3 inner-fast owner，只优化连续输入读和输出写流水，不新建第二套pool datapath。
5. 每项先以prefix板级delta验收；`delta>=0`立即回退，不叠加到下一项。

预算：

```text
CONV_ROW_DATAPATH <=34.0M
PPU_ROW_CONSUME <=0.8M
PPU_BLOCK5_FINAL <=7.0M
VEC_FIXED <=5.0M
AVGPOOL <=2.5M
RTL_STAGE_TOTAL <50M
```

中间只在总周期首次低于 `60M` 时跑一次fullres CSim/综合/实现；最终 `<50M` 时再跑完整闭环。资源门槛保持：URAM不增加、BRAM不高于当前实现、LUT/FF增量不超过3%、无owner clone、无DATAFLOW merge/deadlock、100 MHz合法布线且timing pass。

### 7.6 Round 1B当前硬件实测（2026-07-23）

一次启动完成 `P00-P15`。`P15=64,452,405 cycles`，与基线
`64,452,437`仅差32 cycles；ARM计时为 `21,484,123 ticks`，约
645 ms。所有prefix均以 `IDLE/ERROR=0/status=0x5`结束，最终
`MASK.BIN`与golden逐字节一致，PA=`0.97019055`、
mIoU=`0.92065819`。

| 新增exec | 总增量 | 主要stage增量 |
|---|---:|---|
| baseline `P00` | 383,129 | frame 382,765 |
| U1 pool | 1,355,102 | pool 1,355,091 |
| U2 conv | 11,783,392 | conv 11,782,656 |
| U5 pool | 1,355,104 | pool 1,355,091 |
| U6 pool | 338,592 | pool 338,633 |
| U7 conv | 2,484,958 | conv 2,483,456 |
| U20 block | 12,337,248 | conv 9,690,240；B5 2,637,841 |
| U21 conv | 1,012,994 | conv 1,012,352 |
| U35 block | 12,992,608 | conv 9,690,240；B5 3,293,201 |
| U39 fixed | 5,794,366 | vec 5,794,375 |
| U40 conv | 2,405,890 | conv 2,398,528 |
| U53 block | 3,992,766 | conv 2,985,216；B5 999,115 |
| U54 conv | 474,464 | conv 473,472 |
| U68 block | 4,320,416 | conv 2,985,216；B5 1,326,795 |
| U71 fixed | 1,770,592 | vec 1,770,566 |
| U72 conv/output | 1,650,784 | conv 541,440；PPU 1,107,790 |

结论：

1. U2、U20、U35贡献 `31.163M / 44.043M=70.75%` 的Conv周期，是绝对热点；但绝对昂贵不等于相对R2回退。
2. `PPU_ROW_CONSUME=1.112M` 中约99.6%来自U72的full-resolution输出，普通conv row consumer已不是主瓶颈；原 `PPU<=0.8M` 门槛暂缓，需把U72 upsample/output单独归因后再定。
3. `VEC_FIXED=7.565M` 完全由U39（5.794M）和U71（1.771M）组成；R2 A/B证明两项均无回退，后续冻结fixed路径。
4. 当前run给出了热点构成；同artifact R2 A/B已完成，结果见7.7。
5. 后续只修正差值为正且累计解释至少90%总回退的descriptor；U2/U40及
   fixed/pool路径没有回退，必须冻结。

### 7.7 Round 1B同artifact R2 A/B（2026-07-23）

`platform_p7_0720` 使用与当前版完全相同的 `INPUTQ.BIN` 和
`P00.BIN...P15.BIN`。所有prefix均为 `IDLE/ERROR=0/status=0x5`。
P15完整执行结果为 `63,603,157 cycles`，而不是旧测试的
`60,219,515`；后者恰好等于本次P13的 `60,219,509`（仅差6 cycles），
证明旧R2测试在exec 12后进入ERROR并提前结束，不能作为整网基线。

| 新增exec | 当前版delta | R2 delta | 当前-R2 |
|---|---:|---:|---:|
| U2 conv | 11,783,392 | 11,963,388 | -179,996 |
| U7 conv | 2,484,958 | 2,326,944 | +158,014 |
| U20 block | 12,337,248 | 11,929,888 | +407,360 |
| U21 conv | 1,012,994 | 854,880 | +158,114 |
| U35 block | 12,992,608 | 12,585,250 | +407,358 |
| U39 fixed | 5,794,366 | 5,794,368 | -2 |
| U40 conv | 2,405,890 | 2,658,174 | -252,284 |
| U53 block | 3,992,766 | 3,949,376 | +43,390 |
| U54 conv | 474,464 | 448,194 | +26,270 |
| U68 block | 4,320,416 | 4,277,086 | +43,330 |
| U71 fixed | 1,770,592 | 1,770,560 | +32 |
| U72 conv/output | 1,650,784 | 1,613,088 | +37,696 |
| **P15总计** | **64,452,405** | **63,603,157** | **+849,248** |

stage总量对比：

```text
current CONV=44,042,816  PPU=1,112,352   combined=45,155,168
R2      CONV=33,187,072  PPU=11,118,816  combined=44,305,888
net combined regression = 849,280 cycles
VEC/BLOCK5/POOL = effectively identical
```

U20/U35合计解释95.9%的净回退，是Round 1C唯一首批目标。U7/U21虽各有
约158k正回退，但需在首批板级A/B通过后再处理。U2/U40合计改善约432k，
证明当前行级调度并非整体负优化，不允许整路回退。

### 7.8 Round 1C首轮修补（2026-07-24）

U20/U35各包含五个C12分支卷积，约81,920个paired issue；其约407k周期回退
接近每issue多5周期。源码仍保留已失效的
`issue_begin/issue_count/row_begin`及local-to-global索引，而PARAM已将所有
segment字段清零、硬件每次均执行完整行。

本轮恢复唯一 `scheduled_window_generator_row()`：窄3x3、宽3x3和1x1均直接
执行完整行循环，每行初始化一次cache tag；保留U40已有的wide static cache，
不改FIFO、SA、PPU、Vec、AvgPool、BLOCK5 final或PARAM ABI。TB、contracts和
structure checker已同步禁止旧segment入口。

验证结果：

```text
structure/contracts: PASS
WinGen CSim: 0 errors
U40 prefix CSim: 0 errors
fullres CSim: last_uop=74, last_error=0
fullres mask: 0/524288 mismatch, invalid_labels=0
```

下一步只跑csynth audit，检查full-row WinGen/SA/compact PPU均为single owner、
无DATAFLOW错误且资源/时序不回退。通过后用同一组P00-P15上板；只有U20、
U35和总周期达到7.4 gate，才进入U7/U21或Round 2，不再叠加未归因修改。
