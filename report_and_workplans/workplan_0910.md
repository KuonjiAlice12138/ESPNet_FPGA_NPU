# Workplan 0910 - H256W512 INT8 NPU 性能收敛

更新：2026-09-23。分支：`feature/h256w512`。
最近物理与板级签核：Round 4R，`platform_r4_0922`。
本文只保留上板实测、实现结论、剩余瓶颈和可直接执行的后续步骤。

## 0. 目标与当前结论

**近期目标：** binary2 与 cityscapes20 均达到 `<=10.0M PL cycles`，即
100 MHz 下整网 `<=100 ms`。长期目标仍是提升片上供数和流水调度效率，使工作点
接近 Roofline 岭点；不能通过缩小阵列或降低算术峰值伪造“计算受限”。

Round 4 的增量 WinGen 已通过功能验证并在板上带来稳定收益：相对 Round 3，
Conv 均减少 **461,632 cycles（5.95% / 5.94%）**，整网减少
**461,624 / 461,651 cycles（3.69% / 3.69%）**。两个输出 mask 与同模型 Round 4
CSim 逐字节一致，说明本轮功能正确。

Round 4R 已完成周期中性时序修复并成为当前签核基线：binary2 的 total/Conv 与
Round 4 完全相同，city20 仅多 1 个 total cycle；输出直方图不变，ERROR=0。
09/22 实现为 `WNS=+0.293 ns, WHS=+0.010 ns, routing error=0`，稳定满足 100 MHz。

R4R-3 同时否定了旧 Round 5A 的双 psum context 假设：WIN、SA、POST 的 active
总量均约 4.35M~4.51M，链路没有 FIFO-full backpressure；约 2.79M 周期主要来自
每行重新进入 `shared_conv_row_engine` 的 DATAFLOW 启动/排空。SA compute 仅约
0.42M cycles，即使全部被双 context 隐藏也达不到原定 1.31M~1.51M 节省门槛。
下一项大收益必须改为**单任务多行持续流水**，而不是增加 issue 级 psum 缓冲。

## 1. 上板结果与发布状态

### 1.1 双模型整网演进

| 版本 | binary2 total | city20 total | binary2 Conv | city20 Conv |
|---|---:|---:|---:|---:|
| 0910 POST8 | 13,893,492 | 13,911,890 | 9,139,308 | 9,157,708 |
| Round 1 POST16 | 13,885,589 | 13,903,965 | 9,131,436 | 9,149,836 |
| Round 2 C3 KEEP1 | 13,035,715 | 13,054,116 | 8,281,547 | 8,299,947 |
| Round 3 SA tree | 12,507,939 | 12,526,365 | 7,753,783 | 7,772,183 |
| **Round 4 WinGen incremental** | **12,046,315** | **12,064,714** | **7,292,151** | **7,310,551** |
| **Round 4R timing repair** | **12,046,315** | **12,064,715** | **7,292,151** | **7,310,551** |

Round 4 精确 PL 延时为 **120.46315 ms / 120.64714 ms**；ARM 计时为
4,015,429 / 4,021,562 ticks，与 33,333,000 Hz 换算结果一致。MODE_RUN 包含硬件
frame transfer、整网计算和融合输出，不含 SD 读写、MODE_INIT 和 UART 打印。

相对 0910 初版，Round 4 整网累计下降 **13.30% / 13.28%**，Conv 累计下降约
20.21% / 20.17%。距 100 ms 仍需减少 **2.046M / 2.065M cycles**。

### 1.2 Round 4 周期构成

| 顶层阶段 | binary2 cycles | 占比 | city20 cycles | 占比 |
|---|---:|---:|---:|---:|
| FRAME_LOAD | 95,712 | 0.79% | 95,705 | 0.79% |
| MAIN_CTRL + WEIGHT_LOAD | 35,542 | 0.29% | 35,542 | 0.29% |
| CONV_ROW_DATAPATH | **7,292,151** | **60.53%** | **7,310,551** | **60.59%** |
| PPU_ROW_CONSUME | 302,234 | 2.50% | 302,234 | 2.50% |
| PPU_BLOCK5_FINAL | 2,049,822 | 17.01% | 2,049,822 | 16.99% |
| VEC_FIXED | 1,509,681 | 12.53% | 1,509,681 | 12.51% |
| AVGPOOL | 760,920 | 6.31% | 760,920 | 6.30% |
| IDLE / 其他 | 253 | <0.01% | 259 | <0.01% |
| **总计** | **12,046,315** | **100%** | **12,064,714** | **100%** |

非 Conv 部分稳定在 **4,754,164 / 4,754,163 cycles**，Round 3 到 Round 4
只相差噪声级别。若尾部完全不变，Conv 必须降到约 **5.246M cycles** 才能达到
100 ms；因此仅优化 BLOCK5 或仅继续优化 WinGen 都无法独立闭合目标。

### 1.3 功能与物理状态

- App：`INT8-BOARD-20260916-P7-H256W512-R4-WININC-DUAL-PROF`。
- PARAM v4、75 UOP、16 EXEC、U71/U72 分离；模型 artifact 仍为
  `binary2_int8_h256w512_r2_v4` 与 `cityscapes20_int8_h256w512_r2_v4`。
- `E:/O2H.BIN` 与 Round 4 binary2 CSim SHA-256 均为
  `2518be57ff7f5dc5a2acf5a49e536e66a943970bbc531a88785f7405a1d32a12`。
- `E:/O20H.BIN` 与 Round 4 city20 CSim SHA-256 均为
  `2c018de8d62c6626fbacbe8b0dec3687df26b6fa4638871a890e164bbbe84e1c`。
- XSA：`cb78f7b3752e866003f285a983574138ea90f85fa1debfc7277f58997b885136`；
  bitstream：`33916a05616e8e92abe19af79efa1a920d3ac707d820f2bd7df552074ede651d`。
- 路由合法且 routing error=0；post-route phys-opt 为
  `WNS=-0.437 ns, TNS=-62.488 ns, 300 setup endpoints, WHS=+0.001 ns`。
- placed：总 LUT 75.93%、逻辑 LUT 70.73%、FF 30.22%、BRAM tile 33.06%、
  URAM 50.00%、DSP 39.43%。相对 Round 3，总/逻辑 LUT 均增加约 6.8 个百分点，
  FF 增加约 3.2 个百分点，说明增量协议带来了明显控制与路由代价。

本次单次板测输出正确，但 timing-violating bitstream 不作为稳定发布版本；修复后必须重新
implementation 和双模型上板，不能用“本次能跑”代替时序签核。

### 1.4 Round 4R 物理与板级签核（09/22~09/23）

- 平台：`platform_r4_0922`；XSA SHA-256
  `7096f2cff088c513bacdd82d10a6f89aeaaafbdfe8c803029fa33fd772f23c72`；bitstream
  SHA-256 `c43845f76de83e27578681316690516a1b44b63755364dbb0a41636ce3ec9713`。
- `clk_pl_0=100 MHz`；post-route phys-opt 为 `WNS=+0.293 ns, TNS=0,
  WHS=+0.010 ns, THS=0`，setup/hold failing endpoints 均为 0；464,719 条网络
  全部合法绕线，routing error=0。
- placed：总 LUT 76.13%、逻辑 LUT 70.93%、FF 30.24%、BRAM tile 33.06%、
  URAM 50.00%、DSP 39.43%。资源仍在警戒线附近，后续新增结构必须同步删除旧结构。
- 板上 binary2 为 total/Conv `12,046,315/7,292,151`；city20 为
  `12,064,715/7,310,551`。相对 Round 4 分别为 `0/0` 和 `+1/0` cycles，远低于
  1% 回退门槛；双模型 ERROR=0、类别直方图不变。
- 结论：动态 mask 与 fallback 控制修复只改变组合逻辑和物理路径，没有改变 token、
  II 或状态调度；Round 4R 同时通过功能、周期、时序和合法绕线 gate。

## 2. Round 4 归因与当前工作点

### 2.1 增量窗口传输确实生效

focused CSim 已验证 C12 首次/steady 为 12/6 spatial words，C3 KEEP1 首次/steady
为 15/12 words；双模型 fullres logits 和 mask 均 0 mismatch。板上相对 Round 3：

| Conv owner 状态 | binary2 变化 | city20 变化 |
|---|---:|---:|
| WinGen ACTIVE | **-461,632** | **-461,632** |
| WinGen IDLE | 0 | 0 |
| SA active 合计 | **-461,632** | **-461,632** |
| POST active 合计 | **-457,724** | **-457,724** |
| Conv 总周期 | **-461,632** | **-461,632** |

节省从 WinGen 沿背压链传播到 SA/POST，不能将三项重复相加。token 目标已经达到，
但 Conv 只下降约 5.95%，说明继续盲目扩缓存或减少同类 token 的边际收益不足。

Round 4 active 总量：binary2 WIN=4.494M、SA=4.502M、POST=4.351M；city20
WIN=4.513M、SA=4.521M、POST=4.371M。三者已经接近平衡，剩余约 2.79M 的
owner idle/drain 和 issue 边界成为主要架构嫌疑。现有 bin 含等待，仍需一次聚焦的
RTL handshake 归因后才能修改跨 issue 调度。

### 2.2 当前仍未进入计算受限区

32x32 INT8 峰值为 `102.4 GMAC/s = 204.8 GOPS @100 MHz`。

| 指标 | binary2 | city20 |
|---|---:|---:|
| 有效 Conv MAC | 336,435,200 | 345,872,384 |
| 物理槽位静态填充率 | 70.05% | 70.78% |
| K-tile / Conv cycle | 6.431% | 6.527% |
| Conv 有效吞吐率 | 4.614 GMAC/s | 4.731 GMAC/s |
| Conv 峰值利用率 | 4.506% | 4.620% |
| 端到端峰值利用率 | 2.727% | 2.800% |

外部 frame/weight 搬运合计约 1%，不是主要 DDR 带宽瓶颈。当前仍是片上窗口供数、
psum/POST 服务率和 issue 排空共同造成的调度受限；仅凭 owner active 不能区分 FIFO
empty/full。达到 100 ms 也不自动等于达到岭点，必须以后续 handshake/fire 计数和
服务率敏感性实验确认。

### 2.3 Round 4 实现失败的直接根因

前十条 setup 关键路径均位于 `window_row_loader_narrow_paired/unpaired`，从
`cfg_in_c` 到 256-bit byte mask 数据位；每条含 **16 个 CARRY8、22~23 级逻辑**，
route delay 占 74%~79%。核心来源是 `win_low_byte_mask()` 的动态 256-bit
shift/subtract，加上通用窄路径和增量路径同时常驻后造成的布线扩张。

该问题不是 SA、PPU 或增量 C3/C12 热路径本身的算术关键路径。修复应保留当前周期收益，
只压缩动态 mask 和通用 fallback 的控制/路由，不得回退增量协议。

## 3. 全轮次绝对约束

- 保持 PARAM v4 ABI、75 UOP/16 EXEC、U71/U72 分离、H256W512 和双模型兼容；
  不恢复 PARAM v5、cfg-based WinGen dispatch 或旧 UOP 运行时重建。
- 保持单套 32x32 SA、WinGen、weight buffer、Conv POST、PPU 和 memory owner；
  源码 single call site 与 hierarchy clone count=1，不能靠 Tcl/ALLOCATION 掩盖复制。
- 只执行 PARAM 的 mode/run/复用描述符；禁止按网络层名或
  `in_c/kernel/dilation` 组合在热循环选择专用算法。
- 禁止 activation replay、DDR spill、完整特征图复制、第二套完整行缓存、runtime task
  array、动态大 mux 和递归 DATAFLOW 扩张；不增加 URAM，不加宽全局 psum 总线。
- logits、量化/ReLU、残差、拼接、compact store、上采样与 argmax 必须 bit-exact；
  不修改 golden 或重新 QAT 来掩盖硬件行为变化。
- 100 MHz 必须 routing error=0 且 WNS/WHS>=0。Round 4R 已使用逻辑 LUT 70.93%、
  总 LUT 76.13%，因此后续以其绝对实现值为硬门槛：逻辑 LUT<=242,069、
  总 LUT<=259,800、FF<=206,388、BRAM tile<=246、DSP<=1,391、URAM=56；
  新增结构必须同时删除等价旧逻辑，不能继续堆叠。
- 每轮末扫描所有 CPP 的旧 helper、重复 kernel 和无效接口；主 cfg 不切局部 top，
  不自动运行 csynth/package/implementation，由用户手动执行。

## 4. Round 4R - 周期中性物理收敛与握手归因

Round 4R 是进入任何新性能架构前的强制步骤，目标是保留 Round 4 的 12.05M/7.29M
板上周期，同时恢复 100 MHz 时序裕量。

- [x] **R4R-1：消除 256-bit 动态 mask 关键路径。** 修改
  `ESP_INT8_hls/src/win_gen.cpp::win_low_byte_mask` 及两个调用点：先生成固定宽度
  32-bit lane-valid mask，再按 byte lane 独立门控 256-bit word；禁止
  `(act_vec_t(1) << dynamic_bits) - 1`。保持读取次数、token 顺序、循环 II 和输出值不变。
  检查 csynth/实现中 top path 不再出现 16-CARRY8 的 `and_i_i_i_cast` 链。
- [x] **R4R-2：压缩 fallback 控制锥。** 在 `prepare_window_row_cfg` 一次锁存
  `valid_c/last-k mask` 和窄路径所需的最小字段；paired/unpaired helper 只接收精简局部
  配置，不在每个 token 上重复从完整 `cfg` 推导 mask。增量 C12/C3 和通用 fallback
  仍共享唯一 FMBUF reader/assembler，不复制 engine，不删除仍被 descriptor 调度的
  dilation/C19/C25 路径。
- [x] **R4R-3：做一次聚焦 RTL handshake 归因。** 对一个 C3 行和一个代表性 C12 行
  采集 loader->assembler、act->SA、psum->POST 的 valid/ready 或 empty/full；分别统计
  有效 fire、producer blocked、consumer empty、issue start/drain。优先使用 focused RTL
  co-sim/wave 后处理，不把临时宽 counter 接口带入 release top。

  09/22 focused RTL co-sim 结果：C3/C12 均 PASS，SA K-loop 与 POST 均为 `II=1`。
  两类卷积的 `load_word_stream`、activation stream 和 `psum_stream` 均为
  `write_block=0`；C3 的 load/psum 峰值深度为 2/2，C12 为 1/2。C3/C12 的
  assembler read-stall 分别为 302/1171 cycles，SA read-stall 为 1191/1423，POST
  read-stall 为 1966/2006，但没有 FIFO-full producer backpressure。结论是这些等待主要
  来自 producer 服务节拍及启动/排空，不能通过盲目加深 FIFO消除；本轮不修改 release
  FIFO 深度，也不增加 profiling 接口。

**Round 4R gate：** focused transfer counts 不变；双模型 fullres CSim 0 mismatch；
WinGen/SA/POST singleton、SA K-loop II=1、POST II=1；csynth 资源不高于 Round 4；
实现 WNS/WHS>=0、routing error=0；板上 Conv/total 相对 Round 4 回退<=1%。
任何 mask 修复若增加运行周期或复制 datapath，立即回退。

**09/23 状态：**R4R-1/2/3/4、双模型 bit-exact CSim、顶层综合、实现和双模型上板
均已完成；板上 Conv/total 相对 Round 4 最大变化仅 1 cycle。Round 4R 已冻结为
物理与性能签核基线，后续修改必须能够独立回退到该版本。

- [x] **R4R-4：修片上 packed-reader 的同类时序风险。** 在
  `memory.cpp::make_low_byte_mask/read_tile_packed_word` 中以 32-bit lane-valid mask
  和逐 byte 门控替换动态 256-bit shift/subtract；保持零长度、32-byte 全长、
  跨 word 读取与返回值语义不变。只改共享读取 helper，不复制 PPU/Vec/WinGen
  datapath；focused 边界测试和双模型 fullres CSim 后由用户手动综合、实现，
  检查 `read_count -> merged_word` 的 15-CARRY8 链消失且周期/II 不回退。

  已用 `make_low_byte_lane_mask()` 与固定 byte-slice 门控替换动态宽移位；边界 CSim
  覆盖 0/1/31/32-byte、同 word 与跨 word，0 error。binary2 与 cityscapes20 整网
  lowres logits、fullres mask、upsample 均为 0 mismatch，结构检查通过。正式 cfg 不定义
  R4R-3 probe 宏，诊断逻辑不会进入 release top。顶层实现与板测均已通过，确认该修复
  没有带来周期或功能回退。

## 5. Round 5 - 多行常驻流水与尾部闭合

### 5.1 Round 5A：任务级多行常驻流水

#### 5.1.1 旧方案退出与新目标

旧“depth-2 psum issue context”方案不再执行。Round 4R 板测中 SA compute 仅约
`0.42M` cycles，而 psum emit/POST 已通过现有 depth-8 stream 与 SA 并行；即使把
compute 全部隐藏，理论节省也远小于 `1.31M~1.51M` gate，并会增加 context RAM、
状态机和路由压力。R4R-3 也没有发现 FIFO-full producer backpressure。

当前主要结构性开销是 `run_conv_rows_task()` 对每个输出行重新调用一次
`shared_conv_row_engine()`：局部 stream 和 DATAFLOW region 每行重新启动并排空，形成
约 `2.79M` owner idle/drain。Round 5A 改为每个 conv task 只进入一次多行常驻流水，
首阶段要求双模型整网至少节省 `1.30M` cycles，完成目标至少节省 `1.50M` cycles。

#### 5.1.2 固定微架构

- [ ] 在 `conv_engine.cpp` 将 `shared_conv_row_engine()` 重构为唯一入口
  `shared_conv_rows_engine()`；`run_conv_rows_task()` 每个 task 只调用一次。该入口只有
  一个 DATAFLOW region，WinGen、SA、POST、row consumer 四个物理 owner 各实例化一次，
  每个 owner 在内部按 `row_count` 循环，禁止在外层逐行反复进入嵌套 DATAFLOW。
- [ ] 在 `win_gen.cpp` 与 `sa_core.cpp` 增加多行 sequence 接口：task 配置只锁存一次，
  行号通过窄 `row_token` FIFO 顺序传递；每行仍独立复位窗口行状态和 psum，保持
  first/last-K、odd tail、paired/unpaired 和 dilation 语义。禁止 task array、模板化层专用
  实例、第二套 SA/WinGen 或跨行复用本不应复用的 activation。
- [ ] POST 将每行按原顺序产生的 `act_vec_t` 直接写入有界 pixel stream；row consumer
  从该 stream 顺序读取，不再先写 `s_shared_conv_row_buf` 再由
  `replay_conv_row_to_stream()` 回放。FIFO 只允许吸收短期节拍差，深度上限 32 words，
  禁止容纳完整一行、第二套完整行缓存、DDR spill 或 current-conv-output scratch。
- [ ] `ppu.cpp` 保持单一 consumer owner：compact store 直接复用现有 stream consumer；
  BLOCK5 final 与 upsample 改为逐 word 消费当前卷积流。pre-add 在相同 word 上完成，
  BLOCK5 scratch/residual 仍按原顺序读取并进入同一 common store tail。mode 在 task 入口
  锁存，热循环中不得出现按层名、shape 或 mode 复制的完整 consumer datapath。
- [ ] 保持 PARAM v4、tensor/量化语义、32x32 SA、POST16、512-bit 物理接口和输出顺序
  不变。迁移完成并确认零调用后删除 `s_shared_conv_row_buf`、
  `replay_conv_row_to_stream()` 及仅服务旧逐行路径的 helper，不保留新旧两套路径。
- [ ] 顶层 `prof_stage_id` 接口不加宽、不增加多 writer。整个重叠 region 统一归入
  `CONV_ROW_DATAPATH`；WIN/SA/POST 内部 counter 与 MODE_RUN total 继续有效。融合后的
  PPU/BLOCK5 顶层 bin 不再与 Round 4R 串行 bin 直接相加，Round 5B 以 total delta 和
  focused consumer stall 为依据，禁止把 stage 归属变化误报为性能收益。
- [ ] 将 `csim_dump_u40_prestore_row` 等验证点改为 CSim-only 的逐 word tap；禁止为了
  dump 恢复完整 row buffer、复制 stream 或在综合路径增加广播扇出。dump 的 tensor 语义、
  顺序和文件格式保持不变。

#### 5.1.3 执行与停止 gate

1. **先做静态并发安全审计。** 由现有 PARAM v4 artifact 枚举两个模型全部 26 个 CONV，
   对每个 task 列出 WinGen 源 bank、consumer 目的 bank、add/residual/scratch bank 和每周期
   端口需求；必须证明下一行 WinGen 与上一行 consumer 不会形成同 bank 超端口访问或
   RAW/WAR alias。该审计只生成报告，不修改 ABI。任一活动描述符无法证明安全时停止，
   先重做静态 bank placement，不得保留新旧两套 runtime fallback。
2. **Focused 两行原型。** 先在现有 R4R focused top 上实现 C3 与 C12 的两行 sequence，
   并分别接入 compact、BLOCK5 final、upsample 的代表性真实 consumer。比较“一次两行
   常驻”与“两次单行调用”的 RTL cycles、stream fire/stall 和输出，再按 PARAM v4 中实际
   row_count/mode 加权外推。只有接入 consumer 后预计净节省仍 `>=1.30M` 才修改 release
   top；否则立即停止，也不通过增加 FIFO 深度补救。
3. **生产者常驻化。** 只迁移 WinGen/SA/POST 的 row loop，确认唯一 SA/POST、K-loop
   II=1、POST II=1 和编译时间可控后，再接入直接 row stream consumer。任何阶段若出现
   runtime task mux、函数 clone、动态 complete-partition 写或综合时间超过 Round 4R
   同阶段 2 倍，立即回退该阶段。
4. **消费者流式化。** 按 compact、BLOCK5 final、upsample 顺序迁移，但始终保留一个
   consumer call site 和 common write tail；每完成一种模式即删除对应 replay 路径，
   禁止过渡版本把 row-buffer 与 stream 两套实现同时带入顶层综合。
5. **集中验证。** 执行 dead/legacy 与 singleton 扫描、focused RTL、双模型 fullres
   CSim；要求 logits/mask 0 mismatch、stream 读写计数完全一致、无 deadlock。随后由用户
   手动跑顶层 csynth 和 implementation。
6. **物理 gate。** URAM 保持 56，BRAM tile 不高于 246，DSP 不高于 1391；placed LUT
   不高于 Round 4R 的 259,800，WNS/WHS>=0、routing error=0。若 row-buffer 删除未抵消
   新控制逻辑，先压资源，不允许带着资源增长进入板测。
7. **5A 必须单独上板。** 双模型 ERROR=0、输出直方图/离线 mask 不变；FRAME、Vec、
   AvgPool 不回退>1%。首阶段 total `<=10.75M`，完成目标 `<=10.55M`。未达到
   `10.75M` 则回退 Round 5A，禁止进入 5B；通过后再用实测 total 与 focused consumer
   stall 重算 Round 5B 预算。

### 5.2 Round 5B：按实测 Conv 预算压缩 BLOCK5 尾部

Round 4R 串行计数下，单独优化 BLOCK5 无法闭合 100 ms，因此 5B 只能在 5A 单独
上板且 total `<=10.75M` 后启动。5A 将 Conv 与 row consumer 置于同一持续流水，旧的
串行 Conv/BLOCK5 stage 数不再可直接相加；5B 预算必须使用 5A 的 MODE_RUN 实测差额
和 focused consumer stall 重算，目标是再消除至少 `5A_total - 10.0M` cycles。

- [ ] 在 `ppu.cpp`、`conv_store.cpp`、`memory.cpp` 先测 BLOCK5 scratch/residual read、
  lane arithmetic 和 common store tail 的实际调用间隔。只有算术是服务瓶颈时才将局部
  factor 4->8；否则优化顺序 word 传输、qparam/address 准备，不扩大 lane 算术。
- [ ] 保持单一 arithmetic owner 和 common store tail；qparam 在行入口本地锁存，
  固定分组访问，禁止 32-way 参数 mux、动态 complete-partition 写、runtime segment
  composer 或独立 BLOCK5 finalizer。Vec16 与当前 AvgPool 冻结，除非新板级证据表明回退。

最终 gate 始终是两模型 `MODE_RUN<=10.0M`，Vec/AvgPool 不回退>1%、输出 bit-exact、
WNS/WHS>=0。若 5A 上板 total 高于 `10.75M`，本节不执行。

## 6. 验证、停止与交付

每个中间节点只做一次集中闭环：
`dead/legacy + singleton 扫描 -> contracts/focused test -> 双模型 fullres CSim
-> 用户手动 csynth -> hierarchy/II/resource audit -> 用户手动 implementation`。
Round 5A 必须单独做一次双模型上板 profiling，达到门槛后才进入 5B；不以 HLS latency
估计替代板级计数。每个 round 最多做一次 focused RTL，不反复跑无关脚本。

记录实际 PL Hz、ARM ticks、RTL cycles、counter 状态、XSA/bitstream/artifact hash。
非零 ERROR、非法类别、mask mismatch、clone、II 回退、非法 route 或负 slack 均禁止冻结。
候选若 Conv 或整网回退>1%，停止并回退；仅改变状态归属而总周期不降不算性能收益。

**下一动作：执行 Round 5A 的 focused 两行常驻原型，只在加权外推节省>=1.30M、
无 FIFO-full backpressure 且综合规模可控时迁移 release top。Round 4 增量 C3/C12、
Round 3 SA tree、POST16 和 PARAM v4 全部冻结；旧双 psum context 方案不得恢复。**
