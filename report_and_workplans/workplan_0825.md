# Workplan 0825：P7 归因实验与 U71 融合回退

更新时间：2026-09-07  
当前决策：0906 平台实板提前进入 ERROR，用户停止 U71 融合分支，回退到 `bdb83c6 / P7-CLASS20-BOARD-0806` 完整源码及 `*_int8_0809` PARAM v4 产物。第 14 节为当前交付状态；第 0~13 节保留历史证据，其 v5/U71 保留约束和 RR 后续计划已废止，不再指导下一轮开发。

## 0. 当前目标

在保持 P7 单一数据通路和 PARAM v5/U71 融合语义不变的前提下，恢复 100 MHz 合法布线，并确保总周期不劣于 0901 HLS 基线。当前优先级是先恢复读取路径的周期和物理边界，再处理剩余拥塞；不接受只改善 HLS 估计、却让 Vivado route 失败的方案。

## 1. 已冻结的设计约束

- 保留一个 MainCtrl、一个 Conv Engine、一个 scheduled WinGen、一个 32x32 SA、一个 PPU 和一个 Vec Engine。
- 不降低 `TM/TK`，不把 WinGen、SA、PPU 的关键 `II=1` 改为 `II>1`。
- 不新增 current-conv-output 中间访存，不恢复 P6 cfg dispatch、运行时 shape 推断、旧 generic narrow path 或 cold-RMW fallback。
- 保留 PARAM v5、U71 融合、block-level PPU 和 row-level fused PPU；不新增第二套 Conv/SA/WinGen/PPU datapath。
- `HLS 200-2042` 是 URAM 读回绑定警告，不能单独作为失败根因；任何修复必须同时检查 II、资源、层级复制和实际 route。
- 结论以 100 MHz Vivado legal route 和板级 RTL stage counter 为准，ARM 的 33.333 MHz 计时只用于校验，不用于替代 PL 周期。

## 2. 可信性能基线

### 2.1 板级 P7 基线

2026-08-17 的双模型验证使用 100 MHz PL stage counter：

| 模型 | RTL stage total | 单图 PL 延时 | 备注 |
|---|---:|---:|---|
| binary2 | 51,319,833 cycles | 513.2 ms | 500 张平均值约 513 ms |
| Cityscapes 20 类（失败运行） | 11,631,767 cycles | 不计入整网性能 | 提前 ERROR，116.3 ms 不是完整推理 |

binary2 主要 stage：`FRAME_LOAD=382,743`、`CONV_ROW_DATAPATH=32,229,027`、`PPU_ROW_CONSUME=1,202,400`、`PPU_BLOCK5_FINAL=8,256,942`、`VEC_FIXED=6,155,931`、`AVGPOOL=3,048,809`。这些数值是 P7 性能和结构回归的参考，不是 1A 的新基线。

20 类可用的历史单图基线是 0806 的 `51,393,493` PL cycles（约 `513.935 ms`），不是上述失败的双模型切换结果。

### 2.2 0901 HLS 基线

报告：`ESP_INT8_hls/hls_work_0901_release/hls/syn/report/csynth.rpt`。

| 项目 | 0901 基线 | 当前 1A | 变化 |
|---|---:|---:|---:|
| top LUT | 360,511 | 361,584 | +0.30% |
| top FF | 267,511 | 265,020 | -0.93% |
| top DSP | 1,332 | 1,338 | +0.45% |
| BRAM / URAM | 1,408 / 112 | 1,408 / 112 | 不变 |
| HLS estimated clock | 7.30 ns | 7.30 ns | 不变 |
| `conv_engine_exec` latency | 469,855,391 | 527,533,727 | +12.28% |
| `generate_conv_window_row` | 739,332 | 851,981 | +15.24% |
| `window_row_loader_narrow_paired` | 630,785 | 851,969 | +35.06% |

`window_row_loader_narrow_paired` 的内部读取循环从 0901 的 5 cycles/iteration 变为 7 cycles/iteration；外层最大 latency 从 630,785 变为 851,969，增加量为 `216 cycles × 1024 rows = 221,184 cycles`。这是当前必须先消除的确定性性能回退。

## 3. 2026-09-02 实现失败证据

诊断来源：`D:\ESP_INT8_vp\ESP_INT8_vp.runs\impl_1\runme.log`、`ESP_INT8_wrapper_placed.dcp`、`ESP_INT8_wrapper_routed_error.dcp`，以及 `report_and_workplans/route_diag_0902/`。

| 指标 | 结果 |
|---|---:|
| placement | 完成 |
| phys-opt | 完成，post-place WNS `+0.501 ns`、WHS `-0.238 ns` |
| route elapsed | 约 17 分钟 |
| routable nets | 513,198 |
| fully routed / unrouted | 24 / 513,174 |
| node overlaps | 1,121,867 |
| initial global congestion | North level 7；South level 6；Long North level 7 |
| placed CLB | 41,132 / 42,660 = 96.42% |
| placed CLB LUT / FF | 226,913 / 184,803 |
| placed BRAM / URAM / DSP | 692 / 112 / 1,353 |
| unique control sets | 1,797 |

失败发生在初始 route，不是最终时序报告意义上的 setup 失败。routed-error DCP 只有 24 个 routable net 完全布通，因此其中的 WNS `+0.523 ns`、WHS `-0.202 ns` 不能当作时序通过依据。

## 4. 根因定位

### 4.1 1A 的源码级变化

1A 引入了 `memory_row_context_t` 和 `on_chip_memory_prepare_row_context()`（`memory.cpp`），并让 WinGen/PPU 的热点读取经过 row context、runtime region 选择和 inline 的 `on_chip_memory_read_row_word()`。调用点分布在 `win_gen.cpp` 的 319、523、748、866、2059、2209 行附近，以及 `ppu.cpp` 的 305、600 行附近。

HLS hierarchy/compile log 出现多个 call-site-specialized 版本：
`on_chip_memory_prepare_row_context_4136141144147152157160`、`on_chip_memory_prepare_row_context_5`、`on_chip_memory_prepare_row_context_5189` 和未后缀版本。它们不是算法错误，但说明“共享 reader”没有形成单一物理实例，反而增加了局部乘法器、控制边界和跨区域连线。

1A 同时移除了 0901 `on_chip_memory_read_aligned_tensor_word` 在 narrow paired 热路径中的显式一周期读边界。结果是：

- WinGen 到 FMBUF 的地址/region 选择更直接地暴露在大范围互连中；
- `window_row_loader_narrow_paired` 的内层读取由 5 cycles 退化为 7 cycles；
- WinGen 和 Conv 的层级 latency 增长；
- 物理上 North/South hotspot 直接指向 narrow paired、narrow unpaired、SA 和 row loader。

### 4.2 Vivado 热点

`design_congestion.rpt` 显示的主要贡献者为：

- North global level 7：scheduled WinGen 32%、`execute_issue` 23%、`run_conv_rows_task` 22%；
- South global level 6：`window_row_loader` 52%、SA 17%、BLOCK5 final 8%；
- placer North level 6：narrow paired 19%、FMBUF URAM 15%、narrow unpaired 15%、SA 14%；
- 高扇出：SA DSP `CEA1_BUFG` 39,393 loads，BLOCK5 final FSM state 4,098 loads，PPU affine control 4,096 loads，WinGen FIFO register 2,770 loads。

因此当前问题是“读边界被改坏 + CLB/控制互连接近饱和”的组合，不是 DSP 数量不足，也不能靠更换 router strategy 解决。URAM 仍为 112/112 满占用，BRAM 为 692/744；只增加少量 LUT/控制逻辑就足以使 route 从可搜索退化为大面积冲突。

## 5. 决策：废止 RouteRecovery-1A

当前 1A 不作为实现候选，不进入旧计划中的 1B。保留 PARAM v5/U71 和 P7 的计算架构，但回退 1A 对 WinGen/PPU memory hot path 的改动，目标是恢复 0901 的读时序和已注册的物理边界，而不是回退整个 P7。

## 6. 下一步计划

### RR-2A：恢复注册读边界并先收回性能

修改范围只限 `memory.cpp`、`win_gen.cpp`、必要的 `ppu.cpp` 声明和结构检查脚本：

1. 以 0901 生成报告对应的读取结构为参考，恢复 `on_chip_memory_read_aligned_tensor_word` 的显式 `INLINE off`、单周期/`II=1` 读接口和 WinGen 侧寄存边界；不在窄路径中直接暴露 row context 的 runtime region mux。
2. 从 WinGen 的 `window_row_loader_narrow_paired`、narrow reuse 和常规 3x3 loader 中移除 `on_chip_memory_prepare_row_context()` 热路径调用；地址计算只在行启动阶段完成，内层读取使用已经确定的物理 word offset。
3. 对 PPU 保留必要的 input/row-buffer 写入路径，但不让 PPU 的独立 source read 与 WinGen reader 共享一个带 runtime 分支的聚合 context。任何共享必须通过已有的 registered word boundary，不新增 DATAFLOW region。
4. 删除 1A 引入且不再被调用的 row-context helper/forward declaration；保留 `frame_dma` 仍需要的输入写入 owner。静态检查必须确认不存在多个 `prepare_row_context_*` 物理候选和旧/新 reader 并行主路径。
5. 先做结构检查和 prefix-lite/轻量 CSim，再做 csynth。确认 paired 内层读取回到 5 cycles 后，才允许处理 routing；如果仍为 7 cycles，直接停止本轮。

### RR-2B：仅在 RR-2A 通过后做物理压力微修

不改 SA/TM/TK、WinGen 调度协议和计算周期，只针对报告已经指出的高扇出控制：

1. 在现有 word/row 边界复用寄存器，局部化 WinGen loader/assembler 的 enable 和 valid 控制，避免把完整 row context 或宽数组作为多个函数参数传播。
2. 对 BLOCK5 final 和 row consumer 只做控制信号局部化；不拆成第二套 finalizer，不恢复 Vec standalone BLOCK5，不增加 current-conv-output 访存。
3. qparam 只在已有 execute/row 边界装载为局部标量；不得把完整 PARAM v5 聚合对象传入 WinGen/SA/PPU，也不得引入新的自动 complete-partition 大数组。
4. 一次只生成一个候选，避免同时改变 reader、PPU 和 SA。每次候选都检查资源、层级实例数和 route status。

## 7. 通过门槛与停止规则

### 7.1 HLS/功能门槛

- Top CSim 与现有 binary2 golden/replay 通过；不以未对齐的旧 golden 误判功能。
- 单一 Conv/WinGen/SA/PPU owner；无 P6 dispatch、legacy reader、重复 row-context 主路径或未调用的 debug/冷回退逻辑。
- 相对 0901：`conv_engine_exec <= 470,795,102` cycles，`generate_conv_window_row <= 740,811`，`window_row_loader_narrow_paired <= 632,047`。
- Top 资源不超过 0901 的约 1%：LUT `<=364,117`、FF `<=270,187`、DSP `<=1,346`；BRAM/URAM 不增加。
- 关键 WinGen/SA/PPU loop 保持 `II=1`；综合时间应保持可控，不接受再次进入数小时无进展的搜索状态。

### 7.2 Vivado 门槛

- placed CLB 低于 95% 为目标，控制集不高于 0901 基线；BRAM/URAM 列不能增加压力。
- route status：failed/unrouted nets 为 0、node overlaps 为 0、合法布线完成。
- 100 MHz 最终 timing：WNS、WHS、TNS、THS 均满足约束；仅 post-place WNS 不作为通过。
- 初始 route 若再次出现 global congestion level 7，或在短时间内出现大规模 unrouted/overlap，不等待数小时，保存报告并停止候选。

## 8. 证据文件

- 当前失败日志：`D:\ESP_INT8_vp\ESP_INT8_vp.runs\impl_1\runme.log`
- 当前 placed/routed-error DCP：同一 `impl_1` 目录下的 `ESP_INT8_wrapper_placed.dcp`、`ESP_INT8_wrapper_routed_error.dcp`
- 本轮诊断：`report_and_workplans/route_diag_0902/`
- 当前 HLS 报告：`ESP_INT8_hls/ESP_INT8_hls/hls/syn/report/`
- 0901 基线报告：`ESP_INT8_hls/hls_work_0901_release/hls/syn/report/`

## 9. RR-2A 执行状态（2026-09-03）

本轮已完成源码级修补，修改范围限定为 `memory.cpp`、`win_gen.cpp` 和 RR-2A 结构检查脚本：

- 恢复 `on_chip_memory_prepare_row_base()` 与 `on_chip_memory_read_aligned_tensor_word()`；后者保持 `INLINE off`、`PIPELINE off`，重新形成显式的物理 word 读取边界。
- WinGen 的 narrow paired、narrow reuse、常规 3x3 和 1x1/pre-affine 读取改为 `row_base + row_valid + 固定 word offset`，不再调用 `memory_row_context_t` 聚合 reader。
- 保留 PPU 仍依赖的 row-context/input 写入路径，未改动 `frame_dma` 的输入写入 owner；删除未被调用的 `on_chip_memory_read_aligned_row_tile()`。
- `test_route_recovery_1a_contract.py` 已改为 RR-2A 契约检查，结果为 `RR-2A CONTRACT PASS`。
- 前五个 exec 的 prefix-lite CSim 已通过：`ESP_INT8_CSIM_MAX_UOP=5`，Vitis 返回 `CSim done with 0 errors`。该结果证明当前改动可编译并能执行到首个卷积前缀，但不代表 fullres golden 已完成。

本轮 csynth 已于 2026-09-03 完成，耗时约 10 分 17 秒，日志未出现实际 synthesis error（`SYNCHK` 报告为 0 errors）。与 0901 基线的关键对比如下：

| 指标 | 0901 | RR-2A | 结论 |
|---|---:|---:|---|
| Top LUT / FF / DSP | 360,511 / 267,511 / 1,332 | 360,768 / 266,832 / 1,340 | 门槛内，LUT +0.07%、DSP +0.60% |
| BRAM / URAM | 1,408 / 112 | 1,408 / 112 | 无增长 |
| `conv_engine_exec` | 469,855,391 | 416,384,159 | 降低 11.38% |
| `generate_conv_window_row` | 739,332 | 634,892 | 降低 14.12% |
| `window_row_loader_narrow_paired` | 630,785 | 630,785 | 恢复基线，保持 II=1 |

因此 RR-2A 的 HLS 性能/资源 gate 通过，可以进入功能 gate；但完整 top/golden CSim 和 Vivado 合法布线、最终时序尚未验证。旧的总结构检查器仍因历史 `prof_stage_id` 门槛失败，`--round4ar-only` 仍报告既有 compact-writer 契约问题，本轮未篡改这些无关接口。报告中的 20 条 `HLS 200-2042` 是当前 URAM 绑定告警，URAM 仍为 112/112，不能单独作为 RR-2A 失败依据。

本轮保留 `hls_work_rr2a_light` 作为 CSim 证据目录；临时 `hls_config_rr2a_light.cfg` 已移除，主 `hls_config.cfg` 未修改。

## 10. RR-2B-WG1 执行状态（2026-09-03）

### 主综合结果补充（用户手动综合后）

用户随后用主 `hls_config.cfg` 完成综合，报告时间为 2026-09-03 10:38，HLS 总耗时约 10 分 16 秒，未出现 pre-synthesis 错误。相对 `hls_work_0901_release` 基线：Top LUT `360783`（`+272`，`+0.075%`）、FF `266732`（`-779`）、DSP `1340`（`+8`，`+0.60%`）、BRAM `1408`（不变）、URAM `112`（不变），按 RR-2B 门槛通过资源回退检查。

关键层级为 `window_row_loader_narrow_paired=630785` cycles、`generate_conv_window_row=634892`、`conv_engine_exec=416384159`，均满足 RR-2B 门槛；RTL 日志出现 `fifo_w256_d16_D`，未出现 `fifo_w280_d16_D`。RR-2B-WG1 可保留并进入 Vivado placement/route、timing、high-fanout 和 congestion 验证；在获得实现证据前不叠加新的 PPU/Vec 候选。

### Vivado 实现结果与 RR-2C 计划（2026-09-03）

RR-2B-WG1 的 HLS gate 通过，但 Vivado 实现 gate 失败。`place_design` 和 `phys_opt_design` 完成，`route_design` 从 12:20:23 运行至 16:57:40，耗时 `16535.1 s`（约 4 小时 36 分钟）。初始 global/short/timing congestion 均为 level 6；Phase 5.2 的重叠数量曾从 `427108 -> 133455 -> 55985`，后续又出现 `385504 -> 193565 -> 71507`，最终报告 `23345` 条信号未布通、`29583` 个节点重叠，生成 `ESP_INT8_wrapper_routed_error.dcp`，没有合法 route。

本次 placed 资源为：CLB `40917/42660=95.91%`、CLB LUT `225880`、FF `186803`、BRAM `692/744=93.01%`、URAM `112/112=100%`、DSP `1348`、unique control sets `1802`。因此失败主因是 CLB packing/存储列和跨区域控制、地址网络过密，不是 LUT/DSP 数量本身；最终 `WNS=-4.436 ns` 是在大量未布通网络上的中间结果，不能作为合法时序结论。

最终冲突网络明确指向三组热点：

1. `ppu_consume_conv_stream` 的 pipeline FSM、`ppu_transform_conv_word` 和 `on_chip_memory_read_packed_tile_from_row`；
2. `s_fmbuf_uram` 的地址位、动态 mux/读口控制，分别与 PPU 和 WinGen 长距离相连；
3. WinGen `window_loader_emit_column` 的 word register，以及 SA DSP `PREG/P` 网络。

这说明 RR-2B 的 280-bit 到 256-bit FIFO 收窄已经生效，但只减少了 FIFO 元数据宽度，没有切断主导 route 的 FMBUF 地址/控制和 PPU packed-reader 网络。RR-2B 代码先保留，不再叠加新的 FIFO、PPU 或 Vec 候选。

下一轮执行 `RR-2C Physical Locality Recovery`，只做一个源码候选，且不改变 SA/TM/TK、WinGen schedule、关键 II 或 P7 ABI：

1. **FMBUF 读路径局部化**：在 `memory.cpp` 保持唯一 URAM/BRAM owner 和 pool alias owner；把物理区域选择、bank base 和 row base 的解析提升到 row/task 边界，热内层只接收已确定的 local word offset，禁止每个 word 再经过 `resolve_phys_addr()`/`bank_base()`/URAM-BRAM runtime mux。`frame_dma` 输入写入路径必须保留，不能复制存储阵列。
2. **PPU 热循环局部化**：在 `ppu_consume_conv_stream()` 入口一次解码 mode、valid channels、add/cat 标志和 affine 状态；`ppu_transform_conv_word()` 只接收必要的局部标量/已准备的 source word，不再把完整 consumer、row context 和动态存储选择传播到每个 packed-group issue。保持一个 transform/store tail、现有 `II=1`、不新增中间 buffer 或 current-conv-output 访存。
3. **WinGen/SA 控制局部化**：WinGen 在列/行启动边界锁存物理读参数，assembler 保持 256-bit word-only FIFO；SA 在 K-tile 边界锁存 active-lane/schedule flags，内层仍使用单一 `systolic_array_core_row`、原 TM/TK 和 `SA_K_TILE_II`，不得以复制 SA 或降低吞吐换取布线。
4. **验证和停止规则**：先做静态引用/deadlogic 扫描、结构 contract 和轻量 CSim，再做主 csynth。要求 Top LUT/FF/DSP 不超过 RR-2B 的 `+1%`、BRAM/URAM 不增加、paired/conv latency 不回退、单一 memory/WinGen/SA/PPU owner。实现时 place 后若 CLB 仍 `>=95%`，或初始 route 再出现 level 6/大量 overlap，立即保存诊断并停止，不再等待数小时。

本轮只生成一个 WinGen 物理压力候选，没有改动 SA、TM/TK、WinGen schedule、PPU、Vec 或主配置：

- 将内部 `window_load_word_t` 从 `word + slot + row + chunk` 收窄为仅 256-bit `word`；原有 word 发射顺序和 FIFO 深度不变。
- wide assembler 在每个已存在的列更新边界用两个窄局部计数器按既定 `kh -> chunk` 顺序恢复 row/chunk；slot 仍来自已有 `column_meta_stream`。
- 删除 loader 到 assembler 之间的元数据广播，目标是消除 `load_word_stream` FIFO 元数据寄存器的高扇出和无必要的 FIFO 位宽；不引入第二套 datapath 或新的存储访问。
- 全部 `token.slot/row/chunk` 引用已静态清除；`git diff --check` 和 RR-2A structure contract 通过。
- 独立轻量 CSim（前 5 个 UOP）通过，Vitis 返回 `CSim done with 0 errors`。临时 `hls_config_rr2b_light.cfg` 已删除，主配置保持不变。

独立资源测量未由本轮自动完成，已按“由用户手动综合”的边界停止；此前误启动的隔离 HLS 编译在生成报告前已中止，不能作为 RR-2B 资源结论。下一步只需用主 `hls_config.cfg` 手动综合，并比较 `window_row_loader`/`window_row_assembler` 层级、Top LUT/FF/DSP/BRAM/URAM 及关键 II；若 paired latency 或资源超过 RR-2B 门槛，回退 WG1，不叠加 PPU/Vec 候选。

### RR-2C 源码修复交付状态（2026-09-03）

已完成本轮物理局部性修复，未改变 SA/TM/TK、WinGen schedule、PPU/Vec 数据流或 PARAM ABI：

- `memory_row_context_t` 在行边界一次保存 logical/local word base、物理区域、行宽和通道几何；WinGen/PPU 热读路径不再逐 word 传递完整 descriptor 或调用旧 aligned reader。
- WinGen 的 3x3/1x1/预 affine 读路径统一改用 row context；删除无引用的 `on_chip_memory_prepare_row_base()` 和 `on_chip_memory_read_aligned_tensor_word()`，保留 `frame_dma` 依赖的通用输入写入/存储 owner。
- `window_load_word_t` 保持 256-bit word-only；移除冗余 `row_base0/1/2`，assembler 在既有列更新边界恢复 row/chunk 元数据。
- PPU transform 入口预解码 mode/act type/add context；compact writer 只接收行级目标 context 和 local word index，不再逐 word 解析完整目标 descriptor；未增加中间访存、第二 SA 或第二 PPU。

验证结果：RR-2C reader contract PASS；旧 reader、280-bit token、allocation pragma、cold-RMW 引用扫描 PASS；轻量 `MAX_UOP=4` CSim 返回 `CSim done with 0 errors`。临时 `hls_config_rr2c_light.cfg` 已删除，主 `hls_config.cfg` 未改动。主综合及后续实现留给用户手动执行。

## 11. RR-2C 端口与读取调度修复（2026-09-05）

### 最新报告的两个阻断项

16:28 的主报告虽然完成综合，但不能推进实现：生成的 top RTL 没有三个 `m_axi_gmem*` 和 `s_axi_control`，三个 DDR 指针被当成普通标量端口。日志出现 `HLS 214-450: Ignore address on register port`。源码的 INTERFACE pragma 尚在，但预处理顶层缺少正常基线的 kernel/TOP 标记，reflow 记录也没有这 16 条接口约束。不能把这种产物当作正确 IP；具体的工具标记丢失诱因仍未完全确定。

Top BRAM 从 1408 降至 1363，恰好少了三个 AXI adapter 各 15 个 BRAM18，因此本次资源下降不构成通过证据。同期 paired loader 最大延迟从 630785 增至 851969、内层最大迭代从 5 增至 7。进一步比较报告：基线 aligned reader 是 `II=1` 的函数流水，本次 registered reader 则为非流水函数。两者均显示 latency=1，并不代表调用间握手代价相同。

### 本次源码改动

1. 顶层直接声明 `#pragma HLS TOP name=espnet_encoder_int8_core`，保留主 cfg 的同名 top；仅保留正式的三个 AXI master、AXI-Lite 和既有 profiling 协议。删除已无 cfg 使用的 `ESP_INT8_COSIM_LITE/ap_memory` 分支；CSim prefix 截断宏明确排除 synthesis。没有改动 app ABI、主 cfg、DDR 深度或 burst 参数。TOP 用法依据 [AMD UG1399](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/pragma-HLS-top)。
2. 删除 `on_chip_memory_read_row_word_registered`，改成只接受 `region + 已算好的 local word index` 的单字 reader，并显式恢复 `PIPELINE II=1`。仅约束这个无循环、单次存储读取的叶模块，不给 row/task 大循环增加 II 约束。WinGen cache 在 lookup 前计算最终 word index，命中和跨字第二读共用该地址；memory 内部不再串接 row-base 加法或 descriptor/bank 解析。
3. 保留 RR-2C 的行级 context、PPU 局部化、256-bit word-only FIFO、唯一 BRAM/URAM owner，以及 input DMA/pool alias 路径。未改变 SA/TM/TK、存储容量、PARAM v5、U71 融合、PPU/Vec 调度或缓存发射顺序。
4. 报告审查脚本将 `HLS 214-450` 升为 blocker，并检查生成的 top RTL 是否真的包含所需 AXI/profile 端口，避免再次出现没有 ERROR 就放行。已做本轮旧函数引用和全部 cpp 的本地 static 引用扫描，未发现新增无调用 static helper；没有加入 allocation directive 或第二套计算/存储路径。

### 轻量功能验证与交付边界

仅运行一轮组合轻量 CSim，工作目录 `ESP_INT8_hls/hls_work_rr2c_repair_light`，总耗时约 51 秒，`CSim done with 0 errors`：

- top prefix 改用同目录的 `binary2_int8_u71fused_v5/PARAM.BIN` 与 `input_q.bin`，检查 PARAM 接受状态和运行错误，不再把零输出 checksum 当作功能通过。
- 完成 INIT、输入 DMA 和 U1 前缀；49,152 个输入 word 与实际片上存储读回完全一致。`last_uop=2` 是下一条被 prefix 截断的指令，不表示 U2 卷积已执行。
- 17 组 WinGen 用例直接连接真实 `memory.cpp`，覆盖 C3/C12/C19/C25/C131、1x1、单双像素、行复用、dilation=16、奇数尾部、非对齐跨字、URAM/BRAM 和 pool2 alias。比较 294,528 个 lane，0 mismatch。
- 这不是整网/fullres golden，也不验证 RTL 背压和真实时序；CSim 的 stream 最大占用 4608 来自软件中整行生产后消费，不能直接解释为硬件 FIFO 深度需求。

源码交付后由用户使用原主 `hls_config.cfg` 手动综合。本次没有运行 csynth/实现。下一份报告必须先确认三个 AXI master、AXI-Lite 与 profiling 协议恢复，再按含完整接口的 RR-2B 基线核对资源：LUT/FF/DSP 不超过 +1%、BRAM18 <=1408、URAM <=112；paired <=630785、Conv <=416384159，并检查单字 reader 的 II=1 和内层迭代周期恢复。以上资源/性能/RTL gate **仍待新综合验证，不能用本轮 CSim 代替**。

## 12. RR-2C 缓存地址链修复（2026-09-06）

9 月 5 日 22:30 的主综合完成，三个 AXI master、AXI-Lite/profile 接口恢复；Top LUT/FF/DSP 为 `361766/266434/1342`，BRAM18/URAM 为 `1408/112`，满足相对 RR-2B 的资源增幅门槛。但 paired loader 仍为 `741377`、Conv 为 `470911135`，没有通过性能门槛。本轮不推进实现。

**已定位的原因**：比较当前及 `hls_work_0901_release` 的 `window_row_loader_narrow_paired.verbose.sched.rpt`，基线 State 6 是 `input_col 加法 -> 乘法 -> 字节基址加法 -> cache tag 比较`，关键链 `6.468 ns`。当前 State 6 的最后一步变成 `local word base 加法`，tag 比较被推至 State 7，后续 reader 调用也后移。对应内层最大迭代从 5 拍增至 6 拍；不是单字 reader 的 II 再次退化。

**本轮只改 WinGen 缓存读取的地址表达式**：先用 `(local_word_base << 5) | row_byte0[4:0]` 合成行不变的本地区域字节基址，再加 `input_col * phys_c + c_begin`，最终直接取高位作为 word index。合并基址只需位拼接，不把额外进位加法留在 tag 比较前。对有效 row context，这与原地址等价；原有区域/边界检查、非对齐跨字第二读、缓存 tag 含义及替换次序保持不变。

没有改动顶层接口、主 cfg、memory owner/binding、单字 reader 的 `II=1`、SA/TM/TK、PPU/Vec、PARAM v5、schedule 或 FIFO；没有增加新 helper、buffer、datapath 或大循环流水约束。旧的 post-shift word-base 加法表达式已替换，不保留旁路。全 cpp/include 静态引用扫描未发现无调用的本地 static helper；已移除旧 reader/cold-RMW 等名称和 ALLOCATION pragma 的扫描结果仍为 0。

**验证**：复用 `test_rr2c_recovery.py` 增加地址链结构回归，修复前该项失败、修复后 5 项通过。只运行一轮 `hls_config_rr2c_repair_light.cfg` 组合 CSim，约 20 秒，`CSim done with 0 errors`：PARAM v5 INIT/RUN prefix 输入 DMA `49152` words 全匹配；19 组 WinGen 用例检查 `325888` lanes，0 mismatch，含新增 URAM/BRAM 高基址用例。没有跑 fullres、局部综合或主综合；结构检查和 CSim 不代表性能/资源 gate 已通过。

**下一步由用户手动综合**：仍用原主 `hls_config.cfg`。检查内层最大迭代恢复到 `<=5`、paired `<=630785`、Conv `<=416384159`，并重查第 11 节端口和资源门槛。如果未恢复，先对照新调度报告中地址生成和 tag 比较的状态，不叠加 SA/PPU 改动，也不以放宽目标时钟或强制大循环 II=1 掩盖回退。

## 13. platform_0906 双模型单图对比准备（2026-09-06）

用户已完成实现、导出 `platform_0906`。本次只改 app 和 SD 数据，不重跑 HLS，不重训或重导出模型，**Build app 由用户手动执行**。

1. App 增加独立的双模型单图模式，关闭 prefix/验证集模式。依次对 binary2、cityscapes20 执行各一次参数加载、`MODE_INIT`、输入加载和完整 `MODE_RUN`。每次清零计数器、独立计时，输出 stage/conv-internal CSV 和两模型比较表；不把两次时间累加称作单次延时。
2. 使用各自 `*_int8_u71fused_v5` 的完整 PARAM 和匹配 INPUTQ。SD 文件为 `P2.BIN/B2.BIN`、`P20.BIN/C20.BIN`；输出分别为 `M2.BIN/M20.BIN`。输入量化数据不同，禁止互换；所有文件名满足 FAT 8.3。保留主机 golden，不把它预填到 SD 输出文件中。
3. 保存输出后检查对应的 2/20 类 ID 范围和 ERROR stage，出现错误保留输出用于诊断，不把提前返回计作成功性能。计时口径保持 ARM `CNTFRQ_EL0`、PL 标称 100 MHz 分开：不含 SD、INIT、cache 维护和串口打印，包含 MODE_RUN 内的输入 DMA、计算及输出写回。
4. 增加小规模 app 结构回归，核对两次模型初始化、独立输入/输出及计时边界；备份 SD 已识别的旧根文件后写入两组数据并校验 SHA256，验证集子目录不动。最终交给用户手动 build、下载和测试。

准备已完成：app tag 为 `INT8-BOARD-20260906-P7-DUAL-CONVPROF-SINGLE`，app.yaml、组件、CMake cache、bitstream/FSBL 启动路径均指向 `platform_0906`；5 项 app 静态回归通过。两份 PARAM 均核对为 v5、75 UOP、15 exec（14 条有效操作 + END），无提前 END；SD 四个 BIN 与源产物 SHA256 全匹配。旧根目录 21 个文件已备份到 `backups/sd_0906_dual_single/old_root`，B/C/D/Q 子目录未改动，未预放 M2/M20 输出。样例分别为 Lindau 与 Frankfurt 的既有 golden 样例，不是同一张原图。详见 `E:/RUNINFO.TXT`。

本轮未启动 build；上轮遗留 app build 已停止。**新 ELF 构建、二次 MODE_INIT 后的实板功能及本轮延时仍待用户手动 build/上板验证**，静态检查不能替代这些验证。

## 14. 0907 实板失败与完整 v4 回退

### 实验结论

`platform_0906`、PARAM v5、binary2 第一次运行的 RTL total 为 `7,971,489` cycles，ARM 为 `2,657,154` ticks，约 80 ms。`AVGPOOL=1,355,088`、`CONV_ROW_DATAPATH=6,232,575`，BLOCK5/Vec 均为 0，ERROR 占 75 cycles，最终 stage 为 `0x0c`。这表明仅执行了初始池化和首层卷积附近的前缀便退出，没有完成整网，更没有运行到 U71。不能据此把 U71 算术本身判为根因，也不能把 80 ms 记为性能提升。

`M2.BIN` 全为 0（524288 像素），类别范围检查虽然通过，但不能证明计算正确。App 正确根据 ERROR 拒绝该次性能结果，20 类测试没有继续执行。失败 mask、当时的参数/输入、源码、工具、app 及生成 IP 均备份到 `backups/rollback_u71_0907/`。

### 回退范围

1. **HLS**：`src/`、`include/` 与主 cfg 恢复为 Git `bdb83c6` 的完整 0806 成功基线；32x32 SA、原 P7/PPU、20 类上采样支持不变。去掉此后引入的 U71 pre-affine、PARAM v5 和 RR memory/WinGen 改动，不拼接新旧数据通路。TB 同步回退，仅把已失效的 artifact 目录名指向现存的两组 0809 数据。
2. **编译产物/工具**：恢复 `binary2_int8_0809` 与 `cityscapes20_int8_0809`，不重新导出或改 golden。两份 PARAM 均为 v4、75 UOP、16 exec（15 条有效 + END）；index13 为独立 U71，index14 为 U72，U71 不再与 classifier 融合。exporter、replay、整数数学和 prefix 工具同步恢复该 Git 基线。两组 v5 artifact 移到上述备份目录，不再作为当前输入。
3. **平台与 App**：切回现存 `platform_class20_0806` 的 XSA/bit/FSBL/BSP，保留双模型各一次的单图测试及 ERROR 拒绝机制。此平台只有 stage counter，没有新增 conv-internal counter，capability=0 不再阻断运行。App 只接受 v4，tag 为 `INT8-BOARD-20260907-P7-V4-ROLLBACK-DUAL-SINGLE`。旧 build/compile_commands 和 0906 HLS 生成工作目录已归档，避免旧 ELF/IP 被误用。
4. **SD 与手动入口**：`P2.BIN/B2.BIN -> M2.BIN`、`P20.BIN/C20.BIN -> M20.BIN`；上轮 M2 先备份再清走，不预填 golden。沿用各自已验证样例，B/C/D/Q 验证集目录不动。用户手动 Build App，然后用指向 `platform_class20_0806` 的 Vitis launch 测试；复用已保存平台无需先重跑 HLS/实现。

**版本边界**：0826 是最近一次 binary2 profiling 上板成功的平台，但对应新增计数端口的源码未单独提交，当前没有找到完整源码快照。因此本轮选择最后完整可追溯、且有 20 类成功记录的 0806/0809 Git 基线，不声称源码与 0826 平台完全一致。0826、0906 平台和当前 `D:/ESP_INT8_vp` 工程均保留未覆盖；旧平台的输入 XSA 已固定到备份副本，避免下一次 build platform 误读 Vivado 根目录的新 XSA。

**已完成的核对**：`git diff --exit-code bdb83c6 -- ESP_INT8_hls/src ESP_INT8_hls/include ESP_INT8_hls/hls_config.cfg` 返回 0（只有换行格式提示）；主 cfg 引用文件全部存在。SD 两份 PARAM 均为 v4/16 exec，仅最后一条 END，U71/U72 分离；四个 BIN 与主机来源 SHA256 全匹配。App 的 6 项定向静态回归、`git diff --check` 通过；旧平台 bit/FSBL/XSA 身份、启动路径和 BSP 的 FatFs 头文件/库均已检查。失败 M2 已备份并从 SD 清走，未预放 M2/M20，验证集子目录未改动。

**手动交付与验证边界**：未启动 build、CSim、综合或实现，新 ELF 尚未生成。请手动 Build App，使用 `platform_class20_0806` 的 bit/FSBL 下载；本轮复用已保存的旧平台，不需要从当前 0906 Vivado BD 重新导出。两模型顺序切换仍需实板验证，尤其 0817 曾出现第二次 INIT 后 20 类提前 ERROR。历史单模型成功不等于本次双模型切换已经通过。
