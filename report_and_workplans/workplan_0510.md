# ESPNet_Encoder INT8 NPU 工作进展与计划 (2026-05-09/10 合并版)

## 1. 已完成进展

这两天的工作重点是确认 C/RTL co-sim 的适用边界，并把验证主线切换到可观测的板端 debug。晚间继续推进到第一次板端分段验证和 U02 第一层卷积定位。

新版 debug 硬件平台已完成并导出，XSA 与 Vitis platform 时间戳均为 `2026-05-09 18:24:44`。HLS IP 已在 AXI-Lite control 寄存器中加入 `dbg_status`、`dbg_heartbeat`、`dbg_act_words`、`dbg_wgt_words`、`dbg_psum_words`、`dbg_out_words`，PS app 可在 timeout 时打印 phase、opcode、current/last uop 和 stream counter。当前 app 默认运行分段 debug bring-up，暂不直接整网运行。

完整 `m_axi + AXI VIP` 顶层 co-sim 多次停在 `MODE_INIT` 完成、`MODE_RUN` 不返回的位置；同时板端此前已证明 `U00/U01` 可通过，因此该 co-sim 路径不再适合作为真实硬件卡点判断依据。

随后临时切换到 `COSIM_LITE + dbg_u02_lite_tb.cpp` 并重新综合，报告确认顶层接口已变为 `ap_memory/ap_none`，排除了 AXI VIP 干扰。但真实尺寸 U02 仿真到今天仍只推进到约 `1.86%`，说明即使 lite 版本也无法承担真实 UOP 序列验证。

默认 `hls_config.cfg` 已恢复为最终上板 AXI 配置，即 `m_axi + s_axilite`，`ESP_INT8_COSIM_LITE` 已移除。`verification_plan_int8_hls.md` 也已同步更新为板端 debug 主线。

板端 `DBG05/DBG06` 运行结果显示：`DBG_U00_INPUT` 与 `DBG_U01_POOL1` 均可完成并保存输出；`DBG_U02_C1` timeout 后，新增 `DBG_U02_IN` 进一步排查，结果仍在 `stop_after=2` 处 timeout。这说明当前卡点已经收窄到 U02 第一层卷积 transaction 内部，而不是 B1_CAT 输出 dump 或 frame store 阶段。运行中 `ap_ctrl=0x1` 持续 busy；AXI-Lite debug 输出寄存器在 transaction 未返回前一直读到 0，后续不再把这些寄存器视为可靠 live progress，只作为返回后或 timeout 末端辅助信息。

针对 U02，已在 `win_gen.cpp` 中加入第一层 `3x3 stride=2 in_c=3` 专用窗口生成路径，避免第一层继续走通用 `load_window_line/load_window_vector` 动态窗口路径。默认 top CSim 已通过，随后重新跑默认 `hls_config.cfg` 的 HLS C synthesis，报告时间为 `2026-05-10 22:27-22:30`，确认补丁已进入综合。报告中 `emit_first_layer_3x3_windows` 已被综合进 `window_generator`，第一层专用路径约 `56M cycles`；但通用 `window_generator` 路径仍保留极大静态 max latency，top HLS 资源估计仍偏紧，不能据此认为性能已经收敛。

`2026-05-11` 早间复查实现、platform 和 app 后确认版本链路有效：新 bitstream、XSA、platform、app 的时间戳顺序正确，app ELF 中也包含 `DBG_U02_IN/DBG_U02_C1`。重新上板运行 `DBG06-U02SPLIT` 后，输出仍与上一版一致：`U00/U01` 完成，`DBG_U02_IN` 在 `ap_ctrl=0x1` busy 状态下 timeout。这说明单独替换第一层 `win_gen` 路径并没有解除 U02 卡死。

进一步阅读 HLS 报告后发现更核心的结构问题：`execute_conv_stream_region` 的 dataflow 只把 `push_conv_weights` 拆成独立进程，`window_generator + systolic_array_core + write_conv_post_output` 被 HLS 合并进同一个大块 `Block_newFuncRoot`。在 RTL 中这会使 `window_generator` 先写满有限深度 `act_stream`，而下游消费者尚未并行启动，形成结构性 stream 死锁风险；C 仿真中 stream 近似无界，因此不会暴露该问题。为先把板端验证边界推进过 U02，已在 `int8_core.cpp` 中加入 U02 第一层 `direct/fused` fallback，仅对 `src=T_INPUT, dst=T_B1_CAT, param_id=0, 3x3 stride2 in_c=3` 生效，其余卷积仍保持原 stream 路径。app tag 已更新为 `INT8-BOARD-20260511-DBG07-U02DIRECT`，默认 `hls_config.cfg` 的 CSim 已通过 `0 errors`。

## 2. 当前判断

真实尺寸 C/RTL co-sim 正式放弃，不再作为 package、Vivado system 实现或上板前的阻塞门槛。后续 co-sim 仅保留给小规模模块级 TB、手写小尺寸 sanity case，或必要时的定向局部验证。

当前项目已具备上板定位条件。继续等待真实尺寸 co-sim 的收益很低；板端 debug 能在真实 PS/PL/DDR/SD 环境下直接读取 `phase/uop/act/wgt/psum/out`，定位价值更高。

主要风险仍集中在第一层真实卷积及后续 stream datapath。当前已确认上一版硬件至少卡在 U02 卷积内部；`win_gen` 专用路径本身不足以解除 timeout，根因更可能是 stream dataflow 没有被 HLS 拆成真正并行的 producer/consumer。U02 `direct/fused` fallback 是为了绕过该结构问题、先验证片上 memory、参数读取、卷积数值和写回路径是否能继续推进，不是最终性能架构。

关于 latency 报告需要明确区分：`1e12 cycles` 级 max 是 HLS 对通用动态窗口路径的静态最坏估计，不能直接等同于板端真实延迟；但它确实提示当前通用 `win_gen` 结构不可作为最终性能方案。若第一层专用路径能让 U02 返回，后续再系统优化 `PIPELINE off` 和专用窗口生成结构；若 U02 仍不返回，则说明问题不只是旧版通用窗口路径，而更可能位于 stream/dataflow、片上 memory 访问或写回路径。

## 3. 下一步计划

1. 用 `DBG07-U02DIRECT` 补丁重新跑 HLS C synthesis；若综合通过，再 `Package IP`、Vivado `Upgrade IP`、system implementation/bitstream，并导出新的 XSA。
2. Vitis platform `re-read XSA` 并 rebuild platform/app；上板串口 tag 必须显示 `INT8-BOARD-20260511-DBG07-U02DIRECT`。SD 卡仍保持短文件名 `PARAM.BIN`、`INPUTQ.BIN`。
3. 上板优先验证 `DBG_U02_IN` 与 `DBG_U02_C1`。若 U02 能返回，保存 `D02IN.BIN/D02OUT.BIN` 并在 PC 端离线比对，确认 direct fallback 的第一层输出是否可接受。
4. 若 U02 通过，再按 `U04 -> U20 -> U39 -> U53 -> U69 -> U72` 推进深层 checkpoint，最后恢复完整 `MODE_RUN` 单图闭环。
5. 若 U02 仍 timeout，下一轮不再单纯拉长 timeout，而是转向片上 memory 读写、参数读取、第一层 direct 写回路径和顶层控制状态机；若 U02 通过但后续 stream conv 卡住，再系统重构 stream conv 的 dataflow 边界。
6. 性能优化暂后置。只有在第一层卷积能稳定返回后，再决定是继续推广 direct/fused 专用 conv，还是重写 `win_gen -> SA -> PPU` 为 HLS 可稳定拆分的真正 dataflow 结构。

## 4. 2026-05-11 晚间追加更新

晚间复查发现一次板端实验使用了旧版 `PARAM.BIN`：SD 卡上 `PARAM.BIN` 仍为 `2026-04-28` 版本，而当前 artifact 为 `2026-05-11` 版本；两者大小相同但 SHA256 不同。已将 `D:\ESP_INT8\hw_artifacts\hw_constrained_qat_3ep_single\param_blob.bin` 覆盖到 `E:\PARAM.BIN`，旧文件备份到 `D:\ESP_INT8\sd_backups\PARAM_before_sd_update_20260511_192445.BIN`。`INPUTQ.BIN` 与 golden 输出经哈希确认无需更新。

结合未完成的 `dbg_u02_lite_tb` cosim 进度估算，U02 transaction 若只是正常慢，100 MHz 下应约 `21.5s` 完成；因此板端等待到 `60s` 量级仍不返回时，可以判定不是单纯 timeout 太短，而是 U02 卷积路径仍存在卡死或不可接受的异常慢。

对“为什么前一版仍没跑通 U02”的判断已进一步修正：此前主要修了 `win_gen` 和第一层窗口路径，但顶层 `execute_conv_uop()` 实际仍没有默认调用 stream datapath。换言之，`win_gen` 修补没有进入 U02 的实际执行路径；后续 direct/fused fallback 也只是临时绕路，不是最终 stream 架构。

现已补齐关键顶层修补：`int8_core.cpp` 中 `execute_conv_uop()` 默认调用 `execute_conv_stream_datapath()`，direct conv 仅在显式定义 `ESP_INT8_USE_DIRECT_CONV` 时作为 emergency fallback。`hls_config.cfg` 已确认是默认 AXI 上板配置，无 `ESP_INT8_COSIM_LITE`。默认 CSim 已通过 `top_conv_dispatch_tb passed`，且最大 `hls::stream` 深度为 `32`，说明当前顶层已实际走到 stream 路径。app tag 更新为 `INT8-BOARD-20260511-DBG11-STREAMTOP`。

随后根据 HLS 214-475 的真实根因继续修补：仅切回整层 stream path 仍不充分，因为 `window_generator` 读片上 feature memory、`write_conv_post_output` 写同一物理数组会被 HLS 识别为 feedback 并合并进程。当前已改为 `DBG12-ROWSTREAM`：每个输出行单独执行 `window_generator_row -> push_conv_weights -> systolic_array_core_row -> post_process_row_to_buffer` 的 DATAFLOW，DATAFLOW 区域内只读源 tensor、不写目标 tensor；该行计算完成后再在 DATAFLOW 外调用 `store_conv_output_row` 写回。旧的整层 `execute_conv_stream_region` 和 `write_conv_post_output` 已移除，direct conv 仍仅作为 `ESP_INT8_USE_DIRECT_CONV` emergency fallback。

`DBG12-ROWSTREAM` 默认 CSim 已通过 `top_conv_dispatch_tb passed`，最大 `hls::stream` 深度为 `32`；`hls_config.cfg` 已确认无 `ESP_INT8_COSIM_LITE` 和 `ESP_INT8_USE_DIRECT_CONV`，默认综合路径即 row-stream。下一步以 `DBG12-ROWSTREAM` 为新的硬件导出基线：先跑 HLS C synthesis，重点 grep `HLS 214-475` / `Merging processes` / `feedback on`；若该红灯消失，再检查 BRAM/LUT/URAM 与 latency，随后 package IP、Vivado upgrade IP、system implementation/bitstream、导出 XSA、rebuild platform/app。重新上板时优先验证 `DBG_U02_IN` 与 `DBG_U02_C1`。
