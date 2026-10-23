# workplan_1021：Conv 持续供数与计算/输出解耦

**目标：** 在 H240W240 灰度 INT8 基线上提高 MAC 有效发射率，压缩 Conv 周期，逐步接近算力受限，
不是仅让 WinGen 遇到反压。按本文三轮顺序实施，上一轮 gate 未通过不得越级；使用 executing-plans
逐项执行，普通实施过程不追加长日志，只记录 gate 失败、实现结果和上板结果。

## 0. 已验证基线与目标口径

基线为 `workplan_1011` Round 8、`platform_h240_1021`：PL **4985440 cycles / 49.854400ms**，
ARM 49.854618546ms，ERROR=0；板级 mask 0/57600、CSim logits 0/1800 mismatch。
Conv=3418908、PPU row=289114、BLOCK5=658246、Vec=433904、Pool=135617 cycles。
HLS LUT/FF/DSP/BRAM18K/URAM=343789/178344/1269/661/28；SA=1024 DSP。
100MHz 实现 WNS/WHS=+0.585/+0.010ns，route error=0，CLB95.27%。

模型为 142517700 MAC，204300 次 K-tile issue，40050 个输出 issue，80100 个 psum 半字。
静态填充率 68.12%，Conv 内发射率 5.98%，相对理论峰值的有效利用率 4.07%=68.12%×5.98%。
本轮工作预算为 **Conv<=2.50M、TOTAL<=4.10M cycles**，不是已证明可达的预测。
达到预算时 Conv 发射率仍仅约 8.17%，不是 compute-bound 的终点；归因还须有真实传输/阻塞证据。

## 1. 全轮次约束

- 固定 240x240/C1、binary2 W8A8、100MHz、32x32 SA、16-lane POST；不重训、不改量化、
  激活/饱和顺序、模型或 decoder，不恢复 U71 融合。禁止缩小 SA 来人为获得 compute-bound。
- PARAM v4 ABI、UOP/EXEC、对外 act/psum 顺序与数量不变；复用
  `hw_artifacts/binary2_int8_h240w240_gray_r6_v4`，不重导 golden，不改变 schedule 字段含义。
  Python 改动以审计、协议工作量和验证为主，exporter/replay 只补对应契约检查。
- 保留唯一 Conv/WinGen/SA/POST/PPU datapath、唯一 RAM owner、共同写尾。冻结 PPU、Vec、
  Pool、上采样及片上 FM 布局；不新增生产 CPP，不用 ALLOCATION/Tcl 限制掩盖逻辑复制。
- 仅允许**单行内部、前向 DAG** 的小型弹性队列；新增数据缓冲合计不超过 1KiB，
  depth=2 队列使用寄存器/LUTRAM，不能因宽 FIFO 推断额外 BRAM 或巨大 reset 网络。
  禁止跨行 resident DATAFLOW、反馈 FIFO、共享数组双 owner、任务数组、整行/整图副本、DDR spill。
- 保留现有 profiling 端口及 RTL counter 地址/功能；不为计时而给热循环加入等待、volatile 屏障
  或重复状态写。枚举转 ap_uint 时先转 unsigned，避免再次出现类型转换歧义。
- R8 稳定源码快照标记为 `H240W240-STABLE-20261023`；`H240W240-1020` 是 **R7R，不是 R8**。
  不创建 backups 或重复临时项目；生产 `hls_config.cfg`、顶层综合/package/实现由用户手动操作。

## 2. Round 1：校准归因，建立可信的局部 RTL 基线

**修改范围：** `tools/analyze_conv_cycle_profile.py`、`tools/analyze_sa_utilization.py` 及对应测试；
`ESP_INT8_app/src/hal/int8_npu.c`；`ESP_INT8_hls/tb/r4r_handshake_tb.cpp`、
`run_cosim_0910_round4r3.tcl`，以及 `src/conv_engine.cpp` 中仅测试宏可见的 probe。

- [ ] **纠正计数语义。** 当前生成的 `systolic_array_core_row.v` 中，state24 才写状态 1，
  state25 的 emit 完成后才写状态 2；后者会延续到下一次 K-loop。App 将两个 SA bin 显示为
  `RAW_STATE_1/2` 并提示未校准，数值/寄存器不变。Python 兼容旧 CSV 列名，但标记
  `phase_semantics=uncalibrated`；不得直接交换两个 bin，或按最大 bin 推断瓶颈。
- [ ] **复用工具，不恢复旧硬件。** 现有 r4r runner 引用的 probe 已不在生产源码，不能原样运行。
  在 `conv_engine.cpp` 增加仅 `ESP_INT8_ROW_HANDSHAKE_PROBE` 下编译的 void probe，直接调用
  当前 `shared_conv_row_engine()`；TB 用当前 PARAM 的 U8/C12 和 U40/C129 各两行验证冷/连续调用，
  逐元素比对确定性输入的整数参考，不只检查 checksum。删除 runner 中过期 C3 用例；不使用
  `conv_domain_tb.cpp` 的旧 window_generator/systolic_array_core 接口。
- [ ] **从 RTL 事件取证。** 局部 co-sim 仅记录时钟、FIFO 握手、MAC 迭代有效使能及必要完成信号，
  不 dump 宽数据总线。扩展现有分析器读取 VCD/事件 CSV，并由本次生成 RTL/schedule 解析映射，
  不硬编码 state23/25 等编号。输出 `mac_fire/input_wait/output_wait/control_drain`：
  mac_fire 每次有效 K-tile 迭代计一次，paired 两路只计一次；input_wait 必须是实际取数请求未满足，
  output_wait 必须是待发有效结果被拒绝，不能仅看 FIFO empty/full。保留多因同时发生的 mask，
  不强行归为唯一根因；缺失信号/未知值必须报未解析，不能填零。
- [ ] **修正性能模型。** `analyze_sa_utilization.py` 使用显式 H240 artifact，读取当前局部报告的
  实际 II/latency，不再默认使用旧 K-loop latency=35。输出分开列出数学下界、局部服务模型、
  实测周期；默认打印到终端，仅显式指定输出路径才写文件，禁止默认生成一批 CSV/JSON。
  测试覆盖旧日志兼容、错位 marker、缺信号、paired 计数以及输入/输出同时阻塞。

**Gate：** 两个用例 reference=0 error；FIFO token 与 MAC 事件计数和 PARAM 一致，
冷/连续调用无残留；脚本明确区分实测与下界。修正标签不得影响计算、ABI 或原始 counter。
保留 R8 的同用例 RTL 周期/事件基线供后两轮比较；**只通过状态计数求和不算校准完成**。

## 3. Round 2：C12 传输压缩与窗口准备/输出流水

**修改范围：** `ESP_INT8_hls/src/win_gen.cpp`、`tb/win_gen_tb.cpp`，上述两个分析器及测试；
`tools/export_int8_hw_blob.py`、`hw_param_replay.py` 只补审计/验证，不改变生成的 PARAM 内容。
仅替换现有 `WINDOW_TRANSFER_C12_S1_INCREMENTAL` 内部实现，其他 mode 本轮保持原状。

- [ ] **先压缩内部 token。** 复用 `window_load_word_t`，一个 token 携带同一 kh 的相邻两列 C12：

  ```text
  word[95:0] = column A 的 12 bytes
  word[191:96] = column B 的 12 bytes；word[255:192] = 0
  row = kh；slot = 局部列对起点 0 或 2；chunk = 0
  顺序：列对在外、kh=0/1/2 在内；warmup 发列对 0、2，steady 仅列对 2。
  ```

  warmup 从 12 降为 6 token，steady 从 6 降为 3；这是**内部传输数**，不是源 RAM 读取数。
  原始对齐源读数不得增加，仍使用唯一 memory owner 和缓存游标，边界数据明确补零。
- [ ] **再消除收集与输出串行。** 新增 C12 专用的局部 prepare/snapshot 和 emit 两个前向任务，
  均留在 `win_gen.cpp`。保留现有单 loader，在 assembler 行入口按既有 protocol 分发一次；
  仅 C12 进入新的局部 DATAFLOW，其他分支不变，每条 FIFO 严格单写单读。
  prepare 独占滚动四列缓存，
  生成 `ap_uint<1152>` snapshot（4列×3行×12 bytes），经 depth=2 FIFO 送给唯一 C12 emitter；
  emitter 每份 snapshot 按原 K 顺序输出 4 对 256-bit activation。这样准备下一窗口时不修改
  emitter 正在使用的数据；不用共享快照数组或反馈 credit FIFO。
- [ ] snapshot 使用静态 segment/range 拼接；不对 complete-partition 大数组动态 lane 写入。
  复用已有 C12 K-tile 组装规则，不复制 SA；emit II=1。尾部不足两个像素时，第二路仍发送
  规定数量的零向量，不能提前结束。循环终止由已知 issue/token 数决定，不能把 empty 当作结束。
- [ ] Python 增加协议审计：逻辑新列数保持原值，物理 token 按上述规则计数，snapshot 数等于
  `ceil(out_w/2)`；逐字核对 act0/act1 与原实现一致。调整旧 transfer-count 测试，但不得用新
  计数定义掩盖漏数。清除**已替代的 C12 incremental** 路径；C12 dilation>1、C17/C25 delta
  及 wide 路径仍被调度，不作为 legacy 删除，也不保留新旧 C12 incremental fallback。

**Gate：** 当前 artifact/golden hash 不变；C12 全部 activation 逐字相同、源读数不增；
同一代表行 RTL 周期至少降低 10%，其他 mode 不回退超过 1%。稳态供数目标为 1 对 act/clk，
若真实源服务带宽不足须量出下限；**仅少发 token、II=1 或加深 FIFO 不能算吞吐验收通过**。

## 4. Round 3：SA 行内计算与结果发送解耦

**修改范围：** `ESP_INT8_hls/src/sa_core.cpp`、`tb/sa_core_tb.cpp`、`tb/conv_post_tb.cpp`；
`tools/analyze_sa_utilization.py`、`check_p6_hls_structure.py` 及对应测试。

- [ ] 保留 `systolic_array_core_row()` 对外入口与 weight buffer owner，在文件内形成唯一
  compute actor 和唯一 half-word emitter，二者用 depth=2 的 1024-bit 完成结果 FIFO 相连。
  每个结果仍为 32 个 INT32 psum；新增该队列仅 256 bytes，不新增整行结果或第二套 MAC。
- [ ] compute 将逐像素反复启动的 K-loop 改为一条有界、滚动计数的流水，跨 issue 连续运行：

  ```text
  每个有效 step：接收当前 kt 的 act → 唯一 MAC/tree → 更新唯一 32-lane accumulator
  kt==0：从当前 partial 开始，不沿用上一 issue 的 psum
  kt==Ktiles-1：提交完整结果到 FIFO；下一 step 开始下一个 issue
  ```

  K 顺序、INT32 行为、paired 权重选择及 padding mask 不变；carry 的 first/last 标记必须与
  MAC pipeline 延迟对齐。固定循环上界，不增加 runtime task mux，不展开 issue/K 两个维度。
- [ ] emitter 将结果展开为原顺序的 16-lane psum 半字，连续发送，避免每 issue 启动一个子循环。
  paired/odd-tail 和有效组数契约不变；队列满时反压 compute，不能覆盖未消费结果。POST 和
  外部 psum FIFO 保持原实现。所有 DATAFLOW 子任务为 void，不能再次出现返回值综合错误。
- [ ] 仅 compute actor 写原 SA 诊断端口，emitter 不写同一端口，防止多 writer；它仍是 raw marker，
  不冒充整个 SA 的精确阶段时长。模型改为计算/发送取瓶颈而非机械相加，并保留启动/排空开销。
  Ktiles=1 时 16-lane 输出至少需 2 cycles/paired issue，不以此要求不可能的每拍一个输出 issue。

**Gate：** 无外部阻塞的单元 RTL 中，稳态 issue 间隔目标为 `max(Ktiles, half_count)`，
计算循环与 emitter 实际 II=1，尾部仅一次有界排空；至少覆盖 Ktiles=1/2/4/5/8/37、
正负极值、奇宽、连续两次调用和下游反压。结果逐元素一致；只有 K-loop II=1 不算完成解耦。

## 5. 统一验证与交付门槛

1. **每轮先局部检查。** 修改对应测试并验证失败再实现，复用现有 test 文件；只跑本轮涉及的
   契约和小用例，不为迁就过期测试恢复 C3/RGB/20-class 或旧推理路径。
   Round 1 只跑代表行；Round 2、Round 3 各在全部代码完成后跑一次 fullres CSim。
2. **fullres 验收固定。** logits=0/1800、mask=0/57600 mismatch，last_error=0；覆盖重入、
   INIT reload、行/尾边界，维持 frozen artifact。复用 `run_csim_1011_round6.tcl` 增加明确的
   本计划阶段选择，不改生产 cfg；仿真按权限要求申请 sandbox 外执行。
3. **清理后再交综合。** 扫描全部生产 CPP 的静态引用与 call graph，删除替代 helper、旧宏路径和
   重复 datapath；保留 input DMA 写入及合法 mode。运行 `tools/check_p6_hls_structure.py`、
   相关 unittest、`git diff --check`。局部 RTL 验证可包含局部 csynth，**生产顶层 csynth/
   package/实现仍由用户手动运行**；中间仅 Round 2 末、最终 Round 3 末安排顶层综合检查。
4. **资源/可综合性。** 生成 RTL 实例树确认单 SA、单 POST、单 Vec/PPU，SA=1024 DSP；
   顶层资源目标不增长，LUT/FF 超 R8 3% 须解释，超 5% 暂停；BRAM18K<=661、URAM<=28，
   不新增算术复制。估计周期<=7.300ns，无新增 II/port conflict；综合超过约 21min 先检查
   CPU/日志/报告推进，不按 URAM warning 单独判死。物理必须合法布线且 100MHz setup/hold 通过。
5. **上板后才验收性能。** 同图至少连续 3 次均 ERROR=0、输出 0 mismatch，PL/ARM 换算一致；
   Conv/TOTAL 必须下降，冻结 Stage 不退化。目标 Conv<=2.50M、TOTAL<=4.10M cycles，未达
   则保留局部收益但不标整轮性能收尾；不以 Fmax、状态 bin 或 HLS max latency 代替实测。
   之后完成 500 张精度/延时验证。局部 RTL 不代表整网已算力受限；若仍有大量输入等待或
   控制空隙，明确剩余瓶颈，不把“上游被反压”当最终成功标准。

**本次交付仅为计划；以上三轮尚未实施。旧 workplan 保留历史结果，本计划后续只追加必要的 gate 结论。**
