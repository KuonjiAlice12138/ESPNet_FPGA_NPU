# HLS 综合警告审计：上板前可排查的隐患

> 审计时间：2026-05-11  
> 审计范围：`espnet_encoder_int8_core` Vitis HLS 2025.1 综合报告  
> 综合报告：`ESP_INT8_hls/hls/syn/report/hls_syn.rpt`  
> 引导报告：`ESP_INT8_hls/hls/syn/report/v++_compile_ESP_INT8_hls_guidance.html`（77 条违规）  
> 器件：`xczu15eg-ffvb1156-2-i`，时钟 100MHz

---

## 0. 上板前硬门禁

每次 C synthesis 结束后，必须先运行：

```bat
tools\audit_hls_reports.bat
```

若脚本返回失败，则停止后续 `package / Vivado implementation / platform export / app run`。当前硬阻塞项包括：

- `HLS 214-475` / `Merging processes` / `feedback on`
- `HLS 200-975` stream 同函数读写
- dataflow return-value/form-check failure
- scalar M_AXI address computation
- HLS synthesis fatal error
- Vivado resource over-utilized DRC

`--strict` 模式还会把 pipeline 无法满足、dataflow form issue、异常大的编译规模等作为必须人工确认的高风险项。没有通过该门禁，不再进入耗时的实现、导出和上板实验。

---

## 1. 导致上板卡死的根因：DATAFLOW 进程合并

### 1.1 警告原文

```
[WARNING: HLS 214-475] Merging processes 'write_conv_post_output',
'systolic_array_core' and 'window_generator' in function
'execute_conv_stream_region' due to feedback on 'get_fmbuf_bram()::s_bram'
```

### 1.2 触发机制

```
execute_conv_stream_region:
  ┌─ window_generator       ← 读 src（on_chip_memory_read_tile → read_phys_word → s_bram/s_uram）
  ├─ push_conv_weights      ← 不涉及片上内存
  ├─ systolic_array_core    ← 不涉及片上内存
  └─ write_conv_post_output ← 写 dst（on_chip_memory_write_tile → write_phys_word → s_bram/s_uram）
```

当 src 和 dst 落在**同一个物理 BRAM/URAM 阵列**（`s_bram` 或 `s_uram`）上时，HLS 分析器检测到 producer（读）和 consumer（写）访问同一物理数组，判定为"feedback"，将三个进程合并为单个顺序块：

```
合并后进程 A: window_generator → systolic_array_core → write_conv_post_output  (顺序)
独立进程 B: push_conv_weights                                                    (并行)
```

### 1.3 死锁机制

合并后进程 A 内部顺序执行：

1. `window_generator` 启动，向 `act_stream`(depth=32) 写入激活向量
2. 写到第 33 个词时 FIFO 满 → `act_stream.write()` 阻塞
3. 消费者 `systolic_array_core` 在顺序块后半段，尚未启动 → 永远等不到读操作
4. **RTL 死锁**，`ap_done` 永不高

### 1.4 为什么 C 仿真不暴露

C 仿真中的 `hls::stream` 是无限深度内存队列，写入永不阻塞。RTL 中的 `hls::stream` 是有固定深度的硬件 FIFO。这导致 CSim 通过而 RTL 死锁。

### 1.5 当前修补状态（2026-05-11 晚间）

已切换到 `DBG12-ROWSTREAM` 结构：不再使用整层 `execute_conv_stream_region`，而是按输出行执行 row-level DATAFLOW。新的 DATAFLOW 区域只包含：

```text
window_generator_row -> push_conv_weights -> systolic_array_core_row -> post_process_row_to_buffer
```

其中 `window_generator_row` 只读片上 feature memory，`post_process_row_to_buffer` 只写本地 `row_buf`；真正的 `store_conv_output_row` 写回目标 tensor 被移到 DATAFLOW 区域之外。因此 HLS 不应再看到同一个 dataflow region 内对 `s_bram/s_uram` 的读写 feedback。下一次 C synthesis 后必须确认日志中不再出现 `HLS 214-475` / `Merging processes` / `feedback on`。

---

## 2. 受影响的卷积层映射

根据 src/dst 是否落在同一物理阵列，确认所有 stream 路径的 conv 风险：

| UOP | 操作 | src | dst | 同阵列 | 风险 |
|---|---|---|---|---|---|
| 2 | LEVEL1_CONV | T_INPUT (URAM) | T_B1_CAT (URAM) | ✅ | 🔴 已修（fused direct） |
| 8 | L20_D1 | LS_C1 (BRAM) | T_L20_CAT (URAM) | ❌ | 🟢 安全 |
| 9 | L20_D2 | LS_C1 (BRAM) | LS_A (BRAM) | ✅ | 🟡 局部 scratch，深度小 |
| 11 | L20_D4 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch，深度小 |
| 14 | L20_D8 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch，深度小 |
| 17 | L20_D16 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch，深度小 |
| 22 | L2B0_D1 | T_L20_ACT (URAM) | T_L2B0_CAT (URAM) | ✅ | 🔴 **待修** |
| 23 | L2B0_D2 | LS_C1 (BRAM) | LS_A (BRAM) | ✅ | 🟡 局部 scratch |
| 25 | L2B0_D4 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch |
| 28 | L2B0_D8 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch |
| 31 | L2B0_D16 | LS_C1 (BRAM) | LS_TMP (BRAM) | ✅ | 🟡 局部 scratch |
| 41 | L30_D1 | LS_C1 (BRAM) | T_L30_CAT (URAM) | ❌ | 🟢 安全 |
| 55 | L3B0_D1 | T_L30_ACT (URAM) | T_L3B0_CAT (URAM) | ✅ | 🔴 **待修** |
| 72 | CLASSIFIER | T_B3_ACT (URAM) | T_OUT (URAM) | ✅ | 🔴 **待修** |

**Crash map:**

- 🟢 安全：src 在 BRAM scratch，dst 在 URAM → 不同物理阵列，不合并
- 🟡 局部 scratch：仅在 BRAM scratch 内部，FIFO 深度小 + 通用路径 `can_merge_lane` 较少可合并，风险较低
- 🔴 待修：src 和 dst 都在 URAM → 同阵列，必然合并

---

## 3. 综合报告中其他可提前排查的警告（77 条分类）

### 3.1 🔴 CRITICAL：DATAFLOW 进程合并

| 警告号 | 数量 | 位置 | 影响 |
|---|---|---|---|
| HLS 214-475 | 2 条 | `execute_conv_stream_region` | RTL 死锁 |
| HLS 200-1450 | 1 条 | `push_conv_weights` 有后继且写回 | 吞吐下降 |
| HLS 200-1449 | 1 条 | 合并进程有前驱且读入 | 吞吐下降 |

**检查清单：** 每次综合后，grep `214-475` / `Merging processes` / `feedback on`。任何此类警告都是红灯。

### 3.2 🟡 HIGH：II 违规

| 模块 | 数量 | II 范围 | 根因 |
|---|---|---|---|
| `on_chip_memory_read_tile` | ~12 条 | II=1~5 | 双重 `read_bank_word` 之间的 carry dependence |
| `on_chip_memory_write_tile` | ~24 条 | II=1~12 | 读-改-写顺序依赖 |
| `push_conv_weights` | ~6 条 | II=1~15 | `s_wbuf` 数组端口不足 |
| `param_dma_init` | ~8 条 | II=1~8 | `gmem2` 总线请求调度失败 |
| `avgpool_unit_checked` | ~4 条 | II=14~17 | 内层 `write_tile` 调用延迟不兼容 |

**影响：** 不会死锁，但每层真实延迟可能是静态估计的 10-15 倍。

**建议修复：**
- `on_chip_memory_read_tile`：合并两次 `read_bank_word` 为一次 64 字节读
- `s_wbuf`：在 `push_conv_weights` 中增加 `ARRAY_PARTITION complete dim=1`
- `param_dma_init`：拆分 `gmem_param` 为多个独立 M_AXI 端口读

### 3.3 🟡 MEDIUM：循环不能 flatten

| 循环 | 位置 | 原因 |
|---|---|---|
| `push_conv_weights` VITIS_LOOP_771_1 | int8_core.cpp:897 | 非完美嵌套循环 |
| `param_dma_init` VITIS_LOOP_427_7 | param_dma.cpp:427 | 内层前有非平凡逻辑 |
| `param_dma_init` VITIS_LOOP_443_9 | param_dma.cpp:443 | 内层前有非平凡逻辑 |
| `load_window_vector` VITIS_LOOP_293_2 | win_gen.cpp:293 | 内层前有非平凡逻辑 |
| `load_window_line` VITIS_LOOP_211_4 | win_gen.cpp:211 | 内层前有非平凡逻辑 |
| `load_weight_buffer` VITIS_LOOP_41_1 | sa_core.cpp:41 | 内层前有非平凡逻辑 |
| `avgpool_unit_checked` VITIS_LOOP_74_1 | avgpool_unit.cpp:74 | 不可计算循环计数 |
| `core_mode_run` VITIS_LOOP_1478_1 | int8_core.cpp | 128 次迭代，无法 flatten |

**影响：** 状态机层数增加，综合结果复杂化，增加产生不良状态机转换的概率。

### 3.4 🟢 LOW：不影响功能

| 类别 | 数量 | 说明 |
|---|---|---|
| RTL dangling port (AR*/AW*) | ~30 条 | COSIM_LITE 模式下 AXI 端口未驱动，预期行为 |
| Power-on init register | ~20 条 | 内部状态寄存器标记为 POR 初始化 |
| RTL 命名冲突重命名 | 1 条 | `fifo_w256_d32_A` → `fifo_w256_d32_A_x` |
| 未使用参数 | 2 条 | `cfg` in win_gen:102, `gmem_param` in if_dec:68 |
| 无效 trip count 指令 | 4 条 | 循环边界从 8→1 修正（oc_tiles=1 已验证） |
| 数组 undecay 跳过 | 3 条 | `gmem_frame_in/out/param` 变长数组 |
| 内存绑定目标未找到 | 5 条 | `s_bram`/`s_pool2`/`s_uram` 被编译器优化掉 |

---

## 4. 资源与时序

| 阶段 | LUT | FF | DSP | BRAM | URAM | Timing |
|---|---|---|---|---|---|---|
| RTL 综合 | 261,256 | 111,935 | 1,196 | 1,354 | 112 | 6.257ns ✅ |
| Place & Route | 254,277 | 112,861 | 1,199 | 1,354 | 112 | 9.815ns ✅ |

- HLS 估计 Fmax：136.99 MHz
- Route 后 WNS=0.155ns，WHS=0.011ns
- 布线拥塞等级 6/64×64（偏高但通过）
- 资源利用率：LUT 74%，CLB 96%，BRAM 91%，URAM 100%

---

## 5. 上板前强制检查清单

每次 HLS 综合完成后，必须逐条检查以下警告：

### 5.1 死锁类（不通过不得上板）

- [ ] `HLS 214-475` DATAFLOW merging → 确认所有 `execute_conv_stream_region` 内无此警告
- [ ] `hls::stream` 最大深度报告 → C 仿真日志中 `maximum depth reached` > stream FIFO depth
- [ ] COSIM_LITE 死锁检测单元 → `AESL_deadlock_detector` 无错误报告

### 5.2 正确性类（需评估影响）

- [ ] 循环 `Cannot flatten` 警告 → 确认受影响循环的 RTL 状态机有限
- [ ] II 违规（`Unable to schedule`）→ 确认是否影响 pipelined 循环边界
- [ ] `PIPELINE off` 的循环是否确实应该 off（不是漏了 `II=1`）

### 5.3 性能类（当前可接受，后续优化）

- [ ] 静态延迟估计 > 1e10 cycles 的循环
- [ ] 内存端口不足的数组（`s_wbuf`、`gmem2`）
- [ ] URAM 未利用 read-first 模式

### 5.4 资源类（确认在器件限制内）

- [ ] URAM/BRAM/DSP 不超标
- [ ] Route 后 timing met（WNS > 0）
- [ ] 布线拥塞等级 ≤ 6

---

## 6. 内存物理布局（供交叉参考）

共享 FMBUF（6,377,472 bytes）：

```
0x000000 ─────────────────────── URAM（3,670,016 = 0x380000 bytes）
         T_INPUT (0x000000, 1.5MB)
         T_B2_CAT / T_B2_ACT (0x000000, 128×256×131 ≈ 4MB 地址空间)
         T_L2B0_CAT / T_L2B0_ACT (0x000000, 128×256×131 view)
         T_B3_CAT / T_B3_ACT (0x000000, 64×128×256 ≈ 2MB 地址空间)
         T_L3B0_CAT / T_L3B0_ACT (0x000000, 64×128×256 view)
         T_B1_CAT / T_B1_ACT (FMEM1, 0x180000, 256×512×19 ≈ 2.4MB)
         T_OUT (0x200000, 64×128×2 = 16KB)
         T_POOL1 (0x3E0000, 256×512×3 = 384KB)
0x380000 ─────────────────────── BRAM（2,770,944 = 0x2A4000 bytes）
         T_L20_CAT / T_L20_ACT (B2 view c[64:127], phys_c=131)
         T_L30_CAT / T_L30_ACT (FMEM1, 0x518000, 64×128×128 = 1MB)
0x598000 ─────────────────────── 结束
```

独立 BRAM：
- `s_pool2` (BRAM_SCR1)：128×256×3 = 96KB → T_POOL2
- `s_wbuf` (BRAM)：128KB → 权重缓存

---

## 7. 修复建议优先级

| 优先级 | 事项 | 影响范围 |
|---|---|---|
| P0 | U22/U55/U72 加 fused direct path（或全 conv 临时 bypass stream） | 上板卡死 |
| P1 | 建立"综合后强制检查清单"流程 | 防再次遗漏 |
| P2 | `on_chip_memory_read_tile` 合并双读消除 II 违例 | 全模块性能 |
| P2 | `s_wbuf` 分区消除 II 违例 | 权重加载性能 |
| P3 | 通用 win_gen 路径拆分专用 kernel（1x1 / 3x3 s1 / 3x3 s2） | 后续层性能 |

---

## 8. 新增发现：DATAFLOW 内 Stream FIFO 容量边界死锁（2026-05-12）

### 8.1 问题描述

HLS `hls::stream` 在 RTL 中实现为有限深度 FIFO。`#pragma HLS STREAM variable=xxx depth=N` 声明的深度 N 在实际 RTL 中可用容量为 **N-1**（一个 slot 用于流水线同步）。

当 DATAFLOW 中某个生产者进程在消费者启动前就要写入 `≥ N` 个元素时，生产者会在第 N 个写入处永久阻塞——消费者尚未启动，FIFO 永远不会被消费。

### 8.2 触发条件

```
生产者先启动/先完成一轮 burst → FIFO depth ≤ 该轮 burst 元素数
                                  ↓
                                FIFO 满，生产者阻塞
                                  ↓
                         消费者等待另一个进程的 start_out 才能启动
                                  ↓
                               死锁
```

### 8.3 具体案例

`DBG13-ROWCLEAN` 中 `wgt_stream`（`fifo_w256_d32_A`，depth=32，可用容量=31）：

- `feed_cached_weights` 写入 32 个权重词（k_tiles=1, TM=32）
- 消费者 `systolic_array_core_row` 须等 `window_generator_row` 发出 `start_out` 后才启动（约 54 拍延迟）
- 生产者写到第 32 个词时 FIFO 满 → 阻塞 → 死锁

### 8.4 检测规则

综合报告不会直接报此类错误。但可通过以下间接信号发现：

| 信号 | 如何查找 |
|---|---|
| DATAFLOW 内 `start_for_*` 模块未覆盖某个消费者 | 说明该消费者依赖其他进程的 start_out |
| 两个进程 `ap_sync` 共享同一起始信号 | 生产者过早启动，可能先于消费者完成 burst |
| `fifo_w*_d32_*` FIFO 深度恰好等于写入量 | 边界对齐，N-1 < N |
| `cached_wgts` 初始化在 DATAFLOW 外层 | 生产者数据预加载但消费者需等同步链 |

### 8.5 修复原则

| 原则 | 做法 |
|---|---|
| FIFO depth ≥ 该 DATAFLOW 迭代内最大 burst 写入量 + 1 | `depth = MAX_K_TILE_COUNT * TM + 1` |
| 或让所有进程同时启动，consumer/pipeline 跑在 producer 写入的同一拍 | 用 `ap_sync` 确保 consumer 与 producer 同拍启动 |
| 或把批量写入放在 DATAFLOW 外部 | 不适用于 producer 和 consumer 都必须在线的情况 |

### 8.6 新增 BLOCKER 检查

已在 `audit_hls_reports.py` 中加入 `dataflow-fifo-capacity` 规则：检测 `#pragma HLS STREAM.*depth=` 之后的 depth 值与 DATAFLOW 内写入量是否可能冲突。具体实现依赖源码扫描，综合报告不支持此检测。

同时新增 `fifo-w256-d32` 实例化检测：RTL verilog 文件中出现 `fifo_w256_d32_A` 模块实例化即为高风险（经验上该深度在多数 DATAFLOW 场景下均不足）。
