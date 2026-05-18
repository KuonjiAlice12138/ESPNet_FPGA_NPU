# On-Chip BRAM/URAM Controller (片上缓存控制器)

## 1. 模块职责
该模块负责三件事：

1. 管理片上特征图 bank
2. 管理片上权重/量化参数 bank
3. 为卷积和池化模块提供无冲突读口

它不是简单的“BRAM 控制器”，而是整个单 IP NPU 的片上存储调度中心。

## 2. 存储层次冻结

### 2.1 总体原则
首版实现不再假设“几乎全部特征图空间都交给 URAM”。冻结后的实现原则是：

- **URAM**：负责大块、深度优先、顺序访问为主的 `FMBUF`
- **BRAM**：负责 `WBUF / QBUF / line buffer / window buffer / stream FIFO / small scratch`
- **LUTRAM/寄存器**：负责很小的局部状态和短深度队列

原因：

- Vitis HLS 当前目标器件报告给出 `URAM=112`、`BRAM_18K=1488`
- 若静态预留 `9 MB+` 级大 bank，会直接超过片上 RAM 可实现边界
- 实际实现中还必须给 bank 划分、FIFO、复制、地址控制和布线留余量

### 2.2 FMBUF
特征图全局缓存，采用 **URAM 为主、BRAM 为辅**，使用**稠密 NHWC INT8 布局**。

冻结原因：

- NHWC 与当前 FP32 C 代码一致
- concat 地址计算简单
- 不引入全局 `C pad to 32` 的巨大存储浪费

### 2.3 WBUF
权重缓存，首版固定使用 BRAM。

当前模型整个 INT8 权重仅约 `108 KB`，因此 `v1` 直接采用**一次性整模预加载**，而不是必须做层间 ping-pong。

### 2.4 QBUF
量化参数缓存，首版固定使用 BRAM，用于存放：

- bias
- requant mul/shift
- affine mul/bias/shift
- 指令表
- tensor descriptor 表

## 3. 最终 bank 划分

### 3.1 特征图 bank
首版不再把 `FMEM0/1/2` 实现为三块独立静态大 RAM。综合约束修正后，`FMEM0/1/2` 只保留逻辑 bank/view 语义，物理实现冻结为一个共享 FMBUF：

| Region | Fixed Size | Main Impl | Usage |
|---|---:|---|---|
| `FMBUF_URAM` | `0x380000` bytes | URAM | 输入、B2/B3 主容器低地址主体 |
| `FMBUF_BRAM` | `0x218000` bytes | BRAM | B1、L2/L3 局部 scratch、pool scratch、高地址尾部 |

冻结后的 `FMBUF` 物理总预算为 `0x598000` bytes。该预算依赖三项必须同时满足的调度约束：

1. `POOL_B2_TMP/POOL_B2_OUT` 提前到 B1 结束后、L20 开始前执行，使 `T_INPUT` 可提前释放。
2. `T_L20_CAT/ACT` 固定作为 B2 容器 `c[64:127]` view，避免单独保存 2 MB L20 tensor。
3. `T_L2B0_CAT/ACT` 和 `T_L3B0_CAT/ACT` 固定作为 B2/B3 容器的 channel-slice view，避免再额外保存完整 2 MB 级分支 tensor。

### 3.2 参数 bank

| Bank | Fixed Size | Main Impl | Usage |
|---|---:|---|---|
| `WBUF` | `128 KB` | BRAM | 全部 INT8 权重 |
| `QBUF0` | `64 KB` | BRAM | conv/add/pool qparam |
| `QBUF1` | `64 KB` | BRAM | affine / scale desc / uop / tensor desc |

### 3.3 行缓存与局部缓存

| Buffer | Fixed Impl | Notes |
|---|---|---|
| line buffer | BRAM | `3x3` / dilated conv 必需 |
| window buffer | BRAM | 每次服务一个 `Cin tile=32` |
| stream FIFO | BRAM | 首版不允许把深 FIFO 放成 LUTRAM |
| add scratch | BRAM | residual / branch 累积 |
| `BRAM_SCR0` | BRAM | pool / add / concat 小 tensor scratch |
| `BRAM_SCR1` | BRAM | pool / add / concat 小 tensor scratch |

## 4. 生命周期规划
这是本设计能否压到 `50 ms` 的关键。

### 4.1 生命周期原则

1. 输入帧必须在 B2 构建完成前保留，因为还要走两次 `avgpool`
2. B1 之后不再需要 level1 原始输出，可覆盖
3. B2 完成后可释放 input 占用的大 bank
4. stage3 仅需保存 `b2_act`、`level3_xxx`、`b3_act` 和最终输出

### 4.2 固定复用顺序

| Time | Low FMBUF | High FMBUF |
|---|---|---|
| Start / B1 | input, B1, pool1 | pool tmp |
| Early pool2 | input, B1, pool2 scratch | pool tmp |
| Level2_0 | L20 view at B2 `c[64:127]` | LS_C1 high scratch |
| Level2_B0 / B2 | B2 container, L2B0/L20 views | L2 block scratch |
| Level3_0 | B2 container, low scratch after first conv | L30 compact + LS_C1 scratch |
| Level3_B0 / B3 | B3 container, L3B0 view | L30 compact |
| Classifier | B3 container, output | free |

### 4.3 关键实现约束

- 不允许因为“bank 不够”而退化成层间大规模 DDR spill
- 若某阶段临时 tensor 无法长驻 URAM，则放入 BRAM scratch，而不是回 DDR
- 只有输入和最终输出是默认 DDR 可见对象
- `CAT -> ACT` 物理别名在首版 **固定开启**
- `T_POOL_TMP`、小 residual scratch 和 line/window buffer 全部进入 BRAM scratch，不占用全局 URAM bank

## 5. 地址生成规则

### 5.1 稠密 NHWC 布局
compact tensor 在 FMBUF 中使用：

`addr = base + ((h * W + w) * C + c)`

其中：

- 单位为 byte
- 元素类型为 signed `int8`
- `base` 由 tensor descriptor 给出

view tensor 使用：

`addr = base + ((h * W + w) * physical_c_stride + channel_offset + c)`

其中 `tensor_desc_t.reserved0` 固定解释为 `physical_c_stride`，`tensor_desc_t.reserved1` 固定解释为 `channel_offset`。

### 5.2 concat 写入
concat 不搬运旧数据，不创建临时拼接模块，而是由多个 producer 直接写目标 tensor 不同通道区间：

`dst_c = c_offset + c_local`

例如：

- B1：`level1_act` 写 `c[0:15]`，`pool1` 写 `c[16:18]`
- B2：`block0` 直接写 B2 view 的 `c[0:63]`，`level2_0` 复制到 `c[64:127]`，`pool2` 写 `c[128:130]`
- B3：`level3_block0` 直接写 B3 view 的 `c[128:255]`，`level3_0` 复制到 `c[0:127]`

## 6. window generator 所需局部缓存
`3x3` 卷积不能直接从 FMBUF 按单点随机读，每层都重新拼 `9` 个点会严重拖慢阵列。因此必须提供行缓存。

### 6.1 line buffer 规格

- 支持 kernel=`3`
- 支持 dilation=`1/2/4/8/16`
- 支持 stride=`1/2`
- 每次服务一个 `Cin tile = 32`

### 6.2 实现方式
每个 active `Cin tile` 建立：

- 3 行 line buffer
- 1 个 `3 x 3 x 32` window buffer

窗口更新后输出到 `systolic_array_core`。

### 6.3 1x1 特例
`1x1` 卷积不走 line buffer，只做直读打包。

## 7. 读端与写端仲裁

### 7.1 必需端口
on-chip memory 逻辑上必须支持：

1. `conv read`
2. `pool read`
3. `postproc / concat write`
4. `residual read`

### 7.2 v1 仲裁规则
首版不追求全算子完全并发，冻结为：

- `conv read` 优先级最高
- `postproc write` 与 `conv read` 允许 dataflow 重叠
- `pool read` 与 `conv read` 不要求在同一 bank 同拍并发

这样更符合 HLS 的实际收敛路径。

## 8. DDR 入口封包格式
为提高突发效率，DDR <-> 片上之间统一按 `256-bit` 搬运。

固定类型：

```cpp
typedef ap_uint<256> axi_vec_t;
```

而 FMBUF 内部仍按 byte 语义组织。读写模块负责 pack/unpack。

## 9. HLS 接口冻结

```cpp
struct tensor_desc_t {
    uint8_t bank_id;
    uint32_t base_offset;
    uint16_t h;
    uint16_t w;
    uint16_t c;
};

void on_chip_memory_read_tile(...);
void on_chip_memory_write_tile(...);
void window_generator(...);
```

`bank_id` 只允许：

1. `0` = `FMEM0`
2. `1` = `FMEM1`
3. `2` = `FMEM2`
4. `0x80` = `BRAM_SCR0`
5. `0x81` = `BRAM_SCR1`

## 10. 首版冻结结论
本模块的最终冻结结论是：

1. **URAM/BRAM 必须混用**
2. **物理 `FMBUF` 固定为 `0x598000`，其中 `0x380000` 绑定 URAM，`0x218000` 绑定 BRAM**
3. **`CAT -> ACT` 物理别名默认开启**
4. **B2/B3 的部分分支 tensor 固定采用 channel-slice view，避免重复占用片上大容量**
5. **首版按 `100 MHz` 验收，不再为 `200 MHz` 预留额外结构自由度**

## 11. 必测场景

1. `input -> pool1 -> B1 concat`
2. `B2 concat` 三路写入不互相覆盖
3. `Cin=131` 的尾 tile 读取
4. `dilation=16` 行缓存访问
5. `stride=2` 与 `same padding`
6. 大 tensor 覆盖复用后不污染下一阶段数据
