# Hardware Pooling and Activation (激活与池化固化)

## 1. 模块职责
本文档覆盖 4 个非卷积数据通路模块：

1. `avgpool_unit`
2. `activation_unit`
3. `eltwise_add_path`
4. `concat_writer`

这 4 条路径在当前网络里都是真实存在的，并且如果继续留在 CPU 端，会直接破坏 `50 ms` 目标。

### 1.1 与首版实现目标的关系
这些模块都挂在单一 HLS top 内部，不单独复用板上的 4 个外部 AXI DMA。首版实现目标优先级如下：

1. `100 MHz` 下 bit-exact 可实现
2. 不引入额外 DDR 往返
3. 支持后续 `150 MHz` 优化

因此本文件中的 `pool / add / concat / activation` 都按“浅流水、轻控制、片上完成”的原则定义，而不是按大规模独立 IP 定义。

## 2. activation_unit

### 2.1 支持模式
首版冻结为：

- `ACT_NONE`
- `ACT_RELU`

`PReLU` 仅预留 opcode，不纳入首版验收。

### 2.2 计算规则

```cpp
int8_t relu_i8(int8_t x) {
    return (x < 0) ? 0 : x;
}
```

activation 放在 post process 末端，不单独占一条 DMA 或存储路径。

### 2.3 存储约束
`activation_unit` 不拥有独立大容量 buffer：

- 大 tensor 输出直接写回 `FMBUF`，以 URAM 为主
- 极小 tile 或逐像素中间结果固定放在寄存器/LUTRAM
- 不允许为单独的 `relu`/`identity` 再开一块独立全局 feature-map 存储

## 3. avgpool_unit
当前 FP32 工程中，存在 3 次 CPU avgpool：

1. `input -> pool1`，生成 B1 的 3 通道分支
2. `input -> temp_pool`
3. `temp_pool -> pool2`，生成 B2 的 3 通道分支

INT8 版本必须全部下沉到硬件。

### 3.1 仅需支持的模式

- Kernel：`3x3`
- Stride：`2`
- Padding：`same`
- Channels：`3`

这不是通用 pool IP，而是为当前 encoder 定制的精简模块。

### 3.2 计算公式
`avgpool_unit` 的数值路径冻结为两段：

1. 先在输入 scale 域完成整数平均
2. 若 `pool_qparam.same_scale = 0`，再做一次 requant 到目标 tensor scale

当输入/输出共用同一 tensor scale 时，可直接做整数平均：

```cpp
sum = p0 + p1 + ... + p8;      // int16/int32
avg = round(sum / 9);
out = clamp_int8(avg);
```

为避免除法器，HLS 中使用常数乘法：

```cpp
avg = round_shift(sum * 7282, 16); // 7282 / 65536 ~= 1/9
```

若 `same_scale = 0`，则继续执行：

```cpp
scaled = (int32_t)avg * mult;
rounded = round_shift(scaled, shift);
out = clamp_int8(rounded);
```

### 3.3 边界处理
边界按 zero padding 处理，与卷积的 `same padding` 一致。

### 3.4 原型冻结

```cpp
void avgpool_unit(
    const tensor_desc_t& src,
    const tensor_desc_t& dst,
    const pool_qparam_t& qparam,
    ap_int<8>* fmbuf_base);
```

### 3.5 存储实现冻结
`avgpool` 的局部缓存冻结为：

- 行缓冲和 `3x3` window buffer 使用 BRAM
- 滑窗边界状态可用寄存器/LUTRAM
- 输出根据 tensor 生命周期写回逻辑 `FMEM`，其物理实现由 `04_On_Chip_BRAM_Controller.md` 决定

也就是说，`POOL` 的“源/目标 tensor 在 FMEM”是逻辑语义；真正的 line buffer/window buffer 不应消耗 URAM。

### 3.6 当前网络中的 3 个 pool 语义
当前 Encoder 只允许这 3 个 pool 配置：

1. `POOL_B1`
   - `src = T_INPUT`
   - `dst = T_POOL1`
   - `same_scale = 0`
   - `dst_scale = scale(T_B1_CAT)`
2. `POOL_B2_TMP`
   - `src = T_INPUT`
   - `dst = T_POOL_TMP`
   - `same_scale = 1`
   - `dst_scale = scale(T_INPUT)`
3. `POOL_B2_OUT`
   - `src = T_POOL_TMP`
   - `dst = T_POOL2`
   - `same_scale = 0`
   - `dst_scale = scale(T_B2_CAT)`

## 4. eltwise_add_path
当前网络中的 add 可分成两类：

1. 分支内部累积 add
2. block 输出与 residual 的 add

### 4.1 首版策略

- 输入来自 FMBUF
- 输出直接写回 FMBUF
- 当前网络 14 个 add 全部要求输入 scale 已在离线导出阶段对齐
- 小 tile 累加 scratch 固定使用寄存器或 BRAM scratch

### 4.2 计算位宽

```cpp
int16_t tmp = (int16_t)a + (int16_t)b;
int8_t out = clamp_int8(tmp);
```

如果 `uop.flags.requant_bypass = 0`，则进入 `03_Data_Formatter_and_Quantization.md` 定义的 `ADD_POST` 流程。
但对当前 Encoder，所有 `ADD` 的最终配置冻结为：

- `requant_bypass = 1`
- `act_type` 仍按各节点 uop 指定
- `src_scale_a = src_scale_b = dst_scale`

### 4.3 实现约束
`eltwise_add_path` 不应被实现成“整 tensor 读出到外部再写回”的独立模块，而是：

- 从逻辑 `FMEM` 流式读取两个输入
- 在局部 BRAM scratch / 寄存器中完成对齐与饱和
- 结果立即写回目标逻辑 `FMEM`

这点对 `100 MHz` 首版闭合很关键，因为 residual add 若退化成大范围搬运，会同时伤害性能和时序。

## 5. concat_writer
concat 在当前网络中出现 7 次，且都是性能关键路径。

### 5.1 设计原则
concat 不做“读两路再复制一遍”，而是：

- 各 producer 直接写目标 tensor 的不同 channel 区间
- 目标 tensor 的 descriptor 在执行前已固定
- 目标 tensor 所在 `FMEM` 是逻辑 bank，不代表要为 concat 单独静态保留物理大 bank

### 5.2 地址公式

```cpp
dst_addr =
    dst_base +
    ((h * dst_w + w) * dst_c_total) +
    channel_offset +
    c_local;
```

### 5.3 当前网络里的固定 concat 偏移

| Node | Offsets |
|---|---|
| B1 | `0`, `16` |
| Level2_0 cat | `0`, `16`, `28`, `40`, `52` |
| Level2_Block0 cat | `0`, `16`, `28`, `40`, `52` |
| B2 | `0`, `64`, `128` |
| Level3_0 cat | `0`, `28`, `53`, `78`, `103` |
| Level3_Block0 cat | `0`, `28`, `53`, `78`, `103` |
| B3 | `0`, `128` |

这些偏移来自现有 FP32 `espnet_encoder.c`，在 INT8 版中保持一致。

### 5.4 存储落点冻结
`concat_writer` 的实现重点是地址正确和轻量路由：

- 写指针与 channel offset 计算放在轻逻辑里
- 小型写入队列/FIFO 固定放 BRAM
- 大型 concat 目标 tensor 仍落到 `FMBUF`
- 不允许 concat 结果先暂存 DDR 再回读

## 6. 模块时序要求

### 6.1 avgpool
允许内部使用 line buffer，目标为 `II=1` 输出像素流。

### 6.2 activation / add / concat
这些路径必须视为轻逻辑，目标都是：

- `II=1`
- 不引入额外 DDR 往返
- 尽可能与卷积/post process dataflow 重叠

### 6.3 频率目标
这些非卷积路径的首版频率目标与全系统一致：

- `v1` 验收目标：`100 MHz`
- `v2` 优化目标：`150 MHz`
- `200 MHz` 仅在存储和布线优化充分后再评估

因此首版实现中应优先采用更保守的地址和控制逻辑，避免为了冲击 `200 MHz` 过早引入复杂并行复制或激进手工打包。

## 7. 当前网络的实施建议
为了让 HLS 先收敛，建议按以下顺序实现：

1. `concat_writer`
2. `eltwise_add_path`
3. `avgpool_unit`
4. `relu`

因为：

- concat 地址正确性会直接影响所有 stage
- add 正确性影响 level2/level3 残差
- avgpool 只在输入 3 通道路径上，规模较小，适合作为独立验证模块

## 8. 验证要求

1. B1 concat 与 B2 concat 不覆盖
2. 两次级联 avgpool 后尺寸与 FP32 工程一致
3. `ACT_RELU` 结果与 `alpha_zero` 的 FP32 路径一致
4. level2_block0 residual add 数值正确
5. level3_block0 residual add 数值正确
6. `POOL/ADD/CONCAT` 内部 scratch 不要求 bit 导出，但输入/输出 tensor 必须与 golden sample 对齐
