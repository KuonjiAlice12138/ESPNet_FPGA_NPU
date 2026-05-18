# 2D Systolic Array Engine (脉动阵列引擎)

## 1. 模块职责
该模块负责本项目所有 `1x1` 与 `3x3` 卷积的 INT8 x INT8 -> INT32 累加，不负责：

- DDR 读写
- avgpool
- concat
- standalone affine/BR
- activation

这些功能统一交给其他模块，阵列核心只做“乘加 + 累加完成后吐出 psum”。

## 2. 冻结后的阵列规格

### 2.1 基本参数

- `TM = 32`：每拍并行输出通道数
- `TK = 32`：每拍并行 reduction 宽度
- 输入位宽：`int8`
- 权重位宽：`int8`
- 累加位宽：`int32`
- **首版时钟目标**：`100 MHz`
- **优化目标**：`150 MHz`
- **后续冲刺目标**：`200 MHz`

### 2.2 v1 是否强依赖 DSP 双打包
`v1` 版本**明确不使用** DSP48E2 的双 INT8 packing。

- `v1`：固定为 `1 DSP ~= 1 MAC`
- `v2` 及以后：功能稳定后才允许单独评估双打包

原因：

- 即使按保守实现估算，`32 x 32 x 100 MHz = 102.4 GMAC/s`
- 若中间特征图常驻片上，`50 ms` 目标依然存在可达性
- 不要在功能未收敛前把设计绑死在高风险 DSP packing 上

## 3. 支持的卷积模式

| Field | Supported |
|---|---|
| Kernel | `1` or `3` |
| Stride | `1` or `2` |
| Dilation | `1 / 2 / 4 / 8 / 16` |
| Padding | `same` |
| Cin tail | zero pad to `TK=32` inside tile only |
| Cout tail | lane mask inside `TM=32` tile |

### 3.1 输出尺寸规则

- `1x1`：`H_out = ceil(H_in / stride)`，`W_out = ceil(W_in / stride)`
- `3x3 same`：padding 固定为 `dilation`

### 3.2 累加位宽安全性
本网络最极端情况约为：

- `Cin <= 256`
- `K <= 3x3`
- `|act|, |weight| <= 127`

上界约：

`256 x 9 x 127 x 127 = 37,158,144`

仍在 `int32` 正范围内，因此 `psum` 固定为 `int32`。

## 4. 子模块划分
建议在 `sa_core.cpp` 内部再拆 4 个函数：

1. `load_weight_tile`
2. `broadcast_act_vec`
3. `mac_array_32x32`
4. `drain_psum_tile`

### 4.1 推荐 HLS 原型

```cpp
typedef ap_int<8> i8_t;
typedef ap_int<32> i32_t;
typedef ap_uint<32 * 8> act_vec_t;
typedef ap_uint<32 * 8> weight_vec_t;

struct conv_cfg_t {
    ap_uint<16> in_h;
    ap_uint<16> in_w;
    ap_uint<16> in_c;
    ap_uint<16> out_c;
    ap_uint<2>  kernel;
    ap_uint<2>  stride;
    ap_uint<5>  dilation;
    ap_uint<1>  bias_en;
};

void systolic_array_core(
    hls::stream<act_vec_t>& act_stream,
    hls::stream<weight_vec_t>& weight_stream,
    hls::stream<i32_t>& psum_stream,
    const conv_cfg_t& cfg);
```

## 5. 计算映射方式

### 5.1 逻辑映射

- 行维 `TM=32`：映射输出通道 tile
- 列维 `TK=32`：映射卷积 flatten 后的 reduction 维度 tile

即每个空间位置 `(oh, ow)` 的一次卷积，等价为：

`dot(activation[K_flat], weight[Cout][K_flat])`

其中：

- `K_flat = Cin * kernel * kernel`
- 阵列每拍处理 `32` 个 `K_flat` 元素，并同时更新 `32` 个输出通道

### 5.2 顶层循环顺序
冻结为：

```cpp
for (oc_tile = 0; oc_tile < ceil(Cout / 32); ++oc_tile)
  for (oh = 0; oh < Hout; ++oh)
    for (ow = 0; ow < Wout; ++ow)
      for (k_tile = 0; k_tile < ceil(K_flat / 32); ++k_tile)
```

原因：

- 便于 weight-stationary
- 易于处理 `Cin=3/19/131` 之类非 32 对齐通道
- 易于把 `1x1` 与 `3x3 dilated` 统一成同一种 flatten reduction

## 6. 权重组织格式
离线导出后的 INT8 权重，在 DDR 和片上 WBUF 内统一采用如下顺序：

`[layer][oc_tile][k_tile][tm=32][tk=32]`

规则：

- `tm` 或 `tk` 超出真实 `Cout/Cin*K*K` 的部分补零
- `1x1` 卷积视为 `K_flat = Cin`
- `3x3 dilated` 的 dilation 不编码进权重，只影响 act 读取地址

这样做可以让 `load_weight_tile` 只做突发搬运，不做格式变换。

## 7. 激活向量组织格式
阵列输入激活流统一是长度为 `32` 的 signed INT8 向量。

### 7.1 `1x1`
按 `Cin` 连续读取，尾部补零。

### 7.2 `3x3 / dilated 3x3`
由 `window_generator` 先按 `(kh, kw, cin)` 展平，再输出 32-lane 向量。

flatten 顺序冻结为：

`((kh * kernel) + kw) * Cin + cin`

对同一版本导出脚本与硬件必须严格一致。

## 8. HLS 关键 pragma
以下 pragma 属于强约束，而不是“建议”：

```cpp
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=act_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum_reg complete dim=1
```

如果 `mac_array_32x32` 未能做到 `II=1`，视为实现不达标。

## 9. 末尾通道与零填充规则

### 9.1 Cin 非 32 对齐

- 在 `k_tile` 最后一拍，对超出真实 `K_flat` 的 lane 注入零
- 不允许从 FMBUF 读取越界值后再做 mask

### 9.2 Cout 非 32 对齐

- 每个 `oc_tile` 携带 `valid_tm`
- 对 `tm >= valid_tm` 的 lane：
  - 权重装零
  - 最终写回时屏蔽

## 10. 延迟估算公式
单层卷积的主计算拍数按下式估算：

`cycles_conv ~= Hout x Wout x ceil(Cout / 32) x ceil(Cin x K x K / 32)`

总延迟还需额外加：

- 阵列填充/排空延迟
- window generator 预热
- post process 写回延迟

但这些都必须与主计算流重叠，不能成为独立串行大开销。

## 11. 实现建议

### 11.1 首版资源假设

- `v1` 固定按 `1 DSP ~= 1 MAC` 实现阵列资源
- 即 `32 x 32` 阵列对应约 `1024` 个 DSP 量级
- 该规模在 ZU15EG 总 DSP 资源上仍可接受，但会给布线和时序带来明显压力

### 11.2 频率分阶段验收

建议按以下里程碑推进，而不是一上来就强压 `200 MHz`：

1. `100 MHz`：功能正确、bit-exact、板级可跑
2. `150 MHz`：资源/时序优化版
3. `200 MHz`：仅在存储和布线策略进一步优化后再评估

## 12. 验证要求
阵列模块 HLS 单测至少覆盖：

1. `1x1, stride1, Cin=256, Cout=2`
2. `3x3, stride2, Cin=3, Cout=16`
3. `3x3, dilation=16, Cin=25, Cout=25`
4. `Cin` 非 32 对齐
5. `Cout` 非 32 对齐
6. 全零输入
7. 最大正负值输入
