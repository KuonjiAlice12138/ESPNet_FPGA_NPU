# sa_core Spec

## 1. 对应关系

- 逻辑模块：`systolic_array_core`
- 当前实现文件：`src/sa_core.cpp`

## 2. 来源整理
本文件重组自：

- `02_Systolic_Array_Engine.md`

## 3. 模块职责
`systolic_array_core` 负责所有 `1x1` 与 `3x3` 卷积的：

- `INT8 x INT8 -> INT32` 乘加累加

不负责：

- DDR 读写
- avgpool
- concat
- affine/BR
- activation

## 4. 冻结规格

- `TM = 32`
- `TK = 32`
- 输入：`int8`
- 权重：`int8`
- 累加：`int32`
- 首版目标频率：`100 MHz`

首版明确不使用 DSP 双打包。

## 5. 支持模式

- kernel：`1` 或 `3`
- stride：`1 / 2`
- dilation：`1 / 2 / 4 / 8 / 16`
- padding：`same`
- `Cin tail`：内部零填充
- `Cout tail`：lane mask

## 6. 原型冻结

```cpp
void systolic_array_core(
    hls::stream<act_vec_t>& act_stream,
    hls::stream<weight_vec_t>& weight_stream,
    hls::stream<i32_t>& psum_stream,
    const conv_cfg_t& cfg);
```

## 7. 计算映射

- 行维 `TM=32`：输出通道 tile
- 列维 `TK=32`：reduction 维 tile

固定循环顺序：

```cpp
for (oc_tile)
  for (oh)
    for (ow)
      for (k_tile)
```

## 8. 权重格式
片上与 DDR 中权重统一顺序：

`[layer][oc_tile][k_tile][tm=32][tk=32]`

规则：

- 超出真实 `Cout` 或 `K_flat` 的部分补零

## 9. 激活流假设
输入激活流由 `window_generator` 保证：

- 固定 32-lane
- `1x1` 直读
- `3x3` 按冻结 flatten 顺序展开

## 10. HLS 关键约束

```cpp
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=act_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=psum_reg complete dim=1
```

未做到 `II=1` 视为实现不达标。

## 11. 尾 tile 规则

### 11.1 `Cin` 非 32 对齐

- 最后一个 `k_tile` 超出真实范围的 lane 注零

### 11.2 `Cout` 非 32 对齐

- 通过 `valid_tm` 屏蔽写回无效 lane

## 12. 验收重点

1. `1x1 stride1 Cin=256 Cout=2`
2. `3x3 stride2 Cin=3 Cout=16`
3. `3x3 dilation=16`
4. `Cin/Cout` 非 32 对齐
5. 极值输入和全零输入
