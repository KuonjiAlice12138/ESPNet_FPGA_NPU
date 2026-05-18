# avgpool_unit Spec

## 1. 对应关系

- 逻辑模块：`avgpool_unit`
- 当前实现文件：`src/avgpool_unit.cpp`

## 2. 来源整理
本文件重组自：

- `05_Hardware_Pooling_and_Activation.md` 的 avgpool 内容
- `03_Data_Formatter_and_Quantization.md` 的 pool scale 规则

## 3. 模块职责
`avgpool_unit` 负责当前网络中全部硬件化的平均池化路径：

1. `input -> pool1`
2. `input -> temp_pool`
3. `temp_pool -> pool2`

## 4. 冻结支持范围

- kernel：`3x3`
- stride：`2`
- padding：`same`
- channels：`3`

这不是通用 pool IP，而是为当前 encoder 定制的精简模块。

## 5. 计算路径冻结

### 5.1 先整数平均

`sum -> round(sum/9) -> clamp`

实现中建议用常数乘法近似 `1/9`，避免除法器。

### 5.2 再决定是否 requant

- `same_scale = 1`：直接继承输入 scale
- `same_scale = 0`：末端做一次 requant 到目标 scale

## 6. 边界处理

- 采用 zero padding
- 与卷积 `same padding` 一致

## 7. 原型冻结

```cpp
void avgpool_unit(
    const tensor_desc_t& src,
    const tensor_desc_t& dst,
    const pool_qparam_t& qparam,
    ap_int<8>* fmbuf_base);
```

## 8. 当前网络中允许的 3 个 pool 语义

1. `POOL_B1`
   - `src = T_INPUT`
   - `dst = T_POOL1`
   - `same_scale = 0`
   - `dst_scale = scale(T_B1_CAT)`
2. `POOL_B2_TMP`
   - `src = T_INPUT`
   - `dst = T_POOL_TMP`
   - `same_scale = 1`
3. `POOL_B2_OUT`
   - `src = T_POOL_TMP`
   - `dst = T_POOL2`
   - `same_scale = 0`
   - `dst_scale = scale(T_B2_CAT)`

## 9. 存储约束

- 行缓冲和 `3x3` window buffer 走 BRAM
- 输出逻辑落点由 `tensor_desc_t` 决定
- 不允许通过 DDR 中转

## 10. 验收重点

1. 两次级联 pool 的尺寸与 FP32 工程一致
2. `same_scale=0/1` 两条路径正确
3. B1 / B2 两类输出 scale 正确
