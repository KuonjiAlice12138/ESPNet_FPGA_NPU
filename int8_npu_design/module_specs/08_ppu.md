# ppu Spec

## 1. 对应关系

- 逻辑模块：`post_process_unit`
- 当前实现文件：`src/ppu.cpp`

## 2. 来源整理
本文件重组自：

- `03_Data_Formatter_and_Quantization.md`
- `05_Hardware_Pooling_and_Activation.md` 中 activation 的共享约束

## 3. 模块职责
`post_process_unit` 负责把中间结果变成层间继续流转的 `INT8 activation`，覆盖：

1. `CONV_POST`
2. `ADD_POST`
3. `AFFINE_POST`

以及末端 `activation_unit` 语义。

## 4. 不负责的事情

- 不负责 concat 地址写入
- 不负责 avgpool 滑窗
- 不负责卷积 MAC

## 5. 量化冻结

- Activation：`int8`
- Weight：`int8`
- Bias：`int32`
- Accumulator：`int32`
- Zero-point：固定 `0`
- Activation scale：per tensor
- Weight scale：per output channel

## 6. 中间层共享约束

1. 中间层不允许 dequant 到 float
2. `add` 输入必须预先对齐到同一 scale
3. `concat` 目标 tensor 的所有分支必须先 requant 到同一 scale

## 7. 支持的 3 种路径

### 7.1 `CONV_POST`

`psum + bias -> requant -> clamp -> activation`

### 7.2 `ADD_POST`

当前网络固定要求：

- `requant_bypass = 1`
- add 输入 scale 已对齐

因此对当前 encoder，`ADD` 主路径只做：

- `int16/int32` 求和
- 饱和
- 可选 `relu`

### 7.3 `AFFINE_POST`

用于替代 standalone BR：

`y = x * scale + shift`

但实际实现冻结为整数形式：

`mul + bias + round_shift + clamp + activation`

## 8. 舍入与饱和规则

### 8.1 右移舍入

- `round to nearest, ties away from zero`

### 8.2 饱和

- `>127 -> 127`
- `<-128 -> -128`

## 9. 激活模式

- `ACT_NONE`
- `ACT_RELU`

## 10. 原型冻结

```cpp
void post_process_unit(
    hls::stream<ap_int<32> >& psum_stream,
    hls::stream<ap_int<8> >& add_a_stream,
    hls::stream<ap_int<8> >& add_b_stream,
    hls::stream<ap_int<8> >& out_stream,
    const post_cfg_t& cfg,
    const ap_int<32> bias[32],
    const ap_int<32> mult[32],
    const ap_uint<8> shift[32],
    const ap_int<32> affine_mul[32],
    const ap_int<32> affine_bias[32],
    const ap_uint<8> affine_shift[32]);
```

## 11. 参数表

- `conv_qparam_t`
- `affine_qparam_t`
- `add_qparam_t`
- `pool_qparam_t` 的共享定义仍需理解，但 pool 运算不在本模块实现

## 12. 与 FMBUF 的关系

- 写回按稠密 `NHWC`
- 对 `valid_tm` 尾 tile 只写有效通道
- 结果直接回写逻辑 tensor，不允许中间结果回 DDR

## 13. 验收重点

1. bias=0 / bias!=0
2. `shift=0 / >0`
3. overflow 正负饱和
4. `relu` 开关
5. `ADD bypass`
6. `AFFINE` 路径 bit-exact
