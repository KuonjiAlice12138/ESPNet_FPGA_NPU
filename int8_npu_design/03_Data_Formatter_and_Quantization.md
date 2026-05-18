# Data Formatter and Quantization (数据重组与量化)

## 1. 模块职责
本模块负责把阵列输出的 `INT32 psum` 变成层间继续流转的 `INT8 activation`，同时覆盖当前网络真正需要的 3 类后处理：

1. `conv psum + bias -> requant -> relu`
2. `int8 add -> requant/clip -> relu`
3. `standalone affine(BR) -> requant -> relu`

注意：`concat` 不是算术运算，不在这里做；`concat` 只做地址写入，见 `05_Hardware_Pooling_and_Activation.md`。

## 1.2 与首版实现策略的关系
本模块是 `100 MHz` 首版实现中非常关键的轻量计算路径，因此冻结以下实现原则：

- 量化参数表固定驻留 `QBUF`，使用 BRAM
- `post_process_unit` 必须保持浅流水，不能成为比卷积阵列更难收敛的路径
- 首版优先保 bit-exact 和 `II=1`，而不是为了压缩几个乘法器去引入复杂共享控制

## 1.1 与 `Model_flattened_quantized.py` 的一致性约束
根据 `D:\ESPNet\Model_flattened_quantized.py`：

- 输入端只有一个 `QuantStub`
- 输出端只有一个 `DeQuantStub`
- 中间的 `add` 使用 `nn.quantized.FloatFunctional()`
- 中间的 `cat` 直接发生在量化域

因此本项目硬件必须遵守：

1. **中间层不做 dequant 到 float**
2. `add/cat` 前需要的是 **requant / scale 对齐**，不是反量化
3. 唯一允许的真正 dequant 发生在最终 `classifier` 输出之后；如果软件后处理不需要 float，这一步甚至可以留在 PS 端再做

## 2. 量化规范冻结

### 2.1 数值格式

- Activation：`int8`
- Weight：`int8`
- Bias：`int32`
- Accumulator：`int32`
- Multiplier：`int32`
- Shift：`uint8`

### 2.2 Zero-point 规范
`v1` 硬件冻结为**对称量化**：

- `act_zp = 0`
- `weight_zp = 0`
- `out_zp = 0`

不支持非零 zero-point。这样可以显著降低 datapath 复杂度，并避免在 HLS 中为每次 MAC 增加补偿项。

### 2.3 Scale 粒度

- Activation scale：**per tensor**
- Weight scale：**per output channel**
- Bias：按 `bias_fp32 / (s_act_in * s_w[oc])` 离线量化成 `int32`
- Requant 参数：**per output channel**

### 2.4 Add/Concat 的约束
为了避免通道内 scale 混乱，冻结如下规则：

1. `add` 两个输入必须共享同一 activation scale
2. `concat` 的所有分支在写入目标 tensor 前必须先被 requant 到同一目标 activation scale
3. 下游卷积只看到“一个 tensor 一个 scale”的 activation

导出脚本必须为每个 `add/concat` 节点生成匹配的目标 scale，不允许把这个问题留给硬件临时处理。

### 2.6 Pool 的 scale 规则
`POOL` 也必须遵守单 tensor 单 scale 规则。首版冻结为：

1. `POOL` 先在整数域完成 `3x3` 平均
2. 若 `pool_qparam.same_scale = 1`，则输出继承输入 tensor scale
3. 若 `pool_qparam.same_scale = 0`，则 `POOL` 末端必须做一次 requant，写入目标 tensor scale

对当前 Encoder，固定采用：

1. B1 分支 `POOL(T_INPUT -> T_POOL1)`：`same_scale = 0`，输出 scale = `scale(T_B1_CAT)`
2. B2 临时池化 `POOL(T_INPUT -> T_POOL_TMP)`：`same_scale = 1`，输出 scale = `scale(T_INPUT)`
3. B2 最终池化 `POOL(T_POOL_TMP -> T_POOL2)`：`same_scale = 0`，输出 scale = `scale(T_B2_CAT)`

### 2.5 对当前量化模型的具体解释
对应到 `Model_flattened_quantized.py`，应按下面方式理解：

- `self.qadd.add(a, b)`：
  - 硬件中执行 INT8 域 add
  - 前提是 `a` 和 `b` 已经被 requant 到同一 scale
- `torch.cat([x1, x2, ...], dim=1)`：
  - 硬件中不做算术
  - 前提是 `x1/x2/...` 在写入目标 tensor 前已被 requant 到该目标 tensor 的统一 scale

如果导出脚本不能保证这一点，当前文档定义的单-scale-per-tensor 硬件就不成立。

## 3. 支持的 3 种后处理模式

### 3.1 CONV_POST
输入：

- `psum_int32`
- `bias_int32[oc]`
- `mult[oc]`
- `shift[oc]`
- `act_type`

计算：

```cpp
acc = psum + bias;
scaled = (int64_t)acc * mult;
rounded = round_shift(scaled, shift);
out = clamp_int8(rounded);
out = activation(out);
```

### 3.2 ADD_POST
输入：

- `src_a_int8`
- `src_b_int8`
- `mult/shift` 字段固定保留
- `act_type`

计算：

```cpp
sum16 = (int16_t)src_a + (int16_t)src_b;
if (requant_bypass) {
    out = clamp_int8(sum16);
} else {
    scaled = (int32_t)sum16 * mult;
    rounded = round_shift(scaled, shift);
    out = clamp_int8(rounded);
}
out = activation(out);
```

对于当前 Encoder，导出脚本必须保证 add 输入 scale 已对齐，硬件固定走 `requant_bypass=1` 路径。
对当前 Encoder，14 个 `ADD` uop **全部冻结为 `requant_bypass = 1`**。也就是说：

1. 所有 add 输入必须在进入 add 前已经对齐到同一 scale
2. `ADD` 本身只做 `int16/int32` 加法、饱和和可选 `relu`
3. 首版硬件不得把当前网络中的任一 add 改成运行时 requant add

### 3.3 AFFINE_POST
用于替代当前 FP32 工程里的 standalone BR：

`y = x * scale + shift`

在硬件中不保留 float `scale/shift`，而是离线转换成整数形式：

```cpp
tmp = (int32_t)x * affine_mul[oc] + affine_bias[oc];
rounded = round_shift(tmp, affine_shift[oc]);
out = clamp_int8(rounded);
out = activation(out);
```

这一路用于：

- B1 后处理
- Level2_0 concat 后处理
- Level2_Block0 residual 后处理
- B2 后处理
- Level3_0 concat 后处理
- Level3_Block0 residual 后处理
- B3 后处理

## 4. 舍入与饱和规则

### 4.1 右移舍入
统一使用“round to nearest, ties away from zero”：

```cpp
int32_t round_shift(int64_t x, uint8_t s) {
    if (s == 0) return (int32_t)x;
    int64_t bias = (x >= 0) ? (1LL << (s - 1)) : -(1LL << (s - 1));
    return (int32_t)((x + bias) >> s);
}
```

### 4.2 饱和

```cpp
if (x > 127) x = 127;
if (x < -128) x = -128;
```

不允许依赖编译器隐式截断。

## 5. 激活函数模式
当前模型 `v1` 只要求：

- `ACT_NONE`
- `ACT_RELU`

`PReLU/LeakyReLU` 字段可以预留，但不纳入首版验收范围。当前 FP32 代码中 `alpha_zero` 已说明主流程只需标准 ReLU。

## 5.1 频率目标
本模块跟随系统首版目标：

- `v1` 验收频率：`100 MHz`
- 后续优化目标：`150 MHz`

因此这里的 datapath 设计必须明显偏保守：

- 使用每通道独立乘法器
- 使用 BRAM 参数表直读
- 不引入为了冲 `200 MHz` 才有意义的复杂重定时技巧

## 6. HLS 实现原型

```cpp
enum post_mode_t {
    POST_CONV = 0,
    POST_ADD  = 1,
    POST_AFFINE = 2
};

struct post_cfg_t {
    ap_uint<2> mode;
    ap_uint<2> act_type;
    ap_uint<1> requant_bypass;
    ap_uint<6> valid_tm;
};

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

## 7. 参数表格式
离线导出必须至少生成两类参数表。

### 7.1 `conv_qparam_t`

```cpp
struct conv_qparam_t {
    int32_t bias[32];
    int32_t mult[32];
    uint8_t shift[32];
};
```

每个 `oc_tile` 一份。

### 7.2 `affine_qparam_t`

```cpp
struct affine_qparam_t {
    int32_t mul[32];
    int32_t bias[32];
    uint8_t shift[32];
};
```

每个 `oc_tile` 一份。

### 7.3 `add_qparam_t`
当前网络虽然全部走 `requant_bypass = 1`，但 descriptor 仍保留统一格式，避免导出器和硬件分叉。

```cpp
struct add_qparam_t {
    int32_t mult;
    uint8_t shift;
    uint8_t act_type;
    uint8_t requant_bypass;
    uint8_t reserved0;
    uint32_t src_scale_id_a;
    uint32_t src_scale_id_b;
    uint32_t dst_scale_id;
    uint32_t reserved1;
};
```

### 7.4 `pool_qparam_t`

```cpp
struct pool_qparam_t {
    uint8_t kernel;
    uint8_t stride;
    uint8_t same_scale;
    uint8_t act_type;
    uint32_t src_scale_id;
    uint32_t dst_scale_id;
    int32_t mult;
    uint8_t shift;
    uint8_t reserved[11];
};
```

## 8. 与 FMBUF 的接口规则

### 8.1 写回顺序
所有输出统一按密集 NHWC 顺序写回：

`addr = base + ((h * W + w) * C + c)`

### 8.2 valid channel 规则
对 `C != 32` 整数倍的尾 tile：

- 只写回 `valid_tm` 个通道
- 其余 lane 丢弃

### 8.3 与 bank 规划的关系
本模块只关心逻辑 tensor descriptor，不直接关心底层是 URAM 还是 BRAM。

但结合当前 `04_On_Chip_BRAM_Controller.md`，应按如下原则理解：

- 大 tensor 的 post 结果写回 FMBUF（URAM 主体）
- 小 scratch / add 中间值允许落 BRAM scratch
- 不允许为了简化 post 路径而把中间结果回写 DDR

## 9. 验证向量要求
本模块独立仿真必须覆盖：

1. `bias=0` 和 `bias!=0`
2. 正数溢出、负数溢出
3. `shift=0`
4. `shift>0`
5. `relu` 开/关
6. `add bypass` 和 `add requant`
7. `affine` 路径
