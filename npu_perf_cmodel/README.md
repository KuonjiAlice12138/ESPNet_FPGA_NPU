# ESPNet INT8 NPU Performance C Model

这个目录提供一个面向当前 final spec 的性能 C 模型，用来在 HLS 实现前先估算：

1. `MODE_INIT` 一次性参数加载时延
2. `MODE_RUN` 单帧推理时延
3. 各 stage / 各 uop 的 cycle 占比
4. 在 `100 MHz` 下是否满足 `< 100 ms`

## 模型覆盖范围
模型按当前冻结文档建立：

- `TM = 32`, `TK = 32`
- `100 MHz`
- `256-bit` DDR `m_axi`
- 当前 `01~07` final spec
- 当前 `07_ESPNet_Encoder_Uop_Schedule.md` 的执行顺序

## 主要假设

### 卷积
- 基础公式：`Hout * Wout * ceil(Cout/32) * ceil(Cin*K*K/32)`
- 加入 line-buffer/window warmup
- 加入 1x1 / 3x3 的不同 issue efficiency
- 加入 stride / dilation 对效率的小幅惩罚

### Pool / Add / Affine / Concat
- 按 `32-lane` 向量化吞吐建模
- 保留固定 pipeline tail
- 不允许 DDR 中转

### DDR
- 总线宽度：`256-bit = 32 bytes/cycle`
- 对 `input load / output store / param init` 使用可配置效率

## 构建

```powershell
gcc -O2 -std=c11 D:\ESP_INT8\npu_perf_cmodel\npu_perf_model.c -o D:\ESP_INT8\npu_perf_cmodel\npu_perf_model.exe
```

## 运行

```powershell
D:\ESP_INT8\npu_perf_cmodel\npu_perf_model.exe
```

可选参数：

```powershell
D:\ESP_INT8\npu_perf_cmodel\npu_perf_model.exe --freq-mhz 100 --ddr-eff 0.70 --conv1-eff 0.93 --conv3-eff 0.88 --vec-eff 0.92 --pool-eff 0.90
```

## 输出
程序会打印：

1. 模型假设
2. 每个 uop 的 cycle
3. 每个 stage 的 cycle 和 ms
4. `INIT / RUN / INIT+RUN`
5. 是否满足 `< 100 ms`
