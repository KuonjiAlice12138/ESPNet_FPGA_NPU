# win_gen Spec

## 1. 对应关系

- 逻辑模块：`window_generator`
- 当前实现文件：`src/win_gen.cpp`

## 2. 来源整理
本文件重组自：

- `04_On_Chip_BRAM_Controller.md` 的窗口局部缓存定义
- `02_Systolic_Array_Engine.md` 的激活向量组织格式

## 3. 模块职责
`window_generator` 负责把逻辑 tensor 数据转换成阵列可消费的激活向量：

1. 对 `1x1` 卷积做直接打包
2. 对 `3x3 / dilated 3x3` 生成滑窗
3. 对 `Cin` 非 32 对齐情况做 tile 尾部补零
4. 输出固定长度 `32-lane` 激活向量流

## 4. 不负责的事情

- 不负责权重读取
- 不负责 MAC 累加
- 不负责后处理

## 5. 支持模式冻结

- `kernel = 1 or 3`
- `stride = 1 or 2`
- `dilation = 1 / 2 / 4 / 8 / 16`
- `padding = same`
- 每次服务一个 `Cin tile = 32`

## 6. 局部缓存冻结

### 6.1 对 `3x3`

- 3 行 `line buffer`
- 1 个 `3 x 3 x 32` `window buffer`

### 6.2 对 `1x1`

- 不走 line buffer
- 只做直读和打包

## 7. 激活展平顺序冻结
对 `3x3 / dilated 3x3`，flatten 顺序固定为：

`((kh * kernel) + kw) * Cin + cin`

这必须与离线导出脚本严格一致。

## 8. 零填充规则

### 8.1 边界

- 按 zero padding 处理

### 8.2 `Cin` 尾 tile

- 超出真实 `K_flat` 的 lane 注入零
- 不允许先读越界值后再 mask

## 9. 输出给阵列的格式

- 输出类型：长度 `32` 的 `INT8` 向量
- 对 `1x1`：按 `Cin` 连续读取
- 对 `3x3`：按 `(kh, kw, cin)` 顺序展开

## 10. 与 memory/sa_core 的边界

- 输入由 `on_chip_memory` 提供
- 输出交给 `systolic_array_core`
- 本模块不决定 `oc_tile` 映射，只负责 `k_tile` 方向激活流

## 11. 验收重点

1. `1x1` 打包正确
2. `3x3 dilation=16` 地址正确
3. stride=2 时窗口推进正确
4. `Cin` 非 32 对齐补零正确
