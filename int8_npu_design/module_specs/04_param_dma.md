# param_dma Spec

## 1. 对应关系

- 逻辑模块：`param_dma`
- 当前实现文件：`src/param_dma.cpp`

## 2. 来源整理
本文件重组自：

- `01_System_Architecture.md`
- `06_DMA_and_AXI_Interface.md`

## 3. 模块职责
`param_dma` 负责 `MODE_INIT` 阶段的静态参数预加载：

1. 从 `gmem_param` 读取 `param_blob.bin`
2. 解析 header 中各 section offset
3. 加载权重到 `WBUF`
4. 加载量化参数、descriptor 和 `uop` 表到 `QBUF` 或内部缓存

## 4. 不负责的事情

- 不负责运行期输入帧搬运
- 不负责中间 tensor 读写
- 不负责解释 `uop` 的执行顺序

## 5. 冻结输入

- `PARAM_BASE` 指向单一 `param_blob.bin`
- `MODE_INIT` 时只使用本输入

## 6. `param_blob` 结构冻结

### 6.1 头部

- 固定 `128 bytes`
- 包含 magic/version
- 包含所有 table count 和 section offset

### 6.2 固定 section 顺序

1. `header`
2. `tensor_desc_table`
3. `scale_desc_table`
4. `conv_desc_table`
5. `affine_desc_table`
6. `add_desc_table`
7. `pool_desc_table`
8. `uop_table`
9. `weight_data`
10. `conv_qparam_data`
11. `affine_qparam_data`
12. `add_qparam_data`
13. `pool_qparam_data`

### 6.3 对齐规则

1. 所有 section 起始地址 `64-byte` 对齐
2. 所有 offset 相对于 blob 起始地址

## 7. 片上落点冻结

- 全部 INT8 权重：`WBUF`
- conv/add/pool qparam：`QBUF0`
- affine/scale desc/uop/tensor desc：`QBUF1`
- `tensor_desc.reserved0/reserved1` 不再视为无意义保留位，必须原样保留给 memory 模块解释为 `physical_c_stride/channel_offset`

首版固定为整模一次性预加载，不做层间权重 ping-pong。

## 8. 对当前模型的规模约束

- 全模型 INT8 权重约 `108 KB`
- `WBUF` 固定预算 `128 KB`

因此当前模型允许一次性整模加载。

## 9. 初始化错误检测
本模块必须能参与以下错误发现：

- bad blob magic/version
- count/offset 非法
- section 越界
- 片上 bank overflow

## 10. 验收重点

1. `MODE_INIT` 后 `WBUF/QBUF` 内容与离线导出一致
2. header/offset 解析正确
3. 不需要运行期重新读权重
4. 可被 `instruction_fetch_decode` 和其他模块稳定复用
5. 当前 `param_blob.bin` 解析结果应满足 `uops=75, scales=55`，并保留 B2/B3 view tensor descriptor
