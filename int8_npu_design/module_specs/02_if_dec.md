# if_dec Spec

## 1. 对应关系

- 逻辑模块：`instruction_fetch_decode`
- 当前实现文件：`src/if_dec.cpp`

## 2. 来源整理
本文件重组自：

- `06_DMA_and_AXI_Interface.md`
- `07_ESPNet_Encoder_Uop_Schedule.md`

## 3. 模块职责
`instruction_fetch_decode` 负责把 `param_blob` 中冻结的控制信息转换成运行期可用的内部状态：

1. 解析 `param_blob_header_t`
2. 解析 `tensor_desc_table`
3. 解析 `scale_desc_table`
4. 解析 opcode-specific 参数 descriptor
5. 顺序读取 packed `uop_t`
6. 向其他模块提供当前 `uop`、`tensor_desc`、`param_id`、`scale` 等解释结果

## 4. 不负责的事情

- 不负责 DDR 数据搬运本身
- 不负责权重加载
- 不负责特征图读写
- 不负责执行卷积/池化/后处理运算

## 5. 输入来源

1. `MODE`
2. `UOP_COUNT`
3. `gmem_param` 指向的 `param_blob.bin`

## 6. 需要理解的冻结格式

### 6.1 `uop_t`

- 固定 `32 bytes`
- 小端打包
- `sizeof(uop_t) == 32`

### 6.2 `tensor_desc_t`

- 固定 `16 bytes`
- `bank_id / elem_bytes / base_offset / h / w / c`
- `reserved0` 固定解释为 `physical_c_stride`
- `reserved1` 固定解释为 `channel_offset`
- compact tensor 使用 `physical_c_stride = c, channel_offset = 0`
- `T_L2B0_*` 和 `T_L3B0_*` 必须按 view tensor 解释，不能按独立 FMEM bank 解释

### 6.3 `scale_desc_t`

- `mult + shift`
- 用于 requant 到目标 tensor scale

### 6.4 opcode-specific descriptor

- `conv_param_desc_t`
- `affine_param_desc_t`
- `add_param_desc_t`
- `pool_param_desc_t`

## 7. 共享冻结命名空间
本模块必须严格遵守：

1. `opcode_t`
2. 全局 `tensor id`
3. 局部 `scratch id`
4. opcode-specific `param_id`
5. `flags` 位定义

这些内容全部以 `06_DMA_and_AXI_Interface.md` 与 `07_ESPNet_Encoder_Uop_Schedule.md` 为准。

## 8. 对当前网络的特殊约束

1. 当前 `ESPNet_Encoder v1` 的 `uop.qparam_id` 固定为 `0`
2. `UOP_COUNT` 必须等于 `param_blob` 头中的 `uop_count`
3. 所有 `ADD` 的 `requant_bypass` 对当前网络固定为 `1`
4. `concat` 和 `pool` 的目标 scale 关系要与共享 schedule 一致
5. `uop[5]` 与 `uop[6]` 固定为提前执行的两级 `POOL`，分别生成 `T_POOL_TMP` 和 `T_POOL2`
6. `uop[36]` 和 `uop[70]` 是 view concat 检查，源 tensor 已经位于目标容器对应 channel slice

## 9. 错误检测职责
本模块至少应能触发以下错误来源：

- bad blob magic/version
- uop decode error
- tensor desc out of range
- param desc out of range
- unsupported opcode

## 10. 输出给下游的信息

1. 当前 `uop`
2. 当前 stage / current_uop_id
3. 源/目标 tensor descriptor
4. 当前 opcode 对应的参数表偏移
5. 对 `ADD / POOL / CONV / AFFINE` 的 mode 解释结果

## 11. 验收重点

1. `MODE_INIT` 能完整解析 `param_blob` 各 section
2. `MODE_RUN` 能顺序吐出正确的 `uop`
3. 共享命名空间与离线导出完全一致
4. 出错时能把 `current_uop_id` 和 `error_code` 报出去
5. 对当前导出的 `param_blob.bin`，关键 uop 序号和 tensor view descriptor 与 `07_ESPNet_Encoder_Uop_Schedule.md` 完全一致
