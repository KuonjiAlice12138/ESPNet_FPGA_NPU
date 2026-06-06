# ESP_INT8_hls Component Notes

当前目录是实际使用中的 `Vitis HLS` component 根目录。

## 目录约定

```text
ESP_INT8_hls/
├── include/
├── src/
├── tb/
├── hls_config.cfg
└── vitis-comp.json
```

## 当前逻辑模块与文件名映射

| 逻辑模块 | 当前文件名 |
|---|---|
| `espnet_encoder_int8_core` | `src/int8_core.cpp` |
| `instruction_fetch_decode` | `src/if_dec.cpp` |
| `frame_dma` | `src/frame_dma.cpp` |
| `param_dma` | `src/param_dma.cpp` |
| `on_chip_memory` | `src/memory.cpp` |
| `window_generator_row` | `src/win_gen.cpp` |
| `systolic_array_core_row` | `src/sa_core.cpp` |
| `conv_post_process / row_store` | `src/int8_core.cpp` |
| `avgpool_unit_checked / affine / add / store` | `src/avgpool_unit.cpp` |
| `concat_writer` | `src/concat_unit.cpp` |
| `fullres_upsample` | `src/upsample_unit.cpp` |

公共头文件：

- `include/npu_types.hpp`
- `include/npu_config.hpp`
- `include/npu_uop.hpp`
- `include/npu_q.hpp`

当前测试骨架：

- `tb/top_tb.cpp`
