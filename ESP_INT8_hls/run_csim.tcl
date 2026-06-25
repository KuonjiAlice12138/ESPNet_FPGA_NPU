open_project -reset hls_work_p7_roundfix
add_files -cflags "-DESP_INT8_CSIM_DUMP_DEBUG_SET -DESP_INT8_CSIM_DUMP_L2_SET -DESP_INT8_CSIM_DUMP_U40_PRESTORE -DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" src/int8_core.cpp src/scratch_mgr.cpp src/conv_store.cpp src/frame_dma.cpp src/upsample_unit.cpp src/param_dma.cpp src/memory.cpp src/win_gen.cpp src/sa_core.cpp src/avgpool_unit.cpp
add_files -tb -cflags "-DESP_INT8_CSIM_DUMP_DEBUG_SET -DESP_INT8_CSIM_DUMP_L2_SET -DESP_INT8_CSIM_DUMP_U40_PRESTORE -DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" tb/top_golden_sample_tb.cpp
set_top espnet_encoder_int8_core
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
exit