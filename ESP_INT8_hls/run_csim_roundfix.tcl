# CSim script for round_shift fix verification
# Run: vitis_hls -f run_csim_roundfix.tcl
# Or: D:\Xilinx\2025.1\Vitis\bin\unwrapped\win64.o\vitis_hls.exe -f run_csim_roundfix.tcl

open_project -reset hls_work_p7_roundfix

# Same source files as hls_config_top_golden.cfg
add_files src/int8_core.cpp src/scratch_mgr.cpp src/conv_store.cpp \
         src/frame_dma.cpp src/upsample_unit.cpp src/param_dma.cpp \
         src/memory.cpp src/win_gen.cpp src/sa_core.cpp src/avgpool_unit.cpp
add_files include/npu_types.hpp include/npu_config.hpp \
         include/npu_schedule.hpp include/npu_uop.hpp include/npu_q.hpp

# CSIM dump flags: dump all key checkpoints
set csim_flags "-DESP_INT8_CSIM_DUMP_DEBUG_SET -DESP_INT8_CSIM_DUMP_L2_SET \
                -DESP_INT8_CSIM_DUMP_U40_PRESTORE -DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE"

add_files -cflags $csim_flags

# Testbench (modified to not exit on mask mismatch)
add_files -tb -cflags $csim_flags tb/top_golden_sample_tb.cpp

set_top espnet_encoder_int8_core
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10

csim_design

exit
