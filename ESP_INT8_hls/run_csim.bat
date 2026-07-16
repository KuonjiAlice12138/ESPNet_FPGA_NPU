@echo off
set MSYS_NO_PATHCONV=1
set MSYS2_ARG_CONV_EXCL=*
set RDI_DATADIR=D:\Xilinx\2025.1\Vivado\data
set XILINX_VITIS=D:\Xilinx\2025.1\Vitis
set PATH=D:\Xilinx\2025.1\Vitis\bin;D:\Xilinx\2025.1\Vitis\bin\unwrapped\win64.o;D:\Xilinx\2025.1\Vivado\bin;D:\Xilinx\2025.1\Vivado\bin\unwrapped\win64.o;D:\Xilinx\2025.1\Vitis\tps\win64\msys64\usr\bin;D:\Xilinx\2025.1\Vitis\gnuwin\bin;%PATH%
cd /d D:\ESP_INT8\ESP_INT8_hls
D:\Xilinx\2025.1\Vitis\bin\unwrapped\win64.o\vitis-run.exe --mode hls --config hls_config.cfg --work_dir hls_work_p7_satb --csim