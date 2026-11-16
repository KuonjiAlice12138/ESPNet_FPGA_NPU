ESP INT8 board app: native H512W512 grayscale binary2, adapted QAT10 weights.

Platform: platform_h512_baseline, standalone_psu_cortexa53_0, Cortex-A53 0.
App tag: INT8-BOARD-20261116-P7-H512W512-GRAY-QAT10-B2.
Manual rebuild required after future App source changes. The current ELF
was manually rebuilt on 2026-11-16; its QAT10 tag was confirmed on board.
The platform, App component, app.yaml, BSP and boot recipe remain native512.
Only weights/qparams changed: all tensor/Conv/window/consumer/fixed/BLOCK5
descriptors and the EXEC plan match the implemented hardware's prior model.
No IP repackaging, platform rebuild or bitstream change is needed for QAT10.

One launch runs binary2 once, with all 75 UOPs / 16 EXEC entries of PARAM v4.
Dual-model, EXEC-prefix and validation-set modes are disabled. All 13 stage
bins and the three Conv WIN/SA/POST state counters remain enabled and required.
Input is NHWC signed INT8 512x512x1, with no FPGA input sampler/resize.
Logits are 64x64x2; exact 8x bilinear logits interpolation then argmax produces
a 512x512 uint8 mask. DDR input/output buffers are each 262144 bytes and
64-byte aligned. PARAM capacity is 262144 bytes.

SD files (8.3 names):
P512F.BIN  QAT10 PARAM, 143040 bytes
I512F.BIN  input paired with this export, 262144 bytes
O512F.BIN  fresh board output, 262144 bytes; do not copy golden to this name
BOOT.BIN   not required for JTAG; SD boot requires newly packaged firmware

Source artifact: hw_artifacts/binary2_int8_h512w512_gray_finetuned_v4
Model export: D:/ESPNet/resolution_h512w512_gray/quantized_artifacts_native512_finetuned
Hardware profile: p7_h512w512_gray_c1_v1. The selected checkpoint is the best
of the completed 10-epoch QAT phase, following 40 epochs of native512 FP32
fine-tuning. The stopped extra phase is not used. Labels are vehicles=0,
valid background=1, GT255 ignored. The App validates geometry and complete
EXEC-plan length before INIT and rejects timeout/ERROR/invalid output labels.

Full val500 QAT/PARAM replay: PA=98.2372%, mIoU=86.9902%, vehicle IoU=75.8465%.
All 4096000 INT8 logits match between QAT and replay. Expected first-sample
golden histogram: c0=19021, c1=243123, invalid=0. Histogram equality is not
sufficient: compare O512F.BIN byte-for-byte with golden_fullres_mask.bin.
CSim was not run for this QAT checkpoint. The 2026-11-16 board mask now
matches the integer golden exactly: 0/262144 pixels, invalid=0, ERROR=0.
First-sample board PA=98.9952%, mIoU=93.6241%, vehicle IoU=88.3357% on the
matching512 center-ROI GT. Full val500 board scoring remains pending.
The obsolete H240-direct-transfer native512 hardware artifact was removed;
its historical scores remain in workplan_1110.md, not this release.

ARM timing uses CNTPCT_EL0 / runtime CNTFRQ_EL0, BSP reference 33333000 Hz.
PL uses nominal 100 MHz (10 ns). SD I/O, MODE_INIT, cache maintenance and
printing remain outside MODE_RUN. WIN/SA/POST overlap and must not be summed.
POST staging/commit bins include psum waits, requantization and packing.
QAT10 board timing is 22228515 PL cycles=222.28515 ms; ARM timing is
7409496 ticks/33333000 Hz=222.28710 ms. Only one cycle differs from the old
direct-transfer model; model adaptation improves accuracy, not throughput.

Reused implementation: routing errors=0, WNS=+0.439 ns, WHS=+0.010 ns at 100MHz;
logic/total LUT=205438/223151, FF=172818, BRAM tile=228.5, DSP=1289, URAM=108.
NPU control=0xA0000000; RTL counter=0xA0010000. JTAG launch uses the platform
FSBL, the hash-identical cached bitstream, and ESP_INT8_app/build/ESP_INT8_app.elf.
The optional PSU-init path also points to this platform's psu_init.tcl.
Create a new Run/Debug configuration if the existing one is removed.

Verified data SHA256:
P512F.BIN 7c2175418f1f34e53bb3f12f57f8e182df204d27b2df0a3e136ac488baf45d4a
I512F.BIN b3355dbbd88d11bf34bc984d5591b83a04951a1cf93b7a674c2e908d37bcfe93
Golden    7bf97daf99a91f6dc41e148ac40c73400a93c7559083e20fcef8fb9ad120cf8c

JTAG preparation: manually rebuild, select ESP_INT8_app_app_hw_1, keep FSBL
initialization selected, and verify the new QAT10 tag at startup.
Bitstream: platform_h512_baseline/hw/sdt/ESP_INT8_hwplat.bit
FSBL: platform_h512_baseline/export/platform_h512_baseline/sw/boot/fsbl.elf
PS init: platform_h512_baseline/hw/sdt/psu_init.tcl

For SD boot only, package after the NEW ELF is built, from D:/ESP_INT8:
& D:/Xilinx/2025.1/Vivado/bin/bootgen.bat -arch zynqmp -image ESP_INT8_app/src/boot_h512_native.bif -o ESP_INT8_app/BOOT.BIN -w on
Then copy that new BOOT.BIN to E:/BOOT.BIN and verify its hash. No BOOT.BIN
was generated during this preparation; data files alone are not boot firmware.
Remove obsolete P512N.BIN/O512N.BIN and any stale O512F.BIN before testing.
Preserve unrelated H240/resize comparison data and validation folders B/C/D/Q.
