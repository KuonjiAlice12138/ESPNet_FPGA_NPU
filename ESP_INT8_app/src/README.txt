ESP INT8 board app: H256W512 Round 4R timing-repair dual-single profiling.

Platform: platform_r4_0922, standalone_psu_cortexa53_0.
App tag: INT8-BOARD-20260923-P7-H256W512-R4R-DUAL-PROF.
Build the app manually after these source updates. Download using this
platform's bitstream and FSBL through Vitis; SD contains model data only.

One boot runs binary2 once, then cityscapes20 once. Each model has its own
MODE_INIT and quantized input. Prefix and validation-set modes are disabled.
PARAM v4 ABI remains unchanged: 75 logical UOPs, 16 EXEC records including END,
U71/U72 separate. Input is 256x512x3 INT8 NHWC; output is a 256x512 uint8 mask.
Both PARAM files enable C3 KEEP1 row reuse with 48 packed words; C12 reuse
remains NONE. Round 4R retains the Round 4 incremental C3/C12 WinGen protocol,
while replacing dynamic low-byte masks and narrowing fallback control. PARAM v4,
tensor semantics and execution order are unchanged, so the validated Round 2
PARAM files and inputs remain correct.

Required SD files (8.3 names):
P2H.BIN   binary2 PARAM, 143424 bytes
I2H.BIN   binary2 input, 393216 bytes
P20H.BIN  cityscapes20 PARAM, 148032 bytes
I20H.BIN  cityscapes20 input, 393216 bytes

Generated outputs, 131072 bytes each:
O2H.BIN   binary2 mask
O20H.BIN  cityscapes20 mask

Both models print top-level RTL stage cycles, Conv WIN/SA/POST cycles,
counter consistency checks, ARM timing, and output class histograms.
The counter capability bit for Conv profiling is required.

POST state 1 now classifies a staging/discard word; state 2 classifies a word
that commits a row-buffer pixel. Both include any psum wait, requantization,
and packing; neither is a pure phase-duration counter. State 0 is idle/done.
Only compare aggregate POST activity with the old baseline's POST activity.

ARM timing uses CNTPCT_EL0 / runtime CNTFRQ_EL0, with BSP reference 33333000 Hz.
PL counters use nominal 100 MHz (10 ns), independently of the ARM timer.
The measured interval covers MODE_RUN, including hardware frame transfers.
SD access, MODE_INIT, cache maintenance, and counter/UART printing are outside
that interval. Reject runs with an ERROR stage or invalid output class IDs.

Host accuracy checks use each model's matching H256W512 artifact and target:
hw_artifacts/binary2_int8_h256w512_r2_v4
hw_artifacts/cityscapes20_int8_h256w512_r2_v4
SD outputs are never prefilled with golden data. Weights, inputs, logits and
masks are unchanged. Compare Round 4R total/Conv cycles against the measured
Round 4 baseline: binary2 12046315/7292151, cityscapes20 12064714/7310551.
Round 4R is cycle neutral: each total and Conv count must regress by no more
than 1%. The dual-single PASS label checks successful execution and output
range, not this performance gate.

Expected SHA256 for this run:
P2H.BIN  d2de465a97c802d69b691bd852e3250e13eae1329ce67df14e20fceaeeb507e4
I2H.BIN  5a1f93c1247c7de7c6d499f864f1c8af3f8c2931774cc0a1d338b10246295215
P20H.BIN fe8f9541edcffcb8100582cc7a3b3b0351395a5c18d504f697a24acaed6e71df
I20H.BIN 5a1f93c1247c7de7c6d499f864f1c8af3f8c2931774cc0a1d338b10246295215
Platform XSA: 7096f2cff088c513bacdd82d10a6f89aeaaafbdfe8c803029fa33fd772f23c72
Bitstream: c43845f76de83e27578681316690516a1b44b63755364dbb0a41636ce3ec9713
Implementation: routing errors=0, WNS=+0.293 ns, WHS=+0.010 ns at 100 MHz.
Timing and routing are signed off; board output and cycle neutrality remain to
be checked before accepting Round 4R as the performance baseline.

This README is also copied to SD as RUNINFO.TXT. Existing B/C/D/Q validation
directories are not used or modified. No local backups directory is created.
