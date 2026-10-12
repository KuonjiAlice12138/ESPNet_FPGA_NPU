ESP INT8 board app: H240W240 grayscale binary2 single-image profiling.

Platform: platform_h240_1012, standalone_psu_cortexa53_0, Cortex-A53 0.
App tag: INT8-BOARD-20261011-P7-H240W240-GRAY-B2.
The user rebuilt the App on 2026-10-12 at 17:05. The debug launch uses the
new platform's bitstream and FSBL; it must not use the failed R5A platform.
BOOT.BIN was packaged from boot_h240.bif and the rebuilt App, then copied to E:/.
All three SD files were verified by SHA256. No old ELF or BOOT is used.

One boot runs binary2 once. Dual-model, EXEC-prefix and validation-set modes
are disabled. PARAM v4 has 75 logical UOPs, 16 EXEC records including END,
26 CONV, 7 AFFINE, 14 ADD and 3 POOL; U71/U72 remain separate.
Input is true grayscale NHWC INT8, 240x240x1. Output is a 240x240 uint8 mask.
All DDR buffers are 64-byte aligned; PARAM capacity remains 262144 bytes.

SD files (8.3 names):
BOOT.BIN  freshly packaged FSBL + H240 bitstream + rebuilt App, for SD boot
P2G.BIN   PARAM, 143040 bytes
I2G.BIN   grayscale input, 57600 bytes
O2G.BIN   generated output, 57600 bytes; remove any old copy before running

Copy PARAM.BIN and INPUTQ.BIN from:
hw_artifacts/binary2_int8_h240w240_gray_v4
Do not use either preserved H256/RGB artifact with this hardware.
The input uses center-square crop, resize to 240x240 and the QAT grayscale
normalization. Labels are vehicles=0, valid background=1, ignored target=255.
Golden fullres output contains c0=4105, c1=53495 and no invalid class IDs.
The board mask must match golden_fullres_mask.bin byte for byte. A valid
histogram alone is not sufficient to pass the output-correctness gate.

Top-level stage and Conv WIN/SA/POST counter dumps remain enabled; the Conv
capability bit is required. Reject ERROR stage activity, timeout or invalid IDs.
POST state 1 classifies staging words; state 2 classifies commit words. Both
include psum wait, requantization and packing, not pure phase durations.
WIN/SA/POST overlap within Conv; their cycle totals must not be summed.

ARM timing uses CNTPCT_EL0 / runtime CNTFRQ_EL0, BSP reference 33333000 Hz.
PL counters use nominal 100 MHz (10 ns), independently of the ARM timer.
MODE_RUN includes hardware frame transfers. SD I/O, MODE_INIT, cache
maintenance and counter/UART dumps are outside the measured interval.
Initial target: <=6000000 PL cycles (60 ms); bring-up ceiling: 7000000 (70 ms).
Actual performance and correctness require the board result, not HLS max latency.
Only after the single-image gate passes should the 500-image run be prepared.

Verified SHA256:
P2G.BIN  bb92bafa4ac297a760ee4880887fdeffe8cee0279107f9f01f3fa355e4244fa0
I2G.BIN  c1f3b03245d4bc6514c2a2dfcada898453f0d8eca6292ddcd46b40c1c2b5dd87
Golden   88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce
XSA      67eb53a804bedf6d1f8a12ad5415569251d370a327b7fcec847c7c72eac9ec56
Bit      b3f1ddbf970ed68784f28d7bfb1d336b8894024446cb51dd409dddbce65201f3
App ELF  2f440a0dffa089ab9f845329307e459d23d18252cf98bed4dccb02f7990ae56b
BOOT.BIN f1d82983e24f08694b42369fc76bfaa32358c219be8fdf86c44f12c05a08a6f2

Implementation: routing errors=0, WNS=+0.252 ns, WHS=+0.010 ns at 100 MHz.
Placed resources: logic LUT=219686, total LUT=236675, FF=192017,
BRAM tile=203.5, DSP=1298, URAM=28. Post-route phys-opt did not modify the
netlist. XSA, exported platform XSA and embedded bitstream hashes agree.
NPU control=0xA0000000; RTL counter=0xA0010000. The stage-valid and all three
Conv-state ports are connected to the counter in the exported hardware.
SD root now contains BOOT.BIN, P2G.BIN and I2G.BIN, with O2G.BIN absent.
Old H256 root test files were removed; validation folders B/C/D/Q were preserved.
ELF buffers: input=0x28440, output=0x36540, param=0x44640; aligned and nonoverlapping.
Board execution and output-correctness/performance gates remain pending.
