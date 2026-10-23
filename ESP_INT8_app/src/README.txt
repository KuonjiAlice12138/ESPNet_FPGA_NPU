ESP INT8 board app: H240W240 grayscale binary2 Round 8 single-image profiling.

Platform: platform_h240_1021, standalone_psu_cortexa53_0, Cortex-A53 0.
App tag: INT8-BOARD-20261021-P7-H240W240-GRAY-R8-B2.
app.yaml, the App component, boot_h240.bif and the JTAG launch configuration
select this platform. JTAG bitstream, FSBL and psu_init paths are aligned.
App builds remain manual. Round 8 has been built and passed the single-image
board test. Never reuse a Round 7R ELF or BOOT.BIN for this release; any rebuild
requires packaging a matching BOOT.BIN before SD boot.

One boot runs binary2 once. Dual-model, EXEC-prefix and validation-set modes
are disabled. PARAM v4 has 75 logical UOPs, 16 EXEC records including END,
26 CONV, 7 AFFINE, 14 ADD and 3 POOL; U71/U72 remain separate.
Input is true grayscale NHWC INT8, 240x240x1. Output is a 240x240 uint8 mask.
All DDR buffers are 64-byte aligned; PARAM capacity remains 262144 bytes.

SD files (8.3 names):
BOOT.BIN  matching FSBL + Round 8 bitstream + rebuilt App, required for SD boot
P2G.BIN   PARAM, 143040 bytes
I2G.BIN   grayscale input, 57600 bytes
O2G.BIN   generated output, 57600 bytes; remove any old copy before running

Copy PARAM.BIN and INPUTQ.BIN from:
hw_artifacts/binary2_int8_h240w240_gray_r6_v4
Hardware profile: p7_h240w240_gray_c1_r6_delta.
Round 8 reuses this artifact without re-export/QAT. Schedule word 9 selects
packed-row/delta transfer protocols. The baseline H240 PARAM is incompatible.
Do not use preserved H256/RGB artifacts or the baseline H240 BOOT.
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
The XSA reports 99.999001 MHz, unchanged from the previous platform.
MODE_RUN includes hardware frame transfers. SD I/O, MODE_INIT, cache
maintenance and counter/UART dumps are outside the measured interval.
Board-tested Round 7 baseline: 5334364 cycles (53.34364 ms), Conv=3418908,
PPU row=289114, BLOCK5 final=773550, Vec=667534, Pool=135617 cycles.
Round 7R measured 5097039 PL cycles (50.97039 ms), a 4.45% improvement over R7.
Conv=3418908, PPU row=289114, BLOCK5 final=679846, Vec=523904, Pool=135617.
ARM measured 1699004 ticks / 33333000 Hz = 50.9706297 ms; ERROR=0.
The SD output matches golden byte for byte (0/57600 mismatch).
Round 8 measured 4985440 PL cycles (49.854400 ms), 2.19% faster than R7R.
ARM measured 1661804 ticks / 33333000 Hz = 49.854618546 ms; ERROR=0.
Conv=3418908, PPU row=289114, BLOCK5 final=658246, Vec=433904, Pool=135617.
The board mask matches golden byte for byte (0/57600 mismatch). Fullres CSim
also passed: logits 0/1800, mask 0/57600 mismatches, last_error=0.
Vec<=450000 passes; BLOCK5 final<=600000 remains a future working budget.
Frozen stages retain Round 7 performance. This single result is below 50 ms,
but three consecutive timing runs and 500-image board validation remain open.
Actual performance requires board measurements, not HLS max latency or II=1.

Verified SHA256:
P2G.BIN  9e380cebe305c4945642c4a334f75a7a2ba94eefa1f2ccc7f2964f8d3bf9d878
I2G.BIN  c1f3b03245d4bc6514c2a2dfcada898453f0d8eca6292ddcd46b40c1c2b5dd87
Golden   88bc39cb00f9f6056d76ed85f74b69c86ff4808e6f868d40f6f4e6203ce426ce
XSA      3851987666143fc94cb2c3fc325367d9a6cdd1ad284bc357eb63a7e7507ff0b1
Bit      6988d69a594720394faf6b3a9789ca38d2d0e69b287c48eebe2c09dfd28c89c4
FSBL     d34cb10632763bd7d00ea27f35f4ce4011dc0aff0a92aa3905991e5650c344c7
App ELF and BOOT.BIN must be verified after the manual build/package.

Implementation: routing errors=0, WNS=+0.585 ns, WHS=+0.010 ns at 100 MHz.
Placed resources: logic LUT=203653, total LUT=221366, FF=173049,
BRAM tile=204, DSP=1292, URAM=28, CLB=40642/42660 (95.27%). XSA, exported platform XSA
and embedded bitstream hashes agree with the completed implementation.
Installed Vec/BLOCK5 RTL matches the current Round 8 HLS synthesis. Resources
and setup margin improve relative to R7R; board timing is recorded above.
NPU control=0xA0000000; RTL counter=0xA0010000. The stage-valid and all three
Conv-state ports are connected to the counter in the exported hardware.
Prepare E:/ with P2G.BIN and I2G.BIN plus a newly packaged BOOT.BIN for SD boot.
The Round 8 O2G.BIN has been verified against golden; remove it before
the next board test to prevent mistaking a stale output for the new result.
Preserve validation folders B/C/D/Q; do not reformat the card.
Check the rebuilt ELF buffer addresses for 64-byte alignment and non-overlap.

After manually building ESP_INT8_app on platform_h240_1021:
1. Check the new ELF contains the Round 8 tag above and uses the new BSP.
2. From D:/ESP_INT8, package the corrected boot recipe:
   & 'D:/Xilinx/2025.1/Vivado/bin/bootgen.bat' -arch zynqmp -image ESP_INT8_app/src/boot_h240.bif -o ESP_INT8_app/BOOT.BIN -w on
3. Copy the newly generated ESP_INT8_app/BOOT.BIN to E:/BOOT.BIN and verify hashes.
Do not start SD boot until this final step is complete.
JTAG can use the aligned launch configuration after the build without SD BOOT.
The SD data files alone do not make a bootable card. The stable source checkpoint
is the board-tested Round 8 version, not an unimplemented workplan_1021 upgrade.
Stable <50 ms across three runs and 500-image board validation remain pending.
