ESP INT8 board app.

Required SD files:
PARAM.BIN
INPUTQ.BIN

Generated SD output:
OUTQ.BIN

Debug bring-up output:
D00OUT.BIN, D01OUT.BIN, D02OUT.BIN, D03OUT.BIN, D04OUT.BIN,
D20OUT.BIN, D39OUT.BIN, D53OUT.BIN, D69OUT.BIN, D72OUT.BIN

All numerical comparison is done offline on the host. The PS app does not load
reference files from SD card and does not compare outputs on board.

When the HLS IP is rebuilt with live debug registers, timeout logs also include
phase/opcode/current-uop/last-done-uop and stream word counters. Older exported
platforms build without these prints because the generated register macros are
not present.
