# Focused one-row RTL handshake attribution for Round 4R-3. This project uses
# the production Conv/WinGen/SA/POST call graph under a test-only top wrapper.
set root [file dirname [file normalize [info script]]]
set handle [open [file join $root hls_config.cfg] r]
set cfg [read $handle]
close $handle

set sources {}
foreach line [split $cfg "\n"] {
  if {[regexp {^syn.file=(.+)$} [string trim $line] -> path]} {
    lappend sources [file join $root $path]
  }
}

set probe_case c3
if {[info exists ::env(ESP_INT8_R4R3_CASE)]} {
  set probe_case $::env(ESP_INT8_R4R3_CASE)
}
if {$probe_case ni {c3 c12}} {error "Unknown Round4R-3 case: $probe_case"}

set phase csim
if {[info exists ::env(ESP_INT8_R4R3_PHASE)]} {
  set phase $::env(ESP_INT8_R4R3_PHASE)
}
if {$phase ni {csim cosim}} {error "Unknown Round4R-3 phase: $phase"}

open_project -reset [file join $root hls_work_0910_round4r3_${probe_case}]
foreach source $sources {
  add_files -cflags "-DESP_INT8_R4R_HANDSHAKE_PROBE" $source
}
set_top round4r_conv_row_handshake_probe
add_files -tb [file join $root tb r4r_handshake_tb.cpp]
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
if {$phase eq "cosim"} {
  csynth_design
  cosim_design -rtl verilog -tool xsim -trace_level port_hier \
      -enable_dataflow_profiling
}
close_project
exit
