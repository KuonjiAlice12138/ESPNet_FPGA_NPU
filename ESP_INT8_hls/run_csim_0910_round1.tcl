# CSim only: reuse the GUI source list without changing hls_config.cfg.
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
set tests {focus binary2 cityscapes20}
if {[info exists ::env(ESP_INT8_ROUND1_TEST)]} {
  set tests [list $::env(ESP_INT8_ROUND1_TEST)]
}
foreach test $tests {
  if {$test ni {focus binary2 cityscapes20}} {error "Unknown test: $test"}
  open_project -reset [file join $root hls_work_0910_round1_$test]
  foreach source $sources {
    add_files -cflags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" $source
  }
  if {$test eq "focus"} {
    set_top post_process_conv_row_to_buffer
    add_files -tb [file join $root tb conv_post_tb.cpp]
  } else {
    set_top espnet_encoder_int8_core
    set classes [expr {$test eq "binary2" ? 2 : 20}]
    add_files -tb -cflags "-DESP_INT8_CSIM_CLASS_COUNT=$classes" [file join $root tb top_golden_sample_tb.cpp]
  }
  open_solution -reset solution1 -flow_target vivado
  set_part {xczu15eg-ffvb1156-2-i}
  create_clock -period 10
  csim_design
  close_project
}
exit
