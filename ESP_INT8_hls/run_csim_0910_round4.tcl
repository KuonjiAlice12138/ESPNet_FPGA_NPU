# Round 4 focused and dual-model CSim. Keep the production cfg unchanged.
set root [file dirname [file normalize [info script]]]
set repo [file dirname $root]
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
if {[info exists ::env(ESP_INT8_ROUND4_TEST)]} {
  if {$::env(ESP_INT8_ROUND4_TEST) eq "dual"} {
    set tests {binary2 cityscapes20}
  } else {
    set tests [list $::env(ESP_INT8_ROUND4_TEST)]
  }
}

foreach test $tests {
  if {$test ni {focus binary2 cityscapes20}} {error "Unknown test: $test"}
  set profile [expr {$test eq "focus" ? "binary2" : $test}]
  set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) \
      [file join $repo hw_artifacts ${profile}_int8_h256w512_r2_v4]
  open_project -reset [file join $root hls_work_0910_round4_$test]
  if {$test eq "focus"} {
    add_files -cflags "-DESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER" \
        [file join $root src win_gen.cpp]
    set_top scheduled_window_generator_row
    add_files -tb -cflags "-DESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER" \
        [file join $root tb c3_row_reuse_tb.cpp]
  } else {
    foreach source $sources {
      add_files -cflags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" $source
    }
    set_top espnet_encoder_int8_core
    set classes [expr {$test eq "binary2" ? 2 : 20}]
    add_files -tb \
        -cflags "-DESP_INT8_CSIM_CLASS_COUNT=$classes -DESP_INT8_CSIM_VALIDATE_ROW_REUSE" \
        [file join $root tb top_golden_sample_tb.cpp]
  }
  open_solution -reset solution1 -flow_target vivado
  set_part {xczu15eg-ffvb1156-2-i}
  create_clock -period 10
  csim_design
  close_project
}

unset ::env(ESP_INT8_CSIM_ARTIFACT_DIR)
exit
