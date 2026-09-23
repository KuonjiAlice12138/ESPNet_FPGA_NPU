# Round 3 dual-model full-network CSim only. Keep the production cfg unchanged.
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

foreach profile {binary2 cityscapes20} {
  set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) \
      [file join $repo hw_artifacts ${profile}_int8_h256w512_r2_v4]
  open_project -reset [file join $root hls_work_0910_round3_$profile]
  foreach source $sources {
    add_files -cflags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" $source
  }
  set_top espnet_encoder_int8_core
  set classes [expr {$profile eq "binary2" ? 2 : 20}]
  add_files -tb \
      -cflags "-DESP_INT8_CSIM_CLASS_COUNT=$classes -DESP_INT8_CSIM_VALIDATE_ROW_REUSE" \
      [file join $root tb top_golden_sample_tb.cpp]
  open_solution -reset solution1 -flow_target vivado
  set_part {xczu15eg-ffvb1156-2-i}
  create_clock -period 10
  csim_design
  close_project
}

unset ::env(ESP_INT8_CSIM_ARTIFACT_DIR)
exit
