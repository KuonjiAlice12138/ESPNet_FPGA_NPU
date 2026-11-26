# Optional CSim runner; synthesis, implementation and packaging remain manual.
set root [file dirname [file normalize [info script]]]
set project [file join $root hls_work_unified_csim]
if {[info exists ::env(ESP_INT8_UNIFIED_CSIM_WORK_DIR)]} {
    set project [file normalize $::env(ESP_INT8_UNIFIED_CSIM_WORK_DIR)]
}
if {[file dirname $project] ne $root || [file exists $project]} {
    error "Use a fresh CSim work directory directly inside the HLS component"
}
if {![info exists ::env(ESP_INT8_CSIM_RUNTIME_ROOT)]} {
    set ::env(ESP_INT8_CSIM_RUNTIME_ROOT) [file join [file dirname $root] hw_artifacts unified_int8]
}
if {![file exists [file join $::env(ESP_INT8_CSIM_RUNTIME_ROOT) RUNS.CSV]]} {
    error "Publish and validate a unified runtime catalog first"
}
open_project $project
set flags "-O2 -DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE"
set handle [open [file join $root hls_config.cfg] r]
set cfg [read $handle]
close $handle
foreach line [split $cfg "\n"] {
    if {[regexp {^syn.file=(.+)$} [string trim $line] -> path]} {
        add_files -cflags $flags [file join $root $path]
    }
}
add_files -tb -cflags $flags [file join $root tb top_golden_sample_tb.cpp]
set_top espnet_encoder_int8_core
open_solution solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
close_project
exit
