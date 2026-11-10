# Source/output-extension validation only; production cfg and synthesis remain manual.
set root [file dirname [file normalize [info script]]]
set test full
if {[info exists ::env(ESP_INT8_INPUT_RESIZE_TEST)]} {set test $::env(ESP_INT8_INPUT_RESIZE_TEST)}
if {$test ni {unit mask240 full}} {error "Resize test must be unit, mask240, or full"}
set study D:/ESPNet/resolution_h240w240_gray/output_restore_1104
puts [exec D:/ESPNet/.venv/Scripts/python.exe -B [file join $study output_restore_1104.py] \
    --validate-results [file join $study results_500.json]]
open_project -reset [file join $root hls_work_output_restore_$test]
if {$test eq "unit"} {
    add_files [file join $root src frame_dma.cpp]
    add_files [file join $root src memory.cpp]
    add_files -tb [file join $root tb frame_dma_tb.cpp]
    set_top frame_dma_load
} else {
    set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) \
        [file join [file dirname $root] hw_artifacts binary2_int8_h240w240_gray_r6_v4]
    set ::env(ESP_INT8_CSIM_OUTPUT_REFERENCE_DIR) [file join $study selected_reference]
    unset -nocomplain ::env(ESP_INT8_CSIM_UNITS_ONLY)
    set flags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE"
    if {$test eq "mask240"} {
        append flags " -DESP_INT8_MASK_SIZE=240"
        set ::env(ESP_INT8_CSIM_UNITS_ONLY) 1
    }
    set handle [open [file join $root hls_config.cfg] r]
    set cfg [read $handle]
    close $handle
    foreach line [split $cfg "\n"] {
        if {[regexp {^syn.file=(.+)$} [string trim $line] -> path]} {
            add_files -cflags $flags [file join $root $path]
        }
    }
    add_files -tb -cflags $flags \
        [file join $root tb top_golden_sample_tb.cpp]
    set_top espnet_encoder_int8_core
}
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
close_project
exit
