# Input-extension validation only; production cfg and synthesis remain manual.
set root [file dirname [file normalize [info script]]]
set test full
if {[info exists ::env(ESP_INT8_INPUT_RESIZE_TEST)]} {set test $::env(ESP_INT8_INPUT_RESIZE_TEST)}
if {$test ni {unit full}} {error "Input resize test must be unit or full"}
set study D:/ESPNet/resolution_h240w240_gray/input_resize_1023
puts [exec D:/ESPNet/.venv/Scripts/python.exe -B [file join $study input_resize_1023.py] \
    --validate-prequant-manifest [file join $study selected_int8_reference RESIZEQ.json]]
open_project -reset [file join $root hls_work_input_resize_$test]
if {$test eq "unit"} {
    add_files [file join $root src frame_dma.cpp]
    add_files [file join $root src memory.cpp]
    add_files -tb [file join $root tb frame_dma_tb.cpp]
    set_top frame_dma_load
} else {
    set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) \
        [file join [file dirname $root] hw_artifacts binary2_int8_h240w240_gray_r6_v4]
    set ::env(ESP_INT8_CSIM_INPUT_REFERENCE_DIR) $study
    set handle [open [file join $root hls_config.cfg] r]
    set cfg [read $handle]
    close $handle
    foreach line [split $cfg "\n"] {
        if {[regexp {^syn.file=(.+)$} [string trim $line] -> path]} {
            add_files -cflags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" [file join $root $path]
        }
    }
    add_files -tb -cflags "-DESP_INT8_CSIM_DUMP_LOWRES_LOGITS_SIDE" \
        [file join $root tb top_golden_sample_tb.cpp]
    set_top espnet_encoder_int8_core
}
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
close_project
exit
