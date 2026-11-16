# Native512 CSim only; never synthesize or modify the production configuration.
set root [file dirname [file normalize [info script]]]
set repo [file dirname $root]
set test full
if {[info exists ::env(ESP_INT8_NATIVE512_TEST)]} {set test $::env(ESP_INT8_NATIVE512_TEST)}
if {$test ni {dma focus boundary full}} {error "Unknown native512 CSim test: $test"}
set project [file join $root hls_work_h512w512_gray_finetuned_$test]
if {[info exists ::env(ESP_INT8_NATIVE512_WORK_DIR)]} {
    set project [file normalize $::env(ESP_INT8_NATIVE512_WORK_DIR)]
    if {[file dirname $project] ne $root || [file exists $project]} {
        error "Native512 CSim override requires a fresh work directory inside the HLS project"
    }
}
open_project -reset $project
if {$test eq "dma"} {
    add_files -cflags "-O2" [file join $root src frame_dma.cpp]
    add_files -cflags "-O2" [file join $root src memory.cpp]
    add_files -tb -cflags "-O2" [file join $root tb frame_dma_tb.cpp]
    set_top frame_dma_load
} elseif {$test eq "focus"} {
    set flags "-O2 -DESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER"
    add_files -cflags $flags [file join $root src win_gen.cpp]
    add_files -tb -cflags $flags [file join $root tb win_gen_tb.cpp]
    set_top scheduled_window_generator_row
} else {
    set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) [file join $repo hw_artifacts binary2_int8_h512w512_gray_finetuned_v4]
    if {[info exists ::env(ESP_INT8_NATIVE512_ARTIFACT_DIR)]} {
        set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) [file normalize $::env(ESP_INT8_NATIVE512_ARTIFACT_DIR)]
    }
    if {$test eq "boundary"} {
        set ::env(ESP_INT8_CSIM_BOUNDARY_ONLY) 1
    } else {
        catch {unset ::env(ESP_INT8_CSIM_BOUNDARY_ONLY)}
    }
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
}
open_solution -reset solution1 -flow_target vivado
set_part {xczu15eg-ffvb1156-2-i}
create_clock -period 10
csim_design
close_project
exit
