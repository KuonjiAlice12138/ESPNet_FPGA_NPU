# H240 validation only. Synthesis/package and the production cfg stay manual.
set root [file dirname [file normalize [info script]]]
set repo [file dirname $root]
set round 6
if {[info exists ::env(ESP_INT8_CSIM_ROUND)]} {
    set round $::env(ESP_INT8_CSIM_ROUND)
}
if {$round ni {6 7 7r}} {error "Unknown H240 validation round: $round"}
set tests {focus full}
if {$round in {7 7r}} {set tests {full}}
if {[info exists ::env(ESP_INT8_ROUND6_TEST)]} {
    set tests [list $::env(ESP_INT8_ROUND6_TEST)]
}
if {$round in {7 7r} && $tests ne "full"} {error "Round7/7R runs only the final full-frame CSim"}
set handle [open [file join $root hls_config.cfg] r]
set cfg [read $handle]
close $handle
foreach test $tests {
    if {$test ni {focus full}} {error "Unknown H240 test: $test"}
    open_project -reset [file join $root hls_work_h240_round${round}_$test]
    if {$test eq "focus"} {
        add_files -cflags "-DESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER" \
            [file join $root src win_gen.cpp]
        add_files -tb -cflags "-DESP_INT8_CSIM_VALIDATE_INCREMENTAL_TRANSFER" \
            [file join $root tb win_gen_tb.cpp]
        set_top scheduled_window_generator_row
    } else {
        set ::env(ESP_INT8_CSIM_ARTIFACT_DIR) \
            [file join $repo hw_artifacts binary2_int8_h240w240_gray_r6_v4]
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
}
exit
