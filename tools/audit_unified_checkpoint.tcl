# Read-only failed/placed checkpoint diagnosis. Never synthesize, place or route.
if {$argc != 1} { error "Usage: vivado -mode batch -source audit_unified_checkpoint.tcl -tclargs checkpoint.dcp" }
open_checkpoint [lindex $argv 0]
puts "UNIFIED_CHECKPOINT_ROUTE_BEGIN"
report_route_status
puts "UNIFIED_CHECKPOINT_ROUTE_END"
set paths [get_timing_paths -delay_type max -max_paths 10 -nworst 1]
puts "UNIFIED_CHECKPOINT_PATHS_BEGIN"
set index 0
foreach path $paths {
    if {$index < 20} {
        set fields {}
        foreach property {SLACK DATAPATH_DELAY LOGIC_LEVELS STARTPOINT_PIN ENDPOINT_PIN} {
            if {[catch {get_property $property $path} value]} { set value unavailable }
            lappend fields "$property=$value"
        }
        puts "${index}: [join $fields { | }]"
    }
    incr index
}
puts "UNIFIED_CHECKPOINT_PATHS_END"
puts "UNIFIED_CHECKPOINT_FANOUT_BEGIN"
report_high_fanout_nets -max_nets 10
puts "UNIFIED_CHECKPOINT_FANOUT_END"
puts "UNIFIED_CHECKPOINT_HIERARCHY_BEGIN"
set hierarchy [report_utilization -hierarchical -hierarchical_depth 14 -return_string]
foreach line [split $hierarchy "\n"] {
    set instance [string trim [lindex [split $line |] 1]]
    if {[regexp {(^inst$|espnet_encoder_int8_0|fmbuf|^grp_(core_mode|main_ctrl|execute_issue|conv_engine|run_conv|build_|shared_conv|scheduled_window|systolic|post_process|ppu_consume|ppu_finalize|vec_alu|run_fixed|avgpool)|^ppu_consume)} $instance]} {
        puts $line
    }
}
puts "UNIFIED_CHECKPOINT_HIERARCHY_END"
close_design
exit
