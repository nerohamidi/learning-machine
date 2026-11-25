# quartus_sh -t power_at_fmax.tcl <period_ns> : re-time and re-estimate power with the clock at the achieved Fmax
# (the fitted netlist is unchanged; only the SDC clock period used by the timing/power analyzers changes)
package require ::quartus::project
package require ::quartus::flow
set period [lindex $argv 0]
set f [open corelet_fmax.sdc w]
puts $f "create_clock -name clk -period $period \[get_ports clk\]"
puts $f "derive_clock_uncertainty"
puts $f "set_input_delay  -clock clk 0.0 \[remove_from_collection \[all_inputs\] \[get_ports clk\]\]"
puts $f "set_output_delay -clock clk 0.0 \[all_outputs\]"
close $f
project_open corelet -revision corelet
set_global_assignment -name SDC_FILE corelet.sdc -remove
set_global_assignment -name SDC_FILE corelet_fmax.sdc
export_assignments
execute_module -tool sta
execute_module -tool pow
set_global_assignment -name SDC_FILE corelet_fmax.sdc -remove
set_global_assignment -name SDC_FILE corelet.sdc
export_assignments
project_close
