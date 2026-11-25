# Part 1 Step 5: map corelet.v onto Cyclone IV GX EP4CGX150DF31I7AD with Quartus Prime (spec: 19.1; run here with Lite 20.1 on Windows).
# Usage (from project/hardware/fpga):   quartus_sh -t run_flow.tcl
# Runs analysis & synthesis, fitter (place & route), assembler, timing analyzer and power analyzer
# (20% default toggle rate), then prints Fmax at every corner and the total power.

package require ::quartus::project
package require ::quartus::flow

set proj corelet
set rtl  ../verilog

if {[project_exists $proj]} { project_open $proj -revision $proj } else { project_new $proj -revision $proj }

set_global_assignment -name FAMILY "Cyclone IV GX"
set_global_assignment -name DEVICE EP4CGX150DF31I7AD
set_global_assignment -name TOP_LEVEL_ENTITY corelet
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name MIN_CORE_JUNCTION_TEMP -40
set_global_assignment -name MAX_CORE_JUNCTION_TEMP 100
foreach f {mac.v mac_tile.v mac_row.v mac_array.v fifo_mux_2_1.v fifo_mux_8_1.v fifo_mux_16_1.v
           fifo_depth64.v l0.v ofifo.v sfu.v corelet.v} {
  set_global_assignment -name VERILOG_FILE $rtl/$f
}
set_global_assignment -name SDC_FILE corelet.sdc
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"

# power analysis: 20% toggle rate for all inputs and internal nodes without simulation data
set_global_assignment -name POWER_DEFAULT_TOGGLE_RATE "20%"
set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "20%"
set_global_assignment -name POWER_USE_INPUT_FILES OFF
set_global_assignment -name POWER_REPORT_SIGNAL_ACTIVITY ON
set_global_assignment -name POWER_REPORT_POWER_DISSIPATION ON

export_assignments

execute_module -tool map
execute_module -tool fit
execute_module -tool asm
execute_module -tool sta -args "--do_report_timing"
execute_module -tool pow

project_close
