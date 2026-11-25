# quartus_sta -t report_regreg.tcl : worst register-to-register setup paths at each slow corner
project_open corelet -revision corelet
create_timing_netlist
read_sdc
update_timing_netlist
foreach cond {7_slow_1200mv_100c} {
  set_operating_conditions $cond
  update_timing_netlist
  report_timing -setup -from [all_registers] -to [all_registers] -npaths 5 -detail summary \
                -panel_name "regreg $cond" -file output_files/regreg_$cond.rpt
  report_timing -setup -from [all_registers] -to [all_registers] -npaths 1 -detail full_path \
                -panel_name "regreg path $cond" -file output_files/regreg_path_$cond.rpt
}
delete_timing_netlist
project_close
