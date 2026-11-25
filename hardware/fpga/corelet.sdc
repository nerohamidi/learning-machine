# corelet timing constraints (Part 1 Step 5). The 4 ns target is deliberately tight so the timing analyzer
# reports the achievable Fmax; the frequency we quote is the slow-corner Fmax, not this target.
create_clock -name clk -period 4.000 [get_ports clk]
derive_clock_uncertainty

# I/O of the corelet connects to the SRAMs/controller outside the FPGA mapping: constrain loosely so the
# reported Fmax reflects the internal register-to-register paths.
set_input_delay  -clock clk 0.0 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk 0.0 [all_outputs]
