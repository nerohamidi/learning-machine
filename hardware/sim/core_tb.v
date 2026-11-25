// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
//
// ECE284 project, Part 1: testbench for the squeezed VGG16 layer (features[27], 8 ic x 8 oc, 4x4 map)
// on the 8x8 weight-stationary array. The testbench is the controller (and emulates DRAM).
// Stages: 1 L1 (xmem) loading of activations and weights   2 kernel loading to the PEs (via L0)
//         3 L0 activation loading   4 execution   5 psum movement to pmem (via OFIFO)
//         6 accumulation in SFU + write-back to pmem   7 ReLU in SFU + write-back to pmem
// Stimulus/expected files are in stim/ (software/scripts/gen_stimulus.py).
`timescale 1ns/1ps

module core_tb;

parameter bw = 4;
parameter psum_bw = 16;
parameter len_kij = 9;
parameter len_onij = 16;
parameter col = 8;
parameter row = 8;
parameter len_nij = 36;
parameter out_base = 1024;          // pmem address of the accumulated outputs
parameter w_base   = 1024;          // xmem address of the weights

reg clk = 0;
reg reset = 1;

wire [73:0] inst_q;
assign inst_q[73:53] = 21'b0;   // +alpha fields idle (WS, no gating)
assign inst_q[52:36] = {11'b0, 1'b1, 5'b0};   // Part 2 fields idle (second pmem read port disabled)

reg [1:0]  inst_w_q = 0;
reg [bw*row-1:0] D_xmem_q = 0;
reg CEN_xmem = 1;
reg WEN_xmem = 1;
reg [10:0] A_xmem = 0;
reg CEN_xmem_q = 1;
reg WEN_xmem_q = 1;
reg [10:0] A_xmem_q = 0;
reg CEN_pmem = 1;
reg WEN_pmem = 1;
reg [10:0] A_pmem = 0;
reg CEN_pmem_q = 1;
reg WEN_pmem_q = 1;
reg [10:0] A_pmem_q = 0;
reg ofifo_rd_q = 0;
reg ififo_wr_q = 0;
reg ififo_rd_q = 0;
reg l0_rd_q = 0;
reg l0_wr_q = 0;
reg execute_q = 0;
reg load_q = 0;
reg acc_q = 0;
reg acc = 0;
reg relu_q = 0;
reg relu = 0;
reg pmem_wsel_q = 0;
reg pmem_wsel = 0;

reg [1:0]  inst_w;
reg [bw*row-1:0] D_xmem;
reg [psum_bw*col-1:0] answer;


reg ofifo_rd;
reg ififo_wr;
reg ififo_rd;
reg l0_rd;
reg l0_wr;
reg execute;
reg load;
reg [8*64:1] stringvar;
reg [8*64:1] w_file_name;
wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

integer x_file, x_scan_file ; // file_handler
integer w_file, w_scan_file ; // file_handler
integer acc_file, acc_scan_file ; // file_handler
integer out_file, out_scan_file ; // file_handler
integer captured_data;
integer t, i, j, k, kij;
integer error, n_err_raw, n_err_acc, n_err_relu, n_err_mem;
integer cycle;
integer c_start, c_l1, c_kernel, c_exec, c_ofifo, c_acc, c_relu;

assign inst_q[35] = pmem_wsel_q;
assign inst_q[34] = relu_q;
assign inst_q[33] = acc_q;
assign inst_q[32] = CEN_pmem_q;
assign inst_q[31] = WEN_pmem_q;
assign inst_q[30:20] = A_pmem_q;
assign inst_q[19]   = CEN_xmem_q;
assign inst_q[18]   = WEN_xmem_q;
assign inst_q[17:7] = A_xmem_q;
assign inst_q[6]   = ofifo_rd_q;
assign inst_q[5]   = ififo_wr_q;
assign inst_q[4]   = ififo_rd_q;
assign inst_q[3]   = l0_rd_q;
assign inst_q[2]   = l0_wr_q;
assign inst_q[1]   = execute_q;
assign inst_q[0]   = load_q;


core  #(.bw(bw), .col(col), .row(row)) core_instance (
	.clk(clk),
	.inst(inst_q),
	.ofifo_valid(ofifo_valid),
        .D_xmem(D_xmem_q),
        .sfp_out(sfp_out),
	.reset(reset));

initial cycle = 0;
always @ (posedge clk) cycle = cycle + 1;

// expected raw psums of every kij (pmem address kij*len_nij + nij), checked after stage 5
reg [psum_bw*col-1:0] psum_kij_ref [0:len_kij*len_nij-1];
reg [psum_bw*col-1:0] out_ref      [0:len_onij-1];

initial begin

  inst_w   = 0;
  D_xmem   = 0;
  CEN_xmem = 1;
  WEN_xmem = 1;
  A_xmem   = 0;
  ofifo_rd = 0;
  ififo_wr = 0;
  ififo_rd = 0;
  l0_rd    = 0;
  l0_wr    = 0;
  execute  = 0;
  load     = 0;
  error = 0; n_err_raw = 0; n_err_acc = 0; n_err_relu = 0; n_err_mem = 0;
  c_l1 = 0; c_kernel = 0; c_exec = 0; c_ofifo = 0;

  $dumpfile("core_tb.vcd");
  $dumpvars(0,core_tb);

  x_file = $fopen("stim/activation_tile0.txt", "r");
  // Following three lines are to remove the first three comment lines of the file
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);

  //////// Reset /////////
  #0.5 clk = 1'b0;   reset = 1;
  #0.5 clk = 1'b1;

  for (i=0; i<10 ; i=i+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end

  #0.5 clk = 1'b0;   reset = 0;
  #0.5 clk = 1'b1;

  #0.5 clk = 1'b0;
  #0.5 clk = 1'b1;
  /////////////////////////
  c_start = cycle;

  /////// Stage 1a: Activation data writing to memory ///////
  for (t=0; t<len_nij; t=t+1) begin
    #0.5 clk = 1'b0;  x_scan_file = $fscanf(x_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
    #0.5 clk = 1'b1;
  end

  #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
  #0.5 clk = 1'b1;

  $fclose(x_file);
  c_l1 = c_l1 + (cycle - c_start);
  /////////////////////////////////////////////////


  for (kij=0; kij<9; kij=kij+1) begin  // kij loop

    $sformat(w_file_name, "stim/weight_itile0_otile0_kij%0d.txt", kij);
    w_file = $fopen(w_file_name, "r");
    // Following three lines are to remove the first three comment lines of the file
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);

    #0.5 clk = 1'b0;   reset = 1;
    #0.5 clk = 1'b1;

    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;   reset = 0;
    #0.5 clk = 1'b1;

    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;


    /////// Stage 1b: Kernel data writing to memory ///////
    c_start = cycle;
    A_xmem = w_base;

    for (t=0; t<col; t=t+1) begin
      #0.5 clk = 1'b0;  w_scan_file = $fscanf(w_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1;
    $fclose(w_file);
    c_l1 = c_l1 + (cycle - c_start);
    /////////////////////////////////////


    /////// Stage 2a: Kernel data writing to L0 ///////
    // xmem has a 1-cycle read latency, so l0_wr follows the read address by one cycle.
    c_start = cycle;
    for (t=0; t<col+1; t=t+1) begin
      #0.5 clk = 1'b0;
        if (t<col) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = w_base + t; end
        else       begin CEN_xmem = 1; WEN_xmem = 1; end
        if (t>0) l0_wr = 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  l0_wr = 0; A_xmem = 0;
    #0.5 clk = 1'b1;
    /////////////////////////////////////


    /////// Stage 2b: Kernel loading to PEs ///////
    // L0 is read row by row (row r one cycle after row r-1). Word t lands in column t:
    // a PE keeps the first word it sees while load is high and passes the load flag east.
    // Column j of row r loads ~2j + r cycles after the first read, so load stays high 2*col+row+2 cycles.
    for (t=0; t<2*col+row+2; t=t+1) begin
      #0.5 clk = 1'b0;  load = 1; l0_rd = (t < col);
      #0.5 clk = 1'b1;
    end
    /////////////////////////////////////


    ////// provide some intermission to clear up the kernel loading ///
    #0.5 clk = 1'b0;  load = 0; l0_rd = 0;
    #0.5 clk = 1'b1;


    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end
    c_kernel = c_kernel + (cycle - c_start);
    /////////////////////////////////////


    /////// Stage 3: Activation data writing to L0 ///////
    c_start = cycle;
    for (t=0; t<len_nij+1; t=t+1) begin
      #0.5 clk = 1'b0;
        if (t<len_nij) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = t; end
        else           begin CEN_xmem = 1; WEN_xmem = 1; end
        if (t>0) l0_wr = 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  l0_wr = 0; A_xmem = 0;
    #0.5 clk = 1'b1;
    /////////////////////////////////////


    /////// Stage 4: Execution ///////
    for (t=0; t<len_nij; t=t+1) begin
      #0.5 clk = 1'b0;  execute = 1; l0_rd = 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  execute = 0; l0_rd = 0;
    #0.5 clk = 1'b1;

    // drain: the last activation leaves the bottom-right PE row+col cycles later
    for (t=0; t<row+col+2; t=t+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end
    c_exec = c_exec + (cycle - c_start);
    /////////////////////////////////////


    //////// Stage 5: OFIFO READ -> psum SRAM ////////
    // Ideally, OFIFO should be read while execution, but we have enough ofifo
    // depth so we can fetch out after execution.
    // ofifo's rd_en is registered, so the pmem write follows ofifo_rd by one cycle.
    c_start = cycle;
    if (ofifo_valid !== 1'b1) begin
      $display("kij %0d: ERROR, OFIFO has no complete output vector after execution", kij);
      error = 1;
    end
    for (t=0; t<len_nij+1; t=t+1) begin
      #0.5 clk = 1'b0;
        ofifo_rd = (t < len_nij);
        if (t>0) begin CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 0; A_pmem = kij*len_nij + t-1; end
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; A_pmem = 0;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
    if (ofifo_valid !== 1'b0) begin
      $display("kij %0d: ERROR, OFIFO not empty after reading %0d vectors", kij, len_nij);
      error = 1;
    end
    c_ofifo = c_ofifo + (cycle - c_start);
    /////////////////////////////////////

  end  // end of kij loop


  ////////// Check the raw psums stored in pmem (all kij) /////////
  out_file = $fopen("stim/psum_kij.txt", "r");
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  for (i=0; i<len_kij*len_nij; i=i+1) begin
    out_scan_file = $fscanf(out_file,"%128b", answer);
    if (core_instance.pmem_instance.memory[i] !== answer) begin
      n_err_raw = n_err_raw + 1;
      if (n_err_raw <= 5) begin
        $display("raw psum kij %0d nij %0d ERROR", i/len_nij, i%len_nij);
        $display("  pmem  : %128b", core_instance.pmem_instance.memory[i]);
        $display("  answer: %128b", answer);
      end
    end
  end
  $fclose(out_file);
  $display("Raw psums (pmem[0..%0d], %0d x %0d-bit x %0d cols): %0d mismatches",
           len_kij*len_nij-1, len_kij*len_nij, psum_bw, col, n_err_raw);


  ////////// Stage 6: Accumulation in SFU + write-back /////////
  out_file = $fopen("stim/psum.txt", "r");
  acc_file = $fopen("stim/acc_address.txt", "r");

  // Following three lines are to remove the first three comment lines of the file
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  acc_scan_file = $fscanf(acc_file,"%s", captured_data);
  acc_scan_file = $fscanf(acc_file,"%s", captured_data);
  acc_scan_file = $fscanf(acc_file,"%s", captured_data);

  $display("############ Verification Start during accumulation #############");
  c_start = cycle;

  for (i=0; i<len_onij; i=i+1) begin

    #0.5 clk = 1'b0; reset = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; reset = 0;
    #0.5 clk = 1'b1;

    // read the 9 psums of this output pixel; pmem Q appears one cycle after the address,
    // so acc is raised one cycle after the first read
    for (j=0; j<len_kij+1; j=j+1) begin

      #0.5 clk = 1'b0;
        if (j<len_kij) begin CEN_pmem = 0; WEN_pmem = 1; acc_scan_file = $fscanf(acc_file,"%11b", A_pmem); end
                       else  begin CEN_pmem = 1; WEN_pmem = 1; end

        if (j>0)  acc = 1;
      #0.5 clk = 1'b1;
    end

    // write the accumulated psum back to pmem[out_base + o_nij]
    #0.5 clk = 1'b0; acc = 0; CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 1; A_pmem = out_base + i;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; pmem_wsel = 0;
    #0.5 clk = 1'b1;

    out_scan_file = $fscanf(out_file,"%128b", answer); // reading from out file to answer
    if (sfp_out == answer)
      $display("%2d-th output featuremap Data matched! :D", i);
    else begin
      $display("%2d-th output featuremap Data ERROR!!", i);
      $display("sfpout: %128b", sfp_out);
      $display("answer: %128b", answer);
      n_err_acc = n_err_acc + 1;
    end
  end
  c_acc = cycle - c_start;
  $fclose(acc_file);
  $fclose(out_file);
  //////////////////////////////////


  ////////// Stage 7: ReLU in SFU + write-back /////////
  out_file = $fopen("stim/out.txt", "r");
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);

  $display("############ Verification Start during ReLU #############");
  c_start = cycle;

  for (i=0; i<len_onij; i=i+1) begin
    #0.5 clk = 1'b0; reset = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; reset = 0;
    #0.5 clk = 1'b1;

    // read the accumulated psum
    #0.5 clk = 1'b0; CEN_pmem = 0; WEN_pmem = 1; A_pmem = out_base + i;
    #0.5 clk = 1'b1;
    // load it into the SFU accumulator
    #0.5 clk = 1'b0; CEN_pmem = 1; acc = 1;
    #0.5 clk = 1'b1;
    // ReLU and write back in place
    #0.5 clk = 1'b0; acc = 0; relu = 1; CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1;
    #0.5 clk = 1'b1;

    out_scan_file = $fscanf(out_file,"%128b", answer);
    out_ref[i] = answer;
    if (sfp_out == answer)
      $display("%2d-th output featuremap after ReLU matched! :D", i);
    else begin
      $display("%2d-th output featuremap after ReLU ERROR!!", i);
      $display("sfpout: %128b", sfp_out);
      $display("answer: %128b", answer);
      n_err_relu = n_err_relu + 1;
    end

    #0.5 clk = 1'b0; relu = 0; pmem_wsel = 0;
    #0.5 clk = 1'b1;
  end
  c_relu = cycle - c_start;
  $fclose(out_file);

  // the ReLU outputs must also be in pmem[out_base ...] (stored back by stage 7)
  for (i=0; i<len_onij; i=i+1)
    if (core_instance.pmem_instance.memory[out_base+i] !== out_ref[i]) begin
      n_err_mem = n_err_mem + 1;
      $display("pmem[%0d] (stored ReLU output %0d) ERROR", out_base+i, i);
    end
  //////////////////////////////////

  $display("------------------------------------------------------------");
  $display("raw psum mismatches            : %0d / %0d", n_err_raw, len_kij*len_nij);
  $display("accumulation mismatches        : %0d / %0d", n_err_acc, len_onij);
  $display("ReLU mismatches                : %0d / %0d", n_err_relu, len_onij);
  $display("stored output (pmem) mismatches: %0d / %0d", n_err_mem, len_onij);
  $display("cycles: L1 load %0d | kernel load %0d | L0+exec %0d | OFIFO->pmem %0d | acc %0d | ReLU %0d | total %0d",
           c_l1, c_kernel, c_exec, c_ofifo, c_acc, c_relu, cycle);
  $display("------------------------------------------------------------");

  if (error == 0 && n_err_raw == 0 && n_err_acc == 0 && n_err_relu == 0 && n_err_mem == 0) begin
  	$display("############ No error detected ##############");
  	$display("########### Project Completed !! ############");
  end
  else
  	$display("############ ERRORS DETECTED ##############");

  //////////////////////////////////

  for (t=0; t<10; t=t+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end

  #10 $finish;

end

always @ (posedge clk) begin
   inst_w_q   <= inst_w;
   D_xmem_q   <= D_xmem;
   CEN_xmem_q <= CEN_xmem;
   WEN_xmem_q <= WEN_xmem;
   A_pmem_q   <= A_pmem;
   CEN_pmem_q <= CEN_pmem;
   WEN_pmem_q <= WEN_pmem;
   A_xmem_q   <= A_xmem;
   ofifo_rd_q <= ofifo_rd;
   acc_q      <= acc;
   relu_q     <= relu;
   pmem_wsel_q <= pmem_wsel;
   ififo_wr_q <= ififo_wr;
   ififo_rd_q <= ififo_rd;
   l0_rd_q    <= l0_rd;
   l0_wr_q    <= l0_wr ;
   execute_q  <= execute;
   load_q     <= load;
end


endmodule
