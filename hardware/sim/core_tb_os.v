// +alpha: a 3x3 conv layer executed in OUTPUT-STATIONARY mode on the same core (inst[53] os = 1).
//
// Per output tile (8 pixels x 8 output channels):
//   stream K steps: xmem -> L0 (activations, west) and wmem -> IFIFO (weights, north) in parallel, 1 step/cycle;
//   every PE accumulates its own output pixel / channel in place (no psum traffic, no SFU accumulation);
//   drain: 8 cycles, one result row per cycle into the OFIFO, then OFIFO -> SFU (ReLU) -> pmem.
// Data residency: all activation tiles stay in xmem if n_ptile*K <= 2048, the weights of the current oc tile stay
// in wmem if K <= 2048; otherwise the reduction is split into chunks of `chunk` steps (the PEs simply keep
// their accumulators while the next chunk is loaded from DRAM).
// pmem: outputs at out_base + ot*len_onij + o_nij, compared with out_otile*.txt (same files as the WS flow).
// plusargs: +dir=<stimulus dir>  +zg=0/1 (zero-guard gating)  +vcd
`timescale 1ns/1ps

module core_tb_os;

parameter bw = 4;
parameter psum_bw = 16;
parameter col = 8;
parameter row = 8;
parameter out_base = 1024;
parameter chunk = 1024;
parameter drain_lat = 16;   // cycles from the last execute to the first drain cycle (PE(7,7) accumulates last);
                            // minimum: 15 fails on dense random data (stim_os_rand)

reg clk = 0;
reg reset = 1;

wire [73:0] inst_q;
reg [bw*row-1:0] D_xmem = 0, D_xmem_q = 0;
reg CEN_xmem = 1, WEN_xmem = 1, CEN_xmem_q = 1, WEN_xmem_q = 1;
reg [10:0] A_xmem = 0, A_xmem_q = 0;
reg wmem_ce = 0, wmem_we = 0, wmem_ce_q = 0, wmem_we_q = 0;
reg [10:0] A_wmem = 0, A_wmem_q = 0;
reg CEN_pmem = 1, WEN_pmem = 1, CEN_pmem_q = 1, WEN_pmem_q = 1;
reg [10:0] A_pmem = 0, A_pmem_q = 0;
reg ofifo_rd = 0, ofifo_rd_q = 0;
reg l0_rd = 0, l0_rd_q = 0, l0_wr = 0, l0_wr_q = 0;
reg ififo_rd = 0, ififo_rd_q = 0, ififo_wr = 0, ififo_wr_q = 0;
reg execute = 0, execute_q = 0;
reg relu = 0, relu_q = 0, pmem_wsel = 0, pmem_wsel_q = 0, src_ofifo = 0, src_ofifo_q = 0;
reg drain = 0, drain_q = 0;
reg os = 1, zg = 0;

wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

assign inst_q[73:69] = 5'b0;
assign inst_q[68:58] = A_wmem_q;
assign inst_q[57] = wmem_we_q;
assign inst_q[56] = wmem_ce_q;
assign inst_q[55] = zg;
assign inst_q[54] = drain_q;
assign inst_q[53] = os;
assign inst_q[52:42] = 11'b0;
assign inst_q[41] = 1'b1;          // second pmem read port idle
assign inst_q[40:38] = 3'b0;
assign inst_q[37] = src_ofifo_q;
assign inst_q[36] = 1'b0;          // activations unsigned
assign inst_q[35] = pmem_wsel_q;
assign inst_q[34] = relu_q;
assign inst_q[33] = 1'b0;
assign inst_q[32] = CEN_pmem_q;
assign inst_q[31] = WEN_pmem_q;
assign inst_q[30:20] = A_pmem_q;
assign inst_q[19] = CEN_xmem_q;
assign inst_q[18] = WEN_xmem_q;
assign inst_q[17:7] = A_xmem_q;
assign inst_q[6] = ofifo_rd_q;
assign inst_q[5] = ififo_wr_q;
assign inst_q[4] = ififo_rd_q;
assign inst_q[3] = l0_rd_q;
assign inst_q[2] = l0_wr_q;
assign inst_q[1] = execute_q;
assign inst_q[0] = 1'b0;           // no kernel load in OS

core #(.bw(bw), .col(col), .row(row)) core_instance (
  .clk(clk), .inst(inst_q), .ofifo_valid(ofifo_valid),
  .D_xmem(D_xmem_q), .sfp_out(sfp_out), .reset(reset));

always @ (posedge clk) begin
  D_xmem_q <= D_xmem;  CEN_xmem_q <= CEN_xmem;  WEN_xmem_q <= WEN_xmem;  A_xmem_q <= A_xmem;
  wmem_ce_q <= wmem_ce;  wmem_we_q <= wmem_we;  A_wmem_q <= A_wmem;
  CEN_pmem_q <= CEN_pmem;  WEN_pmem_q <= WEN_pmem;  A_pmem_q <= A_pmem;
  ofifo_rd_q <= ofifo_rd;  l0_rd_q <= l0_rd;  l0_wr_q <= l0_wr;  ififo_rd_q <= ififo_rd;  ififo_wr_q <= ififo_wr;
  execute_q <= execute;  relu_q <= relu;  pmem_wsel_q <= pmem_wsel;  src_ofifo_q <= src_ofifo;  drain_q <= drain;
end

integer cycle = 0;
always @ (posedge clk) cycle = cycle + 1;

`include "toggle_mon.vh"

reg [8*256:1] dir, fname;
integer n_pt, n_ot, K, relu_cfg, len_onij;
integer fd, r, pt, ot, o, j, error, n_mismatch, n_tiles;
integer act_resident, wgt_resident, cur_wgt_ot, cur_chunk_pt, cur_chunk_ot, cur_chunk;
integer c_l1 = 0, c_compute = 0, c0, n_exec = 0;
reg [psum_bw*col-1:0] answer;
reg [8*64:1] junk;

task neg; begin #0.5 clk = 1'b0; end endtask
task pos; begin #0.5 clk = 1'b1; end endtask
task cyc; begin neg; pos; end endtask

task do_reset (input integer n);
  integer q;
  begin
    neg; reset = 1; pos;
    for (q = 1; q < n; q = q + 1) cyc;
    neg; reset = 0; pos;
  end
endtask

// DRAM -> xmem (which = 0) or wmem (which = 1): words [first, first+n) of a file, to address base
task file_to_mem (input [8*256:1] name, input integer which, input integer first, input integer n,
                  input integer base);
  integer fh, q;
  begin
    c0 = cycle;
    fh = $fopen(name, "r");
    if (fh == 0) begin $display("cannot open %0s", name); $finish; end
    r = $fscanf(fh, "%s", junk); r = $fscanf(fh, "%s", junk); r = $fscanf(fh, "%s", junk);
    for (q = 0; q < first; q = q + 1) r = $fscanf(fh, "%32b", D_xmem);
    for (q = 0; q < n; q = q + 1) begin
      neg; r = $fscanf(fh, "%32b", D_xmem);
      if (which == 0) begin CEN_xmem = 0; WEN_xmem = 0; A_xmem = base + q; end
      else            begin wmem_ce = 1;  wmem_we = 1;  A_wmem = base + q; end
      pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1; wmem_ce = 0; wmem_we = 0;
    pos;
    $fclose(fh);
    c_l1 = c_l1 + cycle - c0;
  end
endtask

// stream n steps: xmem[xa .. xa+n) -> L0 and wmem[wa .. wa+n) -> IFIFO, execute one step per cycle
task stream (input integer xa, input integer wa, input integer n);
  integer t;
  begin
    for (t = 0; t < n + 2; t = t + 1) begin
      neg;
      if (t < n) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = xa + t; wmem_ce = 1; wmem_we = 0; A_wmem = wa + t; end
      else       begin CEN_xmem = 1; WEN_xmem = 1; wmem_ce = 0; end
      l0_wr = (t >= 1 && t <= n);  ififo_wr = l0_wr;
      execute = (t >= 2);          l0_rd = execute;  ififo_rd = execute;
      if (execute) n_exec = n_exec + 1;
      pos;
    end
    neg; execute = 0; l0_rd = 0; ififo_rd = 0; l0_wr = 0; ififo_wr = 0; pos;
  end
endtask

// drain the 8x8 results: OFIFO vector j = PE row 7-j = pixel 8*pt + 7 - j -> SFU (ReLU) -> pmem
task drain_tile (input integer pt_i, input integer ot_i);
  integer t, pix;
  begin
    for (t = 1; t < drain_lat; t = t + 1) cyc;          // last execute already issued one cycle ago
    for (t = 0; t < row + 2; t = t + 1) begin
      neg;
      drain = (t < row);
      ofifo_rd = (t >= 1 && t <= row);
      CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; relu = 0;
      if (t >= 2) begin
        pix = 8*pt_i + row - 1 - (t - 2);
        if (pix < len_onij) begin
          CEN_pmem = 0; WEN_pmem = 0; A_pmem = out_base + ot_i*len_onij + pix;
          pmem_wsel = 1; src_ofifo = 1; relu = relu_cfg;
        end
      end
      pos;
    end
    neg; drain = 0; ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; relu = 0; pos;
  end
endtask

task run_tile (input integer pt_i, input integer ot_i);
  integer k0, n;
  begin
    for (k0 = 0; k0 < K; k0 = k0 + n) begin
      n = (act_resident && wgt_resident) ? K : ((K - k0 < chunk) ? K - k0 : chunk);
      if (!act_resident) begin
        $sformat(fname, "%0s/os_act_p%0d.txt", dir, pt_i);  file_to_mem(fname, 0, k0, n, 0);
      end
      if (!wgt_resident) begin
        $sformat(fname, "%0s/os_wgt_o%0d.txt", dir, ot_i);  file_to_mem(fname, 1, k0, n, 0);
      end
      else if (cur_wgt_ot != ot_i) begin
        $sformat(fname, "%0s/os_wgt_o%0d.txt", dir, ot_i);  file_to_mem(fname, 1, 0, K, 0);
        cur_wgt_ot = ot_i;
      end
      c0 = cycle;
      stream(act_resident ? pt_i*K + k0 : 0, wgt_resident ? k0 : 0, n);
      c_compute = c_compute + cycle - c0;
    end
    c0 = cycle;
    drain_tile(pt_i, ot_i);
    c_compute = c_compute + cycle - c0;
    n_tiles = n_tiles + 1;
  end
endtask

initial begin
  error = 0; n_mismatch = 0; n_tiles = 0;
  if (!$value$plusargs("dir=%s", dir)) dir = "stim_os_f27";
  if (!$value$plusargs("zg=%d", zg)) zg = 0;
  if ($test$plusargs("vcd")) begin $dumpfile("core_tb_os.vcd"); $dumpvars(0, core_tb_os); end
  $sformat(fname, "%0s/config.txt", dir);
  fd = $fopen(fname, "r");
  r = $fscanf(fd, "%d %d %d %d %d", n_pt, n_ot, K, relu_cfg, len_onij);
  $fclose(fd);
  act_resident = (n_pt*K <= 2048);
  wgt_resident = (K <= 2048);
  cur_wgt_ot = -1;
  $display("dir=%0s  OS: %0d pixel tiles x %0d oc tiles, K=%0d, relu=%0d, zero gating=%0d", dir, n_pt, n_ot, K,
           relu_cfg, zg);

  do_reset(11); cyc;
  if (act_resident)
    for (pt = 0; pt < n_pt; pt = pt + 1) begin
      $sformat(fname, "%0s/os_act_p%0d.txt", dir, pt);  file_to_mem(fname, 0, 0, K, pt*K);
    end

  mon_en = 1;
  for (ot = 0; ot < n_ot; ot = ot + 1)
    for (pt = 0; pt < n_pt; pt = pt + 1)
      run_tile(pt, ot);
  cyc; cyc; mon_en = 0;   // let the last write-back land inside the window

  cyc; cyc;
  for (ot = 0; ot < n_ot; ot = ot + 1) begin
    $sformat(fname, "%0s/out_otile%0d.txt", dir, ot);
    fd = $fopen(fname, "r");
    r = $fscanf(fd, "%s", junk); r = $fscanf(fd, "%s", junk); r = $fscanf(fd, "%s", junk);
    for (o = 0; o < len_onij; o = o + 1) begin
      r = $fscanf(fd, "%128b", answer);
      if (core_instance.pmem_instance.memory[out_base + ot*len_onij + o] !== answer) begin
        n_mismatch = n_mismatch + 1;
        if (n_mismatch <= 5) begin
          $display("otile %0d o_nij %0d ERROR", ot, o);
          $display("  pmem  : %128b", core_instance.pmem_instance.memory[out_base + ot*len_onij + o]);
          $display("  answer: %128b", answer);
        end
      end
    end
    $fclose(fd);
  end
  $display("outputs checked: %0d vectors (%0d values), mismatches: %0d", n_ot*len_onij, n_ot*len_onij*col,
           n_mismatch);
  $display("cycles: L1 load %0d | compute (stream+drain+write-back) %0d | total %0d | tiles %0d | execute %0d",
           c_l1, c_compute, cycle, n_tiles, n_exec);
  mon_report;
  if (n_mismatch == 0 && error == 0) $display("RESULT: PASS");
  else                               $display("RESULT: FAIL");
  #10 $finish;
end

endmodule
