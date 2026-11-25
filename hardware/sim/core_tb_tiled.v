// ECE284 project, Part 1 Step 6 (+alpha): generalized controller/testbench for tiled 3x3 conv layers
// on the same core.v (8x8 weight-stationary array).
//
//   for ot in otiles:  for it in itiles:  for kij in 0..8:  load W -> PEs, stream activations, OFIFO -> pmem
//                      (after the 9 kij of tile it) SFU: out[ot][o] (+)= sum_kij psum   (partial sums over ic tiles)
//   optional ReLU, then every stored output in pmem is compared with the expected file.
//
// plusargs: +dir=<stimulus dir>  (config.txt gives n_itile n_otile relu len_nij len_onij)
//           +fast=0  template schedule (same timing as core_tb.v)
//           +fast=1  optimized schedule: 1-cycle resets, activations written to L0 while the kernel loads,
//                    OFIFO drained to pmem during execution (started by ofifo_valid), next weights
//                    prefetched into a second xmem buffer during execution, ReLU fused into the write-back
//           +vcd     dump core_tb_tiled.vcd
// xmem map: activations of ic tile it at it*len_nij, weights double-buffered at w_base and w_base+col.
// pmem map: raw psums at kij*len_nij + nij, outputs at out_base + ot*len_onij + o_nij.
`timescale 1ns/1ps

module core_tb_tiled;

parameter bw = 4;
parameter psum_bw = 16;
parameter col = 8;
parameter row = 8;
parameter len_kij = 9;
parameter out_base = 1024;
parameter w_base = 2032;

reg clk = 0;
reg reset = 1;

wire [73:0] inst_q;
reg zg = 0;
assign inst_q[73:53] = {18'b0, zg, 2'b0};   // +alpha fields idle (WS); inst[55] = zero gating (+zg=1)
assign inst_q[52:36] = {11'b0, 1'b1, 5'b0};   // Part 2 fields idle (second pmem read port disabled)
reg [bw*row-1:0] D_xmem = 0, D_xmem_q = 0;
reg CEN_xmem = 1, WEN_xmem = 1, CEN_xmem_q = 1, WEN_xmem_q = 1;
reg [10:0] A_xmem = 0, A_xmem_q = 0;
reg CEN_pmem = 1, WEN_pmem = 1, CEN_pmem_q = 1, WEN_pmem_q = 1;
reg [10:0] A_pmem = 0, A_pmem_q = 0;
reg ofifo_rd = 0, ofifo_rd_q = 0;
reg l0_rd = 0, l0_rd_q = 0, l0_wr = 0, l0_wr_q = 0;
reg execute = 0, execute_q = 0, load = 0, load_q = 0;
reg acc = 0, acc_q = 0, relu = 0, relu_q = 0, pmem_wsel = 0, pmem_wsel_q = 0;

wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

assign inst_q[35] = pmem_wsel_q;
assign inst_q[34] = relu_q;
assign inst_q[33] = acc_q;
assign inst_q[32] = CEN_pmem_q;
assign inst_q[31] = WEN_pmem_q;
assign inst_q[30:20] = A_pmem_q;
assign inst_q[19] = CEN_xmem_q;
assign inst_q[18] = WEN_xmem_q;
assign inst_q[17:7] = A_xmem_q;
assign inst_q[6] = ofifo_rd_q;
assign inst_q[5] = 1'b0;
assign inst_q[4] = 1'b0;
assign inst_q[3] = l0_rd_q;
assign inst_q[2] = l0_wr_q;
assign inst_q[1] = execute_q;
assign inst_q[0] = load_q;

core #(.bw(bw), .col(col), .row(row)) core_instance (
  .clk(clk), .inst(inst_q), .ofifo_valid(ofifo_valid),
  .D_xmem(D_xmem_q), .sfp_out(sfp_out), .reset(reset));

always @ (posedge clk) begin
  D_xmem_q <= D_xmem;  CEN_xmem_q <= CEN_xmem;  WEN_xmem_q <= WEN_xmem;  A_xmem_q <= A_xmem;
  CEN_pmem_q <= CEN_pmem;  WEN_pmem_q <= WEN_pmem;  A_pmem_q <= A_pmem;
  ofifo_rd_q <= ofifo_rd;  l0_rd_q <= l0_rd;  l0_wr_q <= l0_wr;
  execute_q <= execute;  load_q <= load;  acc_q <= acc;  relu_q <= relu;  pmem_wsel_q <= pmem_wsel;
end

integer cycle = 0;
always @ (posedge clk) cycle = cycle + 1;

`include "toggle_mon.vh"

reg [8*256:1] dir, fname;
integer n_itile, n_otile, relu_en, len_nij, len_onij, fast;
integer fd, r, it, ot, kij, t, i, j, o, n_reads, job, n_jobs, wbuf, rd_cnt, wr_cnt, started;
integer error, n_mismatch;
integer c0, c_l1 = 0, c_compute = 0, c_acc = 0, c_relu = 0;
reg [10:0] acc_addr [0:4095];
reg [bw*row-1:0] wnext [0:col-1];
reg [psum_bw*col-1:0] answer;
reg [8*64:1] junk;

task neg; begin #0.5 clk = 1'b0; end endtask
task pos; begin #0.5 clk = 1'b1; end endtask
task cyc; begin neg; pos; end endtask

// synchronous reset held for n rising edges
task do_reset (input integer n);
  integer q;
  begin
    neg; reset = 1; pos;
    for (q = 1; q < n; q = q + 1) cyc;
    neg; reset = 0; pos;
  end
endtask

task open_skip3 (input [8*256:1] name, output integer f);
  begin
    f = $fopen(name, "r");
    if (f == 0) begin $display("cannot open %0s", name); $finish; end
    r = $fscanf(f, "%s", junk); r = $fscanf(f, "%s", junk); r = $fscanf(f, "%s", junk);
  end
endtask

// DRAM -> xmem: n words of a file starting at address base
task file_to_xmem (input [8*256:1] name, input integer base, input integer n);
  integer f, q;
  begin
    open_skip3(name, f);
    for (q = 0; q < n; q = q + 1) begin
      neg; r = $fscanf(f, "%32b", D_xmem); CEN_xmem = 0; WEN_xmem = 0; A_xmem = base + q;
      pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1;
    pos;
    $fclose(f);
  end
endtask

task read_weights (input integer jb);   // weights of job jb (ot, it, kij) into wnext[]
  integer f, q, o2, i2, k2;
  begin
    k2 = jb % len_kij;  i2 = (jb / len_kij) % n_itile;  o2 = jb / (len_kij * n_itile);
    $sformat(fname, "%0s/weight_itile%0d_otile%0d_kij%0d.txt", dir, i2, o2, k2);
    open_skip3(fname, f);
    for (q = 0; q < col; q = q + 1) r = $fscanf(f, "%32b", wnext[q]);
    $fclose(f);
  end
endtask

// ---------------- one kij pass, template schedule (same as core_tb.v) ----------------
task pass_template (input integer act_base, input integer kij_i);
  begin
    do_reset(11); cyc;
    // DRAM -> xmem (weights)
    for (t = 0; t < col; t = t + 1) begin
      neg; D_xmem = wnext[t]; CEN_xmem = 0; WEN_xmem = 0; A_xmem = w_base + t;
      pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1; pos;
    // xmem -> L0 (weights)
    for (t = 0; t < col + 1; t = t + 1) begin
      neg;
      if (t < col) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = w_base + t; end else CEN_xmem = 1;
      if (t > 0) l0_wr = 1;
      pos;
    end
    neg; l0_wr = 0; pos;
    // L0 -> PEs
    for (t = 0; t < 2*col + row + 2; t = t + 1) begin neg; load = 1; l0_rd = (t < col); pos; end
    neg; load = 0; l0_rd = 0; pos;
    for (t = 0; t < 10; t = t + 1) cyc;
    // xmem -> L0 (activations)
    for (t = 0; t < len_nij + 1; t = t + 1) begin
      neg;
      if (t < len_nij) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = act_base + t; end else CEN_xmem = 1;
      if (t > 0) l0_wr = 1;
      pos;
    end
    neg; l0_wr = 0; pos;
    // execution + drain
    for (t = 0; t < len_nij; t = t + 1) begin neg; execute = 1; l0_rd = 1; pos; end
    neg; execute = 0; l0_rd = 0; pos;
    for (t = 0; t < row + col + 2; t = t + 1) cyc;
    // OFIFO -> pmem
    if (ofifo_valid !== 1'b1) begin $display("ERROR: OFIFO not valid after execution"); error = error + 1; end
    for (t = 0; t < len_nij + 1; t = t + 1) begin
      neg;
      ofifo_rd = (t < len_nij);
      if (t > 0) begin CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 0; A_pmem = kij_i*len_nij + t - 1; end
      pos;
    end
    neg; ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; pos;
    cyc;
  end
endtask

// ---------------- one kij pass, optimized schedule ----------------
// cycle t (relative):  0..col-1          xmem read weights        1..col+len_nij  l0_wr
//                      col..col+len-1    xmem read activations    (L0 written while the kernel is loading)
//                      col+1 ..          load (2*col+row+2 cyc), l0_rd for the first col cycles
//                      E = col+1+2*col+row+2+1 ..  execute + l0_rd (len_nij cycles)
//                      OFIFO read starts as soon as ofifo_valid = 1, pmem write follows by one cycle
//                      xmem idle after the activation reads -> prefetch the next job's weights (other buffer)
task pass_fast (input integer act_base, input integer kij_i, input integer cur_buf, input integer has_next);
  integer L0s, E, done, pf;
  begin
    do_reset(1);
    L0s = col + 1;
    E = L0s + 2*col + row + 2 + 1;
    rd_cnt = 0; wr_cnt = 0; started = 0; pf = 0; done = 0;
    t = 0;
    while (!done) begin
      neg;
      // xmem: weights, then activations, then prefetch of the next weights into the other buffer
      if (t < col) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = w_base + cur_buf*col + t; end
      else if (t < col + len_nij) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = act_base + t - col; end
      else if (has_next && pf < col) begin
        CEN_xmem = 0; WEN_xmem = 0; D_xmem = wnext[pf]; A_xmem = w_base + (1 - cur_buf)*col + pf; pf = pf + 1;
      end
      else begin CEN_xmem = 1; WEN_xmem = 1; end
      l0_wr = (t > 0 && t <= col + len_nij);
      // kernel load, then execution
      load    = (t >= L0s && t < L0s + 2*col + row + 2);
      execute = (t >= E && t < E + len_nij);
      l0_rd   = (t >= L0s && t < L0s + col) || (t >= E && t < E + len_nij);
      // OFIFO -> pmem as soon as a complete vector is available
      // (ofifo rd_en is registered: the vector read in cycle t is on ofifo_out in cycle t+1)
      if (ofifo_rd) begin
        CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 0; A_pmem = kij_i*len_nij + wr_cnt; wr_cnt = wr_cnt + 1;
      end
      else begin CEN_pmem = 1; WEN_pmem = 1; end
      if (!started && t > E && ofifo_valid === 1'b1) started = 1;
      ofifo_rd = (started && rd_cnt < len_nij);
      if (ofifo_rd) rd_cnt = rd_cnt + 1;
      pos;
      t = t + 1;
      if (wr_cnt == len_nij && (!has_next || pf == col)) done = 1;
      if (t > 1000) begin $display("ERROR: fast pass timeout"); error = error + 1; done = 1; end
    end
    neg; CEN_pmem = 1; WEN_pmem = 1; CEN_xmem = 1; WEN_xmem = 1; ofifo_rd = 0; l0_wr = 0; pos;
  end
endtask

// ---------------- SFU: accumulate the 9 kij psums (+ previous ic-tile partial) and write back ----------------
task accumulate (input integer ot_i, input integer it_i, input integer fuse_relu);
  integer q;
  begin
    n_reads = len_kij + ((it_i > 0) ? 1 : 0);
    for (o = 0; o < len_onij; o = o + 1) begin
      if (fast) begin neg; reset = 1; pos; end
      else begin neg; reset = 1; pos; neg; reset = 0; pos; end
      for (j = 0; j < n_reads + 1; j = j + 1) begin
        neg;
        reset = 0;
        if (j < len_kij)      begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = acc_addr[o*len_kij + j]; end
        else if (j < n_reads) begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = out_base + ot_i*len_onij + o; end
        else                  begin CEN_pmem = 1; WEN_pmem = 1; end
        acc = (j > 0);
        pos;
      end
      neg; acc = 0; relu = fuse_relu; CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 1; A_pmem = out_base + ot_i*len_onij + o;
      pos;
      neg; CEN_pmem = 1; WEN_pmem = 1;
      pos;
      if (!fast) begin neg; pmem_wsel = 0; relu = 0; pos; end
    end
    neg; pmem_wsel = 0; relu = 0; pos;
  end
endtask

// ---------------- separate ReLU pass (template schedule) ----------------
task relu_pass (input integer ot_i);
  begin
    for (o = 0; o < len_onij; o = o + 1) begin
      neg; reset = 1; pos; neg; reset = 0; pos;
      neg; CEN_pmem = 0; WEN_pmem = 1; A_pmem = out_base + ot_i*len_onij + o; pos;
      neg; CEN_pmem = 1; acc = 1; pos;
      neg; acc = 0; relu = 1; CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 1; pos;
      neg; CEN_pmem = 1; WEN_pmem = 1; pos;
      neg; relu = 0; pmem_wsel = 0; pos;
    end
  end
endtask

initial begin
  error = 0; n_mismatch = 0;
  if (!$value$plusargs("dir=%s", dir)) dir = "stim_f27";
  if (!$value$plusargs("fast=%d", fast)) fast = 0;
  if (!$value$plusargs("zg=%d", zg)) zg = 0;
  if ($test$plusargs("vcd")) begin $dumpfile("core_tb_tiled.vcd"); $dumpvars(0, core_tb_tiled); end

  $sformat(fname, "%0s/config.txt", dir);
  fd = $fopen(fname, "r");
  r = $fscanf(fd, "%d %d %d %d %d", n_itile, n_otile, relu_en, len_nij, len_onij);
  $fclose(fd);
  $display("dir=%0s  itiles=%0d otiles=%0d relu=%0d nij=%0d onij=%0d  schedule=%0s",
           dir, n_itile, n_otile, relu_en, len_nij, len_onij, fast ? "optimized" : "template");

  $sformat(fname, "%0s/acc_address.txt", dir);
  open_skip3(fname, fd);
  for (i = 0; i < len_onij*len_kij; i = i + 1) r = $fscanf(fd, "%11b", acc_addr[i]);
  $fclose(fd);

  do_reset(11); cyc;

  // DRAM -> xmem: activations of every ic tile (reused by all output tiles)
  c0 = cycle;
  for (it = 0; it < n_itile; it = it + 1) begin
    $sformat(fname, "%0s/activation_tile%0d.txt", dir, it);
    file_to_xmem(fname, it*len_nij, len_nij);
  end
  n_jobs = n_otile * n_itile * len_kij;
  if (fast) begin              // first weights to buffer 0
    read_weights(0);
    for (t = 0; t < col; t = t + 1) begin
      neg; D_xmem = wnext[t]; CEN_xmem = 0; WEN_xmem = 0; A_xmem = w_base + t; pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1; pos;
  end
  c_l1 = cycle - c0;

  wbuf = 0;
  mon_en = 1;
  for (ot = 0; ot < n_otile; ot = ot + 1) begin
    for (it = 0; it < n_itile; it = it + 1) begin
      c0 = cycle;
      for (kij = 0; kij < len_kij; kij = kij + 1) begin
        job = (ot*n_itile + it)*len_kij + kij;
        if (fast) begin
          if (job + 1 < n_jobs) read_weights(job + 1);
          pass_fast(it*len_nij, kij, wbuf, job + 1 < n_jobs);
          wbuf = 1 - wbuf;
        end
        else begin
          read_weights(job);
          pass_template(it*len_nij, kij);
        end
      end
      c_compute = c_compute + cycle - c0;
      c0 = cycle;
      accumulate(ot, it, fast && relu_en && (it == n_itile - 1));
      c_acc = c_acc + cycle - c0;
    end
    if (relu_en && !fast) begin
      c0 = cycle;
      relu_pass(ot);
      c_relu = c_relu + cycle - c0;
    end
  end

  cyc; cyc; mon_en = 0;   // let the last write-back land inside the window
  // compare every stored output with the expected file
  for (ot = 0; ot < n_otile; ot = ot + 1) begin
    $sformat(fname, "%0s/out_otile%0d.txt", dir, ot);
    open_skip3(fname, fd);
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

  $display("outputs checked: %0d vectors (%0d values), mismatches: %0d, control errors: %0d",
           n_otile*len_onij, n_otile*len_onij*col, n_mismatch, error);
  $display("cycles: L1 load %0d | compute (kernel+L0+exec+OFIFO) %0d | accumulate %0d | ReLU %0d | total %0d",
           c_l1, c_compute, c_acc, c_relu, cycle);
  mon_report;
  if (n_mismatch == 0 && error == 0) $display("RESULT: PASS");
  else                               $display("RESULT: FAIL");
  #10 $finish;
end

endmodule
