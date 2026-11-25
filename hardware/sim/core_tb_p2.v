// ECE284 project, Part 2: differential processing of video frames X0..X3 on the 8x8 weight-stationary array.
//
// +mode=diff (differential):
//   host/testbench: X0 -> (hi, lo) nibbles, dXi = sat4(Xi - X0)        (delta generation in the testbench = host)
//   for kij in 0..8:  load W[kij] into the PEs ONCE, then 5 back-to-back passes with W kept loaded:
//                     X0_hi (signed, <<4) -> Y0 | X0_lo (unsigned) -> Y0 | dX1 -> D1 | dX2 -> D2 | dX3 -> D3
//                     every raw psum goes OFIFO -> SFU (+ pmem read of the running sum) -> pmem, 1 element/cycle
//   reconstruction:   Y_i = Y0 + D_i (SFU: acc_ld Y0, then add D_i), Y0 stays stored for all three frames
// +mode=conv (conventional baseline): every frame processed at 8 bit, frame by frame:
//   for f in 0..3: DRAM -> xmem (hi_f, lo_f); for kij: load W[kij], 2 passes (hi_f <<4, lo_f) -> Y_f
//
// xmem map: activation pass p at p*len_nij, W[kij] at w_base + kij*col.
// pmem map (128b, 1R1W): Y0 at 0, D_i at i*len_onij, outputs Y_i (i = 1..3) at out_base + (i-1)*len_onij.
// Every stored output (Y0 and Y1..Y3) is compared with y0.txt .. y3.txt from gen_part2_stimulus.py.
// plusargs: +dir=<stimulus dir> +mode=diff|conv +vcd
`timescale 1ns/1ps

module core_tb_p2;

parameter bw = 4;
parameter psum_bw = 16;
parameter col = 8;
parameter row = 8;
parameter len_kij = 9;
parameter K = 3;
parameter out_base = 1024;
parameter w_base = 1664;
parameter max_pass = 5;

reg clk = 0;
reg reset = 1;

wire [73:0] inst_q;
reg zg = 0;
assign inst_q[73:53] = {18'b0, zg, 2'b0};   // +alpha fields idle (WS); inst[55] = zero gating (+zg=1)
reg [bw*row-1:0] D_xmem = 0, D_xmem_q = 0;
reg CEN_xmem = 1, WEN_xmem = 1, CEN_xmem_q = 1, WEN_xmem_q = 1;
reg [10:0] A_xmem = 0, A_xmem_q = 0;
reg CEN_pmem = 1, WEN_pmem = 1, CEN_pmem_q = 1, WEN_pmem_q = 1;
reg [10:0] A_pmem = 0, A_pmem_q = 0;
reg CEN_pmem_rd = 1, CEN_pmem_rd_q = 1;
reg [10:0] A_pmem_rd = 0, A_pmem_rd_q = 0;
reg ofifo_rd = 0, ofifo_rd_q = 0;
reg l0_rd = 0, l0_rd_q = 0, l0_wr = 0, l0_wr_q = 0;
reg execute = 0, execute_q = 0, load = 0, load_q = 0;
reg acc = 0, acc_q = 0, relu = 0, relu_q = 0, pmem_wsel = 0, pmem_wsel_q = 0;
reg act_signed = 0, act_signed_q = 0, src_ofifo = 0, src_ofifo_q = 0, shift4 = 0, shift4_q = 0;
reg add_q = 0, add_q_q = 0, acc_ld = 0, acc_ld_q = 0;

wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

assign inst_q[52:42] = A_pmem_rd_q;
assign inst_q[41] = CEN_pmem_rd_q;
assign inst_q[40] = acc_ld_q;
assign inst_q[39] = add_q_q;
assign inst_q[38] = shift4_q;
assign inst_q[37] = src_ofifo_q;
assign inst_q[36] = act_signed_q;
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
  CEN_pmem_rd_q <= CEN_pmem_rd;  A_pmem_rd_q <= A_pmem_rd;
  ofifo_rd_q <= ofifo_rd;  l0_rd_q <= l0_rd;  l0_wr_q <= l0_wr;
  execute_q <= execute;  load_q <= load;  acc_q <= acc;  relu_q <= relu;  pmem_wsel_q <= pmem_wsel;
  act_signed_q <= act_signed;  src_ofifo_q <= src_ofifo;  shift4_q <= shift4;  add_q_q <= add_q;  acc_ld_q <= acc_ld;
end

integer cycle = 0;
always @ (posedge clk) cycle = cycle + 1;

`include "toggle_mon.vh"

reg [8*256:1] dir, fname, mode_s;
integer len_nij, len_onij, n_frames, pw, diff;
integer fd, r, f, i, o, t, kij, error, n_mismatch;
integer c0, c_l1 = 0, c_compute = 0, c_recon = 0, n_passes = 0, n_wloads = 0, n_exec = 0;
// per-pass descriptors of the current kij phase
integer n_pass;
integer pa_base [0:max_pass-1], pa_sgn [0:max_pass-1], pa_region [0:max_pass-1];
integer pa_shift [0:max_pass-1], pa_first [0:max_pass-1];
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

task open_skip3 (input [8*256:1] name, output integer fh);
  begin
    fh = $fopen(name, "r");
    if (fh == 0) begin $display("cannot open %0s", name); $finish; end
    r = $fscanf(fh, "%s", junk); r = $fscanf(fh, "%s", junk); r = $fscanf(fh, "%s", junk);
  end
endtask

// DRAM -> xmem: n words of a file starting at address base (one word per cycle)
task file_to_xmem (input [8*256:1] name, input integer base, input integer n);
  integer fh, q;
  begin
    open_skip3(name, fh);
    for (q = 0; q < n; q = q + 1) begin
      neg; r = $fscanf(fh, "%32b", D_xmem); CEN_xmem = 0; WEN_xmem = 0; A_xmem = base + q;
      pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1;
    pos;
    $fclose(fh);
  end
endtask

task act_to_xmem (input [8*64:1] name, input integer slot);
  begin
    $sformat(fname, "%0s/act_%0s.txt", dir, name);
    file_to_xmem(fname, slot*len_nij, len_nij);
  end
endtask

// ---------------- one kij phase: load W[kij] once, then n_pass back-to-back passes ----------------
// cycle t (relative):  0..col-1              xmem read W[kij]           1..col+N  l0_wr   (N = n_pass*len_nij)
//                      col..col+N-1          xmem read activations (streamed into L0 while it drains)
//                      L0s..                 kernel load (2*col+row+2 cycles), l0_rd for the first col cycles
//                      E..E+N-1              execute + l0_rd, act_signed of the pass the element belongs to
//                      ofifo_valid ->        OFIFO read (1/cycle); for element n of pass p at nij:
//                                            if nij feeds output o for this kij: read pmem[region_p + o] now,
//                                            write pmem[region_p + o] = (raw << shift_p) + old next cycle
task kij_phase (input integer kij_i);
  integer L0s, E, N, done, rd_cnt, started, n, p, nij, ny, nx, oy, ox, ky, kx;
  integer pend, pend_addr, pend_shift, pend_addq;
  begin
    do_reset(1);
    N = n_pass * len_nij;
    L0s = col + 1;
    E = L0s + 2*col + row + 2 + 1;
    ky = kij_i / K;  kx = kij_i % K;
    rd_cnt = 0; started = 0; done = 0; pend = 0;
    t = 0;
    while (!done) begin
      neg;
      // xmem: W[kij], then the activations of all passes
      if (t < col) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = w_base + kij_i*col + t; end
      else if (t < col + N) begin
        n = t - col;  CEN_xmem = 0; WEN_xmem = 1; A_xmem = pa_base[n / len_nij] + (n % len_nij);
      end
      else begin CEN_xmem = 1; WEN_xmem = 1; end
      l0_wr   = (t > 0 && t <= col + N);
      load    = (t >= L0s && t < L0s + 2*col + row + 2);
      execute = (t >= E && t < E + N);
      l0_rd   = (t >= L0s && t < L0s + col) || (t >= E && t < E + N);
      act_signed = (t >= E && t < E + N) ? pa_sgn[(t - E) / len_nij] : 0;
      if (execute) n_exec = n_exec + 1;
      // write-back of the element read from the OFIFO in the previous cycle
      if (pend) begin
        CEN_pmem = 0; WEN_pmem = 0; A_pmem = pend_addr; pmem_wsel = 1;
        src_ofifo = 1; shift4 = pend_shift; add_q = pend_addq;
      end
      else begin CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; shift4 = 0; add_q = 0; end
      pend = 0;
      // OFIFO read + pmem read of the running sum
      if (!started && t > E && ofifo_valid === 1'b1) started = 1;
      ofifo_rd = (started && rd_cnt < N);
      CEN_pmem_rd = 1;
      if (ofifo_rd) begin
        n = rd_cnt;  p = n / len_nij;  nij = n % len_nij;
        ny = nij / 18;  nx = nij % 18;
        oy = ny - ky;  ox = nx - kx;
        if (oy >= 0 && oy < 16 && ox >= 0 && ox < 16) begin
          pend = 1; pend_addr = pa_region[p] + oy*16 + ox;
          pend_shift = pa_shift[p]; pend_addq = !(kij_i == 0 && pa_first[p]);
          CEN_pmem_rd = 0; A_pmem_rd = pend_addr;
        end
        rd_cnt = rd_cnt + 1;
      end
      pos;
      t = t + 1;
      if (rd_cnt == N && !pend) done = 1;
      if (t > N + 1000) begin $display("ERROR: kij phase timeout"); error = error + 1; done = 1; end
    end
    neg; CEN_pmem = 1; WEN_pmem = 1; CEN_pmem_rd = 1; CEN_xmem = 1; WEN_xmem = 1; ofifo_rd = 0; l0_wr = 0;
    pmem_wsel = 0; src_ofifo = 0; shift4 = 0; add_q = 0; act_signed = 0;
    pos;
    n_passes = n_passes + n_pass;
    n_wloads = n_wloads + 1;
  end
endtask

// ---------------- reconstruction: Y_i[o] = Y0[o] + D_i[o], 4 cycles per output pixel ----------------
//   phase 0: read Y0[o]   (and write Y3[o-1])      phase 1: acc_ld (acc = Y0[o]), read D1[o]
//   phase 2: write Y1[o] = acc + Q, read D2[o]     phase 3: write Y2[o], read D3[o]
task reconstruct;
  integer ph;
  begin
    for (t = 0; t <= 4*len_onij; t = t + 1) begin
      neg;
      o = t / 4;  ph = t % 4;
      CEN_pmem_rd = 1; CEN_pmem = 1; WEN_pmem = 1; acc_ld = 0; add_q = 0; pmem_wsel = 0; src_ofifo = 0;
      if (ph == 0 && o < len_onij) begin CEN_pmem_rd = 0; A_pmem_rd = o; end
      if (ph >= 1) begin CEN_pmem_rd = 0; A_pmem_rd = ph*len_onij + o; end
      if (ph == 1) acc_ld = 1;
      if (ph >= 2 || (ph == 0 && o > 0)) begin
        CEN_pmem = 0; WEN_pmem = 0; pmem_wsel = 1; add_q = 1;
        A_pmem = (ph == 0) ? out_base + 2*len_onij + o - 1 : out_base + (ph - 2)*len_onij + o;
      end
      pos;
    end
    neg; CEN_pmem_rd = 1; CEN_pmem = 1; WEN_pmem = 1; acc_ld = 0; add_q = 0; pmem_wsel = 0; pos;
  end
endtask

task set_pass (input integer p, input integer slot, input integer sgn, input integer region,
               input integer sh, input integer first);   // sh = 1: raw << 4
  begin
    pa_base[p] = slot*len_nij; pa_sgn[p] = sgn; pa_region[p] = region; pa_shift[p] = sh; pa_first[p] = first;
  end
endtask

initial begin
  error = 0; n_mismatch = 0;
  if (!$value$plusargs("dir=%s", dir)) dir = "stim_p2";
  if (!$value$plusargs("mode=%s", mode_s)) mode_s = "diff";
  if (!$value$plusargs("zg=%d", zg)) zg = 0;
  diff = (mode_s == "diff");
  if ($test$plusargs("vcd")) begin $dumpfile("core_tb_p2.vcd"); $dumpvars(1, core_tb_p2); end   // top-level controller signals only (the full design is ~GBs)

  $sformat(fname, "%0s/config.txt", dir);
  fd = $fopen(fname, "r");
  r = $fscanf(fd, "%d %d %d", len_nij, len_onij, n_frames);
  $fclose(fd);
  if (len_nij != 324 || len_onij != 256) begin $display("this tb assumes a 16x16 frame (pad 18x18)"); $finish; end
  $display("dir=%0s  mode=%0s  nij=%0d onij=%0d frames=%0d", dir, diff ? "differential" : "conventional",
           len_nij, len_onij, n_frames);

  do_reset(11); cyc;
  mon_en = 1;

  // DRAM -> xmem: all W[kij] (kept in xmem for the whole run)
  c0 = cycle;
  for (kij = 0; kij < len_kij; kij = kij + 1) begin
    $sformat(fname, "%0s/weight_kij%0d.txt", dir, kij);
    file_to_xmem(fname, w_base + kij*col, col);
  end
  c_l1 = cycle - c0;

  if (diff) begin
    // DRAM -> xmem: X0 nibbles and the three deltas (generated by the host = testbench)
    c0 = cycle;
    act_to_xmem("x0hi", 0); act_to_xmem("x0lo", 1);
    act_to_xmem("d1", 2);   act_to_xmem("d2", 3);   act_to_xmem("d3", 4);
    c_l1 = c_l1 + cycle - c0;
    n_pass = 5;
    set_pass(0, 0, 1, 0,            1, 1);   // X0_hi: signed, Y0 = raw << 4 (first writer of Y0)
    set_pass(1, 1, 0, 0,            0, 0);   // X0_lo: unsigned, Y0 += raw
    set_pass(2, 2, 1, 1*len_onij,   0, 1);   // dX1 -> D1
    set_pass(3, 3, 1, 2*len_onij,   0, 1);   // dX2 -> D2
    set_pass(4, 4, 1, 3*len_onij,   0, 1);   // dX3 -> D3
    c0 = cycle;
    for (kij = 0; kij < len_kij; kij = kij + 1) kij_phase(kij);
    c_compute = cycle - c0;
    c0 = cycle;
    reconstruct;
    c_recon = cycle - c0;
  end
  else begin
    n_pass = 2;
    for (f = 0; f < n_frames; f = f + 1) begin
      c0 = cycle;
      $sformat(junk, "x%0dhi", f); act_to_xmem(junk, 0);
      $sformat(junk, "x%0dlo", f); act_to_xmem(junk, 1);
      c_l1 = c_l1 + cycle - c0;
      pw = (f == 0) ? 0 : out_base + (f - 1)*len_onij;
      set_pass(0, 0, 1, pw, 1, 1);
      set_pass(1, 1, 0, pw, 0, 0);
      c0 = cycle;
      for (kij = 0; kij < len_kij; kij = kij + 1) kij_phase(kij);
      c_compute = c_compute + cycle - c0;
    end
  end

  cyc; cyc; mon_en = 0;   // let the last write-back land inside the window
  cyc; cyc;   // let the last pmem write land before the memory is inspected
  if ($test$plusargs("dump")) begin
    fd = $fopen("pmem_dump_p2.txt", "w");
    for (o = 0; o < 2048; o = o + 1) $fdisplay(fd, "%128b", core_instance.pmem_instance.memory[o]);
    $fclose(fd);
  end
  // compare Y0 (stored at 0) and Y1..Y3 (out_base ..) with the expected files
  for (f = 0; f < n_frames; f = f + 1) begin
    $sformat(fname, "%0s/y%0d.txt", dir, f);
    open_skip3(fname, fd);
    pw = (f == 0) ? 0 : out_base + (f - 1)*len_onij;
    for (o = 0; o < len_onij; o = o + 1) begin
      r = $fscanf(fd, "%128b", answer);
      if (core_instance.pmem_instance.memory[pw + o] !== answer) begin
        n_mismatch = n_mismatch + 1;
        if (n_mismatch <= 5) begin
          $display("Y%0d o_nij %0d ERROR", f, o);
          $display("  pmem  : %128b", core_instance.pmem_instance.memory[pw + o]);
          $display("  answer: %128b", answer);
        end
      end
    end
    $fclose(fd);
  end

  $display("outputs checked: %0d frames x %0d vectors (%0d values), mismatches: %0d, control errors: %0d",
           n_frames, len_onij, n_frames*len_onij*col, n_mismatch, error);
  $display("passes: %0d (%0d per kij x %0d kij%0s) | PE weight loads: %0d | execute cycles: %0d",
           n_passes, n_pass, len_kij, diff ? "" : " x 4 frames", n_wloads, n_exec);
  $display("cycles: DRAM->xmem %0d | compute (kernel load + exec + OFIFO->SFU->pmem) %0d | reconstruction %0d | total %0d",
           c_l1, c_compute, c_recon, cycle);
  mon_report;
  if (n_mismatch == 0 && error == 0) $display("RESULT: PASS");
  else                               $display("RESULT: FAIL");
  #10 $finish;
end

endmodule
