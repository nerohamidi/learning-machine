// +alpha: hardware linear attention (one transformer layer, all heads) on the WS/OS-switchable core.
// Math and data layout: software/transformer/hw_attn.py and gen_attn_stimulus.py.
//
// Per head (N tokens, d_qk = 8, d_v = 7 + the constant-1 lane):
//   phase A  (OS, os=1): S^T = V1^T phiK. L0 <- V1 rows (signed, act_signed=1), IFIFO <- phiK rows (wuns=1),
//            N steps, then drain: OFIFO vector j = PE row 7-j = S[:, 7-j] -> SFU (amax_en: OR of |S|) -> pmem PS+7-j
//   phase R  requantiser, on chip: for t = 0..7 read pmem PS+t, SFU rq -> S8 nibble word -> xmem XS_HI+t / XS_LO+t
//            (the drained vector S[:, t] is exactly the WS kernel word of PE column t: no transpose)
//   phase B  (WS, os=0): hi pass: load S8_hi (signed), stream phiQ (unsigned), OFIFO<<4 -> pmem PP+n
//            lo pass: load S8_lo (wuns=1), stream phiQ, (OFIFO + pmem PP+n) -> divider (norm) -> pmem PO+n
// Every intermediate is checked: S (pmem), S8 hi/lo words (xmem), outputs + normaliser (pmem).
// plusargs: +dir=<stimulus dir>  +zg=0/1  +vcd
`timescale 1ns/1ps

module core_tb_attn;

parameter bw = 4;
parameter psum_bw = 16;
parameter col = 8;
parameter row = 8;
parameter drain_lat = 16;
parameter XV = 0, XQ = 64, XS_HI = 128, XS_LO = 136;     // xmem map
parameter PS = 0, PP = 16, PO = 1024;                     // pmem map (outputs of head h at PO + h*N)

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
reg CEN_pmem_rd = 1, CEN_pmem_rd_q = 1;
reg [10:0] A_pmem_rd = 0, A_pmem_rd_q = 0;
reg ofifo_rd = 0, ofifo_rd_q = 0;
reg l0_rd = 0, l0_rd_q = 0, l0_wr = 0, l0_wr_q = 0;
reg ififo_rd = 0, ififo_rd_q = 0, ififo_wr = 0, ififo_wr_q = 0;
reg execute = 0, execute_q = 0, load = 0, load_q = 0;
reg pmem_wsel = 0, pmem_wsel_q = 0;
reg act_signed = 0, act_signed_q = 0, src_ofifo = 0, src_ofifo_q = 0, shift4 = 0, shift4_q = 0;
reg add_q = 0, add_q_q = 0;
reg os = 0, os_q = 0, drain = 0, drain_q = 0, wuns = 0, wuns_q = 0, rq = 0, rq_q = 0;
reg norm = 0, norm_q = 0, amax_en = 0, amax_en_q = 0, nib_sel = 0, nib_sel_q = 0;
reg zg = 0;

wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

assign inst_q[73] = nib_sel_q;
assign inst_q[72] = amax_en_q;
assign inst_q[71] = norm_q;
assign inst_q[70] = wuns_q;
assign inst_q[69] = rq_q;
assign inst_q[68:58] = A_wmem_q;
assign inst_q[57] = wmem_we_q;
assign inst_q[56] = wmem_ce_q;
assign inst_q[55] = zg;
assign inst_q[54] = drain_q;
assign inst_q[53] = os_q;
assign inst_q[52:42] = A_pmem_rd_q;
assign inst_q[41] = CEN_pmem_rd_q;
assign inst_q[40] = 1'b0;
assign inst_q[39] = add_q_q;
assign inst_q[38] = shift4_q;
assign inst_q[37] = src_ofifo_q;
assign inst_q[36] = act_signed_q;
assign inst_q[35] = pmem_wsel_q;
assign inst_q[34] = 1'b0;
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
assign inst_q[0] = load_q;

core #(.bw(bw), .col(col), .row(row)) core_instance (
  .clk(clk), .inst(inst_q), .ofifo_valid(ofifo_valid),
  .D_xmem(D_xmem_q), .sfp_out(sfp_out), .reset(reset));

always @ (posedge clk) begin
  D_xmem_q <= D_xmem;  CEN_xmem_q <= CEN_xmem;  WEN_xmem_q <= WEN_xmem;  A_xmem_q <= A_xmem;
  wmem_ce_q <= wmem_ce;  wmem_we_q <= wmem_we;  A_wmem_q <= A_wmem;
  CEN_pmem_q <= CEN_pmem;  WEN_pmem_q <= WEN_pmem;  A_pmem_q <= A_pmem;
  CEN_pmem_rd_q <= CEN_pmem_rd;  A_pmem_rd_q <= A_pmem_rd;
  ofifo_rd_q <= ofifo_rd;  l0_rd_q <= l0_rd;  l0_wr_q <= l0_wr;  ififo_rd_q <= ififo_rd;  ififo_wr_q <= ififo_wr;
  execute_q <= execute;  load_q <= load;  pmem_wsel_q <= pmem_wsel;
  act_signed_q <= act_signed;  src_ofifo_q <= src_ofifo;  shift4_q <= shift4;  add_q_q <= add_q;
  os_q <= os;  drain_q <= drain;  wuns_q <= wuns;  rq_q <= rq;  norm_q <= norm;  amax_en_q <= amax_en;
  nib_sel_q <= nib_sel;
end

integer cycle = 0;
always @ (posedge clk) cycle = cycle + 1;

`include "toggle_mon.vh"

reg [8*256:1] dir, fname;
integer n_heads, N, h, fd, r, i, error, n_mismatch;
integer c0, c_l1 = 0, c_a = 0, c_r = 0, c_b = 0;
reg [psum_bw*col-1:0] answer;
reg [bw*row-1:0] answer32;
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

// DRAM -> xmem (which = 0) or wmem (which = 1)
task file_to_mem (input [8*256:1] name, input integer which, input integer n, input integer base);
  integer fh, q;
  begin
    open_skip3(name, fh);
    for (q = 0; q < n; q = q + 1) begin
      neg; r = $fscanf(fh, "%32b", D_xmem);
      if (which == 0) begin CEN_xmem = 0; WEN_xmem = 0; A_xmem = base + q; end
      else            begin wmem_ce = 1;  wmem_we = 1;  A_wmem = base + q; end
      pos;
    end
    neg; CEN_xmem = 1; WEN_xmem = 1; wmem_ce = 0; wmem_we = 0;
    pos;
    $fclose(fh);
  end
endtask

// ---------------- phase A: S^T = V1^T phiK, output stationary ----------------
task phase_a;
  integer t;
  begin
    neg; os = 1; wuns = 1; pos;
    do_reset(1);                                  // clear the accumulators (the array was in WS mode)
    for (t = 0; t < N + 2; t = t + 1) begin
      neg;
      if (t < N) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = XV + t; wmem_ce = 1; wmem_we = 0; A_wmem = t; end
      else       begin CEN_xmem = 1; WEN_xmem = 1; wmem_ce = 0; end
      l0_wr = (t >= 1 && t <= N);  ififo_wr = l0_wr;
      execute = (t >= 2);  l0_rd = execute;  ififo_rd = execute;  act_signed = execute;
      pos;
    end
    neg; execute = 0; l0_rd = 0; ififo_rd = 0; l0_wr = 0; ififo_wr = 0; act_signed = 0; pos;
    for (t = 1; t < drain_lat; t = t + 1) cyc;
    for (t = 0; t < row + 2; t = t + 1) begin
      neg;
      drain = (t < row);
      ofifo_rd = (t >= 1 && t <= row);
      CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; amax_en = 0;
      if (t >= 2) begin
        CEN_pmem = 0; WEN_pmem = 0; A_pmem = PS + row - 1 - (t - 2);
        pmem_wsel = 1; src_ofifo = 1; amax_en = 1;
      end
      pos;
    end
    neg; drain = 0; ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; amax_en = 0;
    os = 0; wuns = 0;
    pos;
  end
endtask

// ---------------- phase R: S (pmem) -> requantise -> S8 nibble words (xmem), on chip ----------------
task phase_r (input integer sel, input integer base);
  integer t;
  begin
    for (t = 0; t <= col; t = t + 1) begin
      neg;
      CEN_pmem_rd = (t < col) ? 0 : 1;  A_pmem_rd = PS + t;
      if (t >= 1) begin CEN_xmem = 0; WEN_xmem = 0; A_xmem = base + t - 1; rq = 1; nib_sel = sel; end
      pos;
    end
    neg; CEN_pmem_rd = 1; CEN_xmem = 1; WEN_xmem = 1; rq = 0; nib_sel = 0; pos;
  end
endtask

// ---------------- phase B pass: WS, kernel = S8 nibble words, activations = phiQ ----------------
// lo = 0: hi pass,  pmem[PP+n]  = OFIFO << 4                       (written the cycle after the OFIFO read)
// lo = 1: lo pass,  pmem[out+n] = norm(OFIFO + pmem[PP+n])          (weights unsigned). The divider is a 2-stage
//         pipeline: `norm` marks the cycle its input (OFIFO + pmem Q) is taken, the result is written 2 cycles later.
task ws_pass (input integer w_addr, input integer lo, input integer out_addr);
  integer L0s, E, done, rd_cnt, started, t, pend, pend_addr, q1, q1_addr, q2, q2_addr;
  begin
    neg; wuns = lo; pos;
    do_reset(1);
    L0s = col + 1;
    E = L0s + 2*col + row + 2 + 1;
    rd_cnt = 0; started = 0; done = 0; pend = 0; q1 = 0; q2 = 0;
    t = 0;
    while (!done) begin
      neg;
      if (t < col)          begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = w_addr + t; end
      else if (t < col + N) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = XQ + t - col; end
      else                  begin CEN_xmem = 1; WEN_xmem = 1; end
      l0_wr   = (t > 0 && t <= col + N);
      load    = (t >= L0s && t < L0s + 2*col + row + 2);
      execute = (t >= E && t < E + N);
      l0_rd   = (t >= L0s && t < L0s + col) || (t >= E && t < E + N);
      CEN_pmem = 1; WEN_pmem = 1; pmem_wsel = 0; src_ofifo = 0; shift4 = 0; add_q = 0; norm = 0;
      if (pend && !lo) begin           // hi pass: write OFIFO << 4 directly
        CEN_pmem = 0; WEN_pmem = 0; A_pmem = pend_addr; pmem_wsel = 1; src_ofifo = 1; shift4 = 1;
      end
      if (pend && lo) begin            // lo pass: divider input this cycle
        src_ofifo = 1; add_q = 1; norm = 1;
      end
      if (q2) begin                    // lo pass: divider output, two cycles after its input
        CEN_pmem = 0; WEN_pmem = 0; A_pmem = q2_addr; pmem_wsel = 1;
      end
      q2 = q1; q2_addr = q1_addr;
      q1 = pend && lo; q1_addr = pend_addr;
      pend = 0;
      if (!started && t > E && ofifo_valid === 1'b1) started = 1;
      ofifo_rd = (started && rd_cnt < N);
      CEN_pmem_rd = 1;
      if (ofifo_rd) begin
        pend = 1;
        pend_addr = lo ? out_addr + rd_cnt : PP + rd_cnt;
        if (lo) begin CEN_pmem_rd = 0; A_pmem_rd = PP + rd_cnt; end
        rd_cnt = rd_cnt + 1;
      end
      pos;
      t = t + 1;
      if (rd_cnt == N && !pend && !q1 && !q2) done = 1;
      if (t > N + 1000) begin $display("ERROR: WS pass timeout"); error = error + 1; done = 1; end
    end
    neg; CEN_pmem = 1; WEN_pmem = 1; CEN_pmem_rd = 1; CEN_xmem = 1; WEN_xmem = 1; ofifo_rd = 0; l0_wr = 0;
    pmem_wsel = 0; src_ofifo = 0; shift4 = 0; add_q = 0; norm = 0; wuns = 0;
    pos;
  end
endtask

task check_pmem (input [8*64:1] what, input integer base, input integer n);
  integer fh, q;
  begin
    $sformat(fname, "%0s/h%0d_%0s.txt", dir, h, what);
    open_skip3(fname, fh);
    for (q = 0; q < n; q = q + 1) begin
      r = $fscanf(fh, "%128b", answer);
      if (core_instance.pmem_instance.memory[base + q] !== answer) begin
        n_mismatch = n_mismatch + 1;
        if (n_mismatch <= 6) begin
          $display("head %0d %0s line %0d ERROR", h, what, q);
          $display("  pmem  : %128b", core_instance.pmem_instance.memory[base + q]);
          $display("  answer: %128b", answer);
        end
      end
    end
    $fclose(fh);
  end
endtask

task check_xmem (input [8*64:1] what, input integer base, input integer n);
  integer fh, q;
  begin
    $sformat(fname, "%0s/h%0d_%0s.txt", dir, h, what);
    open_skip3(fname, fh);
    for (q = 0; q < n; q = q + 1) begin
      r = $fscanf(fh, "%32b", answer32);
      if (core_instance.xmem_instance.memory[base + q] !== answer32) begin
        n_mismatch = n_mismatch + 1;
        if (n_mismatch <= 6)
          $display("head %0d %0s line %0d ERROR: xmem %32b answer %32b", h, what, q,
                   core_instance.xmem_instance.memory[base + q], answer32);
      end
    end
    $fclose(fh);
  end
endtask

initial begin
  error = 0; n_mismatch = 0;
  if (!$value$plusargs("dir=%s", dir)) dir = "stim_attn_rand";
  if (!$value$plusargs("zg=%d", zg)) zg = 0;
  if ($test$plusargs("vcd")) begin $dumpfile("core_tb_attn.vcd"); $dumpvars(0, core_tb_attn); end
  $sformat(fname, "%0s/config.txt", dir);
  fd = $fopen(fname, "r");
  r = $fscanf(fd, "%d %d", n_heads, N);
  $fclose(fd);
  if (N > 64) begin $display("this tb keeps one head's tokens resident in the 64-deep L0/IFIFO"); $finish; end
  $display("dir=%0s  heads=%0d  N=%0d  zero gating=%0d", dir, n_heads, N, zg);

  do_reset(11); cyc;
  for (h = 0; h < n_heads; h = h + 1) begin
    c0 = cycle;
    $sformat(fname, "%0s/h%0d_v1.txt", dir, h);    file_to_mem(fname, 0, N, XV);
    $sformat(fname, "%0s/h%0d_phiq.txt", dir, h);  file_to_mem(fname, 0, N, XQ);
    $sformat(fname, "%0s/h%0d_phik.txt", dir, h);  file_to_mem(fname, 1, N, 0);
    c_l1 = c_l1 + cycle - c0;

    mon_en = 1;
    c0 = cycle;  phase_a;                                   c_a = c_a + cycle - c0;
    c0 = cycle;  phase_r(1, XS_HI);  phase_r(0, XS_LO);     c_r = c_r + cycle - c0;
    c0 = cycle;  ws_pass(XS_HI, 0, 0);  ws_pass(XS_LO, 1, PO + h*N);  c_b = c_b + cycle - c0;
    cyc; cyc; mon_en = 0;
    cyc;
    check_pmem("S", PS, row);
    check_xmem("s8hi", XS_HI, col);
    check_xmem("s8lo", XS_LO, col);
    check_pmem("out", PO + h*N, N);
  end

  $display("checked per head: S (8x8 16b), S8 hi/lo (2x8 words), out+den (%0d x 8 lanes); mismatches: %0d, control errors: %0d",
           N, n_mismatch, error);
  $display("cycles: DRAM->L1 %0d | phase A (OS state) %0d | requant %0d | phase B (2 WS passes + divider) %0d | total %0d",
           c_l1, c_a, c_r, c_b, cycle);
  $display("cycles per head (on-chip, excl. DRAM): %0d", (c_a + c_r + c_b) / n_heads);
  mon_report;
  if (n_mismatch == 0 && error == 0) $display("RESULT: PASS");
  else                               $display("RESULT: FAIL");
  #10 $finish;
end

endmodule
