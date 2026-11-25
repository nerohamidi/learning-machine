// corelet: everything except the SRAMs (this is the block mapped onto the FPGA).
//
//   xmem Q (32b) --> L0    (8 x fifo_depth64, row-wise skewed read)    --> mac_array west input
//   wmem Q (32b) --> IFIFO (8 x fifo_depth64, column-wise skewed read) --> mac_array north input (OS only)
//   mac_array south output + per-column valid --> OFIFO (8 x fifo_depth64, 16b) --> ofifo_out
//   pmem Q (128b) / OFIFO --> SFU (accumulate, ReLU, divider, requantiser) --> sfp_out
//
// Weight-stationary (os = 0): weights enter from the west through L0, the IFIFO is idle and the array's
// north input is 0 (Part 1 / Part 2 behaviour, bit-identical).
// Output-stationary (os = 1): activations from L0 (west), weights from the IFIFO (north), results drained
// down the columns into the OFIFO with `drain`.
//
// Part 2 controls: act_signed, src_ofifo, shift4, add_q, acc_ld (see sfu.v)
// +alpha controls: os, drain, zg (zero-guard gating), wuns (weight unsigned), ififo_wr, ififo_rd,
//                  norm / amax_en / rq / nib_sel (linear-attention SFU functions, see sfu.v)
// sfp_out is the only 128-bit output: it carries the raw OFIFO vector in cycles that write raw psums to pmem
// (wr_raw = 1) and the SFU result otherwise, so pmem's D input is sfp_out (473 -> 346 FPGA pins).
module corelet (clk, reset, load, execute, l0_wr, l0_rd, ofifo_rd, acc, relu,
                act_signed, src_ofifo, shift4, add_q, acc_ld,
                os, drain, zg, wuns, ififo_wr, ififo_rd, norm, amax_en, rq, nib_sel,
                wr_raw, l0_in, ififo_in, pmem_q, ofifo_valid, sfp_out);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  input  load, execute, l0_wr, l0_rd, ofifo_rd, acc, relu;
  input  act_signed, src_ofifo, shift4, add_q, acc_ld;
  input  os, drain, zg, wuns, ififo_wr, ififo_rd, norm, amax_en, rq, nib_sel;
  input  wr_raw;   // this cycle writes a raw OFIFO vector to pmem: sfp_out = OFIFO output (else the SFU output)
  input  [row*bw-1:0]      l0_in;
  input  [col*bw-1:0]      ififo_in;
  input  [psum_bw*col-1:0] pmem_q;
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire [row*bw-1:0]      l0_out;
  wire [col*bw-1:0]      ififo_out;
  wire [psum_bw*col-1:0] array_out;
  wire [psum_bw*col-1:0] ofifo_out;
  wire [psum_bw*col-1:0] sfu_out;
  wire [col-1:0]         array_valid;
  wire l0_full, l0_ready, ififo_full, ififo_ready, ofifo_full, ofifo_ready;

  l0 #(.row(row), .bw(bw)) l0_instance (
    .clk(clk), .reset(reset),
    .in(l0_in), .out(l0_out),
    .rd(l0_rd), .wr(l0_wr),
    .o_full(l0_full), .o_ready(l0_ready));

  // IFIFO: same structure as L0 (one FIFO per column, column c read one cycle after column c-1),
  // which matches the one-cycle-per-column skew of the execute instruction along each row.
  l0 #(.row(col), .bw(bw)) ififo_instance (
    .clk(clk), .reset(reset),
    .in(ififo_in), .out(ififo_out),
    .rd(ififo_rd), .wr(ififo_wr),
    .o_full(ififo_full), .o_ready(ififo_ready));

  mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) mac_array_instance (
    .clk(clk), .reset(reset),
    .in_w(l0_out), .in_wt(ififo_out),
    .inst_w({act_signed, execute, load}),
    .out_s(array_out), .valid(array_valid),
    .os(os), .drain(drain), .zg(zg), .wuns(wuns));

  ofifo #(.col(col), .bw(psum_bw)) ofifo_instance (
    .clk(clk), .reset(reset),
    .in(array_out), .out(ofifo_out),
    .wr(array_valid), .rd(ofifo_rd),
    .o_full(ofifo_full), .o_ready(ofifo_ready), .o_valid(ofifo_valid));

  sfu #(.psum_bw(psum_bw), .col(col)) sfu_instance (
    .clk(clk), .reset(reset),
    .acc(acc), .relu(relu),
    .in(pmem_q), .out(sfu_out),
    .ofifo_in(ofifo_out), .src_ofifo(src_ofifo), .shift4(shift4), .add_q(add_q), .acc_ld(acc_ld),
    .norm(norm), .amax_en(amax_en), .rq(rq), .nib_sel(nib_sel));

  // single 128-bit write-data port to pmem (the pmem D mux used to sit in core.v and needed both buses as pins)
  assign sfp_out = wr_raw ? ofifo_out : sfu_out;

endmodule
