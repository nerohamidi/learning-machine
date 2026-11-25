// corelet: everything except the SRAMs (this is the block mapped onto the FPGA).
//
//   xmem Q (32b) --> L0 (8 x fifo_depth64, row-wise read) --> mac_array 8x8 (west input)
//   mac_array south output + per-column valid --> OFIFO (8 x fifo_depth64, 16b) --> ofifo_out
//   pmem Q (128b) --> SFU (accumulate, ReLU) --> sfp_out
//
// IFIFO is not used: weights enter from the west through L0 (in_n of the array is tied to 0).
// Only the control bits that the corelet needs are ports, to keep the FPGA pin count low.
module corelet (clk, reset, load, execute, l0_wr, l0_rd, ofifo_rd, acc, relu,
                l0_in, pmem_q, ofifo_out, ofifo_valid, sfp_out);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  input  load, execute, l0_wr, l0_rd, ofifo_rd, acc, relu;
  input  [row*bw-1:0]      l0_in;
  input  [psum_bw*col-1:0] pmem_q;
  output [psum_bw*col-1:0] ofifo_out;
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire [row*bw-1:0]      l0_out;
  wire [psum_bw*col-1:0] array_out;
  wire [col-1:0]         array_valid;
  wire l0_full, l0_ready, ofifo_full, ofifo_ready;

  l0 #(.row(row), .bw(bw)) l0_instance (
    .clk(clk), .reset(reset),
    .in(l0_in), .out(l0_out),
    .rd(l0_rd), .wr(l0_wr),
    .o_full(l0_full), .o_ready(l0_ready));

  mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) mac_array_instance (
    .clk(clk), .reset(reset),
    .in_w(l0_out), .in_n({psum_bw*col{1'b0}}),
    .inst_w({execute, load}),
    .out_s(array_out), .valid(array_valid));

  ofifo #(.col(col), .bw(psum_bw)) ofifo_instance (
    .clk(clk), .reset(reset),
    .in(array_out), .out(ofifo_out),
    .wr(array_valid), .rd(ofifo_rd),
    .o_full(ofifo_full), .o_ready(ofifo_ready), .o_valid(ofifo_valid));

  sfu #(.psum_bw(psum_bw), .col(col)) sfu_instance (
    .clk(clk), .reset(reset),
    .acc(acc), .relu(relu),
    .in(pmem_q), .out(sfp_out));

endmodule
