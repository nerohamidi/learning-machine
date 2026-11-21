module corelet (clk, reset, inst, l0_in, ofifo_out, ofifo_valid, sfp_in, sfp_out);
 
  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;
 
  input  clk;
  input  reset;
  input  [33:0] inst;                 // full instruction from core_tb
  input  [row*bw-1:0] l0_in;          // from xmem Q
  output [col*psum_bw-1:0] ofifo_out; // to pmem D
  output ofifo_valid;
  input  [col*psum_bw-1:0] sfp_in;    // from pmem Q
  output [col*psum_bw-1:0] sfp_out;
 
  wire [row*bw-1:0]      l0_out;      // L0 -> array (west input)
  wire [col*psum_bw-1:0] array_out;   // array -> OFIFO (south output)
  wire [col-1:0]         array_valid; // array -> OFIFO write enables
 
  l0 #(.row(row), .bw(bw)) l0_instance (
    .clk(clk), .reset(reset),
    .in(l0_in), .out(l0_out),
    .rd(inst[3]), .wr(inst[2]),
    .o_full(), .o_ready());           // not needed at corelet ports
 
  mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) mac_array_instance (
    .clk(clk), .reset(reset),
    .in_w(l0_out), .in_n({col*psum_bw{1'b0}}),
    .inst_w(inst[1:0]),
    .out_s(array_out), .valid(array_valid));
 
  ofifo #(.col(col), .bw(psum_bw)) ofifo_instance (
    .clk(clk), .reset(reset),
    .in(array_out), .out(ofifo_out),
    .rd(inst[6]), .wr(array_valid),
    .o_valid(ofifo_valid),
    .o_full(), .o_ready());
 
  sfp #(.col(col), .psum_bw(psum_bw)) sfp_instance (
    .clk(clk), .reset(reset),
    .acc(inst[33]), .in(sfp_in), .out(sfp_out));
 
endmodule
 