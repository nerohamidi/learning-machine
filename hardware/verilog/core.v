module core (clk, reset, inst, D_xmem, ofifo_valid, sfp_out);
 
  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;
 
  input  clk;
  input  reset;
  input  [33:0] inst;
  input  [bw*row-1:0] D_xmem;          // activation / weight data from the testbench
  output ofifo_valid;
  output [col*psum_bw-1:0] sfp_out;
 
  // xmem (activation & weight SRAM): inst[19] CEN, inst[18] WEN, inst[17:7] A
  wire [bw*row-1:0] xmem_q;            // xmem -> L0
 
  // pmem (psum SRAM): inst[32] CEN, inst[31] WEN, inst[30:20] A
  wire [col*psum_bw-1:0] pmem_d;       // OFIFO -> pmem
  wire [col*psum_bw-1:0] pmem_q;       // pmem -> SFU
 
  sram_32b_w2048 xmem_instance (
    .CLK(clk),
    .D(D_xmem),
    .Q(xmem_q),
    .CEN(inst[19]),
    .WEN(inst[18]),
    .A(inst[17:7]));
 
  sram_128b_w2048 pmem_instance (
    .CLK(clk),
    .D(pmem_d),
    .Q(pmem_q),
    .CEN(inst[32]),
    .WEN(inst[31]),
    .A(inst[30:20]));
 
  corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance (
    .clk(clk),
    .reset(reset),
    .inst(inst),
    .l0_in(xmem_q),
    .ofifo_out(pmem_d),
    .ofifo_valid(ofifo_valid),
    .sfp_in(pmem_q),
    .sfp_out(sfp_out));
 
endmodule
 