// core: corelet + scratchpads.
//   xmem: sram_32b_w2048  activations (addr 0..) and weights (addr 1024..)
//   pmem: sram_128b_w2048 raw psums (addr kij*len_nij + nij) and outputs (addr 1024 + o_nij)
//
// Instruction word (driven by the testbench, which acts as the controller):
//   inst[35]    pmem_wsel : pmem D source, 0 = OFIFO output (raw psums), 1 = SFU output (write-back)
//   inst[34]    relu      : SFU output = max(acc, 0)
//   inst[33]    acc       : SFU accumulates pmem Q
//   inst[32]    CEN_pmem    inst[31] WEN_pmem    inst[30:20] A_pmem
//   inst[19]    CEN_xmem    inst[18] WEN_xmem    inst[17:7]  A_xmem
//   inst[6]     ofifo_rd    inst[5]  ififo_wr (unused)   inst[4] ififo_rd (unused)
//   inst[3]     l0_rd       inst[2]  l0_wr       inst[1] execute    inst[0] load
// Bits [33:0] are the template's (core_tb.v); bits 35:34 are added for ReLU and SFU write-back.
module core (clk, inst, ofifo_valid, D_xmem, sfp_out, reset);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  input  [35:0] inst;
  input  [bw*row-1:0] D_xmem;
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire [bw*row-1:0]      xmem_q;
  wire [psum_bw*col-1:0] pmem_q;
  wire [psum_bw*col-1:0] ofifo_out;
  wire [psum_bw*col-1:0] pmem_d = inst[35] ? sfp_out : ofifo_out;

  sram_32b_w2048 xmem_instance (
    .CLK(clk), .D(D_xmem), .Q(xmem_q),
    .CEN(inst[19]), .WEN(inst[18]), .A(inst[17:7]));

  sram_128b_w2048 pmem_instance (
    .CLK(clk), .D(pmem_d), .Q(pmem_q),
    .CEN(inst[32]), .WEN(inst[31]), .A(inst[30:20]));

  corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance (
    .clk(clk), .reset(reset),
    .load(inst[0]), .execute(inst[1]), .l0_wr(inst[2]), .l0_rd(inst[3]),
    .ofifo_rd(inst[6]), .acc(inst[33]), .relu(inst[34]),
    .l0_in(xmem_q), .pmem_q(pmem_q),
    .ofifo_out(ofifo_out), .ofifo_valid(ofifo_valid), .sfp_out(sfp_out));

endmodule
