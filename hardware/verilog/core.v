// core: corelet + scratchpads.
//   xmem: sram_32b_w2048        activations and weights (WS), activations (OS)
//   wmem: sram_32b_w2048        +alpha: weights streamed into the IFIFO (OS), in parallel with xmem -> L0
//   pmem: sram_128b_w2048_1r1w  psums (Part 1: raw psums at kij*len_nij + nij, outputs at 1024 + o_nij;
//                               Part 2: Y0 / D_i / Y_i accumulators, see core_tb_p2.v)
//
// Instruction word (driven by the testbench, which acts as the controller):
//   inst[35]    pmem_wsel : pmem D source, 0 = OFIFO output (raw psums), 1 = SFU output (write-back)
//   inst[34]    relu      : SFU output = max(., 0)
//   inst[33]    acc       : SFU accumulates pmem Q
//   inst[32]    CEN_pmem    inst[31] WEN_pmem    inst[30:20] A_pmem
//   inst[19]    CEN_xmem    inst[18] WEN_xmem    inst[17:7]  A_xmem
//   inst[6]     ofifo_rd    inst[5]  ififo_wr    inst[4] ififo_rd
//   inst[3]     l0_rd       inst[2]  l0_wr       inst[1] execute    inst[0] load
// Part 2 extension (all 0 = Part 1 behaviour):
//   inst[36]    act_signed  PE activation sign mode (travels with execute)
//   inst[37]    src_ofifo   SFU adder operand A = OFIFO output (else acc_q)
//   inst[38]    shift4      operand A << 4 (X0 high-nibble pass)
//   inst[39]    add_q       SFU adder operand B = pmem Q (else 0)
//   inst[40]    acc_ld      acc_q <= pmem Q
//   inst[41]    CEN_pmem_rd second pmem address (read port, active low), used while inst[32:20] writes
//   inst[52:42] A_pmem_rd
// +alpha extension (all 0 = Part 1 / Part 2 behaviour):
//   inst[53]    os          1 = output-stationary dataflow, 0 = weight stationary
//   inst[54]    drain       OS: shift the PE accumulators down into the OFIFO
//   inst[55]    zg          zero-guard operand gating enable
//   inst[56]    wmem_ce     wmem chip enable (active high)   inst[57] wmem_we (active high)
//   inst[68:58] A_wmem
//   inst[69]    rq          SFU outputs the requantised nibble word; xmem D = sfp_out[31:0] (on-chip write-back)
//   inst[70]    wuns        PE weight unsigned
//   inst[71]    norm        SFU divider (linear-attention normalisation)
//   inst[72]    amax_en     SFU max-|.| (OR) register update
//   inst[73]    nib_sel     requantiser nibble: 1 = high, 0 = low
// Bits [33:0] are the template's (core_tb.v).
module core (clk, inst, ofifo_valid, D_xmem, sfp_out, reset);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  input  [73:0] inst;
  input  [bw*row-1:0] D_xmem;
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire [bw*row-1:0]      xmem_q;
  wire [bw*col-1:0]      wmem_q;
  wire [psum_bw*col-1:0] pmem_q;
  wire [bw*row-1:0]      xmem_d = inst[69] ? sfp_out[bw*row-1:0] : D_xmem;

  // pmem ports: the template fields (CEN, WEN, A) read (WEN=1) or write (WEN=0); a Part 2 read can be
  // issued through the second address field in the same cycle as a template-field write.
  wire       legacy_rd = !inst[32] &  inst[31];
  wire       legacy_wr = !inst[32] & !inst[31];
  wire       pmem_cenr = !(legacy_rd | !inst[41]);
  wire [10:0] pmem_ar  = legacy_rd ? inst[30:20] : inst[52:42];
  // pmem D source (inst[35]): 0 = raw OFIFO vector, 1 = SFU result. The mux is inside the corelet (sfp_out).
  wire       wr_raw    = legacy_wr & !inst[35];
  wire [psum_bw*col-1:0] pmem_d = sfp_out;

  sram_32b_w2048 xmem_instance (
    .CLK(clk), .D(xmem_d), .Q(xmem_q),
    .CEN(inst[19]), .WEN(inst[18]), .A(inst[17:7]));

  sram_32b_w2048 wmem_instance (
    .CLK(clk), .D(D_xmem), .Q(wmem_q),
    .CEN(!inst[56]), .WEN(!inst[57]), .A(inst[68:58]));

  sram_128b_w2048_1r1w pmem_instance (
    .CLK(clk), .Q(pmem_q), .CENR(pmem_cenr), .AR(pmem_ar),
    .D(pmem_d), .CENW(!legacy_wr), .AW(inst[30:20]));

  corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance (
    .clk(clk), .reset(reset),
    .load(inst[0]), .execute(inst[1]), .l0_wr(inst[2]), .l0_rd(inst[3]),
    .ofifo_rd(inst[6]), .acc(inst[33]), .relu(inst[34]),
    .act_signed(inst[36]), .src_ofifo(inst[37]), .shift4(inst[38]), .add_q(inst[39]), .acc_ld(inst[40]),
    .os(inst[53]), .drain(inst[54]), .zg(inst[55]), .wuns(inst[70]),
    .ififo_wr(inst[5]), .ififo_rd(inst[4]),
    .norm(inst[71]), .amax_en(inst[72]), .rq(inst[69]), .nib_sel(inst[73]),
    .wr_raw(wr_raw), .l0_in(xmem_q), .ififo_in(wmem_q), .pmem_q(pmem_q),
    .ofifo_valid(ofifo_valid), .sfp_out(sfp_out));

endmodule
