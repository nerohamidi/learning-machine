// Part 2: psum SRAM with one read port and one write port (simple dual-port, as an FPGA block RAM or a
// 1R1W SRAM macro). It lets the SFU read psum[o] for the next element while it writes back the previous
// one, so the OFIFO -> SFU -> pmem accumulation runs at one element per cycle.
// Same timing as sram_128b_w2048: the read address is registered, Q = memory[registered address].
module sram_128b_w2048_1r1w (CLK, CENR, AR, Q, CENW, AW, D);

  input  CLK;
  input  CENR;           // read enable, active low
  input  [10:0] AR;
  output [127:0] Q;
  input  CENW;           // write enable, active low
  input  [10:0] AW;
  input  [127:0] D;
  parameter num = 2048;

  reg [127:0] memory [num-1:0];
  reg [10:0] add_q;
  assign Q = memory[add_q];

  always @ (posedge CLK) begin
    if (!CENR)
      add_q <= AR;
    if (!CENW)
      memory[AW] <= D;
  end

endmodule
