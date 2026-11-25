// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
//
// +alpha: one PE that runs either dataflow, selected at run time by `os`, plus zero-guard operand gating.
//
//   WS (os = 0, Part 1/2 behaviour): the weight is loaded from the west and kept in b_q; activations flow
//      west -> east; the psum flows north -> south:  out_s = a_q * b_q + c_q,  c_q <= in_n every cycle.
//   OS (os = 1): activations flow west -> east, weights flow north -> south on the low bw+1 bits of the
//      (otherwise idle) psum wires, and the psum stays in c_q:  c_q <= c_q + a_q * b_q.
//      drain = 1 (broadcast): c_q becomes a shift register down the column (c_q <= in_n, out_s = c_q),
//      so the bottom row emits one row of results per cycle into the OFIFO; zeros shift in from the top,
//      which also clears the accumulators for the next tile.
//   Zero guard: every 4-bit operand travels with a guard bit (in_w[bw], in_n[bw]) set at the array edge
//      when zero gating is enabled and the value is 0. A guarded operand is NOT written into a_q / b_q
//      (the operand registers and the multiplier inputs do not toggle), and the product is bypassed:
//      WS: out_s = c_q (a_q held + b_q stationary -> the multiplier inputs do not change at all);
//      OS: the guarded operand is isolated to 0 at the multiplier input and the accumulator register is not
//      enabled (clock-gated) for that cycle.
//      Results are bit-identical with gating on or off.
module mac_tile (clk, out_s, in_w, out_e, in_n, inst_w, inst_e, reset, os, drain, wuns);

parameter bw = 4;
parameter psum_bw = 16;

output [psum_bw-1:0] out_s;
input  [bw:0]   in_w;   // {zero guard, activation / weight being loaded}
output [bw:0]   out_e;
input  [2:0] inst_w;    // inst[2]: activation signed (Part 2), inst[1]: execute, inst[0]: kernel loading
output [2:0] inst_e;
input  [psum_bw-1:0] in_n;  // WS: psum from the north;  OS: {.., zero guard, weight} (drain: psum)
input  clk;
input  reset;
input  os;     // 1 = output stationary
input  drain;  // OS only: shift the accumulators down the column
input  wuns;   // weight unsigned

// latches (edge-triggered registers)
reg [2:0]         inst_q;        // instruction passed to the east (inst_q[2] = sign mode of a_q)
reg [bw-1:0]      a_q;           // activation
reg [bw-1:0]      b_q;           // weight (WS: stationary, OS: streamed from the north)
reg [psum_bw-1:0] c_q;           // WS: psum from the north;  OS: the stationary output accumulator
reg               load_ready_q;  // 1: this PE has not loaded its weight yet
reg               za_q;          // guard: a_q stands for 0 (a_q itself was not updated)
reg               zb_q;          // guard: b_q stands for 0 (OS streamed weight)

wire          za_w = in_w[bw];
wire [bw-1:0] a_w  = in_w[bw-1:0];
wire          zb_w = in_n[bw];
wire [bw-1:0] b_w  = in_n[bw-1:0];

wire [psum_bw-1:0] mac_out;      // a_q * b_q + c_q

assign out_e  = {za_q, a_q};
assign inst_e = inst_q;
assign out_s  = os ? (drain ? c_q : {{(psum_bw-bw-1){1'b0}}, zb_q, b_q})
                   : (za_q  ? c_q : mac_out);

// OS operand isolation: a guarded operand enters the multiplier as 0, so the multiplier does not switch while
// the other operand keeps streaming (WS needs no isolation: b_q is stationary and a_q is held -> product frozen)
wire [bw-1:0] a_m = (os & za_q) ? {bw{1'b0}} : a_q;
wire [bw-1:0] b_m = (os & zb_q) ? {bw{1'b0}} : b_q;

mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
        .a(a_m),
        .b(b_m),
        .c(c_q),
        .sgn(inst_q[2]),
        .wuns(wuns),
        .out(mac_out)
);

always @ (posedge clk) begin
  if (reset == 1) begin
    inst_q       <= 3'b000;
    load_ready_q <= 1'b1;
    a_q          <= {bw{1'b0}};
    b_q          <= {bw{1'b0}};
    c_q          <= {psum_bw{1'b0}};
    za_q         <= 1'b0;
    zb_q         <= 1'b0;
  end
  else begin
    // execution instruction always flows to the east
    inst_q[1] <= inst_w[1];
    // Part 2: the activation sign mode travels with the activation (latched in the same cycle as a_q)
    inst_q[2] <= inst_w[2];

    // accept a new activation/weight when executing or loading a kernel (a guarded zero is not written)
    if (inst_w[0] | inst_w[1]) begin
      za_q <= za_w;
      if (!za_w) a_q <= a_w;
    end

    // kernel load (WS): keep the first value that arrives while this PE is ready
    if (inst_w[0] & load_ready_q) begin
      b_q          <= za_w ? {bw{1'b0}} : a_w;
      load_ready_q <= 1'b0;
    end

    // forward the kernel-load instruction only after this PE has loaded its weight
    if (load_ready_q == 1'b0)
      inst_q[0] <= inst_w[0];

    // OS: the weight arrives from the north in the same cycle as the activation from the west
    if (os & inst_w[1]) begin
      zb_q <= zb_w;
      if (!zb_w) b_q <= b_w;
    end

    // psum register
    if (!os | drain)
      c_q <= in_n;                                  // WS: psum from the north / OS: drain shift
    else if (inst_q[1] & ~(za_q | zb_q))
      c_q <= mac_out;                               // OS: accumulate (enable off for a guarded zero)
  end
end

endmodule
