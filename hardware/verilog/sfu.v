// Special function unit: one accumulator + one adder + ReLU per array column.
//
// Part 1 (unchanged when the Part 2 / +alpha controls are 0):
//   acc  = 1 : acc_q <= acc_q + in  (in = psum read from the psum SRAM, one 16-bit word per column)
//   relu = 1 : out = max(acc_q, 0), otherwise out = acc_q
//   reset    : acc_q <= 0 (the controller resets the SFU before each output pixel)
//
// Part 2 (differential video frames):
//   out = relu( A + B )
//     A = src_ofifo ? (shift4 ? ofifo_in << 4 : ofifo_in) : acc_q
//     B = add_q     ? in (pmem read port)                : 0
//   * stream accumulation (src_ofifo=1, add_q=1): psum[o] <= psum[o] + raw psum straight from the OFIFO,
//     written back through pmem's write port while pmem's read port fetches the next psum[o]
//     (shift4=1 for the X0 high-nibble pass: Y0 = 16*(W*X0_hi) + W*X0_lo; add_q=0 on the first write)
//   * reconstruction (src_ofifo=0, add_q=1): acc_ld loads acc_q <= Y0[o], then out = Y0[o] + D_i[o] = Y_i[o]
//   acc_ld = 1 : acc_q <= in
//
// +alpha (linear attention, see software/transformer/hw_attn.py):
//   norm = 1   : out[c] = round(sum[c] / sum[col-1]) for c < col-1 (half away from zero, saturated to signed
//                4 bit, sign-extended), out[col-1] = sum[col-1] (the normaliser itself).
//                Pipelined: the result is on `out` TWO cycles after the norm cycle (2 quotient bits per stage);
//                a new element can enter every cycle.
//                Divider: the quotient is only 4 bits, because the attention output is a convex combination of
//                4-bit values, so 4 restoring compare-subtract stages per lane are enough (no reciprocal LUT).
//   amax_en = 1: amax_q |= |sum[c]| for all lanes (OR-reduction = the MSB position of max |S|); pipelined, amax_q and
//                the registered shift sh_q are valid 3 cycles after the last amax_en
//   rq = 1     : out[31:0] = requantised + nibble-split word of the pmem vector `in`:
//                s8[c] = clamp((in[c] + 2^(sh-1)) >>> sh, -128, 127),  sh = max(0, bitlen(amax_q) - 7)
//                nibble[c] = nib_sel ? s8[c][7:4] : s8[c][3:0]; out[4c+3:4c] = nibble[c]
//                (written into xmem as the PE weight word of phase B; out[127:32] = 0)
module sfu (clk, reset, acc, relu, in, out, ofifo_in, src_ofifo, shift4, add_q, acc_ld, norm, amax_en, rq, nib_sel);

  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset, acc, relu;
  input  src_ofifo, shift4, add_q, acc_ld;
  input  norm, amax_en, rq, nib_sel;
  input  [psum_bw*col-1:0] in;
  input  [psum_bw*col-1:0] ofifo_in;
  output [psum_bw*col-1:0] out;

  wire [psum_bw*col-1:0] sum_all;   // A + B of every lane
  wire [psum_bw*col-1:0] lane_out;  // per-lane result (ReLU / divider)
  wire [psum_bw*col-1:0] abs_all;   // |sum| of every lane
  wire [4*col-1:0]       rq_word;

  // divider pipeline: valid flags of the two stages
  reg norm_d1, norm_d2;
  always @ (posedge clk) begin
    if (reset) begin norm_d1 <= 1'b0; norm_d2 <= 1'b0; end
    else       begin norm_d1 <= norm; norm_d2 <= norm_d1; end
  end

  // ---------------- requantiser exponent ----------------
  reg  [psum_bw-1:0] amax_q;
  wire [psum_bw-1:0] abs_or;
  reg  [4:0] msb;                   // bitlen(amax_q)
  wire [4:0] sh = (msb > 5'd7) ? msb - 5'd7 : 5'd0;
  reg  [4:0] sh_q;                  // registered shift amount used by the requantiser
  integer k;
  always @ (*) begin
    msb = 5'd0;
    for (k = 0; k < psum_bw; k = k + 1)
      if (amax_q[k]) msb = k + 1;
  end

  genvar i;
  generate
  for (i=0; i<col; i=i+1) begin : col_num
    reg  signed [psum_bw-1:0] acc_q;
    wire signed [psum_bw-1:0] in_i    = in[psum_bw*(i+1)-1:psum_bw*i];
    wire signed [psum_bw-1:0] ofifo_i = ofifo_in[psum_bw*(i+1)-1:psum_bw*i];
    wire signed [psum_bw-1:0] op_a    = src_ofifo ? (shift4 ? (ofifo_i <<< 4) : ofifo_i) : acc_q;
    wire signed [psum_bw-1:0] op_b    = add_q ? in_i : {psum_bw{1'b0}};
    wire signed [psum_bw-1:0] sum     = op_a + op_b;

    always @ (posedge clk) begin
      if (reset)
        acc_q <= {psum_bw{1'b0}};
      else if (acc_ld)
        acc_q <= in_i;
      else if (acc)
        acc_q <= acc_q + in_i;
    end

    assign sum_all[psum_bw*(i+1)-1:psum_bw*i] = sum;
    assign abs_all[psum_bw*(i+1)-1:psum_bw*i] = sum[psum_bw-1] ? -sum : sum;

    // ---- divider lane: q = round(sum / den), den = sum of the last lane; 2-stage pipeline ----
    //   stage 1 (registered when norm = 1): n = sum, d = den      -> t = 2|n| + d, quotient bits 3..2
    //   stage 2 (registered):               remainder, q[3:2]    -> quotient bits 1..0, sign, saturation
    //   the result appears on `out` two cycles after the norm cycle (norm_d2), the controller writes it then
    wire signed [psum_bw-1:0] den = sum_all[psum_bw*col-1:psum_bw*(col-1)];
    reg  signed [psum_bw-1:0] n1, d1;                  // stage-1 registers
    reg  [psum_bw+4:0] t2;                             // stage-2 registers
    reg  [1:0] q2;
    reg  neg2, bad2;
    reg  [psum_bw-1:0] dd2;
    always @ (posedge clk) begin
      if (norm) begin n1 <= sum; d1 <= den; end
    end
    reg  [psum_bw+4:0] ta, tb;
    reg  [1:0] qa, qb;              // quotient bits 3..2 (stage 1) and 1..0 (stage 2)
    reg  [3:0] qs;                  // signed, saturated 4-bit quotient
    integer b1, b2;
    always @ (*) begin               // stage-1 logic: bits 3, 2
      ta = {(n1[psum_bw-1] ? -n1 : n1), 1'b0} + {1'b0, d1};
      qa = 2'b00;
      for (b1 = 3; b1 >= 2; b1 = b1 - 1)
        if (ta >= ({d1, 1'b0} << b1)) begin
          ta = ta - ({d1, 1'b0} << b1);
          qa[b1-2] = 1'b1;
        end
    end
    always @ (posedge clk) begin
      t2 <= ta; q2 <= qa; neg2 <= n1[psum_bw-1]; bad2 <= d1[psum_bw-1] || (d1 == 0); dd2 <= d1;
    end
    always @ (*) begin               // stage-2 logic: bits 1, 0 and saturation
      tb = t2;
      qb = 2'b00;
      for (b2 = 1; b2 >= 0; b2 = b2 - 1)
        if (tb >= ({dd2, 1'b0} << b2)) begin
          tb = tb - ({dd2, 1'b0} << b2);
          qb[b2] = 1'b1;
        end
      if (bad2)      qs = 4'd0;
      else if (neg2) qs = ({q2, qb} > 4'd8) ? 4'b1000 : (~{q2, qb} + 4'd1);
      else           qs = ({q2, qb} > 4'd7) ? 4'd7 : {q2, qb};
    end

    // ---- requantiser lane: s8 = clamp((in + half) >>> sh, -128, 127) ----
    wire signed [psum_bw:0] in_x   = {in_i[psum_bw-1], in_i};
    wire signed [psum_bw:0] half   = (sh_q == 0) ? 0 : ({{psum_bw{1'b0}}, 1'b1} << (sh_q - 1));
    wire signed [psum_bw:0] shd    = (in_x + half) >>> sh_q;
    wire [7:0] s8 = (shd > 127) ? 8'sd127 : (shd < -128) ? 8'h80 : shd[7:0];
    assign rq_word[4*(i+1)-1:4*i] = nib_sel ? s8[7:4] : s8[3:0];

    wire [psum_bw-1:0] o_relu = (relu && sum[psum_bw-1]) ? {psum_bw{1'b0}} : sum;
    if (i < col-1) begin : divlane
      assign lane_out[psum_bw*(i+1)-1:psum_bw*i] = norm_d2 ? {{(psum_bw-4){qs[3]}}, qs} : o_relu;
    end
    else begin : denlane
      assign lane_out[psum_bw*(i+1)-1:psum_bw*i] = norm_d2 ? dd2 : o_relu;
    end
  end
  endgenerate

  // OR-reduction of |sum| over lanes
  genvar j;
  generate
  for (j=0; j<psum_bw; j=j+1) begin : orred
    wire [col-1:0] bits;
    for (i=0; i<col; i=i+1) begin : ln
      assign bits[i] = abs_all[psum_bw*i+j];
    end
    assign abs_or[j] = |bits;
  end
  endgenerate

  // pipelined: OR of the lanes registered first (absor_q), then folded into amax_q; the shift amount is registered
  // too (sh_q). The requantiser runs several cycles after the last amax_en, so the latency is free.
  reg [psum_bw-1:0] absor_q;
  reg amax_en_d;
  always @ (posedge clk) begin
    if (reset) begin
      amax_q <= {psum_bw{1'b0}}; absor_q <= {psum_bw{1'b0}}; amax_en_d <= 1'b0; sh_q <= 5'd0;
    end
    else begin
      amax_en_d <= amax_en;
      if (amax_en)   absor_q <= abs_or;
      if (amax_en_d) amax_q  <= amax_q | absor_q;
      sh_q <= sh;
    end
  end

  assign out = rq ? {{(psum_bw*col-4*col){1'b0}}, rq_word} : lane_out;

endmodule
