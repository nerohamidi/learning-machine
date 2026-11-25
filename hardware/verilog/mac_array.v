// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
//
// +alpha: WS/OS switchable array with zero-guard bits.
//   in_w  : one 4-bit word per row from L0 (WS: activations or weights being loaded; OS: activations)
//   in_wt : one 4-bit word per column from the IFIFO (OS only: weights entering from the north)
//   The zero guard of every edge operand is generated here: guard = zg & (operand == 0).
//   valid : WS: execute leaving the last row (one psum row per cycle);  OS: drain (one result row per cycle)
module mac_array (clk, reset, out_s, in_w, in_wt, inst_w, valid, os, drain, zg, wuns);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  input  [row*bw-1:0] in_w;
  input  [col*bw-1:0] in_wt;
  input  [2:0] inst_w;  // {act_signed, execute, load}
  output [col-1:0] valid;
  input  os;     // 1 = output stationary
  input  drain;  // OS: shift results out (broadcast to all PEs)
  input  zg;     // zero-guard gating enable
  input  wuns;   // weight unsigned

  wire [psum_bw*col*(row+1)-1:0] temp;        // psum chain (north -> south), one extra slot for the top input
  wire [row*col-1:0]             valid_temp;  // valid vectors of all rows
  reg  [3*row-1:0]               inst_w_temp; // instruction shift register, one stage per row

  // top edge: WS psum in = 0;  OS = {guard, weight} per column (0 while draining)
  genvar c;
  generate
  for (c=0; c < col; c=c+1) begin : top_edge
    wire [bw-1:0] w = in_wt[bw*(c+1)-1:bw*c];
    assign temp[psum_bw*(c+1)-1:psum_bw*c] =
        (os & ~drain) ? {{(psum_bw-bw-1){1'b0}}, zg & ~(|w), w} : {psum_bw{1'b0}};
  end
  endgenerate

  assign out_s = temp[psum_bw*col*(row+1)-1:psum_bw*col*row];
  assign valid = os ? {col{drain}} : valid_temp[row*col-1:row*col-col];   // only the last row's valid is used

  genvar i;
  generate
  for (i=1; i < row+1 ; i=i+1) begin : row_num
      wire [bw-1:0] a = in_w[bw*i-1:bw*(i-1)];
      mac_row #(.bw(bw), .psum_bw(psum_bw), .col(col)) mac_row_instance (
         .clk(clk),
         .reset(reset),
         .in_w({zg & ~(|a), a}),
         .inst_w(inst_w_temp[3*i-1:3*(i-1)]),
         .in_n(temp[psum_bw*col*i-1:psum_bw*col*(i-1)]),
         .out_s(temp[psum_bw*col*(i+1)-1:psum_bw*col*i]),
         .valid(valid_temp[col*i-1:col*(i-1)]),
         .os(os), .drain(drain), .wuns(wuns)
      );
  end
  endgenerate

  integer j;
  always @ (posedge clk) begin
   // inst_w flows to row0 to row7
    if (reset == 1) begin
      inst_w_temp <= {(3*row){1'b0}};
    end
    else begin
      inst_w_temp[2:0] <= inst_w;
      for (j=1; j < row; j=j+1)
        inst_w_temp[3*j +: 3] <= inst_w_temp[3*(j-1) +: 3];
    end
  end

endmodule
