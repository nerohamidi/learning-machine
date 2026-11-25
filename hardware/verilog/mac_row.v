// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_row (clk, out_s, in_w, in_n, valid, inst_w, reset, os, drain, wuns);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  output [col-1:0] valid;
  input  [bw:0] in_w;   // {zero guard, data}
  input  [2:0] inst_w;  // {act_signed, execute, load}
  input  [psum_bw*col-1:0] in_n;
  input  os, drain, wuns;

  wire  [(col+1)*(bw+1)-1:0] temp;   // activation/weight chain (west -> east), guard bit included
  wire  [3*(col+1)-1:0]  inst_temp;  // instruction chain (west -> east)

  assign temp[bw:0]        = in_w;
  assign inst_temp[2:0]    = inst_w;

  genvar i;
  generate
  for (i=1; i < col+1 ; i=i+1) begin : col_num
      mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance (
         .clk(clk),
         .reset(reset),
         .in_w( temp[(bw+1)*i-1:(bw+1)*(i-1)]),
         .out_e(temp[(bw+1)*(i+1)-1:(bw+1)*i]),
         .inst_w(inst_temp[3*i-1:3*(i-1)]),
         .inst_e(inst_temp[3*(i+1)-1:3*i]),
         .in_n(in_n[psum_bw*i-1:psum_bw*(i-1)]),
         .out_s(out_s[psum_bw*i-1:psum_bw*(i-1)]),
         .os(os), .drain(drain), .wuns(wuns));

      // column i's output is valid when its inst_e[1] (execute) is high
      assign valid[i-1] = inst_temp[3*i+1];
  end
  endgenerate

endmodule
