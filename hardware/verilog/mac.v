// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac (out, a, b, c, sgn, wuns);

parameter bw = 4;
parameter psum_bw = 16;

output signed [psum_bw-1:0] out;
input signed  [bw-1:0] a;  // activation
input signed  [bw-1:0] b;  // weight
input signed  [psum_bw-1:0] c;
input sgn;   // Part 2: 1 = activation a is signed (X0 high nibble, dX), 0 = unsigned (Part 1, X0 low nibble)
input wuns;  // +alpha: 1 = weight b is unsigned (OS phase A: phi(K) streamed from the north; S8 low nibble)


wire signed [2*bw+1:0] product;
wire signed [psum_bw-1:0] psum;
wire signed [bw:0]   a_pad;
wire signed [bw:0]   b_pad;

assign a_pad = {sgn & a[bw-1], a};    // sign- or zero-extend the activation
assign b_pad = {~wuns & b[bw-1], b};  // sign- or zero-extend the weight (mixed-sign 5b x 5b multiplier)
assign product = a_pad * b_pad;

assign psum = product + c;
assign out = psum;

endmodule
