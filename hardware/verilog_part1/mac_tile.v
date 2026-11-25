// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_tile (clk, out_s, in_w, out_e, in_n, inst_w, inst_e, reset);

parameter bw = 4;
parameter psum_bw = 16;

output [psum_bw-1:0] out_s;
input  [bw-1:0] in_w; // inst[1]:execute, inst[0]: kernel loading
output [bw-1:0] out_e;
input  [1:0] inst_w;
output [1:0] inst_e;
input  [psum_bw-1:0] in_n;
input  clk;
input  reset;

// latches (edge-triggered registers)
reg [1:0]         inst_q;        // instruction passed to the east
reg [bw-1:0]      a_q;           // activation
reg [bw-1:0]      b_q;           // weight (stationary)
reg [psum_bw-1:0] c_q;           // psum from the north
reg               load_ready_q;  // 1: this PE has not loaded its weight yet

wire [psum_bw-1:0] mac_out;      // a_q * b_q + c_q

assign out_e  = a_q;
assign inst_e = inst_q;
assign out_s  = mac_out;

mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
        .a(a_q),
        .b(b_q),
        .c(c_q),
        .out(mac_out)
);

always @ (posedge clk) begin
  if (reset == 1) begin
    inst_q       <= 2'b00;
    load_ready_q <= 1'b1;
    a_q          <= {bw{1'b0}};
    b_q          <= {bw{1'b0}};
    c_q          <= {psum_bw{1'b0}};
  end
  else begin
    // execution instruction always flows to the east
    inst_q[1] <= inst_w[1];

    // psum from the north is latched every cycle
    c_q <= in_n;

    // accept a new activation/weight when executing or loading a kernel
    if (inst_w[0] | inst_w[1])
      a_q <= in_w;

    // kernel load: keep the first value that arrives while this PE is ready
    if (inst_w[0] & load_ready_q) begin
      b_q          <= in_w;
      load_ready_q <= 1'b0;
    end

    // forward the kernel-load instruction only after this PE has loaded its weight
    if (load_ready_q == 1'b0)
      inst_q[0] <= inst_w[0];
  end
end

endmodule
