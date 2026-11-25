// Special function unit: one accumulator + ReLU per array column.
//   acc  = 1 : acc_q <= acc_q + in  (in = psum read from the psum SRAM, one 16-bit word per column)
//   relu = 1 : out = max(acc_q, 0), otherwise out = acc_q
//   reset    : acc_q <= 0 (the controller resets the SFU before each output pixel)
module sfu (clk, reset, acc, relu, in, out);

  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset, acc, relu;
  input  [psum_bw*col-1:0] in;
  output [psum_bw*col-1:0] out;

  genvar i;
  generate
  for (i=0; i<col; i=i+1) begin : col_num
    reg  signed [psum_bw-1:0] acc_q;
    wire signed [psum_bw-1:0] in_i = in[psum_bw*(i+1)-1:psum_bw*i];

    always @ (posedge clk) begin
      if (reset)
        acc_q <= {psum_bw{1'b0}};
      else if (acc)
        acc_q <= acc_q + in_i;
    end

    assign out[psum_bw*(i+1)-1:psum_bw*i] = (relu && acc_q[psum_bw-1]) ? {psum_bw{1'b0}} : acc_q;
  end
  endgenerate

endmodule
