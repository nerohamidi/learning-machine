module sfp (clk, reset, acc, in, out);

  parameter col = 8;
  parameter psum_bw = 16;

  input  clk;
  input  reset;
  input  acc;                       // inst[33]: add this cycle's input
  input  [col*psum_bw-1:0] in;      // one psum SRAM word = 8 lanes x 16 bits
  output [col*psum_bw-1:0] out;     // goes to sfp_out

  genvar i;

  for (i=0; i<col; i=i+1) begin : lane
    reg  signed [psum_bw-1:0] acc_q;     // this lane's running sum
    wire signed [psum_bw-1:0] in_lane;

    assign in_lane = in[psum_bw*(i+1)-1:psum_bw*i];

    always @ (posedge clk) begin
        if (reset) begin
            acc_q <= 0;
        end
        else begin
            if (acc) begin
                acc_q <= acc_q + in_lane;
            end
        end
    end

    // ReLU
    assign out[psum_bw*(i+1)-1:psum_bw*i] = acc_q[psum_bw-1] ? 0 : acc_q;
  end

endmodule