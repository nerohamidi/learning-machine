// Switching-activity monitor for every PE of the 8x8 array (included inside a testbench module that has
// `clk`, `core_instance`, `row` and `col`). While mon_en = 1 it counts, per clock cycle, the bits that toggle in
//   a_q, b_q (multiplier operand registers), mac product (multiplier output), c_q (psum / accumulator register)
// and the number of PE-cycles with a MAC issued (inst_q[1]) and how many of those were zero-guarded,
// plus the register bits whose write enable is on (reg_en_bits: a_q 4b, b_q 4b, c_q 16b), i.e. the flop clock
// events that clock gating removes.
// Dynamic power of a PE is dominated by these nodes, so the toggle counts are the activity measure used to
// evaluate zero-guard gating (same workload, +zg=0 vs +zg=1).
reg mon_en = 0;
integer tog_a = 0, tog_b = 0, tog_p = 0, tog_c = 0, mac_issued = 0, mac_guarded = 0, reg_en_bits = 0;

function integer popc16 (input [15:0] v);
  integer q;
  begin
    popc16 = 0;
    for (q = 0; q < 16; q = q + 1) popc16 = popc16 + v[q];
  end
endfunction

// SRAM accesses (the dominant energy term besides the array): reads / writes of xmem, wmem, pmem
integer xm_rd = 0, xm_wr = 0, wm_rd = 0, wm_wr = 0, pm_rd = 0, pm_wr = 0;
always @ (posedge clk) begin
  if (mon_en) begin
    if (core_instance.inst[19] === 1'b0) begin if (core_instance.inst[18]) xm_rd = xm_rd + 1; else xm_wr = xm_wr + 1; end
    if (core_instance.inst[56] === 1'b1) begin if (core_instance.inst[57]) wm_wr = wm_wr + 1; else wm_rd = wm_rd + 1; end
    if (core_instance.pmem_cenr === 1'b0) pm_rd = pm_rd + 1;
    if (core_instance.legacy_wr === 1'b1) pm_wr = pm_wr + 1;
  end
end

genvar mon_r, mon_c;
generate
for (mon_r = 1; mon_r <= row; mon_r = mon_r + 1) begin : mon_row
  for (mon_c = 1; mon_c <= col; mon_c = mon_c + 1) begin : mon_col
    wire [3:0]  a  = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.a_q;
    wire [3:0]  b  = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.b_q;
    wire [15:0] c  = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.c_q;
    wire [9:0]  p  = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.mac_instance.product;
    wire        ex = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.inst_q[1];
    wire        zz = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.za_q |
                     (core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.zb_q &
                      core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.os);
    wire [2:0]  iw = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.inst_w;
    wire [4:0]  inw = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.in_w;
    wire [15:0] inn = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.in_n;
    wire        osm = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.os;
    wire        drn = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.drain;
    wire        lrd = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.load_ready_q;
    wire        zab = core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.za_q |
                      core_instance.corelet_instance.mac_array_instance.row_num[mon_r].mac_row_instance.col_num[mon_c].mac_tile_instance.zb_q;
    wire a_en = (iw[0] | iw[1]) & ~inw[4];
    wire b_en = (osm & iw[1] & ~inn[4]) | (iw[0] & lrd);
    wire c_en = ~osm | drn | (ex & ~zab);
    reg [3:0] a_p = 0, b_p = 0;
    reg [15:0] c_p = 0;
    reg [9:0] p_p = 0;
    always @ (posedge clk) begin
      if (mon_en) begin
        tog_a = tog_a + popc16({12'b0, a ^ a_p});
        tog_b = tog_b + popc16({12'b0, b ^ b_p});
        tog_p = tog_p + popc16({6'b0, p ^ p_p});
        tog_c = tog_c + popc16(c ^ c_p);
        reg_en_bits = reg_en_bits + 4*(a_en === 1'b1) + 4*(b_en === 1'b1) + 16*(c_en === 1'b1);
        if (ex === 1'b1) begin
          mac_issued = mac_issued + 1;
          if (zz === 1'b1) mac_guarded = mac_guarded + 1;
        end
      end
      a_p = a; b_p = b; c_p = c; p_p = p;
    end
  end
end
endgenerate

task mon_report;
  begin
    $display("activity: MACs issued %0d, zero-guarded %0d (%0.1f%%) | toggles: a_q %0d  b_q %0d  product %0d  c_q %0d  (sum %0d) | reg write-enable bits %0d",
             mac_issued, mac_guarded, mac_issued ? 100.0*mac_guarded/mac_issued : 0.0,
             tog_a, tog_b, tog_p, tog_c, tog_a + tog_b + tog_p + tog_c, reg_en_bits);
    $display("sram: xmem rd %0d wr %0d | wmem rd %0d wr %0d | pmem rd %0d wr %0d (rd/wr incl. DRAM->L1 fills)",
             xm_rd, xm_wr, wm_rd, wm_wr, pm_rd, pm_wr);
  end
endtask
