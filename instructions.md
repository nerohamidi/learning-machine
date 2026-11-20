Project: VGGNet on 2D Systolic Array with Differential Video-Frame Processing, and Mapping on Cyclone IV GX

Part 1. Vanilla Part (60% + 5% Bonus)

Step 1. Train VGG16 with Quantization-Aware Training (10%)

Train for 4-bit input activation and 4-bit weight to achieve > 90% accuracy.
This time, squeeze (reduce) a certain convolution layer to be narrow, e.g., input channel number 8 and output channel number 8.
Also, remove the batch normalization layer after the squeezed convolution, e.g., replace “conv -> relu -> batchnorm” with “conv -> relu”.
This layer will be mapped on your 8x8 2D systolic array. Reducing it to 8 channels lets the layer map onto the array without tiling.
This time, compute your “psum_recovered” as in HW4, including ReLU, and compare it with your prehooked input for the next layer (instead of your computed psum_ref).
Unlike HW4, your “psum_recovered” should be identical (error < 10^-5) to the next layer’s prehooked input, because the prehooked data includes the quantization error for both input and weight (whereas your “computed_ref” in HW4 included only weight quantization error, since you computed “un-quantized input * quantized weight” at that time).
Hint: Do not place the squeezed conv layer too early, because early layers have a large feature map (nij), which makes verification take many cycles. Recommended location: around the 27th layer, e.g., features[27].
Measure of success: Accuracy > 90% with 8 input/output channels for at least one layer, and error < 10^-3 for psum_recovered.
Step 2. Complete RTL Core Design (5%)
Connect the following blocks:

2D array with MAC units
Scratchpad memories for (1) activation & weight SRAM and (2) psum SRAM (you might need multiple banks for psum)
L0 and output FIFO (note: you do not use IFIFO in this project because the weight is given from west to east via L0)
Special function processor (for accumulation and ReLU)
Hint: Make a core.v that includes corelet.v and the SRAM memories (use your SRAM behavioral .v file from HW7). corelet.v includes all the other blocks. Only corelet.v will be mapped onto the FPGA, so this hierarchy helps in Part 5.
Measure of success: Completion of your entire core design, with no compilation errors after all the connections.
Step 3. Testbench Generation (10%)
Use the testbench template (core_tb.v in the “project” directory in git).
Your testbench has access to the ports of your core.v, so it acts as a controller.
Complete the following stages:
L1 scratchpad loading for weight and activation (e.g., from DRAM, which is emulated by your testbench)
Kernel data loading to PE registers (via L0)
L0 data loading
Execution with PEs
Psum movement to L1 scratchpad (via OFIFO)
Accumulation in SFU and store back to psum SRAM
ReLU in SFU and store back to psum SRAM
Measure of success: Implementation of all the above stages without compilation errors, including all your Verilog and the testbench.
Step 4. Verification (10%)
Generate the text stimulus vectors (input.txt, weight.txt) and expected output (psum.txt) for the squeezed layer, as you did in HW7.
Apply the stimulus files to your testbench (core_tb.v) to run all the stages described in Step 3.
Verify that your results match the expected output file.
Measure of success: Generation of your stimulus and expected output files, and zero verification errors compared with psum.txt.
Step 5. Mapping on FPGA (Cyclone IV GX EP4CGX150DF31I7AD) (5%)
Map your corelet.v (not core.v) onto the FPGA with Quartus Prime 19.1.
Complete synthesis and placement/route.
Measure your frequency at the slow corner.
Measure your power with a 20% input activity factor.
Note: This is not a frequency competition. It is for going through the entire flow.
Measure of success: Reporting the final frequency and power numbers.
Step 6. +alpha (20% + 5% Bounus)
Add anything else, for example:

Any techniques that you learned from the course
A technique from your own idea
Thorough verification, e.g., for multiple layers or tiled layers
Mapping other networks, e.g., ResNet / NLP
Reconfiguration between weight and output stationary
Scalable design, e.g., multi-core for tiled layer processing
Others
Part 2. Differential Input Processing for Video Frames (20%)
Input Data: We will provide the input files of four consecutive 8-bit input frames (X0–X3).Download input files of four consecutive 8-bit input frames (X0–X3).
Step 1: Reference processing (on the systolic array)

Process X0 at 8-bit precision and store its output: Y0 = W × X0.
Y0 must be computed on your systolic array, not in Python.
The PE computes 4×4 bits, so you must handle the 8×4 operation. Split X0 into high and low nibbles and run two passes, then combine them: Y0 = 16 × (W × X0_high) + (W × X0_low).
Store Y0 so that it can be reused for all three differential frames.
Step 2: Differential processing

For i = 1, 2, 3, calculate ΔXi = Xi − X0, process it using the same systolic array (1 pass each)
Keep the same W in the PEs across all passes. Do not reload the weights.
Step 3: Output reconstruction

Recover each output as Yi = Y0 + (W × ΔXi).
Reconstruct the raw convolution outputs (partial sums).
Step 4: RTL changes

Start from your Part 1 RTL (core, corelet, systolic array, SFU, FIFOs) and extend it.
PE/MAC: the X0 nibbles are unsigned, whereas ΔX is signed. Support both activation types in the 4-bit PE (for example, with a mode signal).
SFU: add the shift-and-add of the nibble passes for Y0, and the addition of Y0 to W × ΔXi for reconstruction.
Memory: provide storage for X0, ΔXi and the stored Y0 (for example, a dedicated psum SRAM bank), and check that the accumulator width is sufficient.
Control: sequence the passes (2 for X0, then 1 for each delta) with W kept loaded.
Delta generation: ΔXi may be computed by the testbench (emulating the host/DRAM) or in RTL. State which one you chose and account for its cost in your analysis.
Step 5: Verification

Generate the stimulus files (X0 nibbles, ΔXi) and expected-output files (golden Y0 and reconstructed Yi) in Python from the provided inputs.
Apply them to your testbench, run all the stages, and compare the hardware results with the expected output files.
Report the number of passes (or cycles) for the conventional and the differential method from simulation.
Step 6: Analysis (include in the report)

Accuracy: error of Yi against the reference W × Xi computed with the full 8-bit Xi (max error, MSE or SQNR for each frame), and the fraction of ΔXi values that saturate at 4 bits.
Measure of success: 
The hardware Y0 matches the expected Y0 exactly.
Yi is within the stated error tolerance of the reference (the tolerance will be announced by us later).
The pass reduction is demonstrated in simulation.
Part 3. Presentation and Report (20%)

There is no strict format for the presentation other than the following:

Presentation length: TBD min (without interrupt) + TBD min Q&A.
At TBD min, hard stop by the next team (a stopwatch will be at the front row).
Focus on your unique strength (+alpha part).
Skip general intro / general motivation, e.g., the importance of low-power ML implementation, which everyone knows.
Explain the idea concisely and prove its efficacy.
More details to be added soon
Summary Table Requirements
Frequency, power and accuracy specs (baseline, and frames 1-3 under differential processing)
Verification results (baseline psum.txt and each differential output file)
Differential processing: pass count, measured cycles and speedup (if 7.5 is done)
Benefit of your +alpha idea
Note: The report must be submitted; the presentation is not required.
Document each part with proper, precise explanations.
For each part, include what was done, what results were observed and your inference.
No minimum length; maximum TBD  pages. If the report is mostly figures and graphs, up to TBD  pages.
