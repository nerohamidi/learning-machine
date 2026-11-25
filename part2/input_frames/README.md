# Part 2 input frames

Provided by the course: `differential_input_batch.zip`, extracted on 2026-11-21. The macOS `__MACOSX` metadata was dropped.

`X0.txt` … `X3.txt` hold four consecutive 8-bit frames. Each file has 2048 integers, one decimal value per line.

| frame | min | max | mean |
|---|---|---|---|
| X0 | -96 | 96 | -2.87 |
| X1 | -102 | 103 | -2.95 |
| X2 | -100 | 100 | -2.77 |
| X3 | -100 | 101 | -2.82 |

ΔXi = Xi − X0 always lies in [−7, 7], so every value fits in signed 4 bits (0% saturation).
22–24% of the ΔXi values are zero.

**Note:** the frames are *signed*, but the spec describes the X0 nibbles as unsigned.
The tensor shape (e.g. 8 ch × 16 × 16, or 128 × 4 × 4) and the weights W are not stated in the files.
