# ECE284 Final Project: VGG16 on an 8×8 Systolic Array

Spec: [`instructions.md`](instructions.md).

## Trained model (download first)

The Part 1 checkpoint (4-bit VGG16, `features[27]` squeezed to 8→8, 91.60% on CIFAR-10) is 71 MB. It is not in git.
Download it from Google Drive:

**https://drive.google.com/file/d/1J1xARZzBRBFzddxVDRegyoGBF_UJmCwP/view?usp=sharing**

Put it at:

```
software/result/vgg16_quant_4bit_squeeze8/model_best.pth.tar
```

The Part 1 scripts (`verify_squeeze.py`, `gen_stimulus.py`, `gen_tiled_stimulus.py`, `regression.py`) load it from there.
`.gitignore` keeps it out of commits. To download from the command line:

```bash
pip install gdown
gdown 1J1xARZzBRBFzddxVDRegyoGBF_UJmCwP -O software/result/vgg16_quant_4bit_squeeze8/model_best.pth.tar
```

CIFAR-10 is not in the repo either; the scripts download it into `software/data/` on first run.

## Layout

| Path | What |
|---|---|
| `software/net/` | Quantized, squeezed VGG16 model |
| `software/scripts/` | QAT training, psum verification, stimulus generation (Part 1, tiled, OS, Part 2), regression, dataflow model |
| `software/transformer/` | +alpha: 4-bit linear-attention ViT and its bit-exact hardware attention reference |
| `software/result/` | Training/eval logs; ViT checkpoints in `vit/` |
| `part2/input_frames/` | Provided Part 2 frames X0–X3 |
| `hardware/verilog/` | Final RTL (Parts 1, 2 and +alpha); `verilog_part1/` is the Part 1-only snapshot |
| `hardware/sim/` | Testbenches (`core_tb*.v`), filelists, stimulus folders (`stim*`), regression script and logs |
| `hardware/fpga/` | Quartus project, flow/power scripts, and timing/power reports for each part |
| `report/` | Report source (`report.tex`), figures, figure scripts and `report.pdf` |

## How to run

```bash
# software (Python + PyTorch), from software/
python scripts/verify_squeeze.py                     # needs the checkpoint above
python scripts/gen_stimulus.py                       # -> ../hardware/sim/stim

# RTL (iverilog), from hardware/sim/
iverilog -g2005 -o compiled -c filelist && vvp compiled
./run_alpha.sh                                       # full regression -> alpha_logs/

# report, from report/
latexmk -pdf report.tex
```
