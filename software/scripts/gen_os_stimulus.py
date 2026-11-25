"""+alpha (WS/OS switch): stimulus for running a 3x3 conv layer of the squeezed VGG16 in OUTPUT-STATIONARY mode.

Mapping (course Ch7 p25-27): PE row r = output pixel 8*pt + r, PE column c = output channel 8*ot + c,
time step s = ic*9 + kij (the whole reduction of K = C_in*9 steps happens inside each PE).
  west  (L0, from xmem) at step s : row r = x_pad[ic, oy_r + ky, ox_r + kx]   (4b unsigned, im2col order)
  north (IFIFO, from wmem) at step s: col c = w[8*ot + c, ic, ky, kx]          (4b signed)
After K steps the 8x8 results are drained down the columns: OFIFO vector j = PE row 7-j.

Files in --out:
  config.txt          n_ptile n_otile K relu len_onij
  os_act_p{pt}.txt     K x 32b (row7[msb] .. row0[lsb])
  os_wgt_o{ot}.txt     K x 32b (col7[msb] .. col0[lsb])
  out_otile{ot}.txt    16 x 128b expected outputs (same files and format as gen_tiled_stimulus.py)
  stats.txt            zero fractions of the streamed operands (for the zero-gating analysis)

Usage (from project/software/):
    python scripts/gen_os_stimulus.py --layer 27 --out ../hardware/sim/stim_os_f27
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_tiled_stimulus import load_model_and_data, layer_case, SW  # noqa: E402
from gen_stimulus import pack, write, ARRAY, BW, PSUM_BW  # noqa: E402
from net.vgg_quant_squeeze import SQUEEZED_IDX  # noqa: E402


def generate(model, test, image_idx, layer_idx, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    image, label = test[image_idx]
    x_int, w_int, x_delta, w_delta, saved = layer_case(model, image, layer_idx, device)
    relu = int(layer_idx == SQUEEZED_IDX)
    C_out, C_in, k, _ = w_int.shape
    H = x_int.shape[1]
    P = H + 2
    O = P - k + 1
    a_pad = torch.zeros(C_in, P, P, dtype=torch.long)
    a_pad[:, 1:1 + H, 1:1 + H] = x_int.long()
    w_int = w_int.long()
    n_pt = (O * O + ARRAY - 1) // ARRAY
    n_ot = C_out // ARRAY
    K = C_in * k * k

    steps = [(ic, kij) for ic in range(C_in) for kij in range(k * k)]
    zeros_a = zeros_w = 0
    for pt in range(n_pt):
        lines = []
        for ic, kij in steps:
            ky, kx = divmod(kij, k)
            v = []
            for r in range(ARRAY):
                o = ARRAY * pt + r
                if o < O * O:
                    oy, ox = divmod(o, O)
                    v.append(a_pad[ic, oy + ky, ox + kx].item())
                else:
                    v.append(0)
            zeros_a += sum(1 for t in v if t == 0)
            lines.append(pack(v, BW, False))
        write(os.path.join(out_dir, 'os_act_p{}.txt'.format(pt)),
              ['#row7[msb]...row0[lsb]=pixel#', '#line=step(ic*9+kij)#', '#ptile{}#'.format(pt)], lines)
    for ot in range(n_ot):
        lines = []
        for ic, kij in steps:
            ky, kx = divmod(kij, k)
            v = w_int[ARRAY * ot:ARRAY * (ot + 1), ic, ky, kx].tolist()
            zeros_w += sum(1 for t in v if t == 0)
            lines.append(pack(v, BW, True))
        write(os.path.join(out_dir, 'os_wgt_o{}.txt'.format(ot)),
              ['#col7[msb]...col0[lsb]=oc#', '#line=step(ic*9+kij)#', '#otile{}#'.format(ot)], lines)

    # expected outputs: the integer conv (+ReLU), identical to the WS flow's out_otile files
    out = torch.nn.functional.conv2d(x_int.unsqueeze(0).double(), w_int.double(), padding=1)[0].round().long()
    out = out.reshape(C_out, O * O)
    assert out.abs().max() < 2 ** (PSUM_BW - 1)
    final = out.clamp(min=0) if relu else out
    for ot in range(n_ot):
        write(os.path.join(out_dir, 'out_otile{}.txt'.format(ot)),
              ['#relu={}#'.format(relu), '#col7[msb]...col0[lsb]#', '#line=o_nij#'],
              [pack(final[ARRAY * ot:ARRAY * (ot + 1), o].tolist(), PSUM_BW, True) for o in range(O * O)])
    conv_err = (out.double() * x_delta * w_delta - saved['y'][0].reshape(C_out, -1).double()).abs().max().item()
    assert conv_err < 1e-3
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        f.write('{} {} {} {} {}\n'.format(n_pt, n_ot, K, relu, O * O))
    fa = zeros_a / (n_pt * K * ARRAY)
    fw = zeros_w / (n_ot * K * ARRAY)
    with open(os.path.join(out_dir, 'stats.txt'), 'w') as f:
        f.write('zero_frac_act {:.4f}\nzero_frac_wgt {:.4f}\n'.format(fa, fw))
    print('features[{}] image {}: OS {} pixel tiles x {} oc tiles, K={} relu={} | zero act {:.1%} wgt {:.1%} | '
          'conv-output error {:.2e}'.format(layer_idx, image_idx, n_pt, n_ot, K, relu, fa, fw, conv_err))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type=int, default=SQUEEZED_IDX)
    p.add_argument('--image', type=int, default=0)
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, test = load_model_and_data(args.ckpt, device)
    generate(model, test, args.image, args.layer, args.out, device)


if __name__ == '__main__':
    main()
