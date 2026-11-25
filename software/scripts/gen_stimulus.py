"""Part 1 Step 4: stimulus and expected-output files for the squeezed layer features[27] (8x8 systolic array).

Uses the trained squeezed VGG16 and one CIFAR-10 test image. Everything written here is an integer:
x_int in [0, 15] (unsigned 4-bit), w_int in [-7, 7] (signed 4-bit), psums signed 16-bit.

Files (written to --out, default ../hardware/sim/stim):
  activation_tile0.txt                 36 lines x 32b : padded 6x6 input, one nij per line, row (ic) 7 ... row 0
  weight_itile0_otile0_kij{0..8}.txt    8 lines x 32b : line t = output channel t (array column t), ic 7 ... ic 0
  psum_kij.txt                        324 lines x 128b: raw array output, line kij*36 + nij, col 7 ... col 0
  acc_address.txt                     144 lines x 11b : for o_nij = 0..15 the 9 psum-SRAM addresses to add up
  psum.txt                             16 lines x 128b: accumulated output (before ReLU) per o_nij
  out.txt                              16 lines x 128b: ReLU(psum) per o_nij (what the next layer receives)
Every file starts with three single-token comment lines (the testbench skips them with "%s").

The script also checks that ReLU(out.txt) * x_delta * w_delta matches the pre-hooked input of features[29].

Usage (from project/software/):
    python scripts/gen_stimulus.py [--image 0] [--out ../hardware/sim/stim]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SW)
from net.vgg_quant_squeeze import VGG16_quant_squeeze, SQUEEZED_IDX, NEXT_IDX  # noqa: E402
from models.quant_layer import act_quantization  # noqa: E402

ARRAY, BW, PSUM_BW = 8, 4, 16
PSUM_BASE_OUT = 1024        # pmem address of the accumulated outputs (raw psums use 0 .. 9*36-1)


def twos(v, bits):
    v = int(v)
    assert -(1 << (bits - 1)) <= v < (1 << (bits - 1)), (v, bits)
    return format(v & ((1 << bits) - 1), '0{}b'.format(bits))


def pack(values, bits, signed):
    """values[0] goes to the LSBs (row 0 / column 0)."""
    if signed:
        return ''.join(twos(v, bits) for v in reversed(values))
    return ''.join(format(int(v), '0{}b'.format(bits)) for v in reversed(values))


def write(path, header, lines):
    with open(path, 'w') as f:
        for h in header:
            assert ' ' not in h
            f.write(h + '\n')
        for ln in lines:
            f.write(ln + '\n')


def layer_integers(model, image, device):
    """x_int [8,H,W], w_int [8,8,3,3], deltas and the hooked input of the next layer for one image."""
    layer, nxt = model.features[SQUEEZED_IDX], model.features[NEXT_IDX]
    saved = {}
    h1 = layer.register_forward_pre_hook(lambda m, i: saved.__setitem__('x', i[0].detach().cpu()))
    h2 = nxt.register_forward_pre_hook(lambda m, i: saved.__setitem__('y', i[0].detach().cpu()))
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
        weight_q = layer.weight_quant(layer.weight).detach().cpu()
    h1.remove()
    h2.remove()
    w_delta = layer.weight_quant.wgt_alpha.detach().cpu() / 7
    a_alpha = layer.act_alpha.detach().cpu()
    x_delta = a_alpha / 15
    w_int = (weight_q / w_delta).round()
    x_int = (act_quantization(4)(saved['x'][0], a_alpha) / x_delta).round()
    assert w_int.abs().max() <= 7 and 0 <= x_int.min() and x_int.max() <= 15
    return x_int, w_int, x_delta, w_delta, saved['y'][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=int, default=0, help='CIFAR-10 test image index')
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    p.add_argument('--out', default=os.path.join(SW, '..', 'hardware', 'sim', 'stim'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ampere GPUs run cuDNN convs in TF32 (10-bit mantissa) by default; use true FP32 for the reference
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    test = torchvision.datasets.CIFAR10(root=os.path.join(SW, 'data'), train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    model = VGG16_quant_squeeze()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model.to(device).eval()

    image, label = test[args.image]
    x_int, w_int, x_delta, w_delta, next_in = layer_integers(model, image, device)
    C_out, C_in, k, _ = w_int.shape
    assert C_in == C_out == ARRAY

    # ---- padded activations, nij = y*6 + x ----
    H = x_int.shape[1]
    P = H + 2
    a_pad = torch.zeros(C_in, P, P)
    a_pad[:, 1:1 + H, 1:1 + H] = x_int
    a_pad = a_pad.reshape(C_in, -1).long()                       # [8, 36]
    n_nij = a_pad.shape[1]
    write(os.path.join(args.out, 'activation_tile0.txt'),
          ['#time0row7[msb-lsb],time0row6[msb-lst],....,time0row0[msb-lst]#',
           '#time1row7[msb-lsb],time1row6[msb-lst],....,time1row0[msb-lst]#',
           '#................#'],
          [pack(a_pad[:, nij].tolist(), BW, signed=False) for nij in range(n_nij)])

    # ---- weights per kij: line t = column t = output channel t ----
    w_int = w_int.long()
    for kij in range(k * k):
        ky, kx = divmod(kij, k)
        write(os.path.join(args.out, 'weight_itile0_otile0_kij{}.txt'.format(kij)),
              ['#col0row7[msb-lsb],col0row6[msb-lst],....,col0row0[msb-lst]#',
               '#col1row7[msb-lsb],col1row6[msb-lst],....,col1row0[msb-lst]#',
               '#................#'],
              [pack(w_int[t, :, ky, kx].tolist(), BW, signed=True) for t in range(C_out)])

    # ---- raw psums: psum[kij][nij][oc] = sum_ic w[oc, ic, kij] * a[ic, nij] ----
    psum = torch.zeros(k * k, n_nij, C_out, dtype=torch.long)
    for kij in range(k * k):
        ky, kx = divmod(kij, k)
        psum[kij] = (w_int[:, :, ky, kx] @ a_pad).t()            # [36, 8]
    assert psum.abs().max() < (1 << (PSUM_BW - 1))
    write(os.path.join(args.out, 'psum_kij.txt'),
          ['#line=kij*36+nij#', '#col7[msb-lsb],col6,....,col0[lsb]#', '#16b_signed_per_col#'],
          [pack(psum[kij, nij].tolist(), PSUM_BW, signed=True) for kij in range(k * k) for nij in range(n_nij)])

    # ---- accumulation addresses and outputs ----
    O = P - k + 1
    addr, out = [], torch.zeros(O * O, C_out, dtype=torch.long)
    for o_nij in range(O * O):
        oy, ox = divmod(o_nij, O)
        for kij in range(k * k):
            ky, kx = divmod(kij, k)
            nij = (oy + ky) * P + (ox + kx)
            addr.append(format(kij * n_nij + nij, '011b'))
            out[o_nij] += psum[kij, nij]
    write(os.path.join(args.out, 'acc_address.txt'),
          ['#9_lines_per_o_nij#', '#addr=kij*36+nij#', '#11b#'], addr)
    write(os.path.join(args.out, 'psum.txt'),
          ['#accumulated_psum_before_ReLU#', '#col7[msb-lsb],....,col0[lsb]#', '#line=o_nij#'],
          [pack(out[o].tolist(), PSUM_BW, signed=True) for o in range(O * O)])
    out_relu = out.clamp(min=0)
    write(os.path.join(args.out, 'out.txt'),
          ['#ReLU(psum)#', '#col7[msb-lsb],....,col0[lsb]#', '#line=o_nij#'],
          [pack(out_relu[o].tolist(), PSUM_BW, signed=True) for o in range(O * O)])

    # ---- checks ----
    ref = F.conv2d(x_int.unsqueeze(0), w_int.float(), padding=1)[0].reshape(C_out, -1).t().long()
    assert torch.equal(ref, out), 'psum accumulation does not match conv2d'
    recovered = out_relu.t().float() * x_delta * w_delta                       # [8, 16]
    err = (recovered - next_in.reshape(C_out, -1)).abs().max().item()
    print('image {} (label {}) | x_int range [{}, {}] | psum range [{}, {}] | out range [{}, {}]'.format(
        args.image, label, x_int.min().int(), x_int.max().int(), psum.min(), psum.max(), out.min(), out.max()))
    print('max |ReLU(out) * x_delta * w_delta - prehook(features[{}])| = {:.3e}'.format(NEXT_IDX, err))
    assert err < 1e-3
    print('stimulus written to', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
