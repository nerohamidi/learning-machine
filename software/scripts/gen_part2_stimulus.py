"""Part 2: differential processing of video frames X0..X3 on the 8x8 systolic array.

Working assumptions (PROGRESS.md, pending TA confirmation):
  * each frame (2048 signed ints) is an 8-channel x 16 x 16 tensor, channel-first (index = c*256 + y*16 + x)
  * W = squeezed features[27] (4-bit, 8 -> 8, 3x3, padding 1); Y = conv(X, W) = raw psums (no ReLU)
  * X0 (signed 8 bit) splits as hi = X0 >> 4 (signed, -8..7) and lo = X0 & 15 (unsigned, 0..15),
    so X0 = 16*hi + lo exactly. hi and dX run with the PE in signed mode, lo in unsigned mode.
  * dXi = sat4(Xi - X0) is computed by the host (testbench), clamped to [-8, 7].

Files in --out (xmem words are 32b = 8 ic x 4b, ic7 in the MSBs; pmem words are 128b = 8 oc x 16b, oc7 in the MSBs):
  act_x0hi.txt act_x0lo.txt act_d1..3.txt     324 lines (padded 18x18, line = nij)   differential method
  act_x{f}hi.txt act_x{f}lo.txt (f = 0..3)    324 lines                              conventional method
  weight_kij{k}.txt                           8 lines (line t = oc t)
  y0.txt y1.txt y2.txt y3.txt                 256 lines (line = o_nij = oy*16+ox): expected Y0 (exact) and
                                              reconstructed Yi = Y0 + W*dXi (= W*Xi when no dX saturates)
  config.txt                                  len_nij len_onij n_frames
  analysis.txt                                accuracy / saturation / accumulator-width report

Usage (from project/software/):  python scripts/gen_part2_stimulus.py --out ../hardware/sim/stim_p2
"""

import argparse
import os
import sys

import numpy as np
import torch

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(SW)
sys.path.insert(0, SW)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net.vgg_quant_squeeze import VGG16_quant_squeeze, SQUEEZED_IDX  # noqa: E402
from gen_stimulus import pack, write, ARRAY, BW, PSUM_BW  # noqa: E402

C, H = 8, 16
P = H + 2                 # padded width
K = 3
NIJ, ONIJ = P * P, H * H  # 324, 256


def load_weights(ckpt):
    model = VGG16_quant_squeeze()
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    layer = model.features[SQUEEZED_IDX].eval()
    with torch.no_grad():
        wq = layer.weight_quant(layer.weight)
    w_delta = layer.weight_quant.wgt_alpha.detach() / 7
    w = (wq / w_delta).round()
    assert ((wq / w_delta) - w).abs().max() < 1e-3 and w.abs().max() <= 7
    return w.long().numpy()                                     # [oc, ic, ky, kx]


def conv_raw(x, w):
    """Weight-stationary emulation: per kij the array produces raw[nij, oc] = sum_ic w[oc, ic, kij] * xpad[ic, nij];
    the SFU accumulates raw[(oy+ky)*P + ox+kx] into output o = oy*16+ox. Returns y[o, oc] and the raw psums."""
    xp = np.zeros((C, P, P), dtype=np.int64)
    xp[:, 1:1 + H, 1:1 + H] = x
    xp = xp.reshape(C, NIJ)
    raws = []
    for kij in range(K * K):
        ky, kx = divmod(kij, K)
        raws.append(w[:, :, ky, kx] @ xp)                       # [oc, nij]
    return xp, raws


def gather(raw, kij):
    ky, kx = divmod(kij, K)
    r = raw.reshape(-1, P, P)[:, ky:ky + H, kx:kx + H]          # [oc, oy, ox]
    return r.reshape(-1, ONIJ).T                                # [o, oc]


def sim_accumulate(raws_by_pass, shifts, inits):
    """Emulates the SFU stream accumulation in kij order; returns the final value and the max |intermediate|."""
    acc = np.zeros((ONIJ, ARRAY), dtype=np.int64)
    peak = 0
    for kij in range(K * K):
        for raws, sh in zip(raws_by_pass, shifts):
            acc = acc + (gather(raws[kij], kij) << sh)
            peak = max(peak, int(np.abs(acc).max()))
    return acc, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', default=os.path.join(PROJ, 'part2', 'input_frames'))
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X = [np.loadtxt(os.path.join(args.frames, 'X{}.txt'.format(f)), dtype=np.int64) for f in range(4)]
    assert all(x.size == C * H * H for x in X)
    X = [x.reshape(C, H, H) for x in X]
    w = load_weights(args.ckpt)
    assert w.shape == (ARRAY, ARRAY, K, K)

    def split(x):
        hi, lo = x >> 4, x & 15
        assert np.array_equal(16 * hi + lo, x) and hi.min() >= -8 and hi.max() <= 7
        return hi, lo

    hi0, lo0 = split(X[0])
    dX_raw = [X[i] - X[0] for i in (1, 2, 3)]
    dX = [np.clip(d, -8, 7) for d in dX_raw]
    sat = [float((d != dr).mean()) for d, dr in zip(dX, dX_raw)]

    # ---- golden outputs ----
    ref = [sum(gather(r, k) for k, r in enumerate(conv_raw(x, w)[1])) for x in X]        # W*Xi, full 8-bit, [o, oc]
    raws = {}
    for name, a in [('x0hi', hi0), ('x0lo', lo0), ('d1', dX[0]), ('d2', dX[1]), ('d3', dX[2])]:
        raws[name] = conv_raw(a, w)
    Y0, peak_y0 = sim_accumulate([raws['x0hi'][1], raws['x0lo'][1]], [4, 0], None)
    assert np.array_equal(Y0, ref[0]), 'Y0 from nibbles is not exact'
    Y, D, peak_d = [Y0], [], 0
    for i in (1, 2, 3):
        Di, pk = sim_accumulate([raws['d{}'.format(i)][1]], [0], None)
        peak_d = max(peak_d, pk)
        D.append(Di)
        Y.append(Y0 + Di)
    max_raw = max(int(np.abs(r).max()) for v in raws.values() for r in v[1])
    peak_conv = 0
    for f in range(4):
        h, l = split(X[f])
        _, pk = sim_accumulate([conv_raw(h, w)[1], conv_raw(l, w)[1]], [4, 0], None)
        peak_conv = max(peak_conv, pk)
    peak_out = max(int(np.abs(y).max()) for y in Y)
    lim = 2 ** (PSUM_BW - 1)
    assert max(max_raw, peak_y0, peak_d, peak_conv, peak_out) < lim, 'accumulator overflows 16 bits'

    # ---- stimulus ----
    def act_file(name, a, signed):
        xp = conv_raw(a, w)[0]
        write(os.path.join(args.out, 'act_{}.txt'.format(name)),
              ['#row7[msb]...row0[lsb]#', '#line=nij#', '#{}#'.format('4b_signed' if signed else '4b_unsigned')],
              [pack(xp[:, n].tolist(), BW, signed) for n in range(NIJ)])

    act_file('x0hi', hi0, True)
    act_file('x0lo', lo0, False)
    for i in range(3):
        act_file('d{}'.format(i + 1), dX[i], True)
    for f in range(4):
        h, l = split(X[f])
        act_file('x{}hi'.format(f), h, True)
        act_file('x{}lo'.format(f), l, False)
    for kij in range(K * K):
        ky, kx = divmod(kij, K)
        write(os.path.join(args.out, 'weight_kij{}.txt'.format(kij)),
              ['#col_t=oc#', '#ic7[msb]...ic0[lsb]#', '#4b_signed#'],
              [pack(w[t, :, ky, kx].tolist(), BW, True) for t in range(ARRAY)])
    for f in range(4):
        write(os.path.join(args.out, 'y{}.txt'.format(f)),
              ['#{}#'.format('Y0=16*W*hi+W*lo' if f == 0 else 'Y{0}=Y0+W*dX{0}'.format(f)),
               '#col7[msb]...col0[lsb]#', '#line=o_nij#'],
              [pack(Y[f][o].tolist(), PSUM_BW, True) for o in range(ONIJ)])
    with open(os.path.join(args.out, 'config.txt'), 'w') as fh:
        fh.write('{} {} 4\n'.format(NIJ, ONIJ))

    # ---- analysis ----
    lines = ['Part 2 analysis (layout 8ch x 16 x 16, W = features[27] 4-bit, 3x3, pad 1)',
             'W range [{}, {}]; X0 hi range [{}, {}], lo range [{}, {}]'.format(
                 w.min(), w.max(), hi0.min(), hi0.max(), lo0.min(), lo0.max())]
    for i in range(3):
        d = dX_raw[i]
        lines.append('dX{}: range [{}, {}], zeros {:.1%}, saturated at 4 bit {:.2%}'.format(
            i + 1, d.min(), d.max(), (d == 0).mean(), sat[i]))
    lines.append('Y0 = 16*W*hi + W*lo equals W*X0: exact (max err 0)')
    sig = []
    for i in (1, 2, 3):
        e = (Y[i] - ref[i]).astype(np.float64)
        mse = float((e ** 2).mean())
        p_sig = float((ref[i].astype(np.float64) ** 2).mean())
        sqnr = float('inf') if mse == 0 else 10 * np.log10(p_sig / mse)
        sig.append(p_sig)
        lines.append('Y{}: max |err| {} | MSE {:.4g} | SQNR {} dB | max|Y| {}'.format(
            i, int(np.abs(e).max()), mse, 'inf' if mse == 0 else '{:.2f}'.format(sqnr), int(np.abs(ref[i]).max())))
    lines.append('16-bit accumulator check (limit {}): max|raw psum| {} | max|Y0 partial| {} | max|D partial| {} | '
                 'max|conventional partial| {} | max|Y| {}'.format(lim, max_raw, peak_y0, peak_d, peak_conv, peak_out))
    lines.append('|D_i| max {} vs |Y_i| max {} (deltas use {:.1f} of 16 bits)'.format(
        max(int(np.abs(d).max()) for d in D), peak_out, np.log2(max(int(np.abs(d).max()) for d in D)) + 1))
    with open(os.path.join(args.out, 'analysis.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
