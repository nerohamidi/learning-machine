"""+alpha (WS/OS switch): per-layer cycle / memory-traffic model of both dataflows on the 8x8 core, calibrated
on the RTL simulations, and the resulting per-layer mode choice for the full VGG16 and the tiny ViT.

Calibration (measured in hardware/sim, both match the RTL exactly for every measured layer):
  WS pass (one kernel load + one stream of L positions, OFIFO drained during execution) = L + 58 cycles
        (core_tb_tiled +fast=1: f27 9 x 94 = 846, f24 288 x 94 = 27072, f29 576 x 94 = 54144)
  OS tile (8 outputs x 8 channels, reduction of K steps, drain + write-back)              = K + 29 cycles
        (core_tb_os: f27 2 x 101 = 202, f29 128 x 101 = 12928)
The WS model uses Part 2's stream accumulation (OFIFO -> SFU + pmem read -> pmem), so there is no separate
accumulation phase and only the H x W outputs live in pmem; WS rows/cols that the layer does not fill are idle.
Energy proxy: Eyeriss's normalized access costs (Chen et al., ISCA'16; cited in the course slides Ch4:p10, Ch9:p22),
per 16b word relative to one 16b MAC: global buffer (our L1 SRAMs) 6x, DRAM 200x. Scaled to the word widths used
here (32b SRAM = 12, 128b SRAM = 48, 32b DRAM = 400), and a 4b x 4b MAC = 1/16 of a 16b MAC (cost ~ S^2, Ch5:p14).
Energies are in units of one 16b-MAC energy.
L1 capacity: xmem / wmem 2048 x 32b. An operand set that does not fit is re-fetched from DRAM for every
output-channel tile (WS: padded activations; OS: im2col activations, 9x larger for 3x3 kernels).

Usage (from project/software/):  python scripts/dataflow_model.py  > ../hardware/sim/alpha_logs/dataflow_model.md
"""

import math

OVH_WS, OVH_OS, ARR, L1 = 58, 29, 8, 2048
E_SRAM32, E_SRAM128, E_DRAM32, E_MAC = 12.0, 48.0, 400.0, 1 / 16


def ceil8(x):
    return (x + ARR - 1) // ARR


def ws(cin, cout, h, k=3):
    p = h + (k - 1)                       # padded side (k=1: no padding)
    nit, not_ = ceil8(cin), ceil8(cout)
    passes = nit * not_ * k * k
    cyc = passes * (p * p + OVH_WS)
    x_rd = passes * (p * p + ARR)         # activations + kernel words
    p_rw = 2 * passes * h * h             # stream accumulation: read + write per valid output and pass
    act = nit * p * p
    dram = (act if act <= L1 else act * not_) + nit * not_ * k * k * ARR
    return cyc, x_rd, 0, p_rw, dram


def os_(cin, cout, h, k=3):
    K = cin * k * k                       # no padding of C_in to a multiple of 8 is needed
    npt, not_ = ceil8(h * h), ceil8(cout)
    tiles = npt * not_
    cyc = tiles * (K + OVH_OS)
    act = npt * K                         # im2col (k*k duplication)
    dram = (act if act <= L1 else act * not_) + not_ * K   # weights: one 32b word per step per oc tile
    return cyc, tiles * K, tiles * K, tiles * ARR, dram


def energy(r, macs):
    cyc, x, w, p, dram = r
    return (x + w) * E_SRAM32 + p * E_SRAM128 + dram * E_DRAM32 + macs * E_MAC


def table(name, layers):
    print(f'\n### {name}\n')
    print('| layer | C_in→C_out, H×W | WS cycles | OS cycles | WS SRAM rd/wr (kB) | OS SRAM rd/wr (kB) | '
          'WS DRAM (kB) | OS DRAM (kB) | WS energy (10⁶ MAC₁₆) | OS energy (10⁶ MAC₁₆) | pick (EDP) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    tot = {'ws': [0, 0], 'os': [0, 0], 'sw': [0, 0]}
    for lname, cin, cout, h, k in layers:
        macs = cin * cout * h * h * k * k
        a, b = ws(cin, cout, h, k), os_(cin, cout, h, k)
        ea, eb = energy(a, macs), energy(b, macs)
        pick = 'WS' if a[0] * ea <= b[0] * eb else 'OS'
        kb = lambda r: (r[1] + r[2]) * 4 / 1024 + r[3] * 16 / 1024
        print(f'| {lname} | {cin}→{cout}, {h}×{h}{" (1×1)" if k == 1 else ""} | {a[0]:,} | {b[0]:,} | '
              f'{kb(a):,.0f} | {kb(b):,.0f} | {a[4] * 4 / 1024:,.1f} | {b[4] * 4 / 1024:,.1f} | '
              f'{ea / 1e6:.2f} | {eb / 1e6:.2f} | **{pick}** |')
        tot['ws'][0] += a[0]; tot['ws'][1] += ea
        tot['os'][0] += b[0]; tot['os'][1] += eb
        s = a if pick == 'WS' else b
        tot['sw'][0] += s[0]; tot['sw'][1] += ea if pick == 'WS' else eb
    print(f'\n| policy | cycles | energy (10⁶ MAC₁₆) | EDP vs WS-only |\n|---|---|---|---|')
    base = tot['ws'][0] * tot['ws'][1]
    for kname, lab in (('ws', 'WS only'), ('os', 'OS only'), ('sw', 'switcher (per-layer EDP pick)')):
        c, e = tot[kname]
        print(f'| {lab} | {c:,} | {e / 1e6:.2f} | {c * e / base:.3f} |')


def attention_model():
    """per head, N tokens, d = 8: measured switcher flow vs model estimates of single-dataflow and softmax flows."""
    print('\n### Attention core per head (N tokens, d_qk = 8, d_v = 7)\n')
    print('| N | linear, switcher (A: OS, B: WS) | linear, WS only | linear, OS only | softmax on the same array (model) |'
          ' state / score buffer (linear vs softmax) |')
    print('|---|---|---|---|---|---|')
    for N in (64, 256, 1024, 4096):
        # switcher, measured at N = 64 (core_tb_attn): A = 96, R = 20, B = 248 -> 364 cycles per head
        A = N + 32
        R = 20
        B = 2 * (N + OVH_WS + 1) + 2  # measured 248 at N = 64 (2 passes; +2: divider pipeline latency)
        sw = A + R + B
        # WS only: phase A as a WS GEMM: reduction over tokens on the rows -> N/8 token tiles, each a kernel load
        # of phiK (8 tokens x 8 features) and a stream of the 8 transposed V1 vectors, SFU accumulation
        ws_only = ceil8(N) * (ARR + OVH_WS) + R + B
        # OS only: phase B as OS tiles of 8 tokens, K = 8 (+ the hi/lo passes), and S8 must be transposed (8 cycles)
        os_only = A + R + 8 + 2 * ceil8(N) * (8 + OVH_OS)
        # softmax: QK^T (WS, K^T tiles of 8 keys stationary, stream N queries), SFU softmax (exp LUT, max and
        # sum passes: 3 passes over N^2/8 vectors), P V (OS, 8 x 8 tiles, K = N)
        smx = ceil8(N) * (N + OVH_WS) + 3 * N * ceil8(N) + ceil8(N) * (N + OVH_OS)
        print(f'| {N} | **{sw:,}** | {ws_only:,} | {os_only:,} | {smx:,} | 8×8×16b = 128 B vs N²×16b = '
              f'{N * N * 2 / 1024:,.0f} kB |')


if __name__ == '__main__':
    print('## Dataflow model: weight stationary vs output stationary (8x8 array, 4-bit)')
    vgg = [('conv1', 3, 64, 32, 3), ('conv2', 64, 64, 32, 3), ('conv3', 64, 128, 16, 3), ('conv4', 128, 128, 16, 3),
           ('conv5', 128, 256, 8, 3), ('conv6', 256, 256, 8, 3), ('conv7', 256, 256, 8, 3),
           ('f24 (squeezed)', 256, 8, 4, 3), ('f27 (squeezed)', 8, 8, 4, 3), ('f29 (squeezed)', 8, 512, 4, 3),
           ('conv11', 512, 512, 2, 3), ('conv12', 512, 512, 2, 3), ('conv13', 512, 512, 2, 3)]
    table('VGG16 (squeezed, CIFAR-10, one image)', vgg)
    vit = [('q,k,v proj', 128, 184, 8, 1), ('out proj', 56, 128, 8, 1), ('fc1', 128, 256, 8, 1),
           ('fc2', 256, 128, 8, 1)]
    table('Tiny ViT, one block, one image (N = 64 tokens = 8×8 "pixels", 1×1 GEMMs)', vit)
    attention_model()
