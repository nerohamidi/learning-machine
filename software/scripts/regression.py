"""Part 1 Step 6 (+alpha): RTL regression over many CIFAR-10 test images and three layers.

For every (layer, image) case: generate stimulus (gen_tiled_stimulus.generate), run core_tb_tiled with the
template schedule and the optimized schedule, and record PASS/FAIL and cycle counts.

Usage (from project/software/):
    python scripts/regression.py --images 50 --tiled_images 5
"""

import argparse
import os
import re
import subprocess
import sys
import time

import torch

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_tiled_stimulus import generate, load_model_and_data  # noqa: E402

SIM = os.path.abspath(os.path.join(SW, '..', 'hardware', 'sim'))
VVP = os.path.expanduser('~/miniconda3/envs/hdl/bin/vvp')


def run_sim(stim_dir, fast):
    out = subprocess.run([VVP, '-n', 'compiled_tiled', '+dir=' + stim_dir, '+fast={}'.format(fast)],
                         cwd=SIM, capture_output=True, text=True, timeout=1800).stdout
    ok = 'RESULT: PASS' in out
    total = int(re.search(r'total (\d+)', out).group(1)) if 'total' in out else -1
    mism = re.search(r'mismatches: (\d+)', out)
    return ok, total, int(mism.group(1)) if mism else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=int, default=50, help='images for the squeezed layer features[27]')
    p.add_argument('--tiled_images', type=int, default=5, help='images for the tiled layers features[24]/[29]')
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    args = p.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, test = load_model_and_data(args.ckpt, device)

    cases = [(27, n) for n in range(args.images)] + \
            [(layer, n) for layer in (24, 29) for n in range(args.tiled_images)]
    log = open(os.path.join(SIM, 'regression.log'), 'w')
    summary = {}
    t0 = time.time()
    for layer, n in cases:
        rel = os.path.join('stim_reg', 'f{}_img{}'.format(layer, n))
        generate(model, test, n, layer, os.path.join(SIM, rel), device, quiet=True)
        ok0, cyc0, m0 = run_sim(rel, 0)
        ok1, cyc1, m1 = run_sim(rel, 1)
        line = 'features[{}] image {:3d}: template {} ({} cyc, {} mism) | optimized {} ({} cyc, {} mism)'.format(
            layer, n, 'PASS' if ok0 else 'FAIL', cyc0, m0, 'PASS' if ok1 else 'FAIL', cyc1, m1)
        print(line, flush=True)
        log.write(line + '\n')
        s = summary.setdefault(layer, {'cases': 0, 'pass': 0, 'cyc0': cyc0, 'cyc1': cyc1})
        s['cases'] += 1
        s['pass'] += int(ok0 and ok1)
    lines = ['', 'summary ({:.0f} s):'.format(time.time() - t0)]
    for layer, s in summary.items():
        lines.append('  features[{}]: {}/{} images pass (both schedules) | cycles template {} -> optimized {} '
                     '({:.2f}x)'.format(layer, s['pass'], s['cases'], s['cyc0'], s['cyc1'], s['cyc0'] / s['cyc1']))
    print('\n'.join(lines))
    log.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
