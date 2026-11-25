"""+alpha (linear attention): RTL regression over several ViT layers x test images (all 8 heads each).

For every (layer, image): capture the integer attention operands from the 4-bit ViT (hw_exact), check the
model's own proj pre-hook against the hardware-format result, write the stimulus and run core_tb_attn.
Log: hardware/sim/alpha_logs/attn_regression.log

Usage (from project/software/transformer, env ml; iverilog from env hdl):
    python attn_regression.py --layers 0 1 2 3 4 5 --images 0 1 2
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, '..', '..', 'hardware', 'sim')
VVP = os.path.expanduser('~/miniconda3/envs/hdl/bin/vvp')
sys.path.insert(0, HERE)
from gen_attn_stimulus import from_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    ap.add_argument('--images', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--zg', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(os.path.join(SIM, 'alpha_logs'), exist_ok=True)
    log = open(os.path.join(SIM, 'alpha_logs', 'attn_regression.log'), 'w')
    n_pass = n_run = 0
    for L in args.layers:
        for im in args.images:
            d = os.path.join(SIM, 'regress_alpha', f'attn_l{L}_i{im}')
            from_model(L, im, d)
            res = subprocess.run([VVP, 'compiled_attn', f'+dir={os.path.relpath(d, SIM)}', f'+zg={args.zg}'],
                                 cwd=SIM, capture_output=True, text=True).stdout
            ok = 'RESULT: PASS' in res
            n_pass += ok
            n_run += 1
            act = re.search(r'zero-guarded (\d+) \(([\d.]+)%\)', res)
            mm = re.search(r'mismatches: (\d+)', res)
            line = (f'layer {L} image {im}: {"PASS" if ok else "FAIL"}  mismatches {mm.group(1) if mm else "?"}  '
                    f'zero-guarded MACs {act.group(2) if act else "?"}%')
            print(line, flush=True)
            log.write(line + '\n')
    summary = f'{n_pass}/{n_run} (layer, image) cases pass, 8 heads each'
    print(summary)
    log.write(summary + '\n')


if __name__ == '__main__':
    main()
