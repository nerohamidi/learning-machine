"""Summarise the +alpha regression logs (alpha_logs/*.log written by run_alpha.sh) into markdown tables:
   1. pass/fail of every run
   2. zero-guard gating: same workload with +zg=0 and +zg=1 (activity counters from toggle_mon.vh)
   3. WS vs OS measured cycles / SRAM accesses on the VGG layers
Usage (from hardware/sim):  python3 summarize_alpha.py > alpha_logs/summary.md
"""

import glob
import os
import re

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpha_logs')


def parse(path):
    t = open(path).read()
    d = {'pass': ('RESULT: PASS' in t) or ('No error detected' in t)}
    m = re.search(r'MACs issued (\d+), zero-guarded (\d+).*?a_q (\d+)\s+b_q (\d+)\s+product (\d+)\s+c_q (\d+)\s+\(sum (\d+)\)'
                  r' \| reg write-enable bits (\d+)', t)
    if m:
        d.update(zip(['mac', 'guard', 'ta', 'tb', 'tp', 'tc', 'tsum', 'reg'], map(int, m.groups())))
    m = re.search(r'xmem rd (\d+) wr (\d+) \| wmem rd (\d+) wr (\d+) \| pmem rd (\d+) wr (\d+)', t)
    if m:
        d.update(zip(['xr', 'xw', 'wr_', 'ww', 'pr', 'pw'], map(int, m.groups())))
    m = re.search(r'total (\d+)', t)
    if m:
        d['total'] = int(m.group(1))
    m = re.search(r'compute \([^)]*\) (\d+)', t)
    if m:
        d['compute'] = int(m.group(1))
    m = re.search(r'accumulate (\d+)', t)
    if m:
        d['acc'] = int(m.group(1))
    m = re.search(r'cycles per head \(on-chip, excl. DRAM\): (\d+)', t)
    if m:
        d['per_head'] = int(m.group(1))
    return d


def main():
    runs = {os.path.basename(p)[:-4]: parse(p) for p in sorted(glob.glob(os.path.join(LOG, '*.log')))}
    runs = {k: v for k, v in runs.items() if 'pass' in v and k not in ('summary',)}
    print('## Runs\n')
    print('| run | result |\n|---|---|')
    for k, v in runs.items():
        print(f'| {k} | {"PASS" if v["pass"] else "**FAIL**"} |')

    print('\n## Zero-guard gating (same workload, +zg=0 vs +zg=1; outputs identical in both)\n')
    print('| workload | MACs issued | zero-guarded | a_q toggles | b_q toggles | product toggles | c_q toggles | '
          'operand+product toggles | reg write-enable bits |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k in runs:
        if not k.endswith('_zg0') or k[:-4] + '_zg1' not in runs:
            continue
        a, b = runs[k], runs[k[:-4] + '_zg1']
        if 'mac' not in a:
            continue
        red = lambda x, y: f'{y:,} ({(y - x) / x * 100:+.0f}%)' if x else f'{y:,}'
        op_a = a['ta'] + a['tb'] + a['tp']
        op_b = b['ta'] + b['tb'] + b['tp']
        print(f'| {k[:-4]} | {a["mac"]:,} | {b["guard"] / b["mac"] * 100:.1f}% | {red(a["ta"], b["ta"])} | '
              f'{red(a["tb"], b["tb"])} | {red(a["tp"], b["tp"])} | {red(a["tc"], b["tc"])} | {red(op_a, op_b)} | '
              f'{red(a["reg"], b["reg"])} |')

    print('\n## WS vs OS on the VGG layers (measured, zg=0)\n')
    print('| layer | WS (opt.) compute+acc | OS compute | speedup | WS SRAM rd (xmem / pmem) | OS SRAM rd (xmem / wmem) |'
          ' WS pmem wr | OS pmem wr |')
    print('|---|---|---|---|---|---|---|---|')
    for L in (27, 24, 29):
        w, o = runs.get(f'ws_f{L}_fast_zg0'), runs.get(f'os_f{L}_zg0')
        if not w or not o:
            continue
        wc = w['compute'] + w.get('acc', 0)
        print(f'| features[{L}] | {wc:,} | {o["compute"]:,} | {wc / o["compute"]:.2f}x | {w["xr"]:,} / {w["pr"]:,} | '
              f'{o["xr"]:,} / {o["wr_"]:,} | {w["pw"]:,} | {o["pw"]:,} |')

    heads = [(k, v['per_head']) for k, v in runs.items() if 'per_head' in v]
    if heads:
        print('\n## Linear attention core (cycles per head, on chip)\n')
        for k, v in heads:
            print(f'- {k}: {v}')


if __name__ == '__main__':
    main()
