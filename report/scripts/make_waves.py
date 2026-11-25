"""Timing diagrams for the report, rendered from the simulation VCDs (1 clock = 1000 ps).

    python scripts/make_waves.py   (from project/report/; needs ../hardware/sim/core_tb.vcd and core_tb_p2.vcd)

Signals are the registered instruction fields (the *_q copies the core actually sees) plus ofifo_valid.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType (embedded), not Type 3: NeurIPS font rule
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

INK, INK2, BLUE, ORANGE = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif'})
PERIOD = 1000


def read_vcd(path, names, t_max):
    """Returns {name: [(time, value_str), ...]} for top-scope variables named in `names`."""
    ids, depth, out = {}, 0, {n: [] for n in names}
    with open(path) as f:
        for ln in f:
            if ln.startswith('$scope'):
                depth += 1
            elif ln.startswith('$upscope'):
                depth -= 1
            elif ln.startswith('$var') and depth == 1:
                p = ln.split()
                if p[4] in names:
                    ids.setdefault(p[3], []).append(p[4])
            elif ln.startswith('$enddefinitions'):
                break
        t = 0
        for ln in f:
            c = ln[0]
            if c == '#':
                t = int(ln[1:])
                if t > t_max:
                    break
            elif c in '01xz' and ln[1:].strip() in ids:
                for n in ids[ln[1:].strip()]:
                    out[n].append((t, c))
            elif c == 'b':
                v, i = ln[1:].split()
                if i in ids:
                    for n in ids[i]:
                        out[n].append((t, v))
    return out


def bit_of(changes, bit):
    """Extract one bit (LSB = 0) of a vector signal."""
    res = []
    for t, v in changes:
        res.append((t, v[-1 - bit] if bit < len(v) else '0'))
    return res


def draw(ax, rows, c0, c1, title, pad=4):
    """rows: list of (label, changes, color, invert). Digital waveforms between cycles c0 and c1."""
    n = len(rows)
    for k, (label, ch, color, inv) in enumerate(rows):
        y0 = (n - 1 - k) * 1.5
        xs, ys, v = [], [], 0
        for t, s in ch:
            if t / PERIOD <= c0:
                v = s
        v = (v == '1') != inv
        xs, ys = [c0], [v]
        for t, s in ch:
            x = t / PERIOD
            if c0 < x < c1:
                nv = (s == '1') != inv
                xs += [x, x]; ys += [ys[-1], nv]
        xs.append(c1); ys.append(ys[-1])
        ys = [y0 + 0.9 * y for y in ys]
        ax.fill_between(xs, y0, ys, color=color, alpha=0.18, lw=0, step=None)
        ax.plot(xs, ys, color=color, lw=1.0)
        ax.text(c0 - (c1 - c0) * 0.01, y0 + 0.45, label, ha='right', va='center', color=INK, fontsize=7)
    ax.set_xlim(c0, c1); ax.set_ylim(-0.4, n * 1.5)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK2); ax.tick_params(colors=INK2)
    ax.set_xlabel('clock cycle'); ax.set_title(title, fontsize=8, color=INK, loc='left', pad=pad)


def marker(ax, x, text, top):
    ax.axvline(x, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(x + 6, top, text, fontsize=6.5, color=INK2, va='bottom')


# ---------------- Part 1: one kij of the template schedule (core_tb.v) ----------------
names = ['reset', 'l0_wr_q', 'load_q', 'l0_rd_q', 'execute_q', 'ofifo_valid', 'ofifo_rd_q', 'WEN_pmem_q', 'CEN_pmem_q',
         'acc_q', 'relu_q', 'pmem_wsel_q']
w = read_vcd('../hardware/sim/core_tb.vcd', names, 2300 * PERIOD)
# first kij pass: from the end of the first reset after the L1 load
starts = [t / PERIOD for t, v in w['reset'] if v == '0']
c0 = starts[1] - 13 if len(starts) > 1 else 0
fig, ax = plt.subplots(figsize=(6.8, 2.6))
rows = [('reset', w['reset'], INK2, False), ('l0_wr', w['l0_wr_q'], BLUE, False), ('load', w['load_q'], BLUE, False),
        ('l0_rd', w['l0_rd_q'], BLUE, False), ('execute', w['execute_q'], ORANGE, False),
        ('ofifo_valid', w['ofifo_valid'], ORANGE, False), ('ofifo_rd', w['ofifo_rd_q'], ORANGE, False),
        ('pmem write', w['WEN_pmem_q'], INK2, True)]
draw(ax, rows, c0, c0 + 205, 'Part 1, template schedule: one kij pass (kernel load, L0 fill, execution, OFIFO to pmem)')
fig.tight_layout(); fig.savefig('figs/wave_part1_kij.pdf'); plt.close(fig)

# Part 1: accumulation + ReLU at the end
acc_t = [t / PERIOD for t, v in w['acc_q'] if v == '1']
a0 = acc_t[0] - 3
fig, ax = plt.subplots(figsize=(6.8, 1.6))
rows = [('reset', w['reset'], INK2, False), ('pmem access', w['CEN_pmem_q'], BLUE, True), ('acc', w['acc_q'], ORANGE, False),
        ('pmem write', w['WEN_pmem_q'], INK2, True), ('wsel = SFU', w['pmem_wsel_q'], INK2, False)]
draw(ax, rows, a0, a0 + 45, 'Part 1: SFU accumulation of the 9 kij psums of one output pixel, then write-back')
fig.tight_layout(); fig.savefig('figs/wave_part1_acc.pdf'); plt.close(fig)

# ---------------- Part 2 (differential mode, core_tb_p2.v) ----------------
names = ['reset', 'load_q', 'execute_q', 'act_signed_q', 'ofifo_rd_q', 'CEN_pmem_rd_q', 'WEN_pmem_q', 'shift4_q',
         'add_q_q', 'src_ofifo_q', 'acc_ld_q', 'pmem_wsel_q']
w = read_vcd('../hardware/sim/core_tb_p2.vcd', names, 20000 * PERIOD)
ex = [t / PERIOD for t, v in w['execute_q'] if v == '1']
e0 = ex[0]                           # first execute of kij 0
fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.9), gridspec_kw={'height_ratios': [1.55, 1]})
rows = [('load', w['load_q'], BLUE, False), ('execute', w['execute_q'], ORANGE, False),
        ('act_signed', w['act_signed_q'], ORANGE, False), ('ofifo_rd', w['ofifo_rd_q'], BLUE, False),
        ('pmem read (port R)', w['CEN_pmem_rd_q'], BLUE, True), ('pmem write (port W)', w['WEN_pmem_q'], INK2, True),
        ('shift4', w['shift4_q'], INK2, False), ('add_q', w['add_q_q'], INK2, False)]
draw(axs[0], rows, e0 - 30, e0 + 1700, 'Part 2, kij = 0: weights loaded once, then 5 passes back to back '
     '(X0 hi | X0 lo | dX1 | dX2 | dX3)', pad=14)
for p, name in enumerate(['hi (signed, <<4)', 'lo (unsigned)', 'dX1', 'dX2', 'dX3']):
    marker(axs[0], e0 + 324 * p, name, 8 * 1.5 - 0.35)
# reconstruction: last cycles of the run
ld = [t / PERIOD for t, v in w['acc_ld_q'] if v == '1']
r0 = ld[0] - 3
rows = [('pmem read (port R)', w['CEN_pmem_rd_q'], BLUE, True), ('acc_ld', w['acc_ld_q'], ORANGE, False),
        ('pmem write (port W)', w['WEN_pmem_q'], INK2, True), ('add_q', w['add_q_q'], INK2, False),
        ('src_ofifo', w['src_ofifo_q'], INK2, False)]
draw(axs[1], rows, r0, r0 + 24, 'Part 2 reconstruction: per pixel read Y0, D1, D2, D3; write Y1, Y2, Y3 '
     '(4 cycles / pixel)')
fig.tight_layout(); fig.savefig('figs/wave_part2.pdf'); plt.close(fig)
print('kij0 first execute at cycle', e0, '| reconstruction starts near', r0)
