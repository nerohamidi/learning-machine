"""Report figures: training curve (Part 1 Step 1) and Part 2 cycle breakdown. Run from project/report/."""
import re
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType (embedded), not Type 3: NeurIPS font rule
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})

# ---- training curve ----
ep, tr, te = [], [], []
for ln in open('../software/result/vgg16_quant_4bit_squeeze8/train_log.txt'):
    m = re.match(r'epoch\s+(\d+).*train loss [\d.]+ acc ([\d.]+)% \| test ([\d.]+)%', ln)
    if m:
        ep.append(int(m[1])); tr.append(float(m[2])); te.append(float(m[3]))
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.plot(ep, tr, color=BLUE, lw=1.5, label='train')
ax.plot(ep, te, color=ORANGE, lw=1.5, label='test')
ax.axhline(90, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(1, 90.3, '90% target', color=INK2, fontsize=7)
b = max(range(len(te)), key=lambda i: te[i])
ax.plot(ep[b], te[b], 'o', ms=4, color=ORANGE, mec='white', mew=1)
ax.annotate('best test {:.2f}%'.format(te[b]), (ep[b], te[b]), xytext=(-70, -22),
            textcoords='offset points', fontsize=7, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax.set_xlabel('epoch'); ax.set_ylabel('accuracy (%)'); ax.set_ylim(83, 100.5)
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='lower right', ncol=2)
fig.tight_layout(); fig.savefig('figs/training_curve.pdf'); plt.close(fig)

# ---- Part 2 cycles ----
# measured totals include 14 cycles of initial reset/setup that are not in any segment
rows = [('conventional', 2681, 25384, 0), ('differential', 1706, 15094, 1026)]
totals = {'conventional': 28079, 'differential': 17840}
fig, ax = plt.subplots(figsize=(6.6, 1.45))
labels = ['DRAM$\\rightarrow$xmem', 'compute (weight load + execute + OFIFO$\\rightarrow$SFU$\\rightarrow$pmem)', 'reconstruction $Y_i=Y_0+D_i$']
for k, (name, *segs) in enumerate(rows):
    left = 0
    for s, (v, c) in enumerate(zip(segs, (BLUE, ORANGE, AQUA))):
        if v:
            ax.barh(k, v, left=left, height=0.55, color=c, edgecolor='white', linewidth=1.5,
                    label=labels[s] if k == 1 else None)
        left += v
    ax.text(left + 300, k, '{:,} cycles'.format(totals[name]), va='center', fontsize=7.5, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 32000); ax.set_xlabel('clock cycles (simulated)')
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fontsize=7, handlelength=1)
fig.tight_layout(); fig.savefig('figs/part2_cycles.pdf'); plt.close(fig)
print('ok')
