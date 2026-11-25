"""+alpha figure: attention-core cycles per head vs sequence length N (measured point at N = 64, model elsewhere).
Run from report/ (env ml):  python scripts/make_alpha_plots.py  -> figs/alpha_attn_scaling.pdf"""
import os, sys
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType (embedded), not Type 3: NeurIPS font rule
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'software', 'scripts'))
from dataflow_model import OVH_WS, OVH_OS, ceil8

Ns = [64, 128, 256, 512, 1024, 2048, 4096]
lin = [(N + 32) + 20 + 2 * (N + OVH_WS + 1) + 2 for N in Ns]
smx = [ceil8(N) * (N + OVH_WS) + 3 * N * ceil8(N) + ceil8(N) * (N + OVH_OS) for N in Ns]
fig, ax = plt.subplots(figsize=(4.2, 2.8))
c1, c2, ink = '#2a78d6', '#eb6834', '#555'
ax.plot(Ns, smx, color=c2, lw=2, marker='o', ms=4, label='softmax attention (model)')
ax.plot(Ns, lin, color=c1, lw=2, marker='o', ms=4, label='linear attention, WS/OS switch')
ax.plot([64], [364], 'o', ms=9, mfc='none', mec=ink, mew=1.5)
ax.annotate('measured in RTL: 364', (64, 364), xytext=(85, 1500), fontsize=7.5, color=ink)
ax.text(4096, lin[-1] * 1.4, 'O(N)', color=ink, fontsize=8, ha='right')
ax.text(2048, smx[-2] * 2.5, 'O(N²)', color=ink, fontsize=8, ha='right')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('tokens N', fontsize=8); ax.set_ylabel('cycles per head (8×8 array)', fontsize=8)
ax.tick_params(labelsize=7, colors=ink)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(True, which='major', color='#e5e5e5', lw=0.6)
ax.legend(fontsize=7, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', 'alpha_attn_scaling.pdf'))
fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', 'alpha_attn_scaling.png'), dpi=150)
