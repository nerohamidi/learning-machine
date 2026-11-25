"""Waveform figures for the reports: screenshots of the real GTKWave window (light colour scheme).

    cd project/report && <python> scripts/make_gtkwave_figs.py

<python> needs python-xlib and Pillow (e.g. `pip install python-xlib pillow`). Also needs an X display (WSLg's :0
works), xdotool, and GTKWave (default: the hdl conda env's binary; override with env GTKWAVE=/path/to/gtkwave).
Inputs: ../hardware/sim/core_tb.vcd and core_tb_p2.vcd (core_tb_p2.vcd from `vvp compiled_p2 +mode=diff +vcd`).

For each figure we write a .gtkwaverc (fonts, white background, blue traces), a save file (signals, aliases, inverted
active-low strobes, window size) and a Tcl script (zoom, named markers). GTKWave opens its window, we wait until the
traces are drawn, grab the window with Xlib, close GTKWave, and crop away the menu bar, toolbar and scrollbars.
Output: figs/gtk_*.png. Time axis: the testbench clock period is 1000 ps, so 1 ns = 1 clock cycle.

Why screenshots and not GTKWave's "Print to file": the PostScript printer uses letter-spaced Courier and squashed
traces, which looks unlike GTKWave on screen.
"""
import os
import subprocess
import sys
import tempfile
import time

from PIL import Image
from Xlib import X, display

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.dirname(HERE)
SIM = os.path.join(REPORT, '..', 'hardware', 'sim')
FIGS = os.path.join(REPORT, 'figs')
PERIOD = 1000  # ps per clock
WIN_W = 1500   # window width in px; fonts below are sized for this
TRACE = '1f4e9e'
RC = f"""fontname_signals Sans 20
fontname_waves Sans 15
hide_sst on
color_back ffffff
color_grid e4e4e4
color_trans {TRACE}
color_low {TRACE}
color_high {TRACE}
color_0 {TRACE}
color_1 {TRACE}
color_mid {TRACE}
color_vbox {TRACE}
color_vtrans {TRACE}
color_time 000000
color_timeb ffffff
color_value 000000
color_baseline 888888
color_normal 000000
color_mark b03020
color_umark b03020
"""


def gtkwave_bin():
    if os.environ.get('GTKWAVE'):
        return os.environ['GTKWAVE']
    out = subprocess.run(['conda', 'run', '-n', 'hdl', 'which', 'gtkwave'], capture_output=True, text=True)
    return out.stdout.strip() or 'gtkwave'


def read_vcd(path, names, t_max):
    """{name: [(time_ps, value), ...]} for top-scope 1-bit variables in `names`, up to t_max."""
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
    return out


def rises(changes):
    return [t // PERIOD for t, v in changes if v == '1']


def falls(changes):
    return [t // PERIOD for t, v in changes if v == '0']


def gtkw_file(top, signals):
    """signals: list of (vcd_name, alias or None, invert)."""
    height = 260 + 44 * len(signals)   # menu + toolbar + header + rows, with slack; the crop removes the rest
    lines = [f'[size] {WIN_W} {height}', '[signals_width] 150', '[sst_expanded] 0']
    for name, alias, inv in signals:
        lines.append('@68' if inv else '@28')          # 0x28 = binary, right-justified; 0x40 = invert
        full = f'{top}.{name}'
        lines.append(f'+{{{alias}}} {full}' if alias else full)
    return '\n'.join(lines) + '\n'


def tcl_file(c0, c1, markers):
    lines = [f'gtkwave::setZoomRangeTimes {c0 * PERIOD} {c1 * PERIOD}', 'update',
             f'gtkwave::setZoomRangeTimes {c0 * PERIOD} {c1 * PERIOD}']
    for k, (cyc, label) in enumerate(markers):
        lines.append(f'gtkwave::setNamedMarker {chr(65 + k)} {cyc * PERIOD} "{label}"')
    return '\n'.join(lines) + '\n'


def is_trace(p):
    return p[2] > p[0] + 40 and p[2] > 120


def grab(win):
    g = win.get_geometry()
    raw = win.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff)
    return Image.frombytes('RGB', (g.width, g.height), raw.data, 'raw', 'BGRX')


def wait_for_window(pid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(['xdotool', 'search', '--onlyvisible', '--pid', str(pid)], capture_output=True, text=True)
        ids = [int(x) for x in out.stdout.split()]
        if ids:
            return ids
        time.sleep(0.5)
    sys.exit('GTKWave window did not appear')


def capture(d, ids, timeout=120):
    """Grab until the image shows traces and stops changing (GTKWave renders after loading the VCD)."""
    t0, prev = time.time(), None
    while time.time() - t0 < timeout:
        best = None
        for wid in ids:
            try:
                im = grab(d.create_resource_object('window', wid))
            except Exception:
                continue
            if best is None or im.size[0] * im.size[1] > best.size[0] * best.size[1]:
                best = im
        if best is not None:
            small = best.resize((best.width // 4, best.height // 4))
            raw = small.tobytes()
            has_trace = any(is_trace(raw[i:i + 3]) for i in range(0, len(raw), 3))
            if has_trace and prev is not None and best.tobytes() == prev.tobytes():
                return best
            prev = best
        time.sleep(1.0)
    sys.exit('GTKWave window never finished drawing')


def crop(im):
    """Keep the signal-name pane and the wave pane (time header + signal rows); drop menus, toolbar, scrollbars."""
    w, h = im.size
    px = im.load()
    xs = range(w // 4, w - 40, 2)

    def white_frac(y):
        return sum(min(px[x, y]) >= 225 or is_trace(px[x, y]) for x in xs) / len(xs)   # background, grid or trace

    # the wave pane is the tallest band of mostly-white rows (the toolbar's white entry boxes are too narrow to count)
    band, start = (0, 0), None
    for y in range(h):
        if white_frac(y) > 0.3:
            start = y if start is None else start
            if y - start > band[1] - band[0]:
                band = (start, y)
        else:
            start = None
    y_top, y_end = band
    trace_rows = [y for y in range(y_top, y_end + 1) if any(is_trace(px[x, y]) for x in xs)]
    y_mid = (trace_rows[0] + trace_rows[-1]) // 2
    # right edge = end of the longest run of wave-area pixels on a row through the traces (stops at the scrollbar)
    best, run = (0, 0), None
    for x in range(w):
        p = px[x, y_mid]
        if p == (255, 255, 255) or is_trace(p) or p == (0xe4, 0xe4, 0xe4):
            run = x if run is None else run
            if x - run > best[1] - best[0]:
                best = (run, x)
        else:
            run = None
    return im.crop((3, y_top, best[1] + 1, trace_rows[-1] + 5))


def export(name, vcd, top, signals, c0, c1, markers):
    work = tempfile.mkdtemp(prefix='gtkfig_')
    for fn, text in [('.gtkwaverc', RC), ('fig.gtkw', gtkw_file(top, signals)), ('fig.tcl', tcl_file(c0, c1, markers))]:
        with open(os.path.join(work, fn), 'w') as f:
            f.write(text)
    proc = subprocess.Popen([gtkwave_bin(), '-S', 'fig.tcl', os.path.abspath(vcd), 'fig.gtkw'], cwd=work,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        d = display.Display()
        im = capture(d, wait_for_window(proc.pid))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    if os.environ.get("RAWDIR"):
        im.save(os.path.join(os.environ["RAWDIR"], "raw_" + name))
    out = crop(im)
    out.save(os.path.join(FIGS, name))
    print(f'{name}: cycles {c0}-{c1}, {len(signals)} signals, {out.size[0]} x {out.size[1]} px')


def main():
    os.makedirs(FIGS, exist_ok=True)

    # ---- Part 1, template schedule: one kij pass ----
    vcd1 = os.path.join(SIM, 'core_tb.vcd')
    w = read_vcd(vcd1, ['reset'], 2300 * PERIOD)
    starts = falls(w['reset'])
    c0 = starts[1] - 13 if len(starts) > 1 else 0
    export('gtk_part1_kij.png', vcd1, 'core_tb',
           [('clk', None, False), ('reset', None, False), ('l0_wr_q', 'l0_wr', False), ('load_q', 'load', False),
            ('l0_rd_q', 'l0_rd', False), ('execute_q', 'execute', False), ('ofifo_valid', None, False),
            ('ofifo_rd_q', 'ofifo_rd', False), ('WEN_pmem_q', 'pmem_write', True)],
           c0, c0 + 205, [])

    # ---- Part 2, differential mode: kij 0 (five passes) and reconstruction ----
    vcd2 = os.path.join(SIM, 'core_tb_p2.vcd')
    w = read_vcd(vcd2, ['execute_q', 'acc_ld_q'], 20000 * PERIOD)
    e0 = rises(w['execute_q'])[0]
    export('gtk_part2_kij0.png', vcd2, 'core_tb_p2',
           [('load_q', 'load', False), ('execute_q', 'execute', False), ('act_signed_q', 'act_signed', False),
            ('ofifo_rd_q', 'ofifo_rd', False), ('CEN_pmem_rd_q', 'pmem_read', True),
            ('WEN_pmem_q', 'pmem_write', True), ('shift4_q', 'shift4', False), ('add_q_q', 'add_q', False)],
           e0 - 60, e0 + 1700,
           [(e0 + 324 * p, lab) for p, lab in enumerate(['X0 hi', 'X0 lo', 'dX1', 'dX2', 'dX3'])])
    r0 = rises(w['acc_ld_q'])[0] - 3
    export('gtk_part2_recon.png', vcd2, 'core_tb_p2',
           [('clk', None, False), ('CEN_pmem_rd_q', 'pmem_read', True), ('acc_ld_q', 'acc_ld', False),
            ('WEN_pmem_q', 'pmem_write', True), ('add_q_q', 'add_q', False), ('src_ofifo_q', 'src_ofifo', False)],
           r0, r0 + 24, [])


if __name__ == '__main__':
    main()
