#!/usr/bin/env python3
"""alg29 figures from ~/runs/alg29_main fingerprints.

  fig_unicycle_latency.png  two panels, same x (latency delta, s):
      (a) % trials with a collision   (b) % audited steps whose backup is
      invalid from the true state. Series: G1 (NN during solve), G1 (hold),
      G2 (delay-compensated); noise none (solid) / uniform (dashed).
      Ours (pipelined, delta hidden) drawn as a reference marker at delta=dt.
  fig_di_latency.png        same for the DI.
usage: alg29_figs.py RUN_DIR
"""
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = {'g1': '#2a78d6', 'g1prev': '#eb6834', 'g2': '#1baf7a', 'g3': '#e87ba4',
     'r1': '#7d4fc9', 'r2': '#c9a227', 'ours': '#1a1a19'}
LBL = {'g1': 'G1 naive (NN runs during solve)', 'g1prev': 'G1 naive (hold last command)',
       'g2': 'G2 delay-compensated', 'g3': 'G3 = G2 + tightening',
       'r1': 'R1 robust tube PSF (literature timing)', 'r2': 'R2 robust tube PSF, delay-compensated'}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def rows(p):
    return [r for r in json.load(open(p))['trials'] if not r.get('error')]


def coll(r):
    return bool(r.get('collision', r.get('violation')))


def collect(run, prefix):
    out = {}
    for p in glob.glob(os.path.join(run, f'fp_{prefix}_*.json')):
        arm = os.path.basename(p)[3:-5]
        m = re.match(rf'{prefix}_(g1prev|g1|g2|g3|g0|r1|r2)_?([0-9.]*)_?(none|unif)?$', arm)
        if not m:
            continue
        fam, d, noise = m.group(1), m.group(2), m.group(3) or 'none'
        if fam == 'g0':
            d = '0'
        rs = rows(p)
        n = len(rs)
        k = sum(coll(r) for r in rs)
        ci = sum(r.get('gp_cert_invalid', 0) for r in rs)
        cc = sum(r.get('gp_cert_checks', 0) for r in rs)
        out.setdefault((fam, noise), []).append((float(d), n, k, ci, cc))
    # G0 is the delta = 0 point of every naive/compensated family
    for noise in ('none', 'unif'):
        g0 = out.get(('g0', noise))
        if g0:
            for fam in ('g1', 'g1prev', 'g2'):
                if (fam, noise) in out:
                    out[(fam, noise)] += g0
    return out


def panel(ax, data, which, ours=None, ylabel=''):
    for (fam, noise), pts in sorted(data.items()):
        if fam == 'g0':
            continue
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        if which == 'coll':
            ys = [100 * p[2] / p[1] for p in pts]
            lo = [100 * wilson(p[2], p[1])[0] for p in pts]
            hi = [100 * wilson(p[2], p[1])[1] for p in pts]
            ax.fill_between(xs, lo, hi, color=C[fam], alpha=0.12, linewidth=0)
        else:
            ys = [100 * p[3] / max(p[4], 1) for p in pts]
        ls = '-' if noise == 'none' else '--'
        ax.plot(xs, ys, ls, color=C[fam], lw=2, marker='o', ms=5,
                label=f"{LBL[fam]}{'' if noise == 'none' else ', noisy'}")
    if ours is not None:
        ax.plot([ours[0]], [ours[1]], marker='*', ms=14, color=C['ours'], ls='none',
                label=ours[2])
    ax.set_xlabel('computation latency  δ  (s)')
    ax.set_ylabel(ylabel)
    ax.grid(True, color='#e6e6e3', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_ylim(bottom=0)


def main():
    run = sys.argv[1]
    for prefix, name, ttl in (('u', 'unicycle', 'Unicycle, 500 paired scenes / arm'),
                              ('d', 'di', 'Double integrator (tight start), 500 trials / arm')):
        data = collect(run, prefix)
        if not data:
            continue
        ours = None
        op = os.path.join(run, f'fp_{prefix}_ours_pipe.json')
        if prefix == 'd':
            op = os.path.join(run, 'fp_d_alg22.json')
        if os.path.exists(op):
            rs = rows(op)
            ours = (0.2, 100 * sum(coll(r) for r in rs) / len(rs),
                    'ours (pipelined, δ hidden)' if prefix == 'u' else 'alg22_di (ours, exact state)')
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
        panel(axs[0], data, 'coll', ours, '% trials with a collision (95% CI)')
        panel(axs[1], data, 'cert', (ours[0], 0.0, ours[2] + ': audits 0') if (ours and prefix == 'u') else None,
              '% steps whose backup is invalid\nfrom the TRUE state')
        axs[0].set_title('(a) safety', loc='left', fontsize=11)
        axs[0].text(0.02, 0.97, 'G1 sawtooth: a fractional δ leaves the NN\n'
                    '(δ mod dt)/dt of EVERY step; at δ = dt the late\n'
                    'plan still gets whole steps (dt = 0.2 s)',
                    transform=axs[0].transAxes, va='top', fontsize=8, color='#555550')
        axs[1].set_title('(b) certificate validity', loc='left', fontsize=11)
        h, l = axs[1].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9)
        fig.suptitle(ttl, x=0.01, ha='left', fontsize=12)
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        out = os.path.join(run, f'fig_{name}_latency.png')
        fig.savefig(out, dpi=200)
        print('wrote', out)


if __name__ == '__main__':
    main()
