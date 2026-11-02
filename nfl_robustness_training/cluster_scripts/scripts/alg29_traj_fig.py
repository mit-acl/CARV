#!/usr/bin/env python3
"""One-scene illustration from alg29 traces (TTTCARV_GP_TRACE JSON).

usage: alg29_traj_fig.py OUT.png TRACE.json[:label] [TRACE.json[:label] ...]
Each trace is one panel: obstacles (raw r solid, D dashed), the flown path
(colored by what drove each step), and -- at the first step that flew a
fallback -- the cached plan's predicted path next to where the robot really is.
"""
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

KIND_C = {'plan': '#2a78d6', 'hold': '#8a8985', 'fallback': '#eb6834', 'loiter': '#e87ba4'}


def panel(ax, d, label):
    for o in d['obstacles']:
        ax.add_patch(Circle(o[:2], o[2], color='#c9c8c2', zorder=1))
        ax.add_patch(Circle(o[:2], math.sqrt(o[2] ** 2 + 2 * o[2]), fill=False,
                            ls='--', lw=0.8, ec='#8a8985', zorder=1))
    tr = d['trace']
    xs = [t['x'] for t in tr] + [d['final']]
    for i, t in enumerate(tr):
        ax.plot([xs[i][0], xs[i + 1][0]], [xs[i][1], xs[i + 1][1]],
                color=KIND_C.get(t['kind'], '#8a8985'), lw=2, solid_capstyle='round', zorder=3)
    # Mark the last fallback before the first collision (else the last
    # fallback) -- preferring one whose backup was already invalid from the
    # true state, which is the moment the filter "realises" too late.
    fc = d['stats'].get('first_collision', -1)
    cand = [t for t in tr if t['kind'] == 'fallback' and (fc < 0 or t['k'] < fc)]
    bad = [t for t in cand if t.get('cert_ok') is False]
    fb = (bad or cand or [None])[-1]
    if fc > 0:
        ax.plot(*xs[fc][:2], 'x', color='#d03b3b', ms=11, mew=2.5, zorder=6, label='collision')
    # zoom to the conflict: path within 3 m of any obstacle
    import numpy as _np
    near = [p for p in xs if min(math.hypot(p[0] - o[0], p[1] - o[1]) for o in d['obstacles']) < 2.5]
    if near:
        a = _np.array(near)
        ax.set_xlim(a[:, 0].min() - 0.8, a[:, 0].max() + 0.8)
        ax.set_ylim(a[:, 1].min() - 0.8, a[:, 1].max() + 0.8)
    if fb and fb['cache_states']:
        j = fb['k'] - fb['cache_t0']
        pl = fb['cache_states']
        ax.plot([p[0] for p in pl], [p[1] for p in pl], ':', color='#1a1a19', lw=1.2, zorder=2,
                label='cached backup (as planned)')
        if 0 <= j < len(pl):
            ax.plot(*pl[j][:2], 'o', mfc='white', mec='#1a1a19', ms=8, zorder=4,
                    label='where the cached backup expects the robot')
            ax.annotate(f't={fb["k"]}', fb['x'][:2], textcoords='offset points', xytext=(6, 6), fontsize=8)
        ax.plot(*fb['x'][:2], 'o', color='#eb6834', ms=8, zorder=5,
                label='where it actually is (filled = backup already invalid)'
                if fb.get('cert_ok') is False else 'where it actually is',
                mfc='#eb6834' if fb.get('cert_ok') is False else 'white')
    st = d['stats']
    ax.set_title(f"{label}\ncollision={st['collided']}  invalid-backup steps="
                 f"{st.get('cert_invalid', '?')}/{st.get('cert_checks', '?')}",
                 loc='left', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, color='#eeeeea', lw=0.6)


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    fig, axs = plt.subplots(1, len(specs), figsize=(6 * len(specs), 6.2), squeeze=False)
    for ax, spec in zip(axs[0], specs):
        path, _, label = spec.partition(':')
        panel(ax, json.load(open(path)), label or path)
    for k, c in KIND_C.items():
        axs[0][0].plot([], [], color=c, lw=2, label={'plan': 'fresh plan', 'hold': 'NN / hold during solve',
                                                      'fallback': 'cached-backup fallback',
                                                      'loiter': 'loiter'}[k])
    h, l, seen = [], [], set()
    for ax in axs[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll); h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    fig.savefig(out, dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    main()
