"""Intervention vs safety: ours (alg28) vs literature PSFs with the pin probe
(TTTCARV_GP_PIN=1, best-case minimal intervention), every arm on the textbook
stage-0 cost (rev 7). Reads ~/runs/alg38_textbook, where
both modules count |applied - NN| > THR from the same per-step list.
Writes fig_intervention.png next to this file.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = os.path.expanduser('~/runs')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_intervention.png')


THR = '0.05'   # identical for every arm; ours is flat in THR, see alg29_results.md rev 6


def load(path, ours=False):
    T = json.load(open(os.path.join(RUNS, path)))['trials']
    p = '' if ours else 'gp_'
    n = sum(t[p + 'u_diff_len'] for t in T)
    k = sum(t[p + 'u_diff_gt'][THR] for t in T)
    return 100 * k / n, sum(t['collision'] for t in T)


ARMS = [  # label, colour, marker, uniform path, trunc path, ours?
    ('G0 textbook, δ=0 (point)', '#d62728', 'o', 'alg38_textbook/fp_u_g0_unif_pin.json', 'alg38_textbook/fp_u_g0_trunc_pin.json', False),
    ('G2 textbook + delay comp. (point)', '#ff9896', 'o', 'alg38_textbook/fp_u_g2_0.1_unif_pin.json', 'alg38_textbook/fp_u_g2_0.1_trunc_pin.json', False),
    ('G3 box-tightened, δ=0', '#9467bd', 'D', 'alg38_textbook/fp_u_g3_0.0_unif_pin.json', None, False),
    ('R1 W&Z tube, δ=0', '#1f77b4', 's', 'alg38_textbook/fp_u_r1_0.0_unif_pin.json', 'alg38_textbook/fp_u_r1_0.0_trunc_pin.json', False),
    ('R2 W&Z tube + delay comp.', '#6baed6', 's', 'alg38_textbook/fp_u_r2_0.1_unif_pin.json', 'alg38_textbook/fp_u_r2_0.1_trunc_pin.json', False),
    ('ours, pipelined', '#2ca02c', '*', 'alg38_textbook/fp_u_ours_pipe.json', 'alg38_textbook/fp_u_ours_pipe_trunc.json', True),
]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))

# (a) the cost-design artefact: unpinned vs pinned
pairs = [('G0', 'alg38_textbook/fp_u_g0_unif_nopin.json', 'alg38_textbook/fp_u_g0_unif_pin.json', '#d62728'),
         ('R2', 'alg38_textbook/fp_u_r2_0.1_unif_nopin.json', 'alg38_textbook/fp_u_r2_0.1_unif_pin.json', '#6baed6')]
for i, (lab, a, b, col) in enumerate(pairs):
    (ra, ca), (rb, cb) = load(a), load(b)
    ax[0].bar(i - 0.2, ra, 0.38, color=col, alpha=.35, edgecolor='k', label='single unpinned solve' if i == 0 else None)
    ax[0].bar(i + 0.2, rb, 0.38, color=col, edgecolor='k', label='pinned when certifiable (W&Z optimum)' if i == 0 else None)
    ax[0].text(i - 0.2, ra + 1, f'{ra:.0f}%\n{ca} coll.', ha='center', fontsize=9)
    ax[0].text(i + 0.2, rb + 1, f'{rb:.1f}%\n{cb} coll.', ha='center', fontsize=9)
ro, co = load('alg38_textbook/fp_u_ours_pipe.json', True)
ax[0].bar(2, ro, 0.38, color='#2ca02c', edgecolor='k')
ax[0].text(2, ro + 1, f'{ro:.1f}%\n{co} coll.', ha='center', fontsize=9)
ax[0].set_xticks([0, 1, 2], ['G0 textbook, δ=0', 'R2 W&Z tube + comp.', 'ours, pipelined'])
ax[0].set_ylabel('steps with |applied − NN| > 0.05  [%]')
ax[0].set_ylim(0, 90)
ax[0].set_title('(a) A single solve rarely returns the NN input even when it is safe;\n'
                'pinning it when certifiable gives the PSF optimum (uniform noise)')
ax[0].legend(fontsize=9)
ax[0].grid(alpha=.3, axis='y')

# (b) the trade-off: intervention vs collisions
for lab, col, mk, pu, pt, ours in ARMS:
    for path, fill, law in ((pu, True, 'uniform'), (pt, False, 'bounded Gaussian')):
        if path is None:
            continue
        r, c = load(path, ours)
        ax[1].plot(r, 100 * c / 500, mk, ms=16 if mk == '*' else 10, color=col,
                   mfc=col if fill else 'none', mew=2, mec=col if not fill else 'k',
                   label=lab if fill else None)
        ax[1].annotate(f'{c}', (r, 100 * c / 500), textcoords='offset points', xytext=(8, 4), fontsize=8)
ax[1].plot([], [], 'o', color='gray', mfc='gray', mec='k', label='filled: uniform noise')
ax[1].plot([], [], 'o', color='gray', mfc='none', mew=2, label='hollow: bounded-Gaussian noise')
ax[1].set_xlabel('steps with |applied − NN| > 0.05  [%]   (all filters at best-case minimal intervention)')
ax[1].set_ylabel('collisions  [% of 500 paired scenes]')
ax[1].set_title('(b) Point-based PSFs that intervene minimally are unsafe under noise;\n'
                'every 0-collision filter intervenes more than ours')
ax[1].grid(alpha=.3)
ax[1].legend(fontsize=8, loc='upper right')
ax[1].set_xlim(0, 20)

fig.suptitle('Intervention at equal safety — unicycle, 500 paired scenes per arm, measurement noise bounded at 0.15', fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
