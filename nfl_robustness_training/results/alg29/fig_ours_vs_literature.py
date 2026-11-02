"""Ours (alg28) vs the literature PSFs (alg29 G*/R*), from the rev 3-5 fingerprints.

Reads ~/runs/alg31_term (rev 3), ~/runs/alg32_shrink (rev 4) and ~/runs/alg33_trunc
(rev 5). Writes fig_ours_vs_literature.png next to this file.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = os.path.expanduser('~/runs')
R3, R4, R5 = (os.path.join(RUNS, d) for d in ('alg31_term', 'alg32_shrink', 'alg33_trunc'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_ours_vs_literature.png')

C = dict(G='#d62728', G2='#ff9896', R='#1f77b4', R2='#9ecae1', G3='#9467bd', ours='#2ca02c')


def load(run, arm):
    T = json.load(open(os.path.join(run, f'fp_{arm}.json')))['trials']
    s = lambda k: sum(t.get(k, 0) for t in T)
    d = dict(n=len(T), coll=s('collision'))
    if 'gp_mode' in T[0]:
        d['cert'] = s('gp_cert_invalid') / max(s('gp_cert_checks'), 1)
        d['truth_out'] = s('gp_truth_out') / max(s('gp_cert_checks'), 1)
        d['fail'] = s('gp_solve_fail')
        d['miss'] = s('gp_deadline_miss')
        d['wmax'] = np.array([t['gp_solve_wall_max'] for t in T])
    else:
        d['cert'] = s('snd_audit_miss') / max(s('snd_audit_pts'), 1)
        d['truth_out'] = s('setmem_truth_out_tight') / max(s('setmem_steps'), 1)
    return d


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (p - max(c - h, 0)), 100 * (c + h - p)


fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
ax = ax.ravel()

# ---- (a,b) latency sweep (rev 4, shrink on) --------------------------------
X = ['0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.4', 'meas.']
xi = {x: i for i, x in enumerate(X)}
series = [
    ('G1 textbook PSF, solve at x(k), apply late', C['G'], 'o-',
     {'0': 'u_g0_none', '0.05': 'u_g1_0.05_none', '0.1': 'u_g1_0.1_none', '0.15': 'u_g1_0.15_none',
      '0.2': 'u_g1_0.2_none', '0.3': 'u_g1_0.3_none', '0.4': 'u_g1_0.4_none', 'meas.': 'u_g1_meas_none'}),
    ('R1 W&Z tube PSF, apply late', C['R'], 's-',
     {'0': 'u_r1_0.0_unif', '0.05': 'u_r1_0.05_unif', '0.1': 'u_r1_0.1_unif', '0.15': 'u_r1_0.15_unif',
      '0.2': 'u_r1_0.2_unif', '0.3': 'u_r1_0.3_unif', 'meas.': 'u_r1_meas_unif'}),
    ('G2 textbook + delay compensation', C['G2'], 'o--',
     {'0.1': 'u_g2_0.1_none', '0.2': 'u_g2_0.2_none', '0.4': 'u_g2_0.4_none', 'meas.': 'u_g2_meas_unif'}),
    ('R2 W&Z tube + delay compensation', C['R2'], 's--',
     {'0.1': 'u_r2_0.1_unif', '0.2': 'u_r2_0.2_unif', '0.4': 'u_r2_0.4_unif', 'meas.': 'u_r2_meas_unif'}),
]
for lab, col, sty, arms in series:
    xs, ys, lo, hi, cs = [], [], [], [], []
    for x, a in arms.items():
        d = load(R4, a)
        p, l, h = wilson(d['coll'], d['n'])
        xs.append(xi[x]); ys.append(p); lo.append(l); hi.append(h); cs.append(100 * d['cert'])
    ax[0].errorbar(xs, ys, yerr=[lo, hi], fmt=sty, color=col, label=lab, capsize=3, ms=6)
    ax[1].plot(xs, cs, sty, color=col, label=lab, ms=6)
ours = load(R4, 'u_ours_pipe')
p, l, h = wilson(ours['coll'], ours['n'])
for a, y in ((ax[0], p), (ax[1], 100 * ours['cert'])):
    a.plot([xi['meas.']], [y], '*', color=C['ours'], ms=20, mec='k', zorder=5,
           label='ours, pipelined (real solve time)')
ax[0].errorbar([xi['meas.']], [p], yerr=[[l], [h]], color=C['ours'], capsize=3)
for a in ax[:2]:
    a.set_xticks(range(len(X)), X)
    a.set_xlabel('computation delay δ  [s]   (one control tick = 0.2 s)')
    a.grid(alpha=.3)
ax[0].set_ylabel('collisions  [% of 500 scenes, 95% CI]')
ax[0].set_title('(a) Latency: literature PSFs collide unless the\ndelay is compensated; ours 0 at real solve time')
ax[0].legend(fontsize=8, loc='upper left')
ax[1].set_ylabel('steps whose active certificate is FALSE\nfrom the true state  [%]')
ax[1].set_title('(b) …and their certificates are wrong far more\noften than they collide')

# ---- (c,d,e) noise laws (rev 4 uniform, rev 5 trunc/gauss) ------------------
arms = [('G0\ntextbook\n(point)', 'u_g0', C['G']), ('G2 δ=.1\ncompens.\n(point)', 'u_g2_0.1', C['G2']),
        ('G3\nbox-tight\n(box)', 'u_g3_0.0', C['G3']), ('R1 δ=0\nW&Z tube\n(tube)', 'u_r1_0.0', C['R']),
        ('R2 δ=.1\nW&Z+comp.\n(tube)', 'u_r2_0.1', C['R2']), ('ours\nideal\n(set)', 'u_ours_ideal', C['ours']),
        ('ours\npipelined\n(set)', 'u_ours_pipe', C['ours'])]


def arm_law(a, law):
    if law == 'uniform':
        if a.startswith('u_ours'):
            return load(R4, a)
        return load(R4, a + '_unif')
    return load(R5, f'{a}_{law}')


laws = [('uniform', '//'), ('trunc', ''), ('gauss', 'xx')]
lawlab = {'uniform': 'uniform  (bound holds)', 'trunc': 'bounded Gaussian  (bound holds)',
          'gauss': 'unbounded Gaussian  (bound VIOLATED)'}
D = {(a, law): arm_law(a, law) for _, a, _ in arms for law, _ in laws}
w = 0.27
for j, (law, hatch) in enumerate(laws):
    for i, (lab, a, col) in enumerate(arms):
        d = D[(a, law)]
        x = i + (j - 1) * w
        if law != 'gauss':
            v = 100 * d['cert']
            ax[2].bar(i + (j - .5) * w, v, w, color=col, hatch=hatch, edgecolor='k', lw=.6,
                      label=lawlab[law] if i == 0 else None)
            if v == 0:
                ax[2].text(i + (j - .5) * w, 0.004, '0', ha='center', fontsize=8)
        ax[3].bar(x, d['coll'], w, color=col, hatch=hatch, edgecolor='k', lw=.6,
                  label=lawlab[law] if i == 0 else None)
        ax[3].text(x, d['coll'] + .05, str(d['coll']), ha='center', fontsize=8)
        ax[4].bar(x, 100 * d['truth_out'], w, color=col, hatch=hatch, edgecolor='k', lw=.6,
                  label=lawlab[law] if i == 0 else None)
for a in ax[2:5]:
    a.set_xticks(range(len(arms)), [l for l, _, _ in arms], fontsize=8)
    a.grid(alpha=.3, axis='y')
ax[2].set_ylabel('steps with certificate FALSE from true state  [%]')
ax[2].set_title('(c) Noise: a peaked (bounded-Gaussian) law doubles\nstale point certificates; box/tube/set stay at 0')
ax[2].legend(fontsize=8, title='measurement noise', title_fontsize=8)
ax[3].set_ylabel('collisions  (of 500 scenes)')
ax[3].set_title('(d) Collisions by noise law. Bound holds: ours 0, G0 1.\nBound violated: ours 1 / 3, G0 0')
ax[3].legend(fontsize=8)
ax[4].set_ylabel('steps the truth is OUTSIDE the estimated set  [%]')
ax[4].set_title('(e) Why ours loses under unbounded noise: it relies on\nthe set containing the truth; G0/G2 never use the set')
ax[4].legend(fontsize=8)

# ---- (f) W&Z shrinking-horizon re-solve: what it costs in wall clock --------
for arm, col, lab in (('u_g2_meas_unif', C['G2'], 'G2 compensated'), ('u_r2_meas_unif', C['R2'], 'R2 W&Z + comp.')):
    for run, ls, tag in ((R3, ':', 'cached tail (rev 3)'), (R4, '-', 'W&Z Alg. 2 re-solve (rev 4)')):
        d = load(run, arm)
        v = np.sort(d['wmax']) * 1e3
        ax[5].plot(v, 1 - np.arange(len(v)) / len(v), ls, color=col, lw=2,
                   label=f'{lab}, {tag}: {d["fail"]} failed solves, {d["miss"]} deadline misses')
ax[5].axvline(200, color='k', lw=1)
ax[5].text(203, 0.5, '0.20 s budget', rotation=90, va='center', fontsize=8)
ax[5].set_yscale('log')
ax[5].set_xlabel('worst solve time in the trial  [ms]')
ax[5].set_ylabel('fraction of trials exceeding')
ax[5].set_title('(f) Being faithful to W&Z (shrinking re-solve) fixes\nfailed solves but pushes the worst case past the budget')
ax[5].legend(fontsize=7, loc='lower left')
ax[5].grid(alpha=.3, which='both')

fig.suptitle('Ours (set-based, pipelined) vs literature predictive safety filters (Wabersich & Zeilinger 2021 style) — '
             'unicycle, 500 paired scenes per arm', fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
