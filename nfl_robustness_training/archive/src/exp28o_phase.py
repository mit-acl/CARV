"""DI phase-plane figure for exp28o: certified sets and rule X vs rule S lookahead.

(a) one real rollout of the certified-set loop in (p, v): truth, NN input (centre of S_t), S_t boxes.
(b) lookahead from S_t at t = T0: J256 joint reach boxes under rule X and rule S, with adversarial
    inner samples (x_0 in S_t, e_k in the rule-S box).
(c) reach-box widths vs k (p and v) for X, S, and the adversarial inner estimate.

usage: exp28o_phase.py OUT.png [SEED] [T0]
"""
import itertools
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import exp28o_di_ebox as E


def rollout(seed, T=30):
    rng = np.random.default_rng(seed)
    x = rng.uniform(E.INIT[:, 0], E.INIT[:, 1])
    z = x + rng.uniform(-E.VMAX, E.VMAX, 2)
    S = E.meet(E.INIT, np.stack([z - E.VMAX, z + E.VMAX], 1))
    xs, Ss = [x.copy()], [S.copy()]
    for _ in range(T):
        u = E.nn_u(S.mean(1))[0]
        x = E.f_np(x[None], [u])[0]
        z = x + rng.uniform(-E.VMAX, E.VMAX, 2)
        S = E.meet(E.istep(S, u), np.stack([z - E.VMAX, z + E.VMAX], 1))
        xs.append(x.copy()); Ss.append(S.copy())
    return np.array(xs), Ss


def adv_points(S, rng, n0=150):
    """Same sampler as E.adv_set, but returns the point clouds per step."""
    corners = np.array(list(itertools.product(*[(S[i, 0], S[i, 1]) for i in range(2)])))
    X0 = np.vstack([corners, rng.uniform(S[:, 0], S[:, 1], (n0, 2))])
    hs, h = [], E.h_next(E.hw(S))
    for _ in range(E.K - 1):
        hs.append(h.copy()); h = E.h_next(h)
    seqs = [np.array([np.array(c) * hh for hh in hs]) for c in itertools.product((-1.0, 1.0), repeat=2)]
    seqs += [np.array([np.sign(rng.uniform(-1, 1, 2)) * hh for hh in hs]) for _ in range(24)]
    seqs += [np.array([rng.uniform(-1, 1, 2) * hh for hh in hs]) for _ in range(8)]
    seqs = np.array(seqs)
    X = np.repeat(X0, len(seqs), 0)
    Es = np.tile(seqs, (len(X0), 1, 1))
    X = E.f_np(X, np.full(len(X), E.nn_u(S.mean(1))[0]))
    tr = [X]
    for k in range(E.K - 1):
        X = E.f_np(X, E.nn_u(X + Es[:, k])); tr.append(X)
    return tr


def box(ax, b, **kw):
    ax.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0], fill=False, **kw))


def main():
    torch.set_num_threads(1)
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    rng = np.random.default_rng(1000 * seed + t0)
    xs, Ss = rollout(seed)
    S = Ss[t0]
    s = E.SPLITS['J256']
    chX, _ = E.chain(S, 'X', 'J', s)
    chS, _ = E.chain(S, 'S', 'J', s)
    pts = adv_points(S, rng)
    adv = [np.stack([p.min(0), p.max(0)], 1) for p in pts]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))

    ax = axs[0]
    for b in Ss:
        box(ax, b, ec='tab:green', lw=0.7, alpha=0.6)
    cs = np.array([b.mean(1) for b in Ss])
    ax.plot(xs[:, 0], xs[:, 1], 'k.-', lw=1, ms=3, label='true state $x_t$')
    ax.plot(cs[:, 0], cs[:, 1], 'x', color='tab:green', ms=4, label='NN input = centre of $S_t$')
    box(ax, E.INIT, ec='grey', ls=':', lw=1)
    ax.plot([], [], color='tab:green', lw=0.7, label='certified set $S_t$')
    ax.axvline(0, color='r', lw=1, ls='--', label='constraint $p \\geq 0$')
    ax.plot(*xs[t0], 'o', mfc='none', mec='tab:purple', ms=10, label=f'lookahead start t={t0}')
    ax.set_xlabel('position p'); ax.set_ylabel('velocity v')
    ax.set_title(f'(a) closed loop, seed {seed}, uniform noise $v_{{max}}$={E.VMAX:.2f}')
    ax.legend(fontsize=8, loc='lower right')

    ax = axs[1]
    allp = np.vstack(pts)
    ax.scatter(allp[:, 0], allp[:, 1], s=0.3, c='0.6', alpha=0.25, rasterized=True)
    for k in range(E.K):
        box(ax, chX[k], ec='tab:red', lw=1.0)
        box(ax, chS[k], ec='tab:blue', lw=1.0)
    box(ax, S, ec='tab:green', lw=1.5)
    ax.plot(xs[t0:t0 + E.K + 1, 0], xs[t0:t0 + E.K + 1, 1], 'k.-', lw=1, ms=3)
    ax.plot([], [], color='tab:red', label='rule X reach boxes (J256)')
    ax.plot([], [], color='tab:blue', label='rule S reach boxes (J256)')
    ax.plot([], [], color='tab:green', lw=1.5, label='$S_t$')
    ax.scatter([], [], s=6, c='0.6', label='adversarial inner samples')
    ax.plot([], [], 'k.-', label='actual trajectory')
    lo = np.min([b[:, 0] for b in chX + [S]], 0); hi = np.max([b[:, 1] for b in chX + [S]], 0)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel('position p'); ax.set_ylabel('velocity v')
    ax.set_title(f'(b) 10-step lookahead from t={t0}')
    ax.legend(fontsize=8, loc='best')

    ax = axs[2]
    ks = np.arange(1, E.K + 1)
    for i, (nm, ls) in enumerate((('p', '-'), ('v', '--'))):
        ax.plot(ks, [E.w(b)[i] for b in chX], ls, color='tab:red', label=f'rule X, width {nm}')
        ax.plot(ks, [E.w(b)[i] for b in chS], ls, color='tab:blue', label=f'rule S, width {nm}')
        ax.plot(ks, [E.w(b)[i] for b in adv], ls, color='0.4', label=f'adversarial inner, width {nm}')
    ax.set_xlabel('steps ahead k'); ax.set_ylabel('reach-box width')
    ax.set_title('(c) widths')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    print(f'seed {seed} t0={t0}  hw(S_t) = {E.hw(S)}')
    print(' k   X p|v            S p|v            adv p|v         gain p|v')
    for k in range(E.K):
        a, b, c = E.w(chX[k]), E.w(chS[k]), E.w(adv[k])
        print(f'{k+1:2d}  {a[0]:.3f}|{a[1]:.3f}   {b[0]:.3f}|{b[1]:.3f}   {c[0]:.3f}|{c[1]:.3f}'
              f'   {100*(1-b[0]/a[0]):4.1f}%|{100*(1-b[1]/a[1]):4.1f}%')
    print('min p over rollout', xs[:, 0].min(), ' wrote', out)


if __name__ == '__main__':
    main()
