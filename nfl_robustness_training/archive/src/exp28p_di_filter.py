"""DI phase portrait, nominal vs filtered, with the certified set as the estimator.

Standalone preview for the DI port -- NOT alg22_di / alg28. Same plant, NN, noise model and
set-membership loop as exp28o (uniform noise, |z - x| <= vmax, S' = F(S, u) ∩ (z ± vmax)).

Nominal:  u = NN(centre(S_t)).
Filtered: u = NN(centre(S_t)) if the certificate below holds, else the backup u = +U_B.

Certificate for a candidate u (sound given truth ∈ S_t):
    P = F(S_t, u)                                  exact interval image, contains x_{t+1}
    P satisfies p >= p_min and v >= v_min
    from every state in P, the constant backup u = +U_B keeps p >= p_min forever.
The backup is open-loop, so its trajectories from P are affine in (p, v) with non-negative
coefficients: the worst case is the corner (p_lo, v_lo), checked until v_lo turns non-negative.
Under the backup v only increases, so v >= v_min holds after the first step as well.

Recursive feasibility: if u_{t-1} was certified, then the backup from P_{t-1} is safe, and
S_t ⊆ P_{t-1}, so the backup is safe from S_t and passes the same check (modulo the p/v
check of its own first step, which the certificate of u_{t-1} already covered).

usage: exp28p_di_filter.py OUTDIR [VMIN] [SEED]    (SEED=0: search 1..400 for a nominal violation)
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import exp28o_di_ebox as E

P_MIN = 0.0
U_B = 1.0
T = 45


def backup_ok(P):
    """Constant u = +U_B from box P keeps p >= P_MIN forever (corner (p_lo, v_lo) is worst)."""
    p, v = P[0, 0], P[1, 0]
    for _ in range(200):
        if p < P_MIN - 1e-12:
            return False
        if v >= 0:
            return True
        p, v = p + E.DT * v + E.BV[0] * U_B, v + E.BV[1] * U_B
    return False


def certified(S, u, vmin):
    P = E.istep(S, u)
    return P[0, 0] >= P_MIN and P[1, 0] >= vmin and backup_ok(P)


def run(seed, vmin, filtered):
    rng = np.random.default_rng(seed)
    x = rng.uniform(E.INIT[:, 0], E.INIT[:, 1])
    noise = rng.uniform(-E.VMAX, E.VMAX, (T + 1, 2))     # same noise for both arms
    z = x + noise[0]
    S = E.meet(E.INIT, np.stack([z - E.VMAX, z + E.VMAX], 1))
    xs, Ss, act = [x.copy()], [S.copy()], []
    z0 = z.copy()
    for t in range(T):
        u = E.nn_u(S.mean(1))[0]
        if filtered and not certified(S, u, vmin):
            u = U_B
            act.append(t)
        x = E.f_np(x[None], [u])[0]
        z = x + noise[t + 1]
        S = E.meet(E.istep(S, u), np.stack([z - E.VMAX, z + E.VMAX], 1))
        assert np.all(S[:, 0] <= x + 1e-12) and np.all(x <= S[:, 1] + 1e-12), 'truth lost'
        xs.append(x.copy()); Ss.append(S.copy())
    xs = np.array(xs)
    viol = [i for i in range(len(xs)) if xs[i, 0] < P_MIN or xs[i, 1] < vmin]
    return dict(xs=xs, Ss=Ss, act=act, viol=viol, z0=z0)


def backup_boundary(vs):
    """Smallest p from which the backup keeps p >= P_MIN, for a point state (p, v)."""
    out = []
    for v0 in vs:
        p, v, worst = 0.0, v0, 0.0
        while v < 0:
            p, v = p + E.DT * v + E.BV[0] * U_B, v + E.BV[1] * U_B
            worst = min(worst, p)
        out.append(P_MIN - worst)
    return np.array(out)


def layers(ax, nom, fil, vmin, lw_box, lw_tr, ms, label):
    for i, b in enumerate(nom['Ss']):
        ax.add_patch(Rectangle((b[0, 0], b[1, 0]), *(b[:, 1] - b[:, 0]), fill=False, ec='#e6862a',
                               lw=lw_box, ls=(0, (3, 2)), alpha=0.7,
                               label='certified set $S_t$ — unfiltered' if (label and i == 0) else None))
    for i, b in enumerate(fil['Ss']):
        ax.add_patch(Rectangle((b[0, 0], b[1, 0]), *(b[:, 1] - b[:, 0]), fill=False, ec='#2a64b0',
                               lw=lw_box, alpha=0.6,
                               label='certified set $S_t$ — filtered' if (label and i == 0) else None))
    ax.plot(*nom['xs'].T, '-o', color='#e6862a', lw=lw_tr, ms=ms, label='true state — unfiltered' if label else None)
    ax.plot(*fil['xs'].T, '-o', color='#2a64b0', lw=lw_tr, ms=ms, label='true state — filtered' if label else None)
    a = fil['xs'][fil['act']]
    ax.plot(a[:, 0], a[:, 1], 'o', mfc='none', mec='#2a64b0', ms=ms * 3.2, mew=1.4,
            label='filter applies backup' if label else None)
    v = [i for i in nom['viol']]
    if v:
        ax.plot(nom['xs'][v, 0], nom['xs'][v, 1], 'x', color='#b0202a', ms=ms * 2.4, mew=1.6,
                label='unfiltered violates' if label else None)
    b = E.INIT
    ax.add_patch(Rectangle((b[0, 0], b[1, 0]), *(b[:, 1] - b[:, 0]), fill=False, ec='0.45', ls=':', lw=1.0,
                           label='initial set $X_0$ (prior for $S_0$)' if label else None))
    ax.plot(*fil['xs'][0], marker='*', color='k', ms=ms * 3.2, ls='none', zorder=5,
            label='$t=0$ true state' if label else None)
    c0 = fil['Ss'][0].mean(1)
    ax.plot(*c0, marker='+', color='k', ms=ms * 3.2, mew=1.4, ls='none', zorder=5,
            label='$t=0$ NN input (centre of $S_0$)' if label else None)
    z0 = fil['z0']
    ax.plot(*z0, marker='D', mfc='none', mec='k', ms=ms * 1.8, mew=1.1, ls='none', zorder=5,
            label='$t=0$ measurement $z_0$' if label else None)
    ax.axhline(vmin, color='#b0202a', ls='--', lw=1.4, label=r'$v \geq v_{\min}$' if label else None)
    ax.axvline(P_MIN, color='0.15', lw=1.4, label=r'$p \geq p_{\min}$' if label else None)
    ax.axhspan(vmin - 5, vmin, color='#b0202a', alpha=0.06)
    vs = np.linspace(vmin - 0.4, 0.3, 200)
    ax.plot(backup_boundary(vs), vs, color='#7a4ea8', lw=1.4,
            label='backup-safe boundary (point state)' if label else None)


def main():
    torch.set_num_threads(1)
    outdir = sys.argv[1]
    vmin = float(sys.argv[2]) if len(sys.argv) > 2 else -0.9
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if seed == 0:
        for sd in range(1, 401):
            nom = run(sd, vmin, False)
            if nom['viol']:
                seed = sd
                break
        else:
            sys.exit(f'no nominal violation in seeds 1..400 at v_min={vmin}')
    nom, fil = run(seed, vmin, False), run(seed, vmin, True)
    print(f'seed {seed} v_min {vmin}: nominal violations at t={nom["viol"]} '
          f'(min v {nom["xs"][:, 1].min():.3f}, min p {nom["xs"][:, 0].min():.3f})')
    print(f'  filtered: backup at t={fil["act"]}, violations {fil["viol"]} '
          f'(min v {fil["xs"][:, 1].min():.3f}, min p {fil["xs"][:, 0].min():.3f})')
    print(f'  z0 {fil["z0"]}, centre S0 {fil["Ss"][0].mean(1)}, x0 {fil["xs"][0]}')
    print(f'  final p: nominal {nom["xs"][-1, 0]:.3f}, filtered {fil["xs"][-1, 0]:.3f}')

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    layers(ax, nom, fil, vmin, 0.6, 1.3, 3.0, True)
    allx = np.vstack([nom['xs'], fil['xs']])
    ax.set_xlim(P_MIN - 0.22, max(allx[:, 0].max(), E.INIT[0, 1], E.INIT[0, 1] + E.VMAX) + 0.1)
    ax.set_ylim(vmin - 0.2, max(allx[:, 1].max(), E.INIT[1, 1]) + 0.1)
    ax.set_xlabel('position $p$'); ax.set_ylabel('velocity $v$')
    ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    span = sorted(set(nom['viol']) | set(fil['act'])) or [int(np.argmin(nom['xs'][:, 1]))]
    idx = list(range(max(0, span[0] - 1), min(T + 1, span[-1] + 2)))
    px = np.concatenate([nom['xs'][idx, 0], fil['xs'][idx, 0]])
    axins = ax.inset_axes([0.30, 0.52, 0.40, 0.36])
    layers(axins, nom, fil, vmin, 0.7, 1.5, 3.6, False)
    axins.set_xlim(px.min() - 0.12, px.max() + 0.12)
    axins.set_ylim(vmin - 0.12, vmin + 0.25)
    axins.set_facecolor('white'); axins.tick_params(labelsize=7.5)
    axins.set_title(r'zoom: the $v_{\min}$ crossing', fontsize=8.0, pad=3.0)
    ax.indicate_inset_zoom(axins, edgecolor='0.4', lw=0.9)

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=7.0,
               frameon=False, columnspacing=1.2)
    fig.suptitle(rf'Double integrator, certified-set estimator: nominal vs filtered   '
                 rf'($p_{{\min}}=0$, $v_{{\min}}={vmin:g}$, seed {seed})', fontsize=10, y=0.985)
    fig.subplots_adjust(left=0.082, right=0.988, top=0.70, bottom=0.115)
    out = os.path.join(outdir, f'exp28p_di_phase_nominal_vs_filtered_vmin{vmin:g}.png')
    fig.savefig(out, dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
