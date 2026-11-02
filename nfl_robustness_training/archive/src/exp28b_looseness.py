"""
exp28b -- follow-up to exp28: how much of the lookahead/tube growth is the
system and how much is the bound?

For each starting box (a TIGHT one from the uniform-noise run and a
REALISTIC one from Gaussian noise with vmax = 4 sigma) it compares every
sound bound against an inner estimate of the true reachable set: the hull of
simulated trajectories that include the adversarial cases a random draw
misses -- every corner of the start box combined with every CONSTANT error
e in {-vmax, 0, vmax}^3 (a persistent sensor bias), plus random draws.

The true reachable set lies between the hull (inner) and the bound (outer).

Usage:  python exp28b_looseness.py <outdir>
"""
import sys
import json
import itertools
import numpy as np
import exp28_stateunc as E

VM = E.VMAX


def start_boxes():
    """Tight: uniform noise, t=20. Realistic: Gaussian, vmax=4 sigma, t=20."""
    out = {}
    _, _, keep = E.exp1(None, n_roll=1, kind='uniform', seed=0)
    out['tight'] = (keep['C'][20], keep['z'][20], E.VMAX)
    E.VMAX = 4 * E.SIGMA
    _, _, keep = E.exp1(None, n_roll=1, kind='gauss', vmax=E.VMAX, seed=5)
    out['realistic'] = (keep['C'][20], keep['z'][20], E.VMAX)
    E.VMAX = VM
    return out


def corners(b):
    return np.array(list(itertools.product(*[(b[i, 0], b[i, 1]) for i in range(3)])))


def e_sequences(K, vmax, rng, n_rand):
    """Constant biases (27) + random uniform + random bang-bang, each (K,3)."""
    seqs = [np.tile(np.array(c) * vmax, (K, 1)) for c in itertools.product([-1, 0, 1], repeat=3)]
    for _ in range(n_rand):
        seqs.append(rng.uniform(-vmax, vmax, size=(K, 3)))
        seqs.append(np.sign(rng.uniform(-1, 1, size=(K, 3))) * vmax)
    return seqs


def hull(traj):
    return [np.stack([X.min(0), X.max(0)], 1) for X in traj]


def lookahead(C_t, z_t, vmax, K=10, seed=3):
    rng = np.random.default_rng(seed)
    E.VMAX = vmax
    modes, _, _, _ = E.exp2(C_t, z_t, K=K, n_mc=200)
    u_t = E.nn_u(z_t)[0]
    X0 = np.vstack([corners(C_t), rng.uniform(C_t[:, 0], C_t[:, 1], size=(300, 3))])
    seqs = e_sequences(K - 1, vmax, rng, 150)
    # every start state x every error sequence
    X = np.repeat(X0, len(seqs), 0)
    S = np.tile(np.stack(seqs), (len(X0), 1, 1))
    X = E.f_np(X, np.full(len(X), u_t))
    traj = [X]
    for k in range(K - 1):
        X = E.f_np(X, E.nn_u(X + S[:, k]))
        traj.append(X)
    # the current RSOA's own model: NN reads x (e = 0)
    Xa = E.f_np(X0, np.full(len(X0), u_t))
    tra = [Xa]
    for k in range(K - 1):
        Xa = E.f_np(Xa, E.nn_u(Xa)); tra.append(Xa)
    E.VMAX = VM
    miss = {k: [int((~E.contains(b, traj[i])).sum()) for i, b in enumerate(v)]
            for k, v in modes.items()}
    return modes, hull(traj), hull(tra), miss, len(X)


def tube_hull(C_t, vseq, mode, boxes, vmax, kth, ky, seed=4, n_rand=100):
    """Inner hull for a backup tube: box corners x constant/random sensor errors."""
    rng = np.random.default_rng(seed)
    K = len(vseq)
    Z = E.nominal(C_t.mean(1), vseq)
    G, Bv = E.fb_gains(Z, vseq, kth, ky)
    X0 = np.vstack([corners(C_t), rng.uniform(C_t[:, 0], C_t[:, 1], size=(200, 3))])
    seqs = e_sequences(K, vmax, rng, n_rand)
    X = np.repeat(X0, len(seqs), 0)
    V = np.tile(np.stack(seqs), (len(X0), 1, 1))
    n = len(X)
    lo = np.repeat(C_t[None, :, 0], n, 0)
    hi = np.repeat(C_t[None, :, 1], n, 0)
    traj, miss = [X], [0]
    for i, v in enumerate(vseq):
        if mode == 'ol':
            u = np.full(n, v)
        elif mode == 'fb_z':
            u = np.clip((X + V[:, i]) @ G[i] + Bv[i], -E.UMAX, E.UMAX)
        else:
            u = np.clip(((lo + hi) / 2) @ G[i] + Bv[i], -E.UMAX, E.UMAX)
        Xn = E.f_np(X, u)
        if mode == 'fb_c':
            th_lo, th_hi = lo[:, 2], hi[:, 2]
            c_lo, c_hi = np.cos(th_lo), np.cos(th_hi)
            # vectorised istep (heading intervals here are far below pi)
            cl, ch = _cos_rng(th_lo, th_hi)
            sl, sh = _cos_rng(th_lo - np.pi / 2, th_hi - np.pi / 2)
            plo = np.stack([lo[:, 0] + E.DT * E.V * cl, lo[:, 1] + E.DT * E.V * sl, th_lo + E.DT * u], 1)
            phi = np.stack([hi[:, 0] + E.DT * E.V * ch, hi[:, 1] + E.DT * E.V * sh, th_hi + E.DT * u], 1)
            zz = Xn + V[:, i]
            lo = np.maximum.reduce([plo, zz - vmax, np.repeat(boxes[i + 1][None, :, 0], n, 0)])
            hi = np.minimum.reduce([phi, zz + vmax, np.repeat(boxes[i + 1][None, :, 1], n, 0)])
        X = Xn
        traj.append(X)
        miss.append(int((~E.contains(boxes[i + 1], X)).sum()))
    return hull(traj), miss


def _cos_rng(lo, hi):
    """Vectorised cos range; exact for intervals shorter than pi."""
    a, b = np.cos(lo), np.cos(hi)
    mn, mx = np.minimum(a, b), np.maximum(a, b)
    k1 = 2 * np.pi * np.ceil(lo / (2 * np.pi)) <= hi
    k2 = np.pi + 2 * np.pi * np.ceil((lo - np.pi) / (2 * np.pi)) <= hi
    return np.where(k2, -1.0, mn), np.where(k1, 1.0, mx)


def width_xy(b):
    return float(max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]))


def main():
    out = sys.argv[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    boxes0 = start_boxes()
    summary = {}
    fig, ax = plt.subplots(2, 4, figsize=(22, 9.5))
    for row, (name, (C_t, z_t, vmax)) in enumerate(boxes0.items()):
        modes, h, ha, miss, n = lookahead(C_t, z_t, vmax)
        summary[f'lookahead_{name}'] = {
            'C_t_width': (C_t[:, 1] - C_t[:, 0]).tolist(), 'vmax': vmax, 'n_traj': n,
            'miss': miss,
            'xy_width': {k: [width_xy(b) for b in v] for k, v in modes.items()},
            'th_width': {k: [float(b[2, 1] - b[2, 0]) for b in v] for k, v in modes.items()},
            'hull_xy': [width_xy(b) for b in h], 'hull_th': [float(b[2, 1] - b[2, 0]) for b in h],
            'hullA_xy': [width_xy(b) for b in ha]}
        a = ax[row, 0]
        cols = {'A_coupled_no_e': '#E24B4A', 'B_decoupled': '#BA7517',
                'C_joint_concrete': '#1D9E75', 'C_joint_symbolic': '#534AB7'}
        K = len(h)
        for k, c in cols.items():
            a.plot(range(1, K + 1), [width_xy(b) for b in modes[k]], color=c, marker='o', ms=3, label=k)
        a.plot(range(1, K + 1), [width_xy(b) for b in h], 'k--', label='true hull (NN reads x+e)')
        a.plot(range(1, K + 1), [width_xy(b) for b in ha], 'k:', label='hull if NN read x')
        a.set_yscale('log'); a.set_title(f'{name}: lookahead max(x,y) width'); a.legend(fontsize=7)
        a.set_xlabel('steps ahead')
        a = ax[row, 1]
        for k, c in cols.items():
            a.plot(range(1, K + 1), [b[2, 1] - b[2, 0] for b in modes[k]], color=c, marker='o', ms=3)
        a.plot(range(1, K + 1), [b[2, 1] - b[2, 0] for b in h], 'k--')
        a.set_yscale('log'); a.set_title(f'{name}: lookahead θ width'); a.set_xlabel('steps ahead')

        vseq = [0.6] * 8 + [0.0] * 12
        E.VMAX = vmax
        a = ax[row, 2]
        tcols = {'ol': '#BA7517', 'fb_c': '#1D9E75'}
        for m, c in tcols.items():
            tb, Z, Es = E.tube(C_t, vseq, m)
            th, tm = tube_hull(C_t, vseq, m, tb, vmax, 2.5, 2.0)
            summary[f'tube_{name}_{m}'] = {'miss': sum(tm),
                                           'bound_xy': [width_xy(b) for b in tb],
                                           'hull_xy': [width_xy(b) for b in th]}
            a.plot([width_xy(b) for b in tb], color=c, label=f'{m} bound')
            a.plot([width_xy(b) for b in th], color=c, ls='--', label=f'{m} true hull')
            if m == 'ol':
                b3 = ax[row, 3]
                b3.plot(Z[:, 0], Z[:, 1], 'k--', lw=0.8)
                for b in tb:
                    b3.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                           fill=False, color=c, lw=0.8))
                for b in th:
                    b3.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                           fill=True, color='#D85A30', alpha=0.3, lw=0))
                b3.set_aspect('equal')
                b3.set_title(f'{name}: open-loop backup, bound (box) vs true hull (fill)\n'
                             f'start θ width {C_t[2,1]-C_t[2,0]:.3f} rad')
        E.VMAX = VM
        a.set_yscale('log'); a.set_title(f'{name}: backup tube max(x,y) width'); a.legend(fontsize=7)
        a.set_xlabel('plan step')
    fig.suptitle('exp28b: sound bounds (solid) vs inner estimate of the true reachable set (dashed); '
                 'top = tight start box (uniform noise), bottom = realistic (Gaussian, vmax = 4σ)')
    fig.tight_layout(); fig.savefig(f'{out}/exp28b_looseness.png', dpi=120)
    with open(f'{out}/summary_b.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    for k, v in summary.items():
        print(k, json.dumps({kk: (vv if not isinstance(vv, list) or len(vv) < 4 else
                                  [round(x, 3) if isinstance(x, float) else x for x in vv])
                             for kk, vv in v.items() if kk != 'miss' or True}))


if __name__ == '__main__':
    main()
