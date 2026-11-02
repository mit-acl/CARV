"""
exp28l -- two questions raised in review of the exp28j/k write-up.

(A) Is the reachable set R_k conservative BEFORE any bounding, because it
    quantifies over a fresh adversarial e_k in Err(.) each step instead of the
    error that an actual measurement realisation produces?
    Three arms from the same certified set S_t of a real J256 run:
      real     the closed loop as it actually runs: z = x + v, v ~ U(-vmax,vmax),
               S_{k+1} = F(S_k,u_k) ∩ (z ± vmax), u_k = NN(centre S_k),
               so e_k = centre(S_k) - x_k is whatever the draw gives
      free_S   adversarial e_k (constant bias / bang-bang / uniform) bounded by
               the MEDIAN REALISED half-width of S_k from the real arm
      free_J   adversarial e_k bounded by min(vmax, half-width of the J256 box),
               i.e. exactly the e model exp28j/k call "true"
    free_J / real  = the definitional conservatism of R.
    free_S / real  = the part that is adversarial-vs-actual alone.
    free_J / free_S = the part that is "the lookahead cannot know the future
                      certified set, only its own box".

(B) Does the heading cell count follow from a contraction criterion?
    theta' = theta + dt*NN(x+e), so d theta'/d theta = 1 + dt*g with
    g = dNN/d s_theta at s = x+e. Over a cell the true map is a contraction in
    theta iff sup |1 + dt*g| < 1. This measures sup|1+dt*g| per heading cell for
    1..32 cells on a real lookahead box, next to the per-step theta width factor
    the joint bound actually achieves with that many cells.

Usage:  python exp28l_emodel.py <dump_J256.json> <outdir>
"""
import sys
import json
import numpy as np
import torch
import exp28_stateunc as E
import exp28b_looseness as B
import exp28e_diagnose as D
import exp28f_split as Fs
import exp28j_partition as P

K = 10
VMAX = 0.15
N = 20000
STARTS = (36, 45)
SPLIT = (2, 2, 16, 1, 1, 4)


def istep_vec(lo, hi, u):
    cl, ch = B._cos_rng(lo[:, 2], hi[:, 2])
    sl, sh = B._cos_rng(lo[:, 2] - np.pi / 2, hi[:, 2] - np.pi / 2)
    return (np.stack([lo[:, 0] + E.DT * E.V * cl, lo[:, 1] + E.DT * E.V * sl, lo[:, 2] + E.DT * u], 1),
            np.stack([hi[:, 0] + E.DT * E.V * ch, hi[:, 1] + E.DT * E.V * sh, hi[:, 2] + E.DT * u], 1))


def hull(X):
    return np.stack([X.min(0), X.max(0)], 1)


def w(b):
    return max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]), b[2, 1] - b[2, 0]


def arm_real(S, rng):
    """The loop as it runs: NN reads centre(S_k), S_k from real measurements."""
    X = np.vstack([B.corners(S), rng.uniform(S[:, 0], S[:, 1], size=(N, 3))])
    n = len(X)
    lo = np.repeat(S[None, :, 0], n, 0)
    hi = np.repeat(S[None, :, 1], n, 0)
    u = np.full(n, E.nn_u(S.mean(1))[0])
    traj, ehw, shw = [], [], []
    for k in range(K):
        X = E.f_np(X, u)
        lo, hi = istep_vec(lo, hi, u)
        z = X + rng.uniform(-VMAX, VMAX, (n, 3))
        lo = np.maximum(lo, z - VMAX)
        hi = np.minimum(hi, z + VMAX)
        c = (lo + hi) / 2
        e = c - X
        u = E.nn_u(c)
        traj.append(hull(X))
        ehw.append(np.abs(e).max(0))
        shw.append(np.median((hi - lo) / 2, 0))
    return traj, np.array(ehw), np.array(shw)


def arm_free(S, hw_seq, rng):
    """Adversarial e_k in [-hw_k, hw_k]: constant bias, bang-bang, uniform."""
    X0 = np.vstack([B.corners(S), rng.uniform(S[:, 0], S[:, 1], size=(600, 3))])
    seqs = []
    for c in np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing='ij')).reshape(3, -1).T:
        seqs.append(np.tile(c, (K, 1)) * hw_seq)
    for _ in range(80):
        seqs.append(rng.uniform(-1, 1, (K, 3)) * hw_seq)
        seqs.append(np.sign(rng.uniform(-1, 1, (K, 3))) * hw_seq)
    Sq = np.stack(seqs)
    X = np.repeat(X0, len(seqs), 0)
    Sq = np.tile(Sq, (len(X0), 1, 1))
    X = E.f_np(X, np.full(len(X), E.nn_u(S.mean(1))[0]))
    traj = [hull(X)]
    for k in range(1, K):
        X = E.f_np(X, E.nn_u(X + Sq[:, k]))
        traj.append(hull(X))
    return traj


def part_a(dump, summary):
    st = {r['t']: r for r in json.load(open(dump))['steps']}
    for t0 in STARTS:
        rng = np.random.default_rng(0)
        S = np.array(st[t0]['bound_t'])
        real, ehw, shw = arm_real(S, rng)
        # J256 chain -> the e budget the lookahead is allowed to assume
        X1 = E.istep(S, E.nn_u(S.mean(1))[0])
        J = [X1]
        for _ in range(K - 1):
            J.append(P.hull(D.crown_batch(*P.cells(np.vstack([J[-1], P.ebox(J[-1])]), SPLIT))))
        hw_J = np.array([np.minimum(VMAX, (b[:, 1] - b[:, 0]) / 2) for b in J])
        free_J = arm_free(S, hw_J, rng)
        free_S = arm_free(S, shw, rng)
        ks = (1, 3, 5, 8, 10)
        print(f'\n[A] start t={t0}   xy width m | theta width rad   at k = {ks}')
        rows = {'real (actual measurements)': real, 'free_S (adv e, realised hw)': free_S,
                'free_J (adv e, J256 hw)': free_J}
        for n_, ch in rows.items():
            print(f'  {n_:30s} ' + '  '.join(f'{w(ch[k - 1])[0]:.3f}|{w(ch[k - 1])[1]:.3f}' for k in ks))
        for n_ in ('free_S (adv e, realised hw)', 'free_J (adv e, J256 hw)'):
            print(f'  {n_ + " / real":30s} ' + '  '.join(
                f'{w(rows[n_][k - 1])[0] / w(real[k - 1])[0]:.2f}x|{w(rows[n_][k - 1])[1] / w(real[k - 1])[1]:.2f}x'
                for k in ks))
        print(f'  {"e half-width: realised max":30s} ' + '  '.join(f'{ehw[k - 1][2]:.4f}' for k in ks)
              + '   (theta component)')
        print(f'  {"e half-width: S_k realised":30s} ' + '  '.join(f'{shw[k - 1][2]:.4f}' for k in ks))
        print(f'  {"e half-width: J256 budget":30s} ' + '  '.join(f'{hw_J[k - 1][2]:.4f}' for k in ks))
        summary[f'A_t{t0}'] = {
            'real': [list(map(float, w(b))) for b in real],
            'free_S': [list(map(float, w(b))) for b in free_S],
            'free_J': [list(map(float, w(b))) for b in free_J],
            'e_realised_max': ehw.tolist(), 'S_hw_realised_median': shw.tolist(),
            'e_budget_J256': hw_J.tolist()}


def part_b(summary):
    rng = np.random.default_rng(11)
    C_t, z_t, vmax = B.start_boxes()['realistic']
    eb = E.ebox(np.full(3, vmax))
    b = E.istep(C_t, E.nn_u(z_t)[0])
    for _ in range(2):                       # the box panel (c) dissects
        b = Fs.split_step(b, eb, SPLIT)
    n = 120000
    X = rng.uniform(b[:, 0], b[:, 1], (n, 3))
    ee = rng.uniform(eb[:, 0], eb[:, 1], (n, 3))
    s = torch.tensor(X + ee, dtype=torch.float32, requires_grad=True)
    g = torch.autograd.grad(E.CTRL(s).sum(), s)[0].numpy()[:, 2]
    wth = b[2, 1] - b[2, 0]
    print(f'\n[B] box theta = [{b[2, 0]:.3f}, {b[2, 1]:.3f}] ({wth:.3f} rad), '
          f'dNN/dtheta over the whole box [{g.min():.2f}, {g.max():.2f}]')
    print('  cells  cell w_th   g range on the worst cell    sup|1+dt g|   joint bound w_th\'/w_th   ms')
    rec = {}
    for m in (1, 2, 4, 8, 16, 32):
        idx = np.clip(((X[:, 2] - b[2, 0]) / wth * m).astype(int), 0, m - 1)
        cert, worst = 0.0, None
        for c in range(m):
            gg = g[idx == c]
            if not len(gg):
                continue
            v = max(abs(1 + E.DT * gg.min()), abs(1 + E.DT * gg.max()))
            if v > cert:
                cert, worst = v, (gg.min(), gg.max())
        import time
        s_ = (1, 1, m, 1, 1, 4)
        Fs.split_step(b, eb, s_)
        t0 = time.perf_counter()
        nb = Fs.split_step(b, eb, s_)
        ms = (time.perf_counter() - t0) * 1e3
        f = (nb[2, 1] - nb[2, 0]) / wth
        print(f'  {m:5d}  {wth / m:8.4f}   [{worst[0]:7.2f}, {worst[1]:7.2f}]   {cert:11.3f}   '
              f'{f:20.3f}   {ms:5.1f}')
        rec[m] = {'cell_w_theta': float(wth / m), 'g_worst_cell': [float(worst[0]), float(worst[1])],
                  'sup_abs_1_plus_dt_g': float(cert), 'joint_theta_factor': float(f), 'ms': ms}
    lo, hi = E.nn_bounds(b + eb)
    d = E.istep(b, lo, hi)
    print(f'  decoupled (any cells): u in [{lo:.3f}, {hi:.3f}] -> w_th\'/w_th = '
          f'{(d[2, 1] - d[2, 0]) / wth:.3f}')
    # e bang-bang: the extreme of theta' is attained at a corner of the e box,
    # and row-wise pairing of the two corner lists misses it (see exp28j_figure).
    Xs = np.vstack([B.corners(b), rng.uniform(b[:, 0], b[:, 1], (200000, 3))])
    Es = rng.uniform(eb[:, 0], eb[:, 1], (len(Xs), 3))
    _m = rng.integers(0, 2, len(Xs))[:, None]
    Es = np.where(_m, np.sign(Es) * eb[:, 1], Es)
    thn = Xs[:, 2] + E.DT * E.nn_u(Xs + Es)
    print(f'  true image w_th\'/w_th >= {(thn.max() - thn.min()) / wth:.3f}   '
          f'(INNER estimate)   least-squares slope {np.polyfit(Xs[:, 2], thn, 1)[0]:.3f}')
    rec['decoupled_factor'] = float((d[2, 1] - d[2, 0]) / wth)
    rec['true_factor'] = float((thn.max() - thn.min()) / wth)
    summary['B'] = rec


def main():
    torch.set_num_threads(1)
    dump, outdir = sys.argv[1], sys.argv[2]
    summary = {}
    part_a(dump, summary)
    part_b(summary)
    with open(f'{outdir}/summary_l.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
