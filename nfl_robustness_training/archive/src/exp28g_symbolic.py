"""
exp28g -- would a noise-aware SYMBOLIC refinement add anything over mode J?

The committed symbolic step (REAL_integrated_sim.symbolic) bounds k closed-loop
steps in one CROWN pass with u = NN(x): no sensor error, so it is unsound once
the NN reads an estimate (alg28 turns it off). The sound version takes the
start box AND one error box per step as inputs, [x0 | e_0 | ... | e_{k-1}]
(3 + 3k dims), u_i = NN(x_i + e_i). This script bounds it for k = 1..5 from the
same start as the lookahead, with the start box split into cells (the e's are
not split: 3k dims), and compares with the mode-J concrete chain (J256) and
with the true hull. The PSF intersects symbolic and concrete bounds, so the
column that matters is sym ∩ J256.

Usage:  python exp28g_symbolic.py <outdir>
"""
import sys
import json
import itertools
import time
import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import exp28_stateunc as E
import exp28b_looseness as B
import exp28e_diagnose as D
import exp28f_split as Fs

KMAX = 5
XSPLITS = [(1, 1, 1), (1, 1, 16), (2, 2, 16), (4, 4, 32)]


def sym_bound(R1, eb, k, xs):
    m = E.CL(k, 'nn', True)
    edges = [np.linspace(R1[i, 0], R1[i, 1], xs[i] + 1) for i in range(3)]
    lo, hi = [], []
    for idx in itertools.product(*[range(n) for n in xs]):
        lo.append([edges[i][j] for i, j in enumerate(idx)] + list(eb[:, 0]) * k)
        hi.append([edges[i][j + 1] for i, j in enumerate(idx)] + list(eb[:, 1]) * k)
    lo = torch.tensor(np.array(lo), dtype=torch.float32)
    hi = torch.tensor(np.array(hi), dtype=torch.float32)
    bm = BoundedModule(m, torch.zeros_like(lo), device='cpu', verbose=False)
    t0 = time.perf_counter()
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = bm.compute_bounds(x=(x,), method='backward')
    dt = time.perf_counter() - t0
    lb, ub = lb.detach().numpy(), ub.detach().numpy()
    return np.stack([lb.min(0) - E.PAD, ub.max(0) + E.PAD], 1), dt


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(11)
    summary = {}
    for name, (C_t, z_t, vmax) in B.start_boxes().items():
        eb = E.ebox(np.full(3, vmax))
        R1 = E.istep(C_t, E.nn_u(z_t)[0])
        h = B.hull(D.true_traj(C_t, z_t, vmax, rng))
        b = R1.copy(); J = [b]
        for _ in range(KMAX):
            b = Fs.split_step(b, eb, (2, 2, 16, 1, 1, 4)); J.append(b)
        S = summary[name] = {'true': [D.w(x) for x in h[:KMAX + 1]], 'J256': [D.w(x) for x in J]}
        print(f'{name}: true th ' + ' '.join(f'{h[k][2, 1] - h[k][2, 0]:.3f}' for k in range(1, KMAX + 1)))
        print(f'{name}: J256 th ' + ' '.join(f'{J[k][2, 1] - J[k][2, 0]:.3f}' for k in range(1, KMAX + 1)))
        for xs in XSPLITS:
            row, both, tm = [], [], []
            for k in range(1, KMAX + 1):
                sb, dt = sym_bound(R1, eb, k, xs)
                row.append(sb); tm.append(dt)
                both.append(np.stack([np.maximum(sb[:, 0], J[k][:, 0]), np.minimum(sb[:, 1], J[k][:, 1])], 1))
            ok = all(bool(np.all(row[k - 1][:, 0] <= h[k][:, 0] + 1e-6) and np.all(row[k - 1][:, 1] >= h[k][:, 1] - 1e-6))
                     for k in range(1, KMAX + 1))
            S[str(xs)] = {'sym': [D.w(x) for x in row], 'sym_and_J256': [D.w(x) for x in both],
                          'sec': tm, 'contains_true': ok}
            print(f'{name}: sym{str(xs):12s} th ' + ' '.join(f'{r[2, 1] - r[2, 0]:.3f}' for r in row)
                  + '   sym∩J th ' + ' '.join(f'{r[2, 1] - r[2, 0]:.3f}' for r in both)
                  + '   xy k5: sym %.3f  sym∩J %.3f  J %.3f  true %.3f' % (
                      max(D.w(row[-1])[:2]), max(D.w(both[-1])[:2]), max(D.w(J[-1])[:2]), max(D.w(h[KMAX])[:2]))
                  + '   sec ' + ' '.join(f'{t:.3f}' for t in tm) + f'  sound {ok}', flush=True)
    with open(f'{out}/summary_g.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
