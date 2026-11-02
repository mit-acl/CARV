"""
exp28h -- follow-up to exp28g: can a STRONGER noise-aware symbolic bound beat
(or at least tighten) the mode-J concrete chain?

exp28g split only the start box. Here the per-step heading errors e_theta_i are
split too (es cells per step, es^k cells in total), and alpha-CROWN
(CROWN-Optimized) is tried on top. Same start boxes, e box and J256 chain as
exp28g. The PSF would intersect symbolic with the concrete chain, so the
number that matters is sym ∩ J256 against J256 alone.

Usage:  python exp28h_symsplit.py <outdir>
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
# (name, x0 split, e_theta cells per step, method, max k)
VARIANTS = [('x(2,2,16)', (2, 2, 16), 1, 'backward', 5),
            ('x(1,1,8) eθ2', (1, 1, 8), 2, 'backward', 5),
            ('x(1,1,4) eθ4', (1, 1, 4), 4, 'backward', 3),
            ('x(1,1,8) eθ2 α', (1, 1, 8), 2, 'CROWN-Optimized', 5)]


def sym_bound(R1, eb, k, xs, es, method):
    m = E.CL(k, 'nn', True)
    box = np.vstack([R1] + [eb] * k)
    s = list(xs) + [1, 1, es] * k
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(len(s))]
    lo, hi = [], []
    for idx in itertools.product(*[range(n) for n in s]):
        lo.append([edges[i][j] for i, j in enumerate(idx)])
        hi.append([edges[i][j + 1] for i, j in enumerate(idx)])
    lo = torch.tensor(np.array(lo), dtype=torch.float32)
    hi = torch.tensor(np.array(hi), dtype=torch.float32)
    bm = BoundedModule(m, torch.zeros_like(lo), device='cpu', verbose=False,
                       bound_opts={'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    t0 = time.perf_counter()
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = bm.compute_bounds(x=(x,), method=method)
    dt = time.perf_counter() - t0
    lb, ub = lb.detach().numpy(), ub.detach().numpy()
    return np.stack([lb.min(0) - E.PAD, ub.max(0) + E.PAD], 1), dt, len(lo)


def main():
    out = sys.argv[1]
    torch.set_num_threads(1)
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
        f = lambda xs: ' '.join(f'{v:6.3f}' for v in xs)
        print(f'\n{name}  (θ width at k=1..5, xy = max(x,y) width)', flush=True)
        print(f'  {"true":22s} θ {f([h[k][2, 1] - h[k][2, 0] for k in range(1, KMAX + 1)])}'
              f'   xy {f([max(D.w(h[k])[:2]) for k in range(1, KMAX + 1)])}', flush=True)
        print(f'  {"J256":22s} θ {f([J[k][2, 1] - J[k][2, 0] for k in range(1, KMAX + 1)])}'
              f'   xy {f([max(D.w(J[k])[:2]) for k in range(1, KMAX + 1)])}', flush=True)
        for vname, xs, es, method, kmax in VARIANTS:
            row, both, tm, nc = [], [], [], []
            for k in range(1, kmax + 1):
                sb, dt, n = sym_bound(R1, eb, k, xs, es, method)
                row.append(sb); tm.append(dt); nc.append(n)
                both.append(np.stack([np.maximum(sb[:, 0], J[k][:, 0]), np.minimum(sb[:, 1], J[k][:, 1])], 1))
            ok = all(bool(np.all(row[k - 1][:, 0] <= h[k][:, 0] + 1e-6) and np.all(row[k - 1][:, 1] >= h[k][:, 1] - 1e-6))
                     for k in range(1, kmax + 1))
            gain = [1 - D.w(both[k - 1])[2] / D.w(J[k])[2] for k in range(1, kmax + 1)]
            S[vname] = {'sym': [D.w(x) for x in row], 'sym_and_J256': [D.w(x) for x in both],
                        'sec': tm, 'cells': nc, 'contains_true': ok, 'theta_gain_vs_J256': gain}
            print(f'  {vname:22s} θ {f([r[2, 1] - r[2, 0] for r in row])}'
                  f'   xy {f([max(D.w(r)[:2]) for r in row])}'
                  f'   ∩J θ gain {" ".join(f"{g * 100:4.0f}%" for g in gain)}'
                  f'   sec {" ".join(f"{t:.2f}" for t in tm)}  cells {nc[-1]}  sound {ok}', flush=True)
    with open(f'{out}/summary_h.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
