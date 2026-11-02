"""
exp28i -- follow-up to exp28h: the only symbolic variant that tightened J was
alpha-CROWN, and it already won at k=1, so the gain is alpha-CROWN, not the
multi-step unroll. Here alpha-CROWN is put on the ONE-STEP joint (x, e) bound
itself (the mode-J step), chained k=1..5, to see what it buys and what it costs.

Usage:  python exp28i_alphastep.py <outdir>
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

KMAX = 5
VARIANTS = [('J256 CROWN', (2, 2, 16, 1, 1, 4), 0), ('J32 α5', (1, 1, 8, 1, 1, 4), 5),
            ('J32 α20', (1, 1, 8, 1, 1, 4), 20), ('J256 α5', (2, 2, 16, 1, 1, 4), 5),
            ('J256 α20', (2, 2, 16, 1, 1, 4), 20)]
_BM = {}


def step(b, eb, s, it):
    box = np.vstack([b, eb])
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(6)]
    cells = list(itertools.product(*[range(n) for n in s]))
    lo = torch.tensor([[edges[i][j] for i, j in enumerate(c)] for c in cells], dtype=torch.float32)
    hi = torch.tensor([[edges[i][j + 1] for i, j in enumerate(c)] for c in cells], dtype=torch.float32)
    key = (len(cells), it)
    if key not in _BM:
        _BM[key] = BoundedModule(D.STEP, torch.zeros(len(cells), 6), device='cpu', verbose=False,
                                 bound_opts={'optimize_bound_args': {'iteration': max(it, 1), 'lr_alpha': 0.1}})
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = _BM[key].compute_bounds(x=(x,), method='CROWN-Optimized' if it else 'backward')
    return np.stack([lb.detach().numpy().min(0) - E.PAD, ub.detach().numpy().max(0) + E.PAD], 1)


def main():
    out = sys.argv[1]
    torch.set_num_threads(1)
    rng = np.random.default_rng(11)
    summary = {}
    for name, (C_t, z_t, vmax) in B.start_boxes().items():
        eb = E.ebox(np.full(3, vmax))
        R1 = E.istep(C_t, E.nn_u(z_t)[0])
        h = B.hull(D.true_traj(C_t, z_t, vmax, rng))
        S = summary[name] = {'true': [D.w(x) for x in h[:KMAX + 1]]}
        f = lambda xs: ' '.join(f'{v:6.3f}' for v in xs)
        print(f'\n{name}  (θ width at k=1..5)', flush=True)
        print(f'  {"true":12s} θ {f([h[k][2, 1] - h[k][2, 0] for k in range(1, KMAX + 1)])}', flush=True)
        for vname, s, it in VARIANTS:
            step(R1, eb, s, it)                       # build once, time the rest
            b, ch, tm = R1.copy(), [R1.copy()], []
            for _ in range(KMAX):
                t0 = time.perf_counter(); b = step(b, eb, s, it); tm.append(time.perf_counter() - t0)
                ch.append(b)
            ok = all(bool(np.all(ch[k][:, 0] <= h[k][:, 0] + 1e-6) and np.all(ch[k][:, 1] >= h[k][:, 1] - 1e-6))
                     for k in range(KMAX + 1))
            S[vname] = {'width': [D.w(x) for x in ch], 'sec_per_step': tm, 'contains_true': ok}
            print(f'  {vname:12s} θ {f([c[2, 1] - c[2, 0] for c in ch[1:]])}   xy {f([max(D.w(c)[:2]) for c in ch[1:]])}'
                  f'   {np.median(tm) * 1e3:7.1f} ms/step  sound {ok}', flush=True)
    with open(f'{out}/summary_i.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
