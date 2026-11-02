"""
exp28k -- symbolic refinement back under estimate control, "symbolic in the
state, interval in the noise".

The sensor error only reaches the dynamics through the scalar control:
    u_i = NN(x_i + e_i) = NN(x_i) + d_i,   d_i = NN(x_i + e_i) - NN(x_i).
d_i is bounded by an interval D_i: joint CROWN of NN(x+e) - NN(x) over the
J256 box J_i (sound, it contains x_i) and the e box E_i, split into cells.
The symbolic bound then unrolls k steps with input [x_1 (3) | d_1..d_k (k)],
u_i = NN(x_i) + d_i: 3 + k inputs instead of 3 + 3k, and the NN's feedback is
kept symbolically across all k steps (no re-boxing).

Sound: for the true trajectory x_i in J_i and e_i in E_i, so d_i in D_i.
Compared at in-run certified sets (dump_J256.json, t=36 and t=45) with J256,
the exact box-per-step floor and the sampled true set.

Usage:  python exp28k_symd.py <dump_J256.json> <outdir>
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import exp28_stateunc as E
import exp28e_diagnose as D
import exp28j_partition as P
import exp28b_looseness as B

K = 10
DSPLIT = (2, 2, 16, 1, 1, 4)


class DNN(nn.Module):
    """[x | e] -> NN(x + e) - NN(x)."""

    def __init__(self):
        super().__init__()
        self.ctrl = E.CTRL
        eye = torch.eye(6)
        self.register_buffer('Px', eye[:, :3].clone())
        self.register_buffer('Pe', eye[:, 3:].clone())

    def forward(self, inp):
        x = inp @ self.Px
        return self.ctrl(x + inp @ self.Pe) - self.ctrl(x)


class SymD(nn.Module):
    """k steps, input [x (3) | d_0..d_{k-1}], u_i = NN(x_i) + d_i."""

    def __init__(self, k):
        super().__init__()
        self.k, self.ctrl = k, E.CTRL
        eye = torch.eye(3 + k)
        self.register_buffer('Px', eye[:, :3].clone())
        for i in range(k):
            self.register_buffer(f'Pd{i}', eye[:, 3 + i:4 + i].clone())
        for j in range(3):
            self.register_buffer(f'S{j}', torch.eye(3)[:, j:j + 1].clone())

    def forward(self, inp):
        x = inp @ self.Px
        for i in range(self.k):
            u = self.ctrl(x) + inp @ getattr(self, f'Pd{i}')
            x0, x1, x2 = x @ self.S0, x @ self.S1, x @ self.S2
            x = torch.cat([x0 + E.DT * E.V * torch.cos(x2), x1 + E.DT * E.V * torch.sin(x2),
                           x2 + E.DT * u], 1)
        return x


_BM = {}


def crown(model, key, lo, hi):
    lo = torch.tensor(lo, dtype=torch.float32); hi = torch.tensor(hi, dtype=torch.float32)
    if key not in _BM:
        _BM[key] = BoundedModule(model, torch.zeros_like(lo), device='cpu', verbose=False)
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = _BM[key].compute_bounds(x=(x,), method='backward')
    return lb.detach().numpy(), ub.detach().numpy()


def d_range(Jb):
    lo, hi = P.cells(np.vstack([Jb, P.ebox(Jb)]), DSPLIT)
    l, u = crown(DNN(), ('d', len(lo)), lo, hi)
    return float(l.min()) - E.PAD, float(u.max()) + E.PAD


def d_range_exact(Jb, rng):
    Z = D.sample_box(np.vstack([Jb, P.ebox(Jb)]), 40000, rng)
    d = E.nn_u(Z[:, :3] + Z[:, 3:]) - E.nn_u(Z[:, :3])
    return float(d.min()), float(d.max())


def symd(X1, Dr, k, xs):
    box = np.vstack([X1, np.array(Dr[:k])])
    lo, hi = P.cells(box, list(xs) + [1] * k)
    l, u = crown(SymD(k), ('s', k, len(lo)), lo, hi)
    return np.stack([l.min(0) - E.PAD, u.max(0) + E.PAD], 1)


def w(b):
    return max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]), b[2, 1] - b[2, 0]


def main():
    dump, outdir = sys.argv[1], sys.argv[2]
    torch.set_num_threads(1)
    st = {r['t']: r for r in json.load(open(dump))['steps']}
    rng = np.random.default_rng(0)
    summary = {}
    for t0 in (36, 45):
        S = np.array(st[t0]['bound_t'])
        X1 = E.istep(S, E.nn_u(S.mean(1))[0])
        J = [X1]
        for _ in range(K - 1):
            J.append(P.hull(D.crown_batch(*P.cells(np.vstack([J[-1], P.ebox(J[-1])]), (2, 2, 16, 1, 1, 4)))))
        fl = [X1]
        for _ in range(K - 1):
            fl.append(D.image_exact(fl[-1], P.ebox(fl[-1]), rng))
        t_d = time.perf_counter()
        Dr = [d_range(J[i]) for i in range(K - 1)]
        t_d = (time.perf_counter() - t_d) / (K - 1)
        Dx = [d_range_exact(J[i], rng) for i in range(K - 1)]
        rows = {}
        for xs in [(2, 2, 16), (4, 4, 32)]:
            for tag, DD in [('CROWN d', Dr), ('exact d (not sound)', Dx)]:
                if tag.startswith('exact') and xs != (4, 4, 32):
                    continue
                ch, tm = [X1], []
                for k in range(1, K):
                    t1 = time.perf_counter()
                    ch.append(symd(X1, DD, k, xs)); tm.append(time.perf_counter() - t1)
                rows[f'symD x{xs} {tag}'] = (ch, tm)
        # true set: e bounded by the J256 hull (J contains every S_k)
        X = np.vstack([B.corners(S), rng.uniform(S[:, 0], S[:, 1], size=(500, 3))])
        X = np.repeat(X, 120, 0)
        X = E.f_np(X, np.full(len(X), E.nn_u(S.mean(1))[0])); tr = [X]
        for k in range(1, K):
            hw = np.minimum(P.VMAX, (J[k - 1][:, 1] - J[k - 1][:, 0]) / 2)
            m = rng.integers(0, 2, len(X))[:, None]
            e = rng.uniform(-1, 1, (len(X), 3)); e = np.where(m, np.sign(e), e) * hw
            X = E.f_np(X, E.nn_u(X + e)); tr.append(X)
        true = [np.stack([x.min(0), x.max(0)], 1) for x in tr]
        ks = (1, 3, 5, 8, 10)
        print(f'\nstart t={t0}  (x/y width m (ratio to true) | θ width rad, at steps ahead {ks})'
              f'   d-interval widths {[round(b - a, 3) for a, b in Dr[:5]]}  ({t_d * 1e3:.0f} ms each)')
        out = {'true': true, 'floor (box/step, exact)': fl, 'J256': J}
        out.update({n: c for n, (c, _) in rows.items()})
        for n, (c, _) in rows.items():
            if 'CROWN' in n:
                out[n + ' ∩ J256'] = [np.stack([np.maximum(a[:, 0], b[:, 0]), np.minimum(a[:, 1], b[:, 1])], 1)
                                      for a, b in zip(c, J)]
        S_ = summary[t0] = {}
        for n, ch in out.items():
            ok = all(np.all(ch[k][:, 0] <= true[k][:, 0] + 1e-6) and np.all(ch[k][:, 1] >= true[k][:, 1] - 1e-6)
                     for k in range(K))
            tm = rows.get(n, (None, None))[1]
            print(f'  {n:40s} ' + '  '.join(f'{w(ch[k - 1])[0]:.3f} ({w(ch[k - 1])[0] / w(true[k - 1])[0]:4.2f}x)|{w(ch[k - 1])[1]:.2f}'
                                         for k in ks)
                  + (f'   sec k=5 {tm[3]:.2f} k=10 {tm[-1]:.2f}' if tm else '') + f'  contains {ok}', flush=True)
            S_[n] = {'width': [list(map(float, w(b))) for b in ch], 'sec': tm}
    with open(f'{outdir}/summary_k.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
