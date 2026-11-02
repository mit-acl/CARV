"""
exp28o -- does the certified-set error box (rule S) matter on the double integrator?

The gate before porting alg28 to the DI. On the unicycle, rule S removed half the
excess interventions because the lookahead's e bound min(vmax, hw(X_k))
saturated at the raw sensor bound within a few steps while the certified set
stayed narrow. On the DI the width-invariant coordinate is v and position grows
only LINEARLY (hw_p + dt*hw_v per step), so the gap between the two rules may
be much smaller. This measures it.

Plant (matches the DI trials): x = (p, v), x' = A x + b u, A = [[1,dt],[0,1]],
b = [dt^2/2, dt], dt = 0.2, u = NN(xhat) UNCLIPPED (nfl_veripy's discrete
dynamics_step applies no u_limits), NN = di_4layer [30,20,10]
constraint_default_more_data_5hz. Measurement z = x + U(-vmax, vmax) in both
components, sigma = 0.05, vmax = 3 sigma. Start box p in [2.5,3], v in [-.25,.25].

Certified set (as alg28's setmem): S_{t+1} = F(S_t,u_t) ∩ (z_{t+1} +- vmax),
u_t = NN(centre(S_t)). Starts are S_t taken from real rollouts of that loop.

Lookahead from S_t, K steps:
  X_1   = F(S_t, u_t)                      (u_t is known: the filter computed it)
  X_k+1 = hull of joint CROWN of (x,e) -> A x + b NN(x+e) over cells of X_k x E_k
  rule X  E_k = +- min(vmax, hw(X_k))
  rule S  E_k = +- min(vmax, hw(X_k), h_k),  h_0 = hw(S_t),
          h_{k+1} = min(vmax, (h_p + dt*h_v, h_v))       [the lemma, exact on DI]
  dec   decoupled (mode B): CROWN range of NN over X_k (+) E_k, rule X, then
        interval step. Mode B cannot use rule S (see error_box_lemma.md).
True sets (INNER estimates, never sound):
  adv   x_0 in S_t, e_k adversarial in the rule-S box h_k (bang-bang, constant
        corner bias, uniform); the reachable set under admissible error
  real  the actual loop: random measurements, real set-membership, u = NN(centre)

Usage:  python exp28o_di_ebox.py <outdir>
"""
import sys
import json
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

REPO = '/home/ryosei_t/work/CARV'
DT = 0.2
A = np.array([[1.0, DT], [0.0, 1.0]])
BV = np.array([0.5 * DT * DT, DT])
SIGMA = 0.05
VMAX = 3.0 * SIGMA
INIT = np.array([[2.5, 3.0], [-0.25, 0.25]])
K = 10
PAD = 1e-5
KS = (1, 3, 5, 8, 10)
SPLITS = {'J1': (1, 1, 1, 1), 'J256': (4, 8, 2, 4)}     # cells over (p, v, e_p, e_v)


class DICtrl(nn.Module):
    """cl_systems.double_integrator.di_4layer_controller, copied to avoid the JAX import."""

    def __init__(self, npl=(30, 20, 10)):
        super().__init__()
        self.fc1 = nn.Linear(2, npl[0])
        self.fc2 = nn.Linear(npl[0], npl[1])
        self.fc3 = nn.Linear(npl[1], npl[2])
        self.fc4 = nn.Linear(npl[2], 1)

    def forward(self, xt):
        ut = F.relu(self.fc1(xt))
        ut = F.relu(self.fc2(ut))
        ut = F.relu(self.fc3(ut))
        return self.fc4(ut)


def load_ctrl():
    c = DICtrl()
    p = (REPO + '/nfl_robustness_training/src/controller_models/double_integrator/'
         'di_4layer/constraint_default_more_data_5hz.pth')
    sd = torch.load(p, map_location='cpu')
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    c.load_state_dict(sd, strict=True)
    for q in c.parameters():
        q.requires_grad_(False)
    return c.eval()


CTRL = load_ctrl()


def nn_u(X):
    with torch.no_grad():
        return CTRL(torch.tensor(np.atleast_2d(X), dtype=torch.float32)).numpy()[:, 0].astype(float)


def f_np(X, u):
    X = np.atleast_2d(X)
    return X @ A.T + np.asarray(u, dtype=float)[:, None] * BV[None, :]


def istep(S, u):
    """Exact interval image of box S under f(., u) for a scalar u (tight on the box)."""
    p = [S[0, 0] + DT * S[1, 0], S[0, 1] + DT * S[1, 1]]
    return np.array([[p[0] + BV[0] * u, p[1] + BV[0] * u],
                     [S[1, 0] + BV[1] * u, S[1, 1] + BV[1] * u]])


def hw(b):
    return (b[:, 1] - b[:, 0]) / 2.0


def w(b):
    return b[:, 1] - b[:, 0]


def ebox(h):
    return np.stack([-h, h], 1)


def h_next(h):
    return np.minimum(VMAX, np.array([h[0] + DT * h[1], h[1]]))


def meet(a, b):
    return np.stack([np.maximum(a[:, 0], b[:, 0]), np.minimum(a[:, 1], b[:, 1])], 1)


# ── CROWN ───────────────────────────────────────────────────────────────────────
class Step(nn.Module):
    """One closed-loop step, input [x | e] (4), u = NN(x + e), output A x + b u."""

    def __init__(self):
        super().__init__()
        self.ctrl = CTRL
        eye = torch.eye(4)
        self.register_buffer('Px', eye[:, :2].clone())
        self.register_buffer('Pe', eye[:, 2:].clone())
        self.register_buffer('At', torch.tensor(A.T, dtype=torch.float32))
        self.register_buffer('bt', torch.tensor(BV[None, :], dtype=torch.float32))

    def forward(self, inp):
        x = inp @ self.Px
        u = self.ctrl(x + inp @ self.Pe)
        return x @ self.At + u @ self.bt


STEP = Step()
_BM = {}


def _crown(model, key, lo, hi, nin):
    lo = torch.tensor(lo, dtype=torch.float32)
    hi = torch.tensor(hi, dtype=torch.float32)
    if key not in _BM:
        _BM[key] = BoundedModule(model, torch.zeros(lo.shape[0], nin), device='cpu', verbose=False)
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = _BM[key].compute_bounds(x=(x,), method='backward')
    return lb.detach().numpy() - PAD, ub.detach().numpy() + PAD


def cells(box, s):
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(len(s))]
    idx = list(itertools.product(*[range(n) for n in s]))
    lo = np.array([[edges[i][j] for i, j in enumerate(c)] for c in idx])
    hi = np.array([[edges[i][j + 1] for i, j in enumerate(c)] for c in idx])
    return lo, hi


def joint_step(X, E, s):
    lo, hi = cells(np.vstack([X, E]), s)
    l, u = _crown(STEP, ('j', len(lo)), lo, hi, 4)
    return np.stack([l.min(0), u.max(0)], 1)


def dec_step(X, E):
    """Mode B: NN range over X (+) E, then the interval step under that control range."""
    lo, hi = X[:, 0] + E[:, 0], X[:, 1] + E[:, 1]
    l, u = _crown(CTRL, ('u', 1), lo[None], hi[None], 2)
    ul, uh = float(l[0, 0]), float(u[0, 0])
    return np.array([[X[0, 0] + DT * X[1, 0] + BV[0] * ul, X[0, 1] + DT * X[1, 1] + BV[0] * uh],
                     [X[1, 0] + BV[1] * ul, X[1, 1] + BV[1] * uh]])


# ── chains ──────────────────────────────────────────────────────────────────────
def chain(S, rule, mode, s=None):
    X = istep(S, nn_u(S.mean(1))[0])
    h = h_next(hw(S))                       # bound on hw(S_{t+1})
    out, eh = [X], []
    for _ in range(K - 1):
        e = np.minimum(VMAX, hw(X))
        if rule == 'S':
            e = np.minimum(e, h)
        eh.append(e.copy())
        X = joint_step(X, ebox(e), s) if mode == 'J' else dec_step(X, ebox(e))
        out.append(X)
        h = h_next(h)
    return out, eh


def adv_set(S, rng, n0=400):
    """x_0 in S_t, e_k adversarial in the rule-S box. Inner estimate of the true set."""
    corners = np.array(list(itertools.product(*[(S[i, 0], S[i, 1]) for i in range(2)])))
    X0 = np.vstack([corners, rng.uniform(S[:, 0], S[:, 1], (n0, 2))])
    hs, h = [], h_next(hw(S))
    for _ in range(K - 1):
        hs.append(h.copy()); h = h_next(h)
    seqs = []
    for c in itertools.product((-1.0, 1.0), repeat=2):          # constant corner bias
        seqs.append(np.array([np.array(c) * hh for hh in hs]))
    for _ in range(24):                                          # bang-bang
        seqs.append(np.array([np.sign(rng.uniform(-1, 1, 2)) * hh for hh in hs]))
    for _ in range(8):                                           # uniform
        seqs.append(np.array([rng.uniform(-1, 1, 2) * hh for hh in hs]))
    seqs = np.array(seqs)                                        # (m, K-1, 2)
    X = np.repeat(X0, len(seqs), 0)
    Es = np.tile(seqs, (len(X0), 1, 1))
    X = f_np(X, np.full(len(X), nn_u(S.mean(1))[0]))
    tr = [X]
    for k in range(K - 1):
        X = f_np(X, nn_u(X + Es[:, k])); tr.append(X)
    return [np.stack([x.min(0), x.max(0)], 1) for x in tr]


def real_set(S, rng, n=3000):
    """The actual loop from S_t: random measurements, real set-membership."""
    x = rng.uniform(S[:, 0], S[:, 1], (n, 2))
    Ss = np.repeat(S[None], n, 0)
    u = np.full(n, nn_u(S.mean(1))[0])
    tr = []
    for _ in range(K):
        x = f_np(x, u)
        lo = Ss[:, :, 0] @ A.T.clip(0) + Ss[:, :, 1] @ A.T.clip(max=0) + u[:, None] * BV
        hi = Ss[:, :, 1] @ A.T.clip(0) + Ss[:, :, 0] @ A.T.clip(max=0) + u[:, None] * BV
        z = x + rng.uniform(-VMAX, VMAX, x.shape)
        lo, hi = np.maximum(lo, z - VMAX), np.minimum(hi, z + VMAX)
        Ss = np.stack([lo, hi], 2)
        tr.append(x.copy())
        u = nn_u(Ss.mean(2))
    return [np.stack([t.min(0), t.max(0)], 1) for t in tr], float(np.max(Ss[:, :, 1] - Ss[:, :, 0]) / 2)


def rollout_sets(seed, T=30):
    """One real rollout of the certified-set loop. Returns S_t for t = 0..T."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(INIT[:, 0], INIT[:, 1])
    z = x + rng.uniform(-VMAX, VMAX, 2)
    S = meet(INIT, np.stack([z - VMAX, z + VMAX], 1))
    out = [S.copy()]
    for _ in range(T):
        u = nn_u(S.mean(1))[0]
        x = f_np(x[None], [u])[0]
        z = x + rng.uniform(-VMAX, VMAX, 2)
        S = meet(istep(S, u), np.stack([z - VMAX, z + VMAX], 1))
        assert np.all(S[:, 0] <= x + 1e-12) and np.all(x <= S[:, 1] + 1e-12), 'truth lost'
        assert np.all(S[:, 0] <= S[:, 1]), 'empty intersection'
        out.append(S.copy())
    return out


def contains(outer, inner):
    return all(np.all(o[:, 0] <= i[:, 0] + 1e-9) and np.all(o[:, 1] >= i[:, 1] - 1e-9)
               for o, i in zip(outer, inner))


def main():
    torch.set_num_threads(1)
    outdir = sys.argv[1]
    starts = [(sd, t) for sd in (1, 2, 3) for t in (3, 10, 20)]
    summary, agg = {}, {}
    for sd, t0 in starts:
        rng = np.random.default_rng(1000 * sd + t0)
        S = rollout_sets(sd)[t0]
        adv = adv_set(S, rng)
        real, _ = real_set(S, rng)
        rows, ehs, tms = {}, {}, {}
        for rule in ('X', 'S'):
            for nm, s in SPLITS.items():
                t1 = time.perf_counter()
                rows[f'{nm} rule {rule}'], ehs[f'{nm} rule {rule}'] = chain(S, rule, 'J', s)
                tms[f'{nm} rule {rule}'] = (time.perf_counter() - t1) / (K - 1)
        rows['dec rule X'], ehs['dec rule X'] = chain(S, 'X', 'B')
        print(f'\nseed {sd} t={t0}  S_t: p [{S[0,0]:.3f},{S[0,1]:.3f}]  v [{S[1,0]:.3f},{S[1,1]:.3f}]'
              f'  hw(S) = ({hw(S)[0]:.4f}, {hw(S)[1]:.4f})')
        print(f'  {"":16s} ' + '  '.join(f'   k={k:<2d} p|v    ' for k in KS) + '   ms/step  contains adv')
        for nm, ch in list(rows.items()) + [('adv (inner)', adv), ('real (inner)', real)]:
            ok = '' if 'inner' in nm else ('yes' if contains(ch, adv) else 'NO !!')
            ms = f'{tms[nm] * 1e3:7.1f}' if nm in tms else '       '
            print(f'  {nm:16s} ' + '  '.join(f'{w(ch[k-1])[0]:7.3f}|{w(ch[k-1])[1]:6.3f}' for k in KS)
                  + f'  {ms}   {ok}')
        print(f'  e hw budget p|v  rule X ' + '  '.join(
            f'{e[0]:.4f}|{e[1]:.4f}' for e in [ehs["J256 rule X"][k - 2] if k > 1 else [np.nan] * 2 for k in KS]))
        print(f'  e hw budget p|v  rule S ' + '  '.join(
            f'{e[0]:.4f}|{e[1]:.4f}' for e in [ehs["J256 rule S"][k - 2] if k > 1 else [np.nan] * 2 for k in KS]))
        g = [(1 - w(rows['J256 rule S'][k-1]) / w(rows['J256 rule X'][k-1])) * 100 for k in KS]
        print('  rule S gain over X (J256)  ' + '  '.join(f'{a:5.1f}%|{b:5.1f}%' for a, b in g))
        rec = {n: [w(b).tolist() for b in ch] for n, ch in rows.items()}
        rec['adv'] = [w(b).tolist() for b in adv]
        rec['real'] = [w(b).tolist() for b in real]
        rec['e_budget'] = {n: [e.tolist() for e in v] for n, v in ehs.items()}
        rec['ms'] = tms
        rec['sound'] = {n: contains(ch, adv) for n, ch in rows.items()}
        summary[f's{sd}_t{t0}'] = rec
        for n, ch in rows.items():
            for k in KS:
                agg.setdefault((n, k), []).append(w(ch[k - 1]) / np.maximum(w(adv[k - 1]), 1e-9))
    print('\n=== median over starts: width / adversarial true (inner), p | v ===')
    names = list(rows.keys())
    print(f'  {"":16s} ' + '  '.join(f'   k={k:<2d}      ' for k in KS))
    for n in names:
        print(f'  {n:16s} ' + '  '.join(
            f'{np.median([r[0] for r in agg[(n, k)]]):5.2f}x|{np.median([r[1] for r in agg[(n, k)]]):5.2f}x'
            for k in KS))
    print('\n=== median rule-S gain over rule X, J256, p | v ===')
    for k in KS:
        gp = [100 * (1 - a[0] / b[0]) for a, b in zip(agg[('J256 rule S', k)], agg[('J256 rule X', k)])]
        gv = [100 * (1 - a[1] / b[1]) for a, b in zip(agg[('J256 rule S', k)], agg[('J256 rule X', k)])]
        print(f'  k={k:<2d}  p {np.median(gp):5.1f}% (range {min(gp):5.1f}..{max(gp):5.1f})'
              f'   v {np.median(gv):5.1f}% (range {min(gv):5.1f}..{max(gv):5.1f})')
    with open(f'{outdir}/summary_o.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('\nwrote', f'{outdir}/summary_o.json')


if __name__ == '__main__':
    main()
