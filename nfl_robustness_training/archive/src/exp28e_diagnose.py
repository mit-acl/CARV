"""
exp28e -- where exactly does the lookahead bound lose information?

The lookahead bounds x_{k+1} = f(x_k, NN(x_k + e_k)), x_0 in C_t, e_k in E.
Every method below propagates ONE BOX per step; they differ only in how the
next box is computed. Replacing one ingredient at a time by a (near-)exact
version isolates each source of looseness:

  B_crown   decoupled: u-range = CROWN of NN over (b + E), then istep     [alg27/28 mode B]
  B_oracle  decoupled, but u-range = the EXACT NN range over (b + E)
            (dense grid + random + corners; an inner estimate, so this is
            the best ANY sound NN bounder could do in decoupled form)
  C_crown   joint one-step CROWN over (x, e)                              [mode C]
  C_split   joint one-step CROWN, box split into cells, hull of the cells
  C_oracle  the EXACT hull of the one-step image of (b, E)
            (dense samples; the best ANY box-per-step method could do)
  true      hull of simulated trajectories (exp28b adversarial set)

Reading:  B_oracle >> true      -> decoupling (treating u independent of x) is the loss
          C_crown  >> C_oracle  -> CROWN relaxation is the loss (fix: splitting)
          C_oracle >> true      -> ONE BOX PER STEP is the loss (fix: a richer set)
Plus: inflation[k] = oracle one-step image of the TRUE hull box at step k,
divided by the true hull at k+1 -- the per-step price of re-boxing alone.
And the local closed-loop gain dNN/dtheta along the nominal trajectory.

Usage:  python exp28e_diagnose.py <outdir>
"""
import sys
import json
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import exp28_stateunc as E
import exp28b_looseness as B

K = 20
N_RAND = 60000
SPLIT = (2, 2, 8)          # cells in x, y, theta for C_split


class Step(nn.Module):
    """One closed-loop step, input [x | e] (6), u = NN(x + e)."""

    def __init__(self):
        super().__init__()
        self.ctrl = E.CTRL
        eye = torch.eye(6)
        self.register_buffer('Px', eye[:, :3].clone())
        self.register_buffer('Pe', eye[:, 3:].clone())
        for j in range(3):
            self.register_buffer(f'S{j}', torch.eye(3)[:, j:j + 1].clone())

    def forward(self, inp):
        x = inp @ self.Px
        u = self.ctrl(x + inp @ self.Pe)
        x0, x1, x2 = x @ self.S0, x @ self.S1, x @ self.S2
        return torch.cat([x0 + E.DT * E.V * torch.cos(x2), x1 + E.DT * E.V * torch.sin(x2),
                          x2 + E.DT * u], 1)


STEP = Step()
_BM = {}


def crown_batch(lo, hi):
    """CROWN output boxes for a batch of 6-d input boxes lo/hi (n,6). Returns (n,3,2)."""
    lo = torch.tensor(lo, dtype=torch.float32)
    hi = torch.tensor(hi, dtype=torch.float32)
    n = lo.shape[0]
    if n not in _BM:
        _BM[n] = BoundedModule(STEP, torch.zeros(n, 6), device='cpu', verbose=False)
    x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
    lb, ub = _BM[n].compute_bounds(x=(x,), method='backward')
    return np.stack([lb.detach().numpy() - E.PAD, ub.detach().numpy() + E.PAD], 2).astype(float)


def sample_box(b, n, rng, grid=0):
    d = b.shape[0]
    P = [rng.uniform(b[:, 0], b[:, 1], size=(n, d)), B.corners(b) if d == 3 else
         np.array(list(itertools.product(*[(b[i, 0], b[i, 1]) for i in range(d)])))]
    if grid:
        g = [np.linspace(b[i, 0], b[i, 1], grid) for i in range(d)]
        P.append(np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, d))
    return np.vstack(P)


def nn_range_exact(b, rng):
    X = sample_box(b, N_RAND, rng, grid=41)
    u = E.nn_u(X)
    return float(u.min()), float(u.max())


def image_exact(b, eb, rng):
    """Hull of {f(x, NN(x+e)) : x in b, e in eb} by dense sampling (inner estimate)."""
    Z = sample_box(np.vstack([b, eb]), N_RAND, rng)
    # also a 3-d grid over x with every e corner, where extremes usually sit
    g = sample_box(b, 0, rng, grid=17)[8:]
    ec = np.array(list(itertools.product(*[(eb[i, 0], 0.0, eb[i, 1]) for i in range(3)])))
    Z = np.vstack([Z, np.hstack([np.repeat(g, len(ec), 0), np.tile(ec, (len(g), 1))])])
    X, e = Z[:, :3], Z[:, 3:]
    Y = E.f_np(X, E.nn_u(X + e))
    return np.stack([Y.min(0), Y.max(0)], 1)


def c_split(b, eb):
    edges = [np.linspace(b[i, 0], b[i, 1], SPLIT[i] + 1) for i in range(3)]
    lo, hi = [], []
    for ix, iy, it in itertools.product(*[range(s) for s in SPLIT]):
        lo.append([edges[0][ix], edges[1][iy], edges[2][it], *eb[:, 0]])
        hi.append([edges[0][ix + 1], edges[1][iy + 1], edges[2][it + 1], *eb[:, 1]])
    out = crown_batch(np.array(lo), np.array(hi))
    return np.stack([out[:, :, 0].min(0), out[:, :, 1].max(0)], 1)


def w(b):
    return [float(b[i, 1] - b[i, 0]) for i in range(3)]


def run(C_t, z_t, vmax, rng):
    eb = E.ebox(np.full(3, vmax))
    R1 = E.istep(C_t, E.nn_u(z_t)[0])
    ch = {m: [R1.copy()] for m in ['B_crown', 'B_oracle', 'C_crown', 'C_split', 'C_oracle']}
    tight = {'B_crownNN_over_oracleNN': [], 'C_crown_over_oracle_th': [], 'C_split_over_oracle_th': []}
    t0 = time.time()
    for k in range(K - 1):
        b = ch['B_crown'][-1]
        lo, hi = E.nn_bounds(b + eb)
        ch['B_crown'].append(E.istep(b, lo, hi))
        b = ch['B_oracle'][-1]
        lo, hi = nn_range_exact(b + eb, rng)
        ch['B_oracle'].append(E.istep(b, lo, hi))
        ch['C_crown'].append(crown_batch(np.hstack([ch['C_crown'][-1][:, 0], eb[:, 0]])[None],
                                         np.hstack([ch['C_crown'][-1][:, 1], eb[:, 1]])[None])[0])
        ch['C_split'].append(c_split(ch['C_split'][-1], eb))
        b = ch['C_oracle'][-1]
        ch['C_oracle'].append(image_exact(b, eb, rng))
        # same-box tightness: CROWN vs exact on the C_oracle box (isolates the relaxation)
        ex = ch['C_oracle'][-1]
        cr = crown_batch(np.hstack([b[:, 0], eb[:, 0]])[None], np.hstack([b[:, 1], eb[:, 1]])[None])[0]
        sp = c_split(b, eb)
        tight['C_crown_over_oracle_th'].append((cr[2, 1] - cr[2, 0]) / (ex[2, 1] - ex[2, 0]))
        tight['C_split_over_oracle_th'].append((sp[2, 1] - sp[2, 0]) / (ex[2, 1] - ex[2, 0]))
        lo_o, hi_o = nn_range_exact(b + eb, rng)
        lo_c, hi_c = E.nn_bounds(b + eb)
        tight['B_crownNN_over_oracleNN'].append((hi_c - lo_c) / (hi_o - lo_o))
        print(f'  step {k + 2} done {time.time() - t0:.0f}s  th: ' +
              ' '.join(f'{m}={ch[m][-1][2, 1] - ch[m][-1][2, 0]:.3f}' for m in ch), flush=True)
    return ch, tight


def true_traj(C_t, z_t, vmax, rng):
    X0 = np.vstack([B.corners(C_t), rng.uniform(C_t[:, 0], C_t[:, 1], size=(300, 3))])
    seqs = B.e_sequences(K - 1, vmax, rng, 50)
    X = np.repeat(X0, len(seqs), 0)
    S = np.tile(np.stack(seqs), (len(X0), 1, 1))
    X = E.f_np(X, np.full(len(X), E.nn_u(z_t)[0]))
    traj = [X]
    for k in range(K - 1):
        X = E.f_np(X, E.nn_u(X + S[:, k])); traj.append(X)
    return traj


def gains(z_t):
    """dNN/d(x,y,theta) along the nominal (e = 0) trajectory from z_t."""
    x = torch.tensor(np.atleast_2d(z_t), dtype=torch.float32)
    out = []
    for _ in range(K):
        xi = x.clone().requires_grad_(True)
        u = E.CTRL(xi)
        g, = torch.autograd.grad(u.sum(), xi)
        out.append(g.numpy()[0].tolist())
        xn = x.numpy()
        x = torch.tensor(E.f_np(xn, np.array([u.item()])), dtype=torch.float32)
    return out


def main():
    out = sys.argv[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)
    boxes0 = B.start_boxes()
    summary = {}
    cols = {'B_crown': '#BA7517', 'B_oracle': '#E24B4A', 'C_crown': '#1D9E75',
            'C_split': '#378ADD', 'C_oracle': '#534AB7'}
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    steps = np.arange(1, K + 1)
    for row, (name, (C_t, z_t, vmax)) in enumerate(boxes0.items()):
        print(name, 'start width', w(C_t), 'vmax', vmax, flush=True)
        ch, tight = run(C_t, z_t, vmax, rng)
        traj = true_traj(C_t, z_t, vmax, rng)
        h = B.hull(traj)
        eb = E.ebox(np.full(3, vmax))
        infl = []
        for k in range(K - 1):
            im = image_exact(h[k], eb, rng)
            infl.append([(im[i, 1] - im[i, 0]) / (h[k + 1][i, 1] - h[k + 1][i, 0]) for i in range(3)])
        # the no-noise reference: does re-boxing blow up even when the NN reads x exactly?
        zero = E.ebox(np.zeros(3))
        b = E.istep(C_t, E.nn_u(z_t)[0]); co0 = [b]
        for _ in range(K - 1):
            b = image_exact(b, zero, rng); co0.append(b)
        t0 = true_traj(C_t, z_t, 0.0, rng)
        h0 = B.hull(t0)
        S = summary[name] = {
            'start_width': w(C_t), 'vmax': vmax,
            'width': {m: [w(b) for b in v] for m, v in ch.items()},
            'true_width': [w(b) for b in h],
            'same_box_tightness': tight,
            'reboxing_inflation_per_step_from_true_hull': infl,
            'noe_C_oracle_width': [w(b) for b in co0], 'noe_true_width': [w(b) for b in h0],
            'nn_grad_along_nominal': gains(z_t),
            'sound_vs_true': {m: all(bool(np.all(v[k][:, 0] <= h[k][:, 0] + 1e-6) and
                                          np.all(v[k][:, 1] >= h[k][:, 1] - 1e-6)) for k in range(K))
                              for m, v in ch.items()}}
        for m, v in ch.items():
            ax[row, 0].plot(steps, [b[2, 1] - b[2, 0] for b in v], color=cols[m], marker='o', ms=3, label=m)
            ax[row, 1].plot(steps, [max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]) for b in v], color=cols[m],
                            marker='o', ms=3, label=m)
        for j, (i, lab) in enumerate([(2, 'θ width (rad)'), (None, 'max(x,y) width (m)')]):
            f = (lambda b: b[2, 1] - b[2, 0]) if i == 2 else (lambda b: max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]))
            ax[row, j].plot(steps, [f(b) for b in h], 'k--', lw=2, label='true')
            ax[row, j].plot(steps, [f(b) for b in co0], color='#534AB7', ls=':', label='C_oracle, e=0')
            ax[row, j].plot(steps, [f(b) for b in h0], 'k:', label='true, e=0')
            ax[row, j].set_yscale('log'); ax[row, j].set_xlabel('steps ahead')
            ax[row, j].set_title(f'{name}: {lab}'); ax[row, j].legend(fontsize=7)
        a = ax[row, 2]
        a.plot(steps[1:], [r[2] for r in infl], 'k-o', ms=3, label='re-box inflation θ (true hull → 1 step)')
        a.plot(steps[1:], [max(r[0], r[1]) for r in infl], 'k--', label='re-box inflation xy')
        a.plot(steps[1:], tight['C_crown_over_oracle_th'], color='#1D9E75', label='CROWN / exact, same box (θ)')
        a.plot(steps[1:], tight['C_split_over_oracle_th'], color='#378ADD', label='split CROWN / exact, same box')
        a.plot(steps[1:], tight['B_crownNN_over_oracleNN'], color='#BA7517', label='CROWN NN range / exact')
        a.axhline(1, color='k', lw=0.8); a.set_yscale('log'); a.legend(fontsize=7)
        a.set_title(f'{name}: per-step looseness factors'); a.set_xlabel('steps ahead')
    fig.suptitle('exp28e: which ingredient of the lookahead bound loses the information')
    fig.tight_layout(); fig.savefig(f'{out}/exp28e_diagnose.png', dpi=110)
    with open(f'{out}/summary_e.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
