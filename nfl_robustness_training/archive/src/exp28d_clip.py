"""
exp28d -- what happens to the lookahead bounds if the turn rate saturates?

Assumption under test: the PLANT applies u = clip(NN(z), -c, c) (an actuator
limit), and the bound knows it. Each clip level c changes the true system too,
so each bound is compared with the true set of the SAME clipped system.

Per start box (tight / realistic, from exp28b) and per c:
  B        decoupled: CROWN of NN over (box + e-box), clipped to [-c, c], then istep
  C        joint (x, e) CROWN of one clipped closed-loop step, chained
  clip     the NN ignored entirely: u in [-c, c] (the trivial bound)
  true     hull of simulated trajectories: every start corner + 300 random starts
           x (27 constant biases + 100 random/bang error sequences), NN reads x+e
Usage:  python exp28d_clip.py <outdir>
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import exp28_stateunc as E
import exp28b_looseness as B

K = 20
CLIPS = [None, 2.0, 1.5, 1.2, 1.0]


def sat_np(u, c):
    return u if c is None else np.clip(u, -c, c)


class CLsat(nn.Module):
    """One closed-loop step, input [x | e], u = sat_c(NN(x + e))."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ctrl = E.CTRL
        eye = torch.eye(6)
        self.register_buffer('Px', eye[:, :3].clone())
        self.register_buffer('Pe', eye[:, 3:].clone())
        for j in range(3):
            self.register_buffer(f'S{j}', torch.eye(3)[:, j:j + 1].clone())

    def forward(self, inp):
        x = inp @ self.Px
        a = self.ctrl(x + inp @ self.Pe)
        u = a if self.c is None else a - F.relu(a - self.c) + F.relu(-a - self.c)
        x0, x1, x2 = x @ self.S0, x @ self.S1, x @ self.S2
        return torch.cat([x0 + E.DT * E.V * torch.cos(x2), x1 + E.DT * E.V * torch.sin(x2),
                          x2 + E.DT * u], 1)


def bounds(C_t, z_t, vmax, c):
    eb = E.ebox(np.full(3, vmax))
    R1 = E.istep(C_t, float(sat_np(E.nn_u(z_t)[0], c)))
    out = {'B': [R1.copy()], 'C': [R1.copy()], 'clip': [R1.copy()]}
    nsat = 0
    m = CLsat(c)
    b = R1.copy()
    for _ in range(K - 1):
        lo, hi = E.nn_bounds(b + eb)
        if c is not None:
            nsat += (hi > c) or (lo < -c)
            lo, hi = max(lo, -c), min(hi, c)
        b = E.istep(b, lo, hi); out['B'].append(b)
    b = R1.copy()
    for _ in range(K - 1):
        b = E.crown(m, np.vstack([b, eb])); out['C'].append(b)
    b = R1.copy()
    cc = 1e9 if c is None else c
    for _ in range(K - 1):
        b = E.istep(b, -cc, cc); out['clip'].append(b)
    return out, nsat


def true_hull(C_t, z_t, vmax, c, seed=3):
    rng = np.random.default_rng(seed)
    X0 = np.vstack([B.corners(C_t), rng.uniform(C_t[:, 0], C_t[:, 1], size=(300, 3))])
    seqs = B.e_sequences(K - 1, vmax, rng, 50)
    X = np.repeat(X0, len(seqs), 0)
    S = np.tile(np.stack(seqs), (len(X0), 1, 1))
    X = E.f_np(X, np.full(len(X), sat_np(E.nn_u(z_t)[0], c)))
    traj, umax = [X], 0.0
    for k in range(K - 1):
        u = E.nn_u(X + S[:, k])
        umax = max(umax, float(np.abs(u).max()))
        X = E.f_np(X, sat_np(u, c)); traj.append(X)
    return B.hull(traj), umax, len(X0) * len(seqs)


def w_xy(b):
    return float(max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]))


def w_th(b):
    return float(b[2, 1] - b[2, 0])


def main():
    out = sys.argv[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    boxes0 = B.start_boxes()
    summary = {}
    cols = dict(zip([str(c) for c in CLIPS], ['#E24B4A', '#BA7517', '#1D9E75', '#378ADD', '#534AB7']))
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    steps = np.arange(1, K + 1)
    for row, (name, (C_t, z_t, vmax)) in enumerate(boxes0.items()):
        S = summary[name] = {'C_t_width': (C_t[:, 1] - C_t[:, 0]).tolist(), 'vmax': vmax}
        for c in CLIPS:
            bd, nsat = bounds(C_t, z_t, vmax, c)
            h, umax, n = true_hull(C_t, z_t, vmax, c)
            S[str(c)] = {
                'B_xy': [w_xy(b) for b in bd['B']], 'B_th': [w_th(b) for b in bd['B']],
                'C_xy': [w_xy(b) for b in bd['C']], 'C_th': [w_th(b) for b in bd['C']],
                'clip_xy': [w_xy(b) for b in bd['clip']], 'clip_th': [w_th(b) for b in bd['clip']],
                'true_xy': [w_xy(b) for b in h], 'true_th': [w_th(b) for b in h],
                'B_steps_where_clip_binds': int(nsat), 'true_max_abs_nn_u': umax, 'n_traj': n,
                'C_contains_true': all(bool(np.all(bd['C'][k][:, 0] <= h[k][:, 0] + 1e-6) and
                                            np.all(bd['C'][k][:, 1] >= h[k][:, 1] - 1e-6)) for k in range(K)),
                'B_contains_true': all(bool(np.all(bd['B'][k][:, 0] <= h[k][:, 0] + 1e-6) and
                                            np.all(bd['B'][k][:, 1] >= h[k][:, 1] - 1e-6)) for k in range(K))}
            r = S[str(c)]
            col = cols[str(c)]
            lab = 'no clip' if c is None else f'clip ±{c}'
            ax[row, 0].plot(steps, r['B_xy'], color=col, marker='o', ms=3, label=f'{lab} B')
            ax[row, 0].plot(steps, r['C_xy'], color=col, ls='-.', label=f'{lab} C')
            ax[row, 0].plot(steps, r['true_xy'], color=col, ls='--', lw=1, label=f'{lab} true')
            ax[row, 1].plot(steps, r['B_th'], color=col, marker='o', ms=3, label=f'{lab} B')
            ax[row, 1].plot(steps, r['C_th'], color=col, ls='-.')
            ax[row, 1].plot(steps, r['true_th'], color=col, ls='--', lw=1)
            ax[row, 2].plot(steps, np.array(r['B_xy']) / np.array(r['true_xy']), color=col,
                            marker='o', ms=3, label=f'{lab} B / true')
            ax[row, 2].plot(steps, np.array(r['C_xy']) / np.array(r['true_xy']), color=col, ls='-.',
                            label=f'{lab} C / true')
            print(name, lab, 'B xy@5/10/20', [round(r['B_xy'][k], 3) for k in (4, 9, 19)],
                  'C', [round(r['C_xy'][k], 3) for k in (4, 9, 19)],
                  'true', [round(r['true_xy'][k], 3) for k in (4, 9, 19)],
                  'clip-only', [round(r['clip_xy'][k], 3) for k in (4, 9, 19)],
                  'th B@20', round(r['B_th'][-1], 3), 'true th@20', round(r['true_th'][-1], 3),
                  'binds', nsat, 'umax', round(umax, 3), 'sound', r['B_contains_true'], r['C_contains_true'],
                  flush=True)
        for j, t in enumerate(['max(x,y) width (m)', 'θ width (rad)', 'bound / true xy width']):
            a = ax[row, j]
            a.set_yscale('log'); a.set_xlabel('steps ahead'); a.set_title(f'{name}: {t}')
        ax[row, 0].legend(fontsize=6, ncol=3); ax[row, 1].legend(fontsize=7); ax[row, 2].legend(fontsize=6, ncol=2)
        ax[row, 2].axhline(1, color='k', lw=1)
    fig.suptitle('exp28d: lookahead bounds when the turn rate saturates at ±c '
                 '(solid B decoupled, dash-dot C joint, dashed true hull of the same clipped system)')
    fig.tight_layout(); fig.savefig(f'{out}/exp28d_clip.png', dpi=110)
    with open(f'{out}/summary_d.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
