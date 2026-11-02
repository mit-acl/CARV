"""
exp28_stateunc -- standalone checks for the state-uncertainty redesign.

No process noise. The robot knows X0, the controls it sent, and measurements
z = x + v with |v|_inf <= VMAX. Nothing on the robot side reads x_true; it
is used only to simulate the plant and to audit containment.

  exp1  certified set  S_{t+1} = F(S_t, u_t) ∩ (z_{t+1} ± VMAX)
        vs measurement-only, prediction-only (from X0), and the EKF 3-sigma
        box without reset. Containment over many rollouts, widths over time.
  exp2  lookahead RSOA when the NN reads s = x + e, e in E:
        A  coupled f(x, NN(x))            (current RSOA; ignores e)
        B  decoupled u in NN(R ⊕ E)       (sound, loses x-u coupling)
        C  joint CROWN over (x, e)        (sound, keeps coupling)
  exp3  backup tube from a certified box: open-loop controls vs feedback
        u = sat(v_k + K(s - z_k)), s = z (raw) or s = centre of the
        certified set (error bound derived, see _fbc_E).

The controller class is copied from cl_systems/unicycle.py (including the
unused normaliser, so it computes exactly what the plant computes) to avoid
importing cl_systems, which pulls in nfl_veripy/JAX.

Usage:  python exp28_stateunc.py <outdir>
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

torch.set_num_threads(1)
REPO = '/home/ryosei_t/work/CARV'
DT, V = 0.2, 1.0
SIGMA = 0.05
VMAX = 3 * SIGMA
X0 = np.array([[-9.55, -9.45], [3.45, 3.55], [-np.pi / 24, np.pi / 24]])
PAD = 1e-5          # float32 slack on every CROWN output
UMAX = 1.0          # MPC turn-rate limit (unicycle_mpc_acados.py:332)


# ── controller (verbatim architecture of cl_systems.unicycle_4layer_controller) ──
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, xt):
        return (xt - self.mean) / self.std


class UnicycleCtrl(nn.Module):
    def __init__(self, npl, mean, std):
        super().__init__()
        self.norm = Normalizer(mean, std)
        self.fc1 = nn.Linear(3, npl[0])
        self.fc2 = nn.Linear(npl[0], npl[1])
        self.fc3 = nn.Linear(npl[1], npl[2])
        self.fc4 = nn.Linear(npl[2], 1)

    def forward(self, xt):
        ut = F.relu(self.fc1(xt))        # the original computes norm(xt) and discards it
        ut = F.relu(self.fc2(ut))
        ut = F.relu(self.fc3(ut))
        return self.fc4(ut)


def load_ctrl():
    c = UnicycleCtrl([40, 20, 10], torch.tensor([-7.5, 2.5, 0.]),
                     torch.tensor([7.5, 2.5, np.pi / 6]))
    p = (REPO + '/nfl_robustness_training/src/controller_models/Unicycle_NL/'
         'unicycle_nl_4layer/natural_none_default.pth')
    c.load_state_dict(torch.load(p, map_location='cpu')['state_dict'], strict=True)
    for q in c.parameters():
        q.requires_grad_(False)
    return c.eval()


CTRL = load_ctrl()


def nn_u(S):
    """NN control for a batch of inputs S (n,3), float32 like the plant."""
    with torch.no_grad():
        return CTRL(torch.tensor(np.atleast_2d(S), dtype=torch.float32)).numpy()[:, 0].astype(float)


def f_np(X, u):
    """Plant: Unicycle_NL.dynamics_step, vectorised. X (n,3), u (n,)."""
    X = np.atleast_2d(X)
    return np.stack([X[:, 0] + DT * V * np.cos(X[:, 2]),
                     X[:, 1] + DT * V * np.sin(X[:, 2]),
                     X[:, 2] + DT * u], axis=1)


# ── interval arithmetic ─────────────────────────────────────────────────────────
def cos_range(lo, hi):
    if hi - lo >= 2 * np.pi:
        return -1.0, 1.0
    vals = [np.cos(lo), np.cos(hi)]
    if 2 * np.pi * np.ceil(lo / (2 * np.pi)) <= hi:
        vals.append(1.0)
    if np.pi + 2 * np.pi * np.ceil((lo - np.pi) / (2 * np.pi)) <= hi:
        vals.append(-1.0)
    return min(vals), max(vals)


def sin_range(lo, hi):
    return cos_range(lo - np.pi / 2, hi - np.pi / 2)


def istep(b, ulo, uhi=None):
    """Interval enclosure of the plant over box b under u in [ulo, uhi]."""
    uhi = ulo if uhi is None else uhi
    c = cos_range(b[2, 0], b[2, 1])
    s = sin_range(b[2, 0], b[2, 1])
    return np.array([[b[0, 0] + DT * V * c[0], b[0, 1] + DT * V * c[1]],
                     [b[1, 0] + DT * V * s[0], b[1, 1] + DT * V * s[1]],
                     [b[2, 0] + DT * ulo, b[2, 1] + DT * uhi]])


def inter(a, b):
    return np.stack([np.maximum(a[:, 0], b[:, 0]), np.minimum(a[:, 1], b[:, 1])], 1)


def contains(b, X, tol=1e-9):
    X = np.atleast_2d(X)
    return np.all((X >= b[:, 0] - tol) & (X <= b[:, 1] + tol), axis=1)


def hw(b):
    return (b[:, 1] - b[:, 0]) / 2


# ── CROWN closed-loop modules ───────────────────────────────────────────────────
class CL(nn.Module):
    """k plant steps. Input [x (3) | e_0 (3) | ... | e_{k-1} (3)] if use_e.

    law 'nn'  u = NN(x + e)          (e absent => NN(x), the current RSOA)
    law 'ol'  u = v[i]               (known open-loop control)
    law 'fb'  u = sat(g_i . (x+e) + b_i)
    """

    def __init__(self, k, law, use_e, v=None, g=None, b=None):
        super().__init__()
        self.k, self.law, self.use_e = k, law, use_e
        self.ctrl = CTRL
        n = 3 + (3 * k if use_e else 0)
        eye = torch.eye(n)
        self.register_buffer('Px', eye[:, :3].clone())
        for i in range(k if use_e else 0):
            self.register_buffer(f'Pe{i}', eye[:, 3 + 3 * i:6 + 3 * i].clone())
        for j in range(3):
            self.register_buffer(f'S{j}', torch.eye(3)[:, j:j + 1].clone())
        if v is not None:
            self.register_buffer('v', torch.tensor(np.asarray(v), dtype=torch.float32).reshape(-1))
        if g is not None:
            self.register_buffer('g', torch.tensor(np.asarray(g), dtype=torch.float32))   # (k,3)
            self.register_buffer('b', torch.tensor(np.asarray(b), dtype=torch.float32))   # (k,)

    def forward(self, inp):
        x = inp @ self.Px
        for i in range(self.k):
            s = x + inp @ getattr(self, f'Pe{i}') if self.use_e else x
            if self.law == 'nn':
                u = self.ctrl(s)
            elif self.law == 'ol':
                u = x @ self.S0 * 0 + self.v[i]
            else:
                a = s @ self.g[i].reshape(3, 1) + self.b[i]
                u = a - F.relu(a - UMAX) + F.relu(-a - UMAX)
            x0, x1, x2 = x @ self.S0, x @ self.S1, x @ self.S2
            x = torch.cat([x0 + DT * V * torch.cos(x2), x1 + DT * V * torch.sin(x2),
                           x2 + DT * u], 1)
        return x


def crown(model, box):
    """Sound (up to PAD) output box of model over input box (n,2)."""
    lo = torch.tensor(box[:, 0], dtype=torch.float32)
    hi = torch.tensor(box[:, 1], dtype=torch.float32)
    c, eps = ((lo + hi) / 2).reshape(1, -1), (hi - lo) / 2 + 1e-7
    bm = BoundedModule(model, torch.zeros_like(c), device='cpu', verbose=False)
    lb, ub = bm.compute_bounds(x=(BoundedTensor(c, PerturbationLpNorm(eps=eps)),),
                               method='backward')
    return np.stack([lb.detach().numpy()[0] - PAD, ub.detach().numpy()[0] + PAD], 1).astype(float)


def nn_bounds(box):
    return crown(CTRL, box)[0]


def ebox(E):
    E = np.asarray(E, dtype=float)
    return np.stack([-E, E], 1)


# ── noise ───────────────────────────────────────────────────────────────────────
def draw(rng, n, kind):
    if kind == 'uniform':
        return rng.uniform(-VMAX, VMAX, size=(n, 3))
    return rng.normal(0, SIGMA, size=(n, 3))


# ═══ exp1: certified set vs alternatives ═══════════════════════════════════════
class EKF:
    """Same equations as state_estimator.ExtendedKalmanEstimator, Q=0.01^2."""

    def __init__(self, x0, b0, r_std, q_std=0.01):
        self.x = x0.copy()
        self.P = np.diag(((b0[:, 1] - b0[:, 0]) / 6.0) ** 2)
        self.Q, self.R = np.eye(3) * q_std ** 2, np.eye(3) * r_std ** 2

    def predict(self, u):
        th = self.x[2]
        Fj = np.array([[1, 0, -DT * V * np.sin(th)], [0, 1, DT * V * np.cos(th)], [0, 0, 1]])
        self.x = f_np(self.x, np.array([u]))[0]
        self.P = Fj @ self.P @ Fj.T + self.Q

    def update(self, z):
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(3) - K) @ self.P

    def box(self):
        s = 3 * np.sqrt(np.diag(self.P))
        return np.stack([self.x - s, self.x + s], 1)


def exp1(out, n_roll=300, T=40, kind='uniform', vmax=VMAX, seed=0):
    rng = np.random.default_rng(seed)
    r_std = VMAX / np.sqrt(3) if kind == 'uniform' else SIGMA
    names = ['certified', 'meas_only', 'pred_only', 'ekf']
    miss = {k: np.zeros(T + 1, int) for k in names}
    ep_miss = {k: 0 for k in names}
    width = {k: np.zeros((n_roll, T + 1, 3)) for k in names}
    empty = 0
    keep = None
    for r in range(n_roll):
        x = rng.uniform(X0[:, 0], X0[:, 1])
        z = x + draw(rng, 1, kind)[0]
        M = np.stack([z - vmax, z + vmax], 1)
        C, P = inter(X0, M), X0.copy()
        ekf = EKF(X0.mean(1), X0, r_std)
        ekf.update(z)
        hist = {'x': [x], 'z': [z], 'C': [C.copy()], 'u': []}
        had = {k: False for k in names}
        for t in range(T + 1):
            boxes = {'certified': C, 'meas_only': M, 'pred_only': P, 'ekf': ekf.box()}
            for k, b in boxes.items():
                width[k][r, t] = b[:, 1] - b[:, 0]
                if not contains(b, x)[0]:
                    miss[k][t] += 1
                    had[k] = True
            if t == T:
                break
            u = nn_u(z)[0]                       # the NN reads the raw measurement
            x = f_np(x, np.array([u]))[0]        # world
            z = x + draw(rng, 1, kind)[0]        # sensor
            M = np.stack([z - vmax, z + vmax], 1)
            Cp = istep(C, u)
            C = inter(Cp, M)
            if np.any(C[:, 0] > C[:, 1]):
                empty += 1
                C = M                            # counted; only possible if |v| > vmax
            P = istep(P, u)
            ekf.predict(u)
            ekf.update(z)
            hist['x'].append(x); hist['z'].append(z); hist['C'].append(C.copy()); hist['u'].append(u)
        for k in names:
            ep_miss[k] += had[k]
        if r == 0:
            keep = hist
    res = {'kind': kind, 'vmax': vmax, 'n_roll': n_roll, 'T': T,
           'step_miss': {k: int(miss[k].sum()) for k in names},
           'steps': n_roll * (T + 1),
           'rollouts_with_miss': ep_miss, 'empty_intersections': empty,
           'median_width_t20': {k: np.median(width[k][:, 20], 0).tolist() for k in names},
           'median_width_tT': {k: np.median(width[k][:, T], 0).tolist() for k in names}}
    return res, width, keep


# ═══ exp2: lookahead under s = x + e ═══════════════════════════════════════════
def exp2(C_t, z_t, K=10, n_mc=20000, seed=1):
    """Propagate K steps from certified box C_t; u_t = NN(z_t) is known."""
    E = np.full(3, VMAX)
    u_t = nn_u(z_t)[0]
    R1 = istep(C_t, u_t)
    modes = {}
    timing = {}
    # A: coupled, e ignored (the current RSOA)
    t0 = time.time()
    mA = CL(1, 'nn', False)
    b = R1.copy(); A = [b]
    for _ in range(K - 1):
        b = crown(mA, b); A.append(b)
    modes['A_coupled_no_e'] = A; timing['A'] = time.time() - t0
    # B: decoupled
    t0 = time.time()
    b = R1.copy(); B = [b]
    for _ in range(K - 1):
        ub = nn_bounds(b + ebox(E))
        b = istep(b, ub[0], ub[1]); B.append(b)
    modes['B_decoupled'] = B; timing['B'] = time.time() - t0
    # C: joint (x, e), concrete one-step chaining
    t0 = time.time()
    mC = CL(1, 'nn', True)
    b = R1.copy(); Cc = [b]
    for _ in range(K - 1):
        b = crown(mC, np.vstack([b, ebox(E)])); Cc.append(b)
    modes['C_joint_concrete'] = Cc; timing['C'] = time.time() - t0
    # C-symbolic: k-step joint from R1, k = 1..K-1
    t0 = time.time()
    Cs = [R1.copy()]
    for k in range(1, K):
        m = CL(k, 'nn', True)
        Cs.append(crown(m, np.vstack([R1] + [ebox(E)] * k)))
    modes['C_joint_symbolic'] = Cs; timing['Cs'] = time.time() - t0
    # Monte Carlo of the TRUE loop: x0 in C_t, u_t fixed, then u = NN(x + e)
    rng = np.random.default_rng(seed)
    X = rng.uniform(C_t[:, 0], C_t[:, 1], size=(n_mc, 3))
    X = f_np(X, np.full(n_mc, u_t))
    traj = [X]
    for _ in range(K - 1):
        # half the samples use extreme (bang) errors, which stress the bound
        e = rng.uniform(-VMAX, VMAX, size=(n_mc, 3))
        bang = rng.random(n_mc) < 0.5
        e[bang] = np.sign(e[bang]) * VMAX
        X = f_np(X, nn_u(X + e)); traj.append(X)
    miss = {k: [int((~contains(bx, traj[i])).sum()) for i, bx in enumerate(v)]
            for k, v in modes.items()}
    return modes, traj, miss, timing


# ═══ exp3: backup tube, open loop vs feedback ═════════════════════════════════
def nominal(z0, vseq):
    Z = [np.asarray(z0, float)]
    for v in vseq:
        Z.append(f_np(Z[-1], np.array([v]))[0])
    return np.array(Z)


def fb_gains(Z, vseq, kth, ky):
    """u = v - kth (s_th - z_th) - ky * crosstrack(s - z)  as  g.s + b."""
    G, Bv = [], []
    for i, v in enumerate(vseq):
        zx, zy, zt = Z[i]
        g = np.array([ky * np.sin(zt), -ky * np.cos(zt), -kth])
        b = v + kth * zt - ky * np.sin(zt) * zx + ky * np.cos(zt) * zy
        G.append(g); Bv.append(b)
    return np.array(G), np.array(Bv)


def _fbc_E(C_t, B_tau):
    """Error bound for s = centre(S_tau), S_tau = F(S,u) ∩ meas ∩ B_tau.

    x_tau in S_tau and s = centre(S_tau) give |s - x| <= hw(S_tau). S_tau lies in
    the meas box (hw = VMAX) and in B_tau. Its heading interval is carried
    forward by a known shift and only ever intersected, so hw_theta(S_tau) <=
    hw_theta(C_t). Every term is a proven bound; none is a guess.
    """
    E = np.minimum(VMAX, hw(B_tau))
    E[2] = min(E[2], hw(C_t)[2])
    return E


def tube(C_t, vseq, mode, kth=2.5, ky=2.0, chunk=1):
    K = len(vseq)
    Z = nominal(C_t.mean(1), vseq)
    G, Bv = fb_gains(Z, vseq, kth, ky)
    boxes, Es = [C_t.copy()], []
    j = 0
    while j < K:
        start = boxes[j]
        Estart = []
        for m in range(min(chunk, K - j)):
            idx = j + m
            if mode == 'ol':
                mod = CL(m + 1, 'ol', False, v=vseq[j:j + m + 1])
                b = crown(mod, start)
            else:
                E = np.full(3, VMAX) if mode == 'fb_z' else _fbc_E(C_t, boxes[idx])
                Estart.append(E); Es.append(E)
                mod = CL(m + 1, 'fb', True, g=G[j:j + m + 1], b=Bv[j:j + m + 1])
                b = crown(mod, np.vstack([start] + [ebox(e) for e in Estart]))
            boxes.append(b)
        j += min(chunk, K - j)
    return np.array(boxes), Z, Es


def tube_mc(C_t, vseq, mode, boxes, Es, kth=2.5, ky=2.0, n_mc=5000, seed=2):
    rng = np.random.default_rng(seed)
    Z = nominal(C_t.mean(1), vseq)
    G, Bv = fb_gains(Z, vseq, kth, ky)
    X = rng.uniform(C_t[:, 0], C_t[:, 1], size=(n_mc, 3))
    # every sample is a different world consistent with C_t; it carries its own filter
    Slo = np.repeat(C_t[None, :, 0], n_mc, 0)
    Shi = np.repeat(C_t[None, :, 1], n_mc, 0)
    traj, miss, ebreak = [X], [0], 0
    for i, v in enumerate(vseq):
        if mode == 'ol':
            u = np.full(n_mc, v)
        else:
            if mode == 'fb_z':
                s = X + rng.uniform(-VMAX, VMAX, size=X.shape)
            else:
                s = (Slo + Shi) / 2
                ebreak += int(np.any(np.abs(s - X) > Es[i] + 1e-9, axis=1).sum())
            u = np.clip(s @ G[i] + Bv[i], -UMAX, UMAX)
        Xn = f_np(X, u)
        if mode == 'fb_c':       # vectorised certified-set update
            cl, ch = np.empty_like(Slo), np.empty_like(Shi)
            for n in range(n_mc):
                bb = istep(np.stack([Slo[n], Shi[n]], 1), u[n])
                cl[n], ch[n] = bb[:, 0], bb[:, 1]
            zz = Xn + rng.uniform(-VMAX, VMAX, size=Xn.shape)
            Slo = np.maximum.reduce([cl, zz - VMAX, np.repeat(boxes[i + 1][None, :, 0], n_mc, 0)])
            Shi = np.minimum.reduce([ch, zz + VMAX, np.repeat(boxes[i + 1][None, :, 1], n_mc, 0)])
        X = Xn
        traj.append(X)
        miss.append(int((~contains(boxes[i + 1], X)).sum()))
    return traj, miss, ebreak


# ═══ plots ══════════════════════════════════════════════════════════════════════
def plots(out, r1, widths, keep, e2, e3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    col = {'certified': '#1D9E75', 'meas_only': '#888780', 'pred_only': '#BA7517', 'ekf': '#378ADD'}
    lab = {'certified': 'certified  F(S,u) ∩ meas', 'meas_only': 'measurement only',
           'pred_only': 'prediction only (from X0)', 'ekf': 'EKF 3σ (no reset)'}

    # fig 1: exp1
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for d, nm in enumerate(['x', 'y', 'θ']):
        for k in col:
            w = widths[k][:, :, d]
            t = np.arange(w.shape[1])
            ax[d].plot(t, np.median(w, 0), color=col[k], label=lab[k])
            ax[d].fill_between(t, np.percentile(w, 5, 0), np.percentile(w, 95, 0),
                               color=col[k], alpha=0.15)
        ax[d].set_title(f'box width in {nm} (median, 5–95%)')
        ax[d].set_xlabel('timestep'); ax[d].set_ylim(bottom=0)
    ax[0].legend(fontsize=8)
    ks = list(col)
    ax[3].bar(range(4), [100 * r1['rollouts_with_miss'][k] / r1['n_roll'] for k in ks],
              color=[col[k] for k in ks])
    ax[3].set_xticks(range(4)); ax[3].set_xticklabels(['certified', 'meas', 'pred', 'EKF'])
    ax[3].set_title('% rollouts where truth left the box')
    for i, k in enumerate(ks):
        ax[3].text(i, 100 * r1['rollouts_with_miss'][k] / r1['n_roll'] + 1,
                   f"{r1['rollouts_with_miss'][k]}/{r1['n_roll']}", ha='center', fontsize=9)
    ax[3].set_ylim(0, 105)
    fig.suptitle(f"exp1: state sets, bounded noise |v| ≤ {VMAX}, NN reads z, no process noise")
    fig.tight_layout(); fig.savefig(f'{out}/exp1_sets.png', dpi=130); plt.close(fig)

    # fig 2: exp2
    modes, traj, miss, _ = e2
    keys = ['A_coupled_no_e', 'B_decoupled', 'C_joint_concrete', 'C_joint_symbolic']
    titles = ['A: current RSOA, NN(x)\n(ignores that NN reads s≠x)', 'B: decoupled u ∈ NN(R⊕E)',
              'C: joint CROWN over (x,e)\nconcrete chaining', 'C: joint CROWN over (x,e)\nsymbolic k-step']
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    allpts = np.vstack(traj)
    for a, k, ti in zip(ax, keys, titles):
        for i, X in enumerate(traj):
            a.scatter(X[::20, 0], X[::20, 1], s=0.3, color='#D85A30', alpha=0.3)
        for b in modes[k]:
            a.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                  fill=False, color='#185FA5', lw=0.8))
        a.set_title(ti + f"\nmisses: {sum(miss[k])}", fontsize=10)
        a.set_aspect('equal')
        a.set_xlim(allpts[:, 0].min() - 0.6, allpts[:, 0].max() + 0.6)
        a.set_ylim(allpts[:, 1].min() - 0.6, allpts[:, 1].max() + 0.6)
    for k, c in zip(keys, ['#E24B4A', '#BA7517', '#1D9E75', '#534AB7']):
        w = [max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]) for b in modes[k]]
        ax[4].plot(range(1, len(w) + 1), w, color=c, marker='o', ms=3, label=k)
    ax[4].set_title('max(x,y) width per lookahead step'); ax[4].legend(fontsize=8)
    ax[4].set_xlabel('steps ahead')
    fig.suptitle('exp2: lookahead when the NN reads s = x + e, |e| ≤ vmax  (orange = true closed-loop samples)')
    fig.tight_layout(); fig.savefig(f'{out}/exp2_lookahead.png', dpi=130); plt.close(fig)

    # fig 3: exp3
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    cols = {'ol': '#BA7517', 'fb_z': '#378ADD', 'fb_c': '#1D9E75'}
    ttl = {'ol': 'open-loop backup (today)', 'fb_z': 'tube: feedback on raw z',
           'fb_c': 'tube: feedback on certified-set centre'}
    for a, m in zip(ax, ['ol', 'fb_z', 'fb_c']):
        boxes, Z, Es, traj, miss, eb = e3[m]
        for X in traj:
            a.scatter(X[::10, 0], X[::10, 1], s=0.3, color='#D85A30', alpha=0.3)
        for b in boxes:
            a.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                  fill=False, color=cols[m], lw=0.8))
        a.plot(Z[:, 0], Z[:, 1], 'k--', lw=0.8)
        extra = f", |s−x|>E: {eb}" if m == 'fb_c' else ''
        a.set_title(f"{ttl[m]}\nmisses: {sum(miss)}{extra}", fontsize=10)
        a.set_aspect('equal')
    lim = [min(a.get_xlim()[0] for a in ax[:3]), max(a.get_xlim()[1] for a in ax[:3]),
           min(a.get_ylim()[0] for a in ax[:3]), max(a.get_ylim()[1] for a in ax[:3])]
    for a in ax[:3]:
        a.set_xlim(lim[0], lim[1]); a.set_ylim(lim[2], lim[3])
    for m in ['ol', 'fb_z', 'fb_c']:
        boxes = e3[m][0]
        ax[3].plot([b[1, 1] - b[1, 0] for b in boxes], color=cols[m], label=f'{m}: y width')
        ax[3].plot([b[2, 1] - b[2, 0] for b in boxes], color=cols[m], ls=':', label=f'{m}: θ width')
    ax[3].set_title('tube width along the plan'); ax[3].legend(fontsize=8); ax[3].set_xlabel('plan step')
    fig.suptitle('exp3: backup tube from the same certified box (dashed = nominal, orange = true samples)')
    fig.tight_layout(); fig.savefig(f'{out}/exp3_tube.png', dpi=130); plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    summary = {}
    t0 = time.time()
    r1, widths, keep = exp1(out)
    summary['exp1_uniform'] = r1
    for kind, vm in [('gauss', 3 * SIGMA), ('gauss', 4 * SIGMA)]:
        r, _, _ = exp1(out, n_roll=300, kind=kind, vmax=vm, seed=5)
        summary[f'exp1_{kind}_{vm:.2f}'] = r
    print('exp1 done', time.time() - t0, flush=True)

    t_pick = 20
    C_t, z_t = keep['C'][t_pick], keep['z'][t_pick]
    e2 = exp2(C_t, z_t)
    summary['exp2'] = {'C_t': C_t.tolist(), 'miss': e2[2], 'timing_s': e2[3],
                       'final_width': {k: (v[-1][:, 1] - v[-1][:, 0]).tolist() for k, v in e2[0].items()}}
    print('exp2 done', time.time() - t0, flush=True)

    vseq = [0.6] * 8 + [0.0] * 12
    e3 = {}
    for m in ['ol', 'fb_z', 'fb_c']:
        boxes, Z, Es = tube(C_t, vseq, m)
        traj, miss, eb = tube_mc(C_t, vseq, m, boxes, Es)
        e3[m] = (boxes, Z, Es, traj, miss, eb)
        summary[f'exp3_{m}'] = {'miss': miss, 'E_violations': eb,
                                'final_width': (boxes[-1][:, 1] - boxes[-1][:, 0]).tolist(),
                                'max_xy_width': float(max(max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]) for b in boxes))}
    print('exp3 done', time.time() - t0, flush=True)
    plots(out, r1, widths, keep, e2, e3)
    with open(f'{out}/summary.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
