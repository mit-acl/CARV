# README figure: my reach boxes vs R2's tube vs the NN's real spread
# (Monte Carlo), from one start set of the k15 corridor run at noise +-0.15.
#   (a) the three in the plane, 12 steps ahead
#   (b) median widths over 120 start sets, read from the worst_spread.py log
#   python fig_spread.py <dumpdir> <worst_spread log> <out.png>
import ast, glob, json, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from REAL_integrated_sim import setup_analyzer

ddir, logf, out = sys.argv[1:4]
vmax, dt, v, K = 0.15, 0.2, 1.0, 12
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s
cl = setup_analyzer('Unicycle_NL', 'natural_none_default').cl_system
dyn, ctrl = cl.dynamics, cl.controller.cpu()
nn = lambda X: np.asarray(dyn.control_nn(np.asarray(X, float), ctrl)).reshape(len(X))
rng = np.random.default_rng(0)


def step(X, u):
    return np.stack([X[:, 0] + dt * v * np.cos(X[:, 2]), X[:, 1] + dt * v * np.sin(X[:, 2]),
                     X[:, 2] + dt * u], 1)


def cos_range(lo, hi):
    c = np.stack([np.cos(lo), np.cos(hi)], 1)
    mn, mx = c.min(1), c.max(1)
    mx = np.where(2 * np.pi * np.ceil(lo / (2 * np.pi)) <= hi, 1.0, mx)
    mn = np.where(np.pi + 2 * np.pi * np.ceil((lo - np.pi) / (2 * np.pi)) <= hi, -1.0, mn)
    return mn, mx


def box_step(S, u):
    c_lo, c_hi = cos_range(S[:, 2, 0], S[:, 2, 1])
    s_lo, s_hi = cos_range(S[:, 2, 0] - np.pi / 2, S[:, 2, 1] - np.pi / 2)
    O = S.copy()
    O[:, 0, 0] += dt * v * c_lo; O[:, 0, 1] += dt * v * c_hi
    O[:, 1, 0] += dt * v * s_lo; O[:, 1, 1] += dt * v * s_hi
    O[:, 2, :] += dt * u[:, None]
    return O


def rollout(B, X, noise):
    """Same loop as the runs (MEAS_DELAY=1). X (n,3) starts, noise (n,K-1,3)."""
    S = np.repeat(B[None], len(X), 0)
    out = [X]
    for k in range(K - 1):
        u = nn(S.mean(2))
        z = X + noise[:, k]
        S = box_step(np.stack([np.maximum(S[:, :, 0], z - vmax), np.minimum(S[:, :, 1], z + vmax)], 2), u)
        X = step(X, u)
        out.append(X)
    return np.stack(out, 1)


plt.rcdefaults()          # the sim imports restyle matplotlib


def tube_w(B, k):
    hw = (B[:, 1] - B[:, 0]) / 2
    return 2 * (np.hypot(hw[0], hw[1]) + (k - 1) * v * dt * hw[2])


# a TYPICAL start: NN driving, lookahead reaching 12 steps, and a box/tube
# width ratio at 12 steps closest to the median over all such starts
cands = []
for f in sorted(glob.glob(ddir + '/o_*.json')):
    for s in json.load(open(f))['steps']:
        look, b1 = P(s.get('look')), P(s.get('bound_t1'))
        if b1 in (None, 'None') or not look or s.get('kind') != 'empirical':
            continue
        t = int(P(s['t']))
        L = {int(tau) - t: np.array(b) for tau, b in look}
        if all(k in L for k in range(1, K + 1)):
            B = np.array(b1, float)
            r = max(np.ptp(L[K][0]), np.ptp(L[K][1])) / tube_w(B, K)
            cands.append((r, B, L, f, t))
med = np.median([c[0] for c in cands])
r, B, L, src, t0 = min(cands, key=lambda c: abs(c[0] - med))
print(f'start set from {os.path.basename(src)} t={t0}: box/tube at 12 steps {r:.2f}x (median {med:.2f}x, n={len(cands)})')

lo, hi = B[:, 0], B[:, 1]
n = 3000
X0 = rng.uniform(lo, hi, size=(n, 3))
nz = rng.uniform(-vmax, vmax, size=(n, K - 1, 3))
Xr = rollout(B, X0, nz)                                               # random noise
corners = np.array([[(lo, hi)[(i >> d) & 1][d] for d in range(3)] for i in range(8)])
signs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
Xc = rollout(B, np.repeat(corners, len(signs), 0),
             np.tile(np.repeat(signs[:, None] * vmax, K - 1, 1), (8, 1, 1)))  # constant bias
Xe = rollout(B, np.where(rng.random((1000, 3)) < 0.5, lo, hi),
             vmax * rng.choice([-1.0, 1.0], size=(1000, K - 1, 3)))        # extreme signs
allX = np.concatenate([Xr, Xc, Xe])

# R2's tube radius from the same start set, drawn around the NN's centre path
hw = (hi - lo) / 2
ep0, eth = np.hypot(hw[0], hw[1]), hw[2]
centre = rollout(B, ((lo + hi) / 2)[None], np.zeros((1, K - 1, 3)))[0]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
for k in range(1, K + 1):
    b = L[k]
    ax.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0], fill=False,
                           ec='#c0661a', lw=1.2, label='my reach boxes' if k == 1 else None))
    ax.add_patch(Circle(centre[k - 1, :2], ep0 + (k - 1) * v * dt * eth, fill=False, ec='#2a6fdb',
                        lw=1.2, ls='--', label="R2's tube (same start set)" if k == 1 else None))
    ax.scatter(allX[::4, k - 1, 0], allX[::4, k - 1, 1], s=0.5, c='k', alpha=0.25, lw=0,
               label='real NN, sampled (Monte Carlo)' if k == 1 else None)
ax.set_aspect('equal')
ax.set_title('One start set: where the robot could be, 1-12 steps ahead', fontsize=10)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.legend(loc='best', fontsize=8, markerscale=6)
ax.grid(alpha=0.25)

# panel (b): median widths over all start sets, from the worst_spread log
rows, on = [], False
for line in open(logf):
    if 'vmax=0.15' in line and 'starts=120' in line:
        on = True; continue
    if on:
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            rows.append([float(g) for g in m.groups()])
        elif rows:
            break
r = np.array(rows)
bx.plot(r[:, 0], r[:, 1], 'o-', color='#c0661a', label='my reach boxes')
bx.plot(r[:, 0], r[:, 2], 's--', color='#2a6fdb', label="R2's tube")
bx.plot(r[:, 0], r[:, 4], '^-', color='k', label='real NN, worst noise found')
bx.plot(r[:, 0], r[:, 3], '.:', color='0.5', label='real NN, random noise')
bx.set_xlabel('steps ahead (0.2 s each)'); bx.set_ylabel('width of the region (m)')
bx.set_title('Width vs steps ahead, median over 120 start sets', fontsize=10)
bx.grid(alpha=0.25); bx.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=130)
print('wrote', out)
