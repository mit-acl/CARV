# How wide does the NN's REAL closed-loop spread get, vs our reach boxes and
# R2's tube, from the same start sets? Sampling gives the true spread from
# below, so if it is already wider than the tube no reachability method can
# beat the tube with this NN.
#
# Start sets: bound_t1 of every step in the RSOA dumps (the set the filter
# holds for t+1). From each, sample n states and roll 11 steps:
#   truth: NN reads the true state (does the NN pull states together?)
#   noisy: NN reads the centre of the pipelined set-membership estimate,
#          S_{k+1} = F(S_k & (z_k +- vmax), u_k)   (MEAS_DELAY=1, as in the runs)
#   python true_spread.py <dumpdir> <vmax> [n_samples] [max_starts]
import ast, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
os.chdir(os.path.join(os.path.dirname(__file__), '../../..'))   # setup_analyzer wants the repo root
from REAL_integrated_sim import setup_analyzer

ddir, vmax = sys.argv[1], float(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
max_starts = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
dt, v, K = 0.2, 1.0, 12
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s

cl = setup_analyzer('Unicycle_NL', 'natural_none_default').cl_system
dyn, ctrl = cl.dynamics, cl.controller.cpu()
nn = lambda X: np.asarray(dyn.control_nn(np.asarray(X, float), ctrl)).reshape(len(X))

def step(X, u):
    return np.stack([X[:, 0] + dt * v * np.cos(X[:, 2]),
                     X[:, 1] + dt * v * np.sin(X[:, 2]), X[:, 2] + dt * u], 1)

def cos_range(lo, hi):          # vectorised _cos_range from alg29
    c = np.stack([np.cos(lo), np.cos(hi)], 1)
    mn, mx = c.min(1), c.max(1)
    mx = np.where(2 * np.pi * np.ceil(lo / (2 * np.pi)) <= hi, 1.0, mx)
    mn = np.where(np.pi + 2 * np.pi * np.ceil((lo - np.pi) / (2 * np.pi)) <= hi, -1.0, mn)
    return mn, mx

def box_step(S, u):             # S (n,3,2)
    c_lo, c_hi = cos_range(S[:, 2, 0], S[:, 2, 1])
    s_lo, s_hi = cos_range(S[:, 2, 0] - np.pi / 2, S[:, 2, 1] - np.pi / 2)
    O = S.copy()
    O[:, 0, 0] += dt * v * c_lo; O[:, 0, 1] += dt * v * c_hi
    O[:, 1, 0] += dt * v * s_lo; O[:, 1, 1] += dt * v * s_hi
    O[:, 2, :] += dt * u[:, None]
    return O

rng = np.random.default_rng(0)
res = {k: [] for k in range(1, K + 1)}   # (ours_xy, ours_th, tube_xy, tube_th, tr_xy, tr_th, nz_xy, nz_th)
starts = 0
for f in sorted(glob.glob(ddir + '/o_*.json')):
    for s in json.load(open(f))['steps']:
        if starts >= max_starts:
            break
        t, look, b1 = int(P(s['t'])), P(s.get('look')), P(s.get('bound_t1'))
        if b1 in (None, 'None'):
            continue
        starts += 1
        B = np.array(b1, float)
        look = {int(tau) - t: np.array(b) for tau, b in (look or [])}
        hw = (B[:, 1] - B[:, 0]) / 2
        ep0, eth = np.hypot(hw[0], hw[1]), hw[2]
        X0 = rng.uniform(B[:, 0], B[:, 1], size=(n, 3))
        X0[:8] = np.array([[B[0, i], B[1, j], B[2, l]] for i in (0, 1) for j in (0, 1) for l in (0, 1)])
        Xt, Xn = X0.copy(), X0.copy()
        S = np.repeat(B[None], n, 0)
        for k in range(1, K + 1):
            w = lambda X: (max(np.ptp(X[:, 0]), np.ptp(X[:, 1])), np.ptp(X[:, 2]))
            tr, nz = w(Xt), w(Xn)
            lb = look.get(k)
            ow = (np.nan, np.nan) if lb is None else (max(lb[0, 1] - lb[0, 0], lb[1, 1] - lb[1, 0]), lb[2, 1] - lb[2, 0])
            res[k].append((*ow, 2 * (ep0 + (k - 1) * v * dt * eth), 2 * eth, *tr, *nz))
            Xt = step(Xt, nn(Xt))
            u = nn(S.mean(2))
            z = Xn + rng.uniform(-vmax, vmax, size=Xn.shape)
            S = np.stack([np.maximum(S[:, :, 0], z - vmax), np.minimum(S[:, :, 1], z + vmax)], 2)
            S = box_step(S, u)
            Xn = step(Xn, u)
            assert np.all(Xn >= S[:, :, 0] - 1e-9) and np.all(Xn <= S[:, :, 1] + 1e-9), 'estimator lost truth'

print(f'{ddir}  vmax={vmax}  starts={starts}  samples/start={n}   (median full widths)')
print(f"{'k':>2} | {'xy: ours':>8} {'tube':>6} {'NN true':>8} {'NN noisy':>8} | {'th: ours':>8} {'tube':>6} {'NN true':>8} {'NN noisy':>8}")
for k in range(1, K + 1):
    a = np.array(res[k])
    m = np.nanmedian(a, 0)
    print(f"{k:>2} | {m[0]:8.3f} {m[2]:6.3f} {m[4]:8.3f} {m[6]:8.3f} | {m[1]:8.3f} {m[3]:6.3f} {m[5]:8.3f} {m[7]:8.3f}")
