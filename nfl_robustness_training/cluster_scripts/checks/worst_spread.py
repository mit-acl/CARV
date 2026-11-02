# Worst-case version of true_spread.py: search for the start state and noise
# sequence that spread the NN closed loop the most, instead of drawing noise at
# random. Anything found is a real trajectory, so the widths are still a lower
# bound on the true worst case, just a much better one.
#
# Same loop as the runs (MEAS_DELAY=1): u_k = NN(centre S_k),
#   z_k = x_k + n_k,  S_{k+1} = F(S_k & (z_k +- vmax), u_k),  x_{k+1} = f(x_k, u_k).
# Per start set, the widths at step k are max - min over every trajectory tried:
#   random    uniform noise (as true_spread.py)
#   extreme   noise at +-vmax with random signs, and constant-bias sequences
#   search    projected gradient ascent on +-x, +-y, +-theta at k = 4, 8, 12
#   python worst_spread.py <dumpdir> <vmax> [max_starts]
import ast, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
os.chdir(os.path.join(os.path.dirname(__file__), '../../..'))
from REAL_integrated_sim import setup_analyzer

ddir, vmax = sys.argv[1], float(sys.argv[2])
max_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 120
dt, v, K = 0.2, 1.0, 12
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s
torch.manual_seed(0)
rng = np.random.default_rng(0)

model = setup_analyzer('Unicycle_NL', 'natural_none_default').cl_system.controller.cpu()
for p in model.parameters():
    p.requires_grad_(False)
TWO_PI = 2 * np.pi


def cos_range(lo, hi):
    c = torch.stack([torch.cos(lo), torch.cos(hi)], -1)
    mn, mx = c.min(-1).values, c.max(-1).values
    mx = torch.where(TWO_PI * torch.ceil(lo / TWO_PI) <= hi, torch.ones_like(mx), mx)
    mn = torch.where(np.pi + TWO_PI * torch.ceil((lo - np.pi) / TWO_PI) <= hi, -torch.ones_like(mn), mn)
    return mn, mx


def rollout(B, x0, noise):
    """B (3,2) start set; x0 (m,3); noise (m,K-1,3). Returns states (m,K,3)."""
    m = x0.shape[0]
    lo, hi = B[:, 0].expand(m, 3).clone(), B[:, 1].expand(m, 3).clone()
    x, xs = x0, [x0]
    for k in range(K - 1):
        u = model((lo + hi) / 2).reshape(m)
        z = x + noise[:, k]
        lo, hi = torch.maximum(lo, z - vmax), torch.minimum(hi, z + vmax)
        c_lo, c_hi = cos_range(lo[:, 2], hi[:, 2])
        s_lo, s_hi = cos_range(lo[:, 2] - np.pi / 2, hi[:, 2] - np.pi / 2)
        lo = torch.stack([lo[:, 0] + dt * v * c_lo, lo[:, 1] + dt * v * s_lo, lo[:, 2] + dt * u], 1)
        hi = torch.stack([hi[:, 0] + dt * v * c_hi, hi[:, 1] + dt * v * s_hi, hi[:, 2] + dt * u], 1)
        x = torch.stack([x[:, 0] + dt * v * torch.cos(x[:, 2]),
                         x[:, 1] + dt * v * torch.sin(x[:, 2]), x[:, 2] + dt * u], 1)
        xs.append(x)
    return torch.stack(xs, 1)


def candidates(B):
    """Random, extreme-sign and constant-bias trajectories (no search)."""
    lo, hi = B[:, 0].numpy(), B[:, 1].numpy()
    out = []
    n = 1000
    x0 = rng.uniform(lo, hi, size=(n, 3))
    out.append((x0, rng.uniform(-vmax, vmax, size=(n, K - 1, 3)), 'random'))
    x0 = rng.uniform(lo, hi, size=(n, 3))
    x0[:, :] = np.where(rng.random((n, 3)) < 0.5, lo, hi)          # corners
    out.append((x0, vmax * rng.choice([-1.0, 1.0], size=(n, K - 1, 3)), 'extreme'))
    signs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
    corners = np.array([[i, j, l] for i in (0, 1) for j in (0, 1) for l in (0, 1)])
    x0 = np.array([[(lo, hi)[cc[d]][d] for d in range(3)] for cc in corners for _ in signs])
    nz = np.array([np.repeat(s[None] * vmax, K - 1, 0) for _ in corners for s in signs])
    out.append((x0, nz, 'extreme'))
    return out


def search(B, restarts=12, iters=120):
    """Projected Adam ascent on +-x, +-y, +-theta at k = 4, 8, 12."""
    objs = [(k, d, s) for k in (4, 8, 12) for d in range(3) for s in (1.0, -1.0)]
    m = len(objs) * restarts
    lo, hi = B[:, 0], B[:, 1]
    x0 = (lo + (hi - lo) * torch.rand(m, 3)).requires_grad_(True)
    nz = (vmax * (2 * torch.rand(m, K - 1, 3) - 1)).requires_grad_(True)
    kk = torch.tensor([o[0] - 1 for o in objs]).repeat_interleave(restarts)
    dd = torch.tensor([o[1] for o in objs]).repeat_interleave(restarts)
    ss = torch.tensor([o[2] for o in objs]).repeat_interleave(restarts)
    opt = torch.optim.Adam([x0, nz], lr=0.02 * vmax)
    for _ in range(iters):
        X = rollout(B, x0, nz)
        f = ss * X[torch.arange(m), kk, dd]
        opt.zero_grad()
        (-f.sum()).backward()
        opt.step()
        with torch.no_grad():
            x0.copy_(torch.maximum(torch.minimum(x0, hi), lo))
            nz.clamp_(-vmax, vmax)
    return x0.detach().numpy(), nz.detach().numpy()


files = sorted(glob.glob(ddir + '/o_*.json'))
starts = []
for f in files:
    for s in json.load(open(f))['steps']:
        b1 = P(s.get('bound_t1'))
        if b1 in (None, 'None'):
            continue
        t = int(P(s['t']))
        starts.append((np.array(b1, float), {int(tau) - t: np.array(b) for tau, b in (P(s.get('look')) or [])}))
pick = np.linspace(0, len(starts) - 1, min(max_starts, len(starts))).astype(int)

rows = {k: [] for k in range(1, K + 1)}
for i in pick:
    Bn, look = starts[i]
    B = torch.tensor(Bn, dtype=torch.float32)
    hw = (Bn[:, 1] - Bn[:, 0]) / 2
    ep0, eth = np.hypot(hw[0], hw[1]), hw[2]
    trajs = {}
    with torch.no_grad():
        for x0, nz, name in candidates(B):
            X = rollout(B, torch.tensor(x0, dtype=torch.float32), torch.tensor(nz, dtype=torch.float32)).numpy()
            trajs[name] = np.concatenate([trajs[name], X]) if name in trajs else X
    x0, nz = search(B)
    with torch.no_grad():
        trajs['search'] = rollout(B, torch.tensor(x0), torch.tensor(nz)).numpy()
    allX = np.concatenate(list(trajs.values()))
    for k in range(1, K + 1):
        w = lambda X: (max(np.ptp(X[:, k - 1, 0]), np.ptp(X[:, k - 1, 1])), np.ptp(X[:, k - 1, 2]))
        lb = look.get(k)
        ow = (np.nan, np.nan) if lb is None else (max(lb[0, 1] - lb[0, 0], lb[1, 1] - lb[1, 0]), lb[2, 1] - lb[2, 0])
        rows[k].append((*ow, 2 * (ep0 + (k - 1) * v * dt * eth), 2 * eth, *w(trajs['random']), *w(allX)))

print(f'{ddir}  vmax={vmax}  starts={len(pick)}   (median full widths; worst = max over random+extreme+search)')
print(f"{'k':>2} | {'xy: ours':>8} {'tube':>6} {'random':>7} {'worst':>6} {'worst<tube':>10} | "
      f"{'th: ours':>8} {'tube':>6} {'random':>7} {'worst':>6}")
for k in range(1, K + 1):
    a = np.array(rows[k])
    m = np.nanmedian(a, 0)
    print(f"{k:>2} | {m[0]:8.3f} {m[2]:6.3f} {m[4]:7.3f} {m[6]:6.3f} {np.mean(a[:, 6] < a[:, 2]):10.0%} | "
          f"{m[1]:8.3f} {m[3]:6.3f} {m[5]:7.3f} {m[7]:6.3f}")
