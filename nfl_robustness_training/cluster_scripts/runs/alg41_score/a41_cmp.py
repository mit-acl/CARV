"""alg41: how much of the ours-vs-baseline intervention gap is scoring.

Baselines (alg29, scoring-only patch) carry four references per step:
  raw       |u - nn(x_hat)|                      (rev 7 headline)
  clip      |sat(u) - sat(nn(x_hat))|             (same +-1 action space)
  pred      |u - nn(c)|, c = centre the landed solve decided on (R2/G2: x_hat_pred)
  predclip  both
Ours runs with USAT=0 (rev 7: NN flown unsaturated) and USAT=1 (NN saturated at
+-1 in plant, lookahead and reference), same exclusive node.
"""
import json, os
import numpy as np

D = os.path.expanduser('~/runs/alg41_score')
THR = '0.05'
rng = np.random.default_rng(0)


# Ours ran in whichever of excl.sbatch (D) / nx.sbatch (D/nx) started first.
D_OURS = os.environ.get('A41_OURS', D if os.path.exists(f'{D}/fp_o_pipe_unif_sat1.json') else f'{D}/nx')


def load(name):
    p = f'{D_OURS if name.startswith("o_") else D}/fp_{name}.json'
    if not os.path.exists(p):
        return None
    # Crashed trials (an 'error' key, no counters) are dropped; pairing then uses the rest.
    return {t['seed']: t for t in json.load(open(p))['trials'] if 'error' not in t}


def row(t, ref='raw'):
    if 'gp_u_diff_len' in t:
        key = 'gp_u_diff_gt' if ref == 'raw' else f'gp_u_diff_gt_{ref}'
        return t['gp_u_diff_len'], t[key][THR], int(bool(t['collision']))
    return t['u_diff_len'], t['u_diff_gt'][THR], int(bool(t['collision']))


def pct(a):
    return 100 * a[:, 1].sum() / a[:, 0].sum()


def boot(a, b):
    idx = rng.integers(0, len(a), size=(4000, len(a)))
    d = 100 * (b[idx, 1].sum(1) / b[idx, 0].sum(1) - a[idx, 1].sum(1) / a[idx, 0].sum(1))
    return np.percentile(d, [2.5, 97.5])


def pair(A, ra, B, rb):
    s = sorted(set(A) & set(B))
    a = np.array([row(A[k], ra) for k in s], float)
    b = np.array([row(B[k], rb) for k in s], float)
    lo, hi = boot(a, b)
    return pct(a), pct(b), lo, hi, len(s)


print('== baselines: intervention % (> 0.05) under each reference')
base = ['g0_none', 'g0_unif', 'g0_trunc', 'g2_0.1_none', 'g2_0.1_unif', 'g2_0.1_trunc',
        'g3_0.0_unif', 'r1_0.0_unif', 'r1_0.0_trunc', 'r2_0.1_unif', 'r2_0.1_trunc']
B = {}
for n in base:
    T = load(f'b_{n}')
    if T is None:
        print(f'  {n:14s} missing'); continue
    B[n] = T
    tr = list(T.values())
    L = sum(t['gp_u_diff_len'] for t in tr)
    vals = []
    for r in ('raw', 'clip', 'pred', 'predclip'):
        k = 'gp_u_diff_gt' if r == 'raw' else f'gp_u_diff_gt_{r}'
        vals.append(100 * sum(t[k][THR] for t in tr) / L)
    sat = 100 * sum(t.get('gp_nn_sat_steps', 0) for t in tr) / L
    pf = 100 * sum(t['gp_pin_tries'] - t['gp_pin_ok'] for t in tr) / L
    coll = sum(bool(t['collision']) for t in tr)
    print(f'  {n:14s} raw {vals[0]:5.2f}  clip {vals[1]:5.2f}  pred {vals[2]:5.2f}'
          f'  predclip {vals[3]:5.2f}   |nn|>1.05 {sat:5.2f}  pin_fail {pf:5.2f}  coll {coll}')

print('\n== ours: USAT=0 (rev 7) vs USAT=1, same node')
O = {}
for v in ('pipe_unif', 'pipe_trunc', 'ideal_unif'):
    T0, T1 = load(f'o_{v}_sat0'), load(f'o_{v}_sat1')
    O[v] = (T0, T1)
    if T0 is None or T1 is None:
        print(f'  {v}: missing {"sat0" if T0 is None else "sat1"}'); continue
    p0, p1, lo, hi, n = pair(T0, 'raw', T1, 'raw')
    tr1 = list(T1.values())
    S = lambda T, k: sum((t.get(k) or 0) for t in T.values())
    print(f'  {v:11s} sat0 {p0:5.2f}  sat1 {p1:5.2f}  d={p1 - p0:+.2f} [{lo:+.2f},{hi:+.2f}]  n={n}'
          f'  coll {S(T0, "collision")}/{S(T1, "collision")}'
          f'  audit_miss {S(T0, "snd_audit_miss")}/{S(T1, "snd_audit_miss")}'
          f'  fb_uncert {S(T0, "mpc_fallback_uncertified")}/{S(T1, "mpc_fallback_uncertified")}'
          f'  sat_steps {S(T1, "est_sat_steps")}  u_raw_max {max((t.get("est_u_raw_max") or 0) for t in tr1):.2f}')
    w = max(max(np.atleast_1d(t['wall_per_timestep_s'])) for t in tr1
            if t.get('wall_per_timestep_s') is not None)
    print(f'              sat1 wall_max {w:.3f}')

print('\n== gap (baseline - ours), paired, pp at > 0.05')
cmp = [('rev 7 scoring:  raw  vs ours USAT=0', 'raw', 0),
       ('same actions:   clip vs ours USAT=1', 'clip', 1),
       ('+ info-matched: predclip vs ours USAT=1', 'predclip', 1)]
for bn, ov in (('r2_0.1_unif', 'pipe_unif'), ('r2_0.1_trunc', 'pipe_trunc'),
               ('r1_0.0_unif', 'ideal_unif'), ('g0_unif', 'ideal_unif'),
               ('g2_0.1_unif', 'pipe_unif'), ('g3_0.0_unif', 'ideal_unif')):
    if bn not in B or O.get(ov) is None:
        continue
    print(f'  {bn} vs ours {ov}')
    for lab, ref, s in cmp:
        To = O[ov][s]
        if To is None:
            continue
        po, pb, lo, hi, n = pair(To, 'raw', B[bn], ref)
        print(f'    {lab:42s} base {pb:5.2f}  ours {po:5.2f}  gap {pb - po:+.2f} [{lo:+.2f},{hi:+.2f}]'
              f'  ratio {pb / po:.2f}x')
