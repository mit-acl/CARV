"""alg43: neither side saturates. Paired by seed, pp at > 0.05.

Baselines pin the RAW nn(ctr) (TTTCARV_GP_PIN_RAW=1); *_clip = rev 7 clipped pin, same batch.
Ours = alg41 nx USAT=0 (NN unsaturated, sound).
Fair reference for the baselines is 'pred' = nn(ctr) of the landed solve, unclipped: the same
information ours is scored against (centre of F(S_{t-1} cap z_{t-1}, u_{t-1})).
Prediction stated before the run: R2 - ours lands in [+0.1, +1.45] pp.
"""
import json, os
import numpy as np

D = os.environ.get('A43_D', os.path.expanduser('~/runs/alg43_nosat'))
D41 = os.path.expanduser('~/runs/alg41_score')
THR = '0.05'
rng = np.random.default_rng(0)


def load(p):
    if not os.path.exists(p):
        return None
    return {t['seed']: t for t in json.load(open(p))['trials'] if 'error' not in t}


def row(t, ref):
    if 'gp_u_diff_len' in t:
        k = 'gp_u_diff_gt' if ref == 'raw' else f'gp_u_diff_gt_{ref}'
        return t['gp_u_diff_len'], t[k][THR]
    return t['u_diff_len'], t['u_diff_gt'][THR]


def pair(A, ra, B, rb):
    s = sorted(set(A) & set(B))
    a = np.array([row(A[k], ra) for k in s], float)
    b = np.array([row(B[k], rb) for k in s], float)
    idx = rng.integers(0, len(s), size=(4000, len(s)))
    d = 100 * (b[idx, 1].sum(1) / b[idx, 0].sum(1) - a[idx, 1].sum(1) / a[idx, 0].sum(1))
    pa, pb = 100 * a[:, 1].sum() / a[:, 0].sum(), 100 * b[:, 1].sum() / b[:, 0].sum()
    ra_, rb_ = a[:, 1] / a[:, 0], b[:, 1] / b[:, 0]
    return pa, pb, *np.percentile(d, [2.5, 97.5]), len(s), int((rb_ > ra_).sum()), int((rb_ < ra_).sum())


B = {}
print('== baselines (this batch)')
for n in ('r1_0.0_unif_raw', 'r2_0.1_unif_raw', 'r2_0.1_unif_clip',
          'r1_0.0_trunc_raw', 'r2_0.1_trunc_raw', 'r2_0.1_trunc_clip'):
    T = load(f'{D}/fp_b_{n}.json')
    if T is None:
        print(f'  {n:18s} missing'); continue
    B[n] = T
    tr = list(T.values())
    S = lambda k: sum((t.get(k) or 0) for t in tr)
    L = S('gp_u_diff_len')
    v = {r: 100 * sum(t['gp_u_diff_gt' if r == 'raw' else f'gp_u_diff_gt_{r}'][THR] for t in tr) / L
         for r in ('raw', 'pred', 'clip', 'predclip')}
    print(f'  {n:18s} n={len(tr)} raw {v["raw"]:5.2f} pred {v["pred"]:5.2f} clip {v["clip"]:5.2f}'
          f' predclip {v["predclip"]:5.2f}  coll {S("collision")}  pin {S("gp_pin_ok")}/{S("gp_pin_tries")}'
          f'  raw_reject {S("gp_pin_raw_reject")}  land>1 {S("gp_land_ub_over")}'
          f'  u_new_max {max(t.get("gp_u_new_max", 0) for t in tr):.2f}'
          f'  fallback {S("gp_fallback_steps")} loiter {S("gp_loiter_steps")}')

print('\n== clip control vs alg41 (node reproducibility; same code path)')
for nz in ('unif', 'trunc'):
    A, C = load(f'{D41}/fp_b_r2_0.1_{nz}.json'), B.get(f'r2_0.1_{nz}_clip')
    if A and C:
        pa, pb, lo, hi, n, _, _ = pair(A, 'predclip', C, 'predclip')
        print(f'  r2 {nz}: alg41 {pa:5.2f}  alg43 {pb:5.2f}  d={pb - pa:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}')

print('\n== baseline - ours (ours = alg41 nx USAT=0, raw)')
for nz, ov in (('unif', 'pipe_unif'), ('trunc', 'pipe_trunc')):
    O = load(f'{D41}/nx/fp_o_{ov}_sat0.json')
    Oi = load(f'{D41}/nx/fp_o_ideal_unif_sat0.json') if nz == 'unif' else None
    for bn, ref, Ov, lab in ((f'r2_0.1_{nz}_raw', 'pred', O, 'R2 raw pin, info-matched ref  vs ours pipe'),
                             (f'r2_0.1_{nz}_raw', 'raw', O, 'R2 raw pin, rev-7 ref         vs ours pipe'),
                             (f'r2_0.1_{nz}_clip', 'predclip', O, 'R2 clip pin, predclip (alg41) vs ours pipe'),
                             (f'r1_0.0_{nz}_raw', 'raw', Oi if Oi else O,
                              'R1 raw pin (delta=0)          vs ours ' + ('ideal' if Oi else 'pipe'))):
        if bn not in B or Ov is None:
            continue
        po, pb, lo, hi, n, hi_n, lo_n = pair(Ov, 'raw', B[bn], ref)
        print(f'  {nz:5s} {lab}: base {pb:5.2f} ours {po:5.2f} gap {pb - po:+.2f} [{lo:+.2f},{hi:+.2f}]'
              f' {pb / po:.2f}x  n={n}  base higher/lower {hi_n}/{lo_n}')
