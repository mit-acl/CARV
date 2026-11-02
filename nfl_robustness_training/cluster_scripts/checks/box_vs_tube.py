# Ours' lookahead boxes vs R2's open-loop tube built from the SAME start set.
# For every step t and every lookahead box held for t+k, compare its widths
# with the tube box R2 would use k-1 stages after starting from bound_t1
# (R2 starts its tube at the predicted set for t+1).
#   python box_vs_tube.py <dir with o_*.json RSOA dumps>
import ast, glob, json, sys
import numpy as np
v, dt = 1.0, 0.2
P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s
rows = {}   # k -> list of (ours_xy, ours_th, tube_xy, tube_th)
for f in sorted(glob.glob(sys.argv[1] + '/o_*.json')):
    for s in json.load(open(f))['steps']:
        t, look, b1 = int(P(s['t'])), P(s.get('look')), P(s.get('bound_t1'))
        if not look or b1 in (None, 'None'):
            continue
        hw = (np.array(b1)[:, 1] - np.array(b1)[:, 0]) / 2
        ep0, eth = np.hypot(hw[0], hw[1]), hw[2]
        for tau, b in look:
            k = int(tau) - t
            if k < 1 or k > 12:
                continue
            b = np.array(b)
            w = b[:, 1] - b[:, 0]
            e = ep0 + (k - 1) * v * dt * eth
            rows.setdefault(k, []).append((max(w[0], w[1]), w[2], 2 * e, 2 * eth))
print(f"{'k':>2} {'n':>5} | {'ours xy':>8} {'tube xy':>8} | {'ours th':>8} {'tube th':>8} | ours<tube xy")
for k in sorted(rows):
    a = np.array(rows[k])
    m = np.median(a, 0)
    print(f"{k:>2} {len(a):>5} | {m[0]:8.3f} {m[2]:8.3f} | {m[1]:8.3f} {m[3]:8.3f} | {np.mean(a[:,0] < a[:,2]):5.0%}")
