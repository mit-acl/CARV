"""Recompute the lookahead truth audit from RSOA dumps: every lookahead box filed at step s for time tau,
with steps s..tau-1 all NN-driven ('empirical'), must contain the true state at tau. Report escapes."""
import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + '/o_*.json')):
    d = json.load(open(f)); S = {s['t']: s for s in d['steps']}
    out = []
    for s in d['steps']:
        for tau, b in (s.get('look') or []):
            if tau not in S or not S[tau].get('real_t'): continue
            if any(S.get(k, {}).get('kind') != 'empirical' for k in range(s['t'], tau)): continue
            x = S[tau]['real_t']
            ex = [max(b[i][0] - x[i], x[i] - b[i][1]) for i in range(3)]
            if max(ex) > 1e-9:
                out.append((s['t'], tau, [round(e, 4) for e in ex], [round(b[i][1] - b[i][0], 3) for i in range(3)]))
    print(f.split('/')[-1], 'escapes', len(out), out[:4])
