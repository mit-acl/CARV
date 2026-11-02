"""Classify every overridden step (|u_avg - u_ref| > 0.05) by the solve that produced the flown command.

R1 (delta=0): the solve started at step k lands at k -> reason is in events[k].
R2 (delta=0.1, ALIGN -> one tick): the solve started at k-1 lands at k -> reason is in events[k-1].
"""
import glob, json, os, sys
from collections import Counter

D = os.path.dirname(os.path.abspath(__file__))
THR = 0.05

for arm, lag in (('r1', 0), ('r2', 1)):
    files = sorted(glob.glob(f'{D}/{arm}_*.json'))
    steps = over = 0
    by = Counter()
    ex = {}
    for f in files:
        tr = json.load(open(f))['trace']
        for i, s in enumerate(tr):
            steps += 1
            j = i - lag
            solve = [e[1] for e in tr[j]['events'] if e[0] == 'solve'] if j >= 0 else []
            reason = solve[0] if solve else 'no_solve'
            key = (s['kind'], reason if reason.startswith('VALID') else 'FAILED:' + reason)
            if abs(s['u_avg'] - s['u_ref']) > THR:
                over += 1
                by[key] += 1
                ex.setdefault(key, (os.path.basename(f), s['k'], round(s['u_nn'], 3),
                                    round(s['u_ref'], 3), round(s['u_avg'], 3)))
            by[('_all', key)] += 1
    print(f'== {arm}: {len(files)} trials, {steps} steps, overridden {over} ({100 * over / max(1, steps):.2f}%)')
    for k, n in sorted(((k, n) for k, n in by.items() if k[0] != '_all'), key=lambda x: -x[1]):
        tot = by[('_all', k)]
        print(f'   {str(k):55s} over {n:4d} of {tot:5d} steps   e.g. {ex[k]}  (file,k,u_nn,u_ref,u_avg)')
