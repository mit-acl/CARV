"""
exp28f -- follow-up to exp28e: how close does joint (x, e) CROWN with input
splitting get to the best possible box-per-step bound (C_oracle), and at what
cost per lookahead step?

SPLITS are cell counts over (x, y, theta, e_x, e_y, e_theta).
Usage:  python exp28f_split.py <outdir>
"""
import sys
import json
import itertools
import time
import numpy as np
import exp28_stateunc as E
import exp28b_looseness as B
import exp28e_diagnose as D

K = 20
SPLITS = [(1, 1, 1, 1, 1, 1), (1, 1, 8, 1, 1, 1), (2, 2, 8, 1, 1, 1), (1, 1, 16, 1, 1, 1),
          (1, 1, 8, 1, 1, 4), (2, 2, 8, 1, 1, 4), (2, 2, 16, 1, 1, 4), (2, 2, 16, 2, 2, 4),
          (4, 4, 16, 1, 1, 4)]


def split_step(b, eb, s):
    box = np.vstack([b, eb])
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(6)]
    lo, hi = [], []
    for idx in itertools.product(*[range(n) for n in s]):
        lo.append([edges[i][j] for i, j in enumerate(idx)])
        hi.append([edges[i][j + 1] for i, j in enumerate(idx)])
    out = D.crown_batch(np.array(lo), np.array(hi))
    return np.stack([out[:, :, 0].min(0), out[:, :, 1].max(0)], 1)


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(11)
    summary = {}
    for name, (C_t, z_t, vmax) in B.start_boxes().items():
        eb = E.ebox(np.full(3, vmax))
        R1 = E.istep(C_t, E.nn_u(z_t)[0])
        h = B.hull(D.true_traj(C_t, z_t, vmax, rng))
        b = R1.copy(); orc = [b]
        for _ in range(K - 1):
            b = D.image_exact(b, eb, rng); orc.append(b)
        S = summary[name] = {'true': [D.w(x) for x in h], 'C_oracle': [D.w(x) for x in orc]}
        for s in SPLITS:
            b = R1.copy(); ch = [b]; t = []
            split_step(b, eb, s)                       # build the bounded module once
            for _ in range(K - 1):
                t0 = time.perf_counter()
                b = split_step(b, eb, s); ch.append(b)
                t.append(time.perf_counter() - t0)
            ok = all(bool(np.all(ch[k][:, 0] <= h[k][:, 0] + 1e-6) and np.all(ch[k][:, 1] >= h[k][:, 1] - 1e-6))
                     for k in range(K))
            S[str(s)] = {'width': [D.w(x) for x in ch], 'sec_per_step_median': float(np.median(t)),
                         'cells': int(np.prod(s)), 'contains_true': ok}
            xy = lambda x: max(x[0], x[1])
            r = S[str(s)]['width']
            print(f'{name:9s} {str(s):22s} cells {np.prod(s):5d} {np.median(t) * 1e3:7.1f} ms/step  '
                  + '  '.join(f'k{k + 1}: th {r[k][2]:.3f} ({r[k][2] / S["C_oracle"][k][2]:.2f}x orc) '
                              f'xy {xy(r[k]):.3f} ({xy(r[k]) / xy(S["C_oracle"][k]):.2f}x)'
                              for k in (4, 9, 19)) + f'  sound {ok}', flush=True)
        o, tr = S['C_oracle'], S['true']
        print(f'{name:9s} oracle / true: ' + '  '.join(
            f'k{k + 1}: th {o[k][2]:.3f}/{tr[k][2]:.3f} xy {max(o[k][:2]):.3f}/{max(tr[k][:2]):.3f}'
            for k in (4, 9, 19)), flush=True)
    with open(f'{out}/summary_f.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
