"""
exp28j -- can the lookahead go below the one-box-per-step floor?

Start: certified sets from a real alg28 J256 run (dump_J256.json, seed
1443680074), NN at the box centre for the first (known) step, then K steps of
u = NN(x + e), |e_i| <= min(vmax, half-width of the current hull).

  J256 / J2048   one box per step, joint (x, e) CROWN over split cells
  floor          one box per step, EXACT image by dense sampling (inner
                 estimate, not sound): the best any box-per-step chain can do
  part-M         the chain is a UNION of M boxes, never merged into one box.
                 Every part is split into cells, bounded by joint CROWN, and the
                 cell outputs are re-grouped into M boxes (k-d split on cell
                 centres). Sound: every output box is the hull of sound cells.
  true           sampled, with e bounded by the tightest hull's half-width

Usage:  python exp28j_partition.py <dump_J256.json> <outdir>
"""
import sys
import json
import itertools
import time
import numpy as np
import exp28_stateunc as E
import exp28e_diagnose as D
import exp28b_looseness as B

K = 10
VMAX = 0.15
STARTS = [36, 45]


def cells(box, s):
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(len(s))]
    idx = list(itertools.product(*[range(n) for n in s]))
    lo = np.array([[edges[i][j] for i, j in enumerate(c)] for c in idx])
    hi = np.array([[edges[i][j + 1] for i, j in enumerate(c)] for c in idx])
    return lo, hi


def hull(bs):
    bs = np.asarray(bs)
    return np.stack([bs[:, :, 0].min(0), bs[:, :, 1].max(0)], 1)


def ebox(b):
    hw = np.minimum(VMAX, (b[:, 1] - b[:, 0]) / 2)
    return np.stack([-hw, hw], 1)


def joint(parts, s):
    """CROWN image of every part, split into cells s. Returns cell output boxes (n,3,2)."""
    h = hull(parts)
    eb = ebox(h)
    LO, HI = [], []
    for p in parts:
        lo, hi = cells(np.vstack([p, eb]), s)
        LO.append(lo); HI.append(hi)
    return D.crown_batch(np.vstack(LO), np.vstack(HI))


def regroup(out, M):
    """k-d split of cell outputs into M groups by their centres; hull of each group."""
    c = out.mean(2)
    scale = np.maximum(np.ptp(c, 0), 1e-9)
    groups = [np.arange(len(out))]
    while len(groups) < M:
        groups.sort(key=len)
        g = groups.pop()
        if len(g) < 2:
            groups.append(g); break
        d = int(np.argmax(np.ptp(c[g], 0) / scale))
        o = g[np.argsort(c[g, d])]
        groups += [o[:len(o) // 2], o[len(o) // 2:]]
    return [hull(out[g]) for g in groups]


def w(b):
    return max(b[0, 1] - b[0, 0], b[1, 1] - b[1, 0]), b[2, 1] - b[2, 0]


def main():
    dump, outdir = sys.argv[1], sys.argv[2]
    st = {r['t']: r for r in json.load(open(dump))['steps']}
    rng = np.random.default_rng(0)
    methods = {'J256': ('box', (2, 2, 16, 1, 1, 4)), 'J2048': ('box', (4, 4, 32, 1, 1, 4)),
               'part-16': (16, (1, 1, 4, 1, 1, 2)), 'part-64': (64, (1, 1, 4, 1, 1, 2)),
               'part-256': (256, (1, 1, 2, 1, 1, 2))}
    summary = {}
    for t0 in STARTS:
        S = np.array(st[t0]['bound_t'])
        X1 = E.istep(S, E.nn_u(S.mean(1))[0])
        res, tm = {}, {}
        for name, (mode, s) in methods.items():
            D.crown_batch(*cells(np.vstack([X1, ebox(X1)]), s))      # warm the module
            parts, chain, tt = [X1], [X1], []
            for _ in range(K - 1):
                w0 = time.perf_counter()
                out = joint(parts, s)
                parts = [hull(out)] if mode == 'box' else regroup(out, mode)
                tt.append(time.perf_counter() - w0)
                chain.append(hull(parts))
            res[name], tm[name] = chain, tt
        b, fl = X1, [X1]
        for _ in range(K - 1):
            b = D.image_exact(b, ebox(b), rng); fl.append(b)
        res['floor'] = fl
        # true: e bounded by the tightest SOUND hull at each step (all contain S_k)
        sound = [n for n in methods]
        X = np.vstack([B.corners(S), rng.uniform(S[:, 0], S[:, 1], size=(500, 3))])
        X = np.repeat(X, 120, 0)
        X = E.f_np(X, np.full(len(X), E.nn_u(S.mean(1))[0])); tr = [X]
        for k in range(1, K):
            hw = np.min([np.minimum(VMAX, (res[n][k - 1][:, 1] - res[n][k - 1][:, 0]) / 2) for n in sound], 0)
            m = rng.integers(0, 2, len(X))[:, None]
            e = rng.uniform(-1, 1, (len(X), 3)); e = np.where(m, np.sign(e), e) * hw
            X = E.f_np(X, E.nn_u(X + e)); tr.append(X)
        res['true'] = [np.stack([x.min(0), x.max(0)], 1) for x in tr]
        print(f'\nstart t={t0}: certified set xy {w(S)[0]:.3f} m, θ {w(S)[1]:.3f} rad'
              f'   (x/y width in m, then ratio to true, at steps ahead 1 3 5 8 10)')
        S_ = summary[t0] = {}
        for name in ['true', 'floor'] + list(methods):
            ch = res[name]
            ok = '' if name in ('true', 'floor') else (
                '  sound ' + str(all(np.all(ch[k][:, 0] <= res['true'][k][:, 0] + 1e-6) and
                                     np.all(ch[k][:, 1] >= res['true'][k][:, 1] - 1e-6) for k in range(K))))
            cost = f'  {np.median(tm[name]) * 1e3:6.0f} ms/step' if name in tm else ''
            print(f'  {name:9s} ' + '  '.join(f'{w(ch[k - 1])[0]:6.3f} ({w(ch[k - 1])[0] / w(res["true"][k - 1])[0]:4.2f}x)'
                                             for k in (1, 3, 5, 8, 10)) + cost + ok, flush=True)
            S_[name] = {'width': [list(map(float, w(b))) for b in ch],
                        'ms_per_step': float(np.median(tm[name]) * 1e3) if name in tm else None}
    with open(f'{outdir}/summary_j.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
