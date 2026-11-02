"""
exp28n -- does the scalar-disturbance symbolic bound (exp28k) become worth its
cost once the lookahead error box is the certified-set recursion (exp28m)?

d_i = NN(x_i + e_i) - NN(x_i) is bounded over (J_i, E_i), so a smaller E_i gives
a narrower d interval and a tighter symbolic unroll -- but it also gives a
tighter concrete chain to beat. Same comparison as exp28k (sym ∩ J256 vs J256),
with both sides on the new rule.

Usage:  python exp28n_symd_ebox.py <dump_J256.json> <outdir>
"""
import sys
import json
import time
import numpy as np
import torch
import exp28_stateunc as E
import exp28j_partition as P
import exp28k_symd as Ks
import exp28l_emodel as L
import exp28m_ebound as M

K = 10


def d_range(Jb, hw):
    eb = np.stack([-hw, hw], 1)
    lo, hi = P.cells(np.vstack([Jb, eb]), Ks.DSPLIT)
    l, u = Ks.crown(Ks.DNN(), ('d', len(lo)), lo, hi)
    return float(l.min()) - E.PAD, float(u.max()) + E.PAD


def main():
    torch.set_num_threads(1)
    dump, outdir = sys.argv[1], sys.argv[2]
    st = {r['t']: r for r in json.load(open(dump))['steps']}
    ks = (1, 3, 5, 8, 10)
    out = {}
    for t0 in (36, 45):
        rng = np.random.default_rng(0)
        S = np.array(st[t0]['bound_t'])
        J, hw, _ = M.chain(S, 'new')
        adv = L.arm_free(S, np.vstack([(S[:, 1] - S[:, 0]) / 2, hw]), rng)
        t_d = time.perf_counter()
        Dr = [d_range(J[i], hw[i]) for i in range(K - 1)]
        t_d = (time.perf_counter() - t_d) / (K - 1)
        ch, tm = [J[0]], []
        for k in range(1, K):
            t1 = time.perf_counter()
            ch.append(Ks.symd(J[0], Dr, k, (2, 2, 16)))
            tm.append(time.perf_counter() - t1)
        both = [np.stack([np.maximum(a[:, 0], b[:, 0]), np.minimum(a[:, 1], b[:, 1])], 1)
                for a, b in zip(ch, J)]
        ok = all(np.all(ch[k][:, 0] <= adv[k][:, 0] + 1e-9) and
                 np.all(ch[k][:, 1] >= adv[k][:, 1] - 1e-9) for k in range(K))
        f = lambda c: '  '.join(f'{L.w(c[k - 1])[0]:.3f}|{L.w(c[k - 1])[1]:.3f}' for k in ks)
        print(f'\nt={t0}  xy m | theta rad at k={ks}   (new e rule everywhere)')
        print(f'  d intervals k=1..5 {[round(b - a, 4) for a, b in Dr[:5]]}  ({t_d * 1e3:.0f} ms each)')
        for n_, c in (('J256 (new rule)', J), ('symD', ch), ('symD ∩ J256', both),
                      ('adversarial true set', adv)):
            print(f'  {n_:22s} {f(c)}')
        print(f'  {"∩ gain over J256":22s} ' + '  '.join(
            f'{100 * (1 - L.w(both[k - 1])[0] / L.w(J[k - 1])[0]):3.0f}%|'
            f'{100 * (1 - L.w(both[k - 1])[1] / L.w(J[k - 1])[1]):3.0f}%' for k in ks))
        print(f'  sec k=5 {tm[3]:.2f}  k=10 {tm[-1]:.2f}   symD contains true {ok}')
        out[t0] = {'J': [list(map(float, L.w(b))) for b in J],
                   'symD': [list(map(float, L.w(b))) for b in ch],
                   'symD_and_J': [list(map(float, L.w(b))) for b in both],
                   'adv': [list(map(float, L.w(b))) for b in adv],
                   'd': Dr, 'sec': tm, 'contains_true': ok}
    with open(f'{outdir}/summary_n.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
