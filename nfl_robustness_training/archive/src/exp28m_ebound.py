"""
exp28m -- tighten the lookahead's error box.

alg28 (joint_step, alg28_sound.py:655) bounds the estimation error at the k-th
lookahead step by the lookahead box itself:

    |e_k| <= min(vmax, hw(X_k))                                        (current)

X_k grows, so by k ~ 5 the assumed error is the raw sensor bound vmax in every
component -- the NN is allowed to read anything inside a 0.3 m / 0.3 rad box.
But the error is bounded by the half-width of the CERTIFIED set S_k, and the
width of S obeys a recursion that does not depend on the (unknown) future
controls, because the control applied at any step is a single scalar:

    hw_th(F(S,u))   = hw_th(S)                       (translation in theta)
    hw_xy(F(S,u))  <= hw_xy(S) + dt*V*hw_th(S)       (|d cos/d th| <= 1)
    S_{k+1} = F(S_k,u_k) ∩ (z ± vmax)   =>   hw(S_{k+1}) <= min(vmax, hw(F(S_k,u_k)))

so, with hw_th frozen at its value at t,

    |e_k| <= min(vmax, hw(X_k), hwS_k),  hwS_k = min(vmax, hwS_{k-1} + dt*V*[h,h,0])

This script builds the J256 chain under both rules from real certified sets and
checks both against (i) the closed loop with actual measurement draws and (ii)
the adversarial set with |e_k| <= hwS_k.

Caveat checked separately: the recursion assumes S is only ever propagated and
intersected. alg28's empty-intersection REPAIR resets S to the measurement box
(half-width vmax) and would break it -- 0 occurrences in 12000 steps of the
200-trial batch, and a guard is cheap (drop the cached lookahead on repair).

Usage:  python exp28m_ebound.py <dump_J256.json> <outdir>
"""
import sys
import json
import time
import numpy as np
import torch
import exp28_stateunc as E
import exp28b_looseness as B
import exp28e_diagnose as D
import exp28j_partition as P
import exp28l_emodel as L

K = 10
VMAX = 0.15
SPLIT = (2, 2, 16, 1, 1, 4)
STARTS = (36, 45)


def chain(S, rule, K=K):
    """J256 chain from S. rule -> e half-widths at each step."""
    X = E.istep(S, E.nn_u(S.mean(1))[0])
    out, hws, t = [X], [], []
    hwS = (S[:, 1] - S[:, 0]) / 2
    for k in range(K - 1):
        b = out[-1]
        hw_box = np.minimum(VMAX, (b[:, 1] - b[:, 0]) / 2)
        hwS = np.minimum(VMAX, hwS + E.DT * E.V * np.array([hwS[2], hwS[2], 0.0]))
        hw = hw_box if rule == 'cur' else np.minimum(hw_box, hwS)
        eb = np.stack([-hw, hw], 1)
        t0 = time.perf_counter()
        out.append(P.hull(D.crown_batch(*P.cells(np.vstack([b, eb]), SPLIT))))
        t.append(time.perf_counter() - t0)
        hws.append(hw.copy())
    return out, np.array(hws), float(np.median(t))


def contains(ch, tru):
    return all(bool(np.all(ch[k][:, 0] <= tru[k][:, 0] + 1e-9) and
                    np.all(ch[k][:, 1] >= tru[k][:, 1] - 1e-9)) for k in range(len(tru)))


def main():
    torch.set_num_threads(1)
    dump, outdir = sys.argv[1], sys.argv[2]
    st = {r['t']: r for r in json.load(open(dump))['steps']}
    ks = (1, 3, 5, 8, 10)
    summary = {}
    for t0 in STARTS:
        rng = np.random.default_rng(0)
        S = np.array(st[t0]['bound_t'])
        cur, hw_cur, ms_c = chain(S, 'cur')
        new, hw_new, ms_n = chain(S, 'new')
        real, ehw, shw = L.arm_real(S, rng)
        adv = L.arm_free(S, np.vstack([(S[:, 1] - S[:, 0]) / 2, hw_new]), rng)
        print(f'\nstart t={t0}   hw(S_t) = {(S[:, 1] - S[:, 0]) / 2}')
        print(f'  {"":26s} ' + '  '.join(f'   k={k}      ' for k in ks))
        for n, ch in (('J256 e<=min(vmax,hw X)', cur), ('J256 e<=hwS  (new)', new),
                      ('adversarial e<=hwS', adv), ('real measurements', real)):
            print(f'  {n:26s} ' + '  '.join(f'{L.w(ch[k - 1])[0]:.3f}|{L.w(ch[k - 1])[1]:.3f}' for k in ks))
        print(f'  {"new / cur":26s} ' + '  '.join(
            f'{L.w(new[k - 1])[0] / L.w(cur[k - 1])[0]:.2f}x|{L.w(new[k - 1])[1] / L.w(cur[k - 1])[1]:.2f}x'
            for k in ks))
        print(f'  {"new / real":26s} ' + '  '.join(
            f'{L.w(new[k - 1])[0] / L.w(real[k - 1])[0]:.2f}x|{L.w(new[k - 1])[1] / L.w(real[k - 1])[1]:.2f}x'
            for k in ks))
        print(f'  e_theta budget  cur ' + ' '.join(f'{hw_cur[k - 2][2]:.4f}' for k in ks[1:])
              + '   new ' + ' '.join(f'{hw_new[k - 2][2]:.4f}' for k in ks[1:]))
        print(f'  contains real: cur {contains(cur, real)} new {contains(new, real)} | '
              f'contains adversarial(hwS): cur {contains(cur, adv)} new {contains(new, adv)} | '
              f'{ms_c * 1e3:.0f} vs {ms_n * 1e3:.0f} ms/step')
        summary[t0] = {'cur': [list(map(float, L.w(b))) for b in cur],
                       'new': [list(map(float, L.w(b))) for b in new],
                       'adv_hwS': [list(map(float, L.w(b))) for b in adv],
                       'real': [list(map(float, L.w(b))) for b in real],
                       'hw_cur': hw_cur.tolist(), 'hw_new': hw_new.tolist(),
                       'contains': {'cur_real': contains(cur, real), 'new_real': contains(new, real),
                                    'cur_adv': contains(cur, adv), 'new_adv': contains(new, adv)},
                       'ms_per_step': [ms_c, ms_n]}
    with open(f'{outdir}/summary_m.json', 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
