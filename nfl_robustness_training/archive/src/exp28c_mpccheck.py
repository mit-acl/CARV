"""
exp28c -- is the backup-plan check accurate? Two-sided test against the EXACT set.

For an open-loop plan u_0..u_{K-1} and no process noise the reachable set is
known in closed form. With start box [x0] x [y0] x [th0],
    th_k = th0 + DT*sum_{i<k} u_i
    x_k  = x0 + DT*V*sum_{j<k} cos(th0 + DT*sum_{i<j} u_i)   (y_k: sin)
x0 and y0 only translate, so the exact box hull is
    [x0_lo + min_th0 g_k, x0_hi + max_th0 g_k]
found by a dense sweep of th0 (plus a Lipschitz margin, so it is exact to
<1e-6). This is NOT a sample of trajectories; it is the true set.

Compared per step:
  interval   chain of istep (what exp28 used; sound by construction)
  const_box  centre rollout +- start half-widths: what the committed PSF
             acceptance test uses (mpc_safety_filter_acados.py:437-445)

Plans: the exp28 hand-picked plan plus 200 random smooth plans in [-1, 1].
Usage:  python exp28c_mpccheck.py <outdir>
"""
import sys
import json
import numpy as np
import exp28_stateunc as E
import exp28b_looseness as B

K = 20
N_TH = 20001


def exact(C, u, margin=True):
    th = np.linspace(C[2, 0], C[2, 1], N_TH)
    cs = np.concatenate([[0.0], np.cumsum(E.DT * u)])          # heading offsets
    TH = th[:, None] + cs[None, :]                              # (N, K+1)
    gx = np.concatenate([np.zeros((N_TH, 1)), np.cumsum(E.DT * E.V * np.cos(TH[:, :-1]), 1)], 1)
    gy = np.concatenate([np.zeros((N_TH, 1)), np.cumsum(E.DT * E.V * np.sin(TH[:, :-1]), 1)], 1)
    # grid spacing d: |g(th)-g(th')| <= k*DT*V*|th-th'|, so the sweep misses <= k*DT*V*d/2
    d = (C[2, 1] - C[2, 0]) / (N_TH - 1)
    m = np.arange(K + 1) * E.DT * E.V * d / 2 * (1.0 if margin else 0.0)
    boxes = []
    for k in range(K + 1):
        boxes.append(np.array([[C[0, 0] + gx[:, k].min() - m[k], C[0, 1] + gx[:, k].max() + m[k]],
                               [C[1, 0] + gy[:, k].min() - m[k], C[1, 1] + gy[:, k].max() + m[k]],
                               [C[2, 0] + cs[k], C[2, 1] + cs[k]]]))
    return boxes, TH, gx, gy


def interval(C, u):
    b = [C.copy()]
    for v in u:
        b.append(E.istep(b[-1], v))
    return b


def const_box(C, u):
    hw = (C[:, 1] - C[:, 0]) / 2
    Z = E.nominal(C.mean(1), u)
    return [np.stack([z - hw, z + hw], 1) for z in Z]


def area(b):
    return float((b[0, 1] - b[0, 0]) * (b[1, 1] - b[1, 0]))


def covers(outer, inner, tol=1e-9):
    return bool(np.all(outer[:2, 0] <= inner[:2, 0] + tol) and np.all(outer[:2, 1] >= inner[:2, 1] - tol))


def plans(rng, n):
    P = [np.array([0.6] * 8 + [0.0] * 12)]
    for _ in range(n):
        knots = rng.uniform(-1, 1, 5)
        P.append(np.interp(np.linspace(0, 4, K), np.arange(5), knots))
    return P


def main():
    out = sys.argv[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rng = np.random.default_rng(7)
    boxes0 = B.start_boxes()
    P = plans(rng, 200)
    summary = {}
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    for row, (name, (C, _, _)) in enumerate(boxes0.items()):
        r_int, r_cb, int_ok, cb_miss_step = [], [], 0, []
        for u in P:
            ex, *_ = exact(C, u)                   # outer by <1e-6: used for ratios
            exi, *_ = exact(C, u, margin=False)    # inner (actual swept states): used for containment
            iv, cb = interval(C, u), const_box(C, u)
            int_ok += all(covers(iv[k], exi[k]) for k in range(K + 1))
            r_int.append([area(iv[k]) / area(ex[k]) for k in range(K + 1)])
            r_cb.append([area(cb[k]) / area(ex[k]) for k in range(K + 1)])
            bad = [k for k in range(1, K + 1) if not covers(cb[k], exi[k])]
            cb_miss_step.append(bad[0] if bad else None)
        r_int, r_cb = np.array(r_int), np.array(r_cb)
        summary[name] = {
            'start_width': (C[:, 1] - C[:, 0]).tolist(),
            'n_plans': len(P),
            'interval_contains_exact_all_steps': f'{int_ok}/{len(P)}',
            'interval_area_ratio_median_at_K': float(np.median(r_int[:, -1])),
            'interval_area_ratio_max_any_step': float(r_int.max()),
            'constbox_misses_exact_plans': int(sum(s is not None for s in cb_miss_step)),
            'constbox_first_miss_step_median': float(np.median([s for s in cb_miss_step if s is not None]))
                                               if any(s is not None for s in cb_miss_step) else None,
            'constbox_area_ratio_median_at_K': float(np.median(r_cb[:, -1])),
        }
        # example plan (the exp28 one)
        u = P[0]
        ex, TH, gx, gy = exact(C, u)
        iv, cb = interval(C, u), const_box(C, u)
        a = ax[row, 0]
        for k in range(1, K + 1):
            # true set at step k = union over th0 of (start xy box shifted by (gx, gy))
            for j in np.linspace(0, N_TH - 1, 60).astype(int):
                a.add_patch(Rectangle((C[0, 0] + gx[j, k], C[1, 0] + gy[j, k]),
                                      C[0, 1] - C[0, 0], C[1, 1] - C[1, 0],
                                      color='#D85A30', alpha=0.15, lw=0))
            for b, c, ls in ((iv[k], '#1D9E75', '-'), (cb[k], '#534AB7', '--')):
                a.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                      fill=False, color=c, ls=ls, lw=0.8))
        Z = E.nominal(C.mean(1), u)
        a.plot(Z[:, 0], Z[:, 1], 'k.-', ms=3, lw=0.6)
        a.set_aspect('equal')
        a.set_title(f'{name}: exact true set (orange) vs interval check (green)\n'
                    f'vs PSF constant box (purple dashed); start θ width {C[2,1]-C[2,0]:.3f} rad')
        a = ax[row, 1]
        steps = np.arange(K + 1)
        a.plot(steps, [ex[k][0, 1] - ex[k][0, 0] for k in steps], 'k-', lw=2, label='exact x width')
        a.plot(steps, [iv[k][0, 1] - iv[k][0, 0] for k in steps], color='#1D9E75', label='interval x width')
        a.plot(steps, [cb[k][0, 1] - cb[k][0, 0] for k in steps], color='#534AB7', ls='--', label='PSF const box x width')
        a.plot(steps, [ex[k][1, 1] - ex[k][1, 0] for k in steps], 'k:', lw=2, label='exact y width')
        a.plot(steps, [iv[k][1, 1] - iv[k][1, 0] for k in steps], color='#1D9E75', ls=':', label='interval y width')
        a.set_xlabel('plan step'); a.set_ylabel('m'); a.legend(fontsize=8)
        a.set_title(f'{name}: example plan, widths per step')
        a = ax[row, 2]
        for q, c in ((r_int, '#1D9E75'), (r_cb, '#534AB7')):
            a.fill_between(steps, q.min(0), q.max(0), color=c, alpha=0.25)
            a.plot(steps, np.median(q, 0), color=c)
        a.axhline(1, color='k', lw=1)
        a.set_yscale('log'); a.set_xlabel('plan step'); a.set_ylabel('box area / exact box area')
        a.set_title(f'{name}: {len(P)} plans (median, min-max band)\n'
                    f'green interval: >=1 always = sound; purple const box: <1 = MISSES true states')
    fig.suptitle('exp28c: backup-plan check vs the EXACT reachable set (closed form, not sampled)')
    fig.tight_layout(); fig.savefig(f'{out}/exp28c_mpccheck.png', dpi=110)
    with open(f'{out}/summary_c.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
