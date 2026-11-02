"""
exp28j -- paper figure for the estimate-feedback propagation.

(a) decoupled one-step RSOAs: the NN is bounded over the box first, then the
    control interval is plugged into the dynamics -> the feedback correlation
    is gone and the chain expands.
(b) joint one-step RSOAs over the augmented graph (x, e) |-> f(x, NN(x + e)),
    partitioned into cells -> the chain tracks the true set.
(c) the mechanism, in the theta -> theta' plane at one step: the true image,
    the decoupled box, the single-cell joint bound and the partitioned bound.

All boxes are computed, not drawn by hand. Usage:  python exp28j_figure.py <outdir>
"""
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import exp28_stateunc as E
import exp28b_looseness as B
import exp28e_diagnose as D
import exp28f_split as Fs

K = 6
SPLIT = (2, 2, 16, 1, 1, 4)
CELL_STEP = 2                     # which chain step panel (c) dissects
C_TRUE, C_DEC, C_JNT, C_CELL = '#333333', '#E24B4A', '#1D9E75', '#378ADD'


def rect(b, i=0, j=1, **kw):
    return mp.Rectangle((b[i, 0], b[j, 0]), b[i, 1] - b[i, 0], b[j, 1] - b[j, 0], **kw)


def dec_step(b, eb):
    """Decoupled: CROWN range of the NN over box + error box, then interval step."""
    lo, hi = E.nn_bounds(b + eb)
    return E.istep(b, lo, hi)


def cells_of(box, s):
    edges = [np.linspace(box[i, 0], box[i, 1], s[i] + 1) for i in range(len(s))]
    lo, hi = [], []
    for idx in itertools.product(*[range(n) for n in s]):
        lo.append([edges[i][j] for i, j in enumerate(idx)])
        hi.append([edges[i][j + 1] for i, j in enumerate(idx)])
    return np.array(lo), np.array(hi)


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(11)
    C_t, z_t, vmax = B.start_boxes()['realistic']
    eb = E.ebox(np.full(3, vmax))
    R1 = E.istep(C_t, E.nn_u(z_t)[0])
    traj = D.true_traj(C_t, z_t, vmax, rng)          # list of point clouds per step
    hull = B.hull(traj)

    dec, jnt = [R1.copy()], [R1.copy()]
    for _ in range(K):
        dec.append(dec_step(dec[-1], eb))
        jnt.append(Fs.split_step(jnt[-1], eb, SPLIT))

    fig = plt.figure(figsize=(13.2, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.26)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]

    # ── (a), (b): the two chains in the x-y plane, same axes ────────────────
    for a, ch, ttl in ((ax[0], dec, '(a) decoupled: $\\mathcal{A}^{\\mathrm{dec}}$'),
                       (ax[1], jnt, '(b) joint + partition: $\\mathcal{A}^{J}$')):
        for k in range(K + 1):
            P = traj[k]
            a.scatter(P[:, 0], P[:, 1], s=0.6, color=C_TRUE, alpha=0.25, lw=0,
                      label='true reachable states' if k == 0 else None)
            a.add_patch(rect(hull[k], fill=False, ec=C_TRUE, lw=1.0, ls=':',
                             label='true set (hull)' if k == 0 else None))
        col = C_DEC if ch is dec else C_JNT
        for k in range(K + 1):
            a.add_patch(rect(ch[k], fc=col, ec=col, lw=1.2, alpha=0.16))
            a.add_patch(rect(ch[k], fill=False, ec=col, lw=1.2,
                             label='RSOA chain' if k == 0 else None))
        for k in (0, K):
            a.annotate(f'$k={k}$', (ch[k][0, 1], ch[k][1, 1]), textcoords='offset points',
                       xytext=(2, 2), fontsize=8, color=col)
        a.set_title(ttl, fontsize=11)
        a.set_xlabel('$x$ [m]'); a.set_ylabel('$y$ [m]')
        a.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    lim = [min(ax[0].get_xlim()[0], ax[1].get_xlim()[0]), max(ax[0].get_xlim()[1], ax[1].get_xlim()[1])]
    ylim = [min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]), max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])]
    for a in ax[:2]:
        a.set_xlim(*lim); a.set_ylim(*ylim); a.grid(alpha=0.2)
        a.text(0.98, 0.03, f'$w_\\theta$ at $k={K}$: '
               + f'{(dec if a is ax[0] else jnt)[K][2, 1] - (dec if a is ax[0] else jnt)[K][2, 0]:.2f} rad'
               + f'  (true {hull[K][2, 1] - hull[K][2, 0]:.2f})',
               transform=a.transAxes, ha='right', fontsize=8)

    # ── (c): one step in the theta -> theta' plane ──────────────────────────
    b = jnt[CELL_STEP]
    a = ax[2]
    # The extreme of theta' = theta + dt*pi(x+e) sits at a CORNER of the e box,
    # so e must be sampled bang-bang (as exp28j_partition's 'true' already does).
    # Pairing the two corner lists row-wise, as this did, reaches only 8 of the
    # 64 corner-corner combinations and understates the true width by ~15%.
    X = np.vstack([B.corners(b), rng.uniform(b[:, 0], b[:, 1], size=(20000, 3))])
    Ee = rng.uniform(eb[:, 0], eb[:, 1], size=(len(X), 3))
    _m = rng.integers(0, 2, len(X))[:, None]
    Ee = np.where(_m, np.sign(Ee) * eb[:, 1], Ee)
    th = X[:, 2]
    thn = th + E.DT * E.nn_u(X + Ee)
    a.scatter(th, thn, s=0.8, color=C_TRUE, alpha=0.12, lw=0, label="true image (inner est.)")

    d = dec_step(b, eb)
    a.add_patch(mp.Rectangle((b[2, 0], d[2, 0]), b[2, 1] - b[2, 0], d[2, 1] - d[2, 0],
                             fc=C_DEC, ec=C_DEC, alpha=0.13, lw=1.4,
                             label=f"decoupled box ({d[2, 1] - d[2, 0]:.2f} rad)"))
    j1 = Fs.split_step(b, eb, (1, 1, 1, 1, 1, 1))
    a.add_patch(mp.Rectangle((b[2, 0], j1[2, 0]), b[2, 1] - b[2, 0], j1[2, 1] - j1[2, 0],
                             fill=False, ec=C_JNT, lw=1.6, ls='--',
                             label=f"joint, 1 cell, $e$ unsplit ({j1[2, 1] - j1[2, 0]:.2f} rad)"))
    s = (1, 1, SPLIT[2], 1, 1, SPLIT[5])
    lo, hi = cells_of(np.vstack([b, eb]), s)
    o = D.crown_batch(lo, hi)
    for i in range(0, len(lo), s[5]):                      # group the e_theta cells
        g = o[i:i + s[5]]
        a.add_patch(mp.Rectangle((lo[i, 2], g[:, 2, 0].min()), hi[i, 2] - lo[i, 2],
                                 g[:, 2, 1].max() - g[:, 2, 0].min(),
                                 fc=C_CELL, ec=C_CELL, alpha=0.30, lw=0.7,
                                 label='per-cell joint bounds' if i == 0 else None))
    jh = Fs.split_step(b, eb, s)          # hull of exactly the cells drawn above
    a.plot([b[2, 0], b[2, 1], b[2, 1], b[2, 0], b[2, 0]],
           [jh[2, 0], jh[2, 0], jh[2, 1], jh[2, 1], jh[2, 0]], color=C_CELL, lw=1.6,
           label=f"hull of {s[2] * s[5]} cells ({s[2]}$\\times${s[5]} in $\\theta\\times e_\\theta$): {jh[2, 1] - jh[2, 0]:.2f} rad")
    sl = np.polyfit(th, thn.ravel(), 1)[0]
    a.text(0.97, 0.06, "$\\partial\\theta'/\\partial\\theta \\approx %.2f$: a contraction the\n"
           "decoupled box cannot represent" % sl, transform=a.transAxes, ha='right',
           fontsize=8.5, color=C_TRUE)
    a.set_title("(c) one step, $\\theta \\rightarrow \\theta'$", fontsize=11)
    a.set_xlabel('$\\theta_k$ [rad]'); a.set_ylabel("$\\theta_{k+1}$ [rad]")
    a.grid(alpha=0.2); a.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

    fig.savefig(f'{out}/exp28j_propagation.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{out}/exp28j_propagation.pdf', bbox_inches='tight')
    print('theta widths  k=%d  dec %.3f  joint %.3f  true %.3f'
          % (K, dec[K][2, 1] - dec[K][2, 0], jnt[K][2, 1] - jnt[K][2, 0],
             hull[K][2, 1] - hull[K][2, 0]))
    print('panel c  decoupled %.3f  joint 1 cell (e unsplit) %.3f  hull of %d cells %.3f  true >= %.3f (inner est.)'
          % (d[2, 1] - d[2, 0], j1[2, 1] - j1[2, 0], len(lo), jh[2, 1] - jh[2, 0],
             thn.max() - thn.min()))
    print('wrote', out + '/exp28j_propagation.png')


if __name__ == '__main__':
    main()
