"""
Figures + results for alg14_di.py (double-integrator split-terminal PSF).

Produces, into nfl_robustness_training/results_di/ :
    di_phase_portrait.png   — (p, v) phase plane: invariant-set boundary,
                              RSOA boxes, nominal vs PSF trajectory
    di_timeseries.png       — v(t) with the RSOA lower bound, and the
                              applied vs nominal control
    di_results.md / .json   — filtered vs unfiltered violation counts

Run from the project root:
    python nfl_robustness_training/src/alg14_di_figs.py [n_seeds]
"""

import os
import sys
import json
import pickle
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from REAL_integrated_sim import setup_analyzer
import alg14_di as A


# Figure style. One place to change it, and the timeseries figure inherits it.
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'mathtext.fontset':  'dejavuserif',
    'font.size':         9.0,
    'axes.labelsize':    10,
    'axes.linewidth':    0.8,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8,
    'savefig.bbox':      'standard',
})

C_PSF  = '#1f5fa8'   # REACH-PSF
C_NOM  = '#e07b1c'   # unfiltered baseline
C_TERM = '#7a4fa3'   # terminal invariant set
C_VIOL = '#b3202c'   # constraint / violation

# Zoom-inset geometry — the knobs to turn when the figure needs re-fitting.
#
#   INSET_RECT  [x0, y0, w, h] in axes fractions: where the magnified panel
#               sits inside the main axes, and how BIG it is. (0, 0) is the
#               lower-left corner of the main axes, (1, 1) the upper-right.
#   INSET_PAD_P p-units of margin left and right of the timesteps the zoom is
#               built around.
#   INSET_DV    v window, as an offset from vel_min: (bottom, top).
#
# The two interact: magnification is INSET_RECT's size divided by the window
# INSET_PAD_P/INSET_DV carve out, so widening the window at a fixed rect zooms
# out, and enlarging the rect at a fixed window zooms in.
INSET_RECT  = (0.300, 0.575, 0.424, 0.300)
INSET_PAD_P = 0.10
INSET_DV    = (-0.13, 0.26)
INSET_EDGE  = '0.40'

OUT = os.path.join('nfl_robustness_training', 'results_di')
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'run.log')


@contextlib.contextmanager
def quiet(logfile):
    """
    fd-level redirect. acados prints QP status from C, which bypasses
    sys.stdout, so Python-level redirection is not enough.
    """
    fd_out, fd_err = os.dup(1), os.dup(2)
    f = open(logfile, 'a')
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        os.dup2(fd_out, 1); os.dup2(fd_err, 2)
        os.close(fd_out);   os.close(fd_err)
        f.close()


def invariant_boundary(v_grid, dt, u_max, pos_min, buffer):
    """
    Minimum position p0 such that braking at u=+u_max from (p0, v0) keeps
    p >= pos_min + buffer for the whole braking phase (discrete-exact).
    Because v < 0 throughout braking, p is monotonically decreasing, so the
    minimum is the end-of-braking position and the boundary is
        p0 = pos_min + buffer - delta(v0)
    where delta(v0) is the (negative) position change during braking.
    """
    out = []
    for v0 in v_grid:
        p, v = 0.0, float(v0)
        while v < 0.0:
            p = p + dt * v + 0.5 * dt * dt * u_max
            v = v + dt * u_max
        out.append(pos_min + buffer - p)
    return np.array(out)


def parabolic_boundary(v_grid, u_max, pos_min, buffer):
    """
    Boundary of the parabolic max-brake-invariant set — the set the terminal
    safety argument in di_mpc_acados actually uses:

        p - 0.5 * max(0, -v)**2 / u_max  >=  pos_min + buffer

    so the boundary is p(v) = pos_min + buffer + 0.5*max(0,-v)**2/u_max.

    invariant_boundary() above is the discrete-exact version. It is genuinely
    discontinuous: braking takes an integer number of steps, so at each
    v0 = -k*dt*u_max the step count jumps and the boundary jumps with it by
    0.5*dt^2*u_max = 0.02 here. That staircase is correct but unreadable as a
    curve, and it lies at or left of this parabola at every v (checked over
    the plotted range), i.e. the parabola is the conservative inner
    approximation. Plot the parabola; it is what the safety argument states.
    """
    vneg = np.maximum(0.0, -np.asarray(v_grid, dtype=float))
    return pos_min + buffer + 0.5 * vneg ** 2 / u_max


def _boxes(ax, bounds_list, color, alpha, label=None, lw=0.8, ls='-'):
    first = True
    for b in bounds_list:
        if b is None:
            continue
        b = np.asarray(b)
        ax.add_patch(Rectangle(
            (b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
            fill=False, edgecolor=color, alpha=alpha, linewidth=lw, linestyle=ls,
            label=label if first else None))
        first = False


def _draw_layers(a, psf, nom, act, box_lw, traj_lw, ms, label=False):
    """Every layer of the phase portrait, drawn onto one axes.

    Called twice with identical content — main axes and zoom inset — so the two
    can never drift apart. Only the main axes contributes legend handles.
    """
    pos_min, vel_min = psf['pos_min'], psf['vel_min']
    L = (lambda s: s) if label else (lambda s: None)

    ns, ps = np.array(nom['state_history']), np.array(psf['state_history'])

    # Span well beyond both axes' y-limits so the boundary runs off the top and
    # bottom edges instead of stopping in mid-air.
    vg = np.linspace(vel_min - 1.2, 1.5, 1200)
    pb = parabolic_boundary(vg, psf['u_max'], pos_min, psf['buffer'])
    a.fill_betweenx(vg, pos_min - 1.0, pb, color=C_TERM, alpha=0.09,
                    lw=0, zorder=0)
    a.plot(pb, vg, color=C_TERM, lw=1.6, zorder=5,
           label=L('terminal invariant set'))
    a.axvline(pos_min, color='0.15', lw=1.4, zorder=5, label=L(r'$p \geq p_{\min}$'))
    a.axhspan(vel_min - 10.0, vel_min, color=C_VIOL, alpha=0.055, lw=0, zorder=0)
    a.axhline(vel_min, color=C_VIOL, lw=1.5, ls=(0, (5, 2.5)), zorder=5,
              label=L(r'$v \geq v_{\min}$'))

    _boxes(a, nom['bound_history'], C_NOM, 0.50, L('RSOA — unfiltered'),
           box_lw, (0, (3, 2)))
    _boxes(a, psf['bound_history'], C_PSF, 0.50, L('RSOA — REACH-PSF'),
           box_lw, '-')

    a.plot(ns[:, 0], ns[:, 1], '-', color=C_NOM, lw=traj_lw, zorder=7)
    a.plot(ns[:, 0], ns[:, 1], 'o', color=C_NOM, ms=ms, zorder=7,
           label=L('state — unfiltered'))
    a.plot(ps[:, 0], ps[:, 1], '-', color=C_PSF, lw=traj_lw, zorder=8)
    a.plot(ps[:, 0], ps[:, 1], 'o', color=C_PSF, ms=ms, zorder=8,
           label=L('state — REACH-PSF'))

    if act:
        a.plot(ps[act, 0], ps[act, 1], 'o', color=C_PSF, ms=ms * 3.0,
               mfc='none', mew=1.6, zorder=9, label=L('PSF intervenes'))

    # x at the lower-left corner of every unfiltered RSOA box that dips below a
    # half-plane. NOT "certified violation": the unfiltered run certifies
    # nothing, and because the box is an over-approximation, intersecting the
    # unsafe set does not imply the state is unsafe. What it does mean is that
    # safety cannot be certified at that step — which is the honest comparison,
    # since keeping this box out of the unsafe set is precisely the PSF's job.
    first = True
    for v_ in nom['rsoa_viol']:
        b = nom['bound_history'][v_]
        if b is None:
            continue
        a.plot([b[0, 0]], [b[1, 0]], 'x', color=C_VIOL, ms=ms * 2.2, mew=1.6,
               alpha=0.85, zorder=10,
               label=L('RSOA enters unsafe set') if first else None)
        first = False

    # The headline: timesteps where the unfiltered REAL state is inside the
    # unsafe half-plane. Filled, on top, in the constraint colour — this is the
    # thing the filter prevents, and it should read before the boxes do.
    rv = nom['real_viol']
    if rv:
        a.plot(ns[rv, 0], ns[rv, 1], 'o', color=C_VIOL, ms=ms * 1.7,
               mec='white', mew=0.7, zorder=11, label=L('real-state violation'))


def phase_portrait(psf, nom, path):
    """Single-panel phase portrait with the v_min crossing as an inset.

    The trajectory sweeps from the top right down to the origin, which leaves
    the upper-middle of the plane empty; the zoom goes there rather than into a
    second panel, so the figure stays one column wide and the reader keeps the
    global geometry in view while reading the detail.
    """
    pos_min, vel_min = psf['pos_min'], psf['vel_min']
    ns, ps = np.array(nom['state_history']), np.array(psf['state_history'])
    modes = {r['t']: r['mode'] for r in psf['records']}
    act = [i for i in range(len(ps)) if modes.get(i) in ('MPC', 'ACTIVATE')]

    fig, ax = plt.subplots(figsize=(7.1, 4.5))

    _draw_layers(ax, psf, nom, act, box_lw=0.6, traj_lw=1.3, ms=3.0, label=True)

    ax.set_xlim(pos_min - 0.22,
                max(float(ns[:, 0].max()), float(ps[:, 0].max())) + 0.20)
    ax.set_ylim(vel_min - 0.20,
                max(float(ps[:, 1].max()), float(ns[:, 1].max())) + 0.30)
    ax.set_xlabel('position $p$')
    ax.set_ylabel('velocity $v$')
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ── inset: the v_min crossing ────────────────────────────────────────
    # Window covers every uncertifiable step of the unfiltered run plus the
    # PSF's response to it, with enough margin to show both RSOA stacks.
    viol = nom['rsoa_viol'] or [int(np.argmin([b[1, 0] for b in nom['bound_history']
                                               if b is not None]))]
    span = sorted(set(viol) | set(act))
    span = list(range(max(0, min(span) - 1), min(len(ns), max(span) + 2)))
    px = np.concatenate([ns[span, 0], ps[span, 0]])
    z_x0, z_x1 = float(px.min()) - INSET_PAD_P, float(px.max()) + INSET_PAD_P
    z_y0, z_y1 = vel_min + INSET_DV[0], vel_min + INSET_DV[1]

    # Upper-middle of the plane: bounded left by the trajectory as it climbs
    # back toward the origin, right by the RSOA stack around the initial set.
    axins = ax.inset_axes(list(INSET_RECT))
    _draw_layers(axins, psf, nom, act, box_lw=0.7, traj_lw=1.5, ms=3.6)
    axins.set_xlim(z_x0, z_x1)
    axins.set_ylim(z_y0, z_y1)
    axins.set_facecolor('white')
    axins.patch.set_alpha(0.96)
    axins.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    axins.grid(alpha=0.18, lw=0.5)
    axins.set_axisbelow(True)
    for s in axins.spines.values():
        s.set_edgecolor(INSET_EDGE)
        s.set_linewidth(0.9)
    axins.set_title(r'zoom: the $v_{\min}$ crossing', fontsize=8.0, pad=3.0)

    # Indicator rectangle, then leaders drawn by hand. matplotlib's own
    # connectors join corresponding corners (rect lower-left to inset
    # lower-left, and so on), which here means two lines rising out of the
    # bottom of the rectangle and straight back through it. Joining the
    # rectangle's TOP corners to the inset's BOTTOM corners instead gives the
    # usual splayed magnifier cone and leaves the region itself clear. They
    # still cross the trajectory on the way up — unavoidable with the inset
    # above the region — so they are thin, dashed and pale enough to read as
    # guides rather than as data.
    ind = ax.indicate_inset_zoom(axins, edgecolor=INSET_EDGE, lw=0.9,
                                 alpha=0.85)
    for ln in ind.connectors:
        ln.set_visible(False)

    for (rx, ry), corner in (((z_x0, z_y1), (0.0, 0.0)),
                             ((z_x1, z_y1), (1.0, 0.0))):
        ax.add_artist(ConnectionPatch(
            xyA=(rx, ry), coordsA=ax.transData,
            xyB=corner, coordsB=axins.transAxes,
            color=INSET_EDGE, lw=0.8, ls=(0, (4, 2.5)), alpha=0.8, zorder=11))

    # ── legend and title, both outside the axes ──────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.925),
               ncol=4, fontsize=8.0, frameon=False, handlelength=1.9,
               columnspacing=1.5, handletextpad=0.6, borderaxespad=0.0)

    fig.suptitle(
        'Double-integrator split-terminal REACH-PSF   '
        rf'($p_{{\min}}={pos_min:g}$,  $v_{{\min}}={vel_min:g}$,  '
        rf'seed {psf["seed"]})', fontsize=10.5, y=0.985)

    fig.subplots_adjust(left=0.082, right=0.988, top=0.780, bottom=0.115)
    fig.savefig(path, dpi=400)
    fig.savefig(os.path.splitext(path)[0] + '.pdf')
    plt.close(fig)


def timeseries(psf, nom, path):
    vel_min = psf['vel_min']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)

    for res, c, name in ((nom, C_NOM, 'unfiltered'),
                         (psf, C_PSF, 'REACH-PSF')):
        s = np.array(res['state_history'])
        t = np.arange(len(s))
        lo = np.array([b[1, 0] if b is not None else np.nan
                       for b in res['bound_history']])
        hi = np.array([b[1, 1] if b is not None else np.nan
                       for b in res['bound_history']])
        a1.fill_between(t, lo, hi, color=c, alpha=0.18)
        a1.plot(t, lo, color=c, lw=1.0, ls=':')
        a1.plot(t, s[:, 1], color=c, lw=1.8, label=f'$v$ — {name}')

    a1.axhline(vel_min, color=C_VIOL, lw=1.8, ls=(0, (5, 2.5)), label=r'$v_{\min}$')
    for v_ in nom['rsoa_viol']:
        a1.plot([v_], [nom['bound_history'][v_][1, 0]], 'x', color=C_VIOL,
                ms=11, mew=2.6, zorder=9)
    a1.set_ylabel('velocity'); a1.grid(alpha=0.25)
    a1.legend(fontsize=8.0, ncol=3, loc='lower right', framealpha=0.95)
    a1.set_title("Certified RSOA velocity band vs constraint "
                 r"($\times$ = unfiltered violation)", fontsize=10.5)

    act = sorted(r['t'] for r in psf['records'] if r['mode'] in ('MPC', 'ACTIVATE'))
    for a in (a1, a2):
        first = True
        for t_ in act:
            a.axvspan(t_ - 0.5, t_ + 0.5, color=C_PSF, alpha=0.10, lw=0,
                      label='REACH-PSF active' if first else None)
            first = False

    ts = [r['t'] for r in psf['records'] if r['u'] is not None]
    ua = [r['u'] for r in psf['records'] if r['u'] is not None]
    un = [r['u_nn'] for r in psf['records'] if r['u'] is not None]
    a2.step(np.arange(len(psf['u_diffs'])), psf['u_diffs'], where='mid',
            color='tab:green', lw=1.6, label='REACH-PSF vs NN: $|u_{\mathrm{PSF}}-u_{NN}|$')
    if ts:
        a2.plot(ts, ua, 'o', color='tab:red', ms=6, label='$u$ applied (REACH-PSF)')
        a2.plot(ts, un, 'o', color='k', ms=4, mfc='none', label='$u$ nominal')
    a2.set_xlabel('timestep'); a2.set_ylabel('control')
    a2.grid(alpha=0.25); a2.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    n_seeds  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    VEL_MIN  = -0.90     # tight enough that the nominal RSOA violates
    POS_MIN  = 0.0
    MAX_TIME = 40

    rng   = np.random.default_rng(0)
    seeds = [1401830092] + [int(s) for s in rng.integers(0, 2**31 - 1, n_seeds - 1)]

    open(LOG, 'w').close()
    rows, fig_pair = [], None

    for i, sd in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] seed={sd} ...", flush=True)
        with quiet(LOG):
            # Fresh analyzer per trial — a shared analyzer accumulates state
            # across trials and changes the trajectory.
            az_p = setup_analyzer('DoubleIntegrator',
                                  'constraint_default_more_data_5hz')
            psf = A.test(seed=sd, analyzer=az_p, pos_min=POS_MIN,
                         vel_min=VEL_MIN, MAX_TIME=MAX_TIME, calibrate=True)
            az_n = setup_analyzer('DoubleIntegrator',
                                  'constraint_default_more_data_5hz')
            nom = A.run_nominal(seed=sd, analyzer=az_n, pos_min=POS_MIN,
                                vel_min=VEL_MIN, MAX_TIME=MAX_TIME)
        rows.append({
            'seed': sd,
            'psf_real': len(psf['real_viol']), 'psf_rsoa': len(psf['rsoa_viol']),
            'nom_real': len(nom['real_viol']), 'nom_rsoa': len(nom['rsoa_viol']),
            'mpc_calls': psf['mpc_calls'], 'no_backup': psf['psf_no_diverge'],
            'queue_empty': psf['psf_queue_empty'], 'aborted': psf['aborted'],
            'steps': psf['steps'], 'max_u_diff': float(max(psf['u_diffs'], default=0)),
        })
        print(f"    PSF real={rows[-1]['psf_real']} rsoa={rows[-1]['psf_rsoa']}"
              f" | NOM real={rows[-1]['nom_real']} rsoa={rows[-1]['nom_rsoa']}"
              f" | mpc_calls={rows[-1]['mpc_calls']}", flush=True)
        if fig_pair is None and nom['rsoa_viol']:
            fig_pair = (psf, nom)          # prefer a seed the baseline violates
    if fig_pair is None:
        fig_pair = (psf, nom)

    with open(os.path.join(OUT, 'di_figdata.pkl'), 'wb') as fh:
        pickle.dump(fig_pair, fh)
    phase_portrait(*fig_pair, os.path.join(OUT, 'di_phase_portrait.png'))
    timeseries(*fig_pair,     os.path.join(OUT, 'di_timeseries.png'))

    agg = {
        'n_seeds': len(rows), 'pos_min': POS_MIN, 'vel_min': VEL_MIN,
        'max_time': MAX_TIME,
        'psf_seeds_with_real_viol': sum(r['psf_real'] > 0 for r in rows),
        'psf_seeds_with_rsoa_viol': sum(r['psf_rsoa'] > 0 for r in rows),
        'nom_seeds_with_real_viol': sum(r['nom_real'] > 0 for r in rows),
        'nom_seeds_with_rsoa_viol': sum(r['nom_rsoa'] > 0 for r in rows),
        'psf_aborted': sum(bool(r['aborted']) for r in rows),
        'psf_no_backup_total': sum(r['no_backup'] for r in rows),
        'psf_queue_empty_total': sum(r['queue_empty'] for r in rows),
    }
    json.dump({'agg': agg, 'rows': rows},
              open(os.path.join(OUT, 'di_results.json'), 'w'), indent=2)

    with open(os.path.join(OUT, 'di_results.md'), 'w') as f:
        f.write(f"# alg14_di — split-terminal PSF on the double integrator\n\n")
        f.write(f"`p >= {POS_MIN}`, `v >= {VEL_MIN}`, horizon {MAX_TIME} steps, "
                f"{len(rows)} seeds\n\n")
        f.write("| seed | PSF real | PSF RSOA | nominal real | nominal RSOA "
                "| mpc_calls | no_backup | queue_empty | aborted | max &#124;du&#124; |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['seed']} | {r['psf_real']} | {r['psf_rsoa']} "
                    f"| {r['nom_real']} | {r['nom_rsoa']} | {r['mpc_calls']} "
                    f"| {r['no_backup']} | {r['queue_empty']} "
                    f"| {'yes' if r['aborted'] else 'no'} | {r['max_u_diff']:.3f} |\n")
        f.write("\n## Aggregate\n\n")
        for k, v in agg.items():
            f.write(f"- `{k}` = {v}\n")

    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k:28s} {v}")
    print(f"\nwrote {OUT}/di_phase_portrait.png, di_timeseries.png, "
          f"di_results.md, di_results.json")


def replot():
    """Redraw both figures from di_figdata.pkl without re-running any sims."""
    with open(os.path.join(OUT, 'di_figdata.pkl'), 'rb') as fh:
        psf, nom = pickle.load(fh)
    phase_portrait(psf, nom, os.path.join(OUT, 'di_phase_portrait.png'))
    timeseries(psf, nom, os.path.join(OUT, 'di_timeseries.png'))
    print('replotted from di_figdata.pkl')


def figrun(seed):
    """Run one seed (PSF + unfiltered), pickle it, and draw both figures.
    Does not touch di_results.md / di_results.json."""
    open(LOG, 'w').close()
    with quiet(LOG):
        az_p = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
        psf = A.test(seed=seed, analyzer=az_p, pos_min=0.0, vel_min=-0.90,
                     MAX_TIME=40, calibrate=True)
        az_n = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
        nom = A.run_nominal(seed=seed, analyzer=az_n, pos_min=0.0, vel_min=-0.90,
                            MAX_TIME=40)
    print(f"seed={seed}  PSF real={len(psf['real_viol'])} rsoa={len(psf['rsoa_viol'])}"
          f" | NOM real={len(nom['real_viol'])} rsoa={len(nom['rsoa_viol'])}")
    with open(os.path.join(OUT, 'di_figdata.pkl'), 'wb') as fh:
        pickle.dump((psf, nom), fh)
    replot()


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'figrun':
        figrun(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'replot':
        replot()
    else:
        main()
