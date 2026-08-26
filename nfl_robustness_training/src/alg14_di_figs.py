"""
Figures + results for alg14_di.py (double-integrator split-terminal PSF).

Produces, into nfl_robustness_training/results_di/ :
    di_phase_portrait.png   — (p, v) phase plane: invariant-set boundary,
                              RSOA boxes, nominal vs PSF trajectory
    di_timeseries.png       — v(t) with certified RSOA lower bound, and the
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from REAL_integrated_sim import setup_analyzer
import alg14_di as A


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


def phase_portrait(psf, nom, path):
    pos_min, vel_min = psf['pos_min'], psf['vel_min']
    buffer, dt, u_max = psf['buffer'], psf['dt'], psf['u_max']

    fig, (ax, az) = plt.subplots(1, 2, figsize=(15.0, 5.6),
                                 gridspec_kw={'width_ratios': [1.25, 1.0]})

    ns = np.array(nom['state_history'])
    ps = np.array(psf['state_history'])
    modes = {r['t']: r['mode'] for r in psf['records']}
    act = [i for i in range(len(ps)) if modes.get(i) in ('MPC', 'ACTIVATE')]

    for a in (ax, az):
        # Span well beyond both panels' y-limits so the boundary runs off
        # the top and bottom axes instead of stopping in mid-air.
        vg = np.linspace(vel_min - 1.2, 1.5, 1200)
        pb = parabolic_boundary(vg, u_max, pos_min, buffer)
        a.plot(pb, vg, color='tab:purple', lw=2.0, zorder=6,
               label='terminal invariant-set boundary')
        a.fill_betweenx(vg, pos_min - 1, pb, color='tab:purple', alpha=0.10, zorder=0)
        a.axvline(pos_min, color='k', lw=1.8, zorder=6, label=r'$p \geq p_{\min}$')
        a.axhline(vel_min, color='tab:red', lw=1.8, ls='--', zorder=6,
                  label=r'$v \geq v_{\min}$')
        _boxes(a, nom['bound_history'], 'tab:orange', 0.55, 'RSOA — unfiltered', 0.7, '--')
        _boxes(a, psf['bound_history'], 'tab:blue', 0.55, 'RSOA — REACH-PSF', 0.7, '-')
        a.plot(ns[:, 0], ns[:, 1], 'o-', color='tab:orange', ms=3.0, lw=1.4,
               zorder=7, label='real state — unfiltered')
        a.plot(ps[:, 0], ps[:, 1], 'o-', color='tab:blue', ms=3.0, lw=1.4,
               zorder=8, label='real state — REACH-PSF')
        if act:
            a.plot(ps[act, 0], ps[act, 1], 'o', color='tab:red', ms=8.0,
                   mfc='none', mew=1.8, zorder=9, label='REACH-PSF intervening')
        for v_ in nom['rsoa_viol']:
            b = nom['bound_history'][v_]
            if b is not None:
                a.plot([b[0, 0]], [b[1, 0]], 'x', color='darkred', ms=10, mew=2.4,
                       zorder=10, label=None)
        a.grid(alpha=0.25)
        a.set_xlabel('position $p$')

    ax.set_ylabel('velocity $v$')
    # headroom above the data so the legend sits in empty space, not on the plot
    _vtop = max(0.35, float(ps[:, 1].max()) + 0.1)
    ax.set_ylim(vel_min - 0.20, _vtop + 0.58)
    ax.set_xlim(pos_min - 0.25,
                max(float(ns[:, 0].max()), float(ps[:, 0].max())) + 0.25)
    ax.set_title('(a) full phase portrait', fontsize=11)
    ax.legend(loc='upper center', fontsize=7.8, framealpha=0.95, ncol=3,
              borderaxespad=0.5, handlelength=1.6, columnspacing=1.3)

    # zoom on the v_min crossing — where the constraint actually binds
    lo = np.array([b[1, 0] for b in nom['bound_history'] if b is not None])
    i_worst = int(np.argmin(lo))
    pc = float(ns[i_worst, 0])
    az.set_xlim(pc - 0.85, pc + 0.85)
    az.set_ylim(vel_min - 0.12, vel_min + 0.28)
    az.set_title(r'(b) zoom on the $v_{\min}$ crossing', fontsize=11)
    az.legend().set_visible(False)

    fig.suptitle(f"Double-integrator split-terminal REACH-PSF   "
                 f"($p_{{\min}}={pos_min}$, $v_{{\min}}={vel_min}$, seed {psf['seed']})   "
                 f"\u00d7 = certified-bound violation (unfiltered)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=180); plt.close(fig)


def timeseries(psf, nom, path):
    vel_min = psf['vel_min']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)

    for res, c, name in ((nom, 'tab:orange', 'unfiltered'),
                         (psf, 'tab:blue', 'REACH-PSF')):
        s = np.array(res['state_history'])
        t = np.arange(len(s))
        lo = np.array([b[1, 0] if b is not None else np.nan
                       for b in res['bound_history']])
        hi = np.array([b[1, 1] if b is not None else np.nan
                       for b in res['bound_history']])
        a1.fill_between(t, lo, hi, color=c, alpha=0.18)
        a1.plot(t, lo, color=c, lw=1.0, ls=':')
        a1.plot(t, s[:, 1], color=c, lw=1.8, label=f'$v$ — {name}')

    a1.axhline(vel_min, color='tab:red', lw=2.0, ls='--', label=r'$v_{\min}$')
    for v_ in nom['rsoa_viol']:
        a1.plot([v_], [nom['bound_history'][v_][1, 0]], 'x', color='darkred',
                ms=11, mew=2.6, zorder=9)
    a1.set_ylabel('velocity'); a1.grid(alpha=0.25)
    a1.legend(fontsize=8.0, ncol=3, loc='lower right', framealpha=0.95)
    a1.set_title("Certified RSOA velocity band vs constraint "
                 r"($\times$ = unfiltered violation)", fontsize=10.5)

    act = sorted(r['t'] for r in psf['records'] if r['mode'] in ('MPC', 'ACTIVATE'))
    for a in (a1, a2):
        first = True
        for t_ in act:
            a.axvspan(t_ - 0.5, t_ + 0.5, color='tab:red', alpha=0.11, lw=0,
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
