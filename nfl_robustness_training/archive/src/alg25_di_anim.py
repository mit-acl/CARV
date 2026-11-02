"""
Animation for the DI reachable sets without the estimator box (alg25).

Two panels, one timestep per frame:

  LEFT   the (p, v) phase plane -- terminal invariant set, both half-plane
         constraints, and the boxes revealed one step at a time so you can see
         each one grow out of its parent instead of reading 40 of them at once.
  RIGHT  box width vs t, one line per arm per coordinate. This is the panel
         that answers "do the bounds just keep getting bigger?" -- growth,
         contraction, or a step change when the PSF takes over is a slope here,
         not something you have to eyeball off overlapping rectangles.

Reads results_di_nokf/di_figdata.pkl, the same pickle alg25_di_figs writes, so
the animation and the static figure are always the same run.

Usage:
    python nfl_robustness_training/src/alg25_di_anim.py [pkl_dir]
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alg14_di_figs import (parabolic_boundary, C_PSF, C_NOM, C_TERM, C_VIOL)

FPS = 6


def _widths(hist):
    """(dp, dv) per step; NaN where a horizon carried no bound."""
    dp, dv = [], []
    for b in hist:
        if b is None:
            dp.append(np.nan); dv.append(np.nan)
        else:
            b = np.asarray(b, dtype=float)
            dp.append(b[0, 1] - b[0, 0]); dv.append(b[1, 1] - b[1, 0])
    return np.array(dp), np.array(dv)


def animate(psf, nom, out_base):
    pos_min, vel_min = psf['pos_min'], psf['vel_min']
    ps, ns = np.array(psf['state_history']), np.array(nom['state_history'])
    pb_hist, nb_hist = psf['bound_history'], nom['bound_history']
    modes = {r['t']: r['mode'] for r in psf['records']}
    n = min(len(ps), len(ns), len(pb_hist), len(nb_hist))

    pdp, pdv = _widths(pb_hist[:n])
    ndp, ndv = _widths(nb_hist[:n])
    ts = np.arange(n)

    fig, (ax, aw) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                 gridspec_kw={'width_ratios': [1.35, 1.0]})

    # ---- static layers on the phase plane -------------------------------
    vg = np.linspace(vel_min - 1.2, 1.5, 1200)
    pbnd = parabolic_boundary(vg, psf['u_max'], pos_min, psf['buffer'])
    ax.fill_betweenx(vg, pos_min - 1.0, pbnd, color=C_TERM, alpha=0.09, lw=0, zorder=0)
    ax.plot(pbnd, vg, color=C_TERM, lw=1.6, zorder=5, label='terminal invariant set')
    ax.axvline(pos_min, color='0.15', lw=1.4, zorder=5, label=r'$p \geq p_{\min}$')
    ax.axhspan(vel_min - 10.0, vel_min, color=C_VIOL, alpha=0.055, lw=0, zorder=0)
    ax.axhline(vel_min, color=C_VIOL, lw=1.5, ls=(0, (5, 2.5)), zorder=5,
               label=r'$v \geq v_{\min}$')

    allp = np.concatenate([ps[:n, 0], ns[:n, 0]])
    allv = np.concatenate([ps[:n, 1], ns[:n, 1]])
    mp = 0.10 * (allp.max() - allp.min() + 1e-9)
    ax.set_xlim(min(pos_min - 0.15, allp.min() - mp), allp.max() + mp)
    ax.set_ylim(vel_min - 0.35, max(allv.max() + 0.25, 0.4))
    ax.set_xlabel('position $p$'); ax.set_ylabel('velocity $v$')
    ax.set_title('Forward reachable sets — no estimator box')

    # ---- static layers on the width panel -------------------------------
    aw.set_xlim(0, max(n - 1, 1))
    wmax = np.nanmax(np.concatenate([pdp, pdv, ndp, ndv]))
    aw.set_ylim(0, (wmax if np.isfinite(wmax) and wmax > 0 else 1.0) * 1.15)
    aw.set_xlabel('timestep'); aw.set_ylabel('box width')
    aw.set_title('Width of the reachable set')
    aw.grid(alpha=0.25, lw=0.6)

    lines = {
        'ndp': aw.plot([], [], '-',  color=C_NOM, lw=1.8, label=r'unfiltered  $\Delta p$')[0],
        'ndv': aw.plot([], [], '--', color=C_NOM, lw=1.5, label=r'unfiltered  $\Delta v$')[0],
        'pdp': aw.plot([], [], '-',  color=C_PSF, lw=1.8, label=r'REACH-PSF  $\Delta p$')[0],
        'pdv': aw.plot([], [], '--', color=C_PSF, lw=1.5, label=r'REACH-PSF  $\Delta v$')[0],
    }
    aw.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Shade the timesteps where the PSF is in control, on both panels' terms.
    for t in range(n):
        if modes.get(t) in ('MPC', 'ACTIVATE'):
            aw.axvspan(t - 0.5, t + 0.5, color=C_PSF, alpha=0.10, lw=0, zorder=0)

    hud = ax.text(0.015, 0.975, '', transform=ax.transAxes, va='top', ha='left',
                  fontsize=9, family='monospace',
                  bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.7', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    drawn = []          # artists cleared between frames

    def draw_frame(k):
        for a in drawn:
            a.remove()
        drawn.clear()

        # Boxes 0..k, oldest faintest, so the growth reads as a sequence.
        for (hist, color, ls) in ((nb_hist, C_NOM, (0, (3, 2))),
                                  (pb_hist, C_PSF, '-')):
            for i in range(k + 1):
                b = hist[i]
                if b is None:
                    continue
                b = np.asarray(b, dtype=float)
                fade = 0.22 + 0.63 * (i / max(k, 1))
                lw = 1.5 if i == k else 0.7
                drawn.append(ax.add_patch(Rectangle(
                    (b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                    fill=False, edgecolor=color, alpha=fade, lw=lw, ls=ls, zorder=6)))

        drawn.append(ax.plot(ns[:k + 1, 0], ns[:k + 1, 1], '-o', color=C_NOM,
                             lw=1.3, ms=3.0, zorder=7)[0])
        drawn.append(ax.plot(ps[:k + 1, 0], ps[:k + 1, 1], '-o', color=C_PSF,
                             lw=1.3, ms=3.0, zorder=8)[0])

        act = [i for i in range(k + 1) if modes.get(i) in ('MPC', 'ACTIVATE')]
        if act:
            drawn.append(ax.plot(ps[act, 0], ps[act, 1], 'o', color=C_PSF, ms=9.0,
                                 mfc='none', mew=1.6, zorder=9)[0])

        rv = [i for i in nom['real_viol'] if i <= k]
        if rv:
            drawn.append(ax.plot(ns[rv, 0], ns[rv, 1], 'x', color=C_VIOL, ms=9.0,
                                 mew=2.4, zorder=11)[0])

        for key, arr in (('ndp', ndp), ('ndv', ndv), ('pdp', pdp), ('pdv', pdv)):
            lines[key].set_data(ts[:k + 1], arr[:k + 1])

        mode = modes.get(k, 'NOMINAL')
        hud.set_text(f't = {k:2d}   mode {mode:>8s}\n'
                     f'PSF   dp {pdp[k]:6.3f}  dv {pdv[k]:6.3f}\n'
                     f'unfil dp {ndp[k]:6.3f}  dv {ndv[k]:6.3f}')

    frames = []
    for k in range(n):
        draw_frame(k)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
    # Hold the final state so the last frame is readable on loop.
    frames.extend([frames[-1]] * FPS)

    gif, mp4 = out_base + '.gif', out_base + '.mp4'
    imageio.mimsave(gif, frames, fps=FPS, loop=0)
    try:
        w = imageio.get_writer(mp4, fps=FPS, codec='libx264',
                               macro_block_size=1, quality=8)
        for f in frames:
            w.append_data(f)
        w.close()
    except Exception as e:                       # ffmpeg not always present
        mp4 = f'(mp4 skipped: {e})'
    plt.close(fig)
    print(f'wrote {gif}\nwrote {mp4}')
    return gif, mp4


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        'nfl_robustness_training', 'results_di_nokf')
    with open(os.path.join(d, 'di_figdata.pkl'), 'rb') as fh:
        psf, nom = pickle.load(fh)
    animate(psf, nom, os.path.join(d, 'di_reachable_sets'))
