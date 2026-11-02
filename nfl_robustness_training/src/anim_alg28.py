"""
Animation of an alg28 run from a TTTCARV_RSOA_DUMP file.

The point of the picture is the BACKUP BUFFER: at every timestep the filter is
holding a whole fan of escape routes, one per future branch time tau, not just
the one at t+1. Four panels, all sharing the same frame:

  world (top)     the x-y scene. Certified set, lookahead chain (the RSOA), and
                  every backup plan in the buffer drawn as a route out, coloured
                  by how far ahead it branches.
  ladder (middle) lead time tau - t vs t (now). One cell per (t, lead): green if
                  a certified backup is in hand, red if that lead came back
                  INFEASIBLE, blank if nothing is filed. Read a COLUMN to see
                  what the robot was holding at that instant; read across to see
                  the buffer get built, purged and re-anchored.
  strip (bottom)  who is flying, when a conflict is live, and the re-anchors.

Usage:  python anim_alg28.py <out_prefix> <label>=<dump.json> [...]
        one output set per dump.
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
try:                                   # the cluster has no system ffmpeg
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

C_TRUE   = 'k'
C_CERT   = '#378ADD'
C_CERT_E = '#185FA5'
C_MEAS   = '#bbb'
C_LOOK   = '#BA7517'
C_LOOKF  = '#EF9F27'
C_MPC    = '#C0302F'
C_INFEAS = '#D6453F'
C_RA     = '#2E6FD6'
from matplotlib.colors import LinearSegmentedColormap
# near-term backups in the green already used for "backup", far-term in indigo:
# both stay clear of the orange lookahead and the red MPC.
BACKUP_CMAP = LinearSegmentedColormap.from_list(
    'backup', ['#17A06B', '#1C7FA8', '#3B2E8C'])
WORLD_K = 20                           # lookahead boxes drawn in the world view


def kmax_of(steps):
    """Deepest lead time the dump files anywhere."""
    return max((k - r['t'] for r in steps for k, _ in r['look']), default=1)


def rect(b, **kw):
    b = np.asarray(b)
    return mp.Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0], **kw)


def load(path):
    d = json.load(open(path))
    steps = sorted(d['steps'], key=lambda r: r['t'])
    true = np.array([steps[0]['real_t']] + [r['real_t1'] for r in steps])
    if 'buffer' not in steps[0]:
        raise SystemExit(
            f"{path}: dump has no 'buffer' field — it predates the buffer dump. "
            "Re-run alg28_sound.py with the current source.")
    return d, steps, true


def width(b, j):
    """Width of box b along axis j (0=x, 1=y)."""
    b = np.asarray(b)
    return float(b[j, 1] - b[j, 0])


def danger_margin(b, obstacles, R):
    """Per obstacle: (index, gap to its DANGER circle). <= 0 means in conflict.

    The conflict test is against D = sqrt(r^2 + 2rR), the turning-radius
    inflation that guarantees the robot can always loop away -- NOT against the
    obstacle itself. A box can sit well clear of the grey disc and still be in
    conflict, which is exactly what it looks like on screen.
    """
    b = np.asarray(b)
    out = []
    for i, (cx, cy, r) in enumerate(obstacles):
        dx = max(b[0, 0] - cx, 0.0, cx - b[0, 1])
        dy = max(b[1, 0] - cy, 0.0, cy - b[1, 1])
        out.append((i, float(np.hypot(dx, dy) - np.sqrt(r * r + 2 * r * R))))
    return out


def look_box(r, k):
    """The lookahead box k steps ahead of r['t'], or None."""
    return dict((kk, bb) for kk, bb in r['look']).get(r['t'] + k)


# ── precomputed, frame-independent panels ────────────────────────────────────

def buffer_matrix(steps):
    """(lead, t) -> 0 absent / 1 valid backup / 2 INFEASIBLE, plus the lead range.

    Rows are LEAD TIME tau - t, not absolute tau: every column then reads the
    same way ("how far ahead can I still branch off?"), the diagonal band
    flattens out, and the rows line up with the RSOA panel underneath, whose
    y axis is the same k.
    """
    leads = [tau - r['t'] for r in steps for tau, _ in r['buffer']]
    leads = [k for k in leads if k >= 1]
    if not leads:
        return np.zeros((1, len(steps))), 1, 1
    lo, hi = 1, max(leads)
    M = np.zeros((hi - lo + 1, len(steps)))
    for i, r in enumerate(steps):
        for tau, plan in r['buffer']:
            k = tau - r['t']
            if k >= 1:
                M[k - lo, i] = 1 if plan is not None else 2
    return M, lo, hi




def events_of(steps, name):
    """[(t, event), ...] for every event of the given kind."""
    return [(r['t'], e) for r in steps for e in r.get('events', []) if e[0] == name]


# ── the figure ───────────────────────────────────────────────────────────────

def build(lab, d, steps, true):
    T = len(steps)
    # The scene is wide and flat, so the world gets a full-width row; the three
    # time panels stack below it on one shared timestep axis.
    fig = plt.figure(figsize=(16.0, 10.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.185],
                          height_ratios=[1.75, 1.05, 0.32],
                          left=0.052, right=0.978, top=0.868, bottom=0.062,
                          wspace=0.025, hspace=0.26)
    ax_w = fig.add_subplot(gs[0, :])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[2, 0], sharex=ax_l)
    ax_k = fig.add_subplot(gs[1, 1]); ax_k.axis('off')

    # world limits: true path, obstacles, every backup plan and lookahead box
    hard = [true[:, :2],
            np.array([np.array(o)[:2] + sg * o[2] for o in d['obstacles'] for sg in (1, -1)]),
            np.zeros((1, 2))]
    soft = []
    for r in steps:
        if r['conflict_time'] is not None:
            cb = look_box(r, r['conflict_time'] - r['t'])
            if cb is not None:
                soft.append(np.asarray(cb)[:2].T)
        if r.get('backup_plan'):
            soft.append(np.asarray(r['backup_plan'])[:, :2, :].reshape(-1, 2))
        for k, b in r['look']:
            if k - r['t'] <= WORLD_K:
                soft.append(np.asarray(b)[:2].T)
        for _, plan in r['buffer']:
            if plan:
                soft.append(np.asarray(plan)[:, :2, :].reshape(-1, 2))
    hard = np.vstack(hard)
    soft = np.vstack(soft) if soft else hard
    # The path, the goal and the obstacles always fit. Boxes and plans only up
    # to a percentile, and never further than one hard span past the scene, so
    # a runaway loiter plan cannot shrink everything else to a smudge.
    lo_s, hi_s = np.percentile(soft, [1.0, 99.0], axis=0)
    lim = []
    for j in (0, 1):
        h0, h1 = hard[:, j].min(), hard[:, j].max()
        slack = 0.35 * max(h1 - h0, 1.0)
        lim.append((min(h0, max(lo_s[j], h0 - slack)) - 0.5,
                    max(h1, min(hi_s[j], h1 + slack)) + 0.5))
    XL, YL = lim

    # ── ladder: the backup buffer over the whole run ─────────────────────────
    M, lo, hi_buf = buffer_matrix(steps)
    # The cells only reach as far as the buffer is filed, but the lines drawn
    # over them -- conflict, the scan frontier -- run further ahead, so the axis
    # has to cover every series or they are silently cropped.
    hi = max([hi_buf] + [v - r['t'] for r in steps
                         for v in (r['t_diverge'], r['conflict_time'],
                                   r['wall_tau'], r.get('concrete_until'))
                         if v is not None])
    ax_l.imshow(M, origin='lower', aspect='auto', interpolation='nearest',
                extent=(-0.5, T - 0.5, lo - 0.5, hi_buf + 0.5), zorder=0,
                cmap=ListedColormap(['#FFFFFF', '#2E9E6B', C_INFEAS]),
                norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    def lead(vals):
        return np.array([np.nan if v is None else v - r['t']
                         for r, v in zip(steps, vals)], float)

    ax_l.step(np.arange(T), lead([r['t_diverge'] for r in steps]), where='mid',
              color='k', lw=1.7,
              label='t_diverge: furthest branch still certified')
    ax_l.step(np.arange(T), lead([r.get('concrete_until') for r in steps]),
              where='mid', color=C_RA, lw=1.2, ls='--',
              label='concrete_until: how far the lookahead is scanned')
    ax_l.step(np.arange(T), lead([r['conflict_time'] for r in steps]), where='mid',
              color='#B23AC4', lw=1.7, ls=':',
              label='conflict: where the NN path hits danger')
    for mk, col, lb in (('v', '#2E9E6B', 'conflict cleared'),
                        ('^', C_INFEAS, 'conflict RECONFIRMED (stays live)'),
                        ('D', '#8A8A8A', 'conflict went stale (driven past)')):
        ax_l.plot([], [], marker=mk, ls='', ms=7, color=col, mec='white',
                  mew=0.8, label=lb)
    wt = [(r['t'], r['wall_tau'] - r['t']) for r in steps
          if r.get('wall_tau') is not None]
    if wt:
        ax_l.plot(*zip(*wt), marker='x', ls='', ms=6, mew=1.6, color='#7A1C17',
                  label='wall: first lead with no backup')
    for t, e in events_of(steps, 'reanchor_fire'):
        ax_l.annotate('', xy=(t, e[2] - t), xytext=(t, 0.2),
                      arrowprops=dict(arrowstyle='-|>', color=C_RA, lw=1.9,
                                      shrinkA=0, shrinkB=0))
        ax_l.plot([t], [e[2] - t], marker='o', ms=5.5, mfc='none', mec=C_RA, mew=1.8)
    # How each conflict ENDED. These three look nothing alike on purpose: a
    # re-walk that cleared a conflict, one that reconfirmed it (the conflict
    # stays live, often for many steps), and a conflict the robot simply drove
    # past without it ever being cleared.
    STYLE = {'reanchor_cleared':      ('v', '#2E9E6B', 'cleared'),
             'reanchor_reconfirmed':  ('^', C_INFEAS,  'RECONFIRMED'),
             'conflict_deconflicted': ('v', '#2E9E6B', 'deconflicted'),
             'conflict_stale':        ('D', '#8A8A8A', 'stale')}
    # one pass over all of them in time order, so neighbouring outcomes can be
    # staggered instead of printing on top of each other
    outcomes = sorted((t, k, e) for k in STYLE for t, e in events_of(steps, k))
    for j, (t, k, e) in enumerate(outcomes):
        marker, col, tag = STYLE[k]
        y = hi + (0.5 if j % 2 == 0 else 1.6)
        ax_l.plot([t], [y], marker=marker, ms=8, color=col, mfc=col,
                  mec='white', mew=0.8, zorder=7)
        ax_l.annotate(f'{tag} {e[1]}', xy=(t, y), xytext=(0, 6),
                      textcoords='offset points', fontsize=6.4, color=col,
                      ha='center', va='bottom', zorder=7)
    ax_l.set_xlim(-0.5, T - 0.5); ax_l.set_ylim(lo - 0.5, hi + 3.4)
    ax_l.set_ylabel('lead time  tau - t\n(steps ahead)', fontsize=9)
    ax_l.set_title('BACKUP BUFFER — every escape route in hand, at every instant '
                   '(green = certified backup, red = tau with none)',
                   fontsize=10.5, pad=6)
    # into the quiet top-right of the ladder, where nothing is ever filed
    # parked in the white block on the right, clear of both the cells and the
    # outcome markers along the top
    ax_l.legend(fontsize=6.6, ncol=1, loc='center right', framealpha=0.90,
                bbox_to_anchor=(0.999, 0.56), handlelength=1.8, borderpad=0.5)
    ax_l.set_yticks([k for k in range(lo, hi + 1)
                     if k == 1 or k % (1 + (hi - lo) // 9) == 0])
    ax_l.tick_params(labelsize=8, labelbottom=False)
    ax_l.grid(alpha=0.12)
    band_l = ax_l.axvspan(-0.5, 0.5, color='k', alpha=0.10, zorder=4)

    ax_k.text(0.0, 1.0,
              'READING THE LADDER\n\n'
              'One column = one instant.\n'
              'Its green cells are every\n'
              'lead time the robot can\n'
              'still peel off at and stop\n'
              'safely.\n\n'
              'Red = the MPC could not\n'
              'certify that lead. The\n'
              'lowest red cell is the WALL.\n\n'
              'Conflict is tested against\n'
              'the DOTTED danger circle,\n'
              'not the grey obstacle.\n\n'
              'A RE-ANCHOR (blue arrow)\n'
              'rewinds the dashed scan\n'
              'line and re-walks from a\n'
              'fresher certified set, so\n'
              'red can turn green.',
              fontsize=7.4, va='top', ha='left', linespacing=1.38,
              transform=ax_k.transAxes)
    # ── strip: who is flying, conflicts, re-anchors ──────────────────────────
    mpc = np.array([r['kind'] == 'mpc' for r in steps], float)
    ax_s.bar(np.arange(T), mpc, width=1.0, color=C_MPC, alpha=0.40, zorder=1)
    live = np.array([r['conflict_time'] is not None for r in steps], float)
    ax_s.bar(np.arange(T), live * 0.30, width=1.0, color='#B23AC4', alpha=0.30, zorder=0)
    for j, (t, e) in enumerate(events_of(steps, 'reanchor_fire')):
        ax_s.annotate(f"re-anchor\nwall tau={e[1]}",
                      xy=(t, 1.02), xytext=(t, 1.15 + 0.46 * (j % 2)),
                      fontsize=6.8, color=C_RA, ha='center', va='bottom',
                      arrowprops=dict(arrowstyle='-|>', color=C_RA, lw=1.4))
    nint = int(sum(1 for i in range(1, T) if mpc[i] and not mpc[i - 1]) + mpc[0])
    ax_s.set_ylim(0, 2.45); ax_s.set_yticks([]); ax_s.set_xlim(-0.5, T - 0.5)
    ax_s.set_xlabel('timestep   (0.2 s each)', fontsize=9)
    ax_s.tick_params(labelsize=8)
    ax_s.set_title(f'WHO IS FLYING — backup in control {int(mpc.sum())} steps over '
                   f'{nint} interventions (red), conflict live (purple), '
                   f're-anchors (blue)', fontsize=10.5, pad=6)

    hud = fig.text(0.5, 0.922, '', ha='center', va='bottom', fontsize=12.5)
    sub = fig.text(0.5, 0.901, '', ha='center', va='bottom', fontsize=9.5,
                   bbox=dict(fc='white', ec='#ccc', alpha=0.95, pad=3.0))
    banner = fig.text(0.5, 0.878, '', ha='center', va='bottom', fontsize=11.5,
                      color='white', visible=False,
                      bbox=dict(fc=C_RA, ec='none', pad=4.0, alpha=0.95))
    cursors = [a.axvline(0, color='k', lw=1.3, zorder=6) for a in (ax_l, ax_s)]
    bands = [band_l]

    # ── per-frame drawing ────────────────────────────────────────────────────
    R = d['turning_radius']

    def draw(i):
        r = steps[min(i, T - 1)]
        t = r['t']
        ax_w.clear()
        ax_w.set_xlim(*XL); ax_w.set_ylim(*YL)
        # 'box': shrink the axes to the scene's aspect instead of padding the
        # data limits, so a wide flat scene is not buried in empty sky
        ax_w.set_aspect('equal', adjustable='box')
        ax_w.grid(alpha=0.18)
        ax_w.tick_params(labelsize=8)
        for cx, cy, rad in d['obstacles']:
            ax_w.add_patch(mp.Circle((cx, cy), rad, color='#555', alpha=0.55, zorder=2))
            ax_w.add_patch(mp.Circle((cx, cy), np.sqrt(rad * rad + 2 * rad * R),
                                     fill=False, ls=':', color='#555', lw=0.8, zorder=2))
        ax_w.plot(0, 0, marker='*', ms=15, color='gold', mec='k', zorder=3)

        # the lookahead chain: the RSOA the NN path is certified against. Only
        # the first WORLD_K boxes, or the scene turns to mush -- but the box AT
        # the conflict is always drawn, however far ahead it is, so "conflict at
        # tau=27" has something to point at.
        ct = r['conflict_time']
        for k, b in r['look']:
            if k - t > WORLD_K and k != ct:
                continue
            a = max(0.12, 0.80 - 0.035 * (k - t))
            ax_w.add_patch(rect(b, fc=C_LOOKF, ec=C_LOOK, lw=1.0, alpha=a * 0.35, zorder=1))
            ax_w.add_patch(rect(b, fill=False, ec=C_LOOK, lw=1.0, alpha=a, zorder=1))
        cb_box = look_box(r, ct - t) if ct is not None else None
        if ct is not None and cb_box is None:
            # The conflict is still on the books but its bound is gone: an MPC
            # phase deletes every horizon past 'now', and the re-scan has not
            # reached tau yet. Say so, rather than drawing nothing and leaving
            # the header claiming a conflict with nothing to point at.
            chain = dict(r['look'])
            if chain:
                last = max(chain)
                cc = np.asarray(chain[last]).mean(1)
                ax_w.annotate(f'conflict tau={ct} (+{ct - t})\n'
                              f'bound discarded, not re-scanned yet\n'
                              f'(chain only reaches tau={last})',
                              xy=(cc[0], cc[1]), xytext=(0, 20),
                              textcoords='offset points', fontsize=8,
                              color='#8E22A0', ha='center', va='bottom', zorder=8,
                              bbox=dict(fc='white', ec='#B23AC4', ls='--',
                                        alpha=0.88, pad=2.2),
                              arrowprops=dict(arrowstyle='-', color='#B23AC4',
                                              ls='--', lw=1.1))
        if cb_box is not None:
            ax_w.add_patch(rect(cb_box, fill=False, ec='#B23AC4', lw=2.4, ls='--',
                                zorder=7))
            # Call out WHICH boundary it violates: the danger circle, not the
            # obstacle. Without this the box looks like it is in open space.
            marg = danger_margin(cb_box, d['obstacles'], R)
            hits = [i for i, g in marg if g <= 0]
            gap = min(g for _, g in marg)
            for oi in hits:
                cx, cy, rad = d['obstacles'][oi]
                ax_w.add_patch(mp.Circle((cx, cy), np.sqrt(rad * rad + 2 * rad * R),
                                         fill=False, ls='--', color='#B23AC4',
                                         lw=2.0, zorder=6))
            clab = (f'conflict\ntau={ct} (+{ct - t})\n'
                    f'{-gap * 100:.0f} cm INSIDE danger radius' if hits else
                    f'tau={ct} (+{ct - t}) still flagged\n'
                    f'but box is {gap * 100:.0f} cm CLEAR of danger\n'
                    f'(never re-tested)')
            cc = np.asarray(cb_box).mean(1)
            ax_w.annotate(clab, xy=(cc[0], cc[1]),
                          xytext=(0, 16), textcoords='offset points',
                          fontsize=8.5, ha='center', va='bottom', zorder=8,
                          color='#8E22A0' if hits else '#6B6B6B',
                          bbox=dict(fc='white', alpha=0.88, pad=1.8,
                                    ec='#B23AC4' if hits else '#9A9A9A'))

        # THE FAN: every backup in the buffer, not just the one at t+1
        nvalid = ninfeas = 0
        for tau, plan in r['buffer']:
            if tau <= t:
                continue
            if plan is None:
                b = look_box(r, tau - t)
                if b is not None:
                    c = np.asarray(b).mean(1)
                    ax_w.plot(c[0], c[1], marker='x', ms=9, mew=2.2,
                              color=C_INFEAS, zorder=6)
                ninfeas += 1
                continue
            nvalid += 1
            col = BACKUP_CMAP(np.clip((tau - t) / WORLD_K, 0, 1) * 0.82 + 0.08)
            P = np.asarray(plan)                       # (n, 2, 2) -> x,y lo/hi
            c = P.mean(2)                              # centres
            use = (tau == r['t_diverge'])
            ax_w.plot(c[:, 0], c[:, 1], color=col, lw=2.4 if use else 1.1,
                      alpha=0.95 if use else 0.72, zorder=5 if use else 4,
                      solid_capstyle='round')
            ax_w.add_patch(rect(P[-1], fc=col, ec=col, alpha=0.30, lw=0.8, zorder=4))
            ax_w.plot(c[0, 0], c[0, 1], marker='o', ms=3.0, color=col, zorder=5)

        # the plan that is actually flying: the whole remaining route, so the
        # scene does not go blank for the steps the backup is in control
        if r['kind'] == 'mpc' and r.get('backup_plan'):
            P = np.asarray(r['backup_plan'])
            c = P.mean(2)
            for b in P:
                ax_w.add_patch(rect(b, fill=False, ec=C_MPC, lw=0.9,
                                    alpha=0.55, zorder=5))
            ax_w.plot(c[:, 0], c[:, 1], color=C_MPC, lw=2.6, zorder=6,
                      solid_capstyle='round')
            ax_w.add_patch(rect(P[-1], fc=C_MPC, ec=C_MPC, alpha=0.28, zorder=5))
        if r['kind'] == 'mpc' and r.get('backup_bound_t') is not None:
            ax_w.add_patch(rect(r['backup_bound_t'], fc=C_MPC, ec=C_MPC,
                                alpha=0.45, lw=1.8, zorder=7))

        if i > 0 and steps[i - 1].get('meas_t1'):
            ax_w.add_patch(rect(steps[i - 1]['meas_t1'], fc=C_MEAS, ec='#888',
                                alpha=0.5, zorder=3))
        if r['bound_t'] is not None:
            b = np.asarray(r['bound_t'])
            ax_w.add_patch(rect(b, fc=C_CERT, ec=C_CERT_E, alpha=0.60, zorder=4))
            c = b.mean(1)
            ax_w.plot(c[0], c[1], 'x', color=C_CERT_E, ms=8, mew=2, zorder=5)
        ax_w.plot(true[:i + 1, 0], true[:i + 1, 1], color=C_TRUE, lw=1.3, zorder=5)
        x = true[i]
        ax_w.plot(x[0], x[1], 'o', color=C_TRUE, ms=5, zorder=6)
        ax_w.arrow(x[0], x[1], 0.3 * np.cos(x[2]), 0.3 * np.sin(x[2]),
                   head_width=0.08, color=C_TRUE, zorder=6)

        flying = r['kind'] == 'mpc'
        fired = [e for e in r.get('events', []) if e[0] == 'reanchor_fire']
        cleared = [e for e in r.get('events', []) if e[0] == 'reanchor_cleared']
        recon = [e for e in r.get('events', []) if e[0] == 'reanchor_reconfirmed']
        note = ''
        if fired:
            note = (f"   RE-ANCHOR: wall at tau={fired[0][1]}, re-walking to "
                    f"tau={fired[0][2]} from here ({fired[0][3]} steps fresher)")
        elif cleared:
            note = f"   RE-ANCHOR CLEARED: tau={cleared[0][1]} is clean from the new anchor"
        elif recon:
            note = (f"   RE-ANCHOR RECONFIRMED: tau={recon[0][1]} is still blocked "
                    f"— the conflict stays live")
        elif [e for e in r.get('events', []) if e[0] == 'conflict_stale']:
            st = [e for e in r.get('events', []) if e[0] == 'conflict_stale'][0]
            note = (f"   CONFLICT tau={st[1]} WENT STALE: driven past without "
                    f"ever being cleared")
        elif [e for e in r.get('events', []) if e[0] == 'conflict_deconflicted']:
            dc = [e for e in r.get('events', []) if e[0] == 'conflict_deconflicted'][0]
            note = f"   DECONFLICTED: tau={dc[1]} cleared by refinement"
        hud.set_text(f'{lab}    t={t}  ({d["dt"] * t:.1f} s)    '
                     + ('BACKUP (MPC) FLYING' if flying else 'NN in control'))
        hud.set_color(C_MPC if flying else 'k')
        sub.set_text(
            f'{nvalid} backups in hand'
            + (f'   |   {ninfeas} lead(s) with none' if ninfeas else '')
            + (f'   |   certified out to tau={r["t_diverge"]} '
               f'(+{r["t_diverge"] - t})' if r['t_diverge'] is not None else '')
            + (f'   |   conflict at tau={r["conflict_time"]} '
               f'(+{r["conflict_time"] - t})' if r['conflict_time'] is not None else ''))
        banner.set_visible(bool(note))
        if note:
            banner.set_text(note.strip())
            stale_ev = [e for e in r.get('events', [])
                        if e[0] in ('conflict_stale', 'conflict_deconflicted')]
            banner.set_bbox(dict(fc=('#6B6B6B' if stale_ev else
                                     C_RA if (fired or cleared) else C_INFEAS),
                                 ec='none', pad=4.0, alpha=0.95))

        for cur in cursors:
            cur.set_xdata([i, i])
        for bd in bands:              # axvspan is a Rectangle here, not a Polygon
            bd.set_x(i - 0.5); bd.set_width(1.0)
        return []

    handles = [
        mp.Patch(fc=C_MEAS, ec='#888', label='measurement box  z ± vmax'),
        mp.Patch(fc=C_CERT, alpha=0.6, label='certified set S_t (contains the truth)'),
        plt.Line2D([], [], marker='x', ls='', color=C_CERT_E, label='estimate (what the NN reads)'),
        plt.Line2D([], [], marker='o', color=C_TRUE, label='true state / path'),
        mp.Patch(fill=False, ec=C_LOOK, label=f'RSOA lookahead chain t+1..t+{WORLD_K}'),
        plt.Line2D([], [], color=BACKUP_CMAP(0.15), lw=2,
                   label='backup route branching SOON'),
        plt.Line2D([], [], color=BACKUP_CMAP(0.85), lw=2,
                   label='backup route branching LATE'),
        plt.Line2D([], [], marker='x', ls='', color=C_INFEAS, label='lead with NO backup (wall)'),
        plt.Line2D([], [], color=C_MPC, lw=2.5, label='the backup actually flying'),
        mp.Patch(color='#555', alpha=0.55, label='obstacle (dotted: danger radius)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=8.5, frameon=False, columnspacing=1.6,
               handlelength=1.8)
    return fig, draw, T


def main():
    out = sys.argv[1]
    for arg in sys.argv[2:]:
        lab, path = arg.split('=', 1)
        d, steps, true = load(path)
        fig, draw, T = build(lab, d, steps, true)
        pre = f'{out}_{lab}' if len(sys.argv) > 3 else out
        anim = FuncAnimation(fig, draw, frames=T, interval=250)
        anim.save(pre + '.gif', writer=PillowWriter(fps=4), dpi=78)
        print('wrote', pre + '.gif')
        try:
            anim.save(pre + '.mp4', writer=FFMpegWriter(
                fps=4, extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264']), dpi=100)
            print('wrote', pre + '.mp4')
        except Exception as e:
            print('mp4 skipped:', e)
        # stills: a quiet step, each re-anchor, and the first MPC step
        marks = {4}
        marks |= {t for t, _ in events_of(steps, 'reanchor_fire')}
        marks |= {r['t'] for r in steps if r['kind'] == 'mpc'}
        for i in sorted(marks)[:8]:
            if 0 <= i < T:
                draw(i); fig.savefig(f'{pre}_t{i}.png', dpi=100)
        plt.close(fig)


if __name__ == '__main__':
    main()
