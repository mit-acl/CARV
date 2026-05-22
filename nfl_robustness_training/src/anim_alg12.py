"""
Animation for alg12 — Predictive Safety Filter (PSF) with Forward MPC Buffer.
Saves:
    alg12_psf_buffer.gif
    alg12_psf_buffer.mp4

Usage:
    python anim_alg12.py [seed] [--obs "cx,cy,r" "cx,cy,r" ...]
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter
from alg12_mpc_every_timestep import (
    concrete_scan, symbolic_step, VerificationTask,
    build_mpc_backup, psf_valid, extend_mpc_sequence,
)
from time_budget import TimeBudget
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional
import sys
import io
import imageio

# ── Config ──
MAX_TIME             = 60
MAX_SYMBOLIC_HORIZON = 5


_DEFAULT_OBSTACLES = [
    np.array([-6,  1.5,  0.5]),
]

_args = sys.argv[1:]
_seed = 1401830092
if _args and not _args[0].startswith('--'):
    _seed = int(_args.pop(0))

if '--obs' in _args:
    _obs_idx  = _args.index('--obs')
    _obs_strs = _args[_obs_idx + 1:]
    obstacles = [np.array([float(v) for v in s.split(',')]) for s in _obs_strs]
else:
    obstacles = _DEFAULT_OBSTACLES

print(f"Setting up analyzer...  seed={_seed}")
for o in obstacles:
    print(f"  obstacle: center=({o[0]:.3f}, {o[1]:.3f})  r={o[2]}")

analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
tester   = ReachabilityTester(analyzer, obstacles, seed=_seed)
mpc_sf   = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=10, use_safety_radius=True)

budget = TimeBudget(timestep_budget=0.2)
budget.symbolic_costs = {
    1:  0.05942702293395996,  2: 0.0532071590423584,
    3:  0.12308859825134277,  4: 0.2227306365966797,
    5:  0.3548123836517334,   6: 0.5160810947418213,
    7:  0.7076215744018555,   8: 1.046485185623169,
    9:  1.189185619354248,   10: 1.4745268821716309,
}
budget.concrete_cost = 0.015


# ── Frame helpers ──

def snapshot_rsoa(tester):
    snap = {}
    for t, horizon in tester.horizons.items():
        tb = horizon.get_tight_bound()
        if tb is not None:
            calc_type = "empirical"
            for calc_id in sorted(horizon.calculations.keys()):
                calc_type = horizon.calculations[calc_id]["calc_type"].value
            snap[t] = {
                "x1":  (float(tb[0, 0]), float(tb[0, 1])),
                "x2":  (float(tb[1, 0]), float(tb[1, 1])),
                "type": calc_type,
            }
    return snap


def _tdiv_traj():
    """Return traj_bounds to preview: active MPC plan during MPC_ACTIVE,
    else the t_diverge buffer entry."""
    if mpc_started and mpc_state['traj_bounds']:
        return mpc_state['traj_bounds']
    if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
        return mpc_buffer[t_diverge][1]
    return []


def _current_phase():
    if mpc_started:
        return "MPC_ACTIVE"
    if conflict_time is not None:
        return "PRE_CONFLICT"
    return "NOMINAL"


def push(frames, current_t, label, origin="info"):
    frames.append((
        snapshot_rsoa(tester),
        _tdiv_traj(),          # t_diverge backup trajectory (pink preview)
        t_diverge,             # for star marker
        current_t,
        label,
        origin,
        list(mpc_traj_executed),  # accumulated MPC-executed bounds trail
        _current_phase(),      # actual simulation phase for badge
    ))


# ── Module-level PSF state ──
mpc_buffer:        dict            = {}
concrete_until:    int             = 0
mpc_horizon_until: int             = -1
conflict_time:     Optional[int]   = None
t_diverge:         Optional[int]   = None
pending_job:       Optional[VerificationTask] = None

mpc_state = {
    'committed_at':  None,
    'conflict_time': None,
    'controls':      [],
    'traj_bounds':   [],
    'needed':        False,
}
mpc_started      = False
mpc_traj_executed = []   # accumulated executed MPC bound boxes


# ── Simulation with frame capture ──
print("Running alg12 PSF buffer...")
frames = []

current_timestep = 0
push(frames, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()
    t_next = current_timestep + 1

    # ══ MPC ACTIVE PHASE ═════════════════════════════════════════════════
    if mpc_started:
        ctrl_idx = current_timestep - mpc_state['committed_at']
        queue    = mpc_state['controls']

        if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
            extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                ctrl_idx=ctrl_idx, budget=budget)
            queue = mpc_state['controls']
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC extend] queue now {len(queue)}", "mpc")

        # Clear stale horizons; interleave: one concrete step → one MPC build,
        # repeat. First iteration always runs (guarantees mpc_buffer[t_next]).
        for _t in list(tester.horizons.keys()):
            if _t > current_timestep:
                del tester.horizons[_t]
        concrete_until = current_timestep
        _first = True
        while _first or budget.can_afford('concrete'):
            _first = False
            if concrete_until >= MAX_TIME:
                break
            # One concrete step
            collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
            concrete_until += 1
            if collision:
                break
            # One MPC build for this timestep
            if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until)
                mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                if valid:
                    t_diverge = concrete_until
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC PSF] τ={concrete_until}"
                     f"  {'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")
                if not valid:
                    break  # infeasible wall
            else:
                break  # no MPC budget remaining

        # PSF reversion check
        if psf_valid(mpc_buffer, t_next):
            push(frames, current_timestep,
                 f"t={current_timestep}  [PSF REVERT] mpc_buffer[{t_next}] valid"
                 f"  mpc_hz={mpc_horizon_until} — back to nominal", "info")
            mpc_started                = False
            mpc_state['needed']        = False
            mpc_state['committed_at']  = None
            mpc_state['conflict_time'] = None
            # Keep conflict_time and pending_job — conflict is still real;
            # only P4 symbolic deconfliction should clear them.
            # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
            # were built from nominal-trajectory concrete bounds (concrete_scan
            # always uses nominal dynamics) so they remain valid safety backups
            # now that we are reverting to nominal control.
            for _k in list(mpc_buffer.keys()):
                if _k > t_next and mpc_buffer[_k] is None:
                    del mpc_buffer[_k]
            valid_keys = [k for k, v in mpc_buffer.items()
                          if k >= current_timestep and v is not None]
            t_diverge         = max(valid_keys) if valid_keys else None
            mpc_horizon_until = t_diverge if t_diverge is not None else t_next
            concrete_until             = t_next
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1
            push(frames, current_timestep,
                 f"t={current_timestep}  nominal step (post-revert)  t_div={t_diverge}", "info")
            continue

        # Apply MPC control
        if ctrl_idx >= len(queue):
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] ERROR queue exhausted — ABORT", "mpc")
            break
        ctrl = queue[ctrl_idx]
        if ctrl_idx + 1 < len(mpc_state['traj_bounds']):
            mpc_traj_executed.append(mpc_state['traj_bounds'][ctrl_idx + 1].copy())
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC] idx={ctrl_idx}/{len(queue)-1}"
             f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})", "mpc")
        tester.real_state_mpc(current_timestep, ctrl)
        current_timestep += 1
        continue

    # ══ NOMINAL / PRE_CONFLICT PHASE ════════════════════════════════════

    # Stale conflict check — clear if we've already navigated past conflict_time.
    if conflict_time is not None and current_timestep > conflict_time:
        conflict_time = None
        pending_job   = None

    phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
    push(frames, current_timestep,
         f"t={current_timestep}  concrete_until={concrete_until}"
         f"  mpc_hz={mpc_horizon_until}  conflict={conflict_time}"
         f"  t_div={t_diverge}  [{phase}]", "info")

    # P1 — Ensure mpc_buffer[t+1]
    if t_next not in mpc_buffer:
        if t_next not in tester.horizons:
            concrete_scan(tester, concrete_until, t_next)
            concrete_until = max(concrete_until, t_next)
        if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
            valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next)
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            if valid:
                t_diverge = t_next
            push(frames, current_timestep,
                 f"t={current_timestep}  [P1] mpc_buffer[{t_next}]="
                 f"{'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")

    # P2 — Concrete forward (stop at conflict_time)
    scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
    while concrete_until < scan_ceil and budget.can_afford('concrete'):
        end = min(concrete_until + budget.max_affordable_concrete(), scan_ceil, MAX_TIME)
        collision, ct = concrete_scan(tester, concrete_until, end)
        if collision:
            if conflict_time is None or ct < conflict_time:
                conflict_time = ct
                scan_ceil     = ct
                if pending_job is None:
                    pending_job = VerificationTask(current_timestep, conflict_time)
            concrete_until = ct
            push(frames, current_timestep,
                 f"t={current_timestep}  [P2] concrete {concrete_until - (end - concrete_until)}"
                 f"→{ct}  ⚠ conflict@{conflict_time}", "baseline")
            break
        concrete_until = end
        push(frames, current_timestep,
             f"t={current_timestep}  [P2] concrete →{concrete_until}  ✓ clear", "baseline")

    # P3 — MPC buffer forward toward conflict_time
    if conflict_time is not None:
        mpc_horizon_until = max(mpc_horizon_until, current_timestep)
        while (mpc_horizon_until < conflict_time - 1
               and budget.remaining >= budget.mpc_cost):
            buf_t = mpc_horizon_until + 1
            if buf_t not in tester.horizons:
                break
            if mpc_buffer.get(mpc_horizon_until) is None:
                break  # infeasible wall — don't extend further
            valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, buf_t)
            mpc_horizon_until = buf_t
            push(frames, current_timestep,
                 f"t={current_timestep}  [P3] mpc_buffer[{buf_t}]="
                 f"{'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")
            if valid:
                t_diverge = buf_t
            else:
                break

    # P4 — Symbolic (continuous)
    if pending_job is not None and budget.can_afford('symbolic', 1):
        sym_from    = pending_job.symbolic_start
        pending_job, result = symbolic_step(tester, pending_job,
                                            MAX_SYMBOLIC_HORIZON, budget)
        push(frames, current_timestep,
             f"t={current_timestep}  [P4] symbolic {sym_from}→{pending_job.symbolic_start}",
             "baseline")
        if result is not None:
            if result["collision"]:
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] conflict confirmed@{conflict_time}"
                     f" — refreshing buffer", "baseline")
                # Symbolic tightened bounds — drop only INFEASIBLE entries so
                # P3 retries them. VALID entries remain safe.
                for k in list(mpc_buffer.keys()):
                    if k > current_timestep and mpc_buffer[k] is None:
                        del mpc_buffer[k]
                valid_keys = [k for k, v in mpc_buffer.items() if k >= current_timestep and v is not None]
                t_diverge = max(valid_keys) if valid_keys else None
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                pending_job = VerificationTask(current_timestep, conflict_time)
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] confirmed — kept VALID entries"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "mpc")
            else:
                conflict_time = None
                pending_job   = None
                for k in list(mpc_buffer.keys()):
                    if k > current_timestep and mpc_buffer[k] is None:
                        del mpc_buffer[k]
                valid_keys = [k for k, v in mpc_buffer.items() if k >= current_timestep and v is not None]
                t_diverge = max(valid_keys) if valid_keys else None
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] ✓ deconflicted — kept VALID entries"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "baseline")

    # ══ PSF DECISION ════════════════════════════════════════════════════
    if psf_valid(mpc_buffer, t_next):
        push(frames, current_timestep,
             f"t={current_timestep}  [NOMINAL] PSF OK — mpc_buffer[{t_next}] valid", "info")
        tester.real_state_empirical(current_timestep, t_next)
        current_timestep += 1
        push(frames, current_timestep,
             f"t={current_timestep}  nominal step  t_div={t_diverge}", "info")

    else:
        if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
            controls, traj_bounds = mpc_buffer[t_diverge]
            mpc_state['committed_at']  = t_diverge
            mpc_state['conflict_time'] = conflict_time
            mpc_state['controls']      = list(controls)
            mpc_state['traj_bounds']   = list(traj_bounds)
            mpc_state['needed']        = True
            mpc_started                = True

            ctrl_idx = current_timestep - t_diverge
            if ctrl_idx < len(controls):
                ctrl = controls[ctrl_idx]
                if ctrl_idx + 1 < len(traj_bounds):
                    mpc_traj_executed.append(traj_bounds[ctrl_idx + 1].copy())
                push(frames, current_timestep,
                     f"t={current_timestep}  [PSF ACTIVATE] PSF failed —"
                     f" backup t_div={t_diverge}  ctrl_idx={ctrl_idx}"
                     f"  u={np.round(ctrl, 4)}", "mpc")
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    push(frames, current_timestep,
                         f"t={current_timestep}  [PSF ACTIVATE extended]"
                         f"  u={np.round(ctrl, 4)}", "mpc")
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    push(frames, current_timestep,
                         f"t={current_timestep}  [PSF] no backup — nominal fallback", "info")
                    tester.real_state_empirical(current_timestep, t_next)
                    current_timestep += 1
        else:
            push(frames, current_timestep,
                 f"t={current_timestep}  [PSF] no valid backup (t_div={t_diverge})"
                 f" — nominal fallback", "info")
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

print(f"Generated {len(frames)} frames")


# ══════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════

RSOA_FILL = {
    "concrete":  {"fc": "#f59e0b", "fa": 0.45, "ec": "#d97706"},
    "symbolic":  {"fc": "#7c3aed", "fa": 0.45, "ec": "#5b21b6"},
    "sampled":   {"fc": "#0891b2", "fa": 0.45, "ec": "#0e7490"},
    "empirical": {"fc": "#059669", "fa": 0.45, "ec": "#047857"},
}
ORIGIN_BORDER = {
    "baseline": {"lw": 1.2, "ls": "--", "hatch": None,  "alpha_boost": 0.0},
    "mpc":      {"lw": 2.0, "ls": "-",  "hatch": None,  "alpha_boost": 0.12},
    "info":     {"lw": 1.2, "ls": "-",  "hatch": None,  "alpha_boost": 0.0},
}
MPC_EXEC_COLOR = {"fc": "#ec4899", "fa": 0.50, "ec": "#be185d"}
MPC_BUF_COLOR  = {"fc": "#f9a8d4", "fa": 0.30, "ec": "#ec4899"}
_BOUND_LIMIT   = 1e6

t_origin: dict = {}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.tick_params(colors="#333")
for spine in ax.spines.values():
    spine.set_color("#aaa")
ax.set_xlabel("position (x)", color="#222", fontsize=13)
ax.set_ylabel("position (y)", color="#222", fontsize=13)
title = ax.set_title("", fontsize=9)

# Axis limits
all_x1, all_x2 = [], []
for snap, _, _, _, _, _, _, _ in frames:
    for t, entry in snap.items():
        all_x1.extend([v for v in entry["x1"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
        all_x2.extend([v for v in entry["x2"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
for obs in obstacles:
    cx, cy, r = obs[0], obs[1], obs[2]
    all_x1.extend([cx - r, cx + r])
    all_x2.extend([cy - r, cy + r])

mg = 0.3
x1_min = min(all_x1) - mg;  x1_max = max(all_x1) + mg
x2_min = min(all_x2) - mg;  x2_max = max(all_x2) + mg
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

for i, obs in enumerate(obstacles):
    cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
    ax.add_patch(patches.Circle(
        (cx, cy), r,
        linewidth=2.0, edgecolor="#b91c1c", facecolor="#ef4444",
        alpha=0.75, zorder=5, label="Obstacle" if i == 0 else None
    ))

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists, t_origin
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    snap, tdiv_traj, tdiv, ct, label, origin, mpc_exec_trail, phase = frames[frame_idx]
    title.set_text(label)

    if origin not in ("info",):
        for t in snap:
            if t not in t_origin:
                t_origin[t] = origin

    # RSOA boxes
    for t, entry in sorted(snap.items()):
        fill = RSOA_FILL.get(entry["type"], RSOA_FILL["concrete"])
        bdr  = ORIGIN_BORDER.get(t_origin.get(t, "baseline"), ORIGIN_BORDER["baseline"])

        x1_lo, x1_hi = entry["x1"]
        x2_lo, x2_hi = entry["x2"]
        if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
            continue
        x1_lo = max(x1_lo, x1_min); x1_hi = min(x1_hi, x1_max)
        x2_lo = max(x2_lo, x2_min); x2_hi = min(x2_hi, x2_max)
        if x1_hi <= x1_lo or x2_hi <= x2_lo:
            continue

        r = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=bdr["lw"], linestyle=bdr["ls"],
            edgecolor=fill["ec"], facecolor=fill["fc"],
            alpha=fill["fa"] + bdr["alpha_boost"],
            hatch=bdr["hatch"], zorder=2
        )
        ax.add_patch(r)
        dynamic_artists.append(r)
        cxb = (x1_lo + x1_hi) / 2; cyb = (x2_lo + x2_hi) / 2
        txt = ax.text(cxb, cyb, str(t), fontsize=6, color=fill["ec"],
                      alpha=0.85, ha="center", va="center", zorder=3,
                      fontfamily="monospace")
        dynamic_artists.append(txt)

    # t_diverge backup trajectory (light pink preview)
    for b in tdiv_traj:
        if not hasattr(b, 'shape') or b.shape[0] < 2:
            continue
        x1_lo, x1_hi = float(b[0, 0]), float(b[0, 1])
        x2_lo, x2_hi = float(b[1, 0]), float(b[1, 1])
        if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
            continue
        r = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=1.0, linestyle="--",
            edgecolor=MPC_BUF_COLOR["ec"], facecolor=MPC_BUF_COLOR["fc"],
            alpha=MPC_BUF_COLOR["fa"], zorder=3
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

    # Accumulated MPC executed trail
    for b in mpc_exec_trail:
        if not hasattr(b, 'shape') or b.shape[0] < 2:
            continue
        x1_lo, x1_hi = float(b[0, 0]), float(b[0, 1])
        x2_lo, x2_hi = float(b[1, 0]), float(b[1, 1])
        if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
            continue
        r = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=1.5, edgecolor=MPC_EXEC_COLOR["ec"],
            facecolor=MPC_EXEC_COLOR["fc"], alpha=MPC_EXEC_COLOR["fa"], zorder=4
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

    # Star at t_diverge
    if tdiv is not None and tdiv in snap:
        entry = snap[tdiv]
        if all(np.isfinite(v) for v in [*entry["x1"], *entry["x2"]]):
            cxs = sum(entry["x1"]) / 2; cys = sum(entry["x2"]) / 2
            star = ax.plot(cxs, cys, "*", color="#ec4899", markersize=14, zorder=7)[0]
            dynamic_artists.append(star)
            lbl = ax.text(cxs + 0.05, cys + 0.06, f"t_div={tdiv}",
                          color="#ec4899", fontsize=8, fontfamily="monospace", zorder=7)
            dynamic_artists.append(lbl)

    # Blue dot at current timestep
    if ct in snap:
        entry = snap[ct]
        if all(np.isfinite(v) for v in [*entry["x1"], *entry["x2"]]):
            cxd = sum(entry["x1"]) / 2; cyd = sum(entry["x2"]) / 2
            dot = ax.plot(cxd, cyd, "o", color="#3b82f6", markersize=8, zorder=6)[0]
            dynamic_artists.append(dot)
            lbl = ax.text(cxd + 0.05, cyd + 0.04, f"t={ct}",
                          color="#3b82f6", fontsize=9, fontfamily="monospace", zorder=6)
            dynamic_artists.append(lbl)

    # State badge — driven by actual simulation phase, not per-frame origin
    if phase == "MPC_ACTIVE":
        badge_color, badge_text = "#ec4899", "● MPC ACTIVE"
    elif phase == "PRE_CONFLICT":
        badge_color, badge_text = "#f59e0b", "● PRE_CONFLICT"
    else:
        badge_color, badge_text = "#22c55e", "● NOMINAL"

    badge = ax.text(
        0.01, 0.97, badge_text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        color=badge_color, va="top", zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=badge_color, lw=1.2)
    )
    dynamic_artists.append(badge)

    return dynamic_artists


legend_els = [
    Patch(facecolor="#f59e0b", alpha=0.3, edgecolor="#f59e0b", label="Concrete RSOA"),
    Patch(facecolor="#8b5cf6", alpha=0.3, edgecolor="#8b5cf6", label="Symbolic RSOA"),
    Patch(facecolor="#10b981", alpha=0.3, edgecolor="#10b981", label="Empirical (Kalman)"),
    Patch(facecolor="#ec4899", alpha=0.50, edgecolor="#ec4899", label="MPC executed"),
    Patch(facecolor="#f9a8d4", alpha=0.30, edgecolor="#ec4899",
          linewidth=0.8, linestyle="--", label="PSF backup (t_diverge)"),
    Patch(facecolor="#ef4444", alpha=0.5,  edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o",  color="#3b82f6", ls="", markersize=6, label="Current t"),
    Line2D([0], [0], marker="*",  color="#ec4899", ls="", markersize=8, label="t_diverge"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="white", edgecolor="#aaa", labelcolor="#222")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

gif_path = "alg12_psf_buffer.gif"
mp4_path = "alg12_psf_buffer.mp4"

print(f"Rendering {len(frames)} frames...")
images = []
for i in range(len(frames)):
    if i % 50 == 0:
        print(f"  {i}/{len(frames)} rendered...")
    update(i)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=50)
    buf.seek(0)
    images.append(imageio.v2.imread(buf))

imageio.mimsave(gif_path, images, duration=150, loop=0)
print(f"Saved GIF → {gif_path}")

writer = imageio.get_writer(mp4_path, fps=30, codec="libx264")
for img in images:
    writer.append_data(img)
writer.close()
print(f"Saved MP4 → {mp4_path}")

plt.close()
