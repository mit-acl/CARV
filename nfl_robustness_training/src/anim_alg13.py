"""
Animation for alg13 — DI Predictive Safety Filter (PSF)
"""


from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter
from alg12_mpc_every_timestep import (
    concrete_scan, symbolic_step, VerificationTask,
    build_mpc_backup, psf_valid, extend_mpc_sequence,
    _purge_infeasible,
)
from alg13_di import HalfPlaneConstraints, real_state_violates
from time_budget import TimeBudget
import time
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

# ── Config ──cd ../
MAX_TIME             = 35
MAX_SYMBOLIC_HORIZON = 5

# DI constraint set / terminal-set parameters
POS_MIN = 0.0
VEL_MIN = -0.75
BUFFER  = 0.05

# Use the DoubleIntegrator default init range from setup_analyzer
# ([2.5, 3.0] × [-0.25, 0.25]) unless overridden on the command line.
_args = sys.argv[1:]
_seed = 1401830092
if _args and not _args[0].startswith('--'):
    _seed = int(_args.pop(0))

init_range = None
if '--init' in _args:
    i = _args.index('--init')
    vals = [float(v) for v in _args[i + 1].split(',')]
    init_range = np.array([[vals[0], vals[1]], [vals[2], vals[3]]])

print(f"Setting up analyzer...  seed={_seed}")
if init_range is not None:
    print(f"  init range: p=[{init_range[0,0]:.2f},{init_range[0,1]:.2f}],"
          f" v=[{init_range[1,0]:.2f},{init_range[1,1]:.2f}]")
else:
    print(f"  init range: default ([2.5, 3.0] × [-0.25, 0.25])")

analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz',
                          init_range=init_range)
tester   = ReachabilityTester(analyzer, [], seed=_seed)
tester.obstacles = HalfPlaneConstraints(pos_min=POS_MIN, vel_min=VEL_MIN)
mpc_sf   = make_mpc_safety_filter(
    tester, obstacles_list=[], n_horizon=12,
    use_half_plane=True, pos_min=POS_MIN, vel_min=VEL_MIN, buffer=BUFFER,
)

budget = TimeBudget(timestep_budget=0.20)
budget.symbolic_costs = {
    1:  0.05942702293395996,  2: 0.0532071590423584,
    3:  0.12308859825134277,  4: 0.2227306365966797,
    5:  0.3548123836517334,   6: 0.5160810947418213,
    7:  0.7076215744018555,   8: 1.046485185623169,
    9:  1.189185619354248,   10: 1.4745268821716309,
}
budget.concrete_cost = 0.0142


# ── Frame helpers ──

def snapshot_rsoa(tester):
    snap = {}
    for t, horizon in tester.horizons.items():
        tb = horizon.get_tight_bound()
        if tb is None:
            continue
        calc_type = "empirical"
        real_state = None
        for calc_id in sorted(horizon.calculations.keys()):
            calc = horizon.calculations[calc_id]
            calc_type = calc["calc_type"].value
            if real_state is None and "real_state" in calc:
                rs = np.asarray(calc["real_state"]).flatten()
                real_state = (float(rs[0]), float(rs[1]))
        snap[t] = {
            "p":         (float(tb[0, 0]), float(tb[0, 1])),
            "v":         (float(tb[1, 0]), float(tb[1, 1])),
            "type":      calc_type,
            "real_state": real_state,
        }
    return snap


def _tdiv_traj():
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
    _t0 = time.perf_counter()
    frames.append((
        snapshot_rsoa(tester),
        _tdiv_traj(),
        t_diverge,
        current_t,
        label,
        origin,
        list(mpc_traj_executed),
        _current_phase(),
    ))
    budget.exclude_elapsed(time.perf_counter() - _t0)


# ── Module-level PSF state ──
mpc_buffer:        dict          = {}
concrete_until:    int           = 0
mpc_horizon_until: int           = -1
conflict_time:     Optional[int] = None
t_diverge:         Optional[int] = None
pending_job:       Optional[VerificationTask] = None
wall_tau:          Optional[int] = None

mpc_state = {
    'committed_at':  None,
    'conflict_time': None,
    'controls':      [],
    'traj_bounds':   [],
    'needed':        False,
}
mpc_started       = False
mpc_traj_executed = []


# ── Simulation with frame capture ──
print("Running alg13 PSF (DI parabolic invariant)...")
frames = []

current_timestep = 0
push(frames, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()
    t_next   = current_timestep + 1
    wall_tau = None

    # ══ MPC ACTIVE PHASE ════════════════════════════════════════════════
    if mpc_started:
        ctrl_idx = current_timestep - mpc_state['committed_at']
        queue    = mpc_state['controls']

        if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
            extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                ctrl_idx=ctrl_idx, budget=budget)
            queue = mpc_state['controls']
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC extend] queue now {len(queue)}", "mpc")

        for _t in list(tester.horizons.keys()):
            if _t > current_timestep:
                del tester.horizons[_t]
        concrete_until = current_timestep
        while budget.can_afford('concrete'):
            if concrete_until >= MAX_TIME:
                break
            collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
            concrete_until += 1
            if collision:
                break
            if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until)
                mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                if valid:
                    t_diverge = concrete_until
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC PSF] τ={concrete_until}"
                     f"  {'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")
                if not valid:
                    break
            else:
                break

        if psf_valid(mpc_buffer, t_next):
            push(frames, current_timestep,
                 f"t={current_timestep}  [PSF REVERT] mpc_buffer[{t_next}] valid"
                 f"  mpc_hz={mpc_horizon_until} — back to nominal", "info")
            mpc_started                = False
            mpc_state['needed']        = False
            mpc_state['committed_at']  = None
            mpc_state['conflict_time'] = None
            pending_job = (VerificationTask(t_next, conflict_time)
                           if conflict_time is not None else None)
            _purge_infeasible(mpc_buffer, t_next)
            mpc_horizon_until = t_diverge if t_diverge is not None else t_next
            concrete_until    = t_next
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1
            push(frames, current_timestep,
                 f"t={current_timestep}  nominal step (post-revert)  t_div={t_diverge}", "info")
            continue

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

    # ══ NOMINAL / PRE_CONFLICT PHASE ═══════════════════════════════════
    if conflict_time is not None and current_timestep > conflict_time:
        conflict_time = None
        pending_job   = None

    phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
    push(frames, current_timestep,
         f"t={current_timestep}  concrete_until={concrete_until}"
         f"  mpc_hz={mpc_horizon_until}  conflict={conflict_time}"
         f"  t_div={t_diverge}  [{phase}]", "info")

    # P1 — always run (no passthrough concept in DI)
    if t_next not in mpc_buffer:
        if t_next not in tester.horizons:
            concrete_scan(tester, concrete_until, t_next)
            concrete_until = max(concrete_until, t_next)
        if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
            valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next)
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            if valid and (t_diverge is None or t_next > t_diverge):
                t_diverge = t_next
            push(frames, current_timestep,
                 f"t={current_timestep}  [P1] mpc_buffer[{t_next}]="
                 f"{'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")

    # P2 — concrete forward
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
                 f"t={current_timestep}  [P2] concrete →{ct}  ⚠ conflict@{conflict_time}",
                 "baseline")
            break
        concrete_until = end
        push(frames, current_timestep,
             f"t={current_timestep}  [P2] concrete →{concrete_until}  ✓ clear", "baseline")

    # P3 — MPC buffer forward toward conflict_time (no P3b passthrough)
    if conflict_time is not None:
        mpc_horizon_until = max(mpc_horizon_until, current_timestep)
        while (mpc_horizon_until < conflict_time - 1
               and budget.remaining >= budget.mpc_cost):
            buf_t = mpc_horizon_until + 1
            if buf_t not in tester.horizons:
                break
            if mpc_buffer.get(mpc_horizon_until) is None:
                wall_tau = mpc_horizon_until
                break
            valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, buf_t)
            mpc_horizon_until = buf_t
            push(frames, current_timestep,
                 f"t={current_timestep}  [P3] mpc_buffer[{buf_t}]="
                 f"{'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")
            if valid:
                t_diverge = buf_t
            else:
                wall_tau = buf_t
                break

    # P4 — symbolic
    if pending_job is not None and budget.can_afford('symbolic', 1):
        sym_from = pending_job.symbolic_start
        pending_job, result = symbolic_step(tester, pending_job,
                                            MAX_SYMBOLIC_HORIZON, budget)
        push(frames, current_timestep,
             f"t={current_timestep}  [P4] symbolic {sym_from}→{pending_job.symbolic_start}",
             "baseline")
        if result is not None:
            if result["collision"]:
                _purge_infeasible(mpc_buffer, current_timestep)
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                pending_job       = VerificationTask(current_timestep, conflict_time)
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] confirmed — kept VALID entries"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "mpc")
            else:
                conflict_time = None
                pending_job   = None
                _purge_infeasible(mpc_buffer, current_timestep)
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] ✓ deconflicted"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "baseline")

    # ══ PSF DECISION ═══════════════════════════════════════════════════
    if psf_valid(mpc_buffer, t_next):
        push(frames, current_timestep,
             f"t={current_timestep}  [PSF NOMINAL] OK — mpc_buffer[{t_next}] valid", "info")
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
    "baseline": {"lw": 1.2, "ls": "--", "alpha_boost": 0.0},
    "mpc":      {"lw": 2.0, "ls": "-",  "alpha_boost": 0.12},
    "info":     {"lw": 1.2, "ls": "-",  "alpha_boost": 0.0},
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
ax.set_xlabel("position  p", color="#222", fontsize=13)
ax.set_ylabel("velocity  v", color="#222", fontsize=13)
title = ax.set_title("", fontsize=9)

# Axis limits from data + constraints
all_p, all_v = [], []
for snap, _, _, _, _, _, _, _ in frames:
    for _, e in snap.items():
        all_p.extend([v for v in e["p"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
        all_v.extend([v for v in e["v"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])

# Ensure the half-plane / parabolic boundaries are visible.
all_p.extend([POS_MIN - 0.3, POS_MIN + 0.3])
all_v.extend([VEL_MIN - 0.3, VEL_MIN + 0.3, 0.5])

mg = 0.2
p_min = min(all_p) - mg;  p_max = max(all_p) + mg
v_min_ax = min(all_v) - mg;  v_max_ax = max(all_v) + mg
ax.set_xlim(p_min, p_max)
ax.set_ylim(v_min_ax, v_max_ax)

# ── Static constraint geometry ─────────────────────────────────────────
# Raw unsafe half-planes (red)
ax.add_patch(patches.Rectangle(
    (p_min, v_min_ax), POS_MIN - p_min, v_max_ax - v_min_ax,
    facecolor="#ef4444", alpha=0.30, edgecolor="none", zorder=1,
))
ax.add_patch(patches.Rectangle(
    (p_min, v_min_ax), p_max - p_min, VEL_MIN - v_min_ax,
    facecolor="#ef4444", alpha=0.30, edgecolor="none", zorder=1,
))
ax.axvline(POS_MIN, color="#b91c1c", lw=2.0, zorder=2)
ax.axhline(VEL_MIN, color="#b91c1c", lw=2.0, zorder=2)

# Parabolic invariant boundary: p = 0.5 * max(0, -v)^2
# For v >= 0 the curve collapses to p = 0 (already drawn as the
# vertical raw line). For v < 0 it sweeps right as v decreases.
_v_grid = np.linspace(min(0.0, v_min_ax), v_min_ax, 200)
_p_para = 0.5 * np.maximum(0.0, -_v_grid) ** 2 + POS_MIN
ax.plot(_p_para, _v_grid, color="#f97316", lw=2.0,
        linestyle="--", zorder=3, label="Invariant boundary")
# Shade the kinematic-unsafe strip (between parabola and raw p=0).
ax.fill_betweenx(_v_grid, POS_MIN, _p_para,
                 facecolor="#fdba74", alpha=0.30, zorder=1)

# Buffered terminal boundary (where MPC actually plans to)
_p_term = _p_para + BUFFER
ax.plot(_p_term, _v_grid, color="#f97316", lw=1.0,
        linestyle=":", alpha=0.7, zorder=3)

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists, t_origin
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    snap, tdiv_traj, tdiv, ct, label, origin, mpc_exec_trail, phase = frames[frame_idx]
    title.set_text(label)

    if origin != "info":
        for t in snap:
            if t not in t_origin:
                t_origin[t] = origin

    # RSOA boxes
    for t, entry in sorted(snap.items()):
        fill = RSOA_FILL.get(entry["type"], RSOA_FILL["concrete"])
        bdr  = ORIGIN_BORDER.get(t_origin.get(t, "baseline"), ORIGIN_BORDER["baseline"])

        p_lo, p_hi = entry["p"]
        v_lo, v_hi = entry["v"]
        if not all(np.isfinite(x) and abs(x) < _BOUND_LIMIT
                   for x in [p_lo, p_hi, v_lo, v_hi]):
            continue
        p_lo = max(p_lo, p_min); p_hi = min(p_hi, p_max)
        v_lo = max(v_lo, v_min_ax); v_hi = min(v_hi, v_max_ax)
        if p_hi <= p_lo or v_hi <= v_lo:
            continue

        r = patches.Rectangle(
            (p_lo, v_lo), p_hi - p_lo, v_hi - v_lo,
            linewidth=bdr["lw"], linestyle=bdr["ls"],
            edgecolor=fill["ec"], facecolor=fill["fc"],
            alpha=fill["fa"] + bdr["alpha_boost"], zorder=4,
        )
        ax.add_patch(r)
        dynamic_artists.append(r)
        cxb = (p_lo + p_hi) / 2; cyb = (v_lo + v_hi) / 2
        txt = ax.text(cxb, cyb, str(t), fontsize=6, color=fill["ec"],
                      alpha=0.85, ha="center", va="center", zorder=5,
                      fontfamily="monospace")
        dynamic_artists.append(txt)

    # PSF backup preview
    for b in tdiv_traj:
        if not hasattr(b, 'shape') or b.shape[0] < 2:
            continue
        p_lo, p_hi = float(b[0, 0]), float(b[0, 1])
        v_lo, v_hi = float(b[1, 0]), float(b[1, 1])
        if not all(np.isfinite(x) and abs(x) < _BOUND_LIMIT
                   for x in [p_lo, p_hi, v_lo, v_hi]):
            continue
        r = patches.Rectangle(
            (p_lo, v_lo), p_hi - p_lo, v_hi - v_lo,
            linewidth=1.0, linestyle="--",
            edgecolor=MPC_BUF_COLOR["ec"], facecolor=MPC_BUF_COLOR["fc"],
            alpha=MPC_BUF_COLOR["fa"], zorder=5,
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

    # Accumulated MPC-executed trail
    for b in mpc_exec_trail:
        if not hasattr(b, 'shape') or b.shape[0] < 2:
            continue
        p_lo, p_hi = float(b[0, 0]), float(b[0, 1])
        v_lo, v_hi = float(b[1, 0]), float(b[1, 1])
        if not all(np.isfinite(x) and abs(x) < _BOUND_LIMIT
                   for x in [p_lo, p_hi, v_lo, v_hi]):
            continue
        r = patches.Rectangle(
            (p_lo, v_lo), p_hi - p_lo, v_hi - v_lo,
            linewidth=1.5,
            edgecolor=MPC_EXEC_COLOR["ec"], facecolor=MPC_EXEC_COLOR["fc"],
            alpha=MPC_EXEC_COLOR["fa"], zorder=6,
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

    # Star at t_diverge
    if tdiv is not None and tdiv in snap:
        e = snap[tdiv]
        if all(np.isfinite(v) for v in [*e["p"], *e["v"]]):
            cxs = sum(e["p"]) / 2; cys = sum(e["v"]) / 2
            star = ax.plot(cxs, cys, "*", color="#ec4899", markersize=14, zorder=8)[0]
            dynamic_artists.append(star)
            lbl = ax.text(cxs + 0.04, cys + 0.05, f"t_div={tdiv}",
                          color="#ec4899", fontsize=8, fontfamily="monospace", zorder=8)
            dynamic_artists.append(lbl)

    # Real-state dot
    if ct in snap:
        e  = snap[ct]
        rs = e.get("real_state")
        if rs is not None:
            cxd, cyd = rs[0], rs[1]
            dot_color = "#16a34a"
        elif all(np.isfinite(v) for v in [*e["p"], *e["v"]]):
            cxd = sum(e["p"]) / 2; cyd = sum(e["v"]) / 2
            dot_color = "#3b82f6"
        else:
            cxd = cyd = None
        if cxd is not None:
            dot = ax.plot(cxd, cyd, "o", color=dot_color, markersize=8, zorder=7)[0]
            dynamic_artists.append(dot)
            lbl = ax.text(cxd + 0.03, cyd + 0.03, f"t={ct}",
                          color=dot_color, fontsize=9, fontfamily="monospace", zorder=7)
            dynamic_artists.append(lbl)

    # Phase badge
    if phase == "MPC_ACTIVE":
        badge_color, badge_text = "#ec4899", "● MPC ACTIVE"
    elif phase == "PRE_CONFLICT":
        badge_color, badge_text = "#f59e0b", "● PRE_CONFLICT"
    else:
        badge_color, badge_text = "#22c55e", "● NOMINAL"

    badge = ax.text(
        0.01, 0.97, badge_text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        color=badge_color, va="top", zorder=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=badge_color, lw=1.2)
    )
    dynamic_artists.append(badge)
    return dynamic_artists


legend_els = [
    Patch(facecolor="#ef4444", alpha=0.30, edgecolor="#b91c1c", label="Raw unsafe (p≤0, v≤-1)"),
    Patch(facecolor="#fdba74", alpha=0.30, edgecolor="#f97316",
          linewidth=0.8, linestyle="--", label="Kinematic-unsafe (below parabola)"),
    Patch(facecolor="#f59e0b", alpha=0.45, edgecolor="#d97706", label="Concrete RSOA"),
    Patch(facecolor="#7c3aed", alpha=0.45, edgecolor="#5b21b6", label="Symbolic RSOA"),
    Patch(facecolor="#059669", alpha=0.45, edgecolor="#047857", label="Empirical (Kalman)"),
    Patch(facecolor=MPC_EXEC_COLOR["fc"], alpha=0.50,
          edgecolor=MPC_EXEC_COLOR["ec"], label="MPC executed"),
    Patch(facecolor=MPC_BUF_COLOR["fc"],  alpha=0.30,
          edgecolor=MPC_BUF_COLOR["ec"],  linewidth=0.8, linestyle="--",
          label="PSF backup (t_diverge)"),
    Line2D([0], [0], marker="o",  color="#16a34a", ls="", markersize=6, label="Real state (t)"),
    Line2D([0], [0], marker="*",  color="#ec4899", ls="", markersize=8, label="t_diverge"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="white", edgecolor="#aaa", labelcolor="#222")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

gif_path = "alg13_di_psf.gif"
mp4_path = "alg13_di_psf.mp4"

print(f"Rendering {len(frames)} frames...")
images = []
for i in range(len(frames)):
    if i % 50 == 0:
        print(f"  {i}/{len(frames)} rendered...")
    update(i)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
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
