"""
Animation for alg19 (purge) — PSF with Forward MPC Buffer + Split-Terminal MPC.
Backup may route THROUGH the danger region S (path constrained to raw r),
but must end outside S (terminal constrained to S). Saves:
    alg19_psf_buffer.gif
    alg19_psf_buffer.mp4

Usage:
    python anim_alg14.py [seed] [--obs "cx,cy,r" "cx,cy,r" ...]
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter
from alg19_purge import (
    concrete_scan, symbolic_step, RefinementTask,
    build_mpc_backup, psf_valid, extend_mpc_sequence,
    collides_danger, collides_raw, scan_window,
    _purge_infeasible, _max_valid_diverge, _bounds_at,
)
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
import os
import io
import imageio

# ── Config ──
MAX_TIME             = 60
MAX_SYMBOLIC_HORIZON = 5


_DEFAULT_OBSTACLES = [
    np.array([-4.3,  1,  0.5]),
    np.array([-2,  -2,  0.5]),
    np.array([-1,  0.5,  0.5]),
    np.array([-3.5,  0,  0.5]),
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
_obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
print(f"[ANIM] seed={_seed}  n_obs={len(obstacles)}  obs=[{_obs_str}]")
for o in obstacles:
    print(f"  obstacle: center=({o[0]:.3f}, {o[1]:.3f})  r={o[2]}")

analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
tester   = ReachabilityTester(analyzer, obstacles, seed=_seed)
mpc_sf   = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=12,
                                  use_safety_radius=True, split_terminal_D=True)

v_nom = tester.analyzer.cl_system.dynamics.vt
u_max = 1.0
turning_radius = v_nom / u_max

budget = TimeBudget(timestep_budget=0.2)
budget.symbolic_costs = {
    1:  0.05942702293395996,  2: 0.0532071590423584,
    3:  0.12308859825134277,  4: 0.2227306365966797,
    5:  0.3548123836517334,   6: 0.5160810947418213,
    7:  0.7076215744018555,   8: 1.046485185623169,
    9:  1.189185619354248,   10: 1.4745268821716309,
}
budget.concrete_cost = 0.015


# ── Anim/alg14 divergence probe ──
# Monkey-patch _run_mpc_from_bounds to capture the terminal bound of every
# MPC solve — even rejected ones. We render rejected terminals as red dashed
# boxes so the user can see the geometry the rejection is based on.
_pending_terminal_bound = {'value': None}
_rejected_terminals: dict = {}   # tau -> np.ndarray (terminal bound)

_orig_run_mpc = mpc_sf._run_mpc_from_bounds

def _run_mpc_capture(*args, **kwargs):
    result = _orig_run_mpc(*args, **kwargs)
    traj_bounds, _, _ = result
    if traj_bounds is not None and len(traj_bounds) > 0:
        _pending_terminal_bound['value'] = np.array(traj_bounds[-1])
    return result

mpc_sf._run_mpc_from_bounds = _run_mpc_capture


def _anim_build(call_site: str, tau: int):
    """Wrapper that prints input RSOA bounds + seed before invoking
    build_mpc_backup, so anim logs can be diffed against trial logs.
    The wrapped call's [MPC build] τ=... line still prints from inside
    build_mpc_backup itself."""
    h = tester.horizons.get(tau)
    if h is None:
        print(f"  [ANIM probe] {call_site} τ={tau} — NO HORIZON  (seed={_seed})")
        return build_mpc_backup(mpc_sf, tester, mpc_buffer, tau)
    b = h.get_tight_bound()
    if b is None:
        print(f"  [ANIM probe] {call_site} τ={tau} — bound=None  (seed={_seed})")
    else:
        cx = (b[0, 0] + b[0, 1]) / 2.0
        cy = (b[1, 0] + b[1, 1]) / 2.0
        cth = (b[2, 0] + b[2, 1]) / 2.0
        print(f"  [ANIM probe] {call_site} τ={tau}  in_center=({cx:.3f},{cy:.3f},{cth:.3f})"
              f"  x=[{b[0,0]:.3f},{b[0,1]:.3f}]"
              f"  y=[{b[1,0]:.3f},{b[1,1]:.3f}]"
              f"  th=[{b[2,0]:.3f},{b[2,1]:.3f}]  (seed={_seed})")
    _pending_terminal_bound['value'] = None
    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau)
    term = _pending_terminal_bound['value']
    if not valid and term is not None:
        _rejected_terminals[tau] = term
    elif valid:
        _rejected_terminals.pop(tau, None)
    return valid


# ── Frame helpers ──

def snapshot_rsoa(tester):
    snap = {}
    for t, horizon in tester.horizons.items():
        tb = horizon.get_tight_bound()
        if tb is not None:
            calc_type = "empirical"
            real_state = None
            for calc_id in sorted(horizon.calculations.keys()):
                calc = horizon.calculations[calc_id]
                calc_type = calc["calc_type"].value
                if real_state is None and "real_state" in calc:
                    rs = np.asarray(calc["real_state"]).flatten()
                    real_state = (float(rs[0]), float(rs[1]))
            snap[t] = {
                "x1":        (float(tb[0, 0]), float(tb[0, 1])),
                "x2":        (float(tb[1, 0]), float(tb[1, 1])),
                "type":      calc_type,
                "real_state": real_state,  # (x, y) or None
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
    _t0 = time.perf_counter()
    frames.append((
        snapshot_rsoa(tester),
        _tdiv_traj(),          # t_diverge backup trajectory (pink preview)
        t_diverge,             # for star marker
        current_t,
        label,
        origin,
        set(mpc_applied),          # timesteps with MPC control (replaces Kalman RSOA in render)
        _current_phase(),      # actual simulation phase for badge
        {k: v.copy() for k, v in _rejected_terminals.items()},   # tau -> terminal bound
    ))
    budget.exclude_elapsed(time.perf_counter() - _t0)


# ── Module-level PSF state ──
mpc_buffer:        dict            = {}
concrete_until:    int             = 0
mpc_horizon_until: int             = -1
conflict_time:     Optional[int]   = None
t_diverge:         Optional[int]   = None
pending_job:       Optional[RefinementTask] = None
wall_tau:          Optional[int]   = None

mpc_state = {
    'committed_at':  None,
    'conflict_time': None,
    'controls':      [],
    'traj_bounds':   [],
    'needed':        False,
}
mpc_started  = False
# Set of timesteps where MPC control was applied.
# Used to replace the Kalman/green RSOA with a pink one for those steps.
mpc_applied: set = set()


# ── Simulation with frame capture ──
print("Running alg12 PSF buffer...")
frames = []

current_timestep = 0
psf_no_diverge   = 0
psf_queue_empty  = 0
mpc_calls        = 0
push(frames, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()
    t_next   = current_timestep + 1
    wall_tau = None  # reset each iteration; P3 sets it if it hits INFEASIBLE

    # ══ MPC ACTIVE PHASE ═════════════════════════════════════════════════
    if mpc_started:
        ctrl_idx = current_timestep - mpc_state['committed_at']
        queue    = mpc_state['controls']

        # Clear stale horizons; interleave: one concrete step → one MPC build,
        # repeat. First iteration always runs (guarantees mpc_buffer[t_next]).
        for _t in list(tester.horizons.keys()):
            if _t > current_timestep:
                del tester.horizons[_t]
        concrete_until = current_timestep
        while budget.can_afford('concrete'):
            if concrete_until >= MAX_TIME:
                break
            # One concrete step
            collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
            concrete_until += 1
            if collision:
                break
            # One MPC build for this timestep
            if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = _anim_build("[MPC PSF interleave]", concrete_until)
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
            # Reset pending_job to start from the new post-MPC position.
            # The old symbolic chain started from the pre-MPC nominal
            # trajectory; those parent bounds no longer match the MPC-rebuilt
            # horizons ahead, causing empty intersections.  conflict_time is
            # kept — the conflict may still exist; P4 will re-verify from here.
            pending_job = (RefinementTask(t_next, conflict_time)
                           if conflict_time is not None else None)
            # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
            # were built from nominal-trajectory concrete bounds (concrete_scan
            # always uses nominal dynamics) so they remain valid safety backups
            # now that we are reverting to nominal control.
            _purge_infeasible(mpc_buffer, t_next)
            t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep)
            mpc_horizon_until = t_diverge if t_diverge is not None else t_next
            concrete_until             = t_next
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1
            push(frames, current_timestep,
                 f"t={current_timestep}  nominal step (post-revert)  t_div={t_diverge}", "info")
            continue

        # Extend queue whenever <= n_horizon controls remain.
        # Placed after revert check: no point extending a queue we're about to abandon.
        if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
            extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                ctrl_idx=ctrl_idx, budget=budget)
            queue = mpc_state['controls']
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC extend] queue now {len(queue)}", "mpc")

        # Apply MPC control
        if ctrl_idx >= len(queue):
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] ERROR queue exhausted — ABORT", "mpc")
            break
        ctrl = queue[ctrl_idx]
        mpc_applied.add(current_timestep)
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
    # Skip P1 when inside S-region passthrough: MPC would be INFEASIBLE anyway.
    # Saves ~0.034 s so the symbolic budget fires in 2 timesteps instead of 3.
    # Gate reopens as soon as bounds at t+1 exit S.
    _in_passthrough = (t_diverge is not None and t_diverge > t_next)
    _skip_p1 = False
    if _in_passthrough and t_next in tester.horizons:
        _b = tester.horizons[t_next].get_tight_bound()
        if _b is not None and collides_danger(_b, obstacles, turning_radius):
            _skip_p1 = True
    if t_next not in mpc_buffer and not _skip_p1:
        if t_next not in tester.horizons:
            concrete_scan(tester, concrete_until, t_next)
            concrete_until = max(concrete_until, t_next)
        if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
            valid = _anim_build("[P1]", t_next)
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            if valid and (t_diverge is None or t_next > t_diverge):
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
            # scan_ceil is already capped at conflict_time, so ct can
            # never exceed an existing conflict — no < guard needed.
            conflict_time = ct
            if pending_job is None:
                pending_job = RefinementTask(current_timestep, conflict_time)
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
                wall_tau = mpc_horizon_until
                break
            valid = _anim_build("[P3]", buf_t)
            mpc_horizon_until = buf_t
            push(frames, current_timestep,
                 f"t={current_timestep}  [P3] mpc_buffer[{buf_t}]="
                 f"{'VALID' if valid else 'INFEASIBLE'}  t_div={t_diverge}", "mpc")
            if valid:
                t_diverge = buf_t
            else:
                wall_tau = buf_t
                break

    # P3b — Passthrough scan beyond infeasible wall
    # Only invoke when an S-region exists ahead (bounds overlap S but not raw
    # obstacle). Checks all τ in scan range — wall_tau itself may be outside S
    # (n_horizon failure) while the actual S-region starts a few steps later.
    if wall_tau is not None and budget.remaining >= budget.mpc_cost:
        scan_limit = (min(conflict_time - 1, concrete_until)
                      if conflict_time is not None
                      else concrete_until)
        _has_S_ahead = any(
            b is not None
            and collides_danger(b, obstacles, turning_radius)
            and not collides_raw(b, obstacles)
            for t in range(wall_tau, scan_limit + 1)
            for b in [_bounds_at(tester, t)]
        )
        if _has_S_ahead:
            passthrough_td = scan_window(
                tester, mpc_sf, mpc_buffer, obstacles,
                wall_tau, scan_limit, budget, R=turning_radius)
            if passthrough_td is not None:
                t_diverge = passthrough_td
                mpc_horizon_until = passthrough_td
                push(frames, current_timestep,
                     f"t={current_timestep}  [P3b] Passthrough t_div -> {t_diverge}", "mpc")

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
                _purge_infeasible(mpc_buffer, current_timestep)
                # Pull mpc_horizon_until back to t_diverge so P3 re-scans the
                # purged gap with tightened bounds on the next iteration.
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                pending_job       = RefinementTask(current_timestep, conflict_time)
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] confirmed — kept VALID entries"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "mpc")
            else:
                conflict_time = None
                pending_job   = None
                _purge_infeasible(mpc_buffer, current_timestep)
                mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                push(frames, current_timestep,
                     f"t={current_timestep}  [P4] ✓ deconflicted — kept VALID entries"
                     f"  t_div={t_diverge}  mpc_hz={mpc_horizon_until}", "baseline")

    # ══ PSF DECISION ════════════════════════════════════════════════════
    if psf_valid(mpc_buffer, t_next):
        push(frames, current_timestep,
             f"t={current_timestep}  [PSF NOMINAL] PSF OK — mpc_buffer[{t_next}] valid", "info")
        tester.real_state_empirical(current_timestep, t_next)
        current_timestep += 1
        push(frames, current_timestep,
             f"t={current_timestep}  nominal step  t_div={t_diverge}", "info")

    elif (t_diverge is not None and t_diverge > current_timestep
            and psf_valid(mpc_buffer, t_diverge)):
        push(frames, current_timestep,
             f"t={current_timestep}  [PSF PASSTHROUGH] inside S, t_div={t_diverge} ahead"
             f" — nominal safe", "info")
        tester.real_state_empirical(current_timestep, t_next)
        current_timestep += 1
        push(frames, current_timestep,
             f"t={current_timestep}  nominal step (passthrough)  t_div={t_diverge}", "info")

    else:
        if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
            controls, traj_bounds = mpc_buffer[t_diverge]
            mpc_state['committed_at']  = t_diverge
            mpc_state['conflict_time'] = conflict_time
            mpc_state['controls']      = list(controls)
            mpc_state['traj_bounds']   = list(traj_bounds)
            mpc_state['needed']        = True
            mpc_started                = True
            mpc_calls                 += 1

            ctrl_idx = current_timestep - t_diverge
            if ctrl_idx < len(controls):
                ctrl = controls[ctrl_idx]
                mpc_applied.add(current_timestep)
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
                    if ctrl_idx < len(mpc_state['traj_bounds']):
                        mpc_applied[current_timestep] = mpc_state['traj_bounds'][ctrl_idx].copy()
                    push(frames, current_timestep,
                         f"t={current_timestep}  [PSF ACTIVATE extended]"
                         f"  u={np.round(ctrl, 4)}", "mpc")
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    psf_queue_empty += 1
                    push(frames, current_timestep,
                         f"t={current_timestep}  [PSF] no backup — nominal fallback", "info")
                    tester.real_state_empirical(current_timestep, t_next)
                    current_timestep += 1
        else:
            psf_no_diverge += 1
            push(frames, current_timestep,
                 f"t={current_timestep}  [PSF] no valid backup (t_div={t_diverge})"
                 f" — nominal fallback", "info")
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

print(f"{'='*60}")
print(f"Simulation complete at timestep {current_timestep}")
print(f"  mpc_calls={mpc_calls}  psf_no_diverge={psf_no_diverge}"
      f"  psf_queue_empty={psf_queue_empty}")
print(f"{'='*60}")
print(f"Generated {len(frames)} frames")

# ── Collision verdict ────────────────────────────────────────────────────
# The animator otherwise reports no pass/fail, and outcomes are compute
# dependent (the 0.20 s TimeBudget is wall-clock), so a rerun of a seed that
# collided in the batch does not necessarily collide here. Recompute the same
# real-state check alg19_purge.test() does, and let the caller render only the
# runs that actually reproduced the failure.
_hist = []
for _t in sorted(tester.horizons.keys()):
    for _calc in tester.horizons[_t].calculations.values():
        if 'real_state' in _calc:
            _hist.append(np.asarray(_calc['real_state']).flatten())
            break
_hits = [i for i, _s in enumerate(_hist)
         if any((_s[0] - o[0]) ** 2 + (_s[1] - o[1]) ** 2 <= o[2] ** 2
                for o in obstacles)]
print(f"VERDICT seed={_seed} collision={bool(_hits)} at={_hits[:5]} "
      f"steps={len(_hist)} mpc_calls={mpc_calls} psf_no_diverge={psf_no_diverge}")
if os.environ.get("ANIM_ONLY_IF_COLLIDE") == "1" and not _hits:
    print("no collision reproduced -- skipping render")
    sys.exit(0)


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
for snap, _, _, _, _, _, _, _, _ in frames:
    for t, entry in snap.items():
        all_x1.extend([v for v in entry["x1"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
        all_x2.extend([v for v in entry["x2"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
for obs in obstacles:
    cx, cy, r = obs[0], obs[1], obs[2]
    D = float(np.sqrt(r ** 2 + 2 * r * turning_radius))
    all_x1.extend([cx - D, cx + D])
    all_x2.extend([cy - D, cy + D])

mg = 0.7
x1_min = min(all_x1) - mg;  x1_max = max(all_x1) + mg
x2_min = min(all_x2) - mg;  x2_max = max(all_x2) + mg
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

for i, obs in enumerate(obstacles):
    cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
    D = float(np.sqrt(r ** 2 + 2 * r * turning_radius))
    # Danger region D — inflated radius where avoidance can't be guaranteed.
    ax.add_patch(patches.Circle(
        (cx, cy), D,
        linewidth=1.5, edgecolor="#f97316", facecolor="#fed7aa",
        linestyle="--", alpha=0.35, zorder=4,
        label="Danger D" if i == 0 else None
    ))
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

    (snap, tdiv_traj, tdiv, ct, label, origin, mpc_applied_snap, phase,
     rejected_terms) = frames[frame_idx]
    title.set_text(label)

    if origin not in ("info",):
        for t in snap:
            if t not in t_origin:
                t_origin[t] = origin

    # RSOA boxes
    for t, entry in sorted(snap.items()):
        x1_lo_s, x1_hi_s = entry["x1"]
        x2_lo_s, x2_hi_s = entry["x2"]

        # MPC-controlled timesteps: draw pink RSOA using the actual snap bounds
        # (same size as the Kalman box — avoids inflation from traj_bounds propagation).
        if t in mpc_applied_snap:
            x1_lo, x1_hi = x1_lo_s, x1_hi_s
            x2_lo, x2_hi = x2_lo_s, x2_hi_s
            if all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
                x1_lo = max(x1_lo, x1_min); x1_hi = min(x1_hi, x1_max)
                x2_lo = max(x2_lo, x2_min); x2_hi = min(x2_hi, x2_max)
                if x1_hi > x1_lo and x2_hi > x2_lo:
                    r = patches.Rectangle(
                        (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
                        linewidth=1.5, linestyle="-",
                        edgecolor=MPC_EXEC_COLOR["ec"], facecolor=MPC_EXEC_COLOR["fc"],
                        alpha=MPC_EXEC_COLOR["fa"], zorder=2,
                    )
                    ax.add_patch(r)
                    dynamic_artists.append(r)
                    txt = ax.text((x1_lo + x1_hi) / 2, (x2_lo + x2_hi) / 2, str(t),
                                  fontsize=6, color=MPC_EXEC_COLOR["ec"],
                                  alpha=0.85, ha="center", va="center", zorder=3,
                                  fontfamily="monospace")
                    dynamic_artists.append(txt)
            continue  # skip normal Kalman rendering for this timestep

        fill = RSOA_FILL.get(entry["type"], RSOA_FILL["concrete"])
        bdr  = ORIGIN_BORDER.get(t_origin.get(t, "baseline"), ORIGIN_BORDER["baseline"])

        x1_lo, x1_hi = x1_lo_s, x1_hi_s
        x2_lo, x2_hi = x2_lo_s, x2_hi_s
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

    # Rejected MPC terminal boxes (red dashed) — what MPC tried that hit D
    for tau, tb in sorted(rejected_terms.items()):
        x1_lo, x1_hi = float(tb[0, 0]), float(tb[0, 1])
        x2_lo, x2_hi = float(tb[1, 0]), float(tb[1, 1])
        if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
            continue
        rect = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=1.6, linestyle="--",
            edgecolor="#dc2626", facecolor="none",
            alpha=0.9, zorder=6,
        )
        ax.add_patch(rect)
        dynamic_artists.append(rect)
        cxr = (x1_lo + x1_hi) / 2; cyr = (x2_lo + x2_hi) / 2
        lbl = ax.text(cxr, cyr, f"τ{tau}✗",
                      fontsize=6.5, color="#dc2626", ha="center", va="center",
                      fontfamily="monospace", alpha=0.95, zorder=6)
        dynamic_artists.append(lbl)

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

    # Real-state dot at current timestep (green = real position; blue = RSOA center fallback)
    if ct in snap:
        entry = snap[ct]
        rs = entry.get("real_state")
        if rs is not None:
            cxd, cyd = rs[0], rs[1]
            dot_color = "#16a34a"   # green — actual Kalman state
        elif all(np.isfinite(v) for v in [*entry["x1"], *entry["x2"]]):
            cxd = sum(entry["x1"]) / 2; cyd = sum(entry["x2"]) / 2
            dot_color = "#3b82f6"   # blue — RSOA center (no real state available)
        else:
            cxd = cyd = None
        if cxd is not None:
            dot = ax.plot(cxd, cyd, "o", color=dot_color, markersize=8, zorder=6)[0]
            dynamic_artists.append(dot)
            lbl = ax.text(cxd + 0.05, cyd + 0.04, f"t={ct}",
                          color=dot_color, fontsize=9, fontfamily="monospace", zorder=6)
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
    Patch(facecolor="#fed7aa", alpha=0.5,  edgecolor="#f97316", linestyle="--", label="Danger D"),
    Patch(facecolor="none", edgecolor="#dc2626", linewidth=1.6, linestyle="--",
          label="Rejected MPC terminal"),
    Line2D([0], [0], marker="o",  color="#16a34a", ls="", markersize=6, label="Real state (t)"),
    Line2D([0], [0], marker="*",  color="#ec4899", ls="", markersize=8, label="t_diverge"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="white", edgecolor="#aaa", labelcolor="#222")

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

# Fixed filenames would clobber each other when rendering several seeds in a
# row, so let the caller name the output:  ANIM_OUT=/path/case_<seed>
_out = os.environ.get("ANIM_OUT", "alg19_psf_buffer")
gif_path = _out + ".gif"
mp4_path = _out + ".mp4"

print(f"Rendering {len(frames)} frames...")
images = []
for i in range(len(frames)):
    if i % 50 == 0:
        print(f"  {i}/{len(frames)} rendered...")
    update(i)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
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
