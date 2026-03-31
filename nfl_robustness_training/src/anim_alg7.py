"""
Animation for alg7 — alg6 optimizer with MPC safety filter.
Saves: alg7_optimization_mpc.gif

Frame tuple: (rsoa_snap, mpc_traj_bounds, mpc_t_back, current_t, label, origin)
  mpc_traj_bounds: list of bounds arrays for current MPC attempt (empty if not an MPC frame)
  mpc_t_back:      the lookback timestep being attempted (None if not an MPC frame)
  origin:          "baseline" | "optimizer" | "mpc" | "info"
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter import make_mpc_safety_filter
import time
from alg7_optimization_w_mpc import (
    concrete_scan, symbolic_step, VerificationTask,
    _get_volume, get_dynamic_symbolic_horizon
)
from time_budget import TimeBudget
from multi_obj_opt.strategies import ExtensionOptimizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional

# ── Config ──
MAX_TIME             = 40
MIN_SAFE_HORIZON     = 6
MIN_LOOKAHEAD        = 4
MAX_SYMBOLIC_HORIZON = 10
SYMBOLIC_BUFFER      = 5

obstacles = [
    np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),
    np.array([[-np.inf, 0.3],    [-np.inf, np.inf]]),
]

# obstacles = [
#     np.array([[-5.5, -5], [2, 2.2 ], [-np.inf, np.inf]]),
# ]

# ── Setup ──
print("Setting up analyzer...")

# analyzer           = setup_analyzer('Unicycle_NL', 'natural_none_default')
# tester             = ReachabilityTester(analyzer, obstacles)
# tester_calibration = ReachabilityTester(analyzer)
# mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, t_step=0.1)


analyzer           = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
tester             = ReachabilityTester(analyzer, obstacles)
tester_calibration = ReachabilityTester(analyzer)
mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, t_step=0.1, n_horizon=10)

budget = TimeBudget(timestep_budget=1.0)
print("Calibrating time budget...")
budget.calibrate(tester_calibration, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                 max_backward_horizon=0)

ext_optimizer = ExtensionOptimizer()
ext_optimizer.set_timing_params({
    "concrete_slope":     0.006311738129818,
    "concrete_intercept": 0.0035703865687052,
    "ratio_slope":        0.5341915550605524,
    "ratio_intercept":   -0.0578267576662421,
})


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


def push(frames, current_t, label, origin="info", mpc_traj=None, mpc_t_back=None):
    frames.append((snapshot_rsoa(tester), mpc_traj or [], mpc_t_back,
                   current_t, label, origin, list(mpc_traj_bounds_all)))


# ── MPC state (module-level, mirrors alg7 mpc_state dict) ──
mpc_committed_at     = None
mpc_conflict_time    = None
mpc_control_queue    = []
mpc_traj_bounds_all  = []
mpc_needed           = False
mpc_started          = False
_mpc_trail           = []
_mpc_trail_frame_idx = []


def apply_mpc_filter_frames(frames, conflict_time, current_timestep):
    """Run MPC filter, capturing one frame per lookback attempt."""
    global mpc_committed_at, mpc_conflict_time, mpc_control_queue
    global mpc_traj_bounds_all, mpc_needed

    for lookback in range(2, mpc_sf.max_lookback + 1):
        t_back = conflict_time - lookback
        if t_back < 0:
            break
        if t_back not in tester.horizons:
            continue
        bounds_at_back = tester.horizons[t_back].get_tight_bound()
        if bounds_at_back is None:
            continue

        center = (bounds_at_back[:, 0] + bounds_at_back[:, 1]) / 2.0

        try:
            t0 = time.perf_counter()
            traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(bounds_at_back, center)
            print(f"  [MPC timing] t_back={t_back} _run_mpc_from_bounds took {time.perf_counter() - t0:.3f}s")
        except Exception as e:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] failed at t_back={t_back}: {e}", "mpc")
            continue

        collision_found = False
        collision_step  = None
        for step, b in enumerate(traj_bounds[1:], start=1):
            if mpc_sf._collides(b):
                collision_found = True
                collision_step  = step
                break

        label = (
            f"t={current_timestep}  [MPC] t_back={t_back} — "
            + (f"collision at step {collision_step}, going further back"
               if collision_found else f"SAFE — committing t_back={t_back}")
        )
        push(frames, current_timestep, label, "mpc",
             mpc_traj=traj_bounds, mpc_t_back=t_back)

        if not collision_found:
            if mpc_committed_at is None or t_back >= mpc_committed_at:
                mpc_committed_at    = t_back
                mpc_conflict_time   = conflict_time
                mpc_control_queue   = controls
                mpc_traj_bounds_all = traj_bounds
                mpc_needed          = True
                print(f"  [MPC COMMIT] t_back={t_back}")
            else:
                print(f"  [MPC] Keeping existing plan at t={mpc_committed_at}")
            return t_back

    push(frames, current_timestep,
         f"t={current_timestep}  [MPC] no safe timestep — fallback to t={conflict_time - 1}",
         "mpc")
    return conflict_time - 1


# ── Instrumented helpers ──

def try_extend_anim(frames, validated_until, max_time, budget,
                    max_symbolic_horizon, current_timestep, min_lookahead,
                    origin="baseline", safe_horizon_ceiling=None):
    scan_ceiling = safe_horizon_ceiling if safe_horizon_ceiling is not None else max_time
    vu = validated_until

    while vu < scan_ceiling:
        if not budget.can_afford('concrete'):
            push(frames, current_timestep,
                 f"t={current_timestep}  [extend] budget exhausted", origin)
            break

        end_check_time = min(vu + budget.max_affordable_concrete(), scan_ceiling, max_time)
        result         = tester.concrete(vu, end_check_time)
        collision      = result["collision"]
        conflict_time  = result.get("collision_timestep")
        push(frames, current_timestep,
             f"t={current_timestep}  [extend] concrete {vu}→{end_check_time}"
             + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
             origin)

        if not collision:
            vu = end_check_time
            break

        if not budget.can_afford('symbolic', 1):
            vu = conflict_time - 1
            return vu, VerificationTask(symbolic_start=conflict_time - 1,
                                        conflict_time=conflict_time)

        job       = VerificationTask(symbolic_start=vu, conflict_time=conflict_time)
        sym_start = job.symbolic_start
        job, result_s = symbolic_step(tester, job, max_symbolic_horizon,
                                      budget.max_affordable_symbolic())
        push(frames, current_timestep,
             f"t={current_timestep}  [extend] symbolic {sym_start}→{job.symbolic_start}",
             origin)

        if result_s is None:
            vu = job.symbolic_start
            return vu, job

        if result_s["collision"]:
            push(frames, current_timestep,
                 f"t={current_timestep}  [extend] conflict confirmed@{conflict_time} — MPC filter",
                 origin)
            vu = max(vu, conflict_time - 1)
            apply_mpc_filter_frames(frames, conflict_time, current_timestep)
            return vu, None
        else:
            vu = conflict_time
            push(frames, current_timestep,
                 f"t={current_timestep}  [extend] ✓ deconflicted vu={vu}", origin)

    return vu, None


def optimized_step_anim(frames, validated_until, max_time, budget,
                        max_symbolic_horizon, current_timestep, min_lookahead,
                        ext_optimizer):
    target = validated_until + 1
    if target > max_time:
        return validated_until, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    method = ext_optimizer.get_strategy(
        current_timestep=current_timestep,
        verified_until=validated_until,
        current_vol=current_vol,
        verified_vol=verified_vol,
        time_budget=budget.remaining,
    )
    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] → {method}  "
         f"cur_vol={current_vol:.3f}  ver_vol={verified_vol:.3f}",
         "optimizer")

    if method == "concrete":
        result        = tester.concrete(validated_until, target)
        collision     = result["collision"]
        conflict_time = result.get("collision_timestep")
        push(frames, current_timestep,
             f"t={current_timestep}  [OPT] concrete {validated_until}→{target}"
             + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
             "optimizer")
    else:
        full_span     = target - current_timestep
        actual_k      = min(full_span, max_symbolic_horizon)
        target        = current_timestep + actual_k
        result        = tester.symbolic(current_timestep, target)
        collision     = result["collision"]
        conflict_time = result.get("collision_timestep")
        push(frames, current_timestep,
             f"t={current_timestep}  [OPT] symbolic {current_timestep}→{target} (span={actual_k})"
             + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clean"),
             "optimizer")

    if not collision:
        push(frames, current_timestep,
             f"t={current_timestep}  [OPT] clean — vu={target}", "optimizer")
        return target, None

    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] collision@{conflict_time} — deconflicting from t={current_timestep}",
         "optimizer")

    if not budget.can_afford('symbolic', 1):
        return conflict_time - 1, VerificationTask(
            symbolic_start=conflict_time - 1, conflict_time=conflict_time)

    job       = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    sym_start = job.symbolic_start
    job, result_s = symbolic_step(tester, job, max_symbolic_horizon,
                                  budget.max_affordable_symbolic())
    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] deconflict symbolic {sym_start}→{job.symbolic_start}",
         "optimizer")

    if result_s is None:
        return job.symbolic_start, job

    if result_s["collision"]:
        push(frames, current_timestep,
             f"t={current_timestep}  [OPT] conflict confirmed@{conflict_time} — MPC filter",
             "optimizer")
        apply_mpc_filter_frames(frames, conflict_time, current_timestep)
        return conflict_time - 1, None

    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] deconflicted vu={conflict_time}", "optimizer")
    return conflict_time, None


# ══════════════════════════════════════════════
# Run alg7, capturing frames
# ══════════════════════════════════════════════
print("Running alg7...")
frames = []

current_timestep = 0
validated_until  = 0
pending_job: Optional[VerificationTask] = None

push(frames, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()

    safety_margin = validated_until - current_timestep
    dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
        safety_margin,
        max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
        symbolic_buffer=SYMBOLIC_BUFFER
    )

    # ── MPC locked ────────────────────────────────────────────────────
    if mpc_started:
        ctrl_idx = current_timestep - mpc_committed_at
        queue    = mpc_control_queue
        if ctrl_idx < len(queue):
            ctrl = queue[ctrl_idx]
            _mpc_trail.append(mpc_traj_bounds_all[ctrl_idx].copy())
            _mpc_trail_frame_idx.append(len(frames))
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] idx={ctrl_idx}  u={np.round(ctrl, 4)}"
                 f"  (plan from t={mpc_committed_at})",
                 "mpc", mpc_t_back=mpc_committed_at)
            current_timestep += 1
        else:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] queue exhausted — done", "info")
            break
        continue

    mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
    push(frames, current_timestep,
         f"t={current_timestep}  vu={validated_until}  margin={safety_margin}  [{mode}]",
         "info")

    # ── Phase 1: carry-over ────────────────────────────────────────────
    if pending_job is not None:
        sym_from = pending_job.symbolic_start
        pending_job, result = symbolic_step(tester, pending_job,
                                            dynamic_symbolic_horizon,
                                            budget.max_affordable_symbolic())
        push(frames, current_timestep,
             f"t={current_timestep}  [carry] symbolic {sym_from}→{pending_job.symbolic_start}",
             "baseline")

        if result is not None:
            conflict_time = pending_job.conflict_time
            pending_job   = None
            if result["collision"]:
                push(frames, current_timestep,
                     f"t={current_timestep}  [carry] conflict confirmed@{conflict_time} — MPC filter",
                     "baseline")
                validated_until = max(validated_until, conflict_time - 1)
                apply_mpc_filter_frames(frames, conflict_time, current_timestep)
            else:
                push(frames, current_timestep,
                     f"t={current_timestep}  [carry] ✓ deconflicted vu={conflict_time}",
                     "baseline")
                validated_until = conflict_time
                if budget.remaining > 0:
                    validated_until, pending_job = try_extend_anim(
                        frames, validated_until, MAX_TIME, budget,
                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                        origin="baseline",
                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                    )

    # ── Phase 2: optimized or baseline ────────────────────────────────
    else:
        safety_margin = validated_until - current_timestep
        dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
            safety_margin,
            max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
            symbolic_buffer=SYMBOLIC_BUFFER
        )

        if safety_margin >= MIN_SAFE_HORIZON:
            while (validated_until - current_timestep >= MIN_SAFE_HORIZON
                   and validated_until < MAX_TIME
                   and pending_job is None
                   and budget.remaining > 0):
                validated_until, pending_job = optimized_step_anim(
                    frames, validated_until, MAX_TIME, budget,
                    dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                    ext_optimizer
                )
                dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                    validated_until - current_timestep,
                    max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                    symbolic_buffer=SYMBOLIC_BUFFER
                )

            new_margin = validated_until - current_timestep
            if new_margin < MIN_SAFE_HORIZON and pending_job is None \
                    and budget.remaining > 0:
                push(frames, current_timestep,
                     f"t={current_timestep}  [OPT] margin dropped to {new_margin} — recovering",
                     "info")
                validated_until, pending_job = try_extend_anim(
                    frames, validated_until, MAX_TIME, budget,
                    dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                    origin="baseline",
                    safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                )

        else:
            explore_from   = max(validated_until, current_timestep)
            end_check_time = min(
                explore_from + budget.max_affordable_concrete(),
                current_timestep + MIN_SAFE_HORIZON,
                MAX_TIME
            )
            result        = tester.concrete(explore_from, end_check_time)
            collision     = result["collision"]
            conflict_time = result.get("collision_timestep")
            push(frames, current_timestep,
                 f"t={current_timestep}  [BASE] concrete {explore_from}→{end_check_time}"
                 + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
                 "baseline")

            if not collision:
                validated_until = end_check_time
                new_margin      = validated_until - current_timestep
                if new_margin >= MIN_SAFE_HORIZON and budget.remaining > 0:
                    push(frames, current_timestep,
                         f"t={current_timestep}  [BASE→OPT] reached MIN_SAFE_HORIZON",
                         "info")
                    validated_until, pending_job = optimized_step_anim(
                        frames, validated_until, MAX_TIME, budget,
                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                        ext_optimizer
                    )
            else:
                pending_job = VerificationTask(symbolic_start=current_timestep,
                                               conflict_time=conflict_time)
                sym_start   = pending_job.symbolic_start
                pending_job, result_s = symbolic_step(tester, pending_job,
                                                      dynamic_symbolic_horizon,
                                                      budget.max_affordable_symbolic())
                push(frames, current_timestep,
                     f"t={current_timestep}  [BASE] symbolic {sym_start}→"
                     f"{pending_job.symbolic_start}",
                     "baseline")

                if result_s is not None:
                    pending_job = None
                    if result_s["collision"]:
                        push(frames, current_timestep,
                             f"t={current_timestep}  [BASE] conflict confirmed@{conflict_time} — MPC filter",
                             "baseline")
                        validated_until = max(validated_until, conflict_time - 1)
                        apply_mpc_filter_frames(frames, conflict_time, current_timestep)
                    else:
                        push(frames, current_timestep,
                             f"t={current_timestep}  [BASE] ✓ deconflicted vu={conflict_time}",
                             "baseline")
                        validated_until = conflict_time
                        if budget.remaining > 0:
                            new_margin = validated_until - current_timestep
                            if new_margin >= MIN_SAFE_HORIZON:
                                push(frames, current_timestep,
                                     f"t={current_timestep}  [BASE→OPT] deconflicted past MIN_SAFE_HORIZON",
                                     "info")
                                validated_until, pending_job = optimized_step_anim(
                                    frames, validated_until, MAX_TIME, budget,
                                    dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                    ext_optimizer
                                )
                            else:
                                validated_until, pending_job = try_extend_anim(
                                    frames, validated_until, MAX_TIME, budget,
                                    dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                    origin="baseline",
                                    safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                                )
                else:
                    validated_until = pending_job.symbolic_start
                    push(frames, current_timestep,
                         f"t={current_timestep}  [BASE] budget exhausted@{validated_until}, defer",
                         "baseline")

    if validated_until >= MAX_TIME:
        push(frames, current_timestep, "Done ✓", "info")
        break

    # ── Cancel + recompute MPC if deconflicted past conflict ──────────
    if (mpc_conflict_time is not None
            and validated_until >= mpc_conflict_time):
        if mpc_needed:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] deconflicted past t={mpc_conflict_time}, "
                 f"cancelling plan from t={mpc_committed_at}",
                 "mpc")
            if current_timestep >= mpc_committed_at and validated_until < MAX_TIME:
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC] recomputing for horizon boundary t={validated_until}",
                     "mpc")
                apply_mpc_filter_frames(frames, validated_until, current_timestep)
        mpc_needed = False

    # ── Advance real state ────────────────────────────────────────────
    if (mpc_committed_at is not None
            and current_timestep >= mpc_committed_at
            and mpc_needed):
        ctrl_idx = current_timestep - mpc_committed_at
        if ctrl_idx < len(mpc_control_queue):
            mpc_started = True
            ctrl        = mpc_control_queue[ctrl_idx]
            _mpc_trail.append(mpc_traj_bounds_all[ctrl_idx].copy())
            _mpc_trail_frame_idx.append(len(frames))
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC FIRST FIRE] idx={ctrl_idx}"
                 f"  u={np.round(ctrl, 4)}  (plan from t={mpc_committed_at})",
                 "mpc", mpc_t_back=mpc_committed_at)
            current_timestep += 1
        else:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] queue exhausted — ERROR", "info")
            break
    else:
        tester.real_state_empirical(current_timestep, current_timestep + 1)
        current_timestep += 1
        push(frames, current_timestep,
             f"t={current_timestep}  empirical step  vu={validated_until}", "info")

print(f"Generated {len(frames)} frames")


# ══════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════

RSOA_FILL = {
    "concrete":  {"fc": "#f59e0b", "fa": 0.18, "ec": "#f59e0b"},
    "symbolic":  {"fc": "#8b5cf6", "fa": 0.25, "ec": "#8b5cf6"},
    "sampled":   {"fc": "#06b6d4", "fa": 0.18, "ec": "#06b6d4"},
    "empirical": {"fc": "#10b981", "fa": 0.18, "ec": "#10b981"},
}
ORIGIN_BORDER = {
    "baseline":  {"lw": 0.8,  "ls": "--", "hatch": None, "alpha_boost": 0.0},
    "optimizer": {"lw": 2.0,  "ls": "-",  "hatch": "//", "alpha_boost": 0.12},
    "mpc":       {"lw": 1.5,  "ls": "-",  "hatch": None, "alpha_boost": 0.10},
    "info":      {"lw": 0.8,  "ls": "-",  "hatch": None, "alpha_boost": 0.0},
}
MPC_TRAJ_COLOR = {"fc": "#f97316", "fa": 0.35, "ec": "#f97316"}
_BOUND_LIMIT   = 1e6

t_origin: dict = {}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.set_xlabel("position (p)", color="#ccc", fontsize=13)
ax.set_ylabel("velocity (v)", color="#ccc", fontsize=13)
title = ax.set_title("", color="#eee", fontsize=11, fontfamily="monospace", pad=12)

# Axis limits
all_x1, all_x2 = [], []
for snap, _, _, _, _, _, _ in frames:
    for t, entry in snap.items():
        all_x1.extend([v for v in entry["x1"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
        all_x2.extend([v for v in entry["x2"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
for obs in obstacles:
    all_x1.extend([v for v in [obs[0, 0], obs[0, 1]] if np.isfinite(v)])
    all_x2.extend([v for v in [obs[1, 0], obs[1, 1]] if np.isfinite(v)])

mg = 0.3
x1_min = min(all_x1) - mg; x1_max = max(all_x1) + mg
x2_min = min(all_x2) - mg; x2_max = max(all_x2) + mg
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

for i, obs in enumerate(obstacles):
    x_lo = float(np.clip(obs[0, 0], x1_min, x1_max))
    x_hi = float(np.clip(obs[0, 1], x1_min, x1_max))
    y_lo = float(np.clip(obs[1, 0], x2_min, x2_max))
    y_hi = float(np.clip(obs[1, 1], x2_min, x2_max))
    ax.add_patch(patches.Rectangle(
        (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
        linewidth=1.5, edgecolor="#ef4444", facecolor="#ef4444",
        alpha=0.5, zorder=5, label="Obstacle" if i == 0 else None
    ))

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists, t_origin
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    snap, mpc_traj_data, mpc_t_back, ct, label, origin, stashed_mpc_traj = frames[frame_idx]
    title.set_text(label)

    if origin not in ("info",):
        for t in snap:
            if t not in t_origin:
                t_origin[t] = origin

    # RSOA boxes
    for t, entry in sorted(snap.items()):
        fill  = RSOA_FILL.get(entry["type"], RSOA_FILL["concrete"])
        orign = t_origin.get(t, "baseline")
        bdr   = ORIGIN_BORDER[orign]

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
        cx = (x1_lo + x1_hi) / 2; cy = (x2_lo + x2_hi) / 2
        txt = ax.text(cx, cy, str(t), fontsize=6, color=fill["ec"], alpha=0.85,
                      ha="center", va="center", zorder=3, fontfamily="monospace",
                      fontweight="bold" if orign == "optimizer" else "normal")
        dynamic_artists.append(txt)

    # Accumulated MPC executed trail
    trail_len = sum(1 for fi in _mpc_trail_frame_idx if fi <= frame_idx)
    for b in _mpc_trail[:trail_len]:
        x1_lo, x1_hi = float(b[0, 0]), float(b[0, 1])
        x2_lo, x2_hi = float(b[1, 0]), float(b[1, 1])
        if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                   for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
            continue
        r = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=1.2, edgecolor=MPC_TRAJ_COLOR["ec"],
            facecolor=MPC_TRAJ_COLOR["fc"], alpha=MPC_TRAJ_COLOR["fa"], zorder=4
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

    # Current MPC attempt trajectory (lookahead preview), or stashed committed plan
    traj_to_draw = mpc_traj_data if mpc_traj_data else stashed_mpc_traj
    if traj_to_draw:
        for b in traj_to_draw:
            if b.shape[0] < 2:
                continue
            x1_lo, x1_hi = float(b[0, 0]), float(b[0, 1])
            x2_lo, x2_hi = float(b[1, 0]), float(b[1, 1])
            if not all(np.isfinite(v) and abs(v) < _BOUND_LIMIT
                       for v in [x1_lo, x1_hi, x2_lo, x2_hi]):
                continue
            r = patches.Rectangle(
                (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
                linewidth=1.2, edgecolor=MPC_TRAJ_COLOR["ec"],
                facecolor=MPC_TRAJ_COLOR["fc"], alpha=MPC_TRAJ_COLOR["fa"], zorder=4
            )
            ax.add_patch(r)
            dynamic_artists.append(r)

    # Star at MPC committed t_back
    if mpc_t_back is not None and mpc_t_back in snap:
        entry = snap[mpc_t_back]
        if all(np.isfinite(v) for v in [*entry["x1"], *entry["x2"]]):
            cx = sum(entry["x1"]) / 2; cy = sum(entry["x2"]) / 2
            star = ax.plot(cx, cy, "*", color="#f97316", markersize=12, zorder=7)[0]
            dynamic_artists.append(star)

    # Blue dot at current timestep
    if ct in snap:
        entry = snap[ct]
        if all(np.isfinite(v) for v in [*entry["x1"], *entry["x2"]]):
            cx = sum(entry["x1"]) / 2; cy = sum(entry["x2"]) / 2
            dot = ax.plot(cx, cy, "o", color="#3b82f6", markersize=8, zorder=6)[0]
            dynamic_artists.append(dot)
            lbl = ax.text(cx + 0.05, cy + 0.04, f"t={ct}", color="#3b82f6",
                          fontsize=9, fontfamily="monospace", zorder=6)
            dynamic_artists.append(lbl)

    # Mode badge
    if origin == "optimizer":
        badge_color, badge_text = "#22c55e", "● OPT"
    elif origin == "mpc":
        badge_color, badge_text = "#f97316", "● MPC"
    elif label and "BASE→OPT" in label:
        badge_color, badge_text = "#f59e0b", "⇒ BASE→OPT"
    else:
        badge_color, badge_text = "#64748b", "● BASE"
    badge = ax.text(
        0.01, 0.97, badge_text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        color=badge_color, va="top", zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#0e1117", ec=badge_color, lw=1.2)
    )
    dynamic_artists.append(badge)

    return dynamic_artists


legend_els = [
    Patch(facecolor="#f59e0b", alpha=0.3, edgecolor="#f59e0b", label="Concrete RSOA"),
    Patch(facecolor="#8b5cf6", alpha=0.3, edgecolor="#8b5cf6", label="Symbolic RSOA"),
    Patch(facecolor="#10b981", alpha=0.3, edgecolor="#10b981", label="Empirical (Kalman)"),
    Patch(facecolor="#f97316", alpha=0.35, edgecolor="#f97316", label="MPC Kalman bounds"),
    Patch(facecolor="#ef4444", alpha=0.5,  edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o", color="#3b82f6", ls="", markersize=6, label="Current t"),
    Line2D([0], [0], marker="*", color="#f97316", ls="", markersize=8, label="MPC t_back"),
    Patch(facecolor="#888", alpha=0.2, edgecolor="#888",
          linewidth=0.8, linestyle="--", label="Baseline calc"),
    Patch(facecolor="#888", alpha=0.3, edgecolor="#22c55e",
          linewidth=2.0, hatch="//", label="Optimizer calc"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="#1a1e28", edgecolor="#333", labelcolor="#aaa")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

out_path = "alg7_optimization_mpc.gif"
ani.save(out_path, writer="pillow", fps=3, dpi=130)
print(f"Saved to {out_path}")
plt.close()
