"""
Animation for alg11 — adaptive MPC with acados inner solver.
Saves:
    alg11_mpc_acados.gif
    alg11_mpc_acados.mp4

Obstacles are in the form: [center_x, center_y, radius]

To call with a specific np seed, do python anim_alg11.py <seed_number>

"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter
import time
from alg11_mpc_acados import (
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
MAX_TIME             = 60
MIN_SAFE_HORIZON     = 12
# MIN_SAFE_HORIZON     = 12
MIN_LOOKAHEAD        = 4
MAX_SYMBOLIC_HORIZON = 10
SYMBOLIC_BUFFER      = 5

obstacles = [
        np.array([-6.5, 2.02,  0.5]),
        np.array([-3.2,  1.21,  0.5]),
        np.array([-2,  -0.3, 0.45]),
        np.array([-2, -1.3, 0.5])
    ]


# ── Setup ──
import sys
_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1401830092
print(f"Setting up analyzer...  seed={_seed}")

analyzer           = setup_analyzer('Unicycle_NL', 'natural_none_default')
tester             = ReachabilityTester(analyzer, obstacles, seed=_seed)
tester_calibration = ReachabilityTester(analyzer)
mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=10, nominal_tracking=True)


budget = TimeBudget(timestep_budget=0.3)
print("Calibrating time budget...")
budget.symbolic_costs = {1: 0.05942702293395996, 2: 0.0532071590423584, 3: 0.12308859825134277, 4: 0.2227306365966797, 5: 0.3548123836517334, 6: 0.5160810947418213, 7: 0.7076215744018555, 8: 1.046485185623169, 9: 1.189185619354248, 10: 1.4745268821716309}
budget.concrete_cost = 0.012865893046061198

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


# ── MPC state (module-level, mirrors alg10 mpc_state dict) ──
mpc_committed_at     = None
mpc_conflict_time    = None
mpc_control_queue    = []
mpc_traj_bounds_all  = []
mpc_needed           = False
mpc_started          = False
_mpc_trail           = []
_mpc_trail_frame_idx = []

pending_mpc_conflict     = None   # conflict_time queued for MPC (deferred)
mpc_turn                 = False  # True = MPC gets this timestep when both pending
mpc_first_run            = True   # first confirmed collision runs MPC immediately
dynamic_symbolic_horizon = MAX_SYMBOLIC_HORIZON  # updated each timestep


def extend_mpc_sequence_anim(frames, current_timestep, ctrl_idx=0):
    """
    Extend the MPC control queue from the end of the current trajectory.
    If extension from the endpoint fails, works backwards along the existing
    trajectory (like find_stopping_timestep) to find a viable extension point.
    """
    global mpc_control_queue, mpc_traj_bounds_all

    n_controls = len(mpc_control_queue)
    mpc_end = mpc_committed_at + n_controls

    if mpc_end >= MAX_TIME:
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC extend] reached MAX_TIME", "mpc")
        return

    # Don't look back past current execution point
    max_lb = min(n_controls - ctrl_idx - 1, mpc_sf.n_horizon)
    max_lb = max(max_lb, 0)

    controls_snapshot = list(mpc_control_queue)
    bounds_snapshot   = list(mpc_traj_bounds_all)

    for lb in range(max_lb + 1):
        try_idx = len(bounds_snapshot) - 1 - lb
        if try_idx < 1:
            break

        try_bounds = bounds_snapshot[try_idx]
        center = (try_bounds[:, 0] + try_bounds[:, 1]) / 2.0

        try:
            new_traj_bounds, _, new_controls = mpc_sf._run_mpc_from_bounds(
                try_bounds, center, extra_inflation=0.0)
        except Exception as e:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC extend] failed at lookback={lb}: {e}", "mpc")
            continue

        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides(b):
                break
            safe_count += 1

        if safe_count <= lb:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC extend] lookback={lb}: "
                 f"only {safe_count} safe (need >{lb}), going back", "mpc")
            continue

        # Replace tail (lb controls) with new safe path
        keep = n_controls - lb
        mpc_control_queue = controls_snapshot[:keep] + list(new_controls[:safe_count])
        mpc_traj_bounds_all = bounds_snapshot[:try_idx + 1] + list(new_traj_bounds[1:safe_count + 1])

        net_gain  = safe_count - lb
        new_total = len(mpc_control_queue)
        new_end   = mpc_committed_at + new_total
        lb_msg = f" (lookback={lb})" if lb > 0 else ""
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC extend{lb_msg}] +{net_gain} → {new_total} total "
             f"(t={mpc_committed_at} to t={new_end})", "mpc")
        return

    push(frames, current_timestep,
         f"t={current_timestep}  [MPC extend] no safe extension after "
         f"{max_lb + 1} lookback attempts", "mpc")


def apply_mpc_filter_frames(frames, conflict_time, current_timestep):
    """Run MPC filter, capturing one frame per lookback attempt."""
    global mpc_committed_at, mpc_conflict_time, mpc_control_queue
    global mpc_traj_bounds_all, mpc_needed

    _mpc_total_t0 = time.perf_counter()
    for lookback in range(3, mpc_sf.max_lookback + 1):
        t_back = conflict_time - lookback
        if t_back < current_timestep:
            break
        if t_back not in tester.horizons:
            continue
        bounds_at_back = tester.horizons[t_back].get_tight_bound()
        if bounds_at_back is None:
            continue

        center = (bounds_at_back[:, 0] + bounds_at_back[:, 1]) / 2.0
        print(f"center: {center}")
        cur_h = tester.horizons.get(current_timestep)
        if cur_h is not None:
            cb   = cur_h.get_tight_bound()
            dx_c = cb[0, 1] - cb[0, 0]
            dy_c = cb[1, 1] - cb[1, 0]
            current_inflation = float(np.sqrt((dx_c / 2) ** 2 + (dy_c / 2) ** 2))
        else:
            current_inflation = 0.0

        try:
            t0 = time.perf_counter()
            traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
                bounds_at_back, center, extra_inflation=current_inflation)
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
                print(f"  [MPC COMMIT] t_back={t_back}  controls={[np.round(c, 4).tolist() for c in controls]}")
            else:
                print(f"  [MPC] Keeping existing plan at t={mpc_committed_at}")
            print(f"  [MPC total] apply_mpc_filter_frames took {time.perf_counter() - _mpc_total_t0:.3f}s")
            return t_back

    print(f"  [MPC total] apply_mpc_filter_frames took {time.perf_counter() - _mpc_total_t0:.3f}s")
    push(frames, current_timestep,
         f"t={current_timestep}  [MPC] no safe timestep at or after t={current_timestep} — continuing nominal",
         "mpc")
    return None


def trigger_mpc_anim(frames, conflict_time, current_timestep):
    """First collision: run MPC now and capture frames. Subsequent: queue and alternate."""
    global mpc_first_run, pending_mpc_conflict
    if mpc_first_run:
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC] first collision — running immediately", "mpc")
        apply_mpc_filter_frames(frames, conflict_time, current_timestep)
        mpc_first_run = False
    else:
        pending_mpc_conflict = conflict_time
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC] queued t_conflict={conflict_time}", "mpc")


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

        # Pre-compute MPC as backup before symbolic verification
        trigger_mpc_anim(frames, conflict_time, current_timestep)

        if not budget.can_afford('symbolic', 1):
            vu = conflict_time - 1
            return vu, VerificationTask(symbolic_start=conflict_time - 1,
                                        conflict_time=conflict_time), conflict_time

        job       = VerificationTask(symbolic_start=vu, conflict_time=conflict_time)
        sym_start = job.symbolic_start
        job, result_s = symbolic_step(tester, job, max_symbolic_horizon,
                                      budget)
        push(frames, current_timestep,
             f"t={current_timestep}  [extend] symbolic {sym_start}→{job.symbolic_start}",
             origin)

        if result_s is None:
            vu = job.symbolic_start
            return vu, job, None

        if result_s["collision"]:
            push(frames, current_timestep,
                 f"t={current_timestep}  [extend] conflict confirmed@{conflict_time} — recomputing MPC",
                 origin)
            trigger_mpc_anim(frames, conflict_time, current_timestep)
            vu = max(vu, conflict_time - 1)
            return vu, None, conflict_time
        else:
            vu = conflict_time
            push(frames, current_timestep,
                 f"t={current_timestep}  [extend] ✓ deconflicted vu={vu}", origin)

    return vu, None, None


def optimized_step_anim(frames, validated_until, max_time, budget,
                        max_symbolic_horizon, current_timestep, min_lookahead,
                        ext_optimizer):
    """
    Mirrors alg10's optimized_step: uses get_strategy_opt_free with pow_vol=3.
    """
    target = validated_until + 1
    if target > max_time:
        return validated_until, None, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    # alg10 uses get_strategy_opt_free (not get_strategy)
    method = ext_optimizer.get_strategy_opt_free(
        current_timestep=current_timestep,
        verified_until=validated_until,
        current_vol=current_vol,
        verified_vol=verified_vol,
        time_budget=budget.remaining,
        w_vol=50.0,
        pow_vol=3,
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
        return target, None, None

    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] collision@{conflict_time} — deconflicting from t={current_timestep}",
         "optimizer")

    # Pre-compute MPC as backup before symbolic verification
    trigger_mpc_anim(frames, conflict_time, current_timestep)

    if not budget.can_afford('symbolic', 1):
        return conflict_time - 1, VerificationTask(
            symbolic_start=current_timestep, conflict_time=conflict_time), conflict_time

    job       = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    sym_start = job.symbolic_start
    job, result_s = symbolic_step(tester, job, max_symbolic_horizon,
                                  budget)
    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] deconflict symbolic {sym_start}→{job.symbolic_start}",
         "optimizer")

    if result_s is None:
        return conflict_time - 1, job, None

    if result_s["collision"]:
        push(frames, current_timestep,
             f"t={current_timestep}  [OPT] conflict confirmed@{conflict_time} — recomputing MPC",
             "optimizer")
        trigger_mpc_anim(frames, conflict_time, current_timestep)
        return conflict_time - 1, None, conflict_time

    push(frames, current_timestep,
         f"t={current_timestep}  [OPT] deconflicted vu={conflict_time}", "optimizer")
    return conflict_time, None, None


def run_opt_loop_anim(frames):
    """Run the optimizer while loop until budget, margin, or a pending job stops it."""
    global validated_until, pending_job, dynamic_symbolic_horizon
    while (validated_until - current_timestep >= MIN_SAFE_HORIZON
           and validated_until < MAX_TIME
           and pending_job is None
           and pending_mpc_conflict is None
           and budget.remaining > 0):
        validated_until, pending_job, mpc_c = optimized_step_anim(
            frames, validated_until, MAX_TIME, budget,
            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
            ext_optimizer
        )
        if mpc_c is not None:
            trigger_mpc_anim(frames, mpc_c, current_timestep)
        dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
            validated_until - current_timestep,
            max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
            symbolic_buffer=SYMBOLIC_BUFFER
        )


# ══════════════════════════════════════════════
# Run alg10, capturing frames
# ══════════════════════════════════════════════
print("Running alg11 (acados)...")
frames = []

current_timestep = 0
validated_until  = 0
pending_job: Optional[VerificationTask] = None

# reset alternating-MPC globals for this run
pending_mpc_conflict     = None
mpc_turn                 = False
mpc_first_run            = True
dynamic_symbolic_horizon = MAX_SYMBOLIC_HORIZON

push(frames, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()

    safety_margin = validated_until - current_timestep
    dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
        safety_margin,
        max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
        symbolic_buffer=SYMBOLIC_BUFFER
    )

    # ── MPC active: re-check each timestep whether conflict persists ──
    if mpc_started:
        ctrl_idx  = current_timestep - mpc_committed_at
        queue     = mpc_control_queue
        check_end = min(current_timestep + MIN_SAFE_HORIZON, MAX_TIME)

        # Extend when remaining controls <= n_horizon
        # Retries each step if previous extend failed
        n_h = mpc_sf.n_horizon
        if len(queue) - ctrl_idx <= n_h:
            extend_mpc_sequence_anim(frames, current_timestep, ctrl_idx=ctrl_idx)
            queue = mpc_control_queue

        for _t in list(tester.horizons.keys()):
            if _t > current_timestep:
                del tester.horizons[_t]

        result_check           = tester.concrete(current_timestep, check_end)
        conflict_still_present = result_check["collision"]
        push(frames, current_timestep,
             f"t={current_timestep}  [MPC check] concrete {current_timestep}→{check_end}"
             + ("  ⚠ conflict persists" if conflict_still_present else "  ✓ clear — reverting"),
             "mpc")

        if not conflict_still_present:
            mpc_started         = False
            mpc_needed          = False
            mpc_committed_at    = None
            mpc_conflict_time    = None
            mpc_traj_bounds_all  = []
            mpc_first_run        = True
            validated_until      = check_end
            pending_job          = None
            pending_mpc_conflict = None
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC REVERT] reverting to nominal  vu={check_end}",
                 "info")
            tester.real_state_empirical(current_timestep, current_timestep + 1)
            current_timestep += 1
            push(frames, current_timestep,
                 f"t={current_timestep}  empirical step (post-revert)  vu={validated_until}",
                 "info")
        elif ctrl_idx < len(queue):
            ctrl = queue[ctrl_idx]
            mpc_bound = mpc_traj_bounds_all[ctrl_idx + 1]
            _mpc_trail.append(mpc_bound.copy())
            _mpc_trail_frame_idx.append(len(frames))
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] idx={ctrl_idx}  u={np.round(ctrl, 4)}"
                 f"  (plan from t={mpc_committed_at})",
                 "mpc", mpc_t_back=mpc_committed_at)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
        else:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] queue nearly exhausted — attempting extension", "mpc")
            extend_mpc_sequence_anim(frames, current_timestep, ctrl_idx=ctrl_idx)
            queue = mpc_control_queue
            if ctrl_idx < len(queue):
                ctrl = queue[ctrl_idx]
                mpc_bound = mpc_traj_bounds_all[ctrl_idx + 1]
                _mpc_trail.append(mpc_bound.copy())
                _mpc_trail_frame_idx.append(len(frames))
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC EXTENDED] idx={ctrl_idx}  u={np.round(ctrl, 4)}",
                     "mpc", mpc_t_back=mpc_committed_at)
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC] queue exhausted even after extension", "info")
                break
        continue

    both_pending = pending_job is not None and pending_mpc_conflict is not None
    mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
    push(frames, current_timestep,
         f"t={current_timestep}  vu={validated_until}  margin={safety_margin}  [{mode}]"
         + (f"  [MPC queued t_conflict={pending_mpc_conflict}]"
            if pending_mpc_conflict is not None else ""),
         "info")

    # ── Decide: MPC this timestep or symbolic? ────────────────────────
    has_commit = mpc_committed_at is not None and mpc_needed
    do_mpc_this_step = (
        pending_mpc_conflict is not None and
        (pending_job is None or not has_commit or mpc_turn)
    )

    if do_mpc_this_step:
        push(frames, current_timestep,
             f"t={current_timestep}  [Alt] MPC turn — conflict_time={pending_mpc_conflict}",
             "mpc")
        apply_mpc_filter_frames(frames, pending_mpc_conflict, current_timestep)
        pending_mpc_conflict = None
        mpc_turn = False

    else:
        if both_pending:
            mpc_turn = True

        # ── Phase 1: carry-over ────────────────────────────────────────
        if pending_job is not None:
            sym_from = pending_job.symbolic_start
            pending_job, result = symbolic_step(tester, pending_job,
                                                dynamic_symbolic_horizon,
                                                budget)
            push(frames, current_timestep,
                 f"t={current_timestep}  [carry] symbolic {sym_from}→{pending_job.symbolic_start}",
                 "baseline")

            if result is not None:
                conflict_time = pending_job.conflict_time
                pending_job   = None
                if result["collision"]:
                    push(frames, current_timestep,
                         f"t={current_timestep}  [carry] conflict confirmed@{conflict_time} — triggering MPC",
                         "baseline")
                    validated_until = max(validated_until, conflict_time - 1)
                    trigger_mpc_anim(frames, conflict_time, current_timestep)
                else:
                    push(frames, current_timestep,
                         f"t={current_timestep}  [carry] ✓ deconflicted vu={conflict_time}",
                         "baseline")
                    validated_until = conflict_time
                    if budget.remaining > 0:
                        validated_until, pending_job, mpc_c = try_extend_anim(
                            frames, validated_until, MAX_TIME, budget,
                            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                            origin="baseline",
                            safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                        )
                        if mpc_c is not None:
                            trigger_mpc_anim(frames, mpc_c, current_timestep)
                        run_opt_loop_anim(frames)

        # ── Phase 2: optimized or baseline ────────────────────────────
        else:
            safety_margin = validated_until - current_timestep
            dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                safety_margin,
                max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                symbolic_buffer=SYMBOLIC_BUFFER
            )

            if safety_margin >= MIN_SAFE_HORIZON:
                run_opt_loop_anim(frames)

                new_margin = validated_until - current_timestep
                if new_margin < MIN_SAFE_HORIZON and pending_job is None \
                        and pending_mpc_conflict is None and budget.remaining > 0:
                    push(frames, current_timestep,
                         f"t={current_timestep}  [OPT] margin dropped to {new_margin} — recovering",
                         "info")
                    validated_until, pending_job, mpc_c = try_extend_anim(
                        frames, validated_until, MAX_TIME, budget,
                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                        origin="baseline",
                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                    )
                    if mpc_c is not None:
                        trigger_mpc_anim(frames, mpc_c, current_timestep)
                    run_opt_loop_anim(frames)

            elif validated_until < MAX_TIME:
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
                    if budget.remaining > 0:
                        run_opt_loop_anim(frames)
                else:
                    # Pre-compute MPC as backup before symbolic verification
                    trigger_mpc_anim(frames, conflict_time, current_timestep)
                    pending_job = VerificationTask(symbolic_start=current_timestep,
                                                   conflict_time=conflict_time)
                    sym_start   = pending_job.symbolic_start
                    pending_job, result_s = symbolic_step(tester, pending_job,
                                                          dynamic_symbolic_horizon,
                                                          budget)
                    push(frames, current_timestep,
                         f"t={current_timestep}  [BASE] symbolic {sym_start}→"
                         f"{pending_job.symbolic_start}",
                         "baseline")

                    if result_s is not None:
                        pending_job = None
                        if result_s["collision"]:
                            push(frames, current_timestep,
                                 f"t={current_timestep}  [BASE] conflict confirmed@{conflict_time} — recomputing MPC",
                                 "baseline")
                            validated_until = max(validated_until, conflict_time - 1)
                            trigger_mpc_anim(frames, conflict_time, current_timestep)
                        else:
                            push(frames, current_timestep,
                                 f"t={current_timestep}  [BASE] ✓ deconflicted vu={conflict_time}",
                                 "baseline")
                            validated_until = conflict_time
                            if budget.remaining > 0:
                                new_margin = validated_until - current_timestep
                                if new_margin < MIN_SAFE_HORIZON:
                                    validated_until, pending_job, mpc_c = try_extend_anim(
                                        frames, validated_until, MAX_TIME, budget,
                                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                        origin="baseline",
                                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                                    )
                                    if mpc_c is not None:
                                        trigger_mpc_anim(frames, mpc_c, current_timestep)
                                run_opt_loop_anim(frames)
                    else:
                        validated_until = pending_job.symbolic_start
                        push(frames, current_timestep,
                             f"t={current_timestep}  [BASE] budget exhausted@{validated_until}, defer",
                             "baseline")

    # ── Cancel + recompute MPC if deconflicted past conflict ──────────
    if (mpc_conflict_time is not None
            and validated_until >= mpc_conflict_time + MIN_SAFE_HORIZON):
        if mpc_needed:
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC] deconflicted past t={mpc_conflict_time}, "
                 f"cancelling plan from t={mpc_committed_at}",
                 "mpc")
            if current_timestep >= mpc_committed_at and validated_until < MAX_TIME:
                push(frames, current_timestep,
                     f"t={current_timestep}  [MPC] recomputing for horizon boundary t={validated_until}",
                     "mpc")
                trigger_mpc_anim(frames, validated_until, current_timestep)
        mpc_needed = False
        mpc_first_run = True

    # ── Advance real state ────────────────────────────────────────────
    if (mpc_committed_at is not None
            and current_timestep >= mpc_committed_at
            and mpc_needed):
        ctrl_idx = current_timestep - mpc_committed_at
        if ctrl_idx < len(mpc_control_queue):
            mpc_started = True
            ctrl      = mpc_control_queue[ctrl_idx]
            mpc_bound = mpc_traj_bounds_all[ctrl_idx + 1]
            _mpc_trail.append(mpc_bound.copy())
            _mpc_trail_frame_idx.append(len(frames))
            push(frames, current_timestep,
                 f"t={current_timestep}  [MPC FIRST FIRE] idx={ctrl_idx}"
                 f"  u={np.round(ctrl, 4)}  (plan from t={mpc_committed_at})",
                 "mpc", mpc_t_back=mpc_committed_at)
            tester.real_state_mpc(current_timestep, ctrl)
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
    "concrete":  {"fc": "#f59e0b", "fa": 0.45, "ec": "#d97706"},
    "symbolic":  {"fc": "#7c3aed", "fa": 0.45, "ec": "#5b21b6"},
    "sampled":   {"fc": "#0891b2", "fa": 0.45, "ec": "#0e7490"},
    "empirical": {"fc": "#059669", "fa": 0.45, "ec": "#047857"},
}
ORIGIN_BORDER = {
    "baseline":  {"lw": 1.2,  "ls": "--", "hatch": None, "alpha_boost": 0.0},
    "optimizer": {"lw": 2.5,  "ls": "-",  "hatch": "//", "alpha_boost": 0.15},
    "mpc":       {"lw": 2.0,  "ls": "-",  "hatch": None, "alpha_boost": 0.12},
    "info":      {"lw": 1.2,  "ls": "-",  "hatch": None, "alpha_boost": 0.0},
}
MPC_TRAJ_COLOR = {"fc": "#ec4899", "fa": 0.50, "ec": "#be185d"}
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
title = ax.set_title("", fontsize=0)

# Axis limits
all_x1, all_x2 = [], []
for snap, _, _, _, _, _, _ in frames:
    for t, entry in snap.items():
        all_x1.extend([v for v in entry["x1"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
        all_x2.extend([v for v in entry["x2"] if np.isfinite(v) and abs(v) < _BOUND_LIMIT])
for obs in obstacles:
    cx, cy, r = obs[0], obs[1], obs[2]
    all_x1.extend([cx - r, cx + r])
    all_x2.extend([cy - r, cy + r])

mg = 0.3
x1_min = min(all_x1) - mg; x1_max = max(all_x1) + mg
x2_min = min(all_x2) - mg; x2_max = max(all_x2) + mg
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
            star = ax.plot(cx, cy, "*", color="#ec4899", markersize=12, zorder=7)[0]
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

    # Mode badge — updated label to reflect OPT
    if origin == "optimizer":
        badge_color, badge_text = "#22c55e", "● OPT"
    elif origin == "mpc":
        badge_color, badge_text = "#ec4899", "● MPC"
    elif label and "BASE→OPT" in label:
        badge_color, badge_text = "#f59e0b", "⇒ BASE→OPT"
    else:
        badge_color, badge_text = "#64748b", "● BASE"
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
    Patch(facecolor="#ec4899", alpha=0.35, edgecolor="#ec4899", label="MPC Kalman bounds"),
    Patch(facecolor="#ef4444", alpha=0.5,  edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o", color="#3b82f6", ls="", markersize=6, label="Current t"),
    Line2D([0], [0], marker="*", color="#ec4899", ls="", markersize=8, label="MPC t_back"),
    Patch(facecolor="#888", alpha=0.2, edgecolor="#888",
          linewidth=0.8, linestyle="--", label="Baseline calc"),
    Patch(facecolor="#888", alpha=0.3, edgecolor="#22c55e",
          linewidth=2.0, hatch="//", label="Opt calc"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="white", edgecolor="#aaa", labelcolor="#222")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

# import io, imageio

# out_path = "alg10_mpc_w_custom_cost.gif"
# print(f"Rendering {len(frames)} frames...")
# images = []
# for i in range(len(frames)):
#     if i % 50 == 0:
#         print(f"{i}/{len(frames)} rendered...")
#     update(i)
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi=100)
#     buf.seek(0)
#     images.append(imageio.v2.imread(buf))
# imageio.mimsave(out_path, images, duration=150, loop=0)
# print(f"Saved to {out_path}")
# plt.close()

import io, imageio

gif_path = "alg11_mpc_acados.gif"
mp4_path = "alg11_mpc_acados.mp4"

print(f"Rendering {len(frames)} frames...")

images = []

# --- frame rendering (shared for both outputs) ---
for i in range(len(frames)):
    if i % 50 == 0:
        print(f"{i}/{len(frames)} rendered...")

    update(i)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    img = imageio.v2.imread(buf)
    images.append(img)

# --- save GIF (original behavior) ---
imageio.mimsave(gif_path, images, duration=150, loop=0)
print(f"Saved GIF to {gif_path}")

# --- save MP4 (new addition) ---
fps = 30  # adjust as needed

writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264")

for img in images:
    writer.append_data(img)

writer.close()

print(f"Saved MP4 to {mp4_path}")

plt.close()
