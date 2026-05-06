"""
Adaptive MPC algorithm
-- based on alg8_mpc_adaptive.py, optimization content removed


Obstacles are in the form: [center_x, center_y, radius]

To call with a specific np seed, do /alg10_mpc_w_custom_cost.py <seed_number>

"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
import time
from multi_obj_opt.strategies import ExtensionOptimizer
import numpy as np
from typing import Optional
from mpc_safety_filter import make_mpc_safety_filter

# ── Colors ──
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

class VerificationTask:
    """Represents an in-progress symbolic verification toward a conflict_time."""
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start
        self.conflict_time  = conflict_time
        self.time_invested = 0

    def done(self):
        return self.symbolic_start >= self.conflict_time


def concrete_scan(tester, from_t, to_t):
    """Run concrete propagation and return (collision: bool, collision_t: int | None)."""
    result = tester.concrete(from_t, to_t)
    if result["collision"]:
        return True, result["collision_timestep"]
    return False, None


def symbolic_step(tester, job: VerificationTask, chunk_size: int, budget):
    """
    Advance a VerificationTask by one chunk.
    time_invested accumulates budget.timestep_budget (seconds) each deferred step.
    Runs when the remaining cost fits within budget.remaining.
    Returns (updated_job, result_dict | None).
    result_dict is non-None only when the job is complete after this step.
    """
    steps = min(chunk_size, job.conflict_time - job.symbolic_start)
    cost  = budget.symbolic_costs.get(steps, float('inf'))
    if cost - job.time_invested <= budget.remaining:
        verify_end = min(job.conflict_time, job.symbolic_start + chunk_size)
        result = tester.symbolic(job.symbolic_start, verify_end)
        job.symbolic_start = verify_end
        job.time_invested = 0
        if job.done():
            return job, result
        return job, None
    else:
        job.time_invested += budget.remaining
        return job, None


def extend_mpc_sequence(mpc_sf, mpc_state: dict, min_safe_horizon: int,
                        max_time: int, ctrl_idx: int = 0):
    """
    Extend the MPC control queue from the end of the current trajectory.
    If extension from the endpoint fails, works backwards along the existing
    trajectory (like find_stopping_timestep) to find a viable extension point.
    """
    committed_at = mpc_state['committed_at']
    controls     = mpc_state['controls']
    traj_bounds  = mpc_state['traj_bounds']
    n_controls   = len(controls)
    mpc_end      = committed_at + n_controls

    if mpc_end >= max_time:
        print(f"  [MPC extend] Reached MAX_TIME, done")
        return

    # Don't look back past current execution point
    max_lb = min(n_controls - ctrl_idx - 1, mpc_sf.n_horizon)
    max_lb = max(max_lb, 0)

    for lb in range(max_lb + 1):
        # lb=0 → extend from the very end; lb>0 → lb steps back
        # traj_bounds has n_controls+1 entries (initial + one per control)
        try_idx = len(traj_bounds) - 1 - lb
        if try_idx < 1:
            break

        try_bounds = traj_bounds[try_idx]
        center     = (try_bounds[:, 0] + try_bounds[:, 1]) / 2.0

        try:
            new_traj_bounds, _, new_controls = mpc_sf._run_mpc_from_bounds(
                try_bounds, center, extra_inflation=0.0)
        except Exception as e:
            print(f"  [MPC extend] Failed at lookback={lb}: {e}")
            continue

        # Keep only the collision-free prefix
        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides(b):
                break
            safe_count += 1

        if safe_count <= lb:
            print(f"  [MPC extend] lookback={lb}: only {safe_count} safe "
                  f"(need >{lb}), going further back")
            continue

        # Replace tail (lb controls) with new safe path
        keep = n_controls - lb
        mpc_state['controls']    = list(controls[:keep]) + list(new_controls[:safe_count])
        mpc_state['traj_bounds'] = list(traj_bounds[:try_idx + 1]) + list(new_traj_bounds[1:safe_count + 1])

        net_gain  = safe_count - lb
        new_total = len(mpc_state['controls'])
        new_end   = committed_at + new_total
        if lb > 0:
            print(f"  [MPC extend] lookback={lb}: +{safe_count} safe, "
                  f"-{lb} replaced, net +{net_gain}")
        print(f"  [MPC extend] +{net_gain} controls → {new_total} total "
              f"(t={committed_at} to t={new_end})")
        return

    print(f"  [MPC extend] No safe extension found after {max_lb + 1} "
          f"lookback attempts")


def apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state: dict, current_t: int = 0,
                     min_safe_horizon: int = 0, max_time: int = 60):
    """
    Run MPC safety filter for a collision at conflict_time.
    Updates mpc_state with the committed plan and sets mpc_needed=True.
    If min_safe_horizon > 0, recursively extends the control queue.
    """
    t0 = time.perf_counter()
    stopping_t, controls, traj_bounds, _ = mpc_sf.find_stopping_timestep(conflict_time, tester.horizons, current_t)
    print(f"  [MPC timing] find_stopping_timestep took {time.perf_counter() - t0:.3f}s")

    if stopping_t is None:
        print(f"  [MPC] No safe stopping timestep — continuing nominal.")
        return None

    committed_at = mpc_state.get('committed_at')
    if committed_at is None or stopping_t >= committed_at:
        mpc_state['committed_at']  = stopping_t
        mpc_state['conflict_time'] = conflict_time
        mpc_state['controls']      = controls
        mpc_state['traj_bounds']   = traj_bounds
        mpc_state['needed']        = True
        print(f"  [MPC COMMIT] t_back={stopping_t}  {len(controls)} controls: "
              f"{[round(float(c[0]), 4) for c in controls]}")
    else:
        print(f"  [MPC] Keeping existing plan at t={committed_at} over t={stopping_t}")
    return stopping_t


def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon,
               current_timestep, min_lookahead, mpc_sf, mpc_state, safe_horizon_ceiling=None):
    """
    Greedily push validated_until toward max_time (or safe_horizon_ceiling).

    Returns (new_validated_until, pending_job | None, mpc_conflict | None).
    mpc_conflict is set when a confirmed collision needs MPC to be run.
    """
    scan_ceiling = safe_horizon_ceiling if safe_horizon_ceiling is not None else max_time

    while validated_until < scan_ceiling:
        if not budget.can_afford('concrete'):
            print(f"[extend] Budget exhausted before concrete scan")
            break

        end_check_time = min(
            validated_until + budget.max_affordable_concrete(),
            scan_ceiling,
            max_time
        )
        print(f"[extend] Concrete scan: t={validated_until} -> t={end_check_time}")
        collision, conflict_time = concrete_scan(tester, validated_until, end_check_time)

        if not collision:
            validated_until = end_check_time
            break

        print(f"[extend] Conflict at t={conflict_time}, attempting symbolic verification")

        # Pre-compute MPC as backup before symbolic verification
        apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state, current_timestep)

        if not budget.can_afford('symbolic', 1):
            print(f"[extend] No budget for symbolic — stopping before conflict")
            validated_until = conflict_time - 1
            return validated_until, VerificationTask(
                symbolic_start=conflict_time - 1,
                conflict_time=conflict_time
            ), conflict_time

        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)

        if result is None:
            print(f"[extend] Symbolic incomplete at t={job.symbolic_start}, carrying over")
            validated_until = job.symbolic_start
            return validated_until, job, None

        if result["collision"]:
            print(f"[extend] Conflict confirmed at t={conflict_time} — recomputing MPC with tighter bounds")
            validated_until = max(validated_until, conflict_time - 1)
            return validated_until, None, conflict_time
        else:
            validated_until = conflict_time
            print(f"[extend] Deconflicted — validated until t={conflict_time}")

    return validated_until, None, None


def old_optimized_step(tester, validated_until, max_time, budget, max_symbolic_horizon,
                   current_timestep, min_lookahead, ext_optimizer, mpc_sf, mpc_state):
    """
    Extend the verified horizon by one step to T+1, using the method chosen
    by the optimizer.

    Returns (new_validated_until, pending_job | None, mpc_conflict | None).
    mpc_conflict is set when a confirmed collision needs MPC to be run.
    """
    target = validated_until + 1
    if target > max_time:
        return validated_until, None, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    method = ext_optimizer.get_strategy_opt_free(
        current_timestep = current_timestep,
        verified_until   = validated_until,
        current_vol      = current_vol,
        verified_vol     = verified_vol,
        # time_budget      = budget.remaining,
        time_budget      = budget.remaining + budget.timestep_budget * (max_symbolic_horizon - 1),
        w_vol            = 50.0,
        pow_vol          = 3, # 2,
    )

    if method == "concrete":
        print(f"[opt_step] Concrete: t={validated_until} -> t={target}")
        # print(f"{RED}[opt_step] Concrete: t={validated_until} -> t={target}{RESET}")
        result        = tester.concrete(validated_until, target)
        conflict_time = result.get("collision_timestep")
    else:
        full_span     = target - current_timestep
        actual_k      = min(full_span, max_symbolic_horizon)
        target        = current_timestep + actual_k
        print(f"[opt_step] Symbolic: t={current_timestep} -> t={target} (span={actual_k})")
        # print(f"{BLUE}[opt_step] Symbolic: t={current_timestep} -> t={target} (span={actual_k}){RESET}")
        result        = tester.symbolic(current_timestep, target)
        conflict_time = result.get("collision_timestep")

    if not result["collision"]:
        print(f"[opt_step] Clean — validated until t={target}")
        return target, None, None

    print(f"[opt_step] Collision at t={conflict_time} — deconflicting from t={current_timestep}")

    if not budget.can_afford('symbolic', 1):
        return conflict_time - 1, VerificationTask(
            symbolic_start=current_timestep,
            conflict_time=conflict_time
        ), None

    job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)

    if result is None:
        return conflict_time - 1, job, None

    if result["collision"]:
        print(f"[opt_step] Conflict confirmed at t={conflict_time} — queuing MPC")
        return conflict_time - 1, None, conflict_time

    print(f"[opt_step] Deconflicted — validated until t={conflict_time}")
    return conflict_time, None, None

def optimized_step(tester, validated_until, max_time, budget, max_symbolic_horizon,
                   current_timestep, min_lookahead, ext_optimizer, mpc_sf, mpc_state):
    """
    Extend the verified horizon by one step to T+1, using the method chosen
    by the optimizer.

    Returns (new_validated_until, pending_job | None, mpc_conflict | None).
    mpc_conflict is set when a confirmed collision needs MPC to be run.
    """
    target = validated_until + 1
    if target > max_time:
        return validated_until, None, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    method = ext_optimizer.get_strategy_opt_free(
        current_timestep = current_timestep,
        verified_until   = validated_until,
        current_vol      = current_vol,
        verified_vol     = verified_vol,
        # time_budget      = budget.remaining,
        time_budget      = budget.remaining + budget.timestep_budget * (max_symbolic_horizon - 1),
        w_vol            = 50.0,
        pow_vol          = 3, # 2,
    )

    # concrete
    if method == "concrete":
        print(f"[opt_step] Concrete: t={validated_until} -> t={target}")
        result        = tester.concrete(validated_until, target)
        conflict_time = result.get("collision_timestep")

        if not result["collision"]:
            print(f"[opt_step] Clean — validated until t={target}")
            return target, None, None
        else:
            print(f"[opt_step] Collision at t={conflict_time} — deconflicting from t={current_timestep}")

            # Pre-compute MPC as backup before symbolic verification
            apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state, current_timestep)

            if not budget.can_afford('symbolic', 1):
                return conflict_time - 1, VerificationTask(
                    symbolic_start=current_timestep,
                    conflict_time=conflict_time
                ), conflict_time

            job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
            job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)

            if result is None:
                print(f"[opt_step] Symbolic incomplete at t={job.symbolic_start}, carrying over")
                return conflict_time - 1, job, None

            if result["collision"]:
                print(f"[opt_step] Conflict confirmed at t={conflict_time} — recomputing MPC with tighter bounds")
                validated_until = max(validated_until, conflict_time - 1)
                return validated_until, None, conflict_time

            validated_until = conflict_time+1
            print(f"[opt_step] Validated until t={validated_until}")
            return validated_until, None, None
    # symbolic
    else:
        job = VerificationTask(symbolic_start=current_timestep, conflict_time=target)
        job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)

        if result is None:
            print(f"[opt_step] Symbolic incomplete at t={job.symbolic_start}, carrying over")
            return validated_until, job, None

        if result["collision"]:
            print(f"[opt_step] Conflict confirmed at t={conflict_time} — queuing MPC")
            return conflict_time - 1, None, conflict_time

        validated_until = target+1
        print(f"[opt_step] Validated until t={validated_until}")
        return validated_until, None, None


def _get_volume(tester, timestep):
    horizon = tester.horizons.get(timestep)
    if horizon is None:
        return 1.0
    vol = horizon.get_tight_volume()
    return float(vol) if vol is not None else 1.0


####  simulation loop  ####

def get_dynamic_symbolic_horizon(safety_margin, max_symbolic_horizon=10, symbolic_buffer=5):
    return max_symbolic_horizon
    return min(max_symbolic_horizon, max(1, safety_margin-symbolic_buffer))
    return max_symbolic_horizon

def test(seed=None):
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')


    obstacles = [
        np.array([-6.5, 2.02,  0.5]),
        np.array([-3.2,  1.21,  0.5]),
        np.array([-1.5,  -0.85, 0.45]),
    ]

    tester             = ReachabilityTester(analyzer, obstacles, seed=seed)
    tester_calibration = ReachabilityTester(analyzer)
    mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=10, nominal_tracking=True)

    MIN_LOOKAHEAD        = 4
    MIN_SAFE_HORIZON     = 8
    # MIN_SAFE_HORIZON     = 12
    MAX_SYMBOLIC_HORIZON = 10
    SYMBOLIC_BUFFER      = 5
    MAX_TIME             = 60

    budget = TimeBudget(timestep_budget=0.50)
    # budget.calibrate(tester_calibration, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
    #                  max_backward_horizon=0)

    budget.symbolic_costs = {1: 0.05942702293395996, 2: 0.0532071590423584, 3: 0.12308859825134277, 4: 0.2227306365966797, 5: 0.3548123836517334, 6: 0.5160810947418213, 7: 0.7076215744018555, 8: 1.046485185623169, 9: 1.189185619354248, 10: 1.4745268821716309}
    budget.concrete_cost = 0.012865893046061198

    ext_optimizer = ExtensionOptimizer()
    ext_optimizer.set_timing_params({
        "concrete_slope":     0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope":        0.5341915550605524,
        "ratio_intercept":   -0.0578267576662421,
    })

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    current_timestep          = 0
    validated_until           = 0
    pending_job: Optional[VerificationTask] = None
    pending_mpc_conflict: Optional[int]   = None   # conflict_time for MPC, used when alternating with symbolic
    mpc_turn                  = False   # True = MPC gets this timestep when both pending
    mpc_first_run              = True   # first confirmed collision runs MPC immediately
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'traj_bounds':   [],   # traj_bounds[i+1] = MPC bounds at committed_at + i + 1
        'needed':        False,
    }
    mpc_started = False

    def trigger_mpc(conflict_time):
        """First collision: run MPC now. Subsequent: queue and alternate."""
        print(f"{RED}Triggered MPC calculation with conflict time {conflict_time}{RESET}")
        nonlocal mpc_first_run, pending_mpc_conflict
        if mpc_first_run:
            apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state, current_timestep,
                             min_safe_horizon=MIN_SAFE_HORIZON, max_time=MAX_TIME)
            mpc_first_run = False
        else:
            pending_mpc_conflict = conflict_time

    def run_opt_loop():
        """Run the optimizer while loop until budget, margin, or a pending job stops it."""
        nonlocal validated_until, pending_job, dynamic_symbolic_horizon
        while (validated_until - current_timestep >= MIN_SAFE_HORIZON
               and validated_until < MAX_TIME
            #    and pending_job is None
               and pending_mpc_conflict is None
               and budget.remaining > 0):
            validated_until, pending_job, mpc_c = optimized_step(
                tester, validated_until, MAX_TIME, budget,
                dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                ext_optimizer, mpc_sf, mpc_state
            )
            if pending_job is not None:
                return pending_job, None
            if mpc_c is not None:
                trigger_mpc(mpc_c)
            dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                validated_until - current_timestep,
                max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                symbolic_buffer=SYMBOLIC_BUFFER
            )

        return None, None


    while current_timestep < MAX_TIME:
        budget.start_timestep()

        safety_margin = validated_until - current_timestep
        dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
            safety_margin,
            max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
            symbolic_buffer=SYMBOLIC_BUFFER
        )

        #  MPC active: re-check each timestep whether conflict persists.
        #  If concrete scan (MIN_SAFE_HORIZON) is clean, revert to nominal controller.
        #  If conflict still present, apply the queued MPC control.
        if mpc_started:
            ctrl_idx  = current_timestep - mpc_state['committed_at']
            queue     = mpc_state['controls']
            check_end = min(current_timestep + MIN_SAFE_HORIZON, MAX_TIME)

            # Extend when remaining controls <= n_horizon
            # Retries each step if previous extend failed
            n_h = mpc_sf.n_horizon
            if len(queue) - ctrl_idx <= n_h:
                extend_mpc_sequence(mpc_sf, mpc_state, MIN_SAFE_HORIZON, MAX_TIME,
                                    ctrl_idx=ctrl_idx)
                queue = mpc_state['controls']

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]

            print(f"[MPC check] t={current_timestep}  Scanning concrete t={current_timestep}→{check_end}")
            conflict_still_present, _ = concrete_scan(tester, current_timestep, check_end)

            if not conflict_still_present:
                # Safe to hand back to nominal controller
                print(f"[MPC REVERT] t={current_timestep}  No conflict in horizon — reverting to nominal")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None
                mpc_first_run              = True
                validated_until            = check_end
                pending_job                = None   # stale — conflict resolved by MPC
                pending_mpc_conflict       = None
                tester.real_state_empirical(current_timestep, current_timestep + 1)
                print(f"Safety margin: {validated_until - current_timestep} steps ahead"
                      f"  |  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
            elif ctrl_idx < len(queue):
                ctrl = queue[ctrl_idx]
                print(f"[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                      f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})")
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                print(f"[MPC] Queue nearly exhausted at t={current_timestep}, attempting extension")
                extend_mpc_sequence(mpc_sf, mpc_state, MIN_SAFE_HORIZON, MAX_TIME,
                                    ctrl_idx=ctrl_idx)
                queue = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    print(f"[MPC EXTENDED] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                          f"  u={np.round(ctrl, 4)}")
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    print(f"[MPC] Queue exhausted even after extension at t={current_timestep}")
                    break
            continue




        both_pending = pending_job is not None and pending_mpc_conflict is not None
        mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
        color = GREEN if mode == "OPTIMIZED" else BLUE
        print(f"\n{color}CURRENT TIMESTEP ==== {current_timestep}  "
              f"vu={validated_until}  margin={safety_margin}  [{mode}]{RESET}"
              + (f"  [MPC queued t_conflict={pending_mpc_conflict}]"
                 if pending_mpc_conflict is not None else ""))

        # Whether to run MPC or symbolic for this timestep.
        # If no commit yet, keep retrying MPC every timestep (don't alternate).
        # Only alternate once a commit has been found.
        has_commit = mpc_state['committed_at'] is not None and mpc_state['needed']
        do_mpc_this_step = (
            pending_mpc_conflict is not None and
            (pending_job is None or not has_commit or mpc_turn)
        )

        if do_mpc_this_step:
            print(f"[Alt] MPC turn — running filter for conflict_time={pending_mpc_conflict}")
            apply_mpc_filter(mpc_sf, pending_mpc_conflict, tester, mpc_state, current_timestep,
                             min_safe_horizon=MIN_SAFE_HORIZON, max_time=MAX_TIME)
            pending_mpc_conflict = None
            mpc_turn = False

        else:
            if both_pending:
                # symbolic gets this timestep; MPC gets the next
                mpc_turn = True

            #  Phase 1: carry-over symbolic
            if pending_job is not None:
                print(f"[Carry-over] Resuming symbolic: "
                      f"t={pending_job.symbolic_start} -> t={pending_job.conflict_time}")
                pending_job, result = symbolic_step(tester, pending_job,
                                                    dynamic_symbolic_horizon,
                                                    budget)

                if result is not None:
                    conflict_time = pending_job.conflict_time
                    pending_job   = None
                    if result["collision"]:
                        print(f"Conflict confirmed at t={conflict_time} — triggering MPC")
                        validated_until = max(validated_until, conflict_time - 1)
                        trigger_mpc(conflict_time)
                    else:
                        print(f"Deconflicted — validated until t={conflict_time}")
                        validated_until = conflict_time
                        if budget.remaining > 0:
                            validated_until, pending_job, mpc_c = try_extend(
                                tester, validated_until, MAX_TIME, budget,
                                dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                mpc_sf, mpc_state,
                                safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON,
                            )
                            if mpc_c is not None:
                                trigger_mpc(mpc_c)
                            pending_job, result = run_opt_loop()

            #  Phase 2: main decision when no carry-over pending
            else:
                safety_margin = validated_until - current_timestep
                dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                    safety_margin,
                    max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                    symbolic_buffer=SYMBOLIC_BUFFER
                )

                if safety_margin >= MIN_SAFE_HORIZON:
                    pending_job, result = run_opt_loop()
                    if pending_job is None:
                        new_margin = validated_until - current_timestep
                        if new_margin < MIN_SAFE_HORIZON and pending_job is None and pending_mpc_conflict is None and budget.remaining > 0:
                            print(f"[opt] Margin dropped — recovering with baseline extend")
                            validated_until, pending_job, mpc_c = try_extend(
                                tester, validated_until, MAX_TIME, budget,
                                dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                mpc_sf, mpc_state,
                                safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON,
                            )
                            if mpc_c is not None:
                                trigger_mpc(mpc_c)
                            pending_job, result = run_opt_loop()

                elif validated_until < MAX_TIME:
                    explore_from   = max(validated_until, current_timestep)
                    end_check_time = min(
                        explore_from + budget.max_affordable_concrete(),
                        current_timestep + MIN_SAFE_HORIZON,
                        MAX_TIME
                    )
                    collision, conflict_time = concrete_scan(tester, explore_from, end_check_time)

                    if not collision:
                        validated_until = end_check_time
                        if budget.remaining > 0:
                            pending_job, result = run_opt_loop()
                    else:
                        print(f"Conflict detected at t={conflict_time}")
                        # Pre-compute MPC as backup before symbolic verification
                        trigger_mpc(conflict_time)
                        pending_job = VerificationTask(
                            symbolic_start=current_timestep,
                            conflict_time=conflict_time
                        )
                        pending_job, result = symbolic_step(tester, pending_job,
                                                            dynamic_symbolic_horizon,
                                                            budget)
                        if result is not None:
                            pending_job = None
                            if result["collision"]:
                                print(f"Conflict confirmed at t={conflict_time} — recomputing MPC with tighter bounds")
                                validated_until = max(validated_until, conflict_time - 1)
                                trigger_mpc(conflict_time)
                            else:
                                print(f"Deconflicted — validated until t={conflict_time}")
                                validated_until = conflict_time
                                if budget.remaining > 0:
                                    new_margin = validated_until - current_timestep
                                    if new_margin < MIN_SAFE_HORIZON:
                                        validated_until, pending_job, mpc_c = try_extend(
                                            tester, validated_until, MAX_TIME, budget,
                                            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                            mpc_sf, mpc_state,
                                            safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON,
                                        )
                                        if mpc_c is not None:
                                            trigger_mpc(mpc_c)
                                    pending_job, result = run_opt_loop()
                        else:
                            validated_until = pending_job.symbolic_start
                            print(f"Budget exhausted at t={validated_until}, deferring")

        # Clear mpc_needed if we deconflicted sufficiently (MIN_SAFE_HORIZON) past the conflict point
        if (mpc_state['conflict_time'] is not None
                and validated_until >= mpc_state['conflict_time'] + MIN_SAFE_HORIZON):
            if mpc_state['needed']:
                print(f"  [MPC] Deconflicted past conflict at t={mpc_state['conflict_time']}, "
                      f"cancelling MPC committed at t={mpc_state['committed_at']}")
                if current_timestep >= mpc_state['committed_at'] and validated_until < MAX_TIME:
                    print(f"  [MPC] Recomputing for new horizon boundary t={validated_until}")
                    trigger_mpc(validated_until)
            mpc_state['needed'] = False
            mpc_first_run = True  # reset so next conflict runs MPC immediately

        #  Advance real state: MPC or NN
        if (mpc_state['committed_at'] is not None
                and current_timestep >= mpc_state['committed_at']
                and mpc_state['needed']):
            #MPC Control

            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']
            if ctrl_idx < len(queue):
                mpc_started = True
                ctrl = queue[ctrl_idx]
                print(f"[MPC FIRST FIRE] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                      f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']},"
                      f" {len(queue)} controls total)")
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                print(f"[MPC] Queue exhausted immediately at t={current_timestep}. ERROR???")
                break
        else:
            # Nominal Controller
            tester.real_state_empirical(current_timestep, current_timestep + 1)
            print(f"Safety margin: {validated_until - current_timestep} steps ahead"
                  f"  |  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

    # collect real states from each timestep's horizon
    state_history = []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(calc['real_state'].copy())
                break

    # check each real state against obstacles
    collision_timesteps = []
    for i, s in enumerate(state_history):
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx)**2 + (s_flat[1] - cy)**2 <= r**2:
                collision_timesteps.append(i)
                break

    had_collision = len(collision_timesteps) > 0
    print(f"\n{'='*60}")
    print(f"Simulation complete at timestep {current_timestep}")
    if had_collision:
        print(f"[SAFETY] COLLISION at real-state timesteps: {collision_timesteps}")
    else:
        print(f"[SAFETY] No real-state collision detected")
    return state_history, had_collision


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=seed)
