"""
Optimized extension algorithm matching anim_alg5_optimization.py.

When validated_until is at least MIN_SAFE_HORIZON ahead of current_timestep,
the ExtensionOptimizer chooses whether to compute T+1 via:
  - concrete: tester.concrete(validated_until, T+1)
  - symbolic: tester.symbolic(current_timestep, T+1)  — tighter bounds

The optimizer loops within each timestep until budget is exhausted, a
pending job is created, or the margin drops below MIN_SAFE_HORIZON.

The baseline concrete scan is capped at current_timestep + MIN_SAFE_HORIZON
so the optimizer always gets a turn once the safe margin is reached.
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
import time
from multi_obj_opt.strategies import ExtensionOptimizer
import numpy as np
from typing import Optional
from mpc_safety_filter import make_mpc_safety_filter


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


# def symbolic_step(tester, job: VerificationTask, chunk_size: int, timestep_length):
#     """
#     Advance a VerificationTask by one chunk.
#     Returns (updated_job, result_dict | None).
#     result_dict is non-None only when the job is complete after this step.
#     """
#     verify_end = min(job.conflict_time, job.symbolic_start + chunk_size)
#     result = tester.symbolic(job.symbolic_start, verify_end)
#     job.symbolic_start = verify_end

#     if job.done():
#         return job, result
#     return job, None


def symbolic_step(tester, job: VerificationTask, chunk_size: int, timestep_length):
    """
    Advance a VerificationTask by one chunk.
    Returns (updated_job, result_dict | None).
    result_dict is non-None only when the job is complete after this step.
    """
    # The chunk can be calculated in a single timestep
    if (chunk_size - job.time_invested <= timestep_length):
        verify_end = min(job.conflict_time, job.symbolic_start + chunk_size)
        result = tester.symbolic(job.symbolic_start, verify_end)
        job.symbolic_start = verify_end
        job.time_invested = 0
        if job.done():
            return job, result
        return job, None

    # The chunk is not complete and must continue to be calculated
    else:
        job.time_invested += timestep_length
        return job, None




def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon,
               current_timestep, min_lookahead, mpc_sf, mpc_state, safe_horizon_ceiling=None):
    """
    Greedily push validated_until toward max_time (or safe_horizon_ceiling)
    using concrete scans, falling back to symbolic to deconflict collisions.

    Returns (new_validated_until, pending_job | None).
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

        if not budget.can_afford('symbolic', 1):
            print(f"[extend] No budget for symbolic — stopping before conflict")
            validated_until = conflict_time - 1
            return validated_until, VerificationTask(
                symbolic_start=conflict_time - 1,
                conflict_time=conflict_time
            )

        # chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        # job, result = symbolic_step(tester, job, chunk_size)
        job, result = symbolic_step(tester, job, max_symbolic_horizon, budget.max_affordable_symbolic())

        if result is None:
            print(f"[extend] Symbolic incomplete at t={job.symbolic_start}, carrying over")
            validated_until = job.symbolic_start
            return validated_until, job

        if result["collision"]:
            print(f"[extend] Conflict confirmed at t={conflict_time} — running MPC filter")
            validated_until = max(validated_until, conflict_time - 1)
            apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state)
            return validated_until, None
        else:
            validated_until = conflict_time
            print(f"[extend] Deconflicted — validated until t={conflict_time}")

    return validated_until, None


def optimized_step(tester, validated_until, max_time, budget, max_symbolic_horizon,
                   current_timestep, min_lookahead, ext_optimizer, mpc_sf, mpc_state):
    """
    Extend the verified horizon by one step to T+1, using the method chosen
    by the optimizer.

      concrete: tester.concrete(validated_until, T+1)
      symbolic: tester.symbolic(current_timestep, T+1) — tighter bounds

    If the chosen method finds a collision, deconflict using the original
    procedure: symbolic from current_timestep to conflict_time, chunked.

    Returns (new_validated_until, pending_job | None).
    """
    target = validated_until + 1
    if target > max_time:
        return validated_until, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    method = ext_optimizer.get_strategy(
        current_timestep = current_timestep,
        verified_until   = validated_until,
        current_vol      = current_vol,
        verified_vol     = verified_vol,
        time_budget      = budget.remaining,
    )

    # ── Perform the chosen extension step ─────────────────────────────
    if method == "concrete":
        print(f"[opt_step] Concrete: t={validated_until} -> t={target}")
        result        = tester.concrete(validated_until, target)
        conflict_time = result.get("collision_timestep")

    else:  # symbolic
        full_span     = target - current_timestep
        # actual_k      = min(full_span, max_symbolic_horizon, budget.max_affordable_symbolic())
        actual_k      = min(full_span, max_symbolic_horizon)
        target        = current_timestep + actual_k   # may be capped by budget
        print(f"[opt_step] Symbolic: t={current_timestep} -> t={target} (span={actual_k})")
        result        = tester.symbolic(current_timestep, target)
        conflict_time = result.get("collision_timestep")

    if not result["collision"]:
        print(f"[opt_step] Clean — validated until t={target}")
        return target, None

    # ── Collision: deconflict with symbolic from current_timestep ─────
    print(f"[opt_step] Collision at t={conflict_time} — deconflicting from t={current_timestep}")

    if not budget.can_afford('symbolic', 1):
        return conflict_time - 1, VerificationTask(
            symbolic_start=conflict_time - 1,
            conflict_time=conflict_time
        )

    # chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
    job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    # job, result = symbolic_step(tester, job, chunk_size)
    job, result = symbolic_step(tester, job, max_symbolic_horizon, budget.max_affordable_symbolic())

    if result is None:
        return job.symbolic_start, job

    if result["collision"]:
        print(f"[opt_step] Conflict confirmed at t={conflict_time} — running MPC filter")
        apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state)
        return conflict_time - 1, None

    print(f"[opt_step] Deconflicted — validated until t={conflict_time}")
    return conflict_time, None


def _get_volume(tester, timestep):
    """Return tight-bound volume at a timestep, or 1.0 as a safe fallback."""
    horizon = tester.horizons.get(timestep)
    if horizon is None:
        return 1.0
    vol = horizon.get_tight_volume()
    return float(vol) if vol is not None else 1.0

def apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state: dict):
    """
    Run MPC safety filter for a collision at conflict_time.
    Updates mpc_state with the committed plan and sets mpc_needed=True.
    Returns latest stopping timestep
    """
    t0 = time.perf_counter()
    stopping_t, controls = mpc_sf.find_stopping_timestep(conflict_time, tester.horizons)
    print(f"  [MPC timing] find_stopping_timestep took {time.perf_counter() - t0:.3f}s\n"
          f"  {conflict_time-2-stopping_t} iterations")

    if stopping_t is not None:
        committed_at = mpc_state.get('committed_at') # previously committed mpc
        if committed_at is None or stopping_t >= committed_at:
            #replace if updated
            mpc_state['committed_at']  = stopping_t
            mpc_state['conflict_time'] = conflict_time
            mpc_state['controls']      = controls
            mpc_state['needed']        = True
            print(f"  [MPC COMMIT] t_back={stopping_t}  {len(controls)} controls: "
                  f"{[round(float(c[0]), 4) for c in controls]}")
        else:
            print(f"  [MPC] Keeping existing plan at t={committed_at} over t={stopping_t}")
        return stopping_t
    else:
        #if mpc fails
        fallback = conflict_time - 1
        print(f"  [MPC] No safe stopping timestep found — falling back to t={fallback}")
        return fallback



####  simulation loop  ####

def get_dynamic_symbolic_horizon(safety_margin, max_symbolic_horizon=10, symbolic_buffer=5):
    return max(max_symbolic_horizon, safety_margin - symbolic_buffer)

def test1():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    obstacles = [
        np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),
        np.array([[-np.inf, 0.3],    [-np.inf, np.inf]]),
    ]

    tester             = ReachabilityTester(analyzer, obstacles)
    tester_calibration = ReachabilityTester(analyzer)

    mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, t_step=0.1, n_horizon=10)

    # obstacles = [
    # np.array([[-5.5, -5], [2, 2.2 ], [-np.inf, np.inf]]),
    # ]
    # analyzer           = setup_analyzer('Unicycle_NL', 'natural_none_default')
    # tester             = ReachabilityTester(analyzer, obstacles)
    # tester_calibration = ReachabilityTester(analyzer)
    # mpc_sf             = make_mpc_safety_filter(tester, obstacles_list=obstacles, t_step=0.1)


    MIN_LOOKAHEAD        = 4
    MIN_SAFE_HORIZON     = 6
    MAX_SYMBOLIC_HORIZON = 10
    SYMBOLIC_BUFFER = 5
    MAX_TIME             = 40

    budget = TimeBudget(timestep_budget=1.0)
    budget.calibrate(tester_calibration, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                     max_backward_horizon=0)

    ext_optimizer = ExtensionOptimizer()
    ext_optimizer.set_timing_params({
        "concrete_slope":     0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope":        0.5341915550605524,
        "ratio_intercept":   -0.0578267576662421,
    })

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    current_timestep = 0
    validated_until  = 0
    pending_job: Optional[VerificationTask] = None
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'needed':        False,
    }
    mpc_started = False  # locked True once first MPC control fires


    while current_timestep < MAX_TIME:
        budget.start_timestep()

        safety_margin = validated_until - current_timestep
        dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
            safety_margin,
            max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
            symbolic_buffer=SYMBOLIC_BUFFER
        )
        #  MPC locked: skip all verification, just apply controls
        if mpc_started:
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']
            if ctrl_idx < len(queue):
                ctrl = queue[ctrl_idx]
                print(f"[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                      f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})")
                current_timestep += 1
            else:
                #all mpc control applied
                print(f"[MPC] Queue exhausted at t={current_timestep} "
                      f"({len(queue)} steps from t={mpc_state['committed_at']})")
                break
            continue

        mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
        print(f"\nCURRENT TIMESTEP ==== {current_timestep}  "
              f"vu={validated_until}  margin={safety_margin}  [{mode}]")

        # ── Phase 1: carry-over symbolic (always runs, regardless of mode) ──
        if pending_job is not None:
            print(f"[Carry-over] Resuming symbolic: "
                  f"t={pending_job.symbolic_start} -> t={pending_job.conflict_time}")
            # chunk_size  = min(budget.max_affordable_symbolic(), dynamic_symbolic_horizon)
            # pending_job, result = symbolic_step(tester, pending_job, chunk_size)
            pending_job, result = symbolic_step(tester, pending_job, dynamic_symbolic_horizon, budget.max_affordable_symbolic())

            if result is not None:
                conflict_time = pending_job.conflict_time
                pending_job   = None
                if result["collision"]:
                    print(f"Conflict confirmed at t={conflict_time} — running MPC filter")
                    validated_until = max(validated_until, conflict_time - 1)
                    apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state)
                else:
                    print(f"Deconflicted — validated until t={conflict_time}")
                    validated_until = conflict_time
                    if budget.remaining > 0:
                        validated_until, pending_job = try_extend(
                            tester, validated_until, MAX_TIME, budget,
                            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                            mpc_sf, mpc_state,
                            safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                        )

        # Phase 2: main decision — optimized or baseline
        else:
            safety_margin = validated_until - current_timestep  # recheck after carry-over
            dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                safety_margin,
                max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                symbolic_buffer=SYMBOLIC_BUFFER
            )

            if safety_margin >= MIN_SAFE_HORIZON:
                #  OPTIMIZED: loop until budget gone, horizon maxed, or
                # a pending job is created (collision mid-step)
                while (validated_until - current_timestep >= MIN_SAFE_HORIZON
                       and validated_until < MAX_TIME
                       and pending_job is None
                       and budget.remaining > 0):

                    validated_until, pending_job = optimized_step(
                        tester, validated_until, MAX_TIME, budget,
                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                        ext_optimizer, mpc_sf, mpc_state
                    )
                    dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                        validated_until - current_timestep,
                        max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                        symbolic_buffer=SYMBOLIC_BUFFER
                    )

                # If margin dropped below threshold (e.g. collision), recover
                new_margin = validated_until - current_timestep
                if new_margin < MIN_SAFE_HORIZON and pending_job is None and budget.remaining > 0:
                    print(f"[opt] Margin dropped to {new_margin} < {MIN_SAFE_HORIZON} "
                          f"— recovering with baseline extend")
                    validated_until, pending_job = try_extend(
                        tester, validated_until, MAX_TIME, budget,
                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                        mpc_sf, mpc_state,
                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                    )

            else:
                #  BASELINE: concrete scan capped at MIN_SAFE_HORIZON
                explore_from   = max(validated_until, current_timestep)
                end_check_time = min(
                    explore_from + budget.max_affordable_concrete(),
                    current_timestep + MIN_SAFE_HORIZON,
                    MAX_TIME
                )
                collision, conflict_time = concrete_scan(tester, explore_from, end_check_time)

                if not collision:
                    validated_until = end_check_time
                    new_margin = validated_until - current_timestep
                    if new_margin >= MIN_SAFE_HORIZON and budget.remaining > 0:
                        print(f"[baseline] Reached MIN_SAFE_HORIZON "
                              f"— handing off to optimizer")
                        validated_until, pending_job = optimized_step(
                            tester, validated_until, MAX_TIME, budget,
                            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                            ext_optimizer, mpc_sf, mpc_state
                        )
                else:
                    print(f"Conflict detected at t={conflict_time}")
                    # chunk_size  = min(budget.max_affordable_symbolic(), dynamic_symbolic_horizon)
                    pending_job = VerificationTask(
                        symbolic_start=current_timestep,
                        conflict_time=conflict_time
                    )
                    # pending_job, result = symbolic_step(tester, pending_job, chunk_size)
                    pending_job, result = symbolic_step(tester, pending_job, dynamic_symbolic_horizon, budget.max_affordable_symbolic())

                    if result is not None:
                        pending_job = None
                        if result["collision"]:
                            print(f"Conflict confirmed at t={conflict_time} — running MPC filter")
                            validated_until = max(validated_until, conflict_time - 1)
                            apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state)
                        else:
                            print(f"Deconflicted — validated until t={conflict_time}")
                            validated_until = conflict_time
                            if budget.remaining > 0:
                                new_margin = validated_until - current_timestep
                                if new_margin >= MIN_SAFE_HORIZON:
                                    print(f"[baseline] Deconflicted past MIN_SAFE_HORIZON "
                                          f"— handing off to optimizer")
                                    validated_until, pending_job = optimized_step(
                                        tester, validated_until, MAX_TIME, budget,
                                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                        ext_optimizer, mpc_sf, mpc_state
                                    )
                                else:
                                    validated_until, pending_job = try_extend(
                                        tester, validated_until, MAX_TIME, budget,
                                        dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                                        mpc_sf, mpc_state,
                                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                                    )
                    else:
                        validated_until = pending_job.symbolic_start
                        print(f"Budget exhausted at t={validated_until}, "
                              f"deferring to next timestep")

        if validated_until >= MAX_TIME:
            break

        # Clear mpc_needed if we deconflicted past the collision point we calculated the mpc for
        if (mpc_state['conflict_time'] is not None
                and validated_until >= mpc_state['conflict_time']):
            if mpc_state['needed']:
                print(f"  [MPC] Deconflicted past conflict at t={mpc_state['conflict_time']}, "
                      f"cancelling MPC committed at t={mpc_state['committed_at']}")

                # Recompute MPC for new horizon boundary only if plan was about to fire
                if current_timestep >= mpc_state['committed_at'] and validated_until < MAX_TIME:
                    print(f"  [MPC] Recomputing for new horizon boundary t={validated_until}")
                    apply_mpc_filter(mpc_sf, validated_until, tester, mpc_state)

                mpc_state['needed'] = False

        #  Advance real state: MPC or NN
        if (mpc_state['committed_at'] is not None
                and current_timestep >= mpc_state['committed_at']
                and mpc_state['needed']):

            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']
            if ctrl_idx < len(queue):
                mpc_started = True
                ctrl = queue[ctrl_idx]
                print(f"[MPC FIRST FIRE] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                      f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']},"
                      f" {len(queue)} controls total)")
                current_timestep += 1
            else:
                print(f"[MPC] Queue exhausted immediately at t={current_timestep}. ERROR???")
                break
        else:
            #propagate regular empirical step
            tester.real_state_empirical(current_timestep, current_timestep + 1)
            print(f"Safety margin: {validated_until - current_timestep} steps ahead"
                  f"  |  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

    print(f"\n{'='*60}")
    print(f"Simulation complete at timestep {current_timestep}")


if __name__ == "__main__":
    test1()
