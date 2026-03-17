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
from multi_obj_opt.strategies import ExtensionOptimizer
import numpy as np
from typing import Optional


class VerificationTask:
    """Represents an in-progress symbolic verification toward a conflict_time."""
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start
        self.conflict_time  = conflict_time

    def done(self):
        return self.symbolic_start >= self.conflict_time


def concrete_scan(tester, from_t, to_t):
    """Run concrete propagation and return (collision: bool, collision_t: int | None)."""
    result = tester.concrete(from_t, to_t)
    if result["collision"]:
        return True, result["collision_timestep"]
    return False, None


def symbolic_step(tester, job: VerificationTask, chunk_size: int):
    """
    Advance a VerificationTask by one chunk.
    Returns (updated_job, result_dict | None).
    result_dict is non-None only when the job is complete after this step.
    """
    verify_end = min(job.conflict_time, job.symbolic_start + chunk_size)
    result = tester.symbolic(job.symbolic_start, verify_end)
    job.symbolic_start = verify_end

    if job.done():
        return job, result
    return job, None


def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon,
               current_timestep, min_lookahead, safe_horizon_ceiling=None):
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

        chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        job, result = symbolic_step(tester, job, chunk_size)

        if result is None:
            print(f"[extend] Symbolic incomplete at t={job.symbolic_start}, carrying over")
            validated_until = job.symbolic_start
            return validated_until, job

        validated_until = conflict_time - 1 if result["collision"] else conflict_time
        force_stop = result["collision"] and (conflict_time - current_timestep) < min_lookahead
        print(
            f"Conflict confirmed at t={conflict_time} using symbolic"
            + (" — FORCED STOP" if force_stop else " — deferring")
            if result["collision"]
            else f"Deconflicted — validated until t={conflict_time}"
        )
        if force_stop or result["collision"]:
            return validated_until, None

    return validated_until, None


def optimized_step(tester, validated_until, max_time, budget, max_symbolic_horizon,
                   current_timestep, min_lookahead, ext_optimizer):
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
        actual_k      = min(full_span, max_symbolic_horizon, budget.max_affordable_symbolic())
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

    chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
    job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    job, result = symbolic_step(tester, job, chunk_size)

    if result is None:
        return job.symbolic_start, job

    if result["collision"]:
        force_stop = (conflict_time - current_timestep) < min_lookahead
        print(f"[opt_step] Conflict confirmed at t={conflict_time}"
              + (" — FORCED STOP" if force_stop else " — deferring"))
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


####  simulation loop  ####

def test1():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    obstacles = [
        np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),
        np.array([[-np.inf, 0.0],    [-np.inf, np.inf]]),
    ]
    tester             = ReachabilityTester(analyzer, obstacles)
    tester_calibration = ReachabilityTester(analyzer)

    MIN_LOOKAHEAD        = 4
    MIN_SAFE_HORIZON     = 6
    MAX_SYMBOLIC_HORIZON = 10
    MAX_TIME             = 40

    budget = TimeBudget(timestep_budget=0.40)
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

    while current_timestep < MAX_TIME:
        budget.start_timestep()

        safety_margin = validated_until - current_timestep
        mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
        print(f"\nCURRENT TIMESTEP ==== {current_timestep}  "
              f"vu={validated_until}  margin={safety_margin}  [{mode}]")

        # ── Phase 1: carry-over symbolic (always runs, regardless of mode) ──
        if pending_job is not None:
            print(f"[Carry-over] Resuming symbolic: "
                  f"t={pending_job.symbolic_start} -> t={pending_job.conflict_time}")
            chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
            pending_job, result = symbolic_step(tester, pending_job, chunk_size)

            if result is not None:
                conflict_time   = pending_job.conflict_time
                validated_until = conflict_time - 1 if result["collision"] else conflict_time
                force_stop      = result["collision"] and \
                                  (conflict_time - current_timestep) < MIN_LOOKAHEAD

                print(
                    f"Conflict confirmed at t={conflict_time}"
                    + (" — FORCED STOP" if force_stop else " — deferring")
                    if result["collision"]
                    else f"Deconflicted — validated until t={conflict_time}"
                )

                if force_stop:
                    break
                if not result["collision"] and budget.remaining > 0:
                    validated_until, pending_job = try_extend(
                        tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                    )
                else:
                    pending_job = None

        # ── Phase 2: main decision — optimized or baseline ─────────────────
        else:
            safety_margin = validated_until - current_timestep  # recheck after carry-over

            if safety_margin >= MIN_SAFE_HORIZON:
                # ── OPTIMIZED: loop until budget gone, horizon maxed, or
                # a pending job is created (collision mid-step) ───────────
                while (validated_until - current_timestep >= MIN_SAFE_HORIZON
                       and validated_until < MAX_TIME
                       and pending_job is None
                       and budget.remaining > 0):
                    validated_until, pending_job = optimized_step(
                        tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                        ext_optimizer
                    )

                # If margin dropped below threshold (e.g. collision), recover
                new_margin = validated_until - current_timestep
                if new_margin < MIN_SAFE_HORIZON and pending_job is None \
                        and budget.remaining > 0:
                    print(f"[opt] Margin dropped to {new_margin} < {MIN_SAFE_HORIZON} "
                          f"— recovering with baseline extend")
                    validated_until, pending_job = try_extend(
                        tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                        safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                    )

            else:
                # ── BASELINE: concrete scan capped at MIN_SAFE_HORIZON ────
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
                            MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                            ext_optimizer
                        )
                else:
                    print(f"Conflict detected at t={conflict_time}")
                    chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
                    pending_job = VerificationTask(
                        symbolic_start=current_timestep,
                        conflict_time=conflict_time
                    )
                    pending_job, result = symbolic_step(tester, pending_job, chunk_size)

                    if result is not None:
                        validated_until = conflict_time - 1 if result["collision"] \
                                          else conflict_time
                        force_stop = result["collision"] and \
                                     (conflict_time - current_timestep) < MIN_LOOKAHEAD
                        print(
                            f"Conflict confirmed at t={conflict_time}"
                            + (" — FORCED STOP" if force_stop else " — deferring")
                            if result["collision"]
                            else f"Deconflicted — validated until t={conflict_time}"
                        )
                        pending_job = None
                        if force_stop:
                            break
                        if not result["collision"] and budget.remaining > 0:
                            new_margin = validated_until - current_timestep
                            if new_margin >= MIN_SAFE_HORIZON:
                                print(f"[baseline] Deconflicted past MIN_SAFE_HORIZON "
                                      f"— handing off to optimizer")
                                validated_until, pending_job = optimized_step(
                                    tester, validated_until, MAX_TIME, budget,
                                    MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                                    ext_optimizer
                                )
                            else:
                                validated_until, pending_job = try_extend(
                                    tester, validated_until, MAX_TIME, budget,
                                    MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                                    safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                                )
                    else:
                        validated_until = pending_job.symbolic_start
                        print(f"Budget exhausted at t={validated_until}, "
                              f"deferring to next timestep")

        if validated_until >= MAX_TIME:
            break

        tester.real_state_empirical(current_timestep, current_timestep + 1)
        print(f"Safety margin: {validated_until - current_timestep} steps ahead")
        print(f"[Budget] {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s used")
        current_timestep += 1

    print(f"\n{'='*60}")
    print(f"Simulation complete at timestep {current_timestep}")


if __name__ == "__main__":
    test1()