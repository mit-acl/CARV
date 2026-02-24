"""
Updated refinement strategy. Keeps greedily extending the validated horizon as long as budget allows
but with more modular structure
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
import numpy as np
from typing import Optional

# Helper classes and functions for managing refinement tasks"""

class VerificationTask:
    """Represents an in-progress symbolic verification toward a conflict_time."""
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start  # where to resume symbolic next
        self.conflict_time  = conflict_time   # the collision timestep we're trying to deconflict

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
        return job, result       # finished — caller gets the result
    return job, None             # incomplete — carry over

# ─────────────────────────────────────────────
#  Extension after deconfliction
#  Pattern: concrete → (collision) → symbolic → (deconflict) → concrete → ...
#  Keeps looping as long as budget allows; returns a carry-over job if
#  symbolic verification couldn't finish within the budget.
# ─────────────────────────────────────────────

def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon, current_timestep, min_lookahead):
    """
    Greedily push validated_until toward max_time.

    Each iteration:
      1. Concrete scan from validated_until to max_time.
         - No conflict → done, validated_until = max_time.
         - Conflict found → try to symbolically verify it.
      2. Symbolic verify the conflict (one chunk, budget-limited).
         - Deconflicted → advance validated_until past the conflict, loop again.
         - Still conflicting → real danger, return immediately.
         - Incomplete (budget ran out mid-symbolic) → return carry-over job.

    Returns (new_validated_until, pending_job | None).
    """
    while validated_until < max_time:
        if not budget.can_afford('concrete'):
            print(f"[extend] Budget exhausted before concrete scan")
            break

        end_check_time = min(validated_until + budget.max_affordable_concrete(), max_time)
        print(f"[extend] Attempting concrete scan: t={validated_until} -> t={end_check_time}")
        collision, conflict_time = concrete_scan(tester, validated_until, end_check_time)

        if not collision:
            validated_until = end_check_time
            break

        print(f"[extend] Conflict at t={conflict_time}, attempting symbolic verification")

        if not budget.can_afford('symbolic', 1):
            print(f"[extend] No budget for symbolic — stopping before conflict")
            validated_until = conflict_time - 1
            return validated_until, VerificationTask(symbolic_start=conflict_time - 1, conflict_time=conflict_time)

        chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time) # can change this to symbolic_start=current_timestep
        job, result = symbolic_step(tester, job, chunk_size)

        if result is None:
            # Symbolic incomplete — carry over
            print(f"[extend] Symbolic incomplete at t={job.symbolic_start}, carrying over")
            validated_until = job.symbolic_start
            return validated_until, job

        # Symbolic finished
        validated_until = conflict_time - 1 if result["collision"] else conflict_time
        force_stop = result["collision"] and (conflict_time - current_timestep) < min_lookahead
        print(f"Conflict confirmed at t={conflict_time} using symbolic" + (" — FORCED STOP" if force_stop else " — deferring") if result["collision"] else f"Deconflicted — validated until t={conflict_time}")
        if force_stop or result["collision"]:
            return validated_until, None   # real danger — stop extending

        # Deconflicted — loop to scan further

    return validated_until, None


####  simulation loop  ####

def test1():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    obstacles = [
        np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),   # v = -1 wall
        np.array([[-np.inf, 0.0],    [-np.inf, np.inf]]), # x = 0 wall
    ]
    tester  = ReachabilityTester(analyzer, obstacles)
    tester_calibration = ReachabilityTester(analyzer)

    MIN_LOOKAHEAD        = 4
    MAX_LOOKAHEAD        = 10
    MAX_SYMBOLIC_HORIZON = 10
    MAX_TIME             = 40

    budget = TimeBudget(timestep_budget=0.40)
    budget.calibrate(tester_calibration, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON, max_backward_horizon=0)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    current_timestep = 0
    validated_until  = 0
    pending_job: Optional[VerificationTask] = None   # carry-over symbolic work

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        print(f"\nCURRENT TIMESTEP ==== {current_timestep}")

        # ── Phase 1: advance any carry-over symbolic work ──────────────────
        if pending_job is not None:
            print(f"[Carry-over] Resuming symbolic: t={pending_job.symbolic_start} -> t={pending_job.conflict_time}")
            chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
            pending_job, result = symbolic_step(tester, pending_job, chunk_size)

            if result is not None:
                # job finished this timestep
                conflict_time = pending_job.conflict_time
                validated_until = conflict_time - 1 if result["collision"] else conflict_time

                force_stop = result["collision"] and (conflict_time - current_timestep) < MIN_LOOKAHEAD
                print(f"Conflict confirmed at t={conflict_time}" + (" — FORCED STOP" if force_stop else " — deferring") if result["collision"] else f"Deconflicted — validated until t={conflict_time}")

                if force_stop:
                    break
                if not result["collision"] and budget.remaining > 0:
                    validated_until, pending_job = try_extend(
                        tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD
                    )
                else:
                    pending_job = None
            # else: used whole timestep but still incomplete, pending_job updated in-place, carry over again to next timestep.
            # Top of loop will handle it.

        # ── Phase 2: normal concrete scan (only if no carry-over work) ─────
        else:
            explore_from   = max(validated_until, current_timestep)
            end_check_time = min(explore_from + budget.max_affordable_concrete(), MAX_TIME)
            collision, conflict_time = concrete_scan(tester, explore_from, end_check_time)

            if not collision:
                validated_until = end_check_time
            else:
                print(f"Conflict detected at t={conflict_time}")
                chunk_size   = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
                pending_job  = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
                pending_job, result = symbolic_step(tester, pending_job, chunk_size)

                if result is not None:
                    # finished in one shot
                    validated_until = conflict_time - 1 if result["collision"] else conflict_time
                    force_stop = result["collision"] and (conflict_time - current_timestep) < MIN_LOOKAHEAD
                    print(f"Conflict confirmed at t={conflict_time}" + (" — FORCED STOP" if force_stop else " — deferring") if result["collision"] else f"Deconflicted — validated until t={conflict_time}")
                    pending_job = None
                    if force_stop:
                        break
                    if not result["collision"] and budget.remaining > 0:
                        validated_until, pending_job = try_extend(
                            tester, validated_until, MAX_TIME, budget,
                            MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD
                        )
                else:
                    # partial — carry over
                    validated_until = pending_job.symbolic_start
                    print(f"Budget exhausted at t={validated_until}, deferring to next timestep")

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
