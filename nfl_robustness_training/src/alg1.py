from REAL_integrated_sim import setup_analyzer, ReachabilityTester, CalculationType
from time_budget import TimeBudget
import numpy as np
import math

def test1():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    obstacles = [
        np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),  # v = -1 wall
        np.array([[-np.inf, 0.0],   [-np.inf, np.inf]]), # x = 0 wall
    ]
    tester = ReachabilityTester(analyzer, obstacles)
    tester1 = ReachabilityTester(analyzer)

    MIN_LOOKAHEAD = 4
    MAX_LOOKAHEAD = 10
    MAX_SYMBOLIC_HORIZON = 10
    MAX_TIME = 40

    budget = TimeBudget(timestep_budget=0.40)
    budget.calibrate(tester1, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON, max_backward_horizon=0)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    current_timestep = 0
    validated_until = 0

    # Carry-over state: incomplete symbolic verification from a previous timestep
    pending_symbolic_start = None  # where to resume symbolic next timestep
    pending_conflict_time  = None  # conflict we're working toward verifying

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        print(f"CURRENT TIMESTEP ==== {current_timestep}\n")

        # === Resume pending symbolic verification from previous timestep ===
        if pending_symbolic_start is not None:
            print(f"[Carry-over] Resuming symbolic: t={pending_symbolic_start} -> t={pending_conflict_time}")
            chunk_size = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
            verify_end = min(pending_conflict_time, pending_symbolic_start + chunk_size)

            symbolic_reach_set = tester.symbolic(pending_symbolic_start, verify_end)

            if verify_end == pending_conflict_time:
                # Finished — resolve the pending conflict
                original_conflict_time = pending_conflict_time
                pending_symbolic_start = None
                pending_conflict_time  = None

                if symbolic_reach_set["collision"]:
                    confirmed_t = symbolic_reach_set['collision_timestep']
                    print(f"Still conflicting at t={confirmed_t}")
                    validated_until = confirmed_t - 1
                    if confirmed_t - current_timestep < MIN_LOOKAHEAD:

                        break
                    print(f"deferring until later")

                else:
                    print("Carry-over symbolic complete — deconflicted")
                    validated_until = original_conflict_time
                    if budget.remaining > 0:
                        validated_until, pending_symbolic_start, pending_conflict_time = extend_validation(tester, validated_until, MAX_TIME, budget)
            else:
                # Still incomplete — fall through to bottom of loop for empirical step
                pending_symbolic_start = verify_end
                print(f"[Carry-over] Still incomplete at t={verify_end}, continuing next timestep")

        else:
            # === Normal operation: concrete lookahead ===
            explore_from = max(validated_until, current_timestep)
            end_check_time = min(explore_from + budget.max_affordable_concrete(), MAX_TIME)
            concrete_reach_set = tester.concrete(explore_from, end_check_time)

            if concrete_reach_set["collision"]:
                conflict_time = concrete_reach_set["collision_timestep"]
                print(f"Conflict detected at timestep {conflict_time}\n")

                max_symbolic_horizon_size = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
                verify_until = min(conflict_time, current_timestep + max_symbolic_horizon_size)
                print(f"Budget: horizon_length={max_symbolic_horizon_size}, verify_until={verify_until}")

                symbolic_reach_set = tester.symbolic(current_timestep, verify_until)

                if verify_until == conflict_time:
                    # Full verification done this timestep
                    if symbolic_reach_set["collision"]:
                        print(f"Conflict confirmed at t={conflict_time}\n")
                        validated_until = confirmed_t - 1

                        if conflict_time - current_timestep < MIN_LOOKAHEAD:
                            print(f"Within MIN_LOOKAHEAD — FORCED STOP")
                            break
                    else:
                        print("Deconflicted with symbolic\n")
                        validated_until = conflict_time
                        if budget.remaining > 0:
                            validated_until, pending_symbolic_start, pending_conflict_time = extend_validation(tester, validated_until, MAX_TIME, budget)
                        else:
                            print(f"Skipping extension — budget exhausted ({budget.elapsed:.3f}s used)")
                else:
                    # Partial verification — carry remainder to next timestep
                    print(f"Budget exhausted at t={verify_until}, deferring t={verify_until}..{conflict_time} to next timestep")
                    validated_until = verify_until
                    pending_symbolic_start = verify_until
                    pending_conflict_time  = conflict_time

            else:
                validated_until = end_check_time

        if validated_until >= MAX_TIME:
            break

        tester.real_state_empirical(current_timestep, current_timestep + 1)
        print(f"Safety margin: {validated_until - current_timestep} steps ahead validated")
        print(f"[Budget] {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s used this timestep")
        current_timestep += 1

    print(f"\n{'='*60}")
    print(f"Simulation complete at timestep {current_timestep}")


def recalculate_with_symbolic(tester, conflict_time, current_timestep, chunk_size):
    """
    Run symbolic analysis in chunks of chunk_size to verify a detected conflict.
    """
    total_horizon = conflict_time - current_timestep
    num_intervals = math.ceil(total_horizon / chunk_size)

    print(f"{conflict_time=}, {current_timestep=}")
    print(f"{num_intervals=}, {chunk_size=}")

    curr_chunk_start = current_timestep
    while curr_chunk_start < conflict_time:
        chunk_end = min(curr_chunk_start + chunk_size, conflict_time)
        symbolic_reach_set = tester.symbolic(curr_chunk_start, chunk_end)
        curr_chunk_start = chunk_end

    return symbolic_reach_set

def extend_validation(tester, validated_until, max_time, budget):
    """
    Extend with concrete propagations
    Returns: New validated_until timestep, pending_symbolic_start, pending_conflict_time
    """
    print("extension after symbolic deconflict\n")
    calc_horizon = budget.max_affordable_concrete()
    end = min(validated_until + calc_horizon, max_time)
    concrete_reach_set = tester.concrete(validated_until, end)
    if concrete_reach_set["collision"]:
        collision_time = concrete_reach_set["collision_timestep"]
        validated_until = collision_time -1
        return validated_until, validated_until, collision_time
    else:
        return end, None, None


def extend_validation2(tester, validated_until, max_time, budget):
    """
    Extend the validated horizon as far as the remaining time budget allows.
    Concrete alone is enough to extend if no conflict is found.
    Symbolic is only needed if concrete finds a conflict.

    Returns: New validated_until timestep
    """

    if validated_until >= max_time:
        return validated_until

    if not budget.can_afford('concrete'):
        print(f"Skipping extension — can't afford concrete ({budget.elapsed:.3f}s used)")
        return validated_until

    print(f"Extending validation: {validated_until} -> {max_time}")
    extension_result = tester.concrete(validated_until, max_time)

    if extension_result["collision"]:
        extension_conflict_time = extension_result['collision_timestep']
        print(f"Conflict detected during extension at timestep {extension_conflict_time}\n")

        symbolic_start = find_nearest_symbolic(tester, extension_conflict_time, validated_until)

        if not budget.can_afford('symbolic', extension_conflict_time - symbolic_start):
            print(f"Skipping symbolic verification — insufficient budget")
            return extension_conflict_time - 1

        print(f"Using timestep {symbolic_start} as starting point for symbolic")
        symbolic_result = tester.symbolic(symbolic_start, extension_conflict_time)

        if symbolic_result["collision"]:
            print(f"Conflict confirmed during extension at t={symbolic_result['collision_timestep']}\n")
            return validated_until
        else:
            print(f"Extension deconflicted, validated until t={extension_conflict_time}\n")
            return extension_conflict_time
    else:
        return max_time


def find_nearest_symbolic(tester, search_from, fallback_time):
    print("searching for nearest previous symbolic\n")
    for t in range(search_from - 1, fallback_time, -1):
        if t in tester.horizons and search_from - t >= 5:
            for calc_id, calc in tester.horizons[t].calculations.items():
                if calc['calc_type'] == CalculationType.SYMBOLIC:
                    return t

    return fallback_time


if __name__ == "__main__":
    test1()
