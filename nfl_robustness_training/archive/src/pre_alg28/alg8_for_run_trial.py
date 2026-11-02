"""
Adaptive MPC algorithm — silent version for batch trials.
Identical logic to alg8_mpc_adaptive.py.
Obstacles are in the form: [center_x, center_y, radius]

"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
import time
from multi_obj_opt.strategies import ExtensionOptimizer
import numpy as np
from typing import Optional
from mpc_safety_filter import make_mpc_safety_filter


class VerificationTask:
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start
        self.conflict_time  = conflict_time
        self.time_invested  = 0

    def done(self):
        return self.symbolic_start >= self.conflict_time


def _get_nn_control(tester, timestep):
    """Query NN nominal control at the real state stored in horizons[timestep]."""
    h = tester.horizons.get(timestep)
    if h is None:
        return None
    for calc_data in h.calculations.values():
        if 'real_state' in calc_data:
            state = np.asarray(calc_data['real_state']).reshape(1, -1)
            cl_sys = tester.analyzer.cl_system
            u = cl_sys.dynamics.control_nn(state, cl_sys.controller.cpu())
            return np.asarray(u).flatten()
    return None


def concrete_scan(tester, from_t, to_t):
    result = tester.concrete(from_t, to_t)
    if result["collision"]:
        return True, result["collision_timestep"]
    return False, None


def symbolic_step(tester, job: VerificationTask, chunk_size: int, budget):
    steps = min(chunk_size, job.conflict_time - job.symbolic_start)
    cost  = budget.symbolic_costs.get(steps, float('inf'))
    if cost - job.time_invested <= budget.remaining:
        verify_end = min(job.conflict_time, job.symbolic_start + chunk_size)
        result = tester.symbolic(job.symbolic_start, verify_end)
        job.symbolic_start = verify_end
        job.time_invested  = 0
        if job.done():
            return job, result
        return job, None
    else:
        job.time_invested += budget.remaining
        return job, None


def apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state: dict, current_t: int = 0):
    stopping_t, controls, traj_bounds = mpc_sf.find_stopping_timestep(conflict_time, tester.horizons, current_t)
    if stopping_t is None:
        return None
    committed_at = mpc_state.get('committed_at')
    if committed_at is None or stopping_t >= committed_at:
        mpc_state['committed_at']  = stopping_t
        mpc_state['conflict_time'] = conflict_time
        mpc_state['controls']      = controls
        mpc_state['traj_bounds']   = traj_bounds
        mpc_state['needed']        = True
    return stopping_t


def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon,
               current_timestep, min_lookahead, mpc_sf, mpc_state, safe_horizon_ceiling=None):
    scan_ceiling = safe_horizon_ceiling if safe_horizon_ceiling is not None else max_time
    while validated_until < scan_ceiling:
        if not budget.can_afford('concrete'):
            break
        end_check_time = min(
            validated_until + budget.max_affordable_concrete(),
            scan_ceiling, max_time
        )
        collision, conflict_time = concrete_scan(tester, validated_until, end_check_time)
        if not collision:
            validated_until = end_check_time
            break
        if not budget.can_afford('symbolic', 1):
            validated_until = conflict_time - 1
            return validated_until, VerificationTask(conflict_time - 1, conflict_time), None
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)
        if result is None:
            validated_until = job.symbolic_start
            return validated_until, job, None
        if result["collision"]:
            validated_until = max(validated_until, conflict_time - 1)
            return validated_until, None, conflict_time
        else:
            validated_until = conflict_time
    return validated_until, None, None


def optimized_step(tester, validated_until, max_time, budget, max_symbolic_horizon,
                   current_timestep, min_lookahead, ext_optimizer, mpc_sf, mpc_state):
    target = validated_until + 1
    if target > max_time:
        return validated_until, None, None
    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)
    method = ext_optimizer.get_strategy(
        current_timestep=current_timestep,
        verified_until=validated_until,
        current_vol=current_vol,
        verified_vol=verified_vol,
        time_budget=budget.remaining,
        w_vol=50.0,
        pow_vol=2,
    )
    if method == "concrete":
        result        = tester.concrete(validated_until, target)
        conflict_time = result.get("collision_timestep")
    else:
        full_span     = target - current_timestep
        actual_k      = min(full_span, max_symbolic_horizon)
        target        = current_timestep + actual_k
        result        = tester.symbolic(current_timestep, target)
        conflict_time = result.get("collision_timestep")
    if not result["collision"]:
        return target, None, None
    if not budget.can_afford('symbolic', 1):
        return conflict_time - 1, VerificationTask(current_timestep, conflict_time), None
    job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
    job, result = symbolic_step(tester, job, max_symbolic_horizon, budget)
    if result is None:
        return conflict_time - 1, job, None
    if result["collision"]:
        return conflict_time - 1, None, conflict_time
    return conflict_time, None, None


def _get_volume(tester, timestep):
    horizon = tester.horizons.get(timestep)
    if horizon is None:
        return 1.0
    vol = horizon.get_tight_volume()
    return float(vol) if vol is not None else 1.0


def get_dynamic_symbolic_horizon(safety_margin, max_symbolic_horizon=10, symbolic_buffer=5):
    return max_symbolic_horizon


def test(seed=None, analyzer=None):
    if analyzer is None:
        analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    obstacles = [
        np.array([-6.5, 2.02,  0.5]),
        np.array([-3.2,  1.21,  0.5]),
        np.array([-1.5,  -0.85, 0.45]),
    ]

    tester  = ReachabilityTester(analyzer, obstacles, seed=seed)
    mpc_sf  = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=15)

    MIN_LOOKAHEAD        = 4
    MIN_SAFE_HORIZON     = 6
    MAX_SYMBOLIC_HORIZON = 10
    SYMBOLIC_BUFFER      = 5
    MAX_TIME             = 60

    budget = TimeBudget(timestep_budget=0.50)
    budget.symbolic_costs = {1: 0.05942702293395996, 2: 0.0532071590423584, 3: 0.12308859825134277, 4: 0.2227306365966797, 5: 0.3548123836517334, 6: 0.5160810947418213, 7: 0.7076215744018555, 8: 1.046485185623169, 9: 1.189185619354248, 10: 1.4745268821716309}
    budget.concrete_cost = 0.012865893046061198

    ext_optimizer = ExtensionOptimizer()
    ext_optimizer.set_timing_params({
        "concrete_slope":     0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope":        0.5341915550605524,
        "ratio_intercept":   -0.0578267576662421,
    })

    current_timestep         = 0
    validated_until          = 0
    u_diffs                  = []
    mpc_calls                = 0
    mpc_over_budget          = 0
    pending_job: Optional[VerificationTask] = None
    pending_mpc_conflict: Optional[int]    = None
    mpc_turn                 = False
    mpc_first_run            = True
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'traj_bounds':   [],
        'needed':        False,
    }
    mpc_started = False

    def trigger_mpc(conflict_time):
        nonlocal mpc_first_run, pending_mpc_conflict, mpc_calls, mpc_over_budget
        if mpc_first_run:
            _t0 = time.perf_counter()
            apply_mpc_filter(mpc_sf, conflict_time, tester, mpc_state, current_timestep)
            mpc_calls += 1
            if time.perf_counter() - _t0 + budget.elapsed > budget.timestep_budget:
                mpc_over_budget += 1
            mpc_first_run = False
        else:
            pending_mpc_conflict = conflict_time

    def run_opt_loop():
        nonlocal validated_until, pending_job, dynamic_symbolic_horizon
        while (validated_until - current_timestep >= MIN_SAFE_HORIZON
               and validated_until < MAX_TIME
               and pending_job is None
               and pending_mpc_conflict is None
               and budget.remaining > 0):
            validated_until, pending_job, mpc_c = optimized_step(
                tester, validated_until, MAX_TIME, budget,
                dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                ext_optimizer, mpc_sf, mpc_state
            )
            if mpc_c is not None:
                trigger_mpc(mpc_c)
            dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                validated_until - current_timestep,
                max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                symbolic_buffer=SYMBOLIC_BUFFER
            )

    while current_timestep < MAX_TIME:
        budget.start_timestep()

        safety_margin = validated_until - current_timestep
        dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
            safety_margin,
            max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
            symbolic_buffer=SYMBOLIC_BUFFER
        )

        if mpc_started:
            ctrl_idx  = current_timestep - mpc_state['committed_at']
            queue     = mpc_state['controls']
            check_end = min(current_timestep + MIN_SAFE_HORIZON, MAX_TIME)
            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
            conflict_still_present, _ = concrete_scan(tester, current_timestep, check_end)
            if not conflict_still_present:
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None
                mpc_first_run              = True
                validated_until            = check_end
                pending_job                = None
                pending_mpc_conflict       = None
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, current_timestep + 1)
                current_timestep += 1
            elif ctrl_idx < len(queue):
                ctrl = queue[ctrl_idx]
                u_nn = _get_nn_control(tester, current_timestep)
                u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                break
            continue

        has_commit = mpc_state['committed_at'] is not None and mpc_state['needed']
        both_pending = pending_job is not None and pending_mpc_conflict is not None
        do_mpc_this_step = (
            pending_mpc_conflict is not None and
            (pending_job is None or not has_commit or mpc_turn)
        )

        if do_mpc_this_step:
            _t0 = time.perf_counter()
            apply_mpc_filter(mpc_sf, pending_mpc_conflict, tester, mpc_state, current_timestep)
            mpc_calls += 1
            if time.perf_counter() - _t0 + budget.elapsed > budget.timestep_budget:
                mpc_over_budget += 1
            pending_mpc_conflict = None
            mpc_turn = False
        else:
            if both_pending:
                mpc_turn = True

            if pending_job is not None:
                pending_job, result = symbolic_step(tester, pending_job,
                                                    dynamic_symbolic_horizon, budget)
                if result is not None:
                    conflict_time = pending_job.conflict_time
                    pending_job   = None
                    if result["collision"]:
                        validated_until = max(validated_until, conflict_time - 1)
                        trigger_mpc(conflict_time)
                    else:
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
                            run_opt_loop()
            else:
                safety_margin = validated_until - current_timestep
                dynamic_symbolic_horizon = get_dynamic_symbolic_horizon(
                    safety_margin,
                    max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                    symbolic_buffer=SYMBOLIC_BUFFER
                )
                if safety_margin >= MIN_SAFE_HORIZON:
                    run_opt_loop()
                    new_margin = validated_until - current_timestep
                    if new_margin < MIN_SAFE_HORIZON and pending_job is None and pending_mpc_conflict is None and budget.remaining > 0:
                        validated_until, pending_job, mpc_c = try_extend(
                            tester, validated_until, MAX_TIME, budget,
                            dynamic_symbolic_horizon, current_timestep, MIN_LOOKAHEAD,
                            mpc_sf, mpc_state,
                            safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON,
                        )
                        if mpc_c is not None:
                            trigger_mpc(mpc_c)
                        run_opt_loop()
                elif validated_until < MAX_TIME:
                    explore_from   = max(validated_until, current_timestep)
                    end_check_time = min(
                        explore_from + budget.max_affordable_concrete(),
                        current_timestep + MIN_SAFE_HORIZON, MAX_TIME
                    )
                    collision, conflict_time = concrete_scan(tester, explore_from, end_check_time)
                    if not collision:
                        validated_until = end_check_time
                        if budget.remaining > 0:
                            run_opt_loop()
                    else:
                        pending_job = VerificationTask(current_timestep, conflict_time)
                        pending_job, result = symbolic_step(tester, pending_job,
                                                            dynamic_symbolic_horizon, budget)
                        if result is not None:
                            pending_job = None
                            if result["collision"]:
                                validated_until = max(validated_until, conflict_time - 1)
                                trigger_mpc(conflict_time)
                            else:
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
                                    run_opt_loop()
                        else:
                            validated_until = pending_job.symbolic_start

        if (mpc_state['conflict_time'] is not None
                and validated_until >= mpc_state['conflict_time'] + MIN_SAFE_HORIZON):
            if mpc_state['needed']:
                if current_timestep >= mpc_state['committed_at'] and validated_until < MAX_TIME:
                    trigger_mpc(validated_until)
            mpc_state['needed'] = False
            mpc_first_run = True

        if (mpc_state['committed_at'] is not None
                and current_timestep >= mpc_state['committed_at']
                and mpc_state['needed']):
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']
            if ctrl_idx < len(queue):
                mpc_started = True
                ctrl = queue[ctrl_idx]
                u_nn = _get_nn_control(tester, current_timestep)
                u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                tester.real_state_mpc(current_timestep, ctrl)
                current_timestep += 1
            else:
                break
        else:
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, current_timestep + 1)
            current_timestep += 1

    state_history = []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(calc['real_state'].copy())
                break

    collision_timesteps = []
    for i, s in enumerate(state_history):
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx)**2 + (s_flat[1] - cy)**2 <= r**2:
                collision_timesteps.append(i)
                break

    had_collision = len(collision_timesteps) > 0
    return state_history, had_collision, u_diffs, mpc_calls, mpc_over_budget


if __name__ == "__main__":
    test()
