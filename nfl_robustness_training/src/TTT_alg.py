"""Implementation of TTT algorithm"""

import numpy as np
from real_reachset_sim import ReachabilityTester
from real_reachset_sim import setup_analyzer

def simulation():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating simulation")
    simulator = ReachabilityTester(analyzer)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(simulator.horizons[0])

    refined_search(budget=2.5, n=50, simulator = simulator)


def refined_search(budget, n, simulator):
        total_time = 0
        phase = "search"
        t_start = 0
        b_steps = 1
        t_curr = 0
        t_est = 0
        while t_curr<n:
            if phase == "search":
                t_elapsed = simulator.symbolic(t_start, t_start + b_steps)
            elif phase == "jump":
                # jump phase ==> fill in timesteps in between jumps with concrete queries
                t_elapsed = simulator.symbolic(t_start, t_start + b_steps)
                t_elapsed+= simulator.concrete(t_start, t_start + b_steps)

            t_curr = t_start + b_steps

            #subtract from time budget
            budget -= t_elapsed

            #update stepsize,
            b_steps_n, phase = calc_steps(t_start, b_steps, budget, t_est, t_elapsed, n, phase)

            if phase == "jump":
                t_start = t_curr

            b_steps = min(b_steps_n, n - t_start)

            total_time +=t_elapsed

        print(f"total time taken: {total_time}")

def calc_steps(t_start, b_steps, rem_budget, t_est, new_calc_time, total_horizon, phase):
    """
    Returns optimal symbolic step size based on the provided time budget

    Returns:
        b_steps: the step size for subsequent iterations
        phase: search or jump
    """

    t_curr = t_start + b_steps #arrived timestep after the symbolic query
    t_est = max(t_est, new_calc_time/b_steps)
    if phase == "search":
        if total_horizon * t_est < rem_budget:
            b_steps +=1 #increase step size
        else:
            print("="*10 + " STARTING JUMP PHASE " + "="*10)
            print(f"step size: {b_steps}\n")
            rem_timesteps = total_horizon-b_steps
            num_jumps = np.ceil(rem_timesteps/b_steps)

            #redistribute step size to avoid short last jump
            b_steps = int(np.ceil(rem_timesteps/num_jumps))

            phase = "jump"
    return b_steps, phase


if __name__ == "__main__":
    simulation()
