"""
Double integrator model MPC using do-mpc
"""

from pyexpat import model
from casadi import *

import numpy as np
import do_mpc


def di_model(A:np.ndarray, B: np.ndarray, t_step: float = 0.1):
    """Create discrete time Double Integrator MPC model"""

    model_type = "discrete"
    model = do_mpc.model.Model(model_type)
    _x = model.set_variable('_x', 'p', shape=(1, 1))  # position
    _v = model.set_variable('_x', 'v', shape=(1, 1))  # velocity
    _u = model.set_variable('_u', 'a', shape=(1, 1))  # acceleration
    x = vertcat(_x, _v)
    x_next = A@x + B@_u
    model.sysA = A
    model.sysB = B

    model.set_rhs('p', x_next[0])
    model.set_rhs('v', x_next[1])
    model.setup()
    return model


def di_mpc(model: do_mpc.model.Model,
           obstacles: list = None,
           t_step: float = 0.1,
           n_horizon: int = 8,
           pos_min: float = 0.0,
           vel_min: float = -1.0,
           buffer: float = 0.0) -> do_mpc.controller.MPC:
    """buffer tightens the state constraints relative to the half-planes the
    verifier checks, so the MPC aims for a margin rather than the boundary."""

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = t_step
    mpc.settings.n_horizon = n_horizon
    mpc.settings.supress_ipopt_output()
    mpc.settings.store_full_solution = True

    p = model.x['p']
    v = model.x['v']
    a = model.u["a"]
    p_goal = DM([0.3])
    v_goal = DM([0])

    #lterm is per timstep
    lterm = (p-p_goal).T @ (p-p_goal) 

    #mterm is final objective
    mterm = 100 * (p-p_goal).T @ (p-p_goal) + 100* (v-v_goal).T @ (v-v_goal)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(a=0.1)

    # These were hardcoded to 0.0 / -1.0, which happened to equal the caller's
    # defaults, so the half-plane arguments alg13 passes were silently inert.
    mpc.bounds['lower', '_x', 'p'] = pos_min + buffer
    mpc.bounds['upper', '_x', 'p'] = 5.0
    mpc.bounds['lower', '_x', 'v'] = vel_min + buffer
    mpc.bounds['upper', '_x', 'v'] = 1.0
    mpc.bounds['lower', '_u', 'a'] = -1.0
    mpc.bounds['upper', '_u', 'a'] = 1.0

    if obstacles:
        p_var = model.x['p']
        v_var = model.x['v']
        state_vars  = [p_var, v_var]
        state_names = ['p',   'v']
        # Inflation margins per dim from Kalman bounds diagnostics:
        # max half-width of bound seen
        inflation = [0.3, 0.2]   # [position, velocity]

        for i_obs, obs in enumerate(obstacles):
            for i_dim in range(min(obs.shape[0], len(state_vars))):
                lo, hi = float(obs[i_dim, 0]), float(obs[i_dim, 1])
                x_dim  = state_vars[i_dim]
                name   = state_names[i_dim]
                m      = inflation[i_dim] if i_dim < len(inflation) else 0.0
                if np.isfinite(hi) and not np.isfinite(lo):
                    # obstacle region: x_dim <= hi  →  avoid: x_dim >= hi + m
                    mpc.set_nl_cons(f'obs{i_obs}_{name}_lo', -x_dim, ub=-(hi + m))
                elif np.isfinite(lo) and not np.isfinite(hi):
                    # obstacle region: x_dim >= lo  →  avoid: x_dim <= lo - m
                    mpc.set_nl_cons(f'obs{i_obs}_{name}_hi', x_dim, ub=lo - m)

    # print("MPC constraints:")
    # print(mpc.nl_cons_list)
    mpc.setup()
    return mpc


def di_simulator(model: do_mpc.model.Model,
                   t_step: float = 1.0) -> do_mpc.simulator.Simulator:
    """Configure a Simulator for the double integrator."""
    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = t_step
    simulator.setup()
    return simulator


if __name__ == '__main__':
    t_step   = 0.1
    n_steps  = 100
    A = np.array([[1,t_step],
                 [0,1]])
    B = np.array([[0.5 * t_step**2],[t_step]])

    x0 = np.array([[1.5],   # close to wall
               [-1.5]]) # moving toward wall (negative velocity)
    # Build discrete model, MPC, and simulator
    model     = di_model(A, B, t_step=t_step)
    mpc       = di_mpc(model, t_step=t_step, n_horizon=20)
    simulator = di_simulator(model, t_step=t_step)

    # Estimator: perfect state feedback
    estimator = do_mpc.estimator.StateFeedback(model)

    print("Discrete A:\n", model.sysA)
    print("Discrete B:\n", model.sysB)

    # Initialise
    mpc.x0       = x0
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    # Simulation loop
    print(f"\n{'Step':>4}  {'p':>8}  {'v':>8}  {'a':>8}")
    print("-" * 36)
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        x0 = estimator.make_step(x0)
        p_k = float(x0[0])
        v_k = float(x0[1])
        a_k = float(u0[0])
        print(f"{k:>4}  {p_k:>8.4f}  {v_k:>8.4f}  {a_k:>8.4f}")
        if abs(p_k) < 1e-3 and abs(v_k) < 1e-3:
            print(f"\nConverged at step {k}.")
            break
