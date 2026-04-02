from pyexpat import model
from casadi import *

import numpy as np
import do_mpc

def unicycle_model(dt: float = 0.1, v: float = 1.0):
    """Create non linear mpc model"""
    model_type = "discrete"
    model = do_mpc.model.Model(model_type)
    # State: [x, y, theta]
    p     = model.set_variable('_x', 'p',     shape=(2, 1))
    theta = model.set_variable('_x', 'theta', shape=(1, 1))

    # Control: omega only (v is fixed)
    omega = model.set_variable('_u', 'omega', shape=(1, 1))

    x_next = p[0] + v * cos(theta) * dt
    y_next = p[1] + v * sin(theta) * dt
    th_next = theta + omega * dt

    p_next = vertcat(x_next,y_next)

    model.set_rhs('p', p_next)
    model.set_rhs('theta', th_next)

    model.setup()
    return model

def unicycle_mpc(model: do_mpc.model.Model,
                 obstacles: list = None,
                 dt: float = 0.1,
                 n_horizon: int = 8) -> do_mpc.controller.MPC:

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.t_step = dt
    mpc.settings.n_horizon = n_horizon
    mpc.settings.supress_ipopt_output()
    mpc.settings.store_full_solution = True

    p = model.x['p']
    omega = model.u["omega"]
    p_goal = DM([2,0])

    lterm = (p-p_goal).T @ (p-p_goal) + 0.1 * omega**2
    mterm = (p-p_goal).T @ (p-p_goal)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.bounds['lower', '_u', 'omega'] = -1.0
    mpc.bounds['upper', '_u', 'omega'] = 1.0

    if obstacles is None:
        # Default hardcoded circular obstacles for standalone testing
        cx1, cy1, r1 = -6.0, -0.5, 2.3
        dist_sq1 = (p[0] - cx1)**2 + (p[1] - cy1)**2
        mpc.set_nl_cons('obs1', -dist_sq1, ub=-r1**2)

        cx2, cy2, r2 = -2.0, 1.5, 1.5
        dist_sq2 = (p[0] - cx2)**2 + (p[1] - cy2)**2
        mpc.set_nl_cons('obs2', -dist_sq2, ub=-r2**2)
    else:
        # Takes in obstacle list from Reachability Tester and adds
        # to MPC obstacle constraints
        pos_names = ['x', 'y']
        for i_obs, obs in enumerate(obstacles):
            x_lo, x_hi = float(obs[0, 0]), float(obs[0, 1])
            y_lo, y_hi = float(obs[1, 0]), float(obs[1, 1])
            x_bounded  = np.isfinite(x_lo) and np.isfinite(x_hi)
            y_bounded  = np.isfinite(y_lo) and np.isfinite(y_hi)

            if x_bounded and y_bounded:
                # Bounded box: ellipse fitted to box shape + conservative buffer
                cx = (x_lo + x_hi) / 2.0
                cy = (y_lo + y_hi) / 2.0
                a  = (x_hi - x_lo) / 2.0 + 0.2
                b  = (y_hi - y_lo) / 2.0 + 0.2

                a = 0.45
                b =0.3

                print(f"center {cx},{cy}  a={a}  b={b}")
                ellipse = (p[0] - cx)**2 / a**2 + (p[1] - cy)**2 / b**2
                mpc.set_nl_cons(f'obs{i_obs}_ellipse', -ellipse, ub=-1.0)
            else:
                # Half-space obstacle: handle each finite bound independently
                for i_dim in range(min(obs.shape[0], 2)):
                    lo, hi = float(obs[i_dim, 0]), float(obs[i_dim, 1])
                    x_dim  = p[i_dim]
                    name   = pos_names[i_dim]
                    if np.isfinite(hi) and not np.isfinite(lo):
                        mpc.set_nl_cons(f'obs{i_obs}_{name}_lo', -x_dim, ub=-hi)
                    elif np.isfinite(lo) and not np.isfinite(hi):
                        mpc.set_nl_cons(f'obs{i_obs}_{name}_hi', x_dim, ub=lo)

    mpc.setup()
    return mpc

def unicycle_simulator(model, dt):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = dt
    simulator.setup()
    return simulator

if __name__ == "__main__":
    dt = 0.1
    nstep = 150

    model = unicycle_model(dt, v= 1.0)
    mpc = unicycle_mpc(model, dt=dt, n_horizon=10)
    simulator = unicycle_simulator(model, dt)

    x0 = np.array([-8.0, 2, -pi/4]).reshape(-1, 1)  # from the plot's start position

    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    for k in range(nstep):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        x_k   = float(x0[0])
        y_k   = float(x0[1])
        th_k  = float(x0[2])
        om_k  = float(u0[0])
        print(f"k={k}  x={x_k:.4f}  y={y_k:.4f}  theta={th_k:.4f}  omega={om_k:.4f}")
