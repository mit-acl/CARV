"""
MPC-based Safety Filter with Kalman Filtering.

Algorithm:
  1. Start at t_back = t_collision - 2
  2. Get center of reachable set at t_back
  3. Run MPC from that center for n horizon steps
  4. Get next position using Kalman filter and controls from MPC
  5. If no bounds collide with obstacles: stopping_timestep = t_back
  6. Else: t_back -= 1, repeat from step 2
"""

import numpy as np

from utils.di_mpc import di_model, di_mpc, di_simulator
from utils.unicycle_mpc import unicycle_model, unicycle_mpc
from state_estimator import LinearKalmanEstimator, ExtendedKalmanEstimator

class MPCSafetyFilter:
    """
    base class for MPC safety filters
    Subclasses implement specific linear/nonlinear dynamics
    """

    def __init__(self, obstacles, tester, max_lookback: int = 10):
        self.obstacles    = obstacles if obstacles is not None else []
        self.max_lookback = max_lookback
        self.tester = tester

    def _fill_nom_ctrl_buf(self, x0: np.ndarray):
        """
        Bootstrap: roll out the NN from x0 for n_horizon steps to get
        per-step nominals. Used only on the first MPC solve when no
        predicted trajectory is available yet.
        """
        cl_sys = self.tester.analyzer.cl_system
        xt = x0.flatten().reshape(1, -1)  # (1, state_dim) numpy
        for k in range(self.n_horizon + 1):
            u = cl_sys.dynamics.control_nn(xt, cl_sys.controller.cpu())  # numpy (1, ctrl_dim)
            self._mpc._nom_ctrl_buf[k] = u.flatten()[:1]
            xt = cl_sys.dynamics.dynamics_step(xt, u)

    def _fill_nom_ctrl_buf_from_states(self, states: list):
        """
        Query the NN at each MPC-predicted state to get per-step nominal control.
        states: MPC predicted trajectory from previous solve, shifted by 1.
        Remaining buffer entries are padded with the last computed value.
        """
        cl_sys = self.tester.analyzer.cl_system
        buf_len = len(self._mpc._nom_ctrl_buf)
        filled = min(len(states), buf_len)

        batch = np.array([np.array(states[k]).flatten() for k in range(filled)])  # (filled, state_dim)
        us = cl_sys.dynamics.control_nn(batch, cl_sys.controller.cpu())            # (filled, ctrl_dim)
        self._mpc._nom_ctrl_buf[:filled] = us[:, :1]
        if filled < buf_len:
            self._mpc._nom_ctrl_buf[filled:] = us[-1, :1]

    def _collides(self, bounds: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.all(bounds[:, 1] >= obs[:, 0]) and np.all(bounds[:, 0] <= obs[:, 1]):
                return True
        return False

    def find_stopping_timestep(self, collision_timestep: int, horizons: dict,
                               current_t: int = 0):
        """
        Find the safe stopping timestep closest to collision.

        Args:
            collision_timestep
            horizons:           dict mapping int timestep -> ReachableSetHorizon
            current_t:          current real timestep — t_back will not go before this

        Returns:
            safe stopping timestep or None
        """
        for lookback in range(2, self.max_lookback + 1):
            t_back = collision_timestep - lookback

            if t_back < max(0, current_t):
                break

            bounds_at_back = horizons[t_back].get_tight_bound()
            # half_widths = (bounds_at_back[:, 1] - bounds_at_back[:, 0]) / 2.0
            # print(f"  [bounds diag] t_back={t_back}: half_widths={np.round(half_widths, 4)}")
            center = (bounds_at_back[:, 0] + bounds_at_back[:, 1]) / 2.0

            try:
                print(f"====== t_back: {t_back} =======")
                print(f"  [MPC parent bounds] p={np.round(bounds_at_back[0], 4)}  "
                      f"v={np.round(bounds_at_back[1], 4)}  center={np.round(center, 4)}")
                traj_bounds, _, controls = self._run_mpc_from_bounds(bounds_at_back, center)
            except Exception as e:
                print(f"  [MPC filter] failed at t_back={t_back}: {e}")
                continue

            collision_found = False
            mpc_collision_step  = None
            for step, b in enumerate(traj_bounds[1:], start=1):
                if self._collides(b):
                    collision_found = True
                    mpc_collision_step  = step
                    break

            if not collision_found:
                print(f"  [MPC filter] Safe. Stopping timestep = {t_back}")
                return t_back, controls, traj_bounds
            else:
                print(f"  [MPC filter] Collision at step {mpc_collision_step} "
                      f"from t_back={t_back}, going further back.")

        print(f"  [MPC filter] No safe stopping timestep found at or after t={current_t}.")
        return None, [], []


class LinearMPCSafetyFilter(MPCSafetyFilter):
    """
    Safety filter for linear DI dynamics.
    """

    def __init__(self, estimator: LinearKalmanEstimator, obstacles, tester,
                 t_step: float = 0.1, n_horizon: int = 10, max_lookback: int = 10):
        super().__init__(obstacles, tester, max_lookback)
        self._estimator = estimator
        self.t_step     = t_step
        self.n_horizon  = n_horizon
        self.A          = estimator.A
        self.B          = estimator.B

        self._model     = di_model(self.A, self.B, t_step=t_step)
        self._mpc       = di_mpc(self._model, obstacles=obstacles, t_step=t_step,
                                 n_horizon=n_horizon, nominal_tracking=True)
        self._simulator = di_simulator(self._model, t_step=t_step)

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray) -> list:
        """
        Receding-horizon MPC + Kalman estimation for n_horizon steps
        MPC gets controls and KF gets bounds -> repeat
        """
        estimator = self._estimator

        saved_x      = estimator.x.copy()
        saved_P      = estimator.P.copy()
        saved_bounds = estimator.bounds.copy()
        try:
            estimator.reset(center.flatten(), initial_bounds)

            x = center.reshape(2, 1)
            self._mpc.x0 = x
            self._mpc.set_initial_guess()

            trajectory_bounds = [initial_bounds.copy()]
            points   = []
            controls = []
            prev_predicted_states = None

            for _ in range(self.n_horizon):
                x = estimator.x.reshape(2, 1)
                if prev_predicted_states is None:
                    self._fill_nom_ctrl_buf(x.flatten())
                else:
                    self._fill_nom_ctrl_buf_from_states(prev_predicted_states[1:])
                u = self._mpc.make_step(x)
                prev_predicted_states = [
                    np.array(self._mpc.opt_x_num['_x', k, 0]).flatten()
                    for k in range(self.n_horizon + 1)
                ]
                _, bounds = estimator.predict(np.array(u).flatten())
                estimator.update(estimator.x)  # synthetic measurement update to keep bounds realistic
                bounds = estimator.bounds.copy()
                trajectory_bounds.append(bounds.copy())
                points.append(x)
                controls.append(np.array(u).flatten().copy())

            print(f"controls: {controls}")
            return trajectory_bounds, points, controls
        finally:
            estimator.x      = saved_x
            estimator.P      = saved_P
            estimator.bounds = saved_bounds


class UniycleMPCSafetyFilter(MPCSafetyFilter):
    """
    Safety filter for unicycle dynamics.
    Uses Unicycle MPC + EKF
    """

    def __init__(self, estimator: ExtendedKalmanEstimator, obstacles, tester,
                 dt: float = 0.1, v: float = 1.0,
                 n_horizon: int = 10, max_lookback: int = 10, nominal_tracking=True):
        super().__init__(obstacles, tester, max_lookback)
        self._estimator = estimator
        self.dt         = dt
        self.n_horizon  = n_horizon
        self.v          = v

        self._nominal_tracking = nominal_tracking
        self._model = unicycle_model(dt=dt, v=v)

        # Relaxed IPOPT tolerances — faster convergence
        _fast_opts = {
            'ipopt.tol':        1e-4,
            'ipopt.max_iter':   50,
            'ipopt.warm_start_init_point': 'yes',
        }

        self._mpc   = unicycle_mpc(self._model, obstacles=obstacles, dt=dt,
                                   n_horizon=n_horizon, nominal_tracking=nominal_tracking,
                                   solver_opts=_fast_opts)

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray) -> list:
        """
        Receding-horizon unicycle MPC + EKF propagation
        """
        estimator = self._estimator

        saved_x      = estimator.x.copy()
        saved_P      = estimator.P.copy()
        saved_bounds = estimator.bounds.copy()
        try:
            estimator.reset(center.flatten(), initial_bounds)

            x = center.reshape(-1, 1)
            self._mpc.x0 = x
            self._mpc.set_initial_guess()

            trajectory_bounds = [initial_bounds.copy()]
            points   = []
            controls = []
            prev_predicted_states = None

            for _ in range(self.n_horizon):
                x = estimator.x.reshape(-1, 1)
                if self._nominal_tracking:
                    if prev_predicted_states is None:
                        self._fill_nom_ctrl_buf(x.flatten())
                    else:
                        self._fill_nom_ctrl_buf_from_states(prev_predicted_states[1:])
                u = self._mpc.make_step(x)
                prev_predicted_states = [
                    np.array(self._mpc.opt_x_num['_x', k, 0]).flatten()
                    for k in range(self.n_horizon + 1)
                ]
                _, bounds = estimator.predict(np.array(u).flatten())
                estimator.update(estimator.x)  # measurement update to keep mpc bounds realistic
                bounds = estimator.bounds.copy()
                trajectory_bounds.append(bounds.copy())
                points.append(x)
                controls.append(np.array(u).flatten().copy())

            return trajectory_bounds, points, controls
        finally:
            estimator.x      = saved_x
            estimator.P      = saved_P
            estimator.bounds = saved_bounds


def make_mpc_safety_filter(tester, obstacles_list=None, t_step: float = 0.1,
                            n_horizon: int = 10, max_lookback: int = 10, nominal_tracking=True):
    """
    Build the MPCSafetyFilter subclass from a tester object.
    """
    if obstacles_list is None:
        obstacles_list = tester.obstacles.obstacle_list

    dynamics_name = type(tester.analyzer.cl_system.dynamics).__name__

    if dynamics_name == 'DoubleIntegrator':
        return LinearMPCSafetyFilter(
            estimator=tester.estimator,
            obstacles=obstacles_list,
            tester=tester,
            t_step=t_step,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
        )

    if dynamics_name == 'Unicycle_NL':
        return UniycleMPCSafetyFilter(
            estimator=tester.estimator,
            obstacles=obstacles_list,
            tester=tester,
            dt=t_step,
            v=tester.analyzer.cl_system.dynamics.vt,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
            nominal_tracking=nominal_tracking
        )

    return None
