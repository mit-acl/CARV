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

    def __init__(self, obstacles, max_lookback: int = 10):
        self.obstacles    = obstacles if obstacles is not None else []
        self.max_lookback = max_lookback

    def collides(self, bounds: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.all(bounds[:, 1] >= obs[:, 0]) and np.all(bounds[:, 0] <= obs[:, 1]):
                return True
        return False

    def find_stopping_timestep(self, collision_timestep: int, horizons: dict):
        """
        Find the safe stopping timestep closest to collision.

        Args:
            collision_timestep
            horizons:           dict mapping int timestep -> ReachableSetHorizon

        Returns:
            safe stopping timestep or None
        """
        for lookback in range(2, self.max_lookback + 1):
            t_back = collision_timestep - lookback

            bounds_at_back = horizons[t_back].get_tight_bound()
            center = (bounds_at_back[:, 0] + bounds_at_back[:, 1]) / 2.0

            try:
                traj_bounds = self.run_mpc_from_bounds(bounds_at_back, center)
            except Exception as e:
                print(f"  [MPC filter] failed at t_back={t_back}: {e}")
                continue

            collision_found = False
            collision_step  = None
            for step, b in enumerate(traj_bounds[1:], start=1):
                if self.collides(b):
                    collision_found = True
                    collision_step  = step
                    break

            if not collision_found:
                print(f"  [MPC filter] Safe! Stopping timestep = {t_back}")
                return t_back
            else:
                print(f"  [MPC filter] Collision at step {collision_step} "
                      f"from t_back={t_back}, going further back.")

        print(f"  [MPC filter] No safe stopping timestep found within "
              f"{self.max_lookback} lookback steps.")
        return None


class LinearMPCSafetyFilter(MPCSafetyFilter):
    """
    Safety filter for linear DI dynamics.

    """

    def __init__(self, estimator: LinearKalmanEstimator, obstacles,
                 t_step: float = 0.1, n_horizon: int = 8, max_lookback: int = 10):
        super().__init__(obstacles, max_lookback)
        self._estimator = estimator
        self.t_step     = t_step
        self.n_horizon  = n_horizon
        self.A          = estimator.A
        self.B          = estimator.B

        self._model     = di_model(self.A, self.B, t_step=t_step)
        self._mpc       = di_mpc(self._model, t_step=t_step, n_horizon=n_horizon)
        self._simulator = di_simulator(self._model, t_step=t_step)

    def run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray) -> list:
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

            for _ in range(self.n_horizon):
                x = estimator.x.reshape(2, 1)
                u = self._mpc.make_step(x)
                _, bounds = estimator.predict(np.array(u).flatten())
                trajectory_bounds.append(bounds.copy())

            return trajectory_bounds
        finally:
            estimator.x      = saved_x
            estimator.P      = saved_P
            estimator.bounds = saved_bounds


class UniycleMPCSafetyFilter(MPCSafetyFilter):
    """
    Safety filter for unicycle dynamics.
    Uses Unicycle MPC + EKF
    """

    def __init__(self, estimator: ExtendedKalmanEstimator, obstacles,
                 dt: float = 0.1, v: float = 1.0,
                 n_horizon: int = 8, max_lookback: int = 10):
        super().__init__(obstacles, max_lookback)
        self._estimator = estimator
        self.dt         = dt
        self.n_horizon  = n_horizon

        self._model = unicycle_model(dt=dt, v=v)
        self._mpc   = unicycle_mpc(self._model, dt=dt, n_horizon=n_horizon)

    def run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray) -> list:
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

            for _ in range(self.n_horizon):
                x = estimator.x.reshape(-1, 1)
                u = self._mpc.make_step(x)
                _, bounds = estimator.predict(np.array(u).flatten())
                trajectory_bounds.append(bounds.copy())

            return trajectory_bounds
        finally:
            estimator.x      = saved_x
            estimator.P      = saved_P
            estimator.bounds = saved_bounds


def make_mpc_safety_filter(tester, obstacles_list, t_step: float = 1.0,
                            n_horizon: int = 8, max_lookback: int = 10):
    """
    Build the MPCSafetyFilter subclass from a tester object.
    """
    dynamics_name = type(tester.analyzer.cl_system.dynamics).__name__

    if dynamics_name == 'DoubleIntegrator':
        return LinearMPCSafetyFilter(
            estimator=tester.estimator,
            obstacles=obstacles_list,
            t_step=t_step,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
        )

    if dynamics_name == 'Unicycle_NL':
        return UniycleMPCSafetyFilter(
            estimator=tester.estimator,
            obstacles=obstacles_list,
            dt=t_step,
            v=tester.analyzer.cl_system.dynamics.vt,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
        )

    return None
