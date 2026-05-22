"""
Acados-based MPC Safety Filter.

Mirrors mpc_safety_filter.py but replaces the do_mpc unicycle MPC with
AcadosUnicycleMPC from utils/unicycle_mpc_acados.py.

The planning logic, collision checking, and overall interface are identical
to the original filter — only the inner MPC solver is replaced.
"""

import os
import numpy as np

from mpc_safety_filter import MPCSafetyFilter, LinearMPCSafetyFilter
from utils.unicycle_mpc_acados import AcadosUnicycleMPC


class UniycleMPCSafetyFilterAcados(MPCSafetyFilter):
    """
    Acados-based safety filter for unicycle dynamics.

    Identical to UniycleMPCSafetyFilter from mpc_safety_filter.py except
    that the inner MPC uses acados (SQP + HPIPM) instead of IPOPT via
    do_mpc.

    Parameters
    ----------
    obstacles : list of array-like [cx, cy, r]
    tester    : ReachabilityTester
    dt        : float  — discretisation step
    v         : float  — fixed forward speed
    n_horizon : int    — MPC horizon (steps)
    max_lookback : int — max look-back when searching for safe stopping time
    nominal_tracking : bool — True → minimise (omega - omega_nom)^2
    """

    def __init__(self, obstacles, tester,
                 dt: float = 0.1, v: float = 1.0,
                 n_horizon: int = 10, max_lookback: int = 10,
                 use_safety_radius: bool = True):
        super().__init__(obstacles, tester, max_lookback)
        self.dt        = dt
        self.n_horizon = n_horizon
        self.v         = v

        # Kinematic safety radius: worst-case heading into obstacle at min turning radius.
        # For turning radius R = v / u_max (u_max = 1.0), the vehicle can guarantee
        # avoidance only if it stays outside S = sqrt(r² + 2rR) of each obstacle center.
        # Set use_safety_radius=False to revert to the raw obstacle radii.
        u_max = 1.0  # omega bounds [-1, 1]
        R     = v / u_max
        if use_safety_radius:
            self.safety_obstacles    = [
                np.array([obs[0], obs[1], float(np.sqrt(obs[2] ** 2 + 2 * obs[2] * R))])
                for obs in obstacles
            ]
            self.buffer_init_heading = 0.0
            for obs, sobs in zip(obstacles, self.safety_obstacles):
                print(f"  [safety radius] r={obs[2]:.3f} → S={sobs[2]:.4f}  (R={R:.3f})")
        else:
            self.safety_obstacles    = [np.array(obs, dtype=float) for obs in obstacles]
            self.buffer_init_heading = 0.4
            print(f"  [safety radius] disabled — using raw obstacle radii + heading buffer={self.buffer_init_heading}")

        # Build the acados MPC (solver is compiled on first call)
        self._mpc = AcadosUnicycleMPC(
            obstacles=self.safety_obstacles,
            dt=dt,
            v=v,
            n_horizon=n_horizon,
            nominal_tracking=True,
            solver_name=f'unicycle_acados_sf_{os.getpid()}',
        )

    def _collides(self, bounds: np.ndarray) -> bool:
        """Circle-box overlap test against kinematic safety radii (expanded obstacles)."""
        for obs in self.safety_obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            closest_x = np.clip(cx, bounds[0, 0], bounds[0, 1])
            closest_y = np.clip(cy, bounds[1, 0], bounds[1, 1])
            dist_sq   = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
            if dist_sq <= r ** 2:
                return True
        return False

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray,
                             center: np.ndarray,
                             extra_inflation: float = 0.0) -> tuple:
        """
        Receding-horizon unicycle MPC with constant-size bounds.

        Inflates obstacle radii by the x-y bounding box diagonal (positional
        uncertainty) plus any extra_inflation (e.g. current-timestep bounds
        diagonal from find_stopping_timestep). Plans from the bounds center.
        At each step produces a constant-size box [point ± hw] for _collides.

        Returns (trajectory_bounds, points, controls).
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_x = half_widths[0]
        hw_y = half_widths[1]

        inflation = float(np.sqrt(hw_x ** 2 + hw_y ** 2)) + extra_inflation + self.buffer_init_heading

        self._mpc._obs_inflation_buf[0] = inflation

        x = center.reshape(-1, 1)
        self._mpc.x0 = x
        self._mpc.set_initial_guess()

        trajectory_bounds     = [initial_bounds.copy()]
        points, controls      = [], []
        prev_predicted_states = None

        for _ in range(self.n_horizon):
            if prev_predicted_states is None:
                self._fill_nom_ctrl_buf(x.flatten())
            else:
                self._fill_nom_ctrl_buf_from_states(prev_predicted_states[1:])

            u = self._mpc.make_step(x)

            # Cache predicted states for warm-starting the next step's nom buf
            prev_predicted_states = [
                np.array(self._mpc.opt_x_num['_x', k, 0]).flatten()
                for k in range(self.n_horizon + 1)
            ]

            u_arr  = np.array(u).flatten()
            cl_sys = self.tester.analyzer.cl_system
            x_np   = x.flatten().reshape(1, -1)
            u_np   = u_arr.reshape(1, -1)
            x_next = cl_sys.dynamics.dynamics_step(x_np, u_np)
            if hasattr(x_next, 'numpy'):
                x_next = x_next.numpy()
            x = np.array(x_next).flatten().reshape(-1, 1)

            xf = x.flatten()
            point_bound = np.array([
                [xf[0] - hw_x,           xf[0] + hw_x          ],
                [xf[1] - hw_y,           xf[1] + hw_y          ],
                [xf[2] - half_widths[2], xf[2] + half_widths[2]],
            ])
            trajectory_bounds.append(point_bound)
            points.append(x.copy())
            controls.append(u_arr.copy())

        return trajectory_bounds, points, controls


def make_mpc_safety_filter_acados(tester, obstacles_list=None,
                                   t_step: float = None,
                                   n_horizon: int = 10,
                                   max_lookback: int = 10,
                                   **kwargs):
    """
    Build the acados-based MPCSafetyFilter subclass from a tester object.

    For DoubleIntegrator dynamics the original LinearMPCSafetyFilter
    (do_mpc) is returned unchanged — only the unicycle uses acados.
    """
    if obstacles_list is None:
        obstacles_list = tester.obstacles.obstacle_list

    dynamics_name = type(tester.analyzer.cl_system.dynamics).__name__

    if t_step is None:
        t_step = tester.analyzer.cl_system.dynamics.dt

    if dynamics_name == 'DoubleIntegrator':
        from utils.di_mpc import di_model, di_mpc
        estimator = tester.estimator
        return LinearMPCSafetyFilter(
            A=estimator.A,
            B=estimator.B,
            obstacles=obstacles_list,
            tester=tester,
            t_step=t_step,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
        )

    if dynamics_name == 'Unicycle_NL':
        return UniycleMPCSafetyFilterAcados(
            obstacles=obstacles_list,
            tester=tester,
            dt=t_step,
            v=tester.analyzer.cl_system.dynamics.vt,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
            use_safety_radius=kwargs.get('use_safety_radius', True),
        )

    return None
