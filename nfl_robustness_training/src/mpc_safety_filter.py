"""
MPC-based Safety Filter.

Planning  (_run_mpc_from_bounds):
  1. Compute half-widths (hw_x, hw_y) of the t_back bounding box.
  2. Inflate obstacle radii by diagonal = sqrt(hw_x² + hw_y²).
  3. Build a fresh MPC with the inflated obstacles.
  4. Start from the center of the t_back bounds and propagate via unicycle kinematics.
  5. At each step store a constant-size bound [point ± hw] (same half-widths as t_back).
  6. Reject the plan if any of those boxes overlaps an original (non-inflated) obstacle.

Execution (real_state_mpc in REAL_integrated_sim):
  Apply queued MPC control + Kalman filter for bounds tracking.
"""

import numpy as np

from utils.di_mpc import di_model, di_mpc, di_simulator
from utils.unicycle_mpc import unicycle_model, unicycle_mpc


class MPCSafetyFilter:
    """Base class for MPC safety filters."""

    def __init__(self, obstacles, tester, max_lookback: int = 10):
        self.obstacles    = obstacles if obstacles is not None else []
        self.max_lookback = max_lookback
        self.tester       = tester

    def _fill_nom_ctrl_buf(self, x0: np.ndarray, mpc=None):
        """
        Initial step: roll out NN control from x0 for n_horizon steps to fill the
        nominal control buffer of mpc (defaults to self._mpc).
        """
        if mpc is None:
            mpc = self._mpc
        cl_sys = self.tester.analyzer.cl_system
        xt = x0.flatten().reshape(1, -1)
        for k in range(self.n_horizon + 1):
            u = cl_sys.dynamics.control_nn(xt, cl_sys.controller.cpu())
            mpc._nom_ctrl_buf[k] = u.flatten()[:1]
            xt = cl_sys.dynamics.dynamics_step(xt, u)

    def _fill_nom_ctrl_buf_from_states(self, states: list, mpc=None):
        """
        Query the NN at each MPC-predicted state to fill the nominal-control
        buffer of mpc (defaults to self._mpc).
        """
        if mpc is None:
            mpc = self._mpc
        cl_sys  = self.tester.analyzer.cl_system
        buf_len = len(mpc._nom_ctrl_buf)
        filled  = min(len(states), buf_len)

        batch = np.array([np.array(states[k]).flatten() for k in range(filled)])
        us    = cl_sys.dynamics.control_nn(batch, cl_sys.controller.cpu())
        mpc._nom_ctrl_buf[:filled] = us[:, :1]
        if filled < buf_len:
            mpc._nom_ctrl_buf[filled:] = us[-1, :1]

    def _collides(self, bounds: np.ndarray) -> bool:
        """Circle-box overlap test against original (non-inflated) obstacles."""
        for obs in self.obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            closest_x = np.clip(cx, bounds[0, 0], bounds[0, 1])
            closest_y = np.clip(cy, bounds[1, 0], bounds[1, 1])
            dist_sq   = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
            if dist_sq <= r ** 2:
                return True
        return False

    def find_stopping_timestep(self, collision_timestep: int, horizons: dict,
                               current_t: int = 0):
        """
        Find the safe stopping timestep closest to collision.

        Returns (t_back, controls, traj_bounds) or (None, [], []).
        """
        for lookback in range(3, self.max_lookback + 1):
            t_back = collision_timestep - lookback

            if t_back < max(0, current_t):
                break

            bounds_at_back = horizons[t_back].get_tight_bound()
            center         = (bounds_at_back[:, 0] + bounds_at_back[:, 1]) / 2.0
            print(f"Initial Center: {center}")


            # Extra inflation from current timestep's KF bounds uncertainty.
            cur_bounds = horizons.get(current_t)
            if cur_bounds is not None:
                cb = cur_bounds.get_tight_bound()
                dx_c = cb[0, 1] - cb[0, 0]
                dy_c = cb[1, 1] - cb[1, 0]
                current_inflation = float(np.sqrt((dx_c / 2) ** 2 + (dy_c / 2) ** 2))
            else:
                current_inflation = 0.0

            try:
                print(f"====== t_back: {t_back} =======")
                theta_info = (f"  theta_center={np.round(center[2], 4)}"
                              f"  theta_bounds=[{np.round(bounds_at_back[2, 0], 4)}, "
                              f"{np.round(bounds_at_back[2, 1], 4)}]"
                              if len(center) > 2 else "")
                print(f"  [MPC parent bounds] p={np.round(bounds_at_back[0], 4)}  "
                      f"center={np.round(center[:2], 4)}  cur_inflation={current_inflation:.4f}"
                      f"{theta_info}")

                traj_bounds, _, controls = self._run_mpc_from_bounds(
                    bounds_at_back, center, extra_inflation=current_inflation)
            except Exception as e:
                print(f"  [MPC filter] failed at t_back={t_back}: {e}")
                continue

            collision_found    = False
            mpc_collision_step = None
            for step, b in enumerate(traj_bounds[1:], start=1):
                if self._collides(b):
                    collision_found    = True
                    mpc_collision_step = step
                    break

            if not collision_found:
                print(f"  [MPC filter] Safe. Stopping timestep = {t_back}")
                return t_back, controls, traj_bounds
            else:
                print(f"  [MPC filter] Collision at step {mpc_collision_step} "
                      f"from t_back={t_back}, going further back.")
                print(f"controls were {controls}")

        print(f"  [MPC filter] No safe stopping timestep found at or after t={current_t}.")
        return None, [], []


class LinearMPCSafetyFilter(MPCSafetyFilter):
    """Safety filter for linear DI dynamics."""

    def __init__(self, A, B, obstacles, tester,
                 t_step: float = 0.1, n_horizon: int = 10, max_lookback: int = 10):
        super().__init__(obstacles, tester, max_lookback)
        self.t_step    = t_step
        self.n_horizon = n_horizon
        self.A         = A
        self.B         = B

        self._model = di_model(self.A, self.B, t_step=t_step)
        self._mpc   = di_mpc(self._model, obstacles=obstacles, t_step=t_step,
                             n_horizon=n_horizon, nominal_tracking=True)

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray,
                             extra_inflation: float = 0.0) -> list:
        """
        Receding-horizon MPC with constant-size bounds.
        MPC is rebuilt with obstacles inflated by t_back diagonal + extra_inflation.
        Trajectory bounds at each step are constant-size boxes [point +- half width].
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_x        = half_widths[0]
        hw_y        = half_widths[1]
        inflation   = float(np.sqrt(hw_x ** 2 + hw_y ** 2)) + extra_inflation

        inflated_obs = [np.array([obs[0], obs[1], obs[2] + inflation])
                        for obs in self.obstacles]

        model = di_model(self.A, self.B, t_step=self.t_step)
        mpc   = di_mpc(model, obstacles=inflated_obs, t_step=self.t_step,
                       n_horizon=self.n_horizon, nominal_tracking=True)

        x = center.reshape(2, 1)
        mpc.x0 = x
        mpc.set_initial_guess()

        trajectory_bounds      = [initial_bounds.copy()]
        points, controls       = [], []
        prev_predicted_states  = None

        for _ in range(self.n_horizon):
            if prev_predicted_states is None:
                self._fill_nom_ctrl_buf(x.flatten(), mpc)
            else:
                self._fill_nom_ctrl_buf_from_states(prev_predicted_states[1:], mpc)
            u = mpc.make_step(x)
            prev_predicted_states = [
                np.array(mpc.opt_x_num['_x', k, 0]).flatten()
                for k in range(self.n_horizon + 1)
            ]
            u_arr = np.array(u).flatten()
            x     = self.A @ x + self.B @ u_arr.reshape(-1, 1)
            xf    = x.flatten()
            point_bound = np.array([
                [xf[0] - hw_x, xf[0] + hw_x],
                [xf[1] - hw_y, xf[1] + hw_y],
            ])
            trajectory_bounds.append(point_bound)
            points.append(x.copy())
            controls.append(u_arr.copy())

        return trajectory_bounds, points, controls


class UniycleMPCSafetyFilter(MPCSafetyFilter):
    """Safety filter for unicycle dynamics."""

    def __init__(self, obstacles, tester,
                 dt: float = 0.1, v: float = 1.0,
                 n_horizon: int = 10, max_lookback: int = 10, nominal_tracking=True):
        super().__init__(obstacles, tester, max_lookback)
        self.dt        = dt
        self.n_horizon = n_horizon
        self.v         = v
        self.buffer_init_heading = 0.2

        self._nominal_tracking = nominal_tracking

        # Relaxed IPOPT tolerances — faster convergence
        self._fast_opts = {
            'ipopt.tol':                   1e-4,
            'ipopt.max_iter':              50,
            'ipopt.warm_start_init_point': 'yes',
        }

        # Baseline MPC (no inflation) — used only for warm-starting _nom_ctrl_buf
        # when no prior predicted trajectory is available.
        self._model = unicycle_model(dt=dt, v=v)
        self._mpc   = unicycle_mpc(self._model, obstacles=obstacles, dt=dt,
                                   n_horizon=n_horizon, nominal_tracking=nominal_tracking,
                                   solver_opts=self._fast_opts)

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray, center: np.ndarray,
                             extra_inflation: float = 0.0) -> list:
        """
        Receding-horizon unicycle MPC with constant-size bounds.

        1. Compute diagonal half-length of the t_back x-y bounding box → t_back inflation.
        2. Add extra_inflation (diagonal half-length of current timestep's KF bounds).
        3. Rebuild MPC obstacle radii via TVP so the planner steers clear by the total.
        4. Propagate the center using dynamics_step (NN).
        5. At each step, produce a constant-size bound [point ± hw] using the
           t_back half-widths. The box-circle _collides test captures worst-case
           positional uncertainty.
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_x        = half_widths[0]
        hw_y        = half_widths[1]

        buffer_for_init_heading = self.buffer_init_heading

        inflation   = float(np.sqrt(hw_x ** 2 + hw_y ** 2)) + extra_inflation + buffer_for_init_heading
        print(f"  [MPC bounds] hw=({hw_x:.4f},{hw_y:.4f})  "
              f"t_back_inf={inflation - extra_inflation:.4f}  "
              f"cur_inf={extra_inflation:.4f}  total={inflation:.4f}")

        # Update the inflation TVP — no model rebuild needed.
        self._mpc._obs_inflation_buf[0] = inflation

        x = center.reshape(-1, 1)
        self._mpc.x0 = x
        self._mpc.set_initial_guess()

        trajectory_bounds     = [initial_bounds.copy()]
        points, controls      = [], []
        prev_predicted_states = None

        for _ in range(self.n_horizon):
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
            u_arr  = np.array(u).flatten()
            cl_sys = self.tester.analyzer.cl_system
            x_np   = x.flatten().reshape(1, -1)
            u_np   = u_arr.reshape(1, -1)
            x_next = cl_sys.dynamics.dynamics_step(x_np, u_np)
            if hasattr(x_next, 'numpy'):
                x_next = x_next.numpy()
            x = np.array(x_next).flatten().reshape(-1, 1)

            # Constant-size bound centered at trajectory point
            xf = x.flatten()
            point_bound = np.array([
                [xf[0] - hw_x,          xf[0] + hw_x         ],
                [xf[1] - hw_y,          xf[1] + hw_y         ],
                [xf[2] - half_widths[2], xf[2] + half_widths[2]],
            ])
            trajectory_bounds.append(point_bound)
            points.append(x.copy())
            controls.append(u_arr.copy())

        return trajectory_bounds, points, controls


def make_mpc_safety_filter(tester, obstacles_list=None, t_step: float = None,
                            n_horizon: int = 10, max_lookback: int = 10, nominal_tracking=True):
    """Build the MPCSafetyFilter subclass from a tester object."""
    if obstacles_list is None:
        obstacles_list = tester.obstacles.obstacle_list

    dynamics_name = type(tester.analyzer.cl_system.dynamics).__name__

    if t_step is None:
        t_step = tester.analyzer.cl_system.dynamics.dt

    if dynamics_name == 'DoubleIntegrator':
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
        return UniycleMPCSafetyFilter(
            obstacles=obstacles_list,
            tester=tester,
            dt=t_step,
            v=tester.analyzer.cl_system.dynamics.vt,
            n_horizon=n_horizon,
            max_lookback=max_lookback,
            nominal_tracking=nominal_tracking,
        )

    return None
