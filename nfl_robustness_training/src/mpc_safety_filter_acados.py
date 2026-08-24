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
                 use_safety_radius: bool = True,
                 split_terminal_D: bool = False,
                 verify_input_bounds: bool = False,
                 verify_heading_bounds: bool = False):
        super().__init__(obstacles, tester, max_lookback)
        # Opt-in so alg14/alg15 keep their measured baseline behaviour.
        self.verify_input_bounds = verify_input_bounds
        self.verify_heading_bounds = verify_heading_bounds
        self.dt        = dt
        self.n_horizon = n_horizon
        self.v         = v

        # Kinematic safety radius: worst-case heading into obstacle at min turning radius.
        # For turning radius R = v / u_max (u_max = 1.0), the vehicle can guarantee
        # avoidance only if it stays outside D = sqrt(r² + 2rR) of each obstacle center.
        # Set use_safety_radius=False to revert to the raw obstacle radii.
        u_max = 1.0  # omega bounds [-1, 1]; MUST match acados lbu/ubu
        R     = v / u_max
        # Single source of truth. The D-geometry below and the input-bound
        # check in _run_mpc_from_bounds must use the SAME u_max: D assumes the
        # vehicle never turns faster than this, so a plan that exceeds it
        # invalidates D rather than merely exceeding the actuator.
        self.u_max            = u_max
        self.turn_radius      = R
        self.split_terminal_D = split_terminal_D

        # Raw (physical) obstacle radii — the true collision boundary.
        self.raw_obstacles = [np.array(obs, dtype=float) for obs in obstacles]
        # D-inflated radii — the danger region where avoidance can't be guaranteed.
        self.safety_obstacles = [
            np.array([obs[0], obs[1], float(np.sqrt(obs[2] ** 2 + 2 * obs[2] * R))])
            for obs in obstacles
        ]

        if split_terminal_D:
            # Bug-trap fix: MPC path constraint uses RAW radii (backup may route
            # through D); terminal constraint uses D (backup must end outside D,
            # where the loiter is control-invariant → persistent feasibility).
            self.buffer_init_heading = 0.0
            solver_obstacles = self.raw_obstacles   # path = raw
            solver_turn_R    = R                     # terminal = D
            for obs, sobs in zip(self.raw_obstacles, self.safety_obstacles):
                print(f"  [split D] path r={obs[2]:.3f}  terminal D={sobs[2]:.4f}  (R={R:.3f})")
        elif use_safety_radius:
            self.buffer_init_heading = 0.0
            solver_obstacles = self.safety_obstacles  # path & terminal both = D
            solver_turn_R    = None
            for obs, sobs in zip(self.raw_obstacles, self.safety_obstacles):
                print(f"  [safety radius] r={obs[2]:.3f} → D={sobs[2]:.4f}  (R={R:.3f})")
        else:
            self.safety_obstacles    = list(self.raw_obstacles)
            self.buffer_init_heading = 0.4
            solver_obstacles = self.raw_obstacles
            solver_turn_R    = None
            print(f"  [safety radius] disabled — using raw obstacle radii + heading buffer={self.buffer_init_heading}")

        # Build the acados MPC (solver is compiled on first call)
        self._mpc = AcadosUnicycleMPC(
            obstacles=solver_obstacles,
            dt=dt,
            v=v,
            n_horizon=n_horizon,
            nominal_tracking=True,
            solver_name=f'unicycle_acados_sf_{os.getpid()}',
            turn_radius=solver_turn_R,
        )

    @staticmethod
    def _overlaps(bounds: np.ndarray, obstacle_set) -> bool:
        """Circle-box overlap test against the given obstacle set."""
        for obs in obstacle_set:
            cx, cy, r = obs[0], obs[1], obs[2]
            closest_x = np.clip(cx, bounds[0, 0], bounds[0, 1])
            closest_y = np.clip(cy, bounds[1, 0], bounds[1, 1])
            dist_sq   = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
            if dist_sq <= r ** 2:
                return True
        return False

    def _collides(self, bounds: np.ndarray) -> bool:
        """Overlap against kinematic safety radii D (expanded obstacles)."""
        return self._overlaps(bounds, self.safety_obstacles)

    def _collides_raw(self, bounds: np.ndarray) -> bool:
        """Overlap against the raw (physical) obstacle radii — true collision."""
        return self._overlaps(bounds, self.raw_obstacles)

    def _collides_D(self, bounds: np.ndarray) -> bool:
        """Overlap against the danger region D — alias of _collides for clarity."""
        return self._overlaps(bounds, self.safety_obstacles)

    def _run_mpc_from_bounds(self, initial_bounds: np.ndarray,
                             center: np.ndarray,
                             extra_inflation: float = 0.0,
                             multi_start: bool = False) -> tuple:
        """
        Receding-horizon unicycle MPC with constant-size bounds.

        Inflates obstacle radii by the x-y bounding box diagonal (positional
        uncertainty) plus any extra_inflation (e.g. current-timestep bounds
        diagonal from find_stopping_timestep). Plans from the bounds center.
        At each step produces a constant-size box [point ± hw] for _collides.

        Cost is the standard PSF min-intervention objective ||u - u_nom||²
        where u_nom is the NN policy's recommended control at each predicted
        state.

        Parameters
        ----------
        multi_start : bool
            If True, runs the MPC from several pre-solve warm-starts
            (NN-rollout, omega=+1, omega=-1, omega=0) and returns the rollout
            whose terminal least violates the terminal D constraint, breaking
            ties by total tracking cost. This rescues build_mpc_backup from
            SQP_RTI local minima where the NN warm-start sits in a basin that
            keeps the terminal inside D regardless of iteration count.

        Returns (trajectory_bounds, points, controls).
        """
        # A heading bound is an OVER-approximation of where theta may lie. Once
        # its half-width reaches pi the interval spans >= 2*pi, i.e. every
        # heading -- it excludes nothing, so it constrains nothing, and any
        # certificate derived from it is vacuous rather than conservative.
        # Refusing here (before the solve) is fail-safe: build_mpc_backup
        # catches this and records mpc_buffer[tau] = None, so the timestep
        # falls back to the certified-clear passthrough window instead of
        # flying a plan built on a bound that proves nothing.
        if self.verify_heading_bounds:
            hw_theta = float(initial_bounds[2, 1] - initial_bounds[2, 0]) / 2.0
            if (not np.isfinite(hw_theta)) or hw_theta >= np.pi:
                raise VacuousHeadingBound(
                    f"vacuous heading bound: hw_theta={hw_theta:.4f} >= pi "
                    f"(interval spans >= 2*pi, constrains nothing)")

        if multi_start:
            return self._run_mpc_multi_start(initial_bounds, center, extra_inflation)
        return self._run_mpc_single(initial_bounds, center, extra_inflation,
                                     warm_start_omega=None)

    def _run_mpc_single(self, initial_bounds, center, extra_inflation,
                        warm_start_omega):
        """
        One MPC rollout. warm_start_omega:
          None  → NN-rollout warm-start (default)
          float → constant ω fills _nom_ctrl_buf before set_initial_guess
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_x = half_widths[0]
        hw_y = half_widths[1]

        inflation = float(np.sqrt(hw_x ** 2 + hw_y ** 2)) + extra_inflation + self.buffer_init_heading

        self._mpc._obs_inflation_buf[0] = inflation

        x = center.reshape(-1, 1)
        self._mpc.x0 = x

        # Force the warm-start basin BEFORE set_initial_guess rolls forward.
        if warm_start_omega is not None:
            self._mpc._nom_ctrl_buf[:] = float(warm_start_omega)
        else:
            self._fill_nom_ctrl_buf(x.flatten())
        self._mpc.set_initial_guess()

        trajectory_bounds     = [initial_bounds.copy()]
        points, controls      = [], []
        prev_predicted_states = None

        for _ in range(self.n_horizon):
            # u_nom for the COST function tracks the NN policy at each
            # predicted state — this is the standard PSF min-intervention
            # objective. The warm-start above only seeds the initial QP
            # iterate; it does not change what we're minimizing.
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

    def _score_rollout(self, traj_bounds, controls):
        """
        Score a rollout for multi-start ranking. Lower is better.
        Primary: terminal D-violation depth (worst obstacle). Zero if outside D.
        Secondary: tracking cost Σω² as a proxy (we don't have ω_nom here at
        rank time; squared-effort breaks ties in favor of gentler swerves).
        """
        terminal = traj_bounds[-1]
        cx = (terminal[0, 0] + terminal[0, 1]) / 2.0
        cy = (terminal[1, 0] + terminal[1, 1]) / 2.0
        worst_depth = 0.0
        for obs_raw, obs_D in zip(self.raw_obstacles, self.safety_obstacles):
            d = float(np.sqrt((cx - obs_raw[0]) ** 2 + (cy - obs_raw[1]) ** 2))
            depth = obs_D[2] - d
            if depth > worst_depth:
                worst_depth = depth
        effort = float(np.sum(np.square(controls)))
        return (worst_depth, effort)

    def _heuristic_swerve_sign(self, center):
        """
        Pick +1 (swerve left) or -1 (swerve right) based on which side of the
        current heading the nearest obstacle lies. Returns None if no obstacles
        or only-far-obstacles.
        """
        if not self.raw_obstacles:
            return None
        x, y, theta = float(center[0]), float(center[1]), float(center[2])
        worst, worst_d2 = None, float('inf')
        for obs in self.raw_obstacles:
            cx, cy = float(obs[0]), float(obs[1])
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if d2 < worst_d2:
                worst_d2 = d2
                worst = (cx, cy)
        dx, dy = worst[0] - x, worst[1] - y
        # body-y > 0 means obstacle is to the LEFT of heading → swerve right (-1)
        body_y = -dx * np.sin(theta) + dy * np.cos(theta)
        return -1.0 if body_y > 0.0 else +1.0

    def _run_mpc_multi_start(self, initial_bounds, center, extra_inflation):
        """
        Try NN warm-start first. If its terminal is inside D (rejected by
        alg14's _collides_D), try a single heuristic swerve warm-start picked
        by obstacle bearing. Returns the rollout with the best score.

        Two solves max (~80ms), down from four. Matches the live MPC budget.
        """
        candidates = []
        rejected   = []
        # 1) NN warm-start (standard PSF behaviour)
        nn_rollout = self._run_mpc_single(initial_bounds, center,
                                          extra_inflation,
                                          warm_start_omega=None)
        if self._input_bounds_ok(nn_rollout[2], rejected):
            nn_score = self._score_rollout(nn_rollout[0], nn_rollout[2])
            candidates.append((nn_score, nn_rollout))
            if nn_score[0] <= 1e-3:
                return nn_rollout    # NN already clears D — no second solve.

        # 2) Heuristic swerve based on obstacle bearing.
        swerve = self._heuristic_swerve_sign(center.flatten())
        if swerve is not None:
            sw_rollout = self._run_mpc_single(initial_bounds, center,
                                              extra_inflation,
                                              warm_start_omega=swerve)
            if self._input_bounds_ok(sw_rollout[2], rejected):
                sw_score = self._score_rollout(sw_rollout[0], sw_rollout[2])
                candidates.append((sw_score, sw_rollout))

        if not candidates:
            raise InputBoundViolation(
                "input_bound: every rollout exceeded |omega| <= "
                f"{self.u_max:.4f} (peaks={['%.4f' % r for r in rejected]}); "
                "D geometry assumes |omega| <= u_max")

        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    def _input_bounds_ok(self, controls, rejected):
        """
        True if every control in the rollout satisfies the box the OCP declares.

        acados enforces lbu/ubu only when it CONVERGES; make_step returns the
        last iterate unclipped otherwise, and the rollout is then integrated
        with the true dynamics — so an out-of-box omega yields a path that is
        geometrically checked but physically unflyable, and one that breaks the
        D = sqrt(r^2 + 2rR) geometry the terminal test rests on.

        Rejecting per-candidate (not per-build) keeps a legal rollout when the
        other one is bad. Off by default: see verify_input_bounds.
        """
        if not self.verify_input_bounds:
            return True
        peak = max((abs(float(np.asarray(c).flatten()[0])) for c in controls),
                   default=0.0)
        if (not np.isfinite(peak)) or peak > self.u_max + 1e-6:
            rejected.append(peak)
            print(f"  [input_bound] candidate discarded: |omega|={peak:.4f} "
                  f"> u_max={self.u_max:.4f}")
            return False
        return True


class InputBoundViolation(Exception):
    """Every candidate rollout violated the OCP's own |omega| <= u_max box."""


class VacuousHeadingBound(Exception):
    """Initial heading interval spans >= 2*pi, so it certifies nothing."""


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
            split_terminal_D=kwargs.get('split_terminal_D', False),
            verify_input_bounds=kwargs.get('verify_input_bounds', False),
            verify_heading_bounds=kwargs.get('verify_heading_bounds', False),
        )

    return None
