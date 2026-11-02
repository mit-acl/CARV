"""
Acados-based MPC Safety Filter.

Mirrors mpc_safety_filter.py but replaces the do_mpc unicycle MPC with
AcadosUnicycleMPC from utils/unicycle_mpc_acados.py.

The planning logic, collision checking, and overall interface are identical
to the original filter — only the inner MPC solver is replaced.
"""

import os
import time
import numpy as np

from mpc_safety_filter import MPCSafetyFilter, LinearMPCSafetyFilter
from utils.unicycle_mpc_acados import AcadosUnicycleMPC

# Single-start solves skip the per-candidate |omega| <= u_max check that the
# multi-start path applies. 'count' (default) keeps that behaviour and only
# counts overshoots; 'reject' raises InputBoundViolation, like multi-start does.
SINGLE_UBOUND = os.environ.get('TTTCARV_SINGLE_UBOUND', 'count')
assert SINGLE_UBOUND in ('count', 'reject'), SINGLE_UBOUND

# Terminal safe set used by _collides_D (acceptance test, extension, ranking).
#   'D'       (default) the box clears every obstacle's D = sqrt(r^2 + 2rR).
#             Invariant for ONE obstacle with the loiter turning away from it;
#             with several obstacles both loiter circles can be blocked.
#   'loiter'  some side s = +-1 exists whose max-turn loiter from EVERY state in
#             the box clears EVERY obstacle forever (see _loiter_side). This set
#             is control-invariant under the loiter itself, for any number of
#             obstacles, on the discrete plant.
TERMINAL = os.environ.get('TTTCARV_TERMINAL', 'D')
assert TERMINAL in ('D', 'loiter'), TERMINAL


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
                 verify_heading_bounds: bool = False,
                 first_step_cost: bool = False,
                 tail_cost_weight: float = 1e-3,
                 oneshot: bool = False,
                 oneshot_refine: int = 2,
                 oneshot_verify: bool = False,
                 oneshot_rti: int = 20):
        super().__init__(obstacles, tester, max_lookback)
        # Opt-in so alg14/alg15 keep their measured baseline behaviour.
        self.verify_input_bounds = verify_input_bounds
        self.verify_heading_bounds = verify_heading_bounds
        self.dt        = dt
        self.n_horizon = n_horizon
        self.v         = v
        # One-shot PSF rollout — see _run_mpc_single_oneshot for why.
        self.oneshot        = oneshot
        self.oneshot_refine = max(1, int(oneshot_refine))
        self.oneshot_verify = oneshot_verify
        # SQP_RTI iterations per one-shot refine pass. Was hard-wired to
        # _RTI_WARMUP_ITERS (20): 2 passes x 20 = 40 iterations = ~69 ms of a
        # 71.5 ms solve on EPYC 7542 (runs/alg22_opcost, job 22706513).
        self.oneshot_rti    = max(1, int(oneshot_rti))
        # Budget hooks, set by the caller (None = unconstrained, the default):
        #   solve_observer(seconds) receives every untruncated single-solve time
        #   solve_gate() -> bool     may another solve start now?
        #   time_left() -> seconds   left in the timestep; a hard deadline for
        #                            the RTI loop and the re-sim rollout
        self.solve_observer = None
        self.solve_gate     = None
        self.time_left      = None
        self.deadline_hits        = 0
        self.truncated_solves     = 0
        self.second_start_skipped = 0
        # single-start |omega| audit (see SINGLE_UBOUND)
        self.single_ub_solves = 0
        self.single_ub_over   = 0
        self.single_ub_peak   = 0.0
        self._solve_truncated     = False
        # Largest observed |opt_x_num - plant re-simulation| over the run.
        # Should stay at machine epsilon; anything else means the OCP model
        # and cl_system.dynamics have drifted apart and the one-shot
        # trajectory is no longer a truthful prediction.
        self.oneshot_model_mismatch = 0.0
        # Same, restricted to plans that PASSED the acceptance test — the only
        # ones ever flown. A mismatch on a plan we reject costs nothing; a
        # mismatch on one we accept is a broken certificate, so the two must
        # not be pooled into a single max.
        self.oneshot_mismatch_accepted = 0.0
        self.oneshot_solves     = 0
        self.oneshot_bad_status = 0

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
        self.first_step_cost = first_step_cost
        self._mpc = AcadosUnicycleMPC(
            obstacles=solver_obstacles,
            dt=dt,
            v=v,
            n_horizon=n_horizon,
            nominal_tracking=True,
            solver_name=f'unicycle_acados_sf_{os.getpid()}',
            turn_radius=solver_turn_R,
            first_step_cost=first_step_cost,
            tail_cost_weight=tail_cost_weight,
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

    def _loiter_side(self, bounds: np.ndarray, prefer=None):
        """
        A side s in {+1, -1} whose max-turn loiter is safe forever from every
        state in `bounds`, or None.

        The loiter is omega = s*u_max on the Euler plant. Each step moves a
        chord of length L = v*dt and turns the heading by phi = u_max*dt, so the
        ticks are the vertices of a regular polygon: they lie on ONE circle of
        radius rho = L / (2 sin(phi/2)), and every chord between them stays at
        least h = rho cos(phi/2) from its centre. From state (p, th) the centre is
            c = p + (L/2)(cos th, sin th) + s*h*(-sin th, cos th),
        a vector of length rho rotated by th. Over the box, c lies within
            e_c = (xy half-diagonal) + 2*rho*sin(hw_th/2)
        of the centre computed at the box centre. The loiter is safe forever if
        every obstacle lies wholly outside the annulus [h, rho] around c:
            |c - o| - e_c >= rho + r   or   |c - o| + e_c <= h - r.
        """
        b = np.asarray(bounds, dtype=float)
        hw = (b[:, 1] - b[:, 0]) / 2.0
        if (not np.all(np.isfinite(hw))) or hw[2] >= np.pi:
            return None
        xc, yc, thc = (b[:, 0] + b[:, 1]) / 2.0
        phi = self.u_max * self.dt
        L = self.v * self.dt
        rho = L / (2.0 * np.sin(phi / 2.0))
        h = rho * np.cos(phi / 2.0)
        e_c = float(np.hypot(hw[0], hw[1])) + 2.0 * rho * np.sin(hw[2] / 2.0)
        sides = (prefer, -prefer) if prefer in (1, -1) else (1, -1)
        for side in sides:
            cx = xc + (L / 2.0) * np.cos(thc) - side * h * np.sin(thc)
            cy = yc + (L / 2.0) * np.sin(thc) + side * h * np.cos(thc)
            ok = True
            for o in self.raw_obstacles:
                d = float(np.hypot(cx - o[0], cy - o[1]))
                if not (d - e_c >= rho + o[2] + 1e-9 or d + e_c <= h - o[2] - 1e-9):
                    ok = False
                    break
            if ok:
                return side
        return None

    def _collides_D(self, bounds: np.ndarray) -> bool:
        if TERMINAL == 'loiter':
            return self._loiter_side(bounds) is None
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
        out = self._run_mpc_single(initial_bounds, center, extra_inflation,
                                   warm_start_omega=None)
        if self.verify_input_bounds:
            peak = max((abs(float(np.asarray(c).flatten()[0])) for c in out[2]),
                       default=0.0)
            self.single_ub_solves += 1
            if (not np.isfinite(peak)) or peak > self.u_max + 1e-6:
                self.single_ub_over += 1
                self.single_ub_peak = max(self.single_ub_peak,
                                          peak if np.isfinite(peak) else float('inf'))
                if SINGLE_UBOUND == 'reject':
                    raise InputBoundViolation(
                        f"input_bound: single-start |omega|={peak:.4f} > "
                        f"u_max={self.u_max:.4f}")
        return out

    def _run_mpc_single(self, initial_bounds, center, extra_inflation,
                        warm_start_omega):
        """Dispatch to the one-shot or the re-simulation rollout."""
        if self.solve_gate is not None and not self.solve_gate():
            self.deadline_hits += 1
            raise MPCDeadline("deadline: no budget left to start a solve")
        self._mpc.time_left   = self.time_left
        self._solve_truncated = False
        _t0 = time.perf_counter()
        if self.oneshot:
            out = self._run_mpc_single_oneshot(
                initial_bounds, center, extra_inflation, warm_start_omega)
        else:
            out = self._run_mpc_single_resim(
                initial_bounds, center, extra_inflation, warm_start_omega)
        # Only complete solves feed the price: a truncated one is short by
        # construction and would bias the admission price low.
        if self._solve_truncated:
            self.truncated_solves += 1
        elif self.solve_observer is not None:
            self.solve_observer(time.perf_counter() - _t0)
        return out

    def _run_mpc_single_resim(self, initial_bounds, center, extra_inflation,
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

        for _k in range(self.n_horizon):
            # Each further step is one RTI iteration plus an NN query. Stop
            # before one that cannot finish; a partial rollout is not a plan.
            if (_k > 0 and self.time_left is not None
                    and self.time_left() < 2.0 * self._mpc._iter_cost + 0.001):
                self.deadline_hits += 1
                raise MPCDeadline("deadline: re-simulation rollout ran out of timestep")
            # u_nom for the COST function tracks the NN policy at each
            # predicted state — this is the standard PSF min-intervention
            # objective. The warm-start above only seeds the initial QP
            # iterate; it does not change what we're minimizing.
            if prev_predicted_states is None:
                self._fill_nom_ctrl_buf(x.flatten())
            else:
                self._fill_nom_ctrl_buf_from_states(prev_predicted_states[1:])

            u = self._mpc.make_step(x)
            if self._mpc.truncated:
                self.deadline_hits += 1
                raise MPCDeadline("deadline: re-simulation warm-up solve cut short")

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

    def _run_mpc_single_oneshot(self, initial_bounds, center, extra_inflation,
                                warm_start_omega):
        """
        One-shot PSF rollout: solve the OCP once and return ITS OWN plan.

        Why this exists
        ---------------
        _run_mpc_single_resim solves the OCP n_horizon times, keeping only u[0]
        each time and stepping the plant forward one step. Each of those solves
        applies its hard terminal-D constraint to its own stage N — which sits
        n_horizon steps beyond wherever the loop currently is. The trajectory
        that comes back only spans n_horizon steps in total, so its endpoint was
        never the terminal stage of any solve. Every solve truthfully promises
        "clear of D in n_horizon steps"; the loop advances one step and the
        promise is renewed rather than kept. The returned endpoint is
        unconstrained, and _rollout_verdict rejects it.

        Solving once and returning the solver's own u[0..N-1] and x[0..N] closes
        that gap: the terminal constraint binds on exactly the endpoint we hand
        back, so a rollout that reaches this point cannot violate terminal-D.

        What is returned is opt_u_seq re-simulated through the PLANT, not
        opt_x_num itself. The two models are identical on paper — disc_dyn_expr
        is x + v*cos(th)*dt, y + v*sin(th)*dt, th + w*dt, and
        Unicycle_NL.dynamics_step is the same explicit Euler with v taken off
        the plant — but that is not enough. opt_x_num is a primal iterate of an
        SQP_RTI solve: its states satisfy the LINEARISED dynamics, and only at
        convergence the nonlinear ones. Measured over 7722 solves, plans that
        passed the full acceptance test differed from their own plant rollout by
        up to 5.2. Returning the iterate would certify a trajectory the robot
        cannot fly.

        Re-simulating instead costs one rollout and is true by construction. The
        terminal-D guarantee still transfers whenever the solve converged, which
        is the entire reason to solve once rather than n_horizon times; when it
        did not converge, the acceptance test sees the real endpoint and rejects
        it. oneshot_verify measures the gap as a convergence diagnostic.

        Bounds are built the same way as the re-simulation path: each nominal
        state carries the INITIAL half-widths, translated rigidly. The box does
        not grow along the horizon in either path, so swapping the state source
        changes nothing about how the certificate is formed.
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_x = half_widths[0]
        hw_y = half_widths[1]

        inflation = float(np.sqrt(hw_x ** 2 + hw_y ** 2)) + extra_inflation + self.buffer_init_heading

        self._mpc._obs_inflation_buf[0] = inflation

        x = center.reshape(-1, 1)
        self._mpc.x0 = x

        if warm_start_omega is not None:
            self._mpc._nom_ctrl_buf[:] = float(warm_start_omega)
        else:
            self._fill_nom_ctrl_buf(x.flatten())
        self._mpc.set_initial_guess()

        # First make_step after set_initial_guess runs _RTI_WARMUP_ITERS
        # iterations, so a single call is already converged. Further passes
        # re-query the NN at the OCP's own predicted states and re-solve —
        # the min-intervention fixed point the re-simulation loop was
        # approaching one step at a time. Each extra pass is 1 RTI iteration.
        predicted_states = None
        for _ in range(self.oneshot_refine):
            if predicted_states is not None:
                self._fill_nom_ctrl_buf_from_states(predicted_states[1:])
            # Force the full RTI budget on every pass. The default drops to a
            # single iteration once _step_count > 0, which is right for a
            # receding-horizon controller warm-started one step at a time but
            # wrong here: each pass changes yref at the SAME x0, and one
            # iteration leaves a primal iterate that does not satisfy the
            # nonlinear dynamics.
            self._mpc.make_step(x, n_rti=self.oneshot_rti)
            predicted_states = [
                np.array(self._mpc.opt_x_num['_x', k, 0]).flatten()
                for k in range(self.n_horizon + 1)
            ]
            if self._mpc.truncated:
                # Out of timestep. The iterate is still re-simulated through the
                # plant below and must pass the acceptance test, so returning it
                # is sound; a further refine pass is not affordable.
                self._solve_truncated = True
                break

        controls = self._mpc.opt_u_seq

        # The trajectory we return is the PLANT's response to these inputs, not
        # the solver's iterate.
        #
        # opt_x_num is a primal iterate of an SQP_RTI solve. Its states satisfy
        # the LINEARISED dynamics at each iteration; only at convergence do they
        # satisfy the nonlinear ones. A partially converged iterate can clear
        # every obstacle and sit outside D at the terminal node while not being
        # a trajectory the robot can fly — measured at |opt_x_num - plant| up to
        # 5.2 on plans that passed the acceptance test. Certifying that would be
        # certifying fiction.
        #
        # Re-simulating opt_u_seq through cl_system.dynamics costs one rollout
        # and makes the returned states true by construction, whatever the
        # solver's convergence state. The terminal-D guarantee still transfers
        # whenever the solve converged — that is the whole point of solving once
        # instead of n_horizon times — and when it did not, the acceptance test
        # sees the real endpoint and rejects it. Sound either way; better
        # whenever acados converged.
        cl_sys      = self.tester.analyzer.cl_system
        xs          = center.flatten().reshape(1, -1)
        plant_states = [center.flatten().copy()]
        for u_arr in controls:
            xs = cl_sys.dynamics.dynamics_step(xs, u_arr.reshape(1, -1))
            if hasattr(xs, 'numpy'):
                xs = xs.numpy()
            xs = np.array(xs).reshape(1, -1)
            plant_states.append(xs.flatten().copy())

        trajectory_bounds = [initial_bounds.copy()]
        points            = []
        for k in range(1, self.n_horizon + 1):
            xf = plant_states[k]
            trajectory_bounds.append(np.array([
                [xf[0] - hw_x,           xf[0] + hw_x          ],
                [xf[1] - hw_y,           xf[1] + hw_y          ],
                [xf[2] - half_widths[2], xf[2] + half_widths[2]],
            ]))
            points.append(xf.reshape(-1, 1).copy())

        if self.oneshot_verify:
            self.oneshot_solves += 1
            if getattr(self._mpc, 'last_status', 0) not in (0, 2):
                self.oneshot_bad_status += 1

            # Distance between the solver's iterate and the real rollout. This
            # is now a CONVERGENCE diagnostic, not a soundness one — the
            # returned trajectory is the plant rollout either way. It still
            # matters: the terminal-D guarantee only transfers from the OCP to
            # the flown plan to the extent these agree, so a large number here
            # predicts term_D rejections rather than unsafe flight.
            worst = max(
                (float(np.max(np.abs(plant_states[k] - predicted_states[k])))
                 for k in range(1, self.n_horizon + 1)),
                default=0.0)

            if worst > self.oneshot_model_mismatch:
                self.oneshot_model_mismatch = worst

            # Split by whether this plan would be accepted. Rejected plans are
            # discarded by the caller, so their mismatch is noise; only the
            # accepted number bears on soundness.
            accepted = not any(self._collides_raw(b)
                               for b in trajectory_bounds[1:-1])
            accepted = accepted and not self._collides_D(trajectory_bounds[-1])
            if accepted and worst > self.oneshot_mismatch_accepted:
                self.oneshot_mismatch_accepted = worst

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
        if TERMINAL == 'loiter' and self._loiter_side(terminal) is None:
            worst_depth += 1.0      # rank loiter-safe terminals first
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
        if (swerve is not None and self.solve_gate is not None
                and not self.solve_gate()):
            # A second solve priced at a whole solve does not fit this timestep.
            self.second_start_skipped += 1
            swerve = None
            if not candidates:
                raise MPCDeadline("deadline: NN start rejected, no budget for the swerve start")
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


class MPCDeadline(Exception):
    """A solve was refused or cut short by the caller's timestep deadline."""


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
            # alg13 passes these; before, **kwargs swallowed them and the DI
            # filter was built against di_mpc's hardcoded defaults instead.
            pos_min=kwargs.get('pos_min', 0.0),
            vel_min=kwargs.get('vel_min', -1.0),
            buffer=kwargs.get('buffer', 0.0),
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
            first_step_cost=kwargs.get('first_step_cost', False),
            oneshot=kwargs.get('oneshot', False),
            oneshot_refine=kwargs.get('oneshot_refine', 2),
            oneshot_verify=kwargs.get('oneshot_verify', False),
            oneshot_rti=kwargs.get('oneshot_rti', 20),
            tail_cost_weight=kwargs.get('tail_cost_weight', 1e-3),
        )

    return None
