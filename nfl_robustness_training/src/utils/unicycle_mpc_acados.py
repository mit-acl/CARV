"""
Acados-based MPC for unicycle dynamics.

Provides AcadosUnicycleMPC — a drop-in replacement for the do_mpc unicycle
MPC used by the safety filter.  The external interface is identical:

  mpc._obs_inflation_buf[0] = inflation   # set before each solve
  mpc._nom_ctrl_buf[k]      = omega_nom   # set before each solve (nominal tracking)
  mpc.x0                    = x           # setter
  mpc.set_initial_guess()                 # warm-start
  u = mpc.make_step(x)                    # solve, return u shape (1,1)
  mpc.opt_x_num['_x', k, 0]              # predicted state at stage k

Cost formulation
----------------
NONLINEAR_LS with GAUSS_NEWTON hessian approximation is used for the
nominal-tracking case.  This produces a positive semi-definite QP at every
SQP iteration, which HPIPM can always solve reliably — unlike EXACT hessian
which can be indefinite when obstacle constraints are active.

  stage cost:    ||omega - omega_nom||^2  (omega_nom set via yref each step)
  terminal cost: 0
  constraints:   (x-cx)^2 + (y-cy)^2 >= (r + obs_inflation)^2  per obstacle
                 -1 <= omega <= 1
"""

import os
import ctypes
import numpy as np
import casadi as ca

# ---------------------------------------------------------------------------
# Pre-load acados shared libraries so ctypes finds them regardless of
# LD_LIBRARY_PATH in the current shell.
# ---------------------------------------------------------------------------
_ACADOS_LIB_DIR    = os.environ.get('ACADOS_LIB_DIR',    '/home/sazhang/acados/lib')
_ACADOS_SOURCE_DIR = os.environ.get('ACADOS_SOURCE_DIR',  '/home/sazhang/acados')

for _lib in ['libblasfeo.so', 'libhpipm.so',
             'libqpOASES_e.so.3.1', 'libqpOASES_e.so', 'libacados.so']:
    _p = os.path.join(_ACADOS_LIB_DIR, _lib)
    if os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

os.environ.setdefault('ACADOS_SOURCE_DIR', _ACADOS_SOURCE_DIR)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel  # noqa: E402


# ---------------------------------------------------------------------------
# Internal accessor to mimic do_mpc's  opt_x_num['_x', k, 0]  interface
# ---------------------------------------------------------------------------
class _OptXAccessor:
    def __init__(self, predicted_x: list):
        self._x = predicted_x   # list of np.ndarray, length N+1

    def __getitem__(self, key):
        # key expected as ('_x', k, 0)
        _, k, _ = key
        return self._x[k]


# ---------------------------------------------------------------------------
# AcadosUnicycleMPC
# ---------------------------------------------------------------------------
class AcadosUnicycleMPC:
    """
    Acados OCP solver for unicycle MPC.

    Parameters
    ----------
    obstacles : list of array-like [cx, cy, r]
        Circular obstacles. Positions/radii are fixed at build time;
        inflation is updated per-solve via _obs_inflation_buf.
    dt : float
        Discretisation time step (seconds).
    v : float
        Fixed forward speed.
    n_horizon : int
        MPC prediction horizon (steps).
    solver_name : str
        Base name for generated C code and JSON file.
    """

    def __init__(self, obstacles: list, dt: float = 0.2, v: float = 1.0,
                 n_horizon: int = 10, nominal_tracking: bool = True,
                 solver_name: str = 'unicycle_acados',
                 turn_radius: float = None,
                 first_step_cost: bool = False,
                 tail_cost_weight: float = 1e-3):
        # nominal_tracking kept as parameter for call-site compatibility but
        # non-nominal path is removed — always uses NONLINEAR_LS tracking cost.
        #
        # turn_radius (R): if given, enables the path/terminal constraint split.
        #   Path nodes (0..N-1) are constrained against the RAW obstacle radii in
        #   `obstacles`, while the terminal node (N) is constrained against the
        #   inflated danger radius S = sqrt(r^2 + 2rR). This lets the optimizer
        #   route a backup THROUGH S (fixing the bug-trap) while still ending
        #   outside S, where the max-turn loiter is control-invariant — which
        #   preserves persistent feasibility. If None, path and terminal share
        #   the same radii in `obstacles` (legacy behaviour).
        self.n_horizon   = n_horizon
        self.dt          = dt
        self.v           = v
        self.obstacles   = obstacles if obstacles is not None else []
        self.turn_radius = turn_radius

        # first_step_cost: charge the min-intervention cost only at stage 0.
        #
        # The default objective is sum_k (omega_k - omega_nom,k)^2 over the whole
        # horizon, so the tail is pulled toward the NN policy even though only
        # the first input is ever applied. Under a receding-horizon filter that
        # is a constraint on the solver with nothing to buy: every later input
        # is re-planned before it would be used. Dropping the tail weight gives
        # the solver the whole horizon to find a safe continuation while still
        # deviating minimally from what the NN wants RIGHT NOW.
        #
        # tail_cost_weight is small but NOT zero. The Gauss-Newton Hessian of a
        # NONLINEAR_LS stage is J^T W J and no Levenberg-Marquardt term is
        # configured (GAUSS_NEWTON, see solver_options below), so W = 0 would
        # make those blocks exactly singular. That is the reason for the floor,
        # but note it was NOT the cause of any observed failure: a 24-trial
        # sweep over 1e-3/1e-2/1e-1 recorded ZERO solver errors at every weight.
        #
        # MEASURED (2026-08-31, 24 trials/arm, seed 20260831). Down-weighting
        # the tail triples re-solve rejections (4 -> 12-14) and the reason is
        # always rollout_rejected, never solver_error. Applied-control deviation
        # from nominal drops with the flag on: mean |u_mpc - u_nn| 1.08 -> 0.76,
        # max 4.20 -> 2.18. Safety was 0/24 unsafe in every arm — this is a
        # feasibility effect, not a safety one.
        #
        # MECHANISM (32 trials/arm, rejections broken down by check and node):
        #   control    13 rejects =  7 path_raw (nodes 2,5,8) +  6 term_D
        #   firststep  24 rejects =  4 path_raw               + 20 term_D
        # path_raw goes DOWN; the entire increase is term_D at the terminal
        # node. The plans do not clip obstacles earlier — they fail to leave the
        # danger region D within the horizon.
        #
        # Why: _run_mpc_single returns a RE-SIMULATION (solve, take u[0], step,
        # repeat), not the OCP's own prediction. Each solve enforces its hard
        # terminal-D constraint on ITS stage-N state, n_horizon steps beyond
        # wherever the inner loop currently sits, while the returned trajectory
        # only reaches step n_horizon in total. So every solve truthfully
        # promises "clear of D in n_horizon steps", the loop advances one step,
        # and the promise is renewed instead of kept. Whole-horizon tracking
        # hides this because the plan follows the NN's goal-seeking path and
        # exits D anyway; with the tail freed the controls are weaker
        # (mean |u_mpc - u_nn| 1.08 -> 0.76) and the endpoint is still in D.
        #
        # NOTE the numbers above were measured on the RE-SIMULATION path and do
        # not transfer to the one-shot path (TTTCARV_ONESHOT=1), where the
        # returned endpoint IS the stage-N state the hard terminal constraint
        # was applied to. Re-measure before quoting them.
        #
        # Fix, since implemented as the one-shot rollout: return the plan the
        # terminal constraint actually bound. The first attempt returned
        # opt_x_num directly, on the argument that this is sound as long as the
        # OCP's DISCRETE model matches cl_system.dynamics. THAT ARGUMENT IS
        # WRONG. The models do match (both are the same explicit Euler step, same
        # dt) and it was still unsound: opt_x_num is an SQP_RTI primal iterate
        # satisfying the LINEARIZED dynamics, and |opt_x_num - plant| measured
        # 5.169 on plans the acceptance test had ACCEPTED. Model fidelity was
        # never the binding gate; solver convergence was.
        #
        # So the one-shot path returns opt_u_seq re-simulated through
        # cl_system.dynamics. Those states are true whatever the solver did.
        # What convergence still buys is the terminal GUARANTEE: the hard
        # constraint transfers to the flown endpoint only insofar as the solve
        # converged, so a partially converged solve shows up as a term_D
        # rejection rather than as an unsafe trajectory.
        #
        # first_step_cost is left OFF by default: on the re-simulation path it
        # cost feasibility for no measured benefit. Safety was 0/32 unsafe in
        # both arms. It is the principled cost only when u[0] alone is applied.
        self.first_step_cost  = first_step_cost
        self.tail_cost_weight = tail_cost_weight

        # Buffers — same attribute names as the do_mpc version
        self._obs_inflation_buf = np.zeros(1)
        self._nom_ctrl_buf      = np.zeros((n_horizon + 1, 1))

        # Internal state
        self._x0          = np.zeros((3, 1))
        self._step_count  = 0
        self._predicted_x = [np.zeros(3) for _ in range(n_horizon + 1)]
        # Full optimal input sequence from the last solve. The re-simulation
        # rollout only ever needs u[0], but a one-shot PSF solve needs the
        # whole sequence: it is the plan whose terminal state the hard
        # terminal constraint was actually applied to.
        self._predicted_u = [np.zeros(1) for _ in range(n_horizon)]
        # acados status of the most recent solve. A nonzero status means
        # _predicted_x/_predicted_u are whatever the solver last held, not a
        # converged plan — the one-shot path must not treat those as a
        # prediction of anything.
        self.last_status = 0

        # ── Per-process codegen paths ─────────────────────────────────
        # run_trials_random_obs runs N worker PROCESSES on one node and every
        # one constructs this class. solver_name was a fixed string, so all N
        # shared /tmp/unicycle_acados_gen and the .so built inside it -- and
        # each worker rmtree(s) that shared directory immediately before
        # compiling into it (just below). Concurrent workers therefore delete
        # each other in-progress codegen, and a worker can load a library
        # built from another process half-written sources. AcadosOcpSolver
        # does NOT raise when this happens: the loser would get a solver that
        # never produces a usable plan, silently, for the life of the process.
        #
        # CORRECTION (2026-09-11): an earlier version of this comment claimed
        # this race was the MEASURED cause of the dead-worker mode, where a
        # worker ran every trial with mpc_calls=0 and flew unfiltered nominal.
        # That claim was wrong and is retracted. The 16-worker "confirmation"
        # was underpowered -- the mode needs ~64 workers to show up at all --
        # and a 64-worker regression arm run WITH the pid suffix in place
        # returned 9 dead workers. The actual cause was found afterwards in
        # alg22: budget.mpc_cost is calibrated once per process, and a worker
        # whose single startup measurement landed above the 0.20s timestep
        # budget could never satisfy `budget.remaining >= budget.mpc_cost`, so
        # it skipped every MPC build for the rest of its life through an
        # uncounted `break`. See the calibration sanity guard in alg22.py.
        #
        # The pid suffix stays on its own merits: a shared /tmp directory that
        # every worker rmtree(s) immediately before compiling into it is a real
        # hazard whether or not it ever fired. Giving each process its own
        # directory, .so and json is what makes the rmtree below safe at all.
        # It is hardening, not a fix for anything observed.
        solver_name = f"{solver_name}_p{os.getpid()}"

        # Delete stale cached files so the new formulation is compiled fresh
        import shutil
        for path in [f'/tmp/{solver_name}_gen', f'/tmp/{solver_name}_ocp.json']:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
        # One codegen tree per process would otherwise accumulate in /tmp over
        # a long pool run.
        import atexit as _atexit
        _atexit.register(shutil.rmtree, f"/tmp/{solver_name}_gen", True)

        self._solver = self._build_solver(solver_name)

        # W is a numerical parameter, not part of the generated code, so this
        # re-weights the compiled solver without a rebuild.
        if self.first_step_cost:
            for k in range(1, self.n_horizon):
                self._solver.cost_set(k, 'W',
                                      np.array([[float(self.tail_cost_weight)]]))
            print(f"  [MPC cost] first-step only: stage 0 weight=1.0,"
                  f" stages 1..{self.n_horizon - 1} weight={self.tail_cost_weight}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_solver(self, solver_name: str) -> AcadosOcpSolver:
        n_obs = len(self.obstacles)

        model = AcadosModel()
        model.name = solver_name

        x_sym = ca.SX.sym('x', 3)   # [x, y, theta]
        u_sym = ca.SX.sym('u', 1)   # [omega]
        # Single parameter: obs_inflation (scalar added to all obstacle radii)
        p_sym = ca.SX.sym('p', 1)
        obs_inflation = p_sym[0]

        # Discrete unicycle kinematics
        x_next = ca.vertcat(
            x_sym[0] + self.v * ca.cos(x_sym[2]) * self.dt,
            x_sym[1] + self.v * ca.sin(x_sym[2]) * self.dt,
            x_sym[2] + u_sym[0] * self.dt,
        )

        model.x             = x_sym
        model.u             = u_sym
        model.p             = p_sym
        model.disc_dyn_expr = x_next
        model.f_expl_expr   = ca.SX.zeros(3)  # unused for DISCRETE

        # Nonlinear obstacle constraints:  dist - (r + inflation) >= 0
        # Using the Euclidean distance (not squared) gives a unit-norm Jacobian
        # [(x-cx)/dist, (y-cy)/dist, 0], which keeps the QP well-conditioned
        # even when the warm-start trajectory passes close to an obstacle.
        # A small epsilon under the sqrt prevents a zero gradient at dist=0.
        if n_obs > 0:
            # Path stages have TWO sets of constraints per obstacle:
            #   (1) raw r — hard collision avoidance (high slack penalty),
            #   (2) D     — danger region; cheap slack so intermediates *may*
            #               enter D when needed but pay a small price.
            # Terminal stage has ONE constraint per obstacle: D, HARD (no slack)
            # so the persistent-feasibility loiter is guaranteed.
            h_list_path_r = []   # raw r soft (path)
            h_list_path_D = []   # D soft, cheap (path)
            h_list_e      = []   # D hard (terminal)
            for obs in self.obstacles:
                cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
                dist = ca.sqrt((x_sym[0] - cx) ** 2 + (x_sym[1] - cy) ** 2 + 1e-6)
                h_list_path_r.append(dist - (r + obs_inflation))
                if self.turn_radius is not None:
                    r_term = float(np.sqrt(r ** 2 + 2 * r * self.turn_radius))
                else:
                    r_term = r
                h_list_path_D.append(dist - (r_term + obs_inflation))
                h_list_e.append(dist - (r_term + obs_inflation))
            # Path: [raw_r constraints..., D constraints...]
            model.con_h_expr   = ca.vertcat(*h_list_path_r, *h_list_path_D)
            model.con_h_expr_e = ca.vertcat(*h_list_e)   # terminal: D only

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.n_horizon

        # ── Cost ──────────────────────────────────────────────────────
        # NONLINEAR_LS: residual y = omega, reference = omega_nom (set via yref).
        # Gauss-Newton hessian is always PSD → HPIPM always succeeds.
        model.cost_y_expr   = u_sym               # shape (1,)
        model.cost_y_expr_e = ca.SX.zeros(1)      # no terminal cost residual
        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W           = np.array([[1.0]])   # weight on (omega - yref)^2
        ocp.cost.W_e         = np.array([[0.0]])   # zero terminal weight
        ocp.cost.yref        = np.array([0.0])     # placeholder; overwritten each solve
        ocp.cost.yref_e      = np.array([0.0])
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

        # ── Solver options ─────────────────────────────────────────────
        ocp.solver_options.tf                  = self.n_horizon * self.dt
        ocp.solver_options.integrator_type     = 'DISCRETE'
        ocp.solver_options.nlp_solver_type     = 'SQP_RTI'
        ocp.solver_options.qp_solver           = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.print_level         = 0

        # ── Constraints ────────────────────────────────────────────────
        ocp.constraints.lbu   = np.array([-1.0])
        ocp.constraints.ubu   = np.array([ 1.0])
        ocp.constraints.idxbu = np.array([0])

        if n_obs > 0:
            n_path = 2 * n_obs   # raw r + D for each obstacle
            ocp.constraints.lh   = np.zeros(n_path)
            ocp.constraints.uh   = np.full(n_path, 1e15)
            ocp.constraints.lh_e = np.zeros(n_obs)
            ocp.constraints.uh_e = np.full(n_obs, 1e15)

            # PATH stages: both raw-r and D constraints are softened.
            #   raw-r slack expensive (true collision must not happen),
            #   D     slack cheap     (intermediates may enter D when needed).
            # TERMINAL stage: D constraint is HARD (no Jsh_e). The QP solver
            # cannot return a terminal inside D — if it can't satisfy, it
            # reports infeasibility and we treat the result as INFEASIBLE.
            _raw_slack_lin    = 1e3   # raw r violation — collision avoidance
            _raw_slack_quad   = 1e3
            _D_path_slack_lin   = 1e1   # intermediate in D — small cost
            _D_path_slack_quad  = 1e1

            ocp.constraints.Jsh = np.eye(n_path)
            ocp.cost.zl = np.concatenate([
                _raw_slack_lin  * np.ones(n_obs),     # raw r
                _D_path_slack_lin * np.ones(n_obs),   # D
            ])
            ocp.cost.zu = np.zeros(n_path)
            ocp.cost.Zl = np.concatenate([
                _raw_slack_quad  * np.ones(n_obs),
                _D_path_slack_quad * np.ones(n_obs),
            ])
            ocp.cost.Zu = np.zeros(n_path)
            # Terminal: NO Jsh_e / zl_e / Zl_e → hard constraint.

        ocp.constraints.x0      = np.zeros(3)
        ocp.parameter_values    = np.zeros(1)   # p = [obs_inflation]
        ocp.code_export_directory = f'/tmp/{solver_name}_gen'

        solver = AcadosOcpSolver(
            ocp,
            json_file=f'/tmp/{solver_name}_ocp.json',
            verbose=False,
        )
        return solver

    # ------------------------------------------------------------------
    # Interface matching do_mpc MPC object
    # ------------------------------------------------------------------
    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = np.array(val).reshape(-1, 1)

    def _rollout(self, omega_seq):
        """Forward-simulate the unicycle from self._x0 with given omega sequence."""
        x = self._x0.flatten().copy()
        states = [x.copy()]
        for k in range(self.n_horizon):
            u_k = float(omega_seq[k])
            x = np.array([
                x[0] + self.v * np.cos(x[2]) * self.dt,
                x[1] + self.v * np.sin(x[2]) * self.dt,
                x[2] + u_k * self.dt,
            ])
            states.append(x.copy())
        return states

    def _terminal_clears_D(self, states):
        """Check if the rollout's terminal state lies outside D for every obstacle."""
        if self.turn_radius is None or not self.obstacles:
            return True
        tx, ty = states[-1][0], states[-1][1]
        inflation = float(self._obs_inflation_buf[0])
        for obs in self.obstacles:
            cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
            D = float(np.sqrt(r ** 2 + 2 * r * self.turn_radius)) + inflation
            if (tx - cx) ** 2 + (ty - cy) ** 2 < D * D:
                return False
        return True

    def _heuristic_swerve_omega(self):
        """
        Pick a swerve direction that points the terminal away from the worst
        obstacle. Looks at obstacle bearing relative to current heading; if
        obstacle is to the left of heading we swerve right (omega = -1), and
        vice versa. The 'worst' obstacle is the one whose D-region the NN
        rollout penetrates most deeply.
        """
        nn_states = self._rollout(self._nom_ctrl_buf[:self.n_horizon, 0])
        tx, ty    = nn_states[-1][0], nn_states[-1][1]
        inflation = float(self._obs_inflation_buf[0])
        worst, worst_depth = None, 0.0
        for obs in self.obstacles:
            cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
            D = float(np.sqrt(r ** 2 + 2 * r * self.turn_radius)) + inflation
            d = float(np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2))
            depth = D - d
            if depth > worst_depth:
                worst_depth = depth
                worst = (cx, cy)
        if worst is None:
            return None
        x0 = self._x0.flatten()
        theta = float(x0[2])
        # Bearing from current position to obstacle, expressed in body frame:
        # positive y means obstacle is to the left of current heading.
        dx, dy = worst[0] - float(x0[0]), worst[1] - float(x0[1])
        body_y = -dx * np.sin(theta) + dy * np.cos(theta)
        # Obstacle on left → swerve right (omega = -1); on right → left (+1).
        return -1.0 if body_y > 0.0 else 1.0

    def set_initial_guess(self):
        """
        Warm-start solver by simulating the unicycle forward with whatever
        controls are currently in _nom_ctrl_buf. The caller is responsible for
        setting that buffer to the desired warm-start (NN rollout or constant
        swerve omega) before calling this.
        """
        omega_seq = self._nom_ctrl_buf[:self.n_horizon, 0]
        x = self._x0.flatten().copy()
        for k in range(self.n_horizon + 1):
            self._solver.set(k, 'x', x)
            if k < self.n_horizon:
                u_k = float(omega_seq[k])
                self._solver.set(k, 'u', np.array([u_k]))
                x = np.array([
                    x[0] + self.v * np.cos(x[2]) * self.dt,
                    x[1] + self.v * np.sin(x[2]) * self.dt,
                    x[2] + u_k * self.dt,
                ])
        self._step_count = 0

    def _push_parameters(self):
        """Push obs_inflation and omega_nom (as yref) into the solver."""
        inflation = float(self._obs_inflation_buf[0])
        for k in range(self.n_horizon):
            self._solver.set(k, 'p', np.array([inflation]))
            self._solver.set(k, 'yref', np.array([float(self._nom_ctrl_buf[k])]))
        self._solver.set(self.n_horizon, 'p', np.array([inflation]))
        self._solver.set(self.n_horizon, 'yref', np.array([0.0]))

    # Number of RTI iterations for the first cold-start solve.
    # Subsequent solves are warm-started and need only 1 RTI iteration.
    _RTI_WARMUP_ITERS = 20

    def make_step(self, x0: np.ndarray, n_rti: int = None) -> np.ndarray:
        """
        Solve the OCP from x0 using SQP_RTI.

        On the first call after set_initial_guess() (cold start), runs
        _RTI_WARMUP_ITERS iterations so the solution converges before the
        receding-horizon loop warm-starts naturally.

        Returns
        -------
        np.ndarray, shape (1, 1)
            Optimal omega at stage 0.
        """
        x0_flat = np.array(x0).flatten()

        # Pin initial state
        self._solver.set(0, 'lbx', x0_flat)
        self._solver.set(0, 'ubx', x0_flat)

        # Push parameters and references
        self._push_parameters()

        if n_rti is None:
            n_rti = self._RTI_WARMUP_ITERS if self._step_count == 0 else 1
        for _ in range(n_rti):
            status = self._solver.solve()
        self.last_status = status
        if status not in (0, 2):
            print(f'  [acados] Solver status {status} (step {self._step_count})')

        for k in range(self.n_horizon + 1):
            self._predicted_x[k] = self._solver.get(k, 'x')
        for k in range(self.n_horizon):
            self._predicted_u[k] = self._solver.get(k, 'u')

        self._step_count += 1
        return self._solver.get(0, 'u').reshape(1, 1)

    @property
    def opt_u_seq(self) -> list:
        """
        Optimal input sequence u[0..N-1] from the most recent solve.

        Applying this whole sequence open-loop from x0 reproduces opt_x_num
        exactly, because disc_dyn_expr above is the same explicit-Euler step
        as Unicycle_NL.forward in cl_systems/unicycle.py. That identity is
        what lets a single solve stand in for the re-simulation rollout.
        """
        return [np.array(u).flatten().copy() for u in self._predicted_u]

    @property
    def opt_x_num(self) -> _OptXAccessor:
        """Provides  opt_x_num['_x', k, 0]  access to cached predicted states."""
        return _OptXAccessor(self._predicted_x)
