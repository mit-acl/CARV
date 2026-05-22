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
                 solver_name: str = 'unicycle_acados'):
        # nominal_tracking kept as parameter for call-site compatibility but
        # non-nominal path is removed — always uses NONLINEAR_LS tracking cost.
        self.n_horizon = n_horizon
        self.dt        = dt
        self.v         = v
        self.obstacles = obstacles if obstacles is not None else []

        # Buffers — same attribute names as the do_mpc version
        self._obs_inflation_buf = np.zeros(1)
        self._nom_ctrl_buf      = np.zeros((n_horizon + 1, 1))

        # Internal state
        self._x0          = np.zeros((3, 1))
        self._step_count  = 0
        self._predicted_x = [np.zeros(3) for _ in range(n_horizon + 1)]

        # Delete stale cached files so the new formulation is compiled fresh
        import shutil
        for path in [f'/tmp/{solver_name}_gen', f'/tmp/{solver_name}_ocp.json']:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)

        self._solver = self._build_solver(solver_name)

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
            h_list = []
            for obs in self.obstacles:
                cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
                dist = ca.sqrt((x_sym[0] - cx) ** 2 + (x_sym[1] - cy) ** 2 + 1e-6)
                h_list.append(dist - (r + obs_inflation))
            h_expr = ca.vertcat(*h_list)
            model.con_h_expr   = h_expr   # stages 0 … N-1
            model.con_h_expr_e = h_expr   # stage N (terminal)

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
            ocp.constraints.lh   = np.zeros(n_obs)
            ocp.constraints.uh   = np.full(n_obs, 1e15)
            ocp.constraints.lh_e = np.zeros(n_obs)
            ocp.constraints.uh_e = np.full(n_obs, 1e15)

            # Soft constraints: slacks on lower bound of h (obstacle avoidance).
            # The QP is always feasible; slack is penalised heavily in cost.
            # Jsh selects which h-constraints are softened (all of them).
            _slack_penalty_lin  = 1e3   # linear penalty on slack
            _slack_penalty_quad = 1e3   # quadratic penalty on slack
            ocp.constraints.Jsh   = np.eye(n_obs)
            ocp.constraints.Jsh_e = np.eye(n_obs)
            # zl / zu: linear slack costs (lower / upper slack)
            ocp.cost.zl   = _slack_penalty_lin  * np.ones(n_obs)
            ocp.cost.zu   = np.zeros(n_obs)        # no upper slack penalty
            ocp.cost.Zl   = _slack_penalty_quad * np.ones(n_obs)
            ocp.cost.Zu   = np.zeros(n_obs)
            ocp.cost.zl_e = _slack_penalty_lin  * np.ones(n_obs)
            ocp.cost.zu_e = np.zeros(n_obs)
            ocp.cost.Zl_e = _slack_penalty_quad * np.ones(n_obs)
            ocp.cost.Zu_e = np.zeros(n_obs)

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

    def set_initial_guess(self):
        """
        Warm-start solver by simulating the unicycle forward with the current
        nominal controls (_nom_ctrl_buf).  This keeps the warm-start trajectory
        on the nominal path rather than stationary at x0, which avoids the QP
        starting deep inside an inflated obstacle region.
        """
        x = self._x0.flatten().copy()
        for k in range(self.n_horizon + 1):
            self._solver.set(k, 'x', x)
            if k < self.n_horizon:
                u_k = float(self._nom_ctrl_buf[k])
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

    def make_step(self, x0: np.ndarray) -> np.ndarray:
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

        n_rti = self._RTI_WARMUP_ITERS if self._step_count == 0 else 1
        for _ in range(n_rti):
            status = self._solver.solve()
        if status not in (0, 2):   # 0=success, 2=max_iter/RTI
            print(f'  [acados] Solver status {status} (step {self._step_count})')

        for k in range(self.n_horizon + 1):
            self._predicted_x[k] = self._solver.get(k, 'x')

        self._step_count += 1
        return self._solver.get(0, 'u').reshape(1, 1)

    @property
    def opt_x_num(self) -> _OptXAccessor:
        """Provides  opt_x_num['_x', k, 0]  access to cached predicted states."""
        return _OptXAccessor(self._predicted_x)
