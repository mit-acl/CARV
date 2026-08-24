"""
Acados-based MPC for double-integrator dynamics, with a terminal set
that lives inside the parabolic max-brake-invariant safe region
(analogous to alg12's S-region for the unicycle).

Invariant safe set
------------------
For position constraint p > 0 with control bound |u| <= 1, the
max-brake-invariant set is
    S_par = { (p, v) : v > -1  AND  p > 0.5 * max(0, -v)^2 }
i.e. states from which applying u = +1 forever keeps p > 0. The control
u = +1 EXACTLY preserves the margin m = p - 0.5*max(0,-v)^2; any softer
control bleeds margin away and bounded control can never recover it.

Terminal constraint
-------------------
We enforce a LINEAR sufficient condition for membership in S_par at
the terminal stage:
    v_corner >= 0          (terminal velocity non-negative)
    p_corner >= buffer     (terminal position above buffer)
where (p_corner, v_corner) = (p_N - hw_p, v_N - hw_v) is the worst
corner of the terminal RSOA box.

Safety
~~~~~~
v_corner >= 0 ⇒ max(0, -v_corner) = 0 ⇒ margin at terminal
    m_N = p_corner - 0 = p_corner >= buffer > 0,
so the terminal state is control-invariant: holding u = 0 keeps v >= 0
and p non-decreasing forever.

NOTE (corrected): an earlier version of this file claimed the terminal
constraint alone was sufficient because "margin is non-increasing under
|u| <= 1", and therefore omitted per-stage constraints. That argument is
WRONG. The margin m = p - 0.5*max(0,-v)^2 is non-increasing only while
v < 0; once v crosses zero, m = p and it GROWS. The trajectory minimum
sits at the v = 0 crossing, which the terminal constraint does not bound.
Counterexample from (p, v) = (0.6, -1.0), a state strictly inside S_par
(margin +0.1), with |u| <= 1 and N = 12, dt = 0.2:

    u = [0, 1, .5, .5, 1, .5, 1, 1, 1, 1, 1, 1]
    p:  +0.60 +0.40 +0.22 +0.07 -0.06 -0.16 -0.23 -0.27 -0.27 ... +0.13
                                 ^^^^^ p >= 0 violated       terminal h OK

A second sequence drives v to -1.10 against vel_min = -1.0. Per-stage
constraints are therefore REQUIRED and are now enforced below.

Why linear rather than the smoothed parabolic constraint:
the smoothed `max(0, -v)` has a near-zero gradient around v = 0, which
makes the resulting QP rank-deficient and causes HPIPM to bail at
status 3.  A linear sufficient condition is a strict subset of S_par
but is well-conditioned for HPIPM and large enough that PSF backups
remain feasible from any reasonable RSOA in S_par.

RSOA inflation
--------------
The safety filter wraps each MPC point in a constant-size box of half
widths (hw_p, hw_v). Two parameters are passed at solve time:
    p[0] = hw_p (position half-width)
    p[1] = hw_v (velocity half-width)

Cost
----
NONLINEAR_LS Gauss-Newton tracking on (u - u_nom)^2, same as the
unicycle backend.
"""

import os
import ctypes
import shutil
import numpy as np
import casadi as ca

# ---------------------------------------------------------------------------
# Pre-load acados shared libraries
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


class _OptXAccessor:
    """Mimic do_mpc's opt_x_num['_x', k, 0] indexing."""
    def __init__(self, predicted_x):
        self._x = predicted_x
    def __getitem__(self, key):
        _, k, _ = key
        return self._x[k]


class AcadosDIMPC:
    """
    Acados OCP solver for double-integrator MPC with parabolic
    max-brake-invariant terminal constraint.

    Parameters
    ----------
    dt : float
        Discretisation time step (seconds).
    n_horizon : int
        MPC prediction horizon (steps).
    pos_min, vel_min : float
        Raw half-plane bounds (defaults: 0.0, -1.0).
    buffer : float
        Safety margin added to the terminal parabola constraint.
    """

    def __init__(self,
                 dt: float = 0.2,
                 n_horizon: int = 12,
                 pos_min: float = 0.0,
                 vel_min: float = -1.0,
                 buffer: float = 0.05,
                 solver_name: str = 'di_acados'):
        self.dt        = dt
        self.n_horizon = n_horizon
        self.pos_min   = pos_min
        self.vel_min   = vel_min
        self.buffer    = buffer

        # Buffers — same names as unicycle backend.
        # _obs_inflation_buf is 2-element: [hw_p, hw_v].
        self._obs_inflation_buf = np.zeros(2)
        self._nom_ctrl_buf      = np.zeros((n_horizon + 1, 1))

        self._x0          = np.zeros((2, 1))
        self._step_count  = 0
        self._predicted_x = [np.zeros(2) for _ in range(n_horizon + 1)]

        for path in [f'/tmp/{solver_name}_gen', f'/tmp/{solver_name}_ocp.json']:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

        self._A = np.array([[1.0, dt], [0.0, 1.0]])
        self._B = np.array([[0.5 * dt * dt], [dt]])

        self._solver = self._build_solver(solver_name)

    # ------------------------------------------------------------------
    def _build_solver(self, solver_name: str) -> AcadosOcpSolver:
        model = AcadosModel()
        model.name = solver_name

        x_sym = ca.SX.sym('x', 2)            # [position, velocity]
        u_sym = ca.SX.sym('u', 1)            # [acceleration]
        p_sym = ca.SX.sym('p', 2)            # [hw_p, hw_v]
        hw_p = p_sym[0]
        hw_v = p_sym[1]

        x_next = ca.vertcat(
            x_sym[0] + self.dt * x_sym[1] + 0.5 * self.dt * self.dt * u_sym[0],
            x_sym[1] + self.dt * u_sym[0],
        )

        model.x             = x_sym
        model.u             = u_sym
        model.p             = p_sym
        model.disc_dyn_expr = x_next
        model.f_expl_expr   = ca.SX.zeros(2)

        # ── Terminal constraints (only constraints besides |u|<=1) ────────
        # Worst-corner of RSOA box at the terminal stage:
        #   p_corner = x[0] - hw_p
        #   v_corner = x[1] - hw_v
        #
        # Use a LINEAR sufficient condition that lives strictly inside the
        # parabolic max-brake-invariant safe set:
        #     v_corner >= 0          (terminal velocity non-negative)
        #     p_corner >= buffer     (terminal position above buffer)
        #
        # Safety argument:
        #   v_corner >= 0  ⇒  max(0, -v_corner) = 0
        #                  ⇒  parabolic margin  p_corner - 0.5·max(0,-v)²
        #                     = p_corner ≥ buffer > 0
        #   So terminal sits in the parabolic invariant set with margin ≥ buffer.
        #   Margin is non-increasing under |u|≤1, so every earlier stage also
        #   has margin ≥ buffer ⇒ every stage above the raw safety boundary.
        #
        # This is a strict subset of the smooth parabolic terminal we tried
        # first, but the constraint is linear in (x, p), so HPIPM no longer
        # has to deal with a degenerate-Jacobian smoothed-max around v≈0
        # (where it was bailing with status 3).
        p_corner = x_sym[0] - hw_p
        v_corner = x_sym[1] - hw_v
        h_v_terminal = v_corner                                 # >= 0
        h_p_terminal = p_corner - (self.pos_min + self.buffer)  # >= 0
        model.con_h_expr_e = ca.vertcat(h_v_terminal, h_p_terminal)

        # Per-stage h: the RAW state constraints on every path node.
        # Required — the terminal constraint alone does NOT imply them
        # (see the corrected safety note in the module docstring).
        # This is the split-terminal contract: path nodes clear the RAW
        # constraint, only the terminal must reach the invariant set.
        model.con_h_expr = ca.vertcat(
            p_corner - self.pos_min,    # >= 0
            v_corner - self.vel_min,    # >= 0
        )

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.n_horizon

        # ── Cost ─────────────────────────────────────────────────────────
        model.cost_y_expr   = u_sym
        model.cost_y_expr_e = ca.SX.zeros(1)
        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W           = np.array([[1.0]])
        ocp.cost.W_e         = np.array([[0.0]])
        ocp.cost.yref        = np.array([0.0])
        ocp.cost.yref_e      = np.array([0.0])
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'

        # ── Solver options ───────────────────────────────────────────────
        ocp.solver_options.tf              = self.n_horizon * self.dt
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.print_level     = 0

        # ── Control bounds ───────────────────────────────────────────────
        ocp.constraints.lbu   = np.array([-1.0])
        ocp.constraints.ubu   = np.array([ 1.0])
        ocp.constraints.idxbu = np.array([0])

        # ── Terminal h bounds (linear: v >= 0, p >= buffer) ──────────────
        n_h_e = 2
        ocp.constraints.lh_e = np.zeros(n_h_e)
        ocp.constraints.uh_e = np.full(n_h_e, 1e15)

        # ── Per-stage h bounds (raw constraints: p >= pos_min, v >= vel_min)
        n_h = 2
        ocp.constraints.lh = np.zeros(n_h)
        ocp.constraints.uh = np.full(n_h, 1e15)

        # Soft slack on BOTH stage and terminal h so the QP is always feasible.
        # Slacks only affect which plan acados proposes; the safety verdict is
        # made independently by the filter's own node check, so a slacked
        # solution can never be certified unsafe-but-accepted.
        slack_lin  = 1e3
        slack_quad = 1e3
        ocp.constraints.Jsh_e = np.eye(n_h_e)
        ocp.cost.zl_e = slack_lin  * np.ones(n_h_e)
        ocp.cost.zu_e = np.zeros(n_h_e)
        ocp.cost.Zl_e = slack_quad * np.ones(n_h_e)
        ocp.cost.Zu_e = np.zeros(n_h_e)

        ocp.constraints.Jsh = np.eye(n_h)
        ocp.cost.zl = slack_lin  * np.ones(n_h)
        ocp.cost.zu = np.zeros(n_h)
        ocp.cost.Zl = slack_quad * np.ones(n_h)
        ocp.cost.Zu = np.zeros(n_h)

        ocp.constraints.x0   = np.zeros(2)
        ocp.parameter_values = np.zeros(2)
        ocp.code_export_directory = f'/tmp/{solver_name}_gen'

        return AcadosOcpSolver(
            ocp,
            json_file=f'/tmp/{solver_name}_ocp.json',
            verbose=False,
        )

    # ------------------------------------------------------------------
    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = np.array(val).reshape(-1, 1)

    def set_initial_guess(self):
        x = self._x0.flatten().copy()
        for k in range(self.n_horizon + 1):
            self._solver.set(k, 'x', x)
            if k < self.n_horizon:
                u_k = float(self._nom_ctrl_buf[k])
                self._solver.set(k, 'u', np.array([u_k]))
                x = (self._A @ x.reshape(-1, 1) + self._B * u_k).flatten()
        self._step_count = 0

    def _push_parameters(self):
        p_vec = np.array([float(self._obs_inflation_buf[0]),
                          float(self._obs_inflation_buf[1])])
        for k in range(self.n_horizon):
            self._solver.set(k, 'p', p_vec)
            self._solver.set(k, 'yref', np.array([float(self._nom_ctrl_buf[k])]))
        self._solver.set(self.n_horizon, 'p', p_vec)
        self._solver.set(self.n_horizon, 'yref', np.array([0.0]))

    # 20 cold-start RTI iters matches the unicycle backend. With the linear
    # terminal the QP converges in 1–2 inner iterations; this is plenty.
    _RTI_WARMUP_ITERS = 20

    def make_step(self, x0: np.ndarray) -> np.ndarray:
        x0_flat = np.array(x0).flatten()

        self._solver.set(0, 'lbx', x0_flat)
        self._solver.set(0, 'ubx', x0_flat)

        self._push_parameters()

        if self._step_count == 0:
            # ── Feasibility pre-solve ────────────────────────────────────
            # Zero out the tracking cost so the only active objective is
            # the terminal slack penalty.  One RTI iteration drives the
            # trajectory into the feasible terminal region before the
            # normal warmup begins, preventing HPIPM NaN/Inf (status 3)
            # that occurs when the cold-start guess violates the terminal
            # constraint and the tracking cost fights it.
            W_zero = np.zeros((1, 1))
            for k in range(self.n_horizon):
                self._solver.cost_set(k, 'W', W_zero)
            self._solver.solve()
            W_one = np.array([[1.0]])
            for k in range(self.n_horizon):
                self._solver.cost_set(k, 'W', W_one)
            # ── Normal warmup ────────────────────────────────────────────
            self._push_parameters()
            for _ in range(self._RTI_WARMUP_ITERS):
                status = self._solver.solve()
        else:
            status = self._solver.solve()

        if status not in (0, 2):
            print(f'  [acados-di] Solver status {status} (step {self._step_count})')

        for k in range(self.n_horizon + 1):
            self._predicted_x[k] = self._solver.get(k, 'x')

        self._step_count += 1
        return self._solver.get(0, 'u').reshape(1, 1)

    @property
    def opt_x_num(self) -> _OptXAccessor:
        return _OptXAccessor(self._predicted_x)
