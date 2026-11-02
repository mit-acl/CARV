"""
alg29 -- generic (textbook) predictive safety filter baselines with explicit
computation latency, on the unicycle.

Purpose
-------
Reach_PSF / alg28 decide the control for step t from a backup that was built
for the state AFTER that step (mpc_buffer[t+1]) out of a reachable SET, so the
decision itself is a lookup. A textbook PSF (Wabersich & Zeilinger; model
predictive shielding) instead solves its OCP from the measured POINT x_t and
applies the result "at t". A real solve takes delta seconds, and the plant does
not wait. This file makes delta explicit and measures what it costs.

Every arm uses the same pieces as alg28, so only the filter logic differs:
  scenes      ReachabilityTester(seed) initial state + the runner's obstacles
  NN          the same controller; it reads the estimate (centre of the
              set-membership box), as in alg28 mode J
  noise       TTTCARV_NOISE=uniform, |v| <= VMAX_SIGMA*0.05 = 0.15 (alg28), or
              none (perfect state; the estimate IS the true state)
  estimator   set-membership box S_t = F(S_{t-1}, u) INTERSECT (z_t +- vmax)
  OCP         the alg28 acados OCP: N = 12, path nodes clear raw r, terminal
              node clears D = sqrt(r^2 + 2rR), |omega| <= 1, one-shot solve
              re-simulated through the plant, refine 2, RTI 10
  acceptance  _rollout_verdict (same function alg28 uses)

Timing model ("continuous time" on the project's plant)
-------------------------------------------------------
The plant is Unicycle_NL: p' = p + dt*v*(cos th, sin th), th' = th + int(omega).
Heading integrates the turn rate exactly, so a command that switches inside a
step is modelled exactly: th_{k+1} = th_k + sum_i d_i * omega_i over the
piecewise-constant pieces of step k. With every switch on a tick this reduces
to the discrete model exactly, which is what all the OCPs assume.

A solve started at tick k (time k*dt, fresh measurement z_k) returns at
k*dt + delta. Until then the plant runs the HOLD command:
  nn      NN(x_hat) -- the NN is the live controller, the filter overrides
          once it knows (optimistic; the scenario "we find out the backup failed
          after we already moved")
  prev    the last applied command (zero-order hold)
  backup  the cached backup plan's control for that step (conservative)
If the solve fails, the filter falls back on the cached plan (the last accepted
one), indexed by the current step. When the cache runs out it loiters at max
turn, as alg22/alg28 do.

Arms (TTTCARV_GP_MODE)
  G0  textbook PSF, delta = 0 (computation is free). Sanity/upper bound.
  G1  textbook PSF with latency: solves from x_hat_k, applies u0 at k*dt+delta
      as if no time had passed. Fallback = cached plan at the current step.
  G2  delay-compensated PSF: holds the cached backup during the solve (known
      controls), predicts x_hat forward through them, plans from the predicted
      state and converts the first control so the step's heading increment is
      what the plan assumed. Nominally exact for any delta; the point
      prediction is what breaks under noise.
  G3  G2 + constraint tightening by the estimate box: the OCP inflates the
      obstacles by the box's xy diagonal and every plan node carries the
      start box (constant size) -- the usual "robust" quick fix, and the
      acceptance test alg22 used before alg28's sound boxes.
  R1  robust (tube) PSF from the literature, with G1's naive timing. Output
      feedback: the start set is the set-membership box S; the tube is the
      open-loop Lipschitz tube of the unicycle (same controls, so the heading
      error stays <= e_th and the position error grows by v*dt*e_th per step):
          e_k = e_p0 + k * v * dt * e_th,  e_p0 = xy half-diagonal of S.
      The OCP tightens stage k by e_k (path raw r + e_k, terminal D + e_N;
      D holds for every heading, so heading error needs nothing more), and the
      acceptance test runs on the tube boxes. Sound when the input lands at the
      tick it was planned for -- the standard zero-computation-time assumption.
  R2  the same tube PSF, delay-compensated as in G2: the start box is S pushed
      through the held controls to the tick the plan starts, so the tube covers
      the latency too.

Knobs
  TTTCARV_GP_MODE      G0 | G1 | G2 | G3                      (default G1)
  TTTCARV_GP_DELAY     seconds, or 'measured' (wall clock)    (default 0.1)
  TTTCARV_GP_ASSUMED   G2/G3 with 'measured': the latency the filter plans
                       for; a solve that overruns it is a deadline miss and is
                       treated as a failed solve                (default 0.2)
  TTTCARV_GP_HOLD      nn | prev | backup  (default G1 nn, G2/G3 backup)
  TTTCARV_GP_MULTISTART 1 (default) = retry a rejected solve with alg28's
                       multi-start (NN, +1, -1, 0 warm starts); 0 = single
  TTTCARV_GP_SHRINK    1 = Algorithm 2 line 6: when the full-horizon solve is
                       rejected, re-solve at the horizon left before the
                       active plan's terminal deadline and certify nodes
                       1..m. Unicycle only. 0 (default) = fly the cached
                       tail, which is only the feasible POINT that argument
                       exhibits. Costs a third solve pass in the timestep,
                       which measured-delay arms pay for in wall clock.
  TTTCARV_NOISE        none | uniform | trunc | gauss         (default uniform)
                       trunc: N(0, (vmax/r)^2) conditioned on |v| <= vmax --
                       the same bound as uniform, peaked instead of flat, and
                       the pessimistic case for set-membership (consecutive
                       measurement boxes nearly coincide, so they cut less).
                       gauss: unbounded, so the assumed bound only holds with
                       probability; truth_out counts the breaks.
  TTTCARV_TRUNC_RATIO  r for trunc/gauss: internal sigma = vmax/r (default 3,
                       which reproduces sigma = 0.05 at vmax = 0.15)
  TTTCARV_GP_ALIGN     1 = G2/G3/R2 apply the new plan at the next whole tick
                       after the solve (m = ceil(delta/dt), no fractional
                       step), so no converted first input exceeds |omega|
                       <= U_MAX. Costs up to one step of extra latency. (default 0)
  TTTCARV_GP_TRACE     path: dump a per-step JSON trace of this trial

test() matches run_trials_random_obs.py; extra stats go to LAST_RUN_STATS.
"""

import json
import os
import time

import numpy as np

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from mpc_safety_filter_acados import make_mpc_safety_filter_acados
from Reach_PSF_FINAL_NL import _rollout_verdict
import mpc_safety_filter_acados as _msfa

MODE = os.environ.get('TTTCARV_GP_MODE', 'G1').strip().upper()
if MODE not in ('G0', 'G1', 'G2', 'G3', 'R1', 'R2'):
    raise ValueError(f"TTTCARV_GP_MODE must be G0..G3, R1 or R2, got {MODE!r}")
ROBUST = MODE in ('R1', 'R2')
NAIVE = MODE in ('G0', 'G1', 'R1')       # G1 timing: plan from x_hat_k, applied late
_DELAY_RAW = os.environ.get('TTTCARV_GP_DELAY', '0.1').strip().lower()
MEASURED = _DELAY_RAW == 'measured'
DELAY = 0.0 if (MODE == 'G0' or MEASURED) else float(_DELAY_RAW)
ASSUMED = float(os.environ.get('TTTCARV_GP_ASSUMED', '0.2'))
HOLD = os.environ.get('TTTCARV_GP_HOLD',
                      'nn' if MODE in ('G1', 'R1') else 'backup').strip().lower()
ALIGN = os.environ.get('TTTCARV_GP_ALIGN', '0') != '0'
if HOLD not in ('nn', 'prev', 'backup'):
    raise ValueError(f"TTTCARV_GP_HOLD must be nn, prev or backup, got {HOLD!r}")
MULTISTART = os.environ.get('TTTCARV_GP_MULTISTART', '1') != '0'
SHRINK = os.environ.get('TTTCARV_GP_SHRINK', '0') != '0'
# Minimal-intervention probe (TTTCARV_GP_PIN=1). Before the normal solve, try
# to certify a plan whose FIRST input is pinned to the NN's command at the
# solve state; if the acceptance test passes, that plan is used and the NN's
# command is applied unchanged. This is what (6a) with a converged solver
# returns whenever the NN's input admits a safe continuation, so it gives the
# baseline its best-case intervention rate independent of our cost design.
PIN = os.environ.get('TTTCARV_GP_PIN', '0') != '0'
# TTTCARV_GP_PIN_RAW=1 (alg43): pin stage 0 to the RAW nn(ctr), not clip(nn, +-U_MAX).
# The plant has no input limit and ours passes the unsaturated NN through, so
# this is the matched action space. Stages 1..N keep the +-U_MAX box (the
# terminal loiter and D only need the filter's own turn capability), and the
# |omega| <= U_MAX audit is applied to those stages only.
PIN_RAW = os.environ.get('TTTCARV_GP_PIN_RAW', '0') != '0'
# TTTCARV_GP_PRESTART=1 (alg43): one zero-delay solve from x_hat_0 BEFORE motion, landing at
# k=0, then the normal pipeline. Without it G2/G3/R2 have no plan at k=0 and fly the loiter,
# while ours builds its t=0 backup before acting (alg28 P1 at current_timestep 0).
PRESTART = os.environ.get('TTTCARV_GP_PRESTART', '0') != '0'
PIN_TOL = float(os.environ.get('TTTCARV_GP_PIN_TOL', '1e-6'))  # stage-0 match; flown plan is re-simulated, so any tol is sound
# TTTCARV_GP_FIRSTSTEP=1: the textbook PSF objective |u_0 - u_NN|^2. Stages
# 1..N-1 keep weight TTTCARV_GP_TAILW (default 1e-3), not 0: with Gauss-Newton
# and no LM term a zero weight makes those Hessian blocks singular. 0 (default)
# = whole-horizon tracking, the cost alg28 uses.
FIRSTSTEP = os.environ.get('TTTCARV_GP_FIRSTSTEP', '0') != '0'
TAILW = float(os.environ.get('TTTCARV_GP_TAILW', '1e-3'))
# R modes put the tube's growth on path stages only; the terminal keeps the
# start radius e_0. The terminal constraint is HARD, so growth there is an
# infeasibility cliff (alg28's FAN_TERMINAL, same reason). Measured null for
# R1/R2 (~/runs/alg39_pinfan): the pin succeeds on ~96% of steps. Kept so both
# sides solve the same OCP. TTTCARV_GP_FAN_TERMINAL=1 = tube radius e_k on every
# stage (rev 7 and earlier).
FAN_TERMINAL = os.environ.get('TTTCARV_GP_FAN_TERMINAL', '0') != '0'
NOISE = os.environ.get('TTTCARV_NOISE', 'uniform').strip().lower()
if NOISE not in ('none', 'uniform', 'trunc', 'gauss'):
    raise ValueError(
        f"alg29 supports TTTCARV_NOISE none|uniform|trunc|gauss, got {NOISE!r}")
VMAX_SIGMA = float(os.environ.get('TTTCARV_VMAX_SIGMA', '3.0'))
TRUNC_RATIO = float(os.environ.get('TTTCARV_TRUNC_RATIO', '3.0'))
if TRUNC_RATIO <= 0:
    raise ValueError(f"TTTCARV_TRUNC_RATIO must be > 0, got {TRUNC_RATIO!r}")

MAX_TIME = 60
N_HORIZON = 12
U_MAX = 1.0          # OCP input box; D assumes it
EPS = 1e-9
# A step counts as filtered when the applied average rate differs from the NN's
# by more than this (RTI iterates are not exactly the NN even when unconstrained).
FILTER_TOL = 0.05

LAST_RUN_STATS: dict = {}


# ── geometry helpers ──────────────────────────────────────────────────────────

def _cos_range(lo, hi):
    if hi - lo >= 2.0 * np.pi:
        return -1.0, 1.0
    vals = [float(np.cos(lo)), float(np.cos(hi))]
    if 2.0 * np.pi * np.ceil(lo / (2.0 * np.pi)) <= hi:
        vals.append(1.0)
    if np.pi + 2.0 * np.pi * np.ceil((lo - np.pi) / (2.0 * np.pi)) <= hi:
        vals.append(-1.0)
    return min(vals), max(vals)


def _interval_step(b, dth, dt, v):
    """Interval enclosure of one plant step whose heading increment is dth."""
    b = np.asarray(b, dtype=float)
    c_lo, c_hi = _cos_range(b[2, 0], b[2, 1])
    s_lo, s_hi = _cos_range(b[2, 0] - np.pi / 2.0, b[2, 1] - np.pi / 2.0)
    out = np.empty_like(b)
    out[0] = [b[0, 0] + dt * v * c_lo, b[0, 1] + dt * v * c_hi]
    out[1] = [b[1, 0] + dt * v * s_lo, b[1, 1] + dt * v * s_hi]
    out[2] = [b[2, 0] + dth, b[2, 1] + dth]
    return out


def _plant_step(x, dth, dt, v):
    return np.array([x[0] + dt * v * np.cos(x[2]),
                     x[1] + dt * v * np.sin(x[2]),
                     x[2] + dth])


def _seg_hits(p0, p1, obstacles):
    """True if the straight segment p0->p1 enters any obstacle (the plant moves
    in a straight line within a step)."""
    d = p1 - p0
    L2 = float(d @ d)
    for o in obstacles:
        c = np.asarray(o[:2], dtype=float)
        s = 0.0 if L2 == 0 else float(np.clip((c - p0) @ d / L2, 0.0, 1.0))
        if np.hypot(*(p0 + s * d - c)) <= o[2]:
            return True
    return False


def _draw_noise(rng, vmax, size):
    """Measurement noise, matching alg28_sound._BoundedRNG so paired arms see
    the same law:

      uniform  U[-vmax, vmax]                      (the certificate's bound)
      trunc    N(0, (vmax/TRUNC_RATIO)^2) conditioned on |v| <= vmax, by
               inverse CDF -- the SAME hard bound, only the shape inside it
               changes. One rng.uniform draw of the same shape as 'uniform',
               so the two arms stay seed-aligned (identical scenes).
      gauss    N(0, (vmax/TRUNC_RATIO)^2) unbounded: the bound the filters
               assume now holds only with probability, and truth_out counts
               the draws that break it.
    """
    if NOISE == 'uniform':
        return rng.uniform(-vmax, vmax, size=size)
    sig = vmax / TRUNC_RATIO
    if NOISE == 'gauss':
        return rng.normal(0.0, sig, size=size)
    from scipy.special import ndtr, ndtri
    a, b = ndtr(-TRUNC_RATIO), ndtr(TRUNC_RATIO)
    v = sig * ndtri(rng.uniform(a, b, size=size))
    return np.clip(v, -vmax, vmax)   # ndtri can hit +-inf in float at the ends


def _pt_hits(p, obstacles):
    return any(np.hypot(p[0] - o[0], p[1] - o[1]) <= o[2] for o in obstacles)


def _install_stage_inflation(mpc):
    """Per-stage obstacle tightening on this solver instance: stage k gets
    the usual scalar inflation plus mpc._stage_infl[k] (None = off)."""
    mpc._stage_infl = None
    base_push = mpc._push_parameters

    def push():
        base_push()
        if mpc._stage_infl is not None:
            base = float(mpc._obs_inflation_buf[0])
            for k in range(mpc.n_horizon + 1):
                kk = k if (FAN_TERMINAL or k < mpc.n_horizon) else 0
                mpc._solver.set(k, 'p', np.array([base + float(mpc._stage_infl[kk])]))
    mpc._push_parameters = push


# Slack weights of the path constraints in utils/unicycle_mpc_acados.py: raw r
# is expensive, D is cheap so a path node may cut through D. Mirrored here
# because acados has no cost_get.
_RAW_SLACK = 1e3
_D_SLACK = 1e1
_D_SLACK_HARD = 1e7


def _set_D_slack(mpc, m):
    """Price the path-D slack at _D_SLACK_HARD from stage m on (m=None: base).

    This is how the shrinking-horizon re-solve of Algorithm 2 is posed on a
    solver whose horizon is fixed at build time. The paper's shorter problem
    lands in S_t at stage m and simply stops; here stages m..N are asked to
    STAY outside D instead, which by control invariance of S_t is feasible
    whenever landing at m is (the max-turn loiter is the witness), so the two
    have the same feasible set. Only the acceptance test certifies, and it is
    run on nodes 1..m alone, so the padding steers the solver and nothing more.
    """
    n = len(mpc.obstacles)
    if n == 0:
        return
    for k in range(mpc.n_horizon):
        w = _D_SLACK_HARD if (m is not None and k >= m) else _D_SLACK
        z = np.concatenate([_RAW_SLACK * np.ones(n), w * np.ones(n)])
        mpc._solver.cost_set(k, 'zl', z)
        mpc._solver.cost_set(k, 'Zl', z)


def _lipschitz_tube(S0, n, v, dt):
    """Open-loop tube radii e_0..e_n of the unicycle from start box S0, and
    the heading half-width. |(cos a, sin a) - (cos b, sin b)| <= |a - b|, and
    identical controls keep the heading error constant."""
    hw = (np.asarray(S0)[:, 1] - np.asarray(S0)[:, 0]) / 2.0
    ep0, e_th = float(np.hypot(hw[0], hw[1])), float(hw[2])
    return [ep0 + k * v * dt * e_th for k in range(n + 1)], e_th


def _tube_boxes(states, tube, e_th):
    return [np.array([[s[0] - e, s[0] + e], [s[1] - e, s[1] + e],
                      [s[2] - e_th, s[2] + e_th]]) for s, e in zip(states, tube)]


# ── the filter ────────────────────────────────────────────────────────────────

class _Plan:
    """An accepted plan: controls[j] is the (effective) rate for step t0 + j;
    states[j] is the state the plan predicts at tick t0 + j."""

    def __init__(self, t0, controls, states, tube=None, e_th=0.0):
        self.t0 = int(t0)
        self.tube = tube          # R modes: position tube radius per node
        self.e_th = float(e_th)
        self.controls = [float(np.asarray(c).flatten()[0]) for c in controls]
        self.states = [np.asarray(s, dtype=float).flatten() for s in states]

    def u(self, step):
        j = step - self.t0
        return self.controls[j] if 0 <= j < len(self.controls) else None

    def state(self, tick):
        j = tick - self.t0
        return self.states[j] if 0 <= j < len(self.states) else None


class _UniSys:
    """Unicycle_NL with circular obstacles (the headline system)."""
    name = 'unicycle'
    max_time = MAX_TIME            # alg28: 60 steps

    def loiter_u(self, x):
        """Max-turn loiter on the side whose DISCRETE loiter (one full turn of
        the Euler plant) keeps the larger clearance from every obstacle.
        D = sqrt(r^2 + 2rR) only makes the away-side continuous circle tangent
        at worst; the Euler plant moves 0.2 m straight before each heading
        change and cuts inside it, so the side is chosen on the plant itself."""
        n = int(np.ceil(2 * np.pi * self.v / (U_MAX * self.v * self.dt))) + 1
        best, best_u = -np.inf, U_MAX
        for u in (U_MAX, -U_MAX):
            xs, clear = np.asarray(x, dtype=float), np.inf
            for _ in range(n):
                xs = _plant_step(xs, self.dt * u, self.dt, self.v)
                clear = min(clear, min((np.hypot(xs[0] - o[0], xs[1] - o[1]) - o[2]
                                        for o in self.obstacles), default=np.inf))
            if clear > best:
                best, best_u = clear, u
        return best_u

    def __init__(self, analyzer, obstacles, seed):
        if obstacles is None:
            obstacles = [np.array([-4.3, 1, 0.5]), np.array([-2, -2, 0.5]),
                         np.array([-1, 0.5, 0.5]), np.array([-3.5, 0, 0.5])]
        self.obstacles = [np.asarray(o, dtype=float) for o in obstacles]
        # Same scene draw as alg28: RandomState(seed) -> initial state.
        self.tester = ReachabilityTester(analyzer, self.obstacles, seed=seed)
        dyn = analyzer.cl_system.dynamics
        self.dt, self.v = float(dyn.dt), float(dyn.vt)
        self.mpc_sf = make_mpc_safety_filter_acados(
            self.tester, obstacles_list=self.obstacles, n_horizon=N_HORIZON,
            verify_input_bounds=True, verify_heading_bounds=True,
            use_safety_radius=True, split_terminal_D=True,
            first_step_cost=FIRSTSTEP, tail_cost_weight=TAILW,
            oneshot=True, oneshot_refine=2, oneshot_verify=False, oneshot_rti=10)

    def step(self, x, pieces):
        return _plant_step(x, float(sum(d * u for d, u in pieces)), self.dt, self.v)

    def box_step(self, S, pieces):
        return _interval_step(S, float(sum(d * u for d, u in pieces)), self.dt, self.v)

    def hits(self, x0, pieces, x1):
        """(node violation at x1, violation anywhere on the step)."""
        return (_pt_hits(x1, self.obstacles),
                _seg_hits(x0[:2], x1[:2], self.obstacles))


class _DISys:
    """DoubleIntegrator with half-plane constraints p > pos_min, v > vel_min.
    The plant is exact LTI, so a command switching inside a step is exact in
    continuous time. Noise: none only (the DI has no set-membership estimator)."""
    name = 'di'
    max_time = 40                  # alg22_di's MAX_TIME

    def __init__(self, analyzer, seed, pos_min, vel_min, buffer=0.05):
        from alg13_di import HalfPlaneConstraints
        from alg22_di import DIMPCSafetyFilterOneShot, TERM_ACCEL
        if NOISE != 'none':
            raise ValueError('alg29 DI runs with TTTCARV_NOISE=none only')
        self.pos_min, self.vel_min = float(pos_min), float(vel_min)
        self._accel = float(TERM_ACCEL)         # alg22_di's terminal controller
        self.tester = ReachabilityTester(analyzer, [], seed=seed)
        self.tester.obstacles = HalfPlaneConstraints(pos_min=pos_min, vel_min=vel_min)
        self.dt = float(analyzer.cl_system.dynamics.dt)
        self.mpc_sf = DIMPCSafetyFilterOneShot(
            self.tester, dt=self.dt, n_horizon=N_HORIZON, pos_min=pos_min,
            vel_min=vel_min, buffer=buffer, u_max=U_MAX, term_accel=TERM_ACCEL,
            oneshot=True, oneshot_refine=2, oneshot_verify=False)

    def loiter_u(self, x):
        return self._accel

    @staticmethod
    def step(x, pieces):
        p, v = float(x[0]), float(x[1])
        for d, u in pieces:
            p, v = p + v * d + 0.5 * u * d * d, v + u * d
        return np.array([p, v])

    def box_step(self, S, pieces):
        raise NotImplementedError('DI runs noise-free')

    def hits(self, x0, pieces, x1):
        bad = lambda p, v: p <= self.pos_min or v <= self.vel_min
        node = bad(x1[0], x1[1])
        p, v, cont = float(x0[0]), float(x0[1]), node
        for d, u in pieces:
            if v < 0.0 < v + u * d:                 # p has its minimum inside
                tau = -v / u
                cont |= (p + v * tau + 0.5 * u * tau * tau) <= self.pos_min
            p, v = p + v * d + 0.5 * u * d * d, v + u * d
            cont |= bad(p, v)
        return node, bool(cont)


def test(seed=None, analyzer=None, obstacles=None, pos_min=0.0, vel_min=-1.0):
    if os.environ.get('TTTCARV_GC_PER_TRIAL', '1') != '0':
        import gc
        gc.collect()
    if analyzer is None:
        if os.environ.get('TTTCARV_GP_SYSTEM', 'unicycle') == 'di':
            analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
        else:
            analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    if seed is None:
        seed = 1401830092
    if type(analyzer.cl_system.dynamics).__name__ == 'DoubleIntegrator':
        sysm = _DISys(analyzer, seed, pos_min, vel_min)
    else:
        sysm = _UniSys(analyzer, obstacles, seed)
    tester, mpc_sf = sysm.tester, sysm.mpc_sf
    if ROBUST:
        if sysm.name != 'unicycle':
            raise ValueError('R1/R2 (tube PSF) are unicycle-only')
        _install_stage_inflation(mpc_sf._mpc)
    obstacles = getattr(sysm, 'obstacles', [])
    rng = tester._rng
    cl = analyzer.cl_system
    dyn, ctrl = cl.dynamics, cl.controller.cpu()
    dt = sysm.dt
    vmax = VMAX_SIGMA * float(tester.measurement_noise_std) if NOISE != 'none' else 0.0

    def nn(xh):
        return float(np.asarray(dyn.control_nn(np.asarray(xh, dtype=float).reshape(1, -1),
                                               ctrl)).flatten()[0])

    h0 = tester.horizons[0]
    X0 = np.asarray(h0.get_tight_bound(), dtype=float)
    x = np.asarray(next(c['real_state'] for c in h0.calculations.values()
                        if 'real_state' in c), dtype=float).flatten()
    # alg28 starts from X0 with no measurement at t = 0; so does this. With
    # noise off the filter knows the state exactly.
    S = X0.copy() if NOISE != 'none' else np.stack([x, x], axis=1)

    st = dict(system=sysm.name, mode=MODE, delay=('measured' if MEASURED else DELAY), hold=HOLD,
              noise=NOISE, vmax=vmax, trunc_ratio=TRUNC_RATIO,
              multistart=int(MULTISTART),
              solves=0, solve_ok=0, solve_fail=0, deadline_miss=0,
              reject_reasons={}, fallback_steps=0, loiter_steps=0,
              cache_empty_steps=0, nn_steps=0, filtered_steps=0,
              activations=0, collided=0, collided_seg=0, first_collision=-1,
              seg_only=0, offplan_max=0.0, offplan_at_collision=-1.0,
              u_new_max=0.0, solve_wall=[], truth_out=0,
              cert_checks=0, cert_invalid=0, cert_invalid_first=-1, cert_invalid_raw=0,
              fb_checks=0, fb_invalid=0,
              # |omega| > U_MAX audit: accepted plans, and filter commands
              # actually flown (landed plan input / converted input / fallback)
              plan_ub_over=0, plan_ub_peak=0.0, land_ub_over=0,
              land_ub_over_conv=0, fb_ub_over=0,
              # R modes: is the true state inside the active plan's tube?
              tube_checks=0, tube_out=0, tube_out_trials=0, tube_e_max=0.0,
              # loiter: steps with no certified side from the estimate box, and
              # steps whose flown side is NOT certified from the TRUE state
              terminal=_msfa.TERMINAL, loiter_uncert=0, loiter_checks=0, loiter_bad=0,
              # Algorithm 2 line 6 (TTTCARV_GP_SHRINK)
              shrink=int(SHRINK), shrink_tries=0, shrink_ok=0, shrink_m={},
              # TTTCARV_GP_PIN, and |applied - NN| summed over all steps
              pin=int(PIN), pin_tries=0, pin_ok=0, dev_sum=0.0,
              firststep=int(FIRSTSTEP), tailw=TAILW, fan_terminal=int(FAN_TERMINAL),
              align=int(ALIGN))
    trace = [] if os.environ.get('TTTCARV_GP_TRACE') else None

    cache = None             # _Plan
    cmd = nn(x if NOISE == 'none' else (S[:, 0] + S[:, 1]) / 2.0)   # command in force
    traj = [x.copy()]
    u_diffs = []
    was_filtered = False
    pending = None           # dict(apply_at, ok, plan, kind, f, t0)
    # alg41 scoring: nn at the centre each solve decided on, and the one landed this step.
    refbox = dict(u=None, land=None)
    ud_clip, ud_pred, ud_predclip = [], [], []
    busy_until = -1.0

    def solve(xs, box, t0, deadline=None):
        """Solve from state xs (box for G3) as a plan for steps t0...
        Returns (ok, plan, reason, wall).

        deadline: the tick by which the ACTIVE cached plan promised to be in
        the terminal set. With SHRINK on, a full-horizon solve that is
        rejected is retried at the horizon left before that deadline
        (Algorithm 2, line 6), which is the problem the paper's recursive
        feasibility argument covers -- the cached tail is a feasible point of
        it. The retry runs inside the same wall-clock window: on a real robot
        it is a second solve in the same timestep."""
        st['solves'] += 1
        tube, e_th = None, 0.0
        if MODE == 'G3':
            b0 = np.asarray(box, dtype=float)
            ctr = (b0[:, 0] + b0[:, 1]) / 2.0
        elif ROBUST:
            ctr = (np.asarray(box)[:, 0] + np.asarray(box)[:, 1]) / 2.0
            b0 = np.stack([ctr, ctr], axis=1)
            tube, e_th = _lipschitz_tube(box, N_HORIZON, sysm.v, dt)
            st['tube_e_max'] = max(st['tube_e_max'], tube[-1])
        else:
            b0 = np.stack([xs, xs], axis=1)
            ctr = np.asarray(xs, dtype=float)
        refbox['u'] = nn(ctr)
        w0 = time.perf_counter()
        ok, reason, plan = False, 'solver_error', None
        if PIN and sysm.name == 'unicycle':
            st['pin_tries'] += 1
            up = float(nn(ctr)) if PIN_RAW else float(np.clip(nn(ctr), -U_MAX, U_MAX))
            ac = mpc_sf._mpc._solver
            vib = mpc_sf.verify_input_bounds
            if PIN_RAW:
                mpc_sf.verify_input_bounds = False
            try:
                ac.constraints_set(0, 'lbu', np.array([up]))
                ac.constraints_set(0, 'ubu', np.array([up]))
                for ms in ([False, True] if MULTISTART else [False]):
                    if ROBUST:
                        mpc_sf._mpc._stage_infl = tube
                    try:
                        tb, pts, controls = mpc_sf._run_mpc_from_bounds(
                            b0, ctr, multi_start=ms)
                    except Exception:                      # noqa: BLE001
                        continue
                    finally:
                        if ROBUST:
                            mpc_sf._mpc._stage_infl = None
                    if controls is None or len(controls) == 0:
                        continue
                    if PIN_RAW:
                        cs = [float(np.asarray(c).flatten()[0]) for c in controls]
                        # stage 0 must be the pin (acados may return an unconverged
                        # iterate); stages 1..N must respect the box.
                        bad = None
                        if not all(np.isfinite(cs)):
                            bad = 'nonfinite'
                        elif abs(cs[0] - up) > PIN_TOL:
                            bad = 'stage0'
                            st['pin_raw_s0_err_max'] = max(st.get('pin_raw_s0_err_max', 0.0), abs(cs[0] - up))
                        elif max((abs(c) for c in cs[1:]), default=0.0) > U_MAX + 1e-6:
                            bad = 'tail'
                        if bad is not None:
                            st['pin_raw_reject'] = st.get('pin_raw_reject', 0) + 1
                            st[f'pin_raw_rej_{bad}'] = st.get(f'pin_raw_rej_{bad}', 0) + 1
                            st['pin_raw_rej_absnn_gt1'] = st.get('pin_raw_rej_absnn_gt1', 0) + int(abs(up) > U_MAX)
                            continue
                    states = [ctr] + [np.asarray(q, dtype=float).flatten() for q in pts]
                    if ROBUST:
                        tb = _tube_boxes(states, tube[:len(states)], e_th)
                    if _rollout_verdict(mpc_sf, tb)[0]:
                        ok, reason = True, 'VALID_PIN'
                        plan = _Plan(t0, controls, states, tube, e_th)
                        st['pin_ok'] += 1
                        break
            finally:
                mpc_sf.verify_input_bounds = vib
                ac.constraints_set(0, 'lbu', np.array([-U_MAX]))
                ac.constraints_set(0, 'ubu', np.array([U_MAX]))
        for ms in (([False, True] if MULTISTART else [False]) if not ok else []):
            if ROBUST:
                mpc_sf._mpc._stage_infl = tube
            try:
                tb, pts, controls = mpc_sf._run_mpc_from_bounds(b0, ctr, multi_start=ms)
            except Exception as e:                         # noqa: BLE001
                reason = f'solver_error({type(e).__name__})'
                continue
            finally:
                if ROBUST:
                    mpc_sf._mpc._stage_infl = None
            if controls is None or len(controls) == 0:
                reason = 'empty_plan'
                continue
            if ROBUST:
                st_nom = [ctr] + [np.asarray(p_, dtype=float).flatten() for p_ in pts]
                tb = _tube_boxes(st_nom, tube[:len(st_nom)], e_th)
            good, kind, node, n = _rollout_verdict(mpc_sf, tb)
            if good:
                pk = max((abs(float(np.asarray(c).flatten()[0])) for c in controls), default=0.0)
                if pk > U_MAX + 1e-6:
                    st['plan_ub_over'] += 1
                    st['plan_ub_peak'] = max(st['plan_ub_peak'], pk)
                states = [ctr] + [np.asarray(p, dtype=float).flatten() for p in pts]
                ok, reason, plan = True, 'VALID', _Plan(t0, controls, states, tube, e_th)
                break
            reason = f'rejected_{kind}'

        # ── Algorithm 2 line 6: re-solve at the horizon left to the deadline ─
        # unicycle only: _set_D_slack mirrors that OCP's slack layout, and
        # the padding argument needs the loiter's control invariance.
        if (not ok) and SHRINK and deadline is not None and sysm.name == 'unicycle':
            m = int(deadline) - int(t0)
            if 1 <= m < N_HORIZON:
                st['shrink_tries'] += 1
                # Stages past m only have to hold the terminal set, so they
                # inherit the terminal tightening rather than a growing tube.
                tube_s = None
                if ROBUST:
                    tube_s = tube[:m + 1] + [tube[m]] * (N_HORIZON - m)
                try:
                    _set_D_slack(mpc_sf._mpc, m)
                    for ms in ([False, True] if MULTISTART else [False]):
                        if ROBUST:
                            mpc_sf._mpc._stage_infl = tube_s
                        try:
                            tb, pts, controls = mpc_sf._run_mpc_from_bounds(
                                b0, ctr, multi_start=ms)
                        except Exception as e:                     # noqa: BLE001
                            reason = f'shrink_solver_error({type(e).__name__})'
                            continue
                        finally:
                            if ROBUST:
                                mpc_sf._mpc._stage_infl = None
                        if controls is None or len(controls) < m:
                            reason = 'shrink_empty_plan'
                            continue
                        states = [ctr] + [np.asarray(q, dtype=float).flatten()
                                          for q in pts]
                        if ROBUST:
                            tb = _tube_boxes(states, tube_s[:len(states)], e_th)
                        # Certify nodes 1..m only: this plan promises the
                        # terminal set at m, exactly as (6f) does at N.
                        good, kind, node, n = _rollout_verdict(mpc_sf, tb[:m + 1])
                        if good:
                            pk = max((abs(float(np.asarray(c).flatten()[0]))
                                      for c in controls[:m]), default=0.0)
                            if pk > U_MAX + 1e-6:
                                st['plan_ub_over'] += 1
                                st['plan_ub_peak'] = max(st['plan_ub_peak'], pk)
                            ok, reason = True, 'VALID_SHRUNK'
                            plan = _Plan(t0, controls[:m], states[:m + 1],
                                         None if tube_s is None else tube_s[:m + 1],
                                         e_th)
                            st['shrink_ok'] += 1
                            st['shrink_m'][m] = st['shrink_m'].get(m, 0) + 1
                            break
                        reason = f'shrink_rejected_{kind}'
                finally:
                    _set_D_slack(mpc_sf._mpc, None)

        wall = time.perf_counter() - w0
        st['solve_wall'].append(wall)
        if ok:
            st['solve_ok'] += 1
        else:
            st['solve_fail'] += 1
            st['reject_reasons'][reason] = st['reject_reasons'].get(reason, 0) + 1
        return ok, plan, reason, wall

    loiter = {'side': None}

    def _deadline():
        """Tick by which the active plan promised to be in the terminal set:
        its terminal node. None when no plan is active."""
        if cache is None or not cache.controls:
            return None
        return cache.t0 + len(cache.controls)


    def fallback_u(step, xs, box=None, commit=True):
        """Cached plan's control for `step`, else the terminal controller
        (max-turn loiter / constant acceleration), chosen at the estimate xs.

        TTTCARV_TERMINAL=loiter: the side is the one whose loiter is certified
        safe forever from the whole estimate box, keeping the side already
        flown when it still qualifies (the loiter stays on one circle). If no
        side is certified, the old clearance heuristic picks one (counted)."""
        u = cache.u(step) if cache is not None else None
        if u is not None:
            return u, 'fallback'
        if _msfa.TERMINAL == 'loiter' and sysm.name == 'unicycle':
            b = box if box is not None else np.stack([xs, xs], axis=1)
            side = mpc_sf._loiter_side(b, prefer=loiter['side'])
            if side is None:
                if commit:
                    st['loiter_uncert'] += 1
                side = loiter['side'] or int(np.sign(sysm.loiter_u(xs)))
            if commit:
                loiter['side'] = side
            return side * U_MAX, 'loiter'
        return sysm.loiter_u(xs), 'loiter'

    def land(k):
        """Apply the pending result at the current instant; returns the kind."""
        nonlocal cache, cmd, pending
        refbox['land'] = pending.get('uref')
        if pending['ok']:
            cache = pending['plan']
            loiter['side'] = None
            u_eff = cache.u(k)
            if not NAIVE and u_eff is not None and pending['f']:
                # Heading increment over step k must be dt*u_eff; f*dt of it was
                # already spent at the hold command now in force. |u_new| can
                # exceed U_MAX (the plant has no input limit); the step's
                # AVERAGE rate is the plan's, so the plan's states are exact.
                f = pending['f']
                u_new = (u_eff - f * cmd) / (1.0 - f)
            else:
                u_new = u_eff if u_eff is not None else cmd
            st['u_new_max'] = max(st['u_new_max'], abs(u_new))
            if u_eff is not None and abs(u_eff) > U_MAX + 1e-6:
                st['land_ub_over'] += 1
            elif abs(u_new) > U_MAX + 1e-6:
                st['land_ub_over_conv'] += 1
            cmd, kind = u_new, 'plan'
        else:
            cmd, kind = fallback_u(k, xh, S)
            st['fallback_steps' if kind == 'fallback' else 'loiter_steps'] += 1
            if kind == 'fallback' and abs(cmd) > U_MAX + 1e-6:
                st['fb_ub_over'] += 1
        pending = None
        return kind

    for k in range(sysm.max_time):
        T = k * dt
        xh = x.copy() if NOISE == 'none' else (S[:, 0] + S[:, 1]) / 2.0
        u_nn = nn(xh)
        refbox['land'] = None
        pieces = []            # (duration, omega) over [T, T+dt)
        events = []
        applied_kind = 'hold'

        if PRESTART and k == 0 and not NAIVE and pending is None:
            ok0, plan0, reason0, _w0 = solve(xh, S, 0, _deadline())
            pending = dict(apply_at=T, ok=ok0, plan=plan0, reason=reason0, f=0.0,
                           uref=refbox['u'])
            events.append(('prestart', reason0))
            st['prestart_ok'] = st.get('prestart_ok', 0) + int(ok0)
        # ── a result due at this tick takes effect first ──────────────────
        if pending is not None and pending['apply_at'] <= T + EPS:
            applied_kind = land(k)

        # ── start a solve at this tick if the solver is idle ──────────────
        if busy_until <= T + EPS:
            if NAIVE:
                ok, plan, reason, wall = solve(xh, S, k, _deadline())
                delta = 0.0 if MODE == 'G0' else (wall if MEASURED else DELAY)
                # G1 executes the plan from the step the result lands in, as if
                # it started there: it does not know time has passed.
                land_step = int(np.floor((T + delta) / dt + EPS))
                if plan is not None:
                    plan.t0 = land_step
                pending = dict(apply_at=T + delta, ok=ok, plan=plan, reason=reason, f=None,
                               uref=refbox['u'])
                busy_until = T + delta
            else:  # G2 / G3 / R2: plan for the tick the new command takes effect
                delta = ASSUMED if MEASURED else DELAY
                if ALIGN:                   # apply on the next whole tick
                    delta = float(np.ceil(delta / dt - EPS)) * dt
                m = int(np.floor(delta / dt + EPS))
                f = delta / dt - m
                # Predict through the controls that will be held until then.
                # G2 knows them exactly for HOLD=backup/prev; for HOLD=nn it
                # predicts the NN at the predicted estimate.
                xp, Sp = xh.copy(), S.copy()
                u_prev = cmd
                for j in range(m):
                    if HOLD == 'backup':
                        uj = fallback_u(k + j, xp, Sp if NOISE != 'none' else None,
                                        commit=False)[0]
                    elif HOLD == 'prev':
                        uj = u_prev
                    else:
                        uj = nn(xp)
                    xp = sysm.step(xp, [(dt, uj)])
                    Sp = sysm.box_step(Sp, [(dt, uj)]) if NOISE != 'none' else Sp
                    u_prev = uj
                ok, plan, reason, wall = solve(xp, Sp, k + m, _deadline())
                if MEASURED and wall > ASSUMED:
                    st['deadline_miss'] += 1
                    ok, plan, reason = False, None, 'deadline_miss'
                pending = dict(apply_at=T + delta, ok=ok, plan=plan, reason=reason, f=f,
                               uref=refbox['u'])
                busy_until = T + delta
            events.append(('solve', reason))
            if pending['apply_at'] <= T + EPS:          # delta = 0
                applied_kind = land(k)

        # ── hold command at the tick while the solver is busy ─────────────
        # A result that landed at this tick is not overridden.
        if applied_kind == 'hold' and pending is not None:
            if HOLD == 'nn':
                cmd = u_nn
            elif HOLD == 'backup':
                cmd, kind = fallback_u(k, xh, S)
                if kind == 'loiter':
                    st['cache_empty_steps'] += 1
                elif abs(cmd) > U_MAX + 1e-6:
                    st['fb_ub_over'] += 1
            # 'prev': unchanged

        # ── results that land inside this step ────────────────────────────
        t_cur = T
        while pending is not None and pending['apply_at'] < T + dt - EPS:
            ta = pending['apply_at']
            if ta > t_cur + EPS:
                pieces.append((ta - t_cur, cmd))
                t_cur = ta
            applied_kind = land(k)
        pieces.append((T + dt - t_cur, cmd))

        # ── plant step ────────────────────────────────────────────────────
        dth = float(sum(d * u for d, u in pieces))
        u_avg = dth / dt
        filtered = abs(u_avg - u_nn) > FILTER_TOL
        st['dev_sum'] += abs(u_avg - u_nn)
        st['filtered_steps' if filtered else 'nn_steps'] += 1
        if filtered and not was_filtered:
            st['activations'] += 1
        was_filtered = filtered
        u_diffs.append(abs(u_avg - u_nn))
        _c = lambda u: float(np.clip(u, -U_MAX, U_MAX))
        _ur = refbox['land'] if refbox['land'] is not None else u_nn
        ud_clip.append(abs(_c(u_avg) - _c(u_nn)))
        ud_pred.append(abs(u_avg - _ur))
        ud_predclip.append(abs(_c(u_avg) - _c(_ur)))
        st['nn_sat_steps'] = st.get('nn_sat_steps', 0) + int(abs(u_nn) > U_MAX + 0.05)

        # How far the true state is from where the cached plan thinks it is --
        # the "we have already moved past the plan" measure (position only).
        off = None
        if cache is not None and cache.state(k) is not None:
            dx = x - cache.state(k)
            off = float(np.hypot(*dx[:2])) if sysm.name == 'unicycle' else abs(float(dx[0]))
            if applied_kind in ('fallback',):
                st['offplan_max'] = max(st['offplan_max'], off)

        x_next = sysm.step(x, pieces)
        hit_pt, hit_seg = sysm.hits(x, pieces, x_next)

        # CERTIFICATE AUDIT. After every step on which a plan is active, is the
        # rest of the filter's current backup still valid from where the robot
        # ACTUALLY is? Re-simulate its remaining controls from the true state
        # and apply the same acceptance test the plan passed when it was built.
        # Zero latency and exact state keep this at 0 by construction; latency
        # (the plan assumed its first control from the tick) and estimation
        # error (the plan starts at x_hat) are what break it. fb_* counts only
        # the steps that flew the cache after a failed solve.
        cert_ok = None
        if cache is not None and cache.u(k) is not None:
            rest = [cache.u(j) for j in range(k + 1, cache.t0 + len(cache.controls))]
            xs, bs = x_next, [np.stack([x_next, x_next], axis=1)]
            for uj in rest:
                xs = sysm.step(xs, [(dt, uj)])
                bs.append(np.stack([xs, xs], axis=1))
            ok_rest, kind_rest, _, _ = _rollout_verdict(mpc_sf, [bs[0]] + bs)
            cert_ok = bool(ok_rest)
            st['cert_checks'] += 1
            if not ok_rest:
                st['cert_invalid'] += 1
                st['cert_invalid_raw'] += int(kind_rest == 'path_raw')
                if st['cert_invalid_first'] < 0:
                    st['cert_invalid_first'] = k + 1
            if applied_kind == 'fallback':
                st['fb_checks'] += 1
                st['fb_invalid'] += int(not ok_rest)
        if applied_kind == 'loiter' or (applied_kind == 'hold' and HOLD == 'backup'
                                         and (cache is None or cache.u(k) is None)):
            if _msfa.TERMINAL == 'loiter' and sysm.name == 'unicycle' and loiter['side']:
                st['loiter_checks'] += 1
                if mpc_sf._loiter_side(np.stack([x_next, x_next], axis=1),
                                       prefer=loiter['side']) != loiter['side']:
                    st['loiter_bad'] += 1
        if ROBUST and cache is not None and cache.tube is not None:
            j = k + 1 - cache.t0
            if 0 <= j < len(cache.states) and j < len(cache.tube):
                sj = cache.states[j]
                st['tube_checks'] += 1
                if (np.hypot(*(x_next[:2] - sj[:2])) > cache.tube[j] + 1e-9
                        or abs(x_next[2] - sj[2]) > cache.e_th + 1e-9):
                    st['tube_out'] += 1
        if (hit_pt or hit_seg) and st['first_collision'] < 0:
            st['first_collision'] = k + 1
            st['offplan_at_collision'] = -1.0 if off is None else off
        st['collided'] |= int(hit_pt)
        st['collided_seg'] |= int(hit_seg)

        # ── measurement + set-membership update ───────────────────────────
        if NOISE == 'none':
            S = np.stack([x_next, x_next], axis=1)
        else:
            z = x_next + _draw_noise(rng, vmax, x_next.shape[0])
            prop = sysm.box_step(S, pieces)
            S = np.stack([np.maximum(prop[:, 0], z - vmax),
                          np.minimum(prop[:, 1], z + vmax)], axis=1)
            if np.any(S[:, 0] > S[:, 1]) or np.any(x_next < S[:, 0] - 1e-9) \
                    or np.any(x_next > S[:, 1] + 1e-9):
                st['truth_out'] += 1           # must stay 0
                S = np.stack([z - vmax, z + vmax], axis=1)

        if trace is not None:
            trace.append(dict(k=k, x=x.tolist(), xh=np.asarray(xh).tolist(),
                              u_nn=u_nn, u_ref=_ur, u_avg=u_avg, pieces=pieces, kind=applied_kind, cert_ok=cert_ok,
                              events=events, off=off,
                              cache_t0=None if cache is None else cache.t0,
                              cache_states=None if cache is None else
                              [s.tolist() for s in cache.states]))
        x = x_next
        traj.append(x.copy())

    st['collided_seg'] = int(st['collided_seg'])
    st['seg_only'] = int(st['collided_seg'] and not st['collided'])
    w = st.pop('solve_wall')
    st['solve_wall_med'] = float(np.median(w)) if w else 0.0
    st['solve_wall_max'] = float(np.max(w)) if w else 0.0
    st['solve_wall_p99'] = float(np.percentile(w, 99)) if w else 0.0
    st['tube_out_trials'] = int(st['tube_out'] > 0)
    for a in ('single_ub_solves', 'single_ub_over', 'single_ub_peak'):
        st[a] = getattr(mpc_sf, a, None)
    # Same intervention counts as alg28 ('u_diff_gt'), from the same per-step
    # |applied - NN| list, so the two are compared at identical thresholds.
    st['u_diff_gt'] = {str(t): int(sum(1 for d in u_diffs if d > t))
                       for t in (0.0, 0.01, 0.05, 0.1)}
    st['u_diff_len'] = len(u_diffs)
    for _nm, _l in (('clip', ud_clip), ('pred', ud_pred), ('predclip', ud_predclip)):
        st[f'u_diff_gt_{_nm}'] = {str(t): int(sum(1 for d in _l if d > t))
                                 for t in (0.0, 0.01, 0.05, 0.1)}
    st.setdefault('nn_sat_steps', 0)
    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update({f'gp_{a}': b for a, b in st.items()})

    if trace is not None:
        with open(os.environ['TTTCARV_GP_TRACE'].replace('{seed}', str(seed)), 'w') as fh:
            json.dump(dict(seed=seed, system=sysm.name,
                           obstacles=[np.asarray(o).tolist() for o in obstacles],
                           stats=st, trace=trace, final=x.tolist()), fh)

    print(f"[alg29 {MODE} delay={st['delay']} hold={HOLD} noise={NOISE}] "
          f"collided={st['collided']} seg={st['collided_seg']} "
          f"solves={st['solves']} fail={st['solve_fail']} fallback={st['fallback_steps']} "
          f"loiter={st['loiter_steps']} act={st['activations']}")
    return (traj, bool(st['collided']), u_diffs, st['activations'], 0, 0, 0, 0)


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    obs = None
    if len(sys.argv) > 2:
        obs = [np.array([float(c) for c in a.split(',')]) for a in sys.argv[2:]]
    test(seed=seed, obstacles=obs)
    print(json.dumps(LAST_RUN_STATS, default=str, indent=1))
