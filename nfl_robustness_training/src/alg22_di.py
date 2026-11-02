"""
Algorithm 22 (DI) — receding-horizon PSF with a constant-acceleration terminal
controller. Double-integrator version of Reach_PSF_FINAL_NL.py.

What this is
------------
alg22's control loop (receding re-solve, one-shot backups, PSF tail,
BOOTSTRAP_RESIM, calibration clamp, the diagnostic counters) on the
DoubleIntegrator with half-plane constraints p > pos_min, v > vel_min. test()
is alg22's loop, specialised only where alg22 reads circle obstacles or applies
the unicycle loiter, so a diff against Reach_PSF_FINAL_NL.py shows every
departure.

SHARED, imported from Reach_PSF_FINAL_NL (system-agnostic given mpc_sf's two
predicates): RefinementTask, concrete_scan, symbolic_step, extend_mpc_sequence,
psf_valid, _rollout_verdict, solve_plan_at, _purge_infeasible,
_max_valid_diverge, _get_nn_control, _bounds_at, _bounds_contain, and the knobs
SINGLE_START_COST_FRAC / RESOLVE_RESERVE / BOOTSTRAP_RESIM.
SHARED, imported from alg14_di: the re-simulation rollout and multi-start of
DIMPCSafetyFilterAcados.

Terminal controller: u = a held constant, 0 < a <= u_max
--------------------------------------------------------
The unicycle's Algorithm 1 branch 3 is a max-turn loiter. The DI analogue is a
constant input u = a > 0. Its safe set

    S_a = { x : the rollout of u = a from x keeps p >= pos_min + buffer
                and v > vel_min at every step }

is FORWARD-INVARIANT under u = a, exactly: the rollout from A x + B a is the
suffix of the rollout from x. (alg22 certifies the unicycle loiter only by
one-step margin; here repeated application is covered.)

Membership is closed form. Under u = a, v_k = v0 + k dt a is increasing, so
min v = v0. p_k = p0 + k dt v0 + dt^2 a k^2 / 2 is convex in k with its vertex
at k = -v0/(dt a) and non-decreasing once v >= 0, so min p is attained at k in
{0, floor(vertex), ceil(vertex)}. Both minima are non-decreasing in (p0, v0),
so a box is in S_a iff its (p_lo, v_lo) corner is. See const_accel_min_p.

With a = u_max this is the set alg14_di's brake-then-hold test certifies
(braking at u_max reaches the same minimum p). What changes is the controller
that realises it: one constant input instead of a two-phase policy.

Where S_a is used
  _collides_D(b)  True iff b is NOT in S_a. The name is kept so the imported
                  alg22 helpers and the fingerprint keys (rej_term_D@..) keep
                  working; for the DI, "D" is the complement of S_a.
  OCP terminal    UNCHANGED. AcadosDIMPC's linear v_corner >= 0,
                  p_corner >= pos_min + buffer is a subset of S_a for every
                  a > 0, and it is linear to keep HPIPM out of status 3. The
                  acceptance test is the exact S_a check, so a plan that reaches
                  S_a without meeting the linear condition is still accepted.
  Branch 3        both sites alg22 has: no backup at all (where alg22 loiters),
                  and PSF tail exhausted (where alg22 logs "branch 3 not
                  implemented" and extends instead).

Admission: the CURRENT bound must lie in S_a, or u = a must have been applied
on the previous timestep. The second clause (the latch) is sound by the
invariance above: the certificate issued at entry covers every later step that
holds u = a, whatever the Kalman bound does afterwards. Without it, measurement
noise widening the bound would drop a certified chain to raw NN control.
Latched steps are counted separately (term_ctrl_latched).

Why P3b passthrough is KEPT (alg14_di removes it)
-------------------------------------------------
alg14_di dropped P3b because the DI unsafe set is absorbing, so "enter D and
come back out" cannot happen. That argues against alg20's D-gated scan. alg22
carries alg21's scan, which uses no D: it flies nominal over a concretely
raw-clear gap to a later tau holding a VALID backup. That is sound on any
system, and a build at tau can fail for reasons that need not persist to a
later tau (solver status, a wide bound that symbolic refinement tightens).

Not supported
-------------
TTTCARV_FIRST_STEP_COST=1 raises: AcadosDIMPC has no stage-dependent cost
weights, so the flag would silently measure nothing. TTTCARV_TAIL_COST_WEIGHT
is ignored for the same reason.

Knobs (in addition to alg22's)
  TTTCARV_DI_TERM_CTRL   1 (default) apply u = a at branch 3; 0 restores
                         alg22's behaviour there (raw NN / extend).
  TTTCARV_DI_TERM_ACCEL  a, default 1.0 (= u_max); requires 0 < a <= u_max.

Usage:
    python nfl_robustness_training/src/alg22_di.py [seed]
    TTTCARV_ALG=alg22_di python nfl_robustness_training/src/run_trials_di.py
"""

import math
import os
import time
from typing import Optional

import numpy as np

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget, calibrated_costs
from alg13_di import HalfPlaneConstraints
from alg14_di import DIMPCSafetyFilterAcados, violates_raw
# System-agnostic PSF machinery, imported unchanged from the unicycle alg22.
from Reach_PSF_FINAL_NL import (
    RefinementTask, concrete_scan, symbolic_step, extend_mpc_sequence,
    psf_valid, _rollout_verdict, _bounds_contain, solve_plan_at,
    _purge_infeasible, _max_valid_diverge, _get_nn_control, _bounds_at,
    SINGLE_START_COST_FRAC, RESOLVE_RESERVE, BOOTSTRAP_RESIM, FLIP_DEADBAND,
    RED, GREEN, BLUE, MAGENTA, CYAN, RESET,
)

# Per-trial counters that do not fit test()'s return tuple; the DI trial driver
# merges this into the fingerprint row. Module-local, not alg22's: the driver
# reads it off whichever module it loaded.
LAST_RUN_STATS: dict = {}

TERM_CTRL  = os.environ.get('TTTCARV_DI_TERM_CTRL', '1') != '0'
TERM_ACCEL = float(os.environ.get('TTTCARV_DI_TERM_ACCEL', '1.0'))


# ── Constant-acceleration safe set S_a ───────────────────────────────────

def const_accel_min_p(p0: float, v0: float, dt: float, accel: float) -> float:
    """Minimum position over the infinite discrete rollout of u = accel > 0."""
    p0, v0 = float(p0), float(v0)
    if v0 >= 0.0:
        return p0
    vertex = -v0 / (dt * accel)
    return min(p0 + k * dt * v0 + 0.5 * dt * dt * accel * k * k
               for k in (0, math.floor(vertex), math.ceil(vertex)))


def in_const_accel_set(bounds, dt: float, accel: float,
                       pos_min: float, vel_min: float, buffer: float) -> bool:
    """Worst-corner membership of the RSOA box in S_a. Exact; see module doc."""
    b = np.asarray(bounds, dtype=float)
    if not np.all(np.isfinite(b)):
        return False
    p_lo, v_lo = b[0, 0], b[1, 0]
    # Strict, matching HalfPlaneConstraints / violates_raw (v <= vel_min is a
    # violation). v is non-decreasing under u = accel, so v0 is the minimum.
    if v_lo <= vel_min:
        return False
    return const_accel_min_p(p_lo, v_lo, dt, accel) >= pos_min + buffer


# ── DI safety filter with the one-shot rollout ───────────────────────────

class DIMPCSafetyFilterOneShot(DIMPCSafetyFilterAcados):
    """
    alg14_di's DI filter with (a) the terminal set replaced by S_a and (b)
    alg22's one-shot rollout, so TTTCARV_ONESHOT / BOOTSTRAP_RESIM mean the
    same thing here as on the unicycle.

    On the DI the one-shot plan is not the approximation it is on the unicycle:
    AcadosDIMPC's model is the plant's exact A, B, so an SQP_RTI iterate is a
    QP solution and |opt_x_num - plant| should sit at solver tolerance.
    oneshot_verify measures that rather than assuming it.
    """

    def __init__(self, tester, dt=0.2, n_horizon=12,
                 pos_min=0.0, vel_min=-1.0, buffer=0.05, u_max=1.0,
                 term_accel=1.0, oneshot=False, oneshot_refine=2,
                 oneshot_verify=False,
                 verify_input_bounds=True, verify_finite_bounds=True):
        term_accel = float(term_accel)
        # a <= 0 is not a terminal controller: for v >= 0 the membership test
        # passes while u < 0 drives v negative. a > u_max is unflyable, and the
        # plant does not clip, so it would be applied as commanded.
        if not (0.0 < term_accel <= u_max):
            raise ValueError(f"term_accel={term_accel} must satisfy "
                             f"0 < a <= u_max={u_max}")
        super().__init__(tester, dt=dt, n_horizon=n_horizon,
                         pos_min=pos_min, vel_min=vel_min, buffer=buffer,
                         u_max=u_max,
                         verify_input_bounds=verify_input_bounds,
                         verify_finite_bounds=verify_finite_bounds)
        self.term_accel     = term_accel
        self.oneshot        = oneshot
        self.oneshot_refine = max(1, int(oneshot_refine))
        self.oneshot_verify = oneshot_verify
        self.oneshot_model_mismatch    = 0.0
        self.oneshot_mismatch_accepted = 0.0
        self.oneshot_solves     = 0
        self.oneshot_bad_status = 0
        print(f"  [alg22_di] terminal set S_a, controller u = {term_accel:g}"
              f" (const accel; overrides the brake-to-rest line above)"
              f"   rollout: {'one-shot' if oneshot else 're-simulation'}")

    def _collides_D(self, bounds) -> bool:
        """True when the bound is NOT in S_a (see module docstring)."""
        return not in_const_accel_set(bounds, self.dt, self.term_accel,
                                      self.pos_min, self.vel_min, self.buffer)

    def _score_rollout(self, traj_bounds):
        """Lower is better: terminal shortfall against S_a. 0 iff accepted."""
        b = np.asarray(traj_bounds[-1], dtype=float)
        if not self._collides_D(b):
            return 0.0
        if not np.all(np.isfinite(b)):
            return float('inf')
        min_p = const_accel_min_p(b[0, 0], b[1, 0], self.dt, self.term_accel)
        # The floor keeps a boundary miss (v_lo == vel_min) above the parent's
        # 1e-9 early-return threshold, which would otherwise accept it.
        return (max(0.0, (self.pos_min + self.buffer) - min_p)
                + max(0.0, self.vel_min - b[1, 0]) + 1e-6)

    def _run_mpc_single(self, initial_bounds, center, extra_inflation,
                        warm_start_u):
        """Dispatch to the one-shot or the re-simulation rollout."""
        if self.oneshot:
            return self._run_mpc_single_oneshot(
                initial_bounds, center, extra_inflation, warm_start_u)
        return super()._run_mpc_single(
            initial_bounds, center, extra_inflation, warm_start_u)

    def _run_mpc_single_oneshot(self, initial_bounds, center, extra_inflation,
                                warm_start_u):
        """
        Solve the OCP once and return the PLANT's response to its own u[0..N-1].
        Port of UniycleMPCSafetyFilterAcados._run_mpc_single_oneshot; see that
        docstring for why the plan is re-simulated rather than taken from
        opt_x_num. Boxes carry the initial half-widths, as in the resim path.
        """
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_p, hw_v  = float(half_widths[0]), float(half_widths[1])
        self._mpc._obs_inflation_buf[0] = hw_p + extra_inflation
        self._mpc._obs_inflation_buf[1] = hw_v

        x = np.asarray(center, dtype=float).reshape(-1, 1)
        self._mpc.x0 = x
        if warm_start_u is not None:
            self._mpc._nom_ctrl_buf[:] = float(warm_start_u)
        else:
            self._fill_nom_ctrl_buf(x.flatten())
        self._mpc.set_initial_guess()

        predicted_states = None
        for _ in range(self.oneshot_refine):
            if predicted_states is not None:
                self._fill_nom_ctrl_buf_from_states(predicted_states[1:])
            self._mpc.make_step(x, n_rti=self._mpc._RTI_WARMUP_ITERS)
            predicted_states = [
                np.array(self._mpc.opt_x_num['_x', k, 0]).flatten()
                for k in range(self.n_horizon + 1)
            ]

        controls = self._mpc.opt_u_seq
        cl_sys   = self.tester.analyzer.cl_system
        xs       = x.flatten().reshape(1, -1)
        plant_states = [x.flatten().copy()]
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
                [xf[0] - hw_p, xf[0] + hw_p],
                [xf[1] - hw_v, xf[1] + hw_v],
            ]))
            points.append(xf.reshape(-1, 1).copy())

        if self.oneshot_verify:
            self.oneshot_solves += 1
            if getattr(self._mpc, 'last_status', 0) not in (0, 2):
                self.oneshot_bad_status += 1
            worst = max(
                (float(np.max(np.abs(plant_states[k] - predicted_states[k])))
                 for k in range(1, self.n_horizon + 1)),
                default=0.0)
            self.oneshot_model_mismatch = max(self.oneshot_model_mismatch, worst)
            if _rollout_verdict(self, trajectory_bounds)[0]:
                self.oneshot_mismatch_accepted = max(
                    self.oneshot_mismatch_accepted, worst)

        return trajectory_bounds, points, controls


# ── Backup build and passthrough scan (DI diagnostics) ───────────────────

def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int,
                     reasons: dict = None) -> bool:
    """
    alg22's build_mpc_backup with DI diagnostics. The verdict is alg22's
    _rollout_verdict (path nodes raw-clear, terminal in S_a), plus alg14_di's
    launch-bound check: traj_bounds[0] is the RSOA at tau, which the verdict
    slices off, so without it a backup could be certified from a bound that
    already violates the constraint.

    Reason labels are `bucket(detail)` so the split('(')[0] counters bucket
    them cleanly: start_raw / path_raw / term_D / solver_error.
    """
    h = tester.horizons.get(tau)
    if h is None:
        if reasons is not None:
            reasons[tau] = "no_bounds(concrete RSOA not propagated to tau)"
        return False
    bounds = h.get_tight_bound()
    if bounds is None:
        if reasons is not None:
            reasons[tau] = "no_bounds(get_tight_bound returned None)"
        return False
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        _t0 = time.time()
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
            bounds, center, multi_start=True)
        elapsed = time.time() - _t0

        label = None
        if mpc_sf._collides_raw(traj_bounds[0]):
            b = traj_bounds[0]
            ok = False
            label = (f"start_raw(p_lo={b[0,0]:+.4f} v_lo={b[1,0]:+.4f};"
                     f" RSOA at tau already violates the constraint)")
        else:
            ok, kind, node, n_nodes = _rollout_verdict(mpc_sf, traj_bounds)
            if not ok and kind == 'path_raw':
                b = traj_bounds[node]
                label = (f"path_raw(node={node}/{n_nodes - 1}"
                         f" p_lo={b[0,0]:+.4f} min {mpc_sf.pos_min:+.3f}"
                         f" v_lo={b[1,0]:+.4f} min {mpc_sf.vel_min:+.3f})")
            elif not ok and kind == 'term_D':
                b = np.asarray(traj_bounds[-1], dtype=float)
                min_p = (const_accel_min_p(b[0, 0], b[1, 0], mpc_sf.dt,
                                           mpc_sf.term_accel)
                         if np.all(np.isfinite(b)) else float('nan'))
                label = (f"term_D(not in S_a: p_lo={b[0,0]:+.4f}"
                         f" v_lo={b[1,0]:+.4f} -> u={mpc_sf.term_accel:g}"
                         f" min_p={min_p:+.4f} need >="
                         f" {mpc_sf.pos_min + mpc_sf.buffer:+.4f})")
            elif not ok:
                label = f"{kind}()"

        print(f"  [MPC build] τ={tau}  {'VALID' if ok else f'INFEASIBLE({label})'}"
              f"  took {elapsed:.3f}s")
        if ok:
            mpc_buffer[tau] = (list(controls), list(traj_bounds))
            if reasons is not None:
                reasons.pop(tau, None)
            return True
        mpc_buffer[tau] = None
        if reasons is not None:
            reasons[tau] = label
        return False
    except Exception as e:
        print(f"  [PSF buf] MPC solve failed at τ={tau}: {e}")
        mpc_buffer[tau] = None
        if reasons is not None:
            reasons[tau] = f"solver_error({e})"
        return False


def scan_window(tester, mpc_sf, mpc_buffer: dict, wall_tau: int,
                scan_limit: int, budget, reasons: dict = None) -> Optional[int]:
    """
    alg22's P3b scan with the half-plane raw test in place of circles. Returns
    the first tau past the wall holding a VALID backup, else None. Aborts on the
    first raw-violating bound: nominal is only concrete-certified before it.
    scan_limit is inclusive.
    """
    for tau in range(wall_tau + 1, scan_limit + 1):
        if tau not in tester.horizons:
            break
        bounds = tester.horizons[tau].get_tight_bound()
        if bounds is None:
            break
        if mpc_sf._collides_raw(bounds):
            print(f"  [Scan window] tau={tau} violates the raw constraint — abort")
            return None
        if tau in mpc_buffer:
            if mpc_buffer[tau] is not None:
                print(f"  [Scan window] tau={tau} already VALID in buffer")
                return tau
            continue
        if budget.remaining < budget.mpc_cost:
            print(f"  [Scan window] Budget exhausted at tau={tau}")
            return None
        if build_mpc_backup(mpc_sf, tester, mpc_buffer, tau, reasons):
            print(f"  [Scan window] tau={tau} VALID — passthrough t_diverge found")
            return tau
        print(f"  [Scan window] tau={tau} raw-clear but MPC INFEASIBLE — continue")
    return None


# ── Simulation loop ──────────────────────────────────────────────────────

def test(seed=None, analyzer=None,
         pos_min: float = 0.0, vel_min: float = -1.0, buffer: float = 0.05,
         MAX_TIME: int = 40, init_range=None):
    if analyzer is None:
        analyzer = setup_analyzer('DoubleIntegrator',
                                  'constraint_default_more_data_5hz',
                                  init_range=init_range)

    if seed is None:
        seed = 1401830092

    # ReachabilityTester needs *some* obstacle list to construct; the half-plane
    # constraints are swapped in afterwards, as in alg13_di / alg14_di.
    def _make_tester(s):
        t = ReachabilityTester(analyzer, [], seed=s)
        t.obstacles = HalfPlaneConstraints(pos_min=pos_min, vel_min=vel_min)
        return t

    tester = _make_tester(seed)
    dt     = float(analyzer.cl_system.dynamics.dt)
    u_max  = 1.0   # MUST match AcadosDIMPC lbu/ubu and DoubleIntegrator.u_limits

    # alg22's first-step-only cost needs stage-dependent weights AcadosDIMPC does
    # not have. Refuse rather than run an A/B arm that silently changes nothing.
    first_step_cost = os.environ.get('TTTCARV_FIRST_STEP_COST', '0') != '0'
    if first_step_cost:
        raise ValueError("TTTCARV_FIRST_STEP_COST=1 is not supported by alg22_di")
    tail_cost_weight = 0.0   # inert on the DI; kept so alg22's logging is unchanged
    # One-shot rollout, same knobs and meaning as alg22 (DIMPCSafetyFilterOneShot).
    oneshot        = os.environ.get('TTTCARV_ONESHOT', '0') != '0'
    oneshot_refine = int(os.environ.get('TTTCARV_ONESHOT_REFINE', '2'))
    oneshot_verify = os.environ.get('TTTCARV_ONESHOT_VERIFY', '0') != '0'
    mpc_sf = DIMPCSafetyFilterOneShot(
        tester, dt=dt, n_horizon=12,
        pos_min=pos_min, vel_min=vel_min, buffer=buffer, u_max=u_max,
        term_accel=TERM_ACCEL,
        oneshot=oneshot, oneshot_refine=oneshot_refine,
        oneshot_verify=oneshot_verify)

    MAX_SYMBOLIC_HORIZON = 5

    budget = TimeBudget(timestep_budget=0.20)
    sym_costs, concrete_cost, mpc_cost = calibrated_costs(
        lambda: _make_tester(0),
        lambda t: (lambda: build_mpc_backup(mpc_sf, t, {}, 1)),
        MAX_SYMBOLIC_HORIZON,
    )
    budget.symbolic_costs = dict(sym_costs)
    budget.concrete_cost  = concrete_cost
    budget.mpc_cost       = mpc_cost

    # ── Calibration sanity guard (2026-09-11) ─────────────────────────────
    # mpc_cost is measured once per process, during the startup stampede when
    # 64 workers initialise acados and TF at the same instant. Measured spread
    # on node5000: healthy median 0.066s, max 0.174s, and one worker 0.295s.
    # An estimate above the WHOLE timestep budget is not a steady-state cost,
    # it is a startup transient — but the gate `remaining >= mpc_cost` reads it
    # as "an MPC build can never be afforded" and is then false forever, since
    # remaining starts at timestep_budget. That silently disables the safety
    # filter for the entire life of the process.
    #
    # Clamping is the conservative direction here, which is not the obvious
    # one. An optimistic mpc_cost risks overrunning a single timestep — bounded,
    # and visible in wall_per_timestep_s. A pessimistic one disables safety
    # outright and is invisible. So prefer the bounded failure.
    #
    # THRESHOLD CORRECTED 2026-09-12. Testing `mpc_cost >= timestep_budget` is
    # too loose, and the old clamp TARGET (0.9 * budget = 0.18s) fails its own
    # test. P1 must first propagate concrete bounds to t+1 -- it cannot build a
    # backup without them -- so by the time it checks `remaining >= mpc_cost`,
    # concrete_cost is already spent. The real requirement is
    #     mpc_cost + concrete_cost <= timestep_budget.
    # Measured case (runs/alg22_wdbg2, worker 1783205): mpc=0.1782s,
    # concrete=0.0235s. 0.1782 < 0.20 so the old guard passed it, then P1 spent
    # 0.0259s on bounds and tested 0.174 >= 0.1782 -> false. That worker could
    # never build at t=0 and produced 7 of the 11 psf_no_diverge events in its
    # run. 0.18 + 0.0235 = 0.2035 > 0.20, so the old clamp could put a worker
    # INTO this state while claiming to rescue it.
    mpc_cost_clamped = 0
    _headroom = budget.timestep_budget - concrete_cost
    if mpc_cost >= _headroom:
        clamped = 0.9 * _headroom
        print(f"[calibrate] WARNING mpc_cost={mpc_cost:.4f}s >= timestep_budget-"
              f"concrete_cost={_headroom:.4f}s — P1 spends concrete_cost="
              f"{concrete_cost:.4f}s on bounds before it tests the gate, so the "
              f"build at t+1 could never be afforded and the filter would be OFF "
              f"for this whole process. Clamping to {clamped:.4f}s.")
        budget.mpc_cost  = clamped
        mpc_cost_clamped = 1

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    # ── PSF buffer state ───────────────────────────────────────────────────
    mpc_buffer:        dict                      = {}   # tau -> (controls, traj_bounds) | None
    mpc_reason:        dict                      = {}   # tau -> why the build at tau was rejected
    concrete_until:    int                       = 0
    mpc_horizon_until: int                       = -1   # last tau MPC was attempted
    conflict_time:     Optional[int]             = None
    t_diverge:         Optional[int]             = None
    pending_job:       Optional[RefinementTask] = None
    wall_tau:          Optional[int]             = None   # first INFEASIBLE tau hit in P3; reset each iteration

    # ── MPC execution state (mirrors alg11) ───────────────────────────────
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'traj_bounds':   [],
        'needed':        False,
    }
    mpc_started      = False

    current_timestep = 0
    u_diffs          = []   # |u_mpc - u_nn| per timestep; 0.0 when nominal
    mpc_calls        = 0    # number of PSF activations
    psf_no_diverge   = 0    # PSF failed, no valid t_diverge at all
    # WHY there was no backup. The counter alone cannot distinguish "the build
    # was never attempted because the budget was spent" from "the build ran and
    # the acceptance test rejected it" -- and those need opposite fixes. The
    # printed diagnostic goes to the worker's stdout, which is /dev/null unless
    # TTTCARV_WORKER_LOG_DIR is set, so on the alg22 10k the reason for the one
    # event was lost. Recorded in the fingerprint from here on.
    psf_no_diverge_reasons: dict = {}   # short key -> count
    # P1's budget gate is the only path that leaves mpc_buffer[t+1] *absent*
    # with no reason recorded, which is exactly what every observed
    # psf_no_diverge diagnostic shows (buf[t+1]=absent, td=never_set).
    # It was uncounted: mpc_skipped_no_budget counts the step-2 rebuild loop,
    # not this. Counted separately so the two are never conflated again.
    p1_skipped_no_budget = 0   # P1 build refused by the budget gate
    p1_skipped_no_bounds = 0   # P1 build impossible, concrete bounds missing
    # Terminal controller (u = TERM_ACCEL). prev_t is the timestep it was last
    # applied at -- the latch that lets an admitted chain continue. refused:
    # branch 3 was due but the current bound was outside S_a and unlatched.
    term_ctrl = {'prev_t': None, 'no_backup': 0, 'tail': 0, 'latched': 0,
                 'refused': 0}
    p1_bootstrap_resim   = 0   # no-fallback build forced onto the re-sim solver
    p1_bootstrap_infeasible = 0  # no-fallback build that still came back INFEASIBLE
    p1_t0_infeasible     = 0   # the t=0 build specifically -- the one with an empty buffer
    psf_unrecoverable    = 0   # no backup AND current RSOA not in S_a, unlatched
    psf_queue_empty  = 0    # PSF failed, t_diverge valid but queue empty after extend
    psf_stale_refused = 0   # PSF failed, only a stale (ctrl_idx>0) plan existed
    passthrough_found = 0   # P3b scan_window returned a raw-clear τ with a valid backup
    passthrough_used  = 0   # PSF PASSTHROUGH branch executed (nominal, no backup at t+1)
    passthrough_found = 0   # P3b scan_window returned a τ outside D after a wall
    passthrough_used  = 0   # PSF PASSTHROUGH branch executed (nominal while in D)

    # ── Receding-horizon active phase (alg22) ─────────────────────────────
    # TTTCARV_RECEDING=0 restores alg21's open-loop active phase, which makes
    # this file a behavioural no-op against alg21 — the regression check that
    # the copy did not drift anywhere outside the control-selection step.
    receding = os.environ.get('TTTCARV_RECEDING', '1') != '0'
    # TTTCARV_PSF_TAIL=1 makes the failure path branch 2 of Algorithm 1 in the
    # PSF literature. A plan committed at k_bar certifies exactly n_horizon
    # controls, and the terminal constraint is what makes its shifted tail a
    # feasible solution to the shrunk problem — so on a failed re-solve the
    # stored u*[k - k_bar] needs no new solve to be safe, but ONLY while
    # k - k_bar < N. alg21 instead calls extend_mpc_sequence and rides on
    # indefinitely, which manufactures plan past the horizon the terminal
    # constraint ever covered, from a predicted rather than a measured bound.
    # Off by default so the receding arm stays byte-comparable to its last run.
    psf_tail = os.environ.get('TTTCARV_PSF_TAIL', '0') != '0'
    mpc_resolved             = 0  # active timesteps whose input came from a fresh solve
    mpc_resolve_failed       = 0  # active timesteps that fell back to the committed plan
    mpc_fallback_uncertified = 0  # fallbacks where the current RSOA left traj_bounds[idx]
    mpc_tail_exhausted       = 0  # k - k_bar reached N: certificate spent, branch 3 due
    # Why a re-solve failed. "the solver failed" and "the solver returned a plan
    # that violates the terminal constraint" are different problems with
    # different fixes, and mpc_resolve_failed alone cannot tell them apart.
    mpc_fail_reason: dict = {}
    # Which acceptance check rejected the plan, and at which node. Separates
    # "the plan clipped an obstacle early" from "the plan never got out of D by
    # the end of the horizon" — different problems with different fixes.
    mpc_reject_detail: dict = {}
    # Homotopy-jump probe. The per-step re-solve warm-starts from a fresh NN
    # rollout (warm_start_omega=None), not from the previous solution, so
    # nothing structurally carries the previous step left/right choice
    # forward. If the solver is flipping between the two ways around an
    # obstacle, the applied turn direction alternates. Two views:
    #   applied_u_* -- sign of the input actually applied, on consecutive
    #                  active timesteps. This is what the robot does.
    #   plan_flips  -- sign of the mean of a fresh plan vs the mean of the
    #                  tail of the plan it replaces. This is what the solver
    #                  decided, and it fires even when u[0] happens to agree.
    # *_pairs is the denominator: comparisons where both sides cleared the
    # deadband. A flip RATE needs it; a flip count alone is uninterpretable.
    applied_u_flips    = 0
    applied_u_pairs    = 0
    plan_flips         = 0
    plan_pairs         = 0
    _prev_applied_u    = None
    _prev_applied_t    = None
    # Times the backup-build loop stopped because the timestep budget could not
    # cover one more MPC build. Uncounted until 2026-09-11, and that is the whole
    # reason the dead-worker mode survived five sessions: mpc_cost is calibrated
    # ONCE per process, so when that single measurement lands above
    # timestep_budget the gate is false on every timestep of every trial the
    # worker ever runs. The filter never builds a backup, the process flies
    # unfiltered nominal, and the results look ordinary. Nothing incremented,
    # nothing printed, nothing in the fingerprint.
    mpc_skipped_no_budget = 0
    # Subset of the above: builds skipped purely to honour resolve_reserve.
    mpc_build_yielded_to_resolve = 0
    print(f"[alg22_di] receding-horizon active phase: {'ON' if receding else 'OFF (alg21 parity)'}"
          f"   first-step-only cost: {'ON' if first_step_cost else 'OFF'}"
          f"{f' (tail_w={tail_cost_weight:g})' if first_step_cost else ''}"
          f"   bounded tail (Alg1 branch 2): {'ON' if psf_tail else 'OFF'}")

    # ── RSOA certificate dump (off unless TTTCARV_RSOA_DUMP names a path) ──
    # Every executed timestep leaves through real_state_empirical or
    # real_state_mpc, so wrapping those two is the only placement that cannot
    # miss a branch — the six call sites in the loop are covered by
    # construction rather than by my having found all of them. The record is
    # taken *before* the step so it holds the bounds as they stood when the
    # step was authorized, not as they were later re-propagated.
    _rsoa_path = os.environ.get('TTTCARV_RSOA_DUMP')
    rsoa_log   = []
    if _rsoa_path:
        def _bnd(t):
            h = tester.horizons.get(t)
            if h is None:
                return None
            b = h.get_tight_bound()
            return None if b is None else np.asarray(b).tolist()

        def _real(t):
            h = tester.horizons.get(t)
            if h is None:
                return None
            for c in h.calculations.values():
                if 'real_state' in c:
                    return np.asarray(c['real_state']).flatten().tolist()
            return None

        def _wrap(fn, kind):
            def inner(t, arg):
                rec = {'t': t, 'kind': kind,
                       'conflict_time': conflict_time, 't_diverge': t_diverge,
                       'mpc_started': mpc_started,
                       'committed_at': mpc_state['committed_at'],
                       'bound_t': _bnd(t), 'real_t': _real(t)}
                if kind == 'mpc' and mpc_state['committed_at'] is not None:
                    # traj_bounds[k] is the backup's bound at committed_at + k.
                    # extend_mpc_sequence splices this list, so the index can go
                    # stale; the checker reports a missing entry rather than
                    # assuming one.
                    tb, i = mpc_state['traj_bounds'], t - mpc_state['committed_at']
                    rec['backup_idx']      = i
                    rec['backup_len']      = len(tb)
                    rec['backup_bound_t']  = (np.asarray(tb[i]).tolist()
                                              if 0 <= i < len(tb) else None)
                    rec['backup_bound_t1'] = (np.asarray(tb[i + 1]).tolist()
                                              if 0 <= i + 1 < len(tb) else None)
                out = fn(t, arg)
                rec['bound_t1'] = _bnd(t + 1)
                rec['real_t1']  = _real(t + 1)
                rsoa_log.append(rec)
                return out
            return inner

        tester.real_state_empirical = _wrap(tester.real_state_empirical, 'empirical')
        tester.real_state_mpc       = _wrap(tester.real_state_mpc,       'mpc')

    # ── Per-timestep budget ledger (opt-in via TTTCARV_BUDGET_DUMP) ──────
    # Charges the four kinds of work the budget actually prices, so a timestep
    # can be scored both as-is and with the deferrable symbolic refinement
    # removed. Wrapping the bound methods catches every call site; the residual
    # (elapsed minus the charges) is Python/geometry overhead.
    _budget_path = os.environ.get('TTTCARV_BUDGET_DUMP')
    if _budget_path:
        budget.ledger_enabled       = True
        tester.symbolic             = budget.charged(tester.symbolic,   'symbolic')
        tester.concrete             = budget.charged(tester.concrete,   'concrete')
        tester.real_state_empirical = budget.charged(tester.real_state_empirical, 'step')
        tester.real_state_mpc       = budget.charged(tester.real_state_mpc,       'step')
        mpc_sf._run_mpc_from_bounds = budget.charged(
            mpc_sf._run_mpc_from_bounds, 'mpc')

    def _term_ctrl_admissible(t):
        """(admissible, latched) for applying u = a at timestep t."""
        b = _bounds_at(tester, t)
        if b is not None and not mpc_sf._collides_D(b):
            return True, False
        if term_ctrl['prev_t'] is not None and term_ctrl['prev_t'] == t - 1:
            return True, True
        return False, False

    def _apply_term_ctrl(t, site, latched):
        """Apply u = a at t and record it. The caller advances the clock."""
        u    = np.array([mpc_sf.term_accel])
        u_nn = _get_nn_control(tester, t)
        u_diffs.append(float(np.abs(u[0] - u_nn[0])) if u_nn is not None else 0.0)
        tester.real_state_mpc(t, u)
        term_ctrl['prev_t'] = t
        term_ctrl[site]    += 1
        if latched:
            term_ctrl['latched'] += 1

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next   = current_timestep + 1
        wall_tau = None  # reset each iteration; P3 sets it if it hits INFEASIBLE

        # ══════════════════════════════════════════════════════════════════
        #  MPC ACTIVE PHASE
        # ══════════════════════════════════════════════════════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}  conflict={conflict_time}"
                  f"  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']

            # Clear future horizons; then interleave: one concrete step -> one MPC build, repeat while budget allows.
            # This is to check if we can revert to nominal.
            # If we apply concrete now and mpc build is valid in the next timestep, we are safe to revert

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
                    # INVARIANT: a certificate in mpc_buffer is only meaningful
                    # while the horizon it was derived from is live. The robot
                    # has deviated from the nominal plan, so every future
                    # horizon is stale -- and so is every backup built from one.
                    # Dropping them here is fail-safe by construction: The rebuild loop below runs
                    # AFTER this purge, so backups re-derived from the actually
                    # flown state survive untouched.

                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)

            # Update mpc_horizon_until to the current timestep to match the purge point, so we don't create a gap in buffer
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            # Held back from the rebuild loop so the re-solve below can always
            # pay for itself. Zero when there is no re-solve to protect.
            resolve_reserve = (budget.mpc_cost * SINGLE_START_COST_FRAC
                               if (receding and RESOLVE_RESERVE) else 0.0)
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                # One concrete step
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                # One MPC build for this timestep.
                #
                # The build at t_next is NOT speculative: it is the revert test
                # itself, the exact entry psf_valid() reads below. It must never
                # be traded for the re-solve, and it cannot usefully be, because
                # the two are mutually exclusive -- if it comes back VALID the
                # revert fires and `continue`s before the re-solve is reached,
                # and if it comes back INFEASIBLE the loop breaks here after one
                # build. Every LATER build is lookahead buffer for after the
                # revert, and that is the only work the reserve should touch.
                #
                # Reserving against the first build instead (as the version
                # measured in runs/alg22_reserve did) blocks the revert
                # decision, so the filter cannot hand control back: active
                # timesteps went 2496 -> 5834 on 2500 trials while
                # mpc_fallback_uncertified barely moved, 425 -> 404.
                need = budget.mpc_cost + (0.0 if concrete_until == t_next
                                          else resolve_reserve)
                if concrete_until in tester.horizons and budget.remaining >= need:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until, mpc_reason)
                    mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                    if valid:
                        t_diverge = concrete_until
                    print(f"  [MPC PSF] τ={concrete_until} {'VALID' if valid else 'INFEASIBLE'}"
                          f"  t_diverge={t_diverge}")
                    if not valid:
                        break  # cannot revert to nominal, so stop building backups
                else:
                    # Only a budget shortfall is worth counting; falling off the
                    # end of tester.horizons is the loop's normal exit.
                    if concrete_until in tester.horizons:
                        mpc_skipped_no_budget += 1
                        # A speculative lookahead build that was affordable on
                        # its own and was given up only to keep the re-solve
                        # payable. Separating this from a genuine shortfall is
                        # what makes the reserve measurable: it is the price
                        # paid, and it should show up against a fall in
                        # fail_no_budget. The revert test at t_next is exempt
                        # above and so can never land here.
                        if budget.remaining >= budget.mpc_cost:
                            mpc_build_yielded_to_resolve += 1
                    break  # no MPC budget remaining

            
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # PSF reversion: if backup from t+1 (from the concrete RSOA) is valid, return to nominal.
            # Build forward buffer with remaining budget so we have lookahead buffer if we can revert
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None

                # Reset pending_job to start from the new post-MPC position.
                # The mpc changed trajectory so prev symbolic propagation is now inaccurate
                # conflict_time is kept. P4 will re-verify from here.
                pending_job = (RefinementTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
                # were built in section above and remain valid backup buffers
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep,
                                                       below=conflict_time)
                if t_diverge is None or t_diverge < t_next:
                    # This branch is entered *because* psf_valid(mpc_buffer, t_next),
                    # and that backup was built this timestep from the concrete RSOA
                    # re-propagated after the maneuver. conflict_time, by contrast,
                    # was computed under the pre-maneuver trajectory and is stale
                    # here by construction (that is why pending_job re-verifies from
                    # t_next). Letting a stale ceiling veto a just-verified backup
                    # leaves t_diverge=None, and the filter then falls through to
                    # unconditional nominal for the remainder of the run.
                    print(f"  [PSF REVERT] ceiling conflict_time={conflict_time}"
                          f" excluded the just-verified backup at τ={t_next}"
                          f" (would give t_diverge={t_diverge}) — using τ={t_next}")
                    t_diverge = t_next
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            # ── Receding-horizon control selection (alg22) ────────────────
            # alg21 applies queue[t - committed_at]: one plan, flown open loop
            # for the whole active phase. The certificate behind that input is
            # traj_bounds[ctrl_idx], computed from the RSOA at committed_at —
            # but the set the robot is actually in is the Kalman bound written
            # by real_state_mpc each step, which is not guaranteed to stay
            # inside it. Re-solving from the current bound and applying only
            # u[0] keeps the applied input backed by the set the robot is in.
            resolved = False
            if receding:
                ok, r_ctrls, r_traj, r_label = solve_plan_at(
                    mpc_sf, tester, current_timestep, budget)
                if (not ok and r_label != "no_budget"
                        and budget.remaining >= budget.mpc_cost):
                    # One retry from the wider warm-start set before accepting a
                    # plan built from a stale bound.
                    ok, r_ctrls, r_traj, r_label = solve_plan_at(
                        mpc_sf, tester, current_timestep, budget, multi_start=True)
                if ok:
                    # Compare the new plan against the tail of the one it
                    # replaces BEFORE overwriting mpc_state.
                    _old_tail = [float(np.asarray(c).flatten()[0])
                                 for c in (mpc_state.get("controls") or [])[ctrl_idx:]]
                    _new_all  = [float(np.asarray(c).flatten()[0]) for c in r_ctrls]
                    if _old_tail and _new_all:
                        _a = float(np.mean(_old_tail))
                        _b = float(np.mean(_new_all))
                        if abs(_a) > FLIP_DEADBAND and abs(_b) > FLIP_DEADBAND:
                            plan_pairs += 1
                            if _a * _b < 0.0:
                                plan_flips += 1
                                print(f"{RED}[HOMOTOPY] t={current_timestep} fresh"
                                      f" plan reverses acceleration sign: old tail"
                                      f" mean {_a:+.3f} -> new plan mean"
                                      f" {_b:+.3f}{RESET}")
                    # Re-committing resets committed_at, so ctrl_idx restarts at
                    # 0 and can never index past a control that was not applied.
                    mpc_state['committed_at'] = current_timestep
                    mpc_state['controls']     = r_ctrls
                    mpc_state['traj_bounds']  = r_traj
                    ctrl_idx = 0
                    queue    = mpc_state['controls']
                    resolved = True
                    mpc_resolved += 1
                    print(f"{MAGENTA}[MPC RESOLVE] t={current_timestep} re-solved from"
                          f" current bound  |plan|={len(queue)}{RESET}")
                else:
                    mpc_resolve_failed += 1
                    _rk = str(r_label).split('(')[0]
                    mpc_fail_reason[_rk] = mpc_fail_reason.get(_rk, 0) + 1
                    if _rk == 'rollout_rejected' and '(' in str(r_label):
                        _d = str(r_label).split('(', 1)[1].rstrip(')')
                        _kind, _, _pos = _d.partition('@')
                        _node = _pos.split('/')[0]
                        for _key in (_kind, f"{_kind}@{_node}"):
                            mpc_reject_detail[_key] = mpc_reject_detail.get(_key, 0) + 1
                    print(f"{RED}[MPC RESOLVE] t={current_timestep} FAILED ({r_label})"
                          f" — falling back to plan from"
                          f" t={mpc_state['committed_at']}  ctrl_idx={ctrl_idx}{RESET}")

            if not resolved:
                # Fallback, and alg21's only path: ride the committed plan.
                # Extend whenever <= n_horizon controls remain. Placed after the
                # revert check: no point extending a queue we're about to abandon.
                #
                # Under psf_tail the extend is withheld while the committed
                # plan's own certificate still covers this index. ctrl_idx is
                # exactly k - k_bar, so the bound of Algorithm 1's branch 2 is
                # ctrl_idx < n_horizon; inside it the stored control is the
                # shifted tail and is safe without a solve. At the bound the
                # certificate is spent and branch 3 is due: the terminal
                # controller u = a, admitted when the current bound is in S_a
                # or the chain is latched. When it is not admissible the old
                # extend runs, as a counted departure from the algorithm.
                tail_ok = (receding and psf_tail
                           and ctrl_idx < mpc_sf.n_horizon
                           and ctrl_idx < len(queue))
                if receding and psf_tail and not tail_ok:
                    mpc_tail_exhausted += 1
                    _adm, _latched = _term_ctrl_admissible(current_timestep)
                    if TERM_CTRL and _adm:
                        print(f"{RED}[PSF TAIL] t={current_timestep} certificate spent"
                              f" (k-k_bar={ctrl_idx}, N={mpc_sf.n_horizon}) —"
                              f" branch 3: terminal controller u={mpc_sf.term_accel:g}"
                              f"{' (latched)' if _latched else ''}  (seed={seed}){RESET}")
                        _apply_term_ctrl(current_timestep, 'tail', _latched)
                        current_timestep += 1
                        continue
                    if TERM_CTRL:
                        term_ctrl['refused'] += 1
                    print(f"{RED}[PSF TAIL] t={current_timestep} certificate spent"
                          f" (k-k_bar={ctrl_idx}, N={mpc_sf.n_horizon},"
                          f" |plan|={len(queue)}) — terminal controller"
                          f" {'not admissible (current RSOA not in S_a)' if TERM_CTRL else 'disabled'},"
                          f" extending as alg21 does  (seed={seed}){RESET}")
                if not tail_ok:
                    if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                        extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                            ctrl_idx=ctrl_idx, budget=budget)
                        queue = mpc_state['controls']

                    # Apply queued MPC control
                    if ctrl_idx >= len(queue):
                        #shouldnt run if everything is working
                        print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                        break

                # The fallback input is certified only while the bound the robot
                # is actually in sits inside the bound the plan predicted for
                # this index. Outside that, traj_bounds[ctrl_idx] describes a set
                # the robot is not in and its clearance check is void. Still the
                # best available action, so it is applied — but counted, never
                # silently treated as verified.
                if receding and ctrl_idx > 0:
                    _tb  = mpc_state['traj_bounds']
                    _now = _bounds_at(tester, current_timestep)
                    if (ctrl_idx >= len(_tb)
                            or not _bounds_contain(_tb[ctrl_idx], _now)):
                        mpc_fallback_uncertified += 1
                        print(f"{RED}[MPC RESOLVE] fallback at idx={ctrl_idx}:"
                              f" current RSOA is NOT inside traj_bounds[{ctrl_idx}]"
                              f" — applied input is uncertified"
                              f"  (seed={seed}, t={current_timestep}){RESET}")

            ctrl = queue[ctrl_idx]
            # Only consecutive active timesteps are comparable: a gap means the
            # filter went back to nominal in between, and a sign change across
            # that gap is the NN doing, not the solver.
            _u0 = float(np.asarray(ctrl).flatten()[0])
            if (_prev_applied_u is not None
                    and _prev_applied_t == current_timestep - 1
                    and abs(_u0) > FLIP_DEADBAND
                    and abs(_prev_applied_u) > FLIP_DEADBAND):
                applied_u_pairs += 1
                if _u0 * _prev_applied_u < 0.0:
                    applied_u_flips += 1
            _prev_applied_u = _u0
            _prev_applied_t = current_timestep
            print(f"{MAGENTA}[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                  f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s{RESET}")
            u_nn = _get_nn_control(tester, current_timestep)
            u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
            continue

        # ═════════════════════════════════════════════════════════════════
        #  NOMINAL / PRE_CONFLICT PHASE
        # ═════════════════════════════════════════════════════════════════

        # Stale conflict check — if we've already navigated past conflict_time
        # safely, clear it so we don't stay in PRE_CONFLICT forever.
        if conflict_time is not None and current_timestep > conflict_time:
            print(f"{GREEN}[STALE] conflict_time={conflict_time} is in the past"
                  f" (t={current_timestep}) — clearing{RESET}")
            conflict_time = None
            pending_job   = None

        phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
        color = BLUE if phase == "PRE_CONFLICT" else GREEN
        print(f"\n{color}CURRENT TIMESTEP ==== {current_timestep}"
              f"  concrete_until={concrete_until}  mpc_hz={mpc_horizon_until}"
              f"  conflict={conflict_time}  t_diverge={t_diverge}  [{phase}]{RESET}")

        # -- P1: Ensure mpc_buffer[t+1] exists (Safety Filter) --

        # alg21: P1 always runs. alg20 skipped the t+1 build whenever the robot
        # was in passthrough AND the RSOA at t+1 overlapped D, to avoid paying
        # mpc_cost for a build it assumed was doomed. That assumption came from
        # the same D-as-precondition reasoning dropped in scan_window, and it
        # cost more than it saved: skipping the build is what KEEPS passthrough
        # alive, because the only way to leave passthrough early is to find a
        # backup at t+1. Running P1 unconditionally can only shorten the stretch
        # flown without a backup in hand. The budget guard below still applies,
        # so the extra work is bounded rather than unconditional.

        # Wall at conflict_time. concrete_scan certified that RSOA(conflict_time)
        # intersects a raw obstacle, so at tau >= conflict_time the robot may
        # already be in collision and "a backup starting from tau" has no valid
        # answer. build_mpc_backup would not notice: it unpacks traj_bounds[1:],
        # dropping the launch bound before the raw check, so a maneuver departing
        # from inside an obstacle still reads VALID. Refuse to build instead.
        past_wall = conflict_time is not None and t_next >= conflict_time
        if past_wall and t_next not in mpc_reason:
            mpc_reason[t_next] = (f"conflict_wall(t+1={t_next} >= conflict_time="
                                  f"{conflict_time}; RSOA there is not raw-clear)")

        # Normal P1 operations
        if t_next not in mpc_buffer and not past_wall:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            # Two distinct signatures reach the no-backup branch. `buf[t+1]=absent
            # mpc_hz=-1` means the budget gate below refused the build -- seen
            # only on starved workers, so it is handled by running uncontended
            # rather than by overrunning the budget. `buf[t+1]=None` means the
            # build RAN and its plan was rejected -- the healthy-run case,
            # addressed by BOOTSTRAP_RESIM. Both read `td=never_set`: the buffer
            # is empty only at t=0, which is why every recorded event sits there.
            _have_fallback = (t_diverge is not None
                              and t_diverge > current_timestep
                              and psf_valid(mpc_buffer, t_diverge))
            _afford = budget.remaining >= budget.mpc_cost
            if t_next not in tester.horizons:
                p1_skipped_no_bounds += 1
            elif _afford:
                # No fallback ahead: this build is load-bearing, so pay for the
                # solver that converges rather than the one that is cheap.
                # Restored in `finally` -- an exception here must not leave the
                # filter permanently on the expensive path.
                _os_saved = mpc_sf.oneshot
                _resim_forced = (BOOTSTRAP_RESIM and not _have_fallback
                                 and _os_saved)
                if _resim_forced:
                    mpc_sf.oneshot = False
                    p1_bootstrap_resim += 1
                try:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next, mpc_reason)
                finally:
                    mpc_sf.oneshot = _os_saved
                # Counted in every arm, whatever solver ran: this is the rate
                # the solver choice is meant to move, and unlike psf_no_diverge
                # it is sampled on every trial rather than once per ~10k.
                if not _have_fallback and not valid:
                    p1_bootstrap_infeasible += 1
                # _have_fallback is false far more often than at t=0 -- every
                # time the robot catches up to t_diverge -- and most of those
                # failures are harmless (nominal passthrough covers them). Only
                # t=0 has a structurally empty buffer, so count it on its own.
                if current_timestep == 0 and not valid:
                    p1_t0_infeasible += 1
                mpc_horizon_until = max(mpc_horizon_until, t_next)
                if valid and (t_diverge is None or t_next > t_diverge):
                    t_diverge = t_next
                print(f"  [P1] mpc_buffer[{t_next}] = {'VALID' if valid else 'INFEASIBLE'}"
                      f"  t_diverge={t_diverge}"
                      f"{'  [RESIM: no fallback]' if _resim_forced else ''}")
            else:
                p1_skipped_no_budget += 1

        # ---- P2: Concrete scan forward (stop at conflict_time) -----
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(
                concrete_until + budget.max_affordable_concrete(),
                scan_ceil, MAX_TIME
            )
            collision, ct = concrete_scan(tester, concrete_until, end)
            if collision:
                conflict_time = ct
                print(f"  [P2] Conflict detected at t={conflict_time}")
                # A t_diverge established before this conflict was known may now
                # sit at or beyond it. Nominal is only concrete-certified for
                # t < conflict_time, so re-derive t_diverge under the new
                # ceiling. Neither the P1 wall nor P3/P3b's bounds can catch
                # this: t_diverge was already set when conflict_time moved.
                if t_diverge is not None and t_diverge >= conflict_time:
                    _stale = t_diverge
                    t_diverge = _max_valid_diverge(mpc_buffer, current_timestep,
                                                   below=conflict_time)
                    print(f"  [P2] t_diverge={_stale} is at/beyond the new conflict"
                          f" — clamped to {t_diverge}")
                if pending_job is None:
                    pending_job = RefinementTask(current_timestep, conflict_time)
                concrete_until = ct
                break
            concrete_until = end

        # ---- P3: Build MPC buffer forward toward conflict_time -------
        if conflict_time is not None:
            mpc_horizon_until = max(mpc_horizon_until, current_timestep)
            while (mpc_horizon_until < conflict_time - 1
                   and budget.remaining >= budget.mpc_cost):
                tau = mpc_horizon_until + 1
                if tau not in tester.horizons:
                    break
                if mpc_buffer.get(mpc_horizon_until) is None:
                    wall_tau = mpc_horizon_until
                    break
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau, mpc_reason)
                mpc_horizon_until = tau
                if valid:
                    t_diverge = tau
                    print(f"  [P3] mpc_buffer[{tau}] VALID  t_diverge → {t_diverge}")
                else:
                    print(f"  [P3] mpc_buffer[{tau}] INFEASIBLE — wall at tau={tau}")
                    wall_tau = tau
                    break

        #  P3b: Passthrough scan beyond infeasible wall
        # alg21: invoked after ANY wall. alg20 first required a D-region ahead
        # (some bound overlapping D but not the raw obstacle), which made a wall
        # with no D-region ahead unrecoverable even when a perfectly good backup
        # sat one tau further on -- an INFEASIBLE build has causes other than D
        # (path_raw, solver error, a vacuous heading bound), and none of them
        # imply the next tau is bad. scan_window is already budget-guarded and
        # aborts on the first raw collision, so the unconditional call is bounded.
        if wall_tau is not None and budget.remaining >= budget.mpc_cost:
            scan_limit = (min(conflict_time - 1, concrete_until)
                          if conflict_time is not None
                          else concrete_until)
            passthrough_td = scan_window(
                tester, mpc_sf, mpc_buffer,
                wall_tau, scan_limit, budget,
                reasons=mpc_reason)
            if passthrough_td is not None:
                t_diverge = passthrough_td
                mpc_horizon_until = passthrough_td
                passthrough_found += 1
                print(f"  [P3b] Passthrough t_diverge -> {t_diverge}")

        # ── P4: Symbolic. Still run even after confirmation
        if (pending_job is not None
                and budget.can_afford('symbolic', 1)):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    # Symbolic tightened bounds so drop INFEASIBLE entries so
                    # P3 retries them with tighter bounds (may extend t_diverge).
                    # VALID entries remain safe
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    # Pull mpc_horizon_until back to t_diverge so P3 re-scans the
                    # purged gap with tightened bounds on the next iteration.
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    # Reset job for continuous refinement
                    pending_job       = RefinementTask(current_timestep, conflict_time)
                else:
                    # Deconflicted -  clear conflict state but keep VALID buffer
                    # entries; they're still collision-free and serve as backups
                    # if a new conflict appears immediately after.
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION: nominal or activate MPC backup
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            # PSF satisfied —> nominal control is safe
            #
            # Record t_next as the divergence point. This branch is entered
            # *because* mpc_buffer[t_next] is valid, so t_next IS a certified
            # place to diverge; not writing it down is what lets the clock
            # overtake t_diverge. P1 normally raises it (same rule, same
            # place), but P1 is skipped whenever buffer[t_next] was already
            # built on an earlier timestep — and that is precisely when this
            # branch fires without it. Once current_timestep > t_diverge the
            # activation site computes ctrl_idx > 0 and joins a plan whose
            # first controls were never applied, so traj_bounds[ctrl_idx]
            # describes a trajectory the robot is not on.
            #
            # Only ever raises t_diverge, exactly as P1 does, so a pending
            # passthrough target further ahead is left alone. After the
            # increment below the invariant t_diverge >= current_timestep
            # holds, which makes ctrl_idx > 0 unreachable rather than merely
            # unobserved.
            if t_diverge is None or t_diverge < t_next:
                t_diverge = t_next
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

        elif (t_diverge is not None and t_diverge > current_timestep
                and psf_valid(mpc_buffer, t_diverge)):
            # Passthrough: no backup at t+1, but a valid one exists at t_diverge.
            #
            # Tripwire for the invariant this branch rests on. Flying nominal to
            # t_diverge is only certified because concrete_scan proved every step
            # below conflict_time is raw-clear. P1's wall, P3's loop bound, P3b's
            # scan_limit and P2's clamp each maintain it; none of them is checked
            # here, so a fifth path to t_diverge must fail loudly, not silently.
            assert conflict_time is None or t_diverge < conflict_time, (
                f"PASSTHROUGH gap [{current_timestep},{t_diverge}] spans the "
                f"conflict at t={conflict_time}: nominal is not concrete-certified "
                f"over this gap")
            #
            # Report the REAL reason there is no backup at t+1. alg20 had a
            # further arm here for the skip_p1 route ("inside D"); with P1 no
            # longer skipped on D membership, every no-backup case now reaches
            # this branch having actually attempted a build, so mpc_reason holds
            # the true cause (path_raw, term_D, solver_error, conflict_wall).
            if t_next in mpc_reason:
                _why = f"no backup at t+1: {mpc_reason[t_next]}"
            elif t_next not in mpc_buffer:
                _why = "no backup at t+1: never built (budget or bounds missing)"
            else:
                _why = "no backup at t+1: reason unrecorded"

            # State the basis for flying nominal this step. The guarantee is
            # concrete raw-clearance over the gap, NOT the D geometry.
            _basis = (f"nominal concrete-clear through t={concrete_until} (no conflict)"
                      if conflict_time is None
                      else f"nominal concrete-clear to conflict t={conflict_time}")

            print(f"  [PSF PASSTHROUGH] t={current_timestep} — {_why}; "
                  f"t_diverge={t_diverge} valid ahead; {_basis}"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            passthrough_used += 1
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

        else:
            # PSF fails — need MPC backup from t_diverge
            #
            # A backup is a plan from a specific state at a specific time:
            # controls[k] and traj_bounds[k] both assume controls[0..k-1] were
            # the inputs actually applied. Joining a plan at ctrl_idx > 0 means
            # those inputs were NOT applied — the filter flew nominal over
            # those steps — so traj_bounds[ctrl_idx] describes a state the
            # robot is not in, and every clearance check made against it is
            # void. t_diverge persists across loop iterations, so it can point
            # at a plan committed on an earlier timestep; that is a stale
            # pointer, not a backup.
            #
            # Prefer the plan committed at this exact timestep. If there isn't
            # one, refuse rather than fly a plan whose bounds do not describe
            # the robot: fall through to the no-backup branch, which at least
            # reports the condition instead of hiding it.
            if (t_diverge is not None and t_diverge != current_timestep
                    and psf_valid(mpc_buffer, current_timestep)):
                print(f"{RED}[PSF ACTIVATE] stale t_diverge={t_diverge} at"
                      f" t={current_timestep} (ctrl_idx would be"
                      f" {current_timestep - t_diverge}) — using the plan"
                      f" committed at t={current_timestep} instead{RESET}")
                t_diverge = current_timestep
            if t_diverge is not None and t_diverge != current_timestep:
                print(f"{RED}[PSF ACTIVATE] REFUSED stale backup:"
                      f" t_diverge={t_diverge} != t={current_timestep}"
                      f" (would join at ctrl_idx={current_timestep - t_diverge});"
                      f" no plan committed at t={current_timestep}"
                      f"  (seed={seed}){RESET}")
                psf_stale_refused += 1
                t_diverge = None

            if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
                controls, traj_bounds = mpc_buffer[t_diverge]
                mpc_state['committed_at']  = t_diverge
                mpc_state['conflict_time'] = conflict_time
                mpc_state['controls']      = list(controls)
                mpc_state['traj_bounds']   = list(traj_bounds)
                mpc_state['needed']        = True
                mpc_started                = True
                mpc_calls                 += 1

                ctrl_idx = current_timestep - t_diverge
                queue    = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    print(f"{RED}[PSF ACTIVATE] t={current_timestep}  PSF failed —"
                          f" backup from t_diverge={t_diverge}"
                          f"  ctrl_idx={ctrl_idx}  u={np.round(ctrl, 4)}{RESET}")
                    u_nn = _get_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    # ctrl_idx out of range — extend first
                    print(f"{RED}[PSF ACTIVATE] ctrl_idx={ctrl_idx} beyond queue"
                          f" ({len(queue)}) — extending{RESET}")
                    extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                        ctrl_idx=ctrl_idx, budget=budget)
                    queue = mpc_state['controls']
                    if ctrl_idx < len(queue):
                        ctrl = queue[ctrl_idx]
                        u_nn = _get_nn_control(tester, current_timestep)
                        u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                        tester.real_state_mpc(current_timestep, ctrl)
                        current_timestep += 1
                    else:
                        print(f"[PSF] Cannot activate — queue still empty. Nominal fallback."
                              f"  (seed={seed}, t={current_timestep})")
                        psf_queue_empty += 1
                        u_diffs.append(0.0)
                        tester.real_state_empirical(current_timestep, t_next)
                        current_timestep += 1
            else:
                # No valid backup yet — mpc_buffer[t+1] not ready (P1 stalled)
                # Apply nominal and warn; P1 will catch up next timestep.
                # SHOULD NOT RUN IF WORKING!!!!!
                _bnext   = tester.horizons[t_next].get_tight_bound() if t_next in tester.horizons else None
                _in_D    = (_bnext is not None and mpc_sf._collides_D(_bnext))
                _in_raw  = (_bnext is not None and mpc_sf._collides_raw(_bnext))
                _buf_t1  = mpc_buffer.get(t_next, 'absent')
                _td_state = (
                    'never_set'        if t_diverge is None
                    else f'stale({t_diverge}<=t)'   if t_diverge <= current_timestep
                    else f'invalid({t_diverge})'   if not psf_valid(mpc_buffer, t_diverge)
                    else f'ok({t_diverge})'
                )
                print(f"{RED}[PSF] No backup at t={current_timestep}"
                      f"  buf[{t_next}]={'None' if _buf_t1 is None else ('absent' if _buf_t1=='absent' else 'VALID')}"
                      f"  td={_td_state}"
                      f"  bnext_not_in_Sa={_in_D}  bnext_in_raw={_in_raw}"
                      f"  conflict={conflict_time}  mpc_hz={mpc_horizon_until}"
                      f"  (seed={seed}){RESET}")
                # TERMINAL SAFE CONTROLLER (Algorithm 1 branch 3): u = a.
                #
                # Admitted on the CURRENT bound, not _bnext: _bnext is the RSOA
                # under the nominal NN control, which this step does not apply.
                # If the current bound is in S_a, holding u = a from anywhere in
                # it is safe forever (S_a is forward-invariant under u = a); if
                # u = a was applied last step, the chain's entry certificate
                # still covers this step. See the module docstring.
                _adm, _latched = _term_ctrl_admissible(current_timestep)
                if TERM_CTRL and _adm:
                    print(f"{RED}[PSF TERM CTRL] t={current_timestep} — no backup;"
                          f" holding u={mpc_sf.term_accel:g}"
                          f"{' (latched)' if _latched else ' (current RSOA in S_a)'}"
                          f"  (seed={seed}){RESET}")
                    _apply_term_ctrl(current_timestep, 'no_backup', _latched)
                    # The active phase purges the future at the top of every
                    # step; the nominal phase never does, and every horizon past
                    # t+1 -- and every backup built from one -- was derived under
                    # NOMINAL control, which was not applied. real_state_mpc
                    # already replaced horizons[t+1]; drop the rest so P1/P2
                    # rebuild from the state actually flown instead of
                    # certifying against a trajectory the robot is not on.
                    for _t in list(tester.horizons.keys()):
                        if _t > t_next:
                            del tester.horizons[_t]
                    for _d in (mpc_buffer, mpc_reason):
                        for _t in list(_d.keys()):
                            if _t > current_timestep:
                                del _d[_t]
                    concrete_until    = t_next
                    mpc_horizon_until = min(mpc_horizon_until, current_timestep)
                    t_diverge         = None
                    pending_job       = (RefinementTask(t_next, conflict_time)
                                         if conflict_time is not None else None)
                    current_timestep += 1
                    continue

                if not _adm:
                    psf_unrecoverable += 1
                    if TERM_CTRL:
                        term_ctrl['refused'] += 1

                psf_no_diverge += 1
                _nd_key = ('never_built' if _buf_t1 == 'absent'
                           else 'rejected:' + str(mpc_reason.get(t_next, 'unrecorded')).split('(')[0]
                           if t_next in mpc_reason
                           else 'built_none' if _buf_t1 is None
                           else 'unrecorded')
                psf_no_diverge_reasons[_nd_key] = psf_no_diverge_reasons.get(_nd_key, 0) + 1
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1



    budget.close_timestep()
    if _budget_path:
        import json
        with open(_budget_path, 'w') as _f:
            json.dump({'seed': seed,
                       'timestep_budget': budget.timestep_budget,
                       'calibrated_costs': {
                           'symbolic': dict(budget.symbolic_costs),
                           'concrete': budget.concrete_cost,
                           'mpc':      budget.mpc_cost},
                       'ticks': budget.ledger}, _f)
        print(f"[BUDGET] wrote {len(budget.ledger)} timestep records to {_budget_path}")

    if _rsoa_path:
        import json
        with open(_rsoa_path, 'w') as _f:
            json.dump({'seed': seed, 'dt': tester.analyzer.cl_system.dynamics.dt,
                       'pos_min': pos_min, 'vel_min': vel_min, 'buffer': buffer,
                       'term_accel': mpc_sf.term_accel,
                       'steps': rsoa_log}, _f)
        print(f"[RSOA] wrote {len(rsoa_log)} step certificates to {_rsoa_path}")

    # ── Collect state history and check real-state collisions ─────────────
    state_history = []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(calc['real_state'].copy())
                break

    # Real-state violation AND certified-bound violation, as alg14_di reports:
    # a trial can stay safe in reality while the RSOA it was certified on did not.
    collision_timesteps = [i for i, s in enumerate(state_history)
                           if np.asarray(s).flatten()[0] <= pos_min
                           or np.asarray(s).flatten()[1] <= vel_min]
    rsoa_viol = [t for t in sorted(tester.horizons.keys())
                 if any('real_state' in c
                        for c in tester.horizons[t].calculations.values())
                 and tester.horizons[t].get_tight_bound() is not None
                 and violates_raw(tester.horizons[t].get_tight_bound(),
                                  pos_min, vel_min)]

    had_collision = len(collision_timesteps) > 0
    print(f"\n{'=' * 60}")
    print(f"Simulation complete at timestep {current_timestep}")
    if had_collision:
        print(f"[SAFETY] REAL-STATE VIOLATION at timesteps: {collision_timesteps}")
    else:
        print(f"[SAFETY] No real-state constraint violation detected")
    if rsoa_viol:
        print(f"[SAFETY] RSOA (certified bound) violation at timesteps: {rsoa_viol}")
    print(f"  mpc_calls={mpc_calls}  psf_no_diverge={psf_no_diverge}"
          f"  psf_queue_empty={psf_queue_empty}"
          f"  psf_stale_refused={psf_stale_refused}"
          f"  passthrough_found={passthrough_found}"
          f"  passthrough_used={passthrough_used}"
          f"  max_u_diff={max(u_diffs, default=0):.3f}")
    print(f"  p1_bootstrap_resim={p1_bootstrap_resim}"
          f"  p1_bootstrap_infeasible={p1_bootstrap_infeasible}"
          f"  p1_t0_infeasible={p1_t0_infeasible}"
          f"  term_ctrl(no_backup/tail/latched/refused)="
          f"{term_ctrl['no_backup']}/{term_ctrl['tail']}/"
          f"{term_ctrl['latched']}/{term_ctrl['refused']}")
    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update({
        'receding':                 int(receding),
        'oneshot':                  int(oneshot),
        'oneshot_refine':           oneshot_refine,
        'oneshot_mismatch':         float(getattr(mpc_sf, 'oneshot_model_mismatch', 0.0)),
        'oneshot_mismatch_acc':     float(getattr(mpc_sf, 'oneshot_mismatch_accepted', 0.0)),
        'oneshot_solves':           int(getattr(mpc_sf, 'oneshot_solves', 0)),
        'oneshot_bad_status':       int(getattr(mpc_sf, 'oneshot_bad_status', 0)),
        'first_step_cost':          int(first_step_cost),
        'tail_cost_weight':         tail_cost_weight,
        'mpc_resolved':             mpc_resolved,
        'mpc_resolve_failed':       mpc_resolve_failed,
        'mpc_fallback_uncertified': mpc_fallback_uncertified,
        'psf_tail':                 int(psf_tail),
        'mpc_tail_exhausted':       mpc_tail_exhausted,
        'single_start_frac':        SINGLE_START_COST_FRAC,
        'applied_u_flips':          applied_u_flips,
        'applied_u_pairs':          applied_u_pairs,
        'plan_flips':               plan_flips,
        'plan_pairs':               plan_pairs,
        # Dead-worker detector. mpc_cost_clamped==1 means this process's
        # once-per-process calibration exceeded the timestep budget and would
        # have disabled the filter outright before the guard existed;
        # mpc_skipped_no_budget counts every build the budget gate refused.
        # A trial with mpc_calls==0 and mpc_skipped_no_budget>0 was flown
        # unfiltered and must not be scored as a filter result.
        'mpc_cost_clamped':         mpc_cost_clamped,
        'mpc_skipped_no_budget':    mpc_skipped_no_budget,
        'mpc_build_yielded':        mpc_build_yielded_to_resolve,
        # An unfiltered step taken at a moment the filter had decided it needed
        # to intervene. Zero is the only acceptable value; the reason breakdown
        # is what says whether the cause is budget or the acceptance test.
        'psf_no_diverge_reasons':   dict(psf_no_diverge_reasons),
        'psf_stale_refused':        psf_stale_refused,
        'p1_skipped_no_budget':     p1_skipped_no_budget,
        'p1_skipped_no_bounds':     p1_skipped_no_bounds,
        'term_ctrl_no_backup':      term_ctrl['no_backup'],
        'term_ctrl_tail':           term_ctrl['tail'],
        'term_ctrl_latched':        term_ctrl['latched'],
        'term_ctrl_refused':        term_ctrl['refused'],
        'p1_bootstrap_resim':       p1_bootstrap_resim,
        'p1_bootstrap_infeasible':  p1_bootstrap_infeasible,
        'p1_t0_infeasible':         p1_t0_infeasible,
        'bootstrap_resim':          int(BOOTSTRAP_RESIM),
        'psf_unrecoverable':        psf_unrecoverable,
        'term_ctrl':                int(TERM_CTRL),
        'term_accel':               float(mpc_sf.term_accel),
        'rsoa_viol_steps':          len(rsoa_viol),
        'resolve_reserve':          int(RESOLVE_RESERVE),
        'mpc_cost_s':               float(budget.mpc_cost),
        # If first_step_cost turns the MPC into a nominal passthrough — the
        # optimizer parking omega_0 at omega_nom and deferring avoidance to the
        # free tail — then the applied control collapses onto the NN control and
        # these go to ~0 while MPC is still nominally "active".
        'u_diff_max':               float(max(u_diffs, default=0.0)),
        'u_diff_mean_active':       float(
            sum(d for d in u_diffs if d > 0.0)
            / max(sum(1 for d in u_diffs if d > 0.0), 1)),
        'u_diff_n_active':          int(sum(1 for d in u_diffs if d > 0.0)),
        **{f'fail_{k}': v for k, v in mpc_fail_reason.items()},
        **{f'rej_{k}': v for k, v in mpc_reject_detail.items()},
    })
    print(f"  [alg22_di] receding={'on' if receding else 'off'}"
          f"  mpc_resolved={mpc_resolved}"
          f"  mpc_resolve_failed={mpc_resolve_failed}"
          f"  mpc_fallback_uncertified={mpc_fallback_uncertified}"
          f"  mpc_tail_exhausted={mpc_tail_exhausted}"
          f"  fail_reasons={mpc_fail_reason or '{}'}"
          f"  rejects={mpc_reject_detail or '{}'}"
          f"  u_flips={applied_u_flips}/{applied_u_pairs}"
          f"  plan_flips={plan_flips}/{plan_pairs}"
          f"  mpc_skipped_no_budget={mpc_skipped_no_budget}"
          f"{'  [MPC COST CLAMPED]' if mpc_cost_clamped else ''}")
    return (state_history, had_collision, u_diffs, mpc_calls,
            psf_no_diverge, psf_queue_empty,
            passthrough_found, passthrough_used)

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=seed)
