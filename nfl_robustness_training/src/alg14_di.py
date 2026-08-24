"""
Algorithm 14 (DI) — Predictive Safety Filter with Forward MPC Buffer,
double-integrator / half-plane version of alg14_split_terminal.py.

Why this file exists
--------------------
alg14_split_terminal.py cannot run on the DoubleIntegrator: its geometry is
circle-based (obstacles [cx, cy, r], danger radius D = sqrt(r^2 + 2rR)),
which is meaningless in a (position, velocity) state space, and the DI branch
of make_mpc_safety_filter_acados was dead code.

What is shared vs. specialised
------------------------------
SHARED (imported verbatim from alg14_split_terminal — proof that the PSF core
is system-agnostic once the filter exposes the right two predicates):
    RefinementTask, concrete_scan, symbolic_step, extend_mpc_sequence,
    psf_valid, _purge_infeasible, _max_valid_diverge, _get_nn_control

SPECIALISED here:
    - the two geometry predicates  _collides_raw / _collides_D
    - the MPC backend (AcadosDIMPC instead of AcadosUnicycleMPC)
    - P3b passthrough is REMOVED (see below)

Split-terminal semantics for the DI
-----------------------------------
Constraints are half-planes:  p >= pos_min  and  v >= vel_min.

    PATH nodes (stages 1..N-1) must clear the RAW constraints.
    TERMINAL node (stage N) must lie in a CONTROL-INVARIANT set.

The invariant set is checked EXACTLY for the discrete system rather than via
the continuous parabola p >= v^2/(2*u_max): from the worst corner
(p_lo, v_lo) of the terminal RSOA box, apply u = +u_max until v >= 0, then
hold u = 0 forever. If p >= pos_min + buffer and v >= vel_min at every step
of that braking phase, the terminal admits an infinite-horizon safe
continuation (after v >= 0, u = 0 leaves v unchanged and p non-decreasing).

Worst-corner is sound because the DI is linear and monotone: p_k and v_k are
both non-decreasing in p_0 and v_0 under a fixed open-loop u, so the (min p,
min v) corner dominates the whole box.

Why there is no passthrough (P3b) here
--------------------------------------
For the unicycle, D is a BOUNDED region per obstacle and its complement is
where the max-turn loiter is invariant, so the robot can enter D and come out
— that is what P3b exploits. For the DI the unsafe set is the braking funnel
{v < 0, p < braking distance}, which is FORWARD-INVARIANT under every
admissible control: once inside, there is no control sequence that leaves it.
A passthrough scan can therefore never succeed, so it is removed rather than
left in to burn budget. This is the one place where the two systems differ
structurally, not just parametrically.

Usage:
    python nfl_robustness_training/src/alg14_di.py [seed]
"""

import os
import sys
import time
from typing import Optional

import numpy as np

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import (TimeBudget, calibrated_costs,
                         FALLBACK_SYMBOLIC_COSTS, FALLBACK_CONCRETE_COST,
                         FALLBACK_MPC_COST)
from mpc_safety_filter import MPCSafetyFilter
from utils.di_mpc_acados import AcadosDIMPC

# System-agnostic PSF machinery, imported unchanged from the unicycle algorithm.
# Imported from the CURRENT head of the unicycle lineage (alg20_vacuity),
# not alg14, so the DI variant inherits the shared fixes automatically.
# _max_valid_diverge gained a `below=` ceiling in alg19; taking it from alg14
# would silently give the old signature back.
from alg20_vacuity import (
    RefinementTask, concrete_scan, symbolic_step, extend_mpc_sequence,
    psf_valid, _purge_infeasible, _max_valid_diverge, _get_nn_control,
)
from alg13_di import HalfPlaneConstraints

RED, GREEN, BLUE, MAGENTA, CYAN, RESET = (
    "\033[31m", "\033[32m", "\033[34m", "\033[35m", "\033[36m", "\033[0m"
)


# ── Discrete-exact invariant-set test (the DI analogue of "outside D") ────

def brake_profile(p0, v0, dt, u_max, pos_min, vel_min, max_steps=500):
    """
    Apply u = +u_max from (p0, v0) until v >= 0. Returns (ok, min_p, min_v).

    ok is True iff p >= pos_min and v >= vel_min hold at EVERY step of the
    braking phase. Once v >= 0 the continuation u = 0 holds v constant and
    leaves p non-decreasing, so no further checking is needed.
    """
    p, v = float(p0), float(v0)
    min_p, min_v = p, v
    steps = 0
    while v < 0.0:
        p = p + dt * v + 0.5 * dt * dt * u_max
        v = v + dt * u_max
        min_p, min_v = min(min_p, p), min(min_v, v)
        steps += 1
        if steps > max_steps:
            return False, min_p, min_v
    ok = (min_p >= pos_min) and (min_v >= vel_min)
    return ok, min_p, min_v


def in_terminal_set(bounds, dt, u_max, pos_min, vel_min, buffer):
    """Worst-corner membership in the control-invariant terminal set."""
    ok, min_p, _ = brake_profile(bounds[0, 0], bounds[1, 0],
                                 dt, u_max, pos_min + buffer, vel_min)
    return ok


def violates_raw(bounds, pos_min, vel_min):
    """RAW half-plane violation of the RSOA box (matches HalfPlaneConstraints)."""
    return bool(bounds[0, 0] <= pos_min or bounds[1, 0] <= vel_min)


# ── DI safety filter ─────────────────────────────────────────────────────

class InputBoundViolation(Exception):
    """Every candidate rollout violated the OCP's own |u| <= u_max box."""


class NonFiniteBound(Exception):
    """RSOA box is not finite, so any certificate derived from it is void."""


class DIMPCSafetyFilterAcados(MPCSafetyFilter):
    """
    Split-terminal PSF backend for the double integrator.

    Exposes exactly the interface alg14's shared helpers require:
        n_horizon, _run_mpc_from_bounds(bounds, center, extra_inflation,
        multi_start), _collides_raw(bounds), _collides_D(bounds).
    """

    def __init__(self, tester, dt=0.2, n_horizon=12,
                 pos_min=0.0, vel_min=-1.0, buffer=0.05, u_max=1.0,
                 verify_input_bounds=True, verify_finite_bounds=True):
        super().__init__([], tester, max_lookback=10)
        self.dt        = dt
        self.n_horizon = n_horizon
        self.pos_min   = pos_min
        self.vel_min   = vel_min
        self.buffer    = buffer
        self.u_max     = u_max
        self.verify_input_bounds  = verify_input_bounds
        self.verify_finite_bounds = verify_finite_bounds

        print(f"  [split terminal DI] path: p>={pos_min:.3f}, v>={vel_min:.3f}"
              f"   terminal: brake-to-rest invariant (buffer={buffer:.3f})")

        self._mpc = AcadosDIMPC(
            dt=dt, n_horizon=n_horizon,
            pos_min=pos_min, vel_min=vel_min, buffer=buffer,
            solver_name=f'di_acados_sf_{os.getpid()}',
        )

    # -- the two predicates the shared PSF core is written against ---------

    def _collides_raw(self, bounds) -> bool:
        return violates_raw(bounds, self.pos_min, self.vel_min)

    def _collides_D(self, bounds) -> bool:
        """True when the bound is NOT in the control-invariant terminal set."""
        return not in_terminal_set(bounds, self.dt, self.u_max,
                                   self.pos_min, self.vel_min, self.buffer)

    # -- rollout ----------------------------------------------------------

    def _input_bounds_ok(self, controls, rejected):
        """
        True if every control in the rollout satisfies the box the OCP declares.

        Ported from alg19 / mpc_safety_filter_acados._input_bounds_ok. acados
        enforces lbu/ubu only when it CONVERGES; make_step returns the last
        iterate unclipped otherwise, and the rollout is then integrated with the
        true dynamics. For the DI this is not cosmetic: the terminal test is
        "brake at u = +u_max and stay clear", so a plan containing |u| > u_max
        is both unflyable and outside the authority that the invariant-set
        argument assumes.
        """
        if not self.verify_input_bounds:
            return True
        peak = max((abs(float(np.asarray(c).flatten()[0])) for c in controls),
                   default=0.0)
        if (not np.isfinite(peak)) or peak > self.u_max + 1e-6:
            rejected.append(peak)
            print(f"  [input_bound] candidate discarded: |u|={peak:.4f} "
                  f"> u_max={self.u_max:.4f}")
            return False
        return True

    def _run_mpc_from_bounds(self, initial_bounds, center,
                             extra_inflation: float = 0.0,
                             multi_start: bool = False):
        # DI counterpart of alg20's verify_heading_bounds. NOT the same check:
        # a heading interval is vacuous once it spans 2*pi because theta wraps,
        # whereas p and v do not wrap, so a WIDE DI bound is merely
        # conservative, not vacuous. The only way a DI box certifies nothing is
        # if it is non-finite, so that is what is refused here.
        if self.verify_finite_bounds and not np.all(np.isfinite(initial_bounds)):
            raise NonFiniteBound(
                f"non-finite RSOA box {np.asarray(initial_bounds).tolist()}")

        if multi_start:
            return self._run_mpc_multi_start(initial_bounds, center,
                                             extra_inflation)
        return self._run_mpc_single(initial_bounds, center, extra_inflation,
                                    warm_start_u=None)

    def _run_mpc_single(self, initial_bounds, center, extra_inflation,
                        warm_start_u):
        half_widths = (initial_bounds[:, 1] - initial_bounds[:, 0]) / 2.0
        hw_p, hw_v  = float(half_widths[0]), float(half_widths[1])

        # AcadosDIMPC takes the two half-widths SEPARATELY (p[0], p[1]) — it
        # does not take a single bounding-box diagonal the way the unicycle
        # backend does, because p and v are not commensurable.
        self._mpc._obs_inflation_buf[0] = hw_p + extra_inflation
        self._mpc._obs_inflation_buf[1] = hw_v

        x = np.asarray(center, dtype=float).reshape(-1, 1)
        self._mpc.x0 = x

        if warm_start_u is not None:
            self._mpc._nom_ctrl_buf[:] = float(warm_start_u)
        else:
            self._fill_nom_ctrl_buf(x.flatten())
        self._mpc.set_initial_guess()

        trajectory_bounds     = [initial_bounds.copy()]
        points, controls      = [], []
        prev_predicted_states = None
        cl_sys                = self.tester.analyzer.cl_system

        for _ in range(self.n_horizon):
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
            x_next = cl_sys.dynamics.dynamics_step(x.flatten().reshape(1, -1),
                                                   u_arr.reshape(1, -1))
            if hasattr(x_next, 'numpy'):
                x_next = x_next.numpy()
            x  = np.array(x_next).flatten().reshape(-1, 1)
            xf = x.flatten()

            trajectory_bounds.append(np.array([
                [xf[0] - hw_p, xf[0] + hw_p],
                [xf[1] - hw_v, xf[1] + hw_v],
            ]))
            points.append(x.copy())
            controls.append(u_arr.copy())

        return trajectory_bounds, points, controls

    def _score_rollout(self, traj_bounds):
        """Lower is better. Terminal shortfall against the invariant set."""
        b = traj_bounds[-1]
        ok, min_p, min_v = brake_profile(b[0, 0], b[1, 0], self.dt, self.u_max,
                                         self.pos_min + self.buffer, self.vel_min)
        if ok:
            return 0.0
        return (max(0.0, (self.pos_min + self.buffer) - min_p)
                + max(0.0, self.vel_min - min_v))

    def _run_mpc_multi_start(self, initial_bounds, center, extra_inflation):
        """
        NN warm-start first; if its terminal misses the invariant set, retry
        from a full-brake warm start. The DI analogue of the unicycle's
        heuristic swerve — there is only one direction that helps here.
        """
        rejected, candidates = [], []

        nn = self._run_mpc_single(initial_bounds, center, extra_inflation,
                                  warm_start_u=None)
        if self._input_bounds_ok(nn[2], rejected):
            nn_score = self._score_rollout(nn[0])
            if nn_score <= 1e-9:
                return nn
            candidates.append((nn_score, nn))

        brake = self._run_mpc_single(initial_bounds, center, extra_inflation,
                                     warm_start_u=self.u_max)
        if self._input_bounds_ok(brake[2], rejected):
            candidates.append((self._score_rollout(brake[0]), brake))

        # Rejecting per-candidate keeps a legal rollout when the other is bad.
        if not candidates:
            raise InputBoundViolation(
                f"input_bound: every rollout exceeded |u| <= {self.u_max:.4f} "
                f"(peaks={['%.4f' % r for r in rejected]}); the brake-to-rest "
                "terminal test assumes that authority")

        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]


# ── MPC backup build (split-terminal verdict, DI diagnostics) ─────────────

def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int,
                     reasons: dict = None) -> bool:
    """
    VALID iff every PATH node clears the RAW half-planes AND the TERMINAL node
    lies in the control-invariant (brake-to-rest) set.

    reasons: optional dict[tau -> str] recording WHY a build was rejected, so
    callers report the real cause instead of guessing (alg19). Kept out of
    mpc_buffer on purpose: psf_valid() is a presence test, so a non-None reason
    stored there would read as a valid backup.
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

        # traj_bounds[0] IS the RSOA at tau — the state the backup branches
        # from. alg14_split_terminal.py slices [1:] and never checks it, so a
        # backup can be certified VALID from a bound that already violates the
        # constraint; the PSF then reads buffer[t+1]==VALID and runs nominal
        # straight into it. Checking it here is required for
        # "the WHOLE RSOA must satisfy the constraint" to actually hold.
        if mpc_sf._collides_raw(traj_bounds[0]):
            b = traj_bounds[0]
            _lbl = (f"start_raw(p_lo={b[0,0]:+.4f} v_lo={b[1,0]:+.4f};"
                    f" RSOA at tau already violates the constraint)")
            print(f"  [MPC build] τ={tau}  INFEASIBLE({_lbl})")
            mpc_buffer[tau] = None
            if reasons is not None:
                reasons[tau] = _lbl
            return False

        *path_bounds, terminal_bound = traj_bounds[1:]

        path_hit = next(
            ((i, b) for i, b in enumerate(path_bounds, start=1)
             if mpc_sf._collides_raw(b)),
            None
        )
        term_hit = mpc_sf._collides_D(terminal_bound)
        collision = (path_hit is not None) or term_hit

        if collision:
            if path_hit is not None:
                idx, b = path_hit
                label = (f"path_raw[node={idx}/{len(path_bounds)}]"
                         f" p_lo={b[0,0]:+.4f}(min {mpc_sf.pos_min:+.3f})"
                         f" v_lo={b[1,0]:+.4f}(min {mpc_sf.vel_min:+.3f})")
            else:
                b = terminal_bound
                ok, min_p, min_v = brake_profile(
                    b[0, 0], b[1, 0], mpc_sf.dt, mpc_sf.u_max,
                    mpc_sf.pos_min + mpc_sf.buffer, mpc_sf.vel_min)
                label = (f"term_invariant p_lo={b[0,0]:+.4f} v_lo={b[1,0]:+.4f}"
                         f" -> brake min_p={min_p:+.4f}"
                         f" (need >= {mpc_sf.pos_min + mpc_sf.buffer:+.4f})"
                         f" min_v={min_v:+.4f}")
            status = f"INFEASIBLE({label})"
        else:
            status = "VALID"
        print(f"  [MPC build] τ={tau}  {status}  took {elapsed:.3f}s")

        if not collision:
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


# ── Simulation loop ──────────────────────────────────────────────────────

def test(seed=None, analyzer=None,
         pos_min: float = 0.0, vel_min: float = -1.0, buffer: float = 0.05,
         n_horizon: int = 12, MAX_TIME: int = 40, calibrate: bool = True,
         init_range=None):

    if analyzer is None:
        analyzer = setup_analyzer('DoubleIntegrator',
                                  'constraint_default_more_data_5hz',
                                  init_range=init_range)
    if seed is None:
        seed = 1401830092

    def _make_tester(s):
        t = ReachabilityTester(analyzer, [], seed=s)
        t.obstacles = HalfPlaneConstraints(pos_min=pos_min, vel_min=vel_min)
        return t

    tester = _make_tester(seed)
    dt     = float(analyzer.cl_system.dynamics.dt)
    u_max  = 1.0

    mpc_sf = DIMPCSafetyFilterAcados(
        tester, dt=dt, n_horizon=n_horizon,
        pos_min=pos_min, vel_min=vel_min, buffer=buffer, u_max=u_max)

    MAX_SYMBOLIC_HORIZON = 5

    budget = TimeBudget(timestep_budget=0.20)
    if calibrate:
        sym_costs, concrete_cost, mpc_cost = calibrated_costs(
            lambda: _make_tester(0),
            lambda t: (lambda: build_mpc_backup(mpc_sf, t, {}, 1)),
            MAX_SYMBOLIC_HORIZON,
        )
        budget.symbolic_costs = dict(sym_costs)
        budget.concrete_cost  = concrete_cost
        budget.mpc_cost       = mpc_cost
    else:
        # Static table. TimeBudget defaults concrete_cost to 0.0, which makes
        # max_affordable_concrete() divide by zero, so it must be set here.
        budget.symbolic_costs = dict(FALLBACK_SYMBOLIC_COSTS)
        budget.concrete_cost  = FALLBACK_CONCRETE_COST
        budget.mpc_cost       = FALLBACK_MPC_COST

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    mpc_buffer:        dict          = {}
    mpc_reason:        dict          = {}   # tau -> why the build at tau was rejected
    concrete_until:    int           = 0
    mpc_horizon_until: int           = -1
    conflict_time:     Optional[int] = None
    t_diverge:         Optional[int] = None
    pending_job:       Optional[RefinementTask] = None
    wall_tau:          Optional[int] = None

    mpc_state = {'committed_at': None, 'conflict_time': None,
                 'controls': [], 'traj_bounds': [], 'needed': False}
    mpc_started = False

    current_timestep = 0
    u_diffs, mpc_calls          = [], 0
    psf_no_diverge              = 0
    psf_queue_empty             = 0
    psf_stale_refused           = 0   # PSF failed, only a stale (ctrl_idx>0) plan existed
    aborted                     = False
    records                     = []   # per-timestep trace for plotting
    plan_log                    = {}   # committed_at -> traj_bounds

    def _log(t, mode, u=None, u_nn=None):
        h = tester.horizons.get(t)
        b = h.get_tight_bound() if h is not None else None
        st = None
        if h is not None:
            for c in h.calculations.values():
                if 'real_state' in c:
                    st = np.asarray(c['real_state']).flatten().copy()
                    break
        records.append({
            't': t, 'mode': mode,
            'state': st,
            'bounds': None if b is None else np.asarray(b).copy(),
            'u': None if u is None else float(np.asarray(u).flatten()[0]),
            'u_nn': None if u_nn is None else float(np.asarray(u_nn).flatten()[0]),
            't_diverge': t_diverge, 'conflict': conflict_time,
        })

    # ── RSOA certificate dump (off unless TTTCARV_RSOA_DUMP names a path) ──
    # Every executed timestep leaves through real_state_empirical or
    # real_state_mpc, so wrapping those two is the only placement that cannot
    # miss a branch. The record is taken *before* the step so it holds the
    # bounds as they stood when the step was authorized. (alg19)
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

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next   = current_timestep + 1
        wall_tau = None

        # ══════════════════ MPC ACTIVE ══════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}"
                  f"  conflict={conflict_time}  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
                    # INVARIANT (alg19): a certificate in mpc_buffer is only
                    # meaningful while the horizon it was derived from is live.
                    # The robot has deviated from the nominal plan, so every
                    # future horizon is stale — and so is every backup built
                    # from one. Fail-safe by construction: psf_valid is a
                    # presence test, so a missing entry reads as "no backup".
                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)
            # Leaving mpc_horizon_until past the purge point would make the
            # forward-fill start from a tau we just erased and read the hole as
            # an infeasible wall.
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until,
                                             mpc_reason)
                    mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                    if valid:
                        t_diverge = concrete_until
                    print(f"  [MPC PSF] τ={concrete_until} {'VALID' if valid else 'INFEASIBLE'}"
                          f"  t_diverge={t_diverge}")
                    if not valid:
                        break
                else:
                    break

            mpc_horizon_until = max(mpc_horizon_until, t_next)
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep} — reverting to nominal{RESET}")
                mpc_started = False
                mpc_state['needed'] = False
                mpc_state['committed_at'] = None
                mpc_state['conflict_time'] = None
                pending_job = (RefinementTask(t_next, conflict_time)
                               if conflict_time is not None else None)
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep,
                                                       below=conflict_time)
                if t_diverge is None or t_diverge < t_next:
                    # This branch is entered *because* psf_valid(mpc_buffer, t_next),
                    # and that backup was built this timestep from the concrete RSOA
                    # re-propagated after the maneuver. conflict_time was computed
                    # under the pre-maneuver trajectory and is stale here by
                    # construction. Letting a stale ceiling veto a just-verified
                    # backup leaves t_diverge=None and the filter then falls through
                    # to unconditional nominal for the rest of the run.
                    print(f"  [PSF REVERT] ceiling conflict_time={conflict_time}"
                          f" excluded the just-verified backup at τ={t_next}"
                          f" (would give t_diverge={t_diverge}) — using τ={t_next}")
                    t_diverge = t_next
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                _log(current_timestep, 'REVERT')
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1
                continue

            if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']

            if ctrl_idx >= len(queue):
                print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                aborted = True
                break

            ctrl = queue[ctrl_idx]
            print(f"{MAGENTA}[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                  f"  u={np.round(ctrl, 4)}"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s{RESET}")
            u_nn = _get_nn_control(tester, current_timestep)
            u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
            plan_log[mpc_state['committed_at']] = [np.asarray(b).copy()
                                                   for b in mpc_state['traj_bounds']]
            _log(current_timestep, 'MPC', u=ctrl, u_nn=u_nn)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
            continue

        # ══════════════════ NOMINAL / PRE_CONFLICT ══════════════════
        if conflict_time is not None and current_timestep > conflict_time:
            print(f"{GREEN}[STALE] conflict_time={conflict_time} in the past — clearing{RESET}")
            conflict_time = None
            pending_job   = None

        phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
        color = BLUE if phase == "PRE_CONFLICT" else GREEN
        print(f"\n{color}CURRENT TIMESTEP ==== {current_timestep}"
              f"  concrete_until={concrete_until}  mpc_hz={mpc_horizon_until}"
              f"  conflict={conflict_time}  t_diverge={t_diverge}  [{phase}]{RESET}")

        # -- P1: ensure mpc_buffer[t+1] exists --
        # No passthrough skip: the DI unsafe set is absorbing, so there is no
        # "inside D but will exit" state to wait out. P1 always runs.
        # Wall at conflict_time (alg19). concrete_scan certified that the RSOA
        # at conflict_time violates the raw constraint, so at tau >= conflict_time
        # "a backup starting from tau" has no valid answer. The start-bound check
        # inside build_mpc_backup would also catch it, but refusing here saves the
        # solve and records the reason.
        past_wall = conflict_time is not None and t_next >= conflict_time
        if past_wall and t_next not in mpc_reason:
            mpc_reason[t_next] = (f"conflict_wall(t+1={t_next} >= conflict_time="
                                  f"{conflict_time}; RSOA there is not raw-clear)")

        if t_next not in mpc_buffer and not past_wall:
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next, mpc_reason)
                mpc_horizon_until = max(mpc_horizon_until, t_next)
                if valid and (t_diverge is None or t_next > t_diverge):
                    t_diverge = t_next
                print(f"  [P1] mpc_buffer[{t_next}] = {'VALID' if valid else 'INFEASIBLE'}"
                      f"  t_diverge={t_diverge}")

        # -- P2: concrete scan forward --
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(concrete_until + budget.max_affordable_concrete(), scan_ceil, MAX_TIME)
            collision, ct = concrete_scan(tester, concrete_until, end)
            if collision:
                conflict_time = ct
                print(f"  [P2] Conflict detected at t={conflict_time}")
                # A t_diverge established before this conflict was known may now
                # sit at or beyond it. Nominal is only concrete-certified for
                # t < conflict_time, so re-derive under the new ceiling (alg19).
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

        # -- P3: build MPC buffer forward --
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

        # -- P3b: DELIBERATELY ABSENT (absorbing unsafe set; see module docstring) --
        if wall_tau is not None:
            print(f"  [P3b] skipped — DI unsafe set is forward-invariant,"
                  f" no passthrough possible past wall_tau={wall_tau}")

        # -- P4: symbolic --
        if pending_job is not None and budget.can_afford('symbolic', 1):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(tester, pending_job,
                                                MAX_SYMBOLIC_HORIZON, budget)
            if result is not None:
                if result["collision"]:
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    pending_job = RefinementTask(current_timestep, conflict_time)
                else:
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════ PSF DECISION ══════════════════
        if psf_valid(mpc_buffer, t_next):
            # Record t_next as the divergence point (alg19). This branch is
            # entered *because* mpc_buffer[t_next] is valid, so t_next IS a
            # certified place to diverge; not writing it down is what lets the
            # clock overtake t_diverge. P1 normally raises it, but P1 is skipped
            # whenever buffer[t_next] was already built on an earlier timestep —
            # exactly when this branch fires without it. Once
            # current_timestep > t_diverge the activation site computes
            # ctrl_idx > 0 and joins a plan whose first controls were never
            # applied, so traj_bounds[ctrl_idx] describes a trajectory the robot
            # is not on. Only ever raises t_diverge.
            if t_diverge is None or t_diverge < t_next:
                t_diverge = t_next
            u_diffs.append(0.0)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            _log(current_timestep, 'NOMINAL')
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

        else:
            # A backup is a plan from a specific state at a specific time:
            # controls[k] and traj_bounds[k] both assume controls[0..k-1] were
            # the inputs actually applied. Joining a plan at ctrl_idx > 0 means
            # those inputs were NOT applied — the filter flew nominal over those
            # steps — so traj_bounds[ctrl_idx] describes a state the robot is not
            # in, and every clearance check made against it is void. t_diverge
            # persists across iterations, so it can point at a plan committed on
            # an earlier timestep: a stale pointer, not a backup. (alg19)
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
                mpc_state.update({'committed_at': t_diverge,
                                  'conflict_time': conflict_time,
                                  'controls': list(controls),
                                  'traj_bounds': list(traj_bounds),
                                  'needed': True})
                mpc_started = True
                mpc_calls  += 1
                plan_log[t_diverge] = [np.asarray(b).copy() for b in traj_bounds]

                ctrl_idx = current_timestep - t_diverge
                queue    = mpc_state['controls']
                if ctrl_idx >= len(queue):
                    print(f"{RED}[PSF ACTIVATE] ctrl_idx={ctrl_idx} beyond queue"
                          f" ({len(queue)}) — extending{RESET}")
                    extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                        ctrl_idx=ctrl_idx, budget=budget)
                    queue = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    print(f"{RED}[PSF ACTIVATE] t={current_timestep} — backup from"
                          f" t_diverge={t_diverge}  ctrl_idx={ctrl_idx}"
                          f"  u={np.round(ctrl, 4)}{RESET}")
                    u_nn = _get_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                    _log(current_timestep, 'ACTIVATE', u=ctrl, u_nn=u_nn)
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    print(f"[PSF] Cannot activate — queue still empty. Nominal fallback."
                          f"  (seed={seed}, t={current_timestep})")
                    psf_queue_empty += 1
                    u_diffs.append(0.0)
                    _log(current_timestep, 'QUEUE_EMPTY')
                    tester.real_state_empirical(current_timestep, t_next)
                    current_timestep += 1
            else:
                # SHOULD NOT RUN IF WORKING
                _bnext = (tester.horizons[t_next].get_tight_bound()
                          if t_next in tester.horizons else None)
                _in_raw = (_bnext is not None
                           and violates_raw(_bnext, pos_min, vel_min))
                _in_D = (_bnext is not None
                         and not in_terminal_set(_bnext, dt, u_max,
                                                 pos_min, vel_min, buffer))
                print(f"{RED}[PSF] No backup at t={current_timestep}"
                      f"  td={t_diverge}  bnext_outside_invariant={_in_D}"
                      f"  bnext_in_raw={_in_raw}  conflict={conflict_time}"
                      f"  (seed={seed}){RESET}")
                psf_no_diverge += 1
                u_diffs.append(0.0)
                _log(current_timestep, 'NO_BACKUP')
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1

    if _rsoa_path:
        import json
        with open(_rsoa_path, 'w') as _f:
            json.dump({'seed': seed, 'dt': dt, 'u_max': u_max,
                       'pos_min': pos_min, 'vel_min': vel_min, 'buffer': buffer,
                       'steps': rsoa_log}, _f)
        print(f"[RSOA] wrote {len(rsoa_log)} step certificates to {_rsoa_path}")

    # ── Safety check on real states AND on certified RSOA bounds ──────────
    state_history, bound_history = [], []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(np.asarray(calc['real_state']).flatten().copy())
                bound_history.append(h.get_tight_bound())
                break

    real_viol = [i for i, s in enumerate(state_history)
                 if s[0] <= pos_min or s[1] <= vel_min]
    rsoa_viol = [i for i, b in enumerate(bound_history)
                 if b is not None and violates_raw(b, pos_min, vel_min)]

    print(f"\n{'=' * 60}")
    print(f"Simulation complete at timestep {current_timestep}"
          f"{' (ABORTED)' if aborted else ''}")
    if real_viol:
        print(f"[SAFETY] REAL-STATE VIOLATION at timesteps: {real_viol}")
    else:
        print(f"[SAFETY] No real-state constraint violation detected")
    if rsoa_viol:
        print(f"[SAFETY] RSOA (certified bound) violation at timesteps: {rsoa_viol}")
    else:
        print(f"[SAFETY] No RSOA violation detected")
    print(f"  mpc_calls={mpc_calls}  psf_no_diverge={psf_no_diverge}"
          f"  psf_queue_empty={psf_queue_empty}"
          f"  psf_stale_refused={psf_stale_refused}  aborted={aborted}"
          f"  max_u_diff={max(u_diffs, default=0):.3f}")

    return {
        'state_history': state_history,
        'bound_history': bound_history,
        'records': records,
        'plan_log': plan_log,
        'real_viol': real_viol,
        'rsoa_viol': rsoa_viol,
        'u_diffs': u_diffs,
        'mpc_calls': mpc_calls,
        'psf_no_diverge': psf_no_diverge,
        'psf_queue_empty': psf_queue_empty,
        'psf_stale_refused': psf_stale_refused,
        'mpc_reason': dict(mpc_reason),
        'aborted': aborted,
        'steps': current_timestep,
        'pos_min': pos_min, 'vel_min': vel_min, 'buffer': buffer,
        'dt': dt, 'u_max': u_max, 'seed': seed,
    }


def run_nominal(seed=None, analyzer=None,
                pos_min: float = 0.0, vel_min: float = -1.0,
                MAX_TIME: int = 40, init_range=None):
    """
    Unfiltered baseline: apply the NN nominal control at every timestep, with
    the SAME estimator / RSOA machinery as test(), and record where the real
    state and the certified bound violate the half-planes.

    This is the apples-to-apples control for the PSF run — identical closed-loop
    re-anchoring of the RSOA at each step, only the safety filter is removed.
    """
    if analyzer is None:
        analyzer = setup_analyzer('DoubleIntegrator',
                                  'constraint_default_more_data_5hz',
                                  init_range=init_range)
    if seed is None:
        seed = 1401830092

    tester = ReachabilityTester(analyzer, [], seed=seed)
    tester.obstacles = HalfPlaneConstraints(pos_min=pos_min, vel_min=vel_min)

    for t in range(MAX_TIME):
        tester.real_state_empirical(t, t + 1)

    state_history, bound_history = [], []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(np.asarray(calc['real_state']).flatten().copy())
                bound_history.append(h.get_tight_bound())
                break

    real_viol = [i for i, s in enumerate(state_history)
                 if s[0] <= pos_min or s[1] <= vel_min]
    rsoa_viol = [i for i, b in enumerate(bound_history)
                 if b is not None and violates_raw(b, pos_min, vel_min)]
    print(f"[NOMINAL] real_viol={real_viol}  rsoa_viol={rsoa_viol}")
    return {'state_history': state_history, 'bound_history': bound_history,
            'real_viol': real_viol, 'rsoa_viol': rsoa_viol,
            'pos_min': pos_min, 'vel_min': vel_min, 'seed': seed,
            'steps': len(state_history)}


if __name__ == "__main__":
    _seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=_seed)
