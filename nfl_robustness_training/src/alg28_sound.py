"""
Reach_PSF_FINAL -- reachability-based predictive safety filter (PSF) for the
unicycle with a neural-network (NN) controller.

Every 0.2 s timestep the filter decides whether the NN's control is safe to
apply. It is safe when a verified MPC backup plan exists from the next state
(mpc_buffer[t+1]): the robot can then always switch to that plan later.

Reachable sets (RSOA) come in two kinds:
  concrete  cheap, looser bounds, propagated one step at a time
  symbolic  expensive, tighter bounds, computed in chunks that may take
            several timesteps of budget

States:
  NOMINAL       no conflict ahead; keep a backup ready at t+1.
  PRE_CONFLICT  the concrete RSOA hits an obstacle at conflict_time. Build
                backups forward toward it and try to rule it out symbolically.
  MPC_ACTIVE    no backup at t+1, so the backup plan flies the robot
                (re-solved every step) until a backup at t+1 exists again.

Work each timestep, in priority order, each only if the budget allows:
  P1   make sure mpc_buffer[t+1] exists (the safety check itself)
  P2   concrete scan forward, up to conflict_time or MAX_TIME
  P3   build backups forward toward conflict_time - 1
  P3b  if P3 hit an infeasible build (a "wall"), look past it for a valid one
  P4   symbolic refinement of the conflict

A backup is accepted when every intermediate node clears the raw obstacle
radius r and the terminal node clears the danger radius D = sqrt(r^2 + 2rR),
from where a max-turn loiter can keep clear of the obstacle.

Obstacles are [center_x, center_y, radius].

Usage (from the repo root):
    python nfl_robustness_training/src/Reach_PSF_FINAL_NL.py <seed> x,y,r ...

alg26b: alg26 (sound set-membership box under bounded noise) PLUS removal
of the estimator's reset-to-truth, so no stage assumes perfect state info.
Original alg26 note: the EKF box is replaced by a SOUND set-membership box under
BOUNDED measurement noise. See _install_setmem below.

alg27: alg26v, but the NN controller reads the ESTIMATE (centre of the
certified box) instead of the true state. See _install_estctrl below.
  TTTCARV_ESTCTRL=A  certificate step sound, lookahead unchanged (diagnostic)
  TTTCARV_ESTCTRL=B  lookahead also sound (NN bounded over the box), P4 off
  TTTCARV_ESTCTRL=0  identical to alg26v
  TTTCARV_ESTCTRL=J  alg28: lookahead by a JOINT one-step CROWN bound over
                     (x, e), split into cells (_install_estctrl, joint_step)

alg28: alg27 with every link of the certificate made sound and audited.
  defaults   ESTCTRL=B (NN reads the certified-box centre; lookahead bounds the
             NN over the whole box), KF_RESET=0 (no reset to the true state)
  new        backup-plan boxes by interval propagation instead of constant-size
             boxes, and a per-stage OCP margin for the heading fan
             (_install_sound_mpc; TTTCARV_SOUND_MPC=0 restores alg27 B)
  noise      TTTCARV_NOISE=uniform (default, |v| <= VMAX_SIGMA*sigma, a hard
             bound) or gauss (the simulator's Gaussian; the bound then only
             holds with probability, and truth_out_meas counts the misses)
  audits     snd_audit_miss (plan boxes vs rollouts of the plan's controls),
             snd_node_truth_out (true state vs the flown plan's node box),
             setmem_truth_out_post (true state vs every certified bound);
             all must be 0 under uniform noise.
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget, calibrated_costs
import numpy as np
import os
import time
from typing import Optional
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter
import mpc_safety_filter_acados as _msfa

RED     = "\033[31m"
GREEN   = "\033[32m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
RESET   = "\033[0m"


# ── Configuration ──────────────────────────────────────────────────────────
# Settings come from TTTCARV_* environment variables. Not every default is the
# final configuration: the final run (results/alg22/alg22_results.md) sets
# TTTCARV_ONESHOT=1, TTTCARV_ONESHOT_RTI=10, TTTCARV_MPC_PRICING=1,
# TTTCARV_BUDGET_MARGIN=0.003, TTTCARV_PSF_TAIL=1, TTTCARV_ONESHOT_VERIFY=1 and
# TTTCARV_SINGLE_START_FRAC=0.50.

# Extra per-trial stats. test() returns a fixed tuple shared with the other
# algorithms, so everything else goes here; the trial runner merges it into the
# results row after each trial.
LAST_RUN_STATS: dict = {}

# Share of mpc_cost charged for a single-start re-solve when pricing is OFF.
# mpc_cost is calibrated on a multi-start build (1-2 solves), so the true share
# is 0.5-1.0 and 0.25 under-reserves. Ignored with TTTCARV_MPC_PRICING=1, which
# charges every solve its measured price.
SINGLE_START_COST_FRAC = float(os.environ.get('TTTCARV_SINGLE_START_FRAC', '0.25'))

# Hold back budget for the active-phase re-solve so the rebuild loop cannot
# spend it. Off by default: it measured as having no effect, because the loop's
# first build is the revert test, which either reverts (so no re-solve is
# needed) or fails and ends the loop.
RESOLVE_RESERVE = os.environ.get('TTTCARV_RESOLVE_RESERVE', '0') != '0'

# Last resort when there is no backup at all: a max-turn loiter instead of the
# raw NN control, used only if the current RSOA clears D. One step moves at most
# v*dt = 0.2 and D - 0.2 = 0.918 > r = 0.5, so that step cannot collide whatever
# the control; the filter re-plans on the next step. Repeated loiter steps are
# not covered by this argument.
PSF_LOITER = os.environ.get('TTTCARV_PSF_LOITER', '1') != '0'

# When there is no earlier backup to fall back on (e.g. t=0, empty buffer),
# build the t+1 backup with the slower re-simulation solver instead of the
# one-shot solver. The one-shot solver is cheaper but sometimes stops short and
# its plan ends inside D. Elsewhere that rejection is covered by an existing
# backup; here it would leave none. No effect when TTTCARV_ONESHOT=0.
BOOTSTRAP_RESIM = os.environ.get('TTTCARV_BOOTSTRAP_RESIM', '1') != '0'


def _cos_range(lo, hi):
    """[min, max] of cos over [lo, hi] radians, including any +-1 turning point."""
    if hi - lo >= 2.0 * np.pi:
        return -1.0, 1.0
    vals = [float(np.cos(lo)), float(np.cos(hi))]
    if 2.0 * np.pi * np.ceil(lo / (2.0 * np.pi)) <= hi:
        vals.append(1.0)
    if np.pi + 2.0 * np.pi * np.ceil((lo - np.pi) / (2.0 * np.pi)) <= hi:
        vals.append(-1.0)
    return min(vals), max(vals)


def _sin_range(lo, hi):
    return _cos_range(lo - np.pi / 2.0, hi - np.pi / 2.0)


def _interval_step(b, u, dt, v):
    """One-step interval enclosure of the unicycle under a KNOWN control u.

    Duplicated from alg25_nokf rather than imported: that file is hash-frozen
    for a running job, and these are 20 self-contained lines.
    """
    b = np.asarray(b, dtype=float)
    c_lo, c_hi = _cos_range(float(b[2, 0]), float(b[2, 1]))
    s_lo, s_hi = _sin_range(float(b[2, 0]), float(b[2, 1]))
    out = np.empty_like(b)
    out[0, 0], out[0, 1] = b[0, 0] + dt * v * c_lo, b[0, 1] + dt * v * c_hi
    out[1, 0], out[1, 1] = b[1, 0] + dt * v * s_lo, b[1, 1] + dt * v * s_hi
    out[2, 0], out[2, 1] = b[2, 0] + dt * u, b[2, 1] + dt * u
    return out


# ── alg26: SOUND set-membership state estimation ──────────────────────────────
# The base algorithm files the EKF's 3-sigma box as a certified bound. A
# Gaussian confidence region is not a containment guarantee -- alg24 measured
# the true state outside the intersected bound in 2491/2500 trials -- and
# alg25 showed that simply deleting it is not the answer either: with no
# measurement at all the forward reachable set runs free from X0 and reaches
# ~7.9 width, so the MPC can no longer certify a backup.
#
# alg26 keeps the measurement and keeps the architecture. It changes only what
# the certificate is allowed to use:
#
#   noise      bounded, |v| <= VMAX, drawn uniform instead of Gaussian
#   meas box   [z - VMAX, z + VMAX], which CONTAINS the true state by
#              construction, since z = x_true + v and |v| <= VMAX
#   certified  (propagated reachable set) INTERSECT (meas box)
#
# Both operands provably contain the truth, so the intersection does too, and
# it can never be empty. The EKF is left running untouched for the POINT
# estimate the MPC plans from -- that is what it is good at. It just no longer
# supplies the box.
#
# Measurement is full-state with identity H (REAL_integrated_sim.py:557,
# "noisy_measurement = true_state_np + measurement_noise"), so the consistent
# set is exactly a box. No linearization, no relaxation, and cheaper than the
# EKF update it sits beside.
SETMEM = os.environ.get('TTTCARV_SETMEM', '1') != '0'
# alg26b: also remove the estimator's reset-to-truth. REAL_integrated_sim.py
# calls estimator.reset(parent_real_state, ...) at :513 and :676, re-anchoring
# the EKF at the TRUE state before every executed step, so the filter is handed
# ground truth once per timestep and its error never accumulates past one step.
# That is the "perfect state information" the paper argues against. With
# KF_RESET=0 the EKF carries its own estimate forward instead.
#
# This is safe here in a way it was NOT in alg24: the certificate no longer
# comes from the EKF, so a drifting estimate can no longer produce an unsound
# or empty bound. alg24 crashed with ReachableSetBoundsError from exactly that.
KF_RESET = os.environ.get('TTTCARV_KF_RESET', '0') != '0'   # alg28: default OFF
VMAX_SIGMA = float(os.environ.get('TTTCARV_VMAX_SIGMA', '3.0'))
NOISE = os.environ.get('TTTCARV_NOISE', 'uniform').strip().lower()
if NOISE not in ('uniform', 'gauss', 'trunc'):
    raise ValueError(f"TTTCARV_NOISE must be uniform, gauss or trunc, got {NOISE!r}")
TRUNC_RATIO = float(os.environ.get('TTTCARV_TRUNC_RATIO', '3.0'))
# alg29: TTTCARV_MEAS_DELAY=1 -- the PIPELINED deployment. Timestep t may only
# use the measurement z_{t-1}, so its whole computation (<= the 0.20 s budget)
# can run while the plant flies step t-1 -> t, and u_t is applied exactly at t
# with no actuation latency. The box filed at t is F(S_{t-1} ∩ (z_{t-1} ± vmax),
# u_{t-1}) instead of z_t ± vmax: both operands of the intersection contain
# x_{t-1} and u_{t-1} is the known applied control, so it contains x_t. The
# certified set is then no longer inside z_t ± vmax, so mode J's bound on
# |x - centre| widens to hw of that box: vmax in theta and vmax + dt*v*vmax in
# x, y (|d cos/d theta| <= 1 over a heading interval of half-width vmax).
MEAS_DELAY = os.environ.get('TTTCARV_MEAS_DELAY', '0') != '0'
if TRUNC_RATIO <= 0:
    raise ValueError(f"TTTCARV_TRUNC_RATIO must be > 0, got {TRUNC_RATIO!r}")


class _BoundedRNG:
    """Delegates to the real RNG but replaces .normal() with a BOUNDED draw.

    Wrapping the RNG rather than editing REAL_integrated_sim keeps the frozen
    simulator untouched and catches both measurement sites (:557 and :698).
    Every other RNG use -- the initial-state and sampling uniforms at :195 and
    :427 -- passes straight through, so seeds still line up with the baselines
    for everything except the noise draw itself.
    """

    def __init__(self, rng, vmax):
        self._rng, self._vmax = rng, vmax
        self.last = None

    def normal(self, loc, scale, size=None):
        # alg28: TTTCARV_NOISE=gauss keeps the simulator's Gaussian noise, so
        # |v| <= vmax only holds with probability (union bound, per rollout);
        # truth_out_meas then counts the draws that broke it.
        if NOISE == 'gauss':
            self.last = self._rng.normal(loc, scale, size=size)
        elif NOISE == 'trunc':
            # N(loc, scale) CONDITIONED on |v| <= vmax, by inverse CDF. The
            # bound the certificate assumes is satisfied exactly as under
            # 'uniform' -- only the shape inside it changes. This separates
            # "the method works because the noise is bounded" from "the method
            # works because uniform noise happens to shrink the certified set
            # fastest": what cuts the set is the OFFSET between consecutive
            # measurement boxes, and a peaked law makes consecutive boxes
            # nearly coincide, so it is the pessimistic case for set-membership.
            #
            # Exactly ONE rng.uniform draw of the same shape as the 'uniform'
            # path, so this arm stays seed-aligned with it.
            # TTTCARV_TRUNC_RATIO r: the internal sigma is vmax/r, i.e. the
            # normal is truncated at r sigma. vmax is UNCHANGED, so every arm
            # satisfies the identical bound and only the shape inside it moves;
            # r -> 0 approaches uniform. Varying the simulator's sigma instead
            # would rescale vmax and change the measurement box width, which
            # moves the certified set directly and confounds the comparison.
            # r = 3 reproduces sigma = 0.05 at vmax = 0.15 exactly.
            from scipy.special import ndtr, ndtri
            sig = self._vmax / TRUNC_RATIO
            a = ndtr((-self._vmax - loc) / sig)
            b = ndtr((self._vmax - loc) / sig)
            u = self._rng.uniform(a, b, size=size)
            v = loc + sig * ndtri(u)
            # ndtri can return +-inf at the very ends in float; clip to the bound.
            self.last = np.clip(v, -self._vmax, self._vmax)
        else:
            self.last = self._rng.uniform(-self._vmax, self._vmax, size=size)
        return self.last

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _install_setmem(tester):
    """Swap the EKF box for a sound set-membership box. Returns a stats dict."""
    stats = {'steps': 0, 'truth_out_tight': 0, 'truth_out_meas': 0, 'kf_reset': 1,
             'meas_substituted': 0, 'mpc_interval_steps': 0, 'empty_intersect': 0,
             'no_parent_bound': 0, 'tight_width_max': 0.0, 'meas_width': 0.0,
             'empty_repaired': 0, 'empty_unrepairable': 0,
             'mb_stash': 0, 'mb_fallback': 0,
             'post_add': 0, 'post_set': 0, 'truth_out_post': 0,
             'meas_delay': int(MEAS_DELAY), 'delay_first': 0, 'delay_empty': 0}
    if not SETMEM:
        return stats
    import torch
    from REAL_integrated_sim import CalculationType

    dyn = tester.analyzer.cl_system.dynamics
    dt, v = float(dyn.dt), float(dyn.vt)
    vmax = VMAX_SIGMA * float(tester.measurement_noise_std)
    stats['meas_width'] = 2.0 * vmax

    # Must run AFTER __init__ has filed X0, which is itself an EMPIRICAL calc
    # ('Initial set') and must NOT be replaced by a measurement box.
    assert 0 in tester.horizons and tester.horizons[0].get_tight_bound() is not None, \
        '_install_setmem must run after ReachabilityTester.__init__'

    brng = _BoundedRNG(tester._rng, vmax)
    tester._rng = brng
    # MEAS_DELAY: raw box of the previous measurement, and the control/bound of
    # the step being filed (set by the step wrappers below before they run).
    delay = {'prev_raw': None, 'u': None, 'S_start': None}

    def _delayed_box(raw):
        u, S0, zb = delay['u'], delay['S_start'], delay['prev_raw']
        delay['prev_raw'], delay['u'], delay['S_start'] = raw, None, None
        assert u is not None and S0 is not None, 'MEAS_DELAY: step wrapper did not run'
        base = np.asarray(S0, dtype=float)
        if zb is None:
            stats['delay_first'] += 1           # t = 1: no earlier measurement
        else:
            nb = np.stack([np.maximum(base[:, 0], zb[:, 0]),
                           np.minimum(base[:, 1], zb[:, 1])], axis=1)
            if np.any(nb[:, 0] > nb[:, 1]):
                stats['delay_empty'] += 1       # cannot happen; both hold x_{t-1}
                nb = zb
            base = nb
        return _interval_step(base, u, dt, v)

    if not KF_RESET:
        est = tester.estimator
        est.reset = lambda state, bounds: None          # instance-level no-op
        # The simulator seeds the EKF at the sampled TRUE initial state and only
        # reset() ever sets .state, so without this the estimator would still
        # start from perfect information. Start at the centre of X0 instead.
        b0 = np.asarray(tester.horizons[0].get_tight_bound(), dtype=float)
        est.x = (b0[:, 0] + b0[:, 1]) / 2.0
        est.state, est.bounds = est.x.copy(), b0.copy()
        stats['kf_reset'] = 0
    else:
        stats['kf_reset'] = 1

    horizon_cls = type(tester.horizons[0])
    if not hasattr(horizon_cls, '_setmem_orig_add'):
        horizon_cls._setmem_orig_add = horizon_cls.add_calculation
    orig_add = horizon_cls._setmem_orig_add
    orig_emp, orig_mpc = tester.real_state_empirical, tester.real_state_mpc

    def _probe(h, key):
        # alg26v AUDIT. _check() runs once per executed step, right after the
        # measurement box is filed. That leaves a hole: anything filed into the
        # same horizon LATER (a symbolic refinement, a re-scan) intersects into
        # tight_bound after the check has passed, and could cut the true state
        # out of a bound the filter then goes on to use. This probe re-tests
        # containment after EVERY mutation of tight_bound. tight_bound is only
        # written in three places -- add_calculation (:123-128), the MPC
        # overwrite (:739), and _set_bound here -- and the first and third both
        # call this, while :739 is always followed by _set_bound in
        # real_state_mpc. So truth_out_post == 0 means containment held at every
        # instant of the trial, not merely at the moments _check sampled it.
        if h is None or h.tight_bound is None:
            return
        truth = None
        for calc in h.calculations.values():
            if 'real_state' in calc:
                truth = np.asarray(calc['real_state'], dtype=float).flatten()
                break
        if truth is None:
            return
        stats[key] += 1
        b = h.tight_bound
        if np.any(truth < b[:, 0] - 1e-9) or np.any(truth > b[:, 1] + 1e-9):
            stats['truth_out_post'] += 1

    def _set_bound(h, b):
        h.tight_bound = None if b is None else np.asarray(b, dtype=float).copy()
        if h.tight_bound is not None:
            h.reachable_set.set_range(torch.tensor(h.tight_bound,
                                                   dtype=torch.float32,
                                                   device=h.device))
        _probe(h, 'post_set')

    def _meas_box(real_state):
        """[z - vmax, z + vmax] for the draw just consumed. Contains real_state."""
        if brng.last is None:
            return None
        z = np.asarray(real_state, dtype=float).flatten() + np.asarray(brng.last,
                                                                      dtype=float).flatten()
        brng.last = None                      # consume, so X0 is never substituted
        return np.stack([z - vmax, z + vmax], axis=1)

    def add_calculation(self, *a, **kw):
        """Substitute the measurement box for the EKF box on measurement steps."""
        calc_type = kw.get('calc_type', a[1] if len(a) > 1 else None)
        rs = kw.get('real_state')
        if (calc_type == CalculationType.EMPIRICAL and rs is not None
                and 'bounds' in kw and kw.get('notes', '') != 'Initial set'):
            mb = _meas_box(rs)
            if mb is not None and MEAS_DELAY:
                mb = _delayed_box(mb)
            if mb is not None:
                kw['bounds'] = mb
                self._setmem_mb = mb.copy()   # kept for the repair below
                stats['meas_substituted'] += 1
                if np.any(np.asarray(rs).flatten() < mb[:, 0] - 1e-12) or \
                   np.any(np.asarray(rs).flatten() > mb[:, 1] + 1e-12):
                    stats['truth_out_meas'] += 1      # must stay 0
        before = None if self.tight_bound is None else self.tight_bound.copy()
        out = orig_add(self, *a, **kw)
        if before is not None and self.tight_bound is not None and \
                np.any(self.tight_bound[:, 0] > self.tight_bound[:, 1]):
            stats['empty_intersect'] += 1
            # REPAIR. The base class prints the empty intersection and then
            # still calls set_range with the inverted box (REAL_integrated_sim
            # :132-136), so the next concrete() propagates from a box with
            # lower > upper and auto_LiRPA aborts the whole trial -- that is
            # the 9/2500 ReachableSetBoundsError in alg26b. The measurement box
            # contains the true state by construction, so falling back to it is
            # sound and keeps propagation anchored. real_state_mpc already does
            # exactly this; this closes the same gap on the empirical path.
            mb_prev = getattr(self, '_setmem_mb', None)
            if mb_prev is not None:
                _set_bound(self, np.asarray(mb_prev, dtype=float).copy())
                stats['empty_repaired'] += 1
            else:
                stats['empty_unrepairable'] += 1
        _probe(self, 'post_add')
        return out
    horizon_cls.add_calculation = add_calculation

    def _check(t):
        h = tester.horizons.get(t)
        if h is None or h.get_tight_bound() is None:
            return
        b = h.get_tight_bound()
        truth = None
        for calc in h.calculations.values():
            if 'real_state' in calc:
                truth = np.asarray(calc['real_state'], dtype=float).flatten()
                break
        if truth is None:
            return
        stats['steps'] += 1
        if np.any(truth < b[:, 0] - 1e-9) or np.any(truth > b[:, 1] + 1e-9):
            stats['truth_out_tight'] += 1
        stats['tight_width_max'] = max(stats['tight_width_max'],
                                       float(np.max(b[:, 1] - b[:, 0])))

    def real_state_mpc(start, control):
        # real_state_mpc REPLACES horizons[end] with a fresh horizon and then
        # overwrites tight_bound with the estimator box (REAL_integrated_sim.py
        # :726, :739), so nothing is intersected there. Rebuild that bound as
        # propagated-under-u INTERSECT measurement, which is sound.
        b_prev = (tester.horizons[start].get_tight_bound()
                  if start in tester.horizons else None)
        if MEAS_DELAY:
            delay['u'] = float(np.asarray(control, dtype=float).flatten()[0])
            delay['S_start'] = None if b_prev is None else np.asarray(b_prev, dtype=float).copy()
        res = orig_mpc(start, control)
        h = tester.horizons.get(start + 1)
        if h is not None:
            # alg26d FIX. alg26c read the measurement box back out of
            # get_tight_bound(), but REAL_integrated_sim :739 assigns
            # `self.horizons[end].tight_bound = stored_bounds.copy()` AFTER
            # add_calculation has run, and :713 sets `stored_bounds = kf_bounds`.
            # So the read-back returned the raw EKF 3-sigma box, not the
            # measurement box, and the MPC bound was rebuilt as
            # propagated INTERSECT ekf_box. With KF_RESET=1 the estimator is
            # re-anchored to the true state every step, so that box happens to
            # contain the truth and the error is invisible; with KF_RESET=0 the
            # estimator free-runs and drifts, the intersection cuts the truth
            # out, and the certificate stops describing the real state. That is
            # the entire truth_out_tight signal in alg26b/alg26c: 1482 misses,
            # every one of them in a trial that used MPC, none in the 2093
            # trials that never left NOMINAL. Take the box add_calculation
            # actually substituted instead; it contains the truth by
            # construction, so the intersection cannot exclude it or be empty.
            mb = getattr(h, '_setmem_mb', None)
            if mb is not None:
                mb = np.asarray(mb, dtype=float).copy()
                stats['mb_stash'] += 1
            else:
                mb = h.get_tight_bound()
                stats['mb_fallback'] += 1
            if b_prev is None:
                stats['no_parent_bound'] += 1
            else:
                u = float(np.asarray(control, dtype=float).flatten()[0])
                prop = _interval_step(b_prev, u, dt, v)
                stats['mpc_interval_steps'] += 1
                if mb is not None:
                    nb = np.stack([np.maximum(prop[:, 0], mb[:, 0]),
                                   np.minimum(prop[:, 1], mb[:, 1])], axis=1)
                    if np.any(nb[:, 0] > nb[:, 1]):
                        stats['empty_intersect'] += 1
                        nb = mb               # both contain truth; never take an
                                              # inverted box forward
                    _set_bound(h, nb)
                else:
                    _set_bound(h, prop)
        _check(start + 1)
        return res

    def real_state_empirical(start, end):
        # No rebuild needed: real_state_empirical ADDS to the existing horizon,
        # so add_calculation's own intersection (:128-129) already combines the
        # propagated set with the measurement box.
        if MEAS_DELAY:
            delay['u'] = float(np.asarray(tester._estctrl_u(start), dtype=float).flatten()[0])
            delay['S_start'] = np.asarray(tester.horizons[start].get_tight_bound(),
                                          dtype=float).copy()
        res = orig_emp(start, end)
        _check(end)
        return res

    tester.real_state_mpc = real_state_mpc
    tester.real_state_empirical = real_state_empirical
    return stats
# ──────────────────────────────────────────────────────────────────────────────


# ── alg27: the NN controller reads the ESTIMATE, not the true state ──────────
# In alg26v the plant applies NN(x_true) (REAL_integrated_sim.py:522). That is
# what the closed-loop propagation assumes: cl_systems/unicycle.py:14 evaluates
# the NN at the same state it propagates, so the set is {f(x, NN(x)) : x in X}.
# Once the plant applies NN(x_hat) with x_hat != x_true, that set no longer
# has to contain the true next state, so both halves change:
#
#   certificate  (executed t -> t+1)  x_hat = centre of the certified box at t
#                is known, so u = NN(x_hat) is one number and the step is
#                _interval_step(X_t, u). Sound, and no looser than before.
#   lookahead    (every later step)   the future x_hat is unknown but lies in
#                the future certified box, which is inside the lookahead box, so
#                u in NN(X) (CROWN bounds) and the step is _interval_step_u.
#                Mode A keeps the old coupled propagation here (NOT sound under
#                estimate control; it isolates the cost of the certificate
#                change). Mode B uses the sound decoupled step and turns P4 off,
#                because the k-step symbolic system is coupled by construction.
#
# The control applied at t is cached the first time it is computed, so the
# plant applies exactly the u the certificate step used even if X_t is later
# tightened.
ESTCTRL = os.environ.get('TTTCARV_ESTCTRL', 'B').strip().upper()   # alg28: default B
if ESTCTRL not in ('0', 'A', 'B', 'J'):
    raise ValueError(f"TTTCARV_ESTCTRL must be 0, A, B or J, got {ESTCTRL!r}")
# alg28 mode J. Mode B bounds u = NN(x_hat) over the whole box and then lets
# every u act on every x. That discards the feedback: dNN/dtheta ~ -2.6 along
# the nominal path, so the true heading spread contracts ~0.48x per step while
# the decoupled bound expands ~1.5x per step (exp28e_diagnose.py: even with the
# EXACT NN range, theta 6.9 rad vs 0.5 true after 20 steps). Mode J bounds one
# closed-loop step jointly: x_{k+1} = f(x, NN(x + e)) over x in X_k and
# |e_i| <= min(vmax, hw_i(X_k)). That e box is sound: the NN reads the centre c
# of the certified set S_k; x and c both lie in S_k, S_k lies in X_k and inside
# the measurement box z +- vmax, so |x - c| <= hw(S_k) <= min(hw(X_k), vmax).
# CROWN over the joint box is split into JSPLIT cells over
# (x, y, theta, e_x, e_y, e_theta), bounded in ONE batched call, and hulled
# (exp28f_split.py: 32 cells ~9.5 ms/step, 256 cells ~13 ms/step on one core).
JSPLIT = tuple(int(c) for c in os.environ.get('TTTCARV_JSPLIT', '1,1,8,1,1,4').split(','))
assert len(JSPLIT) == 6 and min(JSPLIT) >= 1, 'TTTCARV_JSPLIT needs 6 positive ints'

# alg28: how mode J bounds the estimation error e_k at the k-th lookahead step.
#   X  |e_k| <= min(vmax, hw(X_k))                          (default, alg28 as run)
#   S  |e_k| <= min(vmax, hw(X_k), hw(S_k)), hw(S_k) by the recursion below
# X_k grows along the chain, so rule X hands the NN the raw sensor bound vmax in
# every component by k ~ 5. But e_k is the gap between the true state and the
# CENTRE of the certified set S_k, so |e_k| <= hw(S_k), and hw(S) obeys a
# recursion that does not need the (unknown) future controls, because whatever
# drives the robot at a future step applies ONE scalar u:
#   theta' = theta + dt*u          => hw_theta(F(S,u))  = hw_theta(S)
#   |d cos/d theta| <= 1           => hw_xy(F(S,u))    <= hw_xy(S) + dt*v*hw_theta(S)
#   S_{k+1} = F(S_k,u_k) ∩ (z ± vmax) => hw(S_{k+1}) <= min(vmax, hw(F(S_k,u_k)))
# so hw_theta stays at its value at the base of the chain and hw_xy creeps up by
# dt*v*hw_theta per step. Measured on real certified sets (exp28m_ebound.py,
# dump_J256 t=36, 10 steps ahead): theta 0.892 -> 0.279 rad, xy 0.616 -> 0.297 m,
# at the same cost. The realised |e_theta| in a run is hw_theta(S_t) = 0.0067 rad
# for every step of the horizon, against the 0.15 rule X assumes (exp28l).
# Only mode J can use this: mode B is sound because the future estimate lies in
# S_k ⊆ X_k so u in NN(X_k), and a bound on hw(S_k) does not say WHERE in X_k
# that box sits, so it cannot shrink the NN's input box.
# The recursion assumes S is only ever propagated and intersected. The
# empty-intersection REPAIR in _install_setmem resets S to the measurement box
# (half-width vmax) and would break it, so rule S falls back to rule X for the
# rest of a trial in which a repair fires (0 in 12000 steps of the 200-trial
# alg28j batch; an empty intersection already means the propagated set lost the
# truth, which is fatal to the certificate with or without this rule).
EBOX = os.environ.get('TTTCARV_EBOX', 'X').strip().upper()
if EBOX not in ('X', 'S'):
    raise ValueError(f"TTTCARV_EBOX must be X or S, got {EBOX!r}")

# Float32 slack for CROWN: its bounds are sound up to rounding, and the plant
# evaluates the NN in float32 too.
_NN_IN_PAD, _NN_OUT_PAD = 1e-6, 1e-5
# alg28: TTTCARV_USAT=1 saturates the NN turn rate at +-USAT_MAX, the same
# actuator limit the MPC (|omega| <= 1, utils/unicycle_mpc_acados.py:332) and
# the danger radius D = sqrt(r^2 + 2rR), R = v/u_max, already assume. Nothing
# clamps the NN otherwise: REAL_integrated_sim :519-523 applies control_nn
# straight into dynamics_step, and the NN reaches ~4 rad/s on the workspace
# (and far more off it). Saturation is applied to the control the plant
# receives AND to the lookahead's control interval, so both stay consistent:
# sat is monotone, so u in [lo, hi] implies sat(u) in [sat(lo), sat(hi)].
USAT = os.environ.get('TTTCARV_USAT', '0') != '0'
USAT_MAX = float(os.environ.get('TTTCARV_USAT_MAX', '1.0'))


def _sat(u):
    return float(np.clip(u, -USAT_MAX, USAT_MAX)) if USAT else float(u)
_NN_BOUNDER = {}


def _interval_step_u(b, u_lo, u_hi, dt, v):
    """_interval_step under a control INTERVAL [u_lo, u_hi] (dt > 0)."""
    b = np.asarray(b, dtype=float)
    out = _interval_step(b, 0.0, dt, v)
    out[2, 0], out[2, 1] = b[2, 0] + dt * u_lo, b[2, 1] + dt * u_hi
    return out


def _nn_bounder(controller):
    """CROWN bounds of the controller over an input box, built once per process."""
    key = id(controller)
    if key not in _NN_BOUNDER:
        import copy
        import torch
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        ctrl = copy.deepcopy(controller).cpu().eval()
        bm = BoundedModule(ctrl, torch.zeros(1, 3))

        def bounds(b):
            b = np.asarray(b, dtype=float)
            lo = torch.tensor(b[:, 0] - _NN_IN_PAD, dtype=torch.float32).reshape(1, -1)
            hi = torch.tensor(b[:, 1] + _NN_IN_PAD, dtype=torch.float32).reshape(1, -1)
            x = BoundedTensor((lo + hi) / 2.0,
                              PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
            with torch.no_grad():
                l, u = bm.compute_bounds(x=(x,), method='CROWN')
            return float(l.flatten()[0]) - _NN_OUT_PAD, float(u.flatten()[0]) + _NN_OUT_PAD
        _NN_BOUNDER[key] = bounds
    return _NN_BOUNDER[key]


_JOINT_BOUNDER = {}


def _joint_bounder(controller, dt, v):
    """Mode J: sound box of {f(x, NN(x + e)) : x in X, e in E} by CROWN over
    JSPLIT cells in one batched call. Built once per process."""
    key = (id(controller), JSPLIT)
    if key not in _JOINT_BOUNDER:
        import copy
        import itertools
        import torch
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        class _Step(torch.nn.Module):
            def __init__(self, ctrl):
                super().__init__()
                self.ctrl = ctrl
                eye = torch.eye(6)
                self.register_buffer('Px', eye[:, :3].clone())
                self.register_buffer('Pe', eye[:, 3:].clone())
                for j in range(3):
                    self.register_buffer(f'S{j}', torch.eye(3)[:, j:j + 1].clone())

            def forward(self, inp):
                x = inp @ self.Px
                u = self.ctrl(x + inp @ self.Pe)
                x0, x1, x2 = x @ self.S0, x @ self.S1, x @ self.S2
                return torch.cat([x0 + dt * v * torch.cos(x2), x1 + dt * v * torch.sin(x2),
                                  x2 + dt * u], 1)

        n = int(np.prod(JSPLIT))
        step = _Step(copy.deepcopy(controller).cpu().eval())
        bm = BoundedModule(step, torch.zeros(n, 6), device='cpu', verbose=False)
        cells = np.array(list(itertools.product(*[range(k) for k in JSPLIT])))   # (n, 6)
        ks = np.array(JSPLIT, dtype=float)

        def bounds(b, hw_e):
            box = np.vstack([np.asarray(b, dtype=float),
                             np.stack([-hw_e, hw_e], 1)])                        # (6, 2)
            w = (box[:, 1] - box[:, 0]) / ks
            lo = box[:, 0] + cells * w
            hi = np.where(cells == ks - 1, box[:, 1], lo + w)                    # exact outer edge
            lo_t = torch.tensor(lo - _NN_IN_PAD, dtype=torch.float32)
            hi_t = torch.tensor(hi + _NN_IN_PAD, dtype=torch.float32)
            x = BoundedTensor((lo_t + hi_t) / 2.0,
                              PerturbationLpNorm(norm=np.inf, x_L=lo_t, x_U=hi_t))
            with torch.no_grad():
                l, u = bm.compute_bounds(x=(x,), method='CROWN')
            l, u = l.numpy().astype(float), u.numpy().astype(float)
            return np.stack([l.min(0) - _NN_OUT_PAD, u.max(0) + _NN_OUT_PAD], 1)
        _JOINT_BOUNDER[key] = bounds
    return _JOINT_BOUNDER[key]


def _install_estctrl(tester, setmem_stats=None):
    """Make the plant apply NN(centre of certified box). Returns a stats dict."""
    stats = {'mode': ESTCTRL, 'known_steps': 0, 'lookahead_steps': 0,
             'refresh_contained': 0, 'refresh_uncontained': 0,
             # taus whose lookahead bound was replaced by a box the old one
             # does not contain; the PSF loop drains this and drops the
             # backups stored there (they were certified for the old box).
             'uncontained_ts': set(),
             'buffer_invalidated': 0, 'sym_next_restored': 0,
             'ctrl_gap_max': 0.0, 'ctrl_gap_sum': 0.0, 'ctrl_gap_n': 0,
             'nn_width_max': 0.0, 'sat_steps': 0, 'u_raw_max': 0.0,
             'look_checked': 0, 'look_truth_out': 0, 'look_skipped_mpc': 0,
             'joint_wall_max': 0.0, 'ebox': EBOX, 'ebox_tightened': 0,
             'ebox_repair_fallback': 0, 'ebox_hw_theta_max': 0.0}
    if ESTCTRL == '0':
        return stats
    from REAL_integrated_sim import CalculationType

    dyn = tester.analyzer.cl_system.dynamics
    controller = tester.analyzer.cl_system.controller
    dt, v = float(dyn.dt), float(dyn.vt)
    # The analyzer is shared across trials; a leftover instance attribute would
    # mean a previous trial died mid-step with the plant patch still in place.
    assert 'control_nn' not in vars(dyn), 'control_nn patch leaked from a previous trial'
    orig_cnn = dyn.control_nn
    horizon_cls = type(tester.horizons[0])
    nn_bounds = _nn_bounder(controller) if ESTCTRL == 'B' else None
    joint_bounds = _joint_bounder(controller, dt, v) if ESTCTRL == 'J' else None
    vmax_e = VMAX_SIGMA * float(tester.measurement_noise_std)
    if MEAS_DELAY:
        assert ESTCTRL in ('B', 'J'), 'MEAS_DELAY needs the estimate-driven plant (B or J)'
        vmax_e = np.array([vmax_e * (1.0 + dt * v), vmax_e * (1.0 + dt * v), vmax_e])
    look_box = {}      # t -> (last lookahead box filed at t, executed step it chains from)
    chain_base = {}    # t -> executed step the bound filed at t chains from
    hw_base = {}       # executed step -> exact half-width of the certified set there

    def _hw_next(hw):
        """Sound bound on hw(S) one step on, for ANY scalar control (rule S)."""
        g = dt * v * float(hw[2])
        return np.minimum(vmax_e, np.array([hw[0] + g, hw[1] + g, hw[2]], dtype=float))

    def hw_cert(t):
        """Bound on hw(S_t) from the base of the chain that filed the bound at t."""
        base = chain_base.get(t)
        if base is None or base not in hw_base or base > t:
            return None
        if setmem_stats is not None and setmem_stats.get('empty_repaired', 0):
            stats['ebox_repair_fallback'] += 1     # S was reset to the measurement box
            return None
        hw = hw_base[base]
        for _ in range(t - base):
            hw = _hw_next(hw)
        return hw
    nn_steps = set()   # executed steps whose applied control was NN(centre)
    u_cache = {}

    def has_real(t):
        h = tester.horizons.get(t)
        return h is not None and any('real_state' in c for c in h.calculations.values())

    def truth(t):
        for c in tester.horizons[t].calculations.values():
            if 'real_state' in c:
                return np.asarray(c['real_state'], dtype=float).flatten()
        return None

    def u_est(t):
        """NN at the centre of the certified box at executed step t (cached)."""
        if t not in u_cache:
            assert has_real(t), f'u_est({t}) before step {t} was executed'
            b = np.asarray(tester.horizons[t].get_tight_bound(), dtype=float)
            c = (b[:, 0] + b[:, 1]) / 2.0
            u_raw = np.asarray(orig_cnn(c.reshape(1, -1), controller.cpu()),
                               dtype=float).reshape(1, -1)
            if USAT:
                stats['sat_steps'] += int(abs(float(u_raw[0, 0])) > USAT_MAX)
                stats['u_raw_max'] = max(stats['u_raw_max'], abs(float(u_raw[0, 0])))
                u_raw = np.clip(u_raw, -USAT_MAX, USAT_MAX)
            u_cache[t] = u_raw
        return u_cache[t]

    def known_step(t):
        b = tester.horizons[t].get_tight_bound()
        stats['known_steps'] += 1
        # AUDIT: the last lookahead box filed at t must contain the true state
        # whenever the NN actually drove every step of its chain. (Under an MPC
        # backup step the lookahead's model -- NN control -- did not apply.)
        lb = look_box.pop(t, None)
        x = truth(t) if lb is not None else None
        if x is not None:
            P, base = lb
            if all(s in nn_steps for s in range(base, t)):
                stats['look_checked'] += 1
                if np.any(x < P[:, 0] - 1e-9) or np.any(x > P[:, 1] + 1e-9):
                    stats['look_truth_out'] += 1
                    print(f"  [LOOK_AUDIT] truth outside lookahead box at t={t}"
                          f" (chain from t={base})")
            else:
                stats['look_skipped_mpc'] += 1
        chain_base[t + 1] = t
        bb = np.asarray(b, dtype=float)
        hw_base[t] = (bb[:, 1] - bb[:, 0]) / 2.0
        return _interval_step(b, float(u_est(t)[0, 0]), dt, v)

    def joint_step(t):
        b = np.asarray(tester.horizons[t].get_tight_bound(), dtype=float)
        hw_e = np.minimum(vmax_e, (b[:, 1] - b[:, 0]) / 2.0)
        if EBOX == 'S':
            hw_s = hw_cert(t)
            if hw_s is not None:
                stats['ebox_tightened'] += int(bool(np.any(hw_s < hw_e - 1e-12)))
                stats['ebox_hw_theta_max'] = max(stats['ebox_hw_theta_max'], float(hw_s[2]))
                hw_e = np.minimum(hw_e, hw_s)
        w0 = time.perf_counter()
        P = joint_bounds(b, hw_e)
        stats['joint_wall_max'] = max(stats['joint_wall_max'], time.perf_counter() - w0)
        stats['lookahead_steps'] += 1
        stats['nn_width_max'] = max(stats['nn_width_max'], (P[2, 1] - P[2, 0]) / dt)
        chain_base[t + 1] = chain_base.get(t, t)
        look_box[t + 1] = (P.copy(), chain_base[t + 1])
        return P

    def decoupled_step(t):
        if ESTCTRL == 'J':
            return joint_step(t)
        b = tester.horizons[t].get_tight_bound()
        u_lo, u_hi = nn_bounds(b)
        u_lo, u_hi = _sat(u_lo), _sat(u_hi)
        stats['lookahead_steps'] += 1
        stats['nn_width_max'] = max(stats['nn_width_max'], u_hi - u_lo)
        P = _interval_step_u(b, u_lo, u_hi, dt, v)
        chain_base[t + 1] = chain_base.get(t, t)
        look_box[t + 1] = (np.asarray(P, dtype=float).copy(), chain_base[t + 1])
        return P

    def file_bound(t1, P, notes):
        """File P at t1. A lookahead bound there is REPLACED (it may assume the
        old control); an executed bound is intersected (both are sound).
        Returns the replaced bound, or None."""
        h = tester.horizons.get(t1)
        if h is None:
            h = horizon_cls(t1, device=tester.analyzer.device)
            tester.horizons[t1] = h
        old = None
        if h.tight_bound is not None and not has_real(t1):
            old = h.tight_bound.copy()
            h.tight_bound = None
        h.add_calculation(bounds=np.asarray(P, dtype=float),
                          calc_type=CalculationType.CONCRETE,
                          origin_timestep=t1 - 1, step_size=1, notes=notes)
        return old

    def refresh(t, mpc_buffer, mpc_reason):
        """Before P1: make the bound at t+1 the known-control step from t.
        Drops the t+1 backup if it was built for a box that does not contain
        the new one. Returns True if it dropped anything."""
        t1 = t + 1
        if t1 not in tester.horizons or has_real(t1):
            return False
        P = known_step(t)
        old = file_bound(t1, P, f'alg27 refresh from t={t}')
        if old is None:
            return False
        if _bounds_contain(old, P):
            stats['refresh_contained'] += 1
            return False
        stats['refresh_uncontained'] += 1
        if t1 in mpc_buffer:
            del mpc_buffer[t1]
            mpc_reason.pop(t1, None)
            stats['buffer_invalidated'] += 1
            return True
        return False

    orig_concrete = tester.concrete

    def concrete(start_timestep, end=None, time_left=None, step_price=0.0):
        if end is None:
            end = start_timestep + 1
        if ESTCTRL == 'A' and not has_real(start_timestep):
            return orig_concrete(start_timestep, end, time_left=time_left,
                                 step_price=step_price)
        total, bounds, cur, last_wall = 0.0, None, start_timestep, 0.0
        for k in range(end - start_timestep):
            if (k > 0 and time_left is not None
                    and time_left() < max(step_price, last_wall)):
                tester.time += total
                return {'success': True, 'time': total, 'completed_steps': k,
                        'truncated': True, 'collision': False, 'final_bounds': bounds}
            if ESTCTRL == 'A' and k > 0:
                # Mode A: only the step out of the executed state is replaced.
                tester.time += total
                r = orig_concrete(cur, end, time_left=time_left, step_price=step_price)
                if not r:
                    return r
                r = dict(r)
                r['completed_steps'] = k + r.get('completed_steps', end - cur)
                return r
            w0 = time.perf_counter()
            P = known_step(cur) if has_real(cur) else decoupled_step(cur)
            old = file_bound(cur + 1, P, f'alg27 step from t={cur}')
            if old is not None:
                if _bounds_contain(old, P):
                    stats['refresh_contained'] += 1
                else:
                    # A plan verified from `old` covers a box inside `old`; this
                    # one is not, so any backup stored at cur+1 is no longer
                    # certified. refresh() does this at t+1; the re-walk reaches
                    # further, so record it and let the PSF loop drop it.
                    stats['refresh_uncontained'] += 1
                    stats['uncontained_ts'].add(cur + 1)
            cur += 1
            bounds = tester.horizons[cur].get_tight_bound()
            last_wall = time.perf_counter() - w0
            total += last_wall
            collisions, intersections = tester.obstacles.check_collision(bounds)
            if collisions is not None:
                tester.time += total
                return {'success': False, 'time': total, 'completed_steps': k + 1,
                        'collision': True, 'collision_timestep': cur,
                        'collision_obstacles': collisions,
                        'intersections': intersections, 'final_bounds': bounds}
        tester.time += total
        return {'success': True, 'time': total,
                'completed_steps': end - start_timestep,
                'collision': False, 'final_bounds': bounds}

    orig_symbolic = tester.symbolic

    def symbolic(start, end):
        # Mode A: the k-step symbolic system is coupled. It may inform the
        # conflict decision, but it must not tighten the next step's certificate.
        guard = (not has_real(end) and has_real(end - 1)
                 and end in tester.horizons
                 and tester.horizons[end].tight_bound is not None)
        before = tester.horizons[end].tight_bound.copy() if guard else None
        res = orig_symbolic(start, end)
        if guard:
            h = tester.horizons[end]
            if not np.array_equal(h.tight_bound, before):
                import torch
                h.tight_bound = before
                h.reachable_set.set_range(torch.tensor(before, dtype=torch.float32,
                                                       device=h.device))
                stats['sym_next_restored'] += 1
        return res

    prev_emp = tester.real_state_empirical

    def real_state_empirical(start, end):
        assert end == start + 1, 'alg27 applies one control per empirical step'
        nn_steps.add(start)
        u = u_est(start)
        x = truth(start)
        if x is not None:
            gap = float(np.max(np.abs(u - np.vectorize(_sat)(np.asarray(
                orig_cnn(x.reshape(1, -1), controller.cpu()), dtype=float)))))
            stats['ctrl_gap_max'] = max(stats['ctrl_gap_max'], gap)
            stats['ctrl_gap_sum'] += gap
            stats['ctrl_gap_n'] += 1
        # Both the plant (:522) and the EKF predict (:533) call control_nn; the
        # applied control is u in both, and the estimator is told what was applied.
        dyn.control_nn = lambda xs, model: np.tile(
            u, (xs.shape[0] if len(xs.shape) > 1 else 1, 1))
        try:
            return prev_emp(start, end)
        finally:
            del dyn.control_nn

    tester.concrete = concrete
    tester.symbolic = symbolic
    tester.real_state_empirical = real_state_empirical
    tester._estctrl_u = u_est
    stats['refresh'] = refresh
    return stats
# ──────────────────────────────────────────────────────────────────────────────


# ── alg28: SOUND backup-plan boxes ────────────────────────────────────────────
# mpc_safety_filter_acados returns trajectory_bounds[k] = (centre rollout) +-
# (START half-widths): a constant-size box (:437-445 one-shot, :307-313 re-sim).
# Under heading uncertainty the reachable set of a fixed control sequence fans
# out -- a heading offset d rotates the whole path -- so that box misses true
# states from step 1 on (exp28c_mpccheck.py: 201/201 plans; at a realistic
# 0.23 rad start width it covers ~14% of the true area after 20 steps).
#
# Fix, in two parts:
#   acceptance  trajectory_bounds[k+1] = _interval_step(trajectory_bounds[k], u_k).
#               The controls are known numbers and the plant is exactly the
#               unicycle, so this chain contains every state reachable from the
#               start box (exp28c: 201/201, median 1.01-1.07x the exact box).
#               Everything downstream -- _rollout_verdict, _score_rollout, the
#               committed-plan fallback test -- reads these boxes.
#   OCP margin  the solver still plans ONE point (the centre). Each path
#               stage's obstacle margin grows by k*dt*v*hw_theta (not the
#               terminal; see FAN_TERMINAL): a heading offset
#               |d| <= hw_theta moves the k-step position by at most
#               sum_j dt*v*|e^{i(a_j+d)} - e^{i a_j}| <= k*dt*v*hw_theta. This
#               only steers the solver toward plans the acceptance test will
#               pass; soundness comes from the acceptance boxes alone.
#               (Re-sim path: each re-solve restarts the growth at 0, so it is
#               less tightened there -- again only a performance question.)
SOUND_MPC = os.environ.get('TTTCARV_SOUND_MPC', '1') != '0'
FAN_MARGIN = os.environ.get('TTTCARV_FAN_MARGIN', '1') != '0'
# The growth goes on path stages only; the terminal keeps the base inflation.
# Path stages are slacked but the terminal constraint is HARD, so growth there
# is an infeasibility cliff, not a preference. Measured in the loop
# (~/runs/alg39_pinfan, 500 paired scenes): bad one-shot solves 32% -> 18%
# (uniform) and 38% -> 19% (trunc), intervention -0.10 pp, 0 collisions.
# TTTCARV_FAN_TERMINAL=1 restores growth on the terminal too (rev 7 and earlier).
FAN_TERMINAL = os.environ.get('TTTCARV_FAN_TERMINAL', '0') != '0'
# Audit: every accepted-or-not plan is re-checked by rolling the plan's own
# controls out from 64 random points + the 8 corners of its start box. A point
# outside its box would mean the interval chain is wrong. Numpy, ~0.1 ms.
PLAN_AUDIT = os.environ.get('TTTCARV_PLAN_AUDIT', '1') != '0'
# Cells the plan's start box is split into along theta before propagating.
#
# One chain per node re-boxes between nodes, which discards the correlation
# between consecutive states (wrapping). Measured against the EXACT reachable
# set (closed form for an open-loop unicycle plan; ~/runs/alg28_arc), the single
# chain is median 1.01-1.03x the exact box area at node 20 but p90 2.9-4.5x and
# worst 4.1-7.5x: the looseness is a TAIL effect, on plans that turn hard, which
# are exactly the plans near an obstacle. Splitting theta and hulling per node
# removes most of it -- at 16 cells, p90 4.54 -> 1.21x, worst 7.45 -> 1.37x.
#
# Splitting x and y would do nothing here (they enter the chain as a pure
# translation); on another plant they would need splitting too.
#
# Sound either way: every cell chain is sound and the hull of sound boxes is
# sound, so this only ever shrinks the acceptance boxes. Cost is ~0.35 ms per
# cell per plan against a ~68 ms solve. 1 = the unsplit chain.
PLAN_CELLS = max(1, int(os.environ.get('TTTCARV_PLAN_CELLS', '1')))

# TTTCARV_REANCHOR=1: when P3 hits a wall it cannot pass, throw the lookahead
# chain away and re-walk it from the current timestep instead of leaving it
# parked at the horizon.
#
# The chain that found the conflict was propagated from an anchor many steps
# back, and every step of that walk re-boxes and loses inter-step correlation
# (wrapping). By the time P3 has spent several timesteps building backups and
# hit an infeasible one, the same absolute time tau is reachable from a much
# later anchor -- fewer steps of wrapping, and from a certified set that has
# absorbed several more measurements. The rebuilt box at the wall is therefore
# much tighter, which can (a) retire the conflict entirely or (b) leave the
# conflict standing but let the backup that came back INFEASIBLE against a fat
# box succeed against a thin one. (b) is the more common win.
#
# This is the structural replacement for P4, which is off in modes B and J:
# same job (re-adjudicate a conflict with better bounds), freshness instead of
# symbolic tightness. Default off so it is an A/B arm, not a silent change.
REANCHOR = os.environ.get('TTTCARV_REANCHOR', '0') != '0'

# TTTCARV_REANCHOR_REARM=k: let a conflict be re-anchored AGAIN once k timesteps
# have passed since the last re-walk for it. 0 (default) = the original
# once-per-conflict_time behaviour.
#
# Why: after a re-walk reconfirms a conflict, the filter goes idle. P3 breaks at
# the first cached INFEASIBLE tau, P3b's scan_window walks the rest of the span
# and skips every tau already filed as None, and P2 cannot run because
# scan_ceil == conflict_time == concrete_until. Nothing in the chain is
# recomputed and the step costs ~0.001s of the 0.200s budget, for as long as it
# takes the robot to reach the conflict (11 steps on seed 287580328 under trunc
# noise). Meanwhile the certified set keeps absorbing measurements, so the
# boxes the rejections were measured against get staler every step -- and the
# rejections can be narrow: tau=24 there missed on box_dist=1.097 vs D_req=1.118.
#
# Re-arming spends that idle budget on exactly the thing that already works.
# The freshness gain is measured from conflict_found_at, which P2 re-stamps on
# every (re)detection, so a reconfirming re-walk resets the clock and the arm
# paces itself. Sound for the same reason the first re-walk is: conflict_time
# stays set until a re-walk proves the node clear.
# TTTCARV_CONFLICT_RETEST=1: re-test a flagged conflict against its CURRENT
# box every timestep, and drop the flag once that box no longer reaches the
# danger radius. See the [P2a] block for why this is needed and why it is sound.
# Default OFF so every existing arm reproduces exactly.
CONFLICT_RETEST = os.environ.get('TTTCARV_CONFLICT_RETEST', '0') != '0'
# TTTCARV_PASSTHROUGH=0 disables the RUN-TIME passthrough: the step where there
# is no backup at t+1 but a valid one at t_diverge > t, and nominal flies anyway
# because the concrete scan certified every step before conflict_time clear.
# This is NOT P3b (the build-time scan past a wall, counted by passthrough_found);
# it is the branch that consumes that result, counted by passthrough_used.
# Default ON: it is the shipped behaviour and removing it does not degrade
# gracefully -- with t_diverge > t and nothing anchored at t, the step falls into
# the activation path, which refuses a ctrl_idx>0 plan (psf_stale_refused).
# The flag exists to MEASURE that cost, not as a configuration anyone should use.
PASSTHROUGH = os.environ.get('TTTCARV_PASSTHROUGH', '1') != '0'
REANCHOR_DEBUG = os.environ.get('TTTCARV_REANCHOR_DEBUG', '0') != '0'
REANCHOR_REARM = int(os.environ.get('TTTCARV_REANCHOR_REARM', '0'))
if REANCHOR_REARM < 0:
    raise ValueError(f"TTTCARV_REANCHOR_REARM must be >= 0, got {REANCHOR_REARM!r}")


def _plan_boxes(b0, us, dt, v, cells):
    """Acceptance boxes for an open-loop plan, one per node.

    `cells` chains from a theta-split of the start box, hulled at every node.
    cells=1 is the plain interval chain. Sound for any cells: each chain
    contains the states reachable from its own cell, the cells cover b0, and
    the hull of sound boxes is sound.
    """
    def chain(b):
        out = [np.asarray(b, dtype=float).copy()]
        for u in us:
            out.append(_interval_step(out[-1], u, dt, v))
        return out

    b0 = np.asarray(b0, dtype=float)
    if cells <= 1:
        return chain(b0)
    edges = np.linspace(b0[2, 0], b0[2, 1], cells + 1)
    chains = []
    for i in range(cells):
        c = b0.copy()
        c[2, 0], c[2, 1] = edges[i], edges[i + 1]
        chains.append(chain(c))
    out = [b0.copy()]
    for k in range(1, len(us) + 1):
        s = np.stack([ch[k] for ch in chains])
        out.append(np.stack([s[:, :, 0].min(0), s[:, :, 1].max(0)], 1))
    return out


def _install_sound_mpc(mpc_sf, tester, seed):
    stats = {'sound_mpc': int(SOUND_MPC), 'fan_margin': int(FAN_MARGIN),
             'fan_terminal': int(FAN_TERMINAL),
             'plans': 0, 'audit_pts': 0, 'audit_miss': 0, 'plan_cells': PLAN_CELLS,
             'fan_margin_max': 0.0, 'box_growth_max': 0.0, 'split_gain_max': 1.0}
    if not SOUND_MPC:
        return stats
    dyn = tester.analyzer.cl_system.dynamics
    dt, v = float(dyn.dt), float(dyn.vt)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    fan = {'hw_theta': 0.0}
    mpc = mpc_sf._mpc

    if FAN_MARGIN:
        orig_push = mpc._push_parameters

        def _push_parameters():
            orig_push()
            base = float(mpc._obs_inflation_buf[0])
            g = dt * v * fan['hw_theta']
            for k in range(mpc.n_horizon + 1):
                kk = k if (FAN_TERMINAL or k < mpc.n_horizon) else 0
                mpc._solver.set(k, 'p', np.array([base + kk * g]))
            stats['fan_margin_max'] = max(stats['fan_margin_max'], mpc.n_horizon * g)
        mpc._push_parameters = _push_parameters

    def _audit(b0, controls, boxes):
        corners = np.array(np.meshgrid(*[b0[i] for i in range(3)])).reshape(3, -1).T
        X = np.vstack([corners, rng.uniform(b0[:, 0], b0[:, 1], size=(64, 3))])
        stats['audit_pts'] += len(X)
        for k, u in enumerate(controls):
            X = np.stack([X[:, 0] + dt * v * np.cos(X[:, 2]),
                          X[:, 1] + dt * v * np.sin(X[:, 2]),
                          X[:, 2] + dt * u], axis=1)
            b = boxes[k + 1]
            if np.any(X < b[:, 0] - 1e-9) or np.any(X > b[:, 1] + 1e-9):
                stats['audit_miss'] += 1
                print(f"{RED}[SOUND_MPC] AUDIT MISS at node {k + 1}{RESET}")
                return

    def wrap(orig):
        def run(initial_bounds, center, extra_inflation, *a, **kw):
            b0 = np.asarray(initial_bounds, dtype=float)
            fan['hw_theta'] = float(b0[2, 1] - b0[2, 0]) / 2.0
            tb, pts, controls = orig(initial_bounds, center, extra_inflation, *a, **kw)
            us = [float(np.asarray(c, dtype=float).flatten()[0]) for c in controls]
            boxes = _plan_boxes(b0, us, dt, v, PLAN_CELLS)
            stats['plans'] += 1
            if PLAN_CELLS > 1 and PLAN_AUDIT:
                plain = _plan_boxes(b0, us, dt, v, 1)
                for k in (len(boxes) - 1,):
                    a = float(np.max(plain[k][:2, 1] - plain[k][:2, 0]))
                    b = float(np.max(boxes[k][:2, 1] - boxes[k][:2, 0]))
                    if b > 1e-12:
                        stats['split_gain_max'] = max(stats['split_gain_max'], a / b)
                    # a split chain can only shrink the hull
                    assert np.all(boxes[k][:, 0] >= plain[k][:, 0] - 1e-9) and \
                           np.all(boxes[k][:, 1] <= plain[k][:, 1] + 1e-9), 'split chain wider'
            w_old = float(np.max(tb[-1][:2, 1] - tb[-1][:2, 0]))
            w_new = float(np.max(boxes[-1][:2, 1] - boxes[-1][:2, 0]))
            stats['box_growth_max'] = max(stats['box_growth_max'], w_new - w_old)
            if PLAN_AUDIT:
                _audit(b0, us, boxes)
            return boxes, pts, controls
        return run

    mpc_sf._run_mpc_single_oneshot = wrap(mpc_sf._run_mpc_single_oneshot)
    mpc_sf._run_mpc_single_resim = wrap(mpc_sf._run_mpc_single_resim)
    return stats
# ──────────────────────────────────────────────────────────────────────────────


# Turn-direction flip diagnostics ignore controls with |u| below this; a
# near-zero turn has no meaningful direction.
FLIP_DEADBAND = 0.05



# ── Shared helpers ─────────────────────────────────────────────────────────


def _get_nn_control(tester, timestep):
    """Query NN nominal control at the real state stored in horizons[timestep]."""
    if getattr(tester, '_estctrl_u', None) is not None and timestep in tester.horizons:
        return tester._estctrl_u(timestep).flatten()
    h = tester.horizons.get(timestep)
    if h is None:
        return None
    calc = next((c for c in h.calculations.values() if 'real_state' in c), None)
    if calc is None:
        return None
    state  = np.asarray(calc['real_state']).reshape(1, -1)
    cl_sys = tester.analyzer.cl_system
    return np.asarray(cl_sys.dynamics.control_nn(state, cl_sys.controller.cpu())).flatten()


class RefinementTask:
    """Represents an in-progress symbolic verification toward conflict_time."""
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start
        self.conflict_time  = conflict_time
        self.time_invested  = 0

    def done(self):
        return self.symbolic_start >= self.conflict_time


# Hooks set by test() when TTTCARV_MPC_PRICING=1 (None otherwise):
#   _CONCRETE_OBSERVER(sec_per_step)  records the measured concrete cost
#   _CONCRETE_DEADLINE                (time_left, step_price) callables that let
#                                     P2 stop between steps before budget runs out
_CONCRETE_OBSERVER = None
_CONCRETE_DEADLINE = None

def concrete_scan(tester, from_t, to_t):
    """Concrete propagation with no deadline. Returns (collision, collision_t)."""
    collision, ct, _ = concrete_scan_deadline(tester, from_t, to_t, deadline=False)
    return collision, ct


def concrete_scan_deadline(tester, from_t, to_t, deadline=True):
    """Concrete propagation that may stop early at the budget deadline.

    Returns (collision, collision_t, reached); reached is the last timestep that
    now has concrete bounds. Runs all the way to to_t if deadline=False or
    pricing is off.
    """
    if from_t >= to_t:
        return False, None, from_t
    kw = {}
    if deadline and _CONCRETE_DEADLINE is not None:
        time_left, step_price = _CONCRETE_DEADLINE
        kw = dict(time_left=time_left, step_price=step_price())
    t0 = time.perf_counter()
    result = tester.concrete(from_t, to_t, **kw)
    wall = time.perf_counter() - t0
    if not result:
        return False, None, from_t
    done = result.get('completed_steps', to_t - from_t)
    # Cost = wall time of the whole call / steps completed (includes overhead;
    # a collision stop completes fewer steps).
    if _CONCRETE_OBSERVER is not None and done > 0:
        _CONCRETE_OBSERVER(wall / done)
    if not result["collision"]:
        return False, None, from_t + done
    return True, result["collision_timestep"], result["collision_timestep"]


def symbolic_step(tester, job: RefinementTask, chunk_size: int, budget):
    """
    Advance the symbolic job by one chunk, if the budget allows.

    A chunk usually costs more than one timestep's budget. If it doesn't fit,
    this timestep's remaining budget is banked in job.time_invested instead, and
    the chunk runs once banked time + the current budget covers it.
    Returns (job, result); result is set only when the job finishes.
    """
    steps = min(chunk_size, job.conflict_time - job.symbolic_start)
    cost  = budget.symbolic_costs.get(steps, float('inf'))
    if cost - job.time_invested <= budget.remaining:
        verify_end         = min(job.conflict_time, job.symbolic_start + chunk_size)
        result             = tester.symbolic(job.symbolic_start, verify_end)
        job.symbolic_start = verify_end
        job.time_invested  = 0
        if job.done():
            return job, result
        return job, None
    else:
        job.time_invested += budget.remaining
        return job, None


def extend_mpc_sequence(mpc_sf, mpc_state: dict, min_safe_horizon: int,
                        max_time: int, ctrl_idx: int = 0, budget=None):
    """
    Append controls to the committed MPC plan (fallback when a re-solve fails).
    Solves from the end of the plan; if that fails, from progressively earlier
    points on it (lookback), replacing the tail.
    """
    committed_at = mpc_state['committed_at']
    controls     = mpc_state['controls']
    traj_bounds  = mpc_state['traj_bounds']
    n_controls   = len(controls)
    mpc_end      = committed_at + n_controls

    if mpc_end >= max_time:
        print(f"  [MPC extend] Reached MAX_TIME, done")
        return

    max_lb = max(min(n_controls - ctrl_idx - 1, mpc_sf.n_horizon), 0)

    for lb in range(max_lb + 1):
        if budget is not None and budget.remaining < budget.mpc_cost:
            print(f"  [MPC extend] Budget exhausted, will retry next timestep")
            return

        try_idx = len(traj_bounds) - 1 - lb
        if try_idx < 1:
            break

        try_bounds = traj_bounds[try_idx]
        center     = (try_bounds[:, 0] + try_bounds[:, 1]) / 2.0

        try:
            _t0 = time.time()
            new_traj_bounds, _, new_controls = mpc_sf._run_mpc_from_bounds(
                try_bounds, center, extra_inflation=0.0)
            print(f"  [MPC extend] from t={committed_at}  lookback={lb}  took {time.time()-_t0:.3f}s")
        except Exception as e:
            print(f"  [MPC extend] Failed at lookback={lb}: {e}")
            continue

        # The new segment may pass through D but not the raw obstacle, and like
        # any backup it must end outside D. Find:
        #   safe_count  how many new nodes stay clear of the raw obstacle
        #   term_count  the last of those that is also outside D (the new end)
        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides_raw(b):
                break
            safe_count += 1

        # Last node in the raw-safe prefix that clears D.
        term_count = 0
        for k in range(1, safe_count + 1):
            if not mpc_sf._collides_D(new_traj_bounds[k]):
                term_count = k

        if term_count <= lb:
            print(f"  [MPC extend] lookback={lb}: raw-safe={safe_count} but no "
                  f"D-terminal beyond lb={lb}, going further back")
            continue

        keep = n_controls - lb
        _note_ub(mpc_sf, new_controls[:term_count], 'acc_ub_extend')
        mpc_state['controls']    = list(controls[:keep]) + list(new_controls[:term_count])
        mpc_state['traj_bounds'] = (list(traj_bounds[:try_idx + 1])
                                    + list(new_traj_bounds[1:term_count + 1]))

        net_gain  = term_count - lb
        new_total = len(mpc_state['controls'])
        new_end   = committed_at + new_total
        if lb > 0:
            print(f"  [MPC extend] lookback={lb}: +{term_count} (D-terminal), "
                  f"-{lb} replaced, net +{net_gain}")
        print(f"  [MPC extend] +{net_gain} controls → {new_total} total "
              f"(t={committed_at} to t={new_end}, terminal ∉ D)")
        return

    print(f"  [MPC extend] No safe extension found after {max_lb + 1} lookback attempts")


# PSF helpers

def psf_valid(mpc_buffer: dict, t: int):
    """True if a valid backup starting at t is stored."""
    return mpc_buffer.get(t) is not None


def _rollout_verdict(mpc_sf, traj_bounds):
    """
    Acceptance test for an MPC plan, shared by backup builds and re-solves.
    Returns (ok, kind, node, n_nodes).

      kind  None | 'empty' | 'path_raw' | 'term_D'
      node  1-based index of the failing node (n_nodes for the terminal)

    Intermediate nodes must clear the raw obstacle radius (they may pass
    through D); the terminal node must clear D.
    """
    if traj_bounds is None or len(traj_bounds) < 2:
        return False, 'empty', -1, 0
    *path_bounds, terminal_bound = traj_bounds[1:]
    n_nodes = len(path_bounds) + 1
    for i, b in enumerate(path_bounds, start=1):
        if mpc_sf._collides_raw(b):
            return False, 'path_raw', i, n_nodes
    if mpc_sf._collides_D(terminal_bound):
        return False, 'term_D', n_nodes, n_nodes
    return True, None, -1, n_nodes


def _rollout_ok(mpc_sf, traj_bounds) -> bool:
    """Bare verdict, for callers that only need pass/fail."""
    return _rollout_verdict(mpc_sf, traj_bounds)[0]


def _bounds_contain(outer, inner, tol: float = 1e-9) -> bool:
    """True if box `inner` lies inside box `outer`."""
    if outer is None or inner is None:
        return False
    o = np.asarray(outer, dtype=float)
    i = np.asarray(inner, dtype=float)
    if o.shape != i.shape:
        return False
    return bool(np.all(i[:, 0] >= o[:, 0] - tol)
                and np.all(i[:, 1] <= o[:, 1] + tol))


def _note_ub(mpc_sf, controls, key):
    """Count an ACCEPTED plan segment whose |omega| exceeds u_max (the
    single-start path does not check it; see SINGLE_UBOUND)."""
    pk = max((abs(float(np.asarray(c).flatten()[0])) for c in controls), default=0.0)
    if pk > mpc_sf.u_max + 1e-6:
        setattr(mpc_sf, key, getattr(mpc_sf, key, 0) + 1)
        mpc_sf.acc_ub_peak = max(getattr(mpc_sf, 'acc_ub_peak', 0.0), pk)


def solve_plan_at(mpc_sf, tester, tau: int, budget=None, multi_start: bool = False):
    """
    Solve an MPC plan from the bounds at tau without storing it in mpc_buffer.
    Returns (ok, controls, traj_bounds, label).

    Used by the active-phase re-solve. It stays out of mpc_buffer because any
    entry there counts as a backup and would change PSF decisions.
    Single-start by default (cheaper); the caller retries with multi_start=True.
    """
    if budget is not None:
        need = budget.mpc_cost * (1.0 if (multi_start or budget.pricing)
                                  else SINGLE_START_COST_FRAC)
        if budget.remaining < need:
            return False, None, None, "no_budget"
    bounds = _bounds_at(tester, tau)
    if bounds is None:
        return False, None, None, "no_bounds"
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
            bounds, center, multi_start=multi_start)
    except Exception as e:
        return False, None, None, f"solver_error({e})"
    if controls is None or len(controls) == 0:
        return False, None, None, "empty_plan"
    ok, kind, node, n_nodes = _rollout_verdict(mpc_sf, traj_bounds)
    if not ok:
        # Coarse reason before "(", details inside, so counters can split on "(".
        return False, None, None, f"rollout_rejected({kind}@{node}/{n_nodes})"
    _note_ub(mpc_sf, controls, 'acc_ub_resolve')
    return True, list(controls), list(traj_bounds), "VALID"

def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int, reasons: dict = None):
    """
    Solve an MPC backup from the concrete bounds at tau and store it:
    mpc_buffer[tau] = (controls, traj_bounds) if it passes the acceptance test,
    else None. Returns True if valid.

    reasons (optional dict tau -> str) records why a build was rejected. It is
    separate from mpc_buffer because any non-None buffer entry counts as valid.
    """
    h = tester.horizons.get(tau)
    if h is None:
        if reasons is not None:
            reasons[tau] = "no_bounds(concrete RSOA not propagated to tau)"
        return False
    bounds = h.get_tight_bound()
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        _t0 = time.time()
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
            bounds, center, multi_start=True)
        elapsed = time.time() - _t0
        *path_bounds, terminal_bound = traj_bounds[1:]

        # Find which check failed, for the diagnostic label:
        #   path_raw  an intermediate node hits the raw obstacle
        #   term_D    the terminal node is inside D
        path_hit = next(
            ((i, b) for i, b in enumerate(path_bounds, start=1)
             if mpc_sf._collides_raw(b)),
            None
        )
        term_hit = mpc_sf._collides_D(terminal_bound)
        # The verdict itself comes from the shared acceptance test.
        collision = not _rollout_ok(mpc_sf, traj_bounds)

        if collision:
            if path_hit is not None:
                idx, b = path_hit
                cx_arr = (b[0, 0] + b[0, 1]) / 2.0
                cy_arr = (b[1, 0] + b[1, 1]) / 2.0
                # closest obstacle (raw center) + box-to-center distance
                best = min(
                    mpc_sf.raw_obstacles,
                    key=lambda o: (o[0] - cx_arr) ** 2 + (o[1] - cy_arr) ** 2
                )
                dx = best[0] - np.clip(best[0], b[0, 0], b[0, 1])
                dy = best[1] - np.clip(best[1], b[1, 0], b[1, 1])
                box_dist = float(np.sqrt(dx * dx + dy * dy))
                label = (f"path_raw[node={idx}/{len(path_bounds)}]"
                         f" obs=({best[0]:.2f},{best[1]:.2f},r={best[2]:.2f})"
                         f" box_dist={box_dist:.3f}")
            else:
                b = terminal_bound
                cx_arr = (b[0, 0] + b[0, 1]) / 2.0
                cy_arr = (b[1, 0] + b[1, 1]) / 2.0
                # closest obstacle, and required D radius vs actual distance
                pairs = []
                for o_raw, o_D in zip(mpc_sf.raw_obstacles, mpc_sf.safety_obstacles):
                    dx = o_raw[0] - np.clip(o_raw[0], b[0, 0], b[0, 1])
                    dy = o_raw[1] - np.clip(o_raw[1], b[1, 0], b[1, 1])
                    pairs.append((float(np.sqrt(dx * dx + dy * dy)), o_raw, o_D[2]))
                pairs.sort(key=lambda p: p[0])
                box_dist, best, D_req = pairs[0]
                term_xyz = [(b[0, 0] + b[0, 1]) / 2.0,
                            (b[1, 0] + b[1, 1]) / 2.0]
                label = (f"term_D obs=({best[0]:.2f},{best[1]:.2f},r={best[2]:.2f})"
                         f" D_req={D_req:.3f}  box_dist={box_dist:.3f}"
                         f"  term_center=({term_xyz[0]:.2f},{term_xyz[1]:.2f})")
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

#  Helpers

def _box_circle_overlap(bounds: np.ndarray, cx: float, cy: float, r: float) -> bool:
    dx = cx - np.clip(cx, bounds[0, 0], bounds[0, 1]) #center of circle to bound distance
    dy = cy - np.clip(cy, bounds[1, 0], bounds[1, 1])
    return dx * dx + dy * dy <= r * r


def _bounds_at(tester, t):
    """Return tight bound at timestep t, or None if missing."""
    h = tester.horizons.get(t)
    return h.get_tight_bound() if h is not None else None


def collides_danger(bounds: np.ndarray, obstacles: list,
                    R: float = 1.0) -> bool:
    """True if the box overlaps any obstacle's danger radius D = sqrt(r^2 + 2rR).
    r is the obstacle radius and R the turning radius (1 here). From outside D a
    max-turn loiter can keep clear of the obstacle.
    """
    return any(_box_circle_overlap(bounds, obs[0], obs[1],
                                   np.sqrt(obs[2]**2 + 2 * obs[2] * R))
               for obs in obstacles)


def collides_raw(bounds: np.ndarray, obstacles: list) -> bool:
    """Overlap against raw obstacle radius."""
    return any(_box_circle_overlap(bounds, obs[0], obs[1], obs[2])
               for obs in obstacles)


def scan_window(tester, mpc_sf, mpc_buffer: dict, obstacles: list,
                     wall_tau: int, scan_limit: int, budget,
                     R: float = 1.0, reasons: dict = None) -> Optional[int]:
    """
    Passthrough search: look past an infeasible build (a "wall") for a tau that
    has a valid backup. Returns that tau as the new t_diverge, or None.

    For each tau from wall_tau+1 to scan_limit (inclusive): give up if the RSOA
    there hits a raw obstacle, otherwise use an existing valid backup or try to
    build one. tau itself may be inside D; only a backup's terminal node must
    clear D. R is unused (kept for the call signature).
    """
    for tau in range(wall_tau + 1, scan_limit + 1):
        if tau not in tester.horizons:
            break
        bounds = tester.horizons[tau].get_tight_bound()
        if bounds is None:
            break
        if collides_raw(bounds, obstacles):
            print(f"  [Scan window] tau={tau} collides with raw obstacle — abort")
            return None
        if tau in mpc_buffer:
            if mpc_buffer[tau] is not None:
                print(f"  [Scan window] tau={tau} already VALID in buffer")
                return tau
            continue
        if budget.remaining < budget.mpc_cost:
            print(f"  [Scan window] Budget exhausted at tau={tau}")
            return None
        valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau, reasons)
        if valid:
            print(f"  [Scan window] tau={tau} VALID — passthrough t_diverge found")
            return tau
        else:
            print(f"  [Scan window] tau={tau} raw-clear but MPC INFEASIBLE — continue")
    return None

def _purge_infeasible(mpc_buffer: dict, after_t: int):
    """Delete INFEASIBLE (None) entries with key > after_t."""
    for k in list(mpc_buffer):
        if k > after_t and mpc_buffer[k] is None:
            del mpc_buffer[k]

def _max_valid_diverge(mpc_buffer: dict, from_t: int,
                       below: Optional[int] = None) -> Optional[int]:
    """Largest key >= from_t with a valid backup, or None.

    below (exclusive) is normally conflict_time: nominal is only verified safe
    before conflict_time, so a backup at or past it is not a passthrough target.
    """
    keys = [k for k, v in mpc_buffer.items()
            if k >= from_t and v is not None
            and (below is None or k < below)]
    return max(keys) if keys else None


# ── Simulation loop ────────────────────────────────────────────────────────

def test(seed=None, analyzer=None, obstacles=None):
    # Free the previous trial's acados solver first. Its library is loaded under
    # a per-process name, so a leftover reference cycle would make this trial
    # reuse the old solver with the previous trial's obstacles.
    if os.environ.get('TTTCARV_GC_PER_TRIAL', '1') != '0':
        import gc
        gc.collect()
    if analyzer is None:
        analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    if seed is None:
        seed = 1401830092

    if obstacles is None:
        obstacles = [
            np.array([-4.3,  1,   0.5]),
            np.array([-2,   -2,   0.5]),
            np.array([-1,    0.5, 0.5]),
            np.array([-3.5,  0,   0.5]),
        ]

    tester = ReachabilityTester(analyzer, obstacles, seed=seed)
    setmem_stats = _install_setmem(tester)
    est_stats = _install_estctrl(tester, setmem_stats)
    # TTTCARV_FIRST_STEP_COST=1: the stay-close-to-NN cost applies only to the
    # first control (the only one applied under receding horizon).
    first_step_cost = os.environ.get('TTTCARV_FIRST_STEP_COST', '0') != '0'
    # Small cost on the remaining controls when FIRST_STEP_COST is on, to keep
    # the QP well conditioned.
    tail_cost_weight = float(os.environ.get('TTTCARV_TAIL_COST_WEIGHT', '1e-3'))
    # TTTCARV_ONESHOT=1: a backup is one OCP solve that returns the solver's own
    # trajectory, instead of n_horizon solves re-simulated one step at a time.
    # Much cheaper, and the terminal-D constraint applies to the returned end.
    oneshot        = os.environ.get('TTTCARV_ONESHOT', '0') != '0'
    oneshot_refine = int(os.environ.get('TTTCARV_ONESHOT_REFINE', '2'))
    # Compare the one-shot trajectory with a plant re-simulation (diagnostic;
    # costs one extra rollout per solve).
    oneshot_verify = os.environ.get('TTTCARV_ONESHOT_VERIFY', '0') != '0'
    # SQP_RTI iterations per one-shot refine pass (final run: 10).
    oneshot_rti    = int(os.environ.get('TTTCARV_ONESHOT_RTI', '20'))
    mpc_sf = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=12,
                                    verify_input_bounds=True,
                                    verify_heading_bounds=True,
                                    use_safety_radius=True, split_terminal_D=True,
                                    first_step_cost=first_step_cost,
                                    tail_cost_weight=tail_cost_weight,
                                    oneshot=oneshot,
                                    oneshot_refine=oneshot_refine,
                                    oneshot_verify=oneshot_verify,
                                    oneshot_rti=oneshot_rti)
    sound_stats = _install_sound_mpc(mpc_sf, tester, seed)
    sound_stats.update(node_checked=0, node_truth_out=0, node_skipped=0)

    v_nom = tester.analyzer.cl_system.dynamics.vt
    u_max = 1.0
    turning_radius = v_nom / u_max

    MAX_SYMBOLIC_HORIZON = 5
    MAX_TIME             = 60

    budget = TimeBudget(timestep_budget=0.20)
    sym_costs, concrete_cost, mpc_cost = calibrated_costs(
        lambda: ReachabilityTester(analyzer, obstacles, seed=0),
        lambda t: (lambda: build_mpc_backup(mpc_sf, t, {}, 1)),
        MAX_SYMBOLIC_HORIZON,
    )
    budget.symbolic_costs = dict(sym_costs)
    budget.concrete_cost  = concrete_cost
    budget.mpc_cost       = mpc_cost

    # ── Calibration guard ─────────────────────────────────────────────────
    # mpc_cost is calibrated once per process and can be inflated by startup
    # contention. P1 spends concrete_cost on bounds before it checks
    # remaining >= mpc_cost, so if mpc_cost >= timestep_budget - concrete_cost
    # the t+1 build could never run and the filter would silently be off.
    # Clamp it below that.
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
    # ── re-anchor state (TTTCARV_REANCHOR) ────────────────────────────────
    conflict_found_at: Optional[int]             = None   # timestep P2 detected the live conflict
    reanchor_for:      Optional[int]             = None   # conflict_time the in-flight re-walk is re-verifying
    reanchor_done:     dict                      = {}   # conflict_time -> timestep of its last re-walk (thrash guard)
    reanchor_retry_taus: set                     = set()  # taus that were INFEASIBLE when the re-walk started

    # ── MPC execution state ───────────────────────────────────────────────
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'traj_bounds':   [],
        'needed':        False,
    }
    mpc_started      = False

    # alg28 end-to-end audit, at every site that applies a plan control: the
    # committed plan's box at the node being flown to must contain the true
    # state that arrives there. Controls that are not the plan's (the loiter)
    # are skipped.
    _orig_rsm = tester.real_state_mpc

    def _audited_real_state_mpc(start, control):
        node = None
        ca = mpc_state.get('committed_at')
        if ca is not None:
            i = start - ca
            q, tb = mpc_state.get('controls') or [], mpc_state.get('traj_bounds') or []
            if (0 <= i < len(q) and i + 1 < len(tb)
                    and np.allclose(np.asarray(q[i], dtype=float).flatten(),
                                    np.asarray(control, dtype=float).flatten())):
                node = np.asarray(tb[i + 1], dtype=float)
            else:
                sound_stats['node_skipped'] += 1
        res = _orig_rsm(start, control)
        if node is not None:
            x = next((np.asarray(c['real_state'], dtype=float).flatten()
                      for c in tester.horizons[start + 1].calculations.values()
                      if 'real_state' in c), None)
            if x is not None:
                sound_stats['node_checked'] += 1
                if np.any(x < node[:, 0] - 1e-9) or np.any(x > node[:, 1] + 1e-9):
                    sound_stats['node_truth_out'] += 1
                    print(f"{RED}[SOUND_MPC] truth outside plan node {start + 1 - ca}"
                          f" at t={start + 1}{RESET}")
        return res
    tester.real_state_mpc = _audited_real_state_mpc

    current_timestep = 0
    u_diffs          = []   # |u_mpc - u_nn| per timestep; 0.0 when nominal
    # Mode-based intervention (threshold-free): steps on which the filter
    # flew its OWN command (real_state_mpc), not the NN. Counts the
    # MPC_ACTIVE / activate / loiter branches; every real_state_empirical
    # branch (nominal, passthrough, revert, queue-empty, no-diverge) is a
    # pass-through and is not counted. The counterpart of alg29's pinned-
    # stage-0 bookkeeping, so the two are compared without a threshold.
    eng_steps        = 0
    # |applied - clip(NN, +-u_max)| per timestep. alg29's pin clips the NN
    # into the filter's input set, so a baseline is charged for the clip on
    # every step where |NN| > u_max; ours flies the NN unclipped and is not.
    # This list charges both sides against the same clipped reference.
    u_diffs_clip     = []
    nn_over          = 0    # timesteps whose NN command is outside +-u_max
    mpc_calls        = 0    # number of PSF activations
    psf_no_diverge   = 0    # PSF failed, no valid t_diverge at all
    # psf_no_diverge broken down by cause (never built vs rejected build).
    psf_no_diverge_reasons: dict = {}   # short key -> count
    p1_skipped_no_budget = 0   # P1 build refused by the budget gate
    concrete_deadline_hits = 0  # P2 scans stopped between steps at the deadline
    p1_skipped_no_bounds = 0   # P1 build impossible, concrete bounds missing
    psf_loiter_used      = 0   # no-backup step covered by the max-turn loiter
    psf_loiter_neg       = 0   # ... of which turned right (TERMINAL=loiter)
    loiter_side          = None  # side being flown; kept while it stays certified
    p1_bootstrap_resim   = 0   # no-fallback build forced onto the re-sim solver
    p1_bootstrap_infeasible = 0  # no-fallback build that still came back INFEASIBLE
    p1_t0_infeasible     = 0   # the t=0 build specifically -- the one with an empty buffer
    psf_unrecoverable    = 0   # no backup AND current RSOA already inside D
    psf_queue_empty  = 0    # PSF failed, t_diverge valid but queue empty after extend
    psf_stale_refused = 0   # PSF failed, only a stale (ctrl_idx>0) plan existed
    passthrough_found = 0   # P3b found a valid backup past a wall
    passthrough_used  = 0   # steps flown nominal with no backup at t+1 (one exists ahead)
    reanchor_resets      = 0   # walls that triggered a lookahead re-walk
    reanchor_cleared     = 0   # re-walks that reached conflict_time clear (conflict dropped)
    reanchor_reconfirmed = 0   # re-walks that hit a collision again
    reanchor_gain_steps  = 0   # sum of timesteps the new anchor is fresher than the old
    reanchor_retry_valid = 0   # INFEASIBLE -> VALID rebuilds at a tau a re-walk had purged
    reanchor_rearms      = 0   # re-walks that were a REARM of an already-re-anchored conflict
    conflict_retest_cleared = 0   # conflicts dropped by the P2a re-test
    conflict_retest_gain    = 0   # sum of timesteps saved vs waiting for STALE
    # Census (free, always on): how often is a live conflict's box ALREADY clear
    # of the danger radius? That is the entire opportunity for the re-test, and
    # it has to be measured before the re-test is worth having.
    rc_cflive_nom   = 0   # nominal timesteps with a live conflict
    rc_cfclear_nom  = 0   # ... of those, box at conflict_time already clear
    rc_cfnobox_nom  = 0   # ... of those, no box filed at conflict_time
    rc_cflive_mpc   = 0   # MPC-phase timesteps with a live conflict
    rc_cfclear_mpc  = 0   # ... of those, box already clear
    rc_cfnobox_mpc  = 0   # ... of those, no box filed
    # Wall census (free; answers "how often is a wall even on the table?").
    # Counted once per timestep at P3c, after P3b has had its chance to pass.
    rc_conflict_steps    = 0   # timesteps with a live conflict_time
    rc_wall_steps        = 0   # of those, timesteps where P3 left a wall P3b could not pass
    rc_blk_rearm         = 0   # walls refused only by the once-per-conflict / REARM gate
    rc_blk_fresh         = 0   # walls refused only by the "no fresher than the detection" gate
    rc_blk_tdiv          = 0   # walls refused only because t_diverge already reaches the conflict
    rc_blk_inflight      = 0   # walls refused because a re-walk was already in flight
    buffer_dropped_uncontained = 0  # backups dropped because their box was re-filed wider

    # ── Active-phase options ──────────────────────────────────────────────
    # TTTCARV_RECEDING=1 (default): in MPC_ACTIVE, re-solve from the current
    # bound every step and apply u[0]. 0 flies the committed plan open loop.
    receding = os.environ.get('TTTCARV_RECEDING', '1') != '0'
    # TTTCARV_PSF_TAIL=1 (final run): if a re-solve fails, fly the committed
    # plan without extending it while it is within its N-step horizon, the part
    # its terminal constraint covers. Off by default.
    psf_tail = os.environ.get('TTTCARV_PSF_TAIL', '0') != '0'
    mpc_resolved             = 0  # active timesteps whose input came from a fresh solve
    mpc_resolve_failed       = 0  # active timesteps that fell back to the committed plan
    mpc_fallback_uncertified = 0  # fallbacks where the current RSOA left traj_bounds[idx]
    mpc_tail_exhausted       = 0  # k - k_bar reached N: certificate spent, branch 3 due
    # Why re-solves failed (e.g. no_budget, rollout_rejected, solver_error).
    mpc_fail_reason: dict = {}
    # Which acceptance check rejected a re-solved plan, and at which node.
    mpc_reject_detail: dict = {}
    # Turn-direction flip diagnostics: does the controller keep switching which
    # way it goes around an obstacle?
    #   applied_u_*  sign of the applied input on consecutive active steps
    #   plan_*       sign of a new plan's mean vs the old plan's remaining tail
    # *_pairs counts comparisons where both values are outside FLIP_DEADBAND.
    applied_u_flips    = 0
    applied_u_pairs    = 0
    plan_flips         = 0
    plan_pairs         = 0
    _prev_applied_u    = None
    _prev_applied_t    = None
    # Active-phase backup builds skipped for lack of budget. Also flags a worker
    # whose mpc_cost calibration disabled the filter (see LAST_RUN_STATS below).
    mpc_skipped_no_budget = 0
    # Subset skipped only to keep RESOLVE_RESERVE.
    mpc_build_yielded_to_resolve = 0
    print(f"[alg22] receding-horizon active phase: {'ON' if receding else 'OFF (alg21 parity)'}"
          f"   first-step-only cost: {'ON' if first_step_cost else 'OFF'}"
          f"{f' (tail_w={tail_cost_weight:g})' if first_step_cost else ''}"
          f"   bounded tail (Alg1 branch 2): {'ON' if psf_tail else 'OFF'}")

    # ── RSOA certificate dump (only if TTTCARV_RSOA_DUMP is a path) ─────────
    # Every step goes through real_state_empirical or real_state_mpc, so wrapping
    # those two logs every step. Bounds are recorded before the step is applied.
    _rsoa_path = os.environ.get('TTTCARV_RSOA_DUMP')
    rsoa_log   = []
    # Per-timestep event tape for the animation: the re-anchor sites append to
    # it, the dump wrapper drains it. Cleared at the top of every iteration, so
    # it stays empty (and free) when no dump is requested.
    rsoa_events = []
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
                       'bound_t': _bnd(t), 'real_t': _real(t),
                       # re-anchor / scan state, so the animation can show WHY
                       # the buffer changes from one step to the next
                       'concrete_until': concrete_until, 'wall_tau': wall_tau,
                       'conflict_found_at': conflict_found_at,
                       'reanchor_for': reanchor_for,
                       'events': list(rsoa_events)}
                # The WHOLE backup buffer as the PSF decided it this step, not
                # just tau = t+1: tau -> plan boxes (x, y only, 3 decimals) or
                # None where the build came back INFEASIBLE. This is the fan of
                # escape routes the robot is holding at time t.
                rec['buffer'] = [
                    [tau, (None if v is None else
                           [[[round(float(b[0][0]), 3), round(float(b[0][1]), 3)],
                             [round(float(b[1][0]), 3), round(float(b[1][1]), 3)]]
                            for b in np.asarray(v[1])])]
                    for tau in sorted(mpc_buffer)
                    for v in (mpc_buffer[tau],)]
                if kind == 'mpc' and mpc_state['committed_at'] is not None:
                    # traj_bounds[k] is the plan's bound at committed_at + k; it
                    # may be missing if the plan was extended.
                    tb, i = mpc_state['traj_bounds'], t - mpc_state['committed_at']
                    rec['backup_idx']      = i
                    rec['backup_len']      = len(tb)
                    rec['backup_bound_t']  = (np.asarray(tb[i]).tolist()
                                              if 0 <= i < len(tb) else None)
                    rec['backup_bound_t1'] = (np.asarray(tb[i + 1]).tolist()
                                              if 0 <= i + 1 < len(tb) else None)
                    # the REST of the committed plan, so the animation can draw
                    # the route the robot is being flown along, not just one box
                    rec['backup_plan'] = [
                        [[round(float(b[0][0]), 3), round(float(b[0][1]), 3)],
                         [round(float(b[1][0]), 3), round(float(b[1][1]), 3)]]
                        for b in np.asarray(tb[max(i, 0):])]
                # Lookahead chain the filter decided on (before the step
                # overwrites t+1). EVERY filed future bound, not just t+1..t+20:
                # a conflict can sit further ahead than 20 steps, and capping
                # the window left the animation with nothing to point at.
                rec['look'] = [(k, _bnd(k)) for k in sorted(tester.horizons)
                               if k > t and _bnd(k) is not None]
                out = fn(t, arg)
                rec['bound_t1'] = _bnd(t + 1)
                rec['real_t1']  = _real(t + 1)
                # for the animation: measurement box, the MPC control applied
                # and the backup plan ready at t+1
                h1 = tester.horizons.get(t + 1)
                mb = getattr(h1, '_setmem_mb', None) if h1 is not None else None
                rec['meas_t1'] = None if mb is None else np.asarray(mb).tolist()
                rec['u'] = np.asarray(arg).flatten().tolist() if kind == 'mpc' else None
                nb = mpc_buffer.get(t + 1)
                rec['buffer_t1'] = (None if nb is None else
                                    [np.asarray(b).tolist() for b in nb[1]])
                rsoa_log.append(rec)
                return out
            return inner

        tester.real_state_empirical = _wrap(tester.real_state_empirical, 'empirical')
        tester.real_state_mpc       = _wrap(tester.real_state_mpc,       'mpc')

    # ── Per-timestep budget ledger (only if TTTCARV_BUDGET_DUMP is a path) ──
    # Records time spent on symbolic, concrete, plant-step and MPC work each
    # timestep, so steps can be scored with symbolic time taken out.
    _budget_path = os.environ.get('TTTCARV_BUDGET_DUMP')
    if _budget_path:
        budget.ledger_enabled       = True
        tester.symbolic             = budget.charged(tester.symbolic,   'symbolic')
        tester.concrete             = budget.charged(tester.concrete,   'concrete')
        tester.real_state_empirical = budget.charged(tester.real_state_empirical, 'step')
        tester.real_state_mpc       = budget.charged(tester.real_state_mpc,       'step')
        mpc_sf._run_mpc_from_bounds = budget.charged(
            mpc_sf._run_mpc_from_bounds, 'mpc')

    # ── Measured-cost budget (TTTCARV_MPC_PRICING=1, final run) ───────────
    # Off: an MPC solve runs if remaining >= mpc_cost from the one-time
    # calibration. On:
    #   - mpc_cost      = p99.5 of this process's recent solve times + margin
    #   - concrete_cost = p95 of measured seconds per concrete step + margin
    #   - every solve (including the second multi-start one) must fit before it starts
    #   - the RTI loop, re-sim rollout and P2 concrete scan stop at the deadline
    mpc_pricing = os.environ.get('TTTCARV_MPC_PRICING', '0') != '0'
    if mpc_pricing:
        import time_budget as _tbm
        budget.pricing        = True
        budget.margin         = float(os.environ.get('TTTCARV_BUDGET_MARGIN', '0.003'))
        mpc_sf.solve_observer = _tbm.record_solve
        mpc_sf.solve_gate     = lambda: budget.remaining >= budget.mpc_cost
        mpc_sf.time_left      = lambda: budget.remaining - budget.margin
    globals()['_CONCRETE_OBSERVER'] = _tbm.record_concrete if mpc_pricing else None
    globals()['_CONCRETE_DEADLINE'] = ((lambda: budget.remaining - budget.margin,
                                        lambda: budget.concrete_cost)
                                       if mpc_pricing else None)

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next   = current_timestep + 1
        wall_tau = None  # reset each iteration; P3 sets it if it hits INFEASIBLE
        rsoa_events.clear()

        # ══════════════════════════════════════════════════════════════════
        #  MPC ACTIVE PHASE
        # ══════════════════════════════════════════════════════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}  conflict={conflict_time}"
                  f"  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']
            # The robot has left the nominal trajectory: any in-flight re-walk
            # was anchored on a chain that no longer applies. Both the purge
            # below and the REVERT path rewind concrete_until themselves.
            reanchor_for      = None
            conflict_found_at = None
            # This branch purges the buffer wholesale; stale marks must not leak
            # into the next nominal tick and delete a legitimately rebuilt backup.
            est_stats.get('uncontained_ts', set()).clear()

            # The robot has left the nominal trajectory, so future bounds and the
            # backups built from them are stale: drop them. Then alternate one
            # concrete step and one backup build while budget allows; a valid
            # build at t+1 means we can revert to nominal.

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)

            # Keep the horizon counters consistent with the purge.
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            # Budget held back for the re-solve (only with RESOLVE_RESERVE).
            resolve_reserve = (budget.mpc_cost * (1.0 if budget.pricing else SINGLE_START_COST_FRAC)
                               if (receding and RESOLVE_RESERVE) else 0.0)
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                # One concrete step
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                # One backup build. The first one (at t+1) is the revert test,
                # so resolve_reserve never holds it back; later builds are
                # lookahead for after the revert.
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
                    # Running out of bounds is the normal exit; only count budget shortfalls.
                    if concrete_until in tester.horizons:
                        mpc_skipped_no_budget += 1
                        # Affordable on its own, skipped only to keep resolve_reserve.
                        if budget.remaining >= budget.mpc_cost:
                            mpc_build_yielded_to_resolve += 1
                    break  # no MPC budget remaining

            
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # Revert: a valid backup at t+1 means nominal is safe again.
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None

                # The trajectory changed, so restart symbolic refinement from t+1
                # (conflict_time is kept).
                pending_job = (RefinementTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                # Drop infeasible entries past t+1; valid ones stay as lookahead.
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep,
                                                       below=conflict_time)
                if t_diverge is None or t_diverge < t_next:
                    # conflict_time was computed for the old trajectory, so it must
                    # not exclude the backup at t+1 that was just verified.
                    print(f"  [PSF REVERT] ceiling conflict_time={conflict_time}"
                          f" excluded the just-verified backup at τ={t_next}"
                          f" (would give t_diverge={t_diverge}) — using τ={t_next}")
                    t_diverge = t_next
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                _unn = _get_nn_control(tester, current_timestep)
                nn_over += int(_unn is not None and abs(float(_unn[0])) > u_max)
                u_diffs_clip.append(0.0 if _unn is None else
                                   max(0.0, abs(float(_unn[0])) - u_max))
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            # ── Receding horizon: re-solve from the current bound, apply u[0] ──
            # The committed plan's bounds come from an older RSOA; re-solving keeps
            # the applied input backed by the set the robot is actually in.
            resolved = False
            if receding:
                ok, r_ctrls, r_traj, r_label = solve_plan_at(
                    mpc_sf, tester, current_timestep, budget)
                if (not ok and r_label != "no_budget"
                        and budget.remaining >= budget.mpc_cost):
                    # Single-start failed (not for budget): retry with multi-start.
                    ok, r_ctrls, r_traj, r_label = solve_plan_at(
                        mpc_sf, tester, current_timestep, budget, multi_start=True)
                if ok:
                    # Compare with the old plan's tail before overwriting it.
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
                                      f" plan reverses turn direction: old tail"
                                      f" mean {_a:+.3f} -> new plan mean"
                                      f" {_b:+.3f}{RESET}")
                    # Re-commit at the current step, so ctrl_idx is 0.
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
                # No fresh plan: keep flying the committed plan, extending it when
                # <= n_horizon controls remain. With PSF_TAIL, don't extend while
                # ctrl_idx < n_horizon (still covered by the plan's own terminal
                # constraint); past that, extend anyway and count it as
                # mpc_tail_exhausted.
                tail_ok = (receding and psf_tail
                           and ctrl_idx < mpc_sf.n_horizon
                           and ctrl_idx < len(queue))
                if receding and psf_tail and not tail_ok:
                    mpc_tail_exhausted += 1
                    print(f"{RED}[PSF TAIL] t={current_timestep} certificate spent"
                          f" (k-k_bar={ctrl_idx}, N={mpc_sf.n_horizon},"
                          f" |plan|={len(queue)}) — branch 3 (terminal controller)"
                          f" not implemented, extending as alg21 does"
                          f"  (seed={seed}){RESET}")
                if not tail_ok:
                    if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                        extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                            ctrl_idx=ctrl_idx, budget=budget)
                        queue = mpc_state['controls']

                    # Apply queued MPC control
                    if ctrl_idx >= len(queue):
                        # Should not happen.
                        print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                        break

                # The fallback input is only certified if the current RSOA lies
                # inside the bound the plan predicted for this step. Apply it
                # anyway, but count it.
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
            # Count flips only across consecutive active steps; a gap means
            # nominal ran in between.
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
            if conflict_time is not None:
                rc_cflive_mpc += 1
                _cb = _bounds_at(tester, conflict_time)
                if _cb is None:
                    rc_cfnobox_mpc += 1
                elif not collides_danger(_cb, obstacles, turning_radius):
                    rc_cfclear_mpc += 1
            u_nn = _get_nn_control(tester, current_timestep)
            u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
            eng_steps += 1
            nn_over += int(u_nn is not None and abs(float(u_nn[0])) > u_max)
            u_diffs_clip.append(
                float(np.abs(ctrl[0] - np.clip(u_nn[0], -u_max, u_max)))
                if u_nn is not None else 0.0)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
            continue

        # ═════════════════════════════════════════════════════════════════
        #  NOMINAL / PRE_CONFLICT PHASE
        # ═════════════════════════════════════════════════════════════════

        # Already past conflict_time without incident: clear the stale conflict.
        if conflict_time is not None and current_timestep > conflict_time:
            print(f"{GREEN}[STALE] conflict_time={conflict_time} is in the past"
                  f" (t={current_timestep}) — clearing{RESET}")
            rsoa_events.append(['conflict_stale', conflict_time])
            conflict_time     = None
            pending_job       = None
            reanchor_for      = None
            conflict_found_at = None

        phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
        color = BLUE if phase == "PRE_CONFLICT" else GREEN
        print(f"\n{color}CURRENT TIMESTEP ==== {current_timestep}"
              f"  concrete_until={concrete_until}  mpc_hz={mpc_horizon_until}"
              f"  conflict={conflict_time}  t_diverge={t_diverge}  [{phase}]{RESET}")

        # ---- alg27: the t+1 bound is the known-control step from t ----
        if 'refresh' in est_stats and est_stats['refresh'](current_timestep, mpc_buffer,
                                                           mpc_reason):
            t_diverge = _max_valid_diverge(mpc_buffer, current_timestep,
                                           below=conflict_time)

        # ---- P1: make sure mpc_buffer[t+1] exists (the safety check) ----

        # No backup can start at or past conflict_time: the RSOA there may already
        # overlap the obstacle, and build_mpc_backup doesn't check the start node.
        # Record the reason instead of building.
        past_wall = conflict_time is not None and t_next >= conflict_time
        if past_wall and t_next not in mpc_reason:
            mpc_reason[t_next] = (f"conflict_wall(t+1={t_next} >= conflict_time="
                                  f"{conflict_time}; RSOA there is not raw-clear)")

        # Build the backup at t+1 if it is missing.
        if t_next not in mpc_buffer and not past_wall:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            # Is there a later valid backup to fall back on if this build fails?
            _have_fallback = (t_diverge is not None
                              and t_diverge > current_timestep
                              and psf_valid(mpc_buffer, t_diverge))
            _afford = budget.remaining >= budget.mpc_cost
            if t_next not in tester.horizons:
                p1_skipped_no_bounds += 1
            elif _afford:
                # No fallback: use the slower re-sim solver, which fails less often
                # (BOOTSTRAP_RESIM). Restore the setting even if the build raises.
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
                # A fresh build is certified against the CURRENT box, so an earlier
                # stale mark for this tau no longer applies. Without this the drain
                # (which runs after P1) would delete the backup just built here.
                est_stats.get('uncontained_ts', set()).discard(t_next)
                # Diagnostics: failed builds with no fallback, and at t=0.
                if not _have_fallback and not valid:
                    p1_bootstrap_infeasible += 1
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

        # ---- P2a: re-test the flagged conflict against its CURRENT box ----
        # conflict_time is written once and then never re-examined. P2's ceiling
        # below IS conflict_time, so the scan stops exactly at the flagged node
        # and never revisits it; the only paths that drop a conflict are a
        # re-anchor re-walk (heavily gated, and off in modes B/J) and the STALE
        # check, which fires only after the robot has already driven past.
        #
        # Meanwhile the box at that node keeps being re-filed from fresher
        # certified sets and shrinks steadily. On seed 287580328 the box at
        # tau=29 goes from 0.78 x 2.26 at t=10 to 0.18 x 0.18 at t=28 -- clear of
        # the danger radius by t=28 -- yet the conflict stays on the books until
        # it goes stale at t=30. While it is on the books it caps t_diverge and
        # gates P1, so the filter intervenes for a collision that its own
        # tightened bounds no longer support.
        #
        # The test is the same predicate that raised the conflict, on the box
        # that is already filed: a handful of box-circle comparisons, free next
        # to a concrete step. Clearing drops only the FLAG -- concrete_until is
        # left where it is, so P2 resumes scanning forward from the old node and
        # re-raises a conflict if anything past it collides. Nothing before the
        # node is assumed any safer than the existing code already assumes.
        if conflict_time is not None:
            rc_cflive_nom += 1
            _cb = _bounds_at(tester, conflict_time)
            if _cb is None:
                rc_cfnobox_nom += 1
            elif not collides_danger(_cb, obstacles, turning_radius):
                rc_cfclear_nom += 1
        if CONFLICT_RETEST and conflict_time is not None:
            _cbox = _bounds_at(tester, conflict_time)
            if _cbox is not None and not collides_danger(_cbox, obstacles,
                                                         turning_radius):
                print(f"{GREEN}[RETEST] conflict_time={conflict_time}: its box no"
                      f" longer reaches the danger radius (t={current_timestep})"
                      f" — clearing{RESET}")
                rsoa_events.append(['conflict_retest_cleared', conflict_time])
                conflict_retest_cleared += 1
                conflict_retest_gain += conflict_time - current_timestep
                conflict_time     = None
                conflict_found_at = None
                pending_job       = None
                reanchor_for      = None
                _purge_infeasible(mpc_buffer, current_timestep)
                t_diverge = _max_valid_diverge(mpc_buffer, current_timestep)
                mpc_horizon_until = (t_diverge if t_diverge is not None
                                     else current_timestep)

        # ---- P2: concrete scan forward, up to conflict_time ----
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(
                concrete_until + budget.max_affordable_concrete(),
                scan_ceil, MAX_TIME
            )
            collision, ct, reached = concrete_scan_deadline(tester, concrete_until, end)
            if collision:
                conflict_time = ct
                # Always re-stamp: a re-walk that reconfirms has re-verified the
                # conflict from *this* anchor, so any later re-anchor should
                # measure its freshness gain from here, not from the first sighting.
                conflict_found_at = current_timestep
                print(f"  [P2] Conflict detected at t={conflict_time}")
                # A t_diverge set before this conflict was known may be at or
                # past it. Nominal is only verified before conflict_time, so
                # pull t_diverge back below it.
                if t_diverge is not None and t_diverge >= conflict_time:
                    _stale = t_diverge
                    t_diverge = _max_valid_diverge(mpc_buffer, current_timestep,
                                                   below=conflict_time)
                    print(f"  [P2] t_diverge={_stale} is at/beyond the new conflict"
                          f" — clamped to {t_diverge}")
                if pending_job is None:
                    pending_job = RefinementTask(current_timestep, conflict_time)
                if reanchor_for is not None:
                    # The re-walk hit a collision too, so the conflict stands.
                    # The boxes are still tighter than the ones the failed builds
                    # were rejected against, so drop the INFEASIBLE entries and
                    # let P3 retry them (P4's "conflict confirmed" branch).
                    reanchor_reconfirmed += 1
                    rsoa_events.append(['reanchor_reconfirmed', reanchor_for])
                    reanchor_for = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = (t_diverge if t_diverge is not None
                                         else current_timestep)
                    print(f"  [REANCHOR] conflict reconfirmed at t={ct} —"
                          f" retrying INFEASIBLE entries against the tighter chain")
                concrete_until = ct
                break
            concrete_until = reached
            if reached < end:  # stopped early at the deadline
                concrete_deadline_hits += 1
                break

        # ---- drop backups whose box was replaced by one it does not contain ----
        # Unconditional: any re-filed lookahead bound can invalidate the plan
        # stored there, whether or not a re-anchor put it there.
        _stale_ts = est_stats.get('uncontained_ts')
        if _stale_ts:
            for _t in sorted(_stale_ts):
                if _t in mpc_buffer:
                    del mpc_buffer[_t]
                    mpc_reason.pop(_t, None)
                    buffer_dropped_uncontained += 1
                    mpc_horizon_until = min(mpc_horizon_until, _t - 1)
                    print(f"  [REANCHOR] backup at tau={_t} dropped:"
                          f" the rebuilt box is not inside the one it was verified for")
            _stale_ts.clear()
            t_diverge = _max_valid_diverge(mpc_buffer, current_timestep,
                                           below=conflict_time)

        # The re-walk got all the way to the conflict without a collision: the
        # conflict was an artefact of the older, more wrapped chain. Clear it and
        # let P3 retry the builds that were rejected against the fat boxes. A
        # re-walk that ran out of budget leaves reanchor_for set and resumes next
        # timestep; the untouched boxes past the frontier stay sound meanwhile.
        if reanchor_for is not None and concrete_until >= reanchor_for:
            print(f"{GREEN}[REANCHOR] Deconflicted — re-walk reached t={reanchor_for}"
                  f" clear from anchor t={current_timestep}{RESET}")
            reanchor_cleared += 1
            rsoa_events.append(['reanchor_cleared', reanchor_for])
            reanchor_for      = None
            conflict_time     = None
            conflict_found_at = None
            pending_job       = None
            _purge_infeasible(mpc_buffer, current_timestep)
            mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ---- P3: build backups forward toward conflict_time ----
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
                est_stats.get('uncontained_ts', set()).discard(tau)
                mpc_horizon_until = tau
                if valid:
                    t_diverge = tau
                    if tau in reanchor_retry_taus:
                        reanchor_retry_valid += 1
                        reanchor_retry_taus.discard(tau)
                    print(f"  [P3] mpc_buffer[{tau}] VALID  t_diverge → {t_diverge}")
                else:
                    print(f"  [P3] mpc_buffer[{tau}] INFEASIBLE — wall at tau={tau}")
                    wall_tau = tau
                    break

        # ---- P3b: passthrough scan past an infeasible build (a "wall") ----
        # One infeasible tau says little about the next, so look further ahead.
        if wall_tau is not None and budget.remaining >= budget.mpc_cost:
            scan_limit = (min(conflict_time - 1, concrete_until)
                          if conflict_time is not None
                          else concrete_until)
            passthrough_td = scan_window(
                tester, mpc_sf, mpc_buffer, obstacles,
                wall_tau, scan_limit, budget, R=turning_radius,
                reasons=mpc_reason)
            if passthrough_td is not None:
                t_diverge = passthrough_td
                mpc_horizon_until = passthrough_td
                passthrough_found += 1
                print(f"  [P3b] Passthrough t_diverge -> {t_diverge}")

        # ---- P3c: re-anchor the lookahead at a wall P3b could not pass ----
        # Fires once per conflict (reanchor_done), and only after at least one
        # timestep has elapsed since the conflict was found -- otherwise the
        # re-walk would start from the same certified set and reproduce the same
        # boxes at ~one CROWN call per step for nothing. With REANCHOR_REARM=k
        # the same conflict may be re-walked again every k timesteps, which is
        # what spends the idle budget of a reconfirmed conflict.
        _last_reanchor = reanchor_done.get(conflict_time) if conflict_time is not None else None
        if conflict_time is not None:
            rc_conflict_steps += 1
            if wall_tau is not None:
                rc_wall_steps += 1
                # Attribute the refusal to the FIRST gate that would stop it, in
                # the same order the condition below tests them. rc_blk_rearm is
                # the one that matters: it is exactly how many more re-walks a
                # fire-on-every-wall policy would buy.
                if reanchor_for is not None:
                    rc_blk_inflight += 1
                elif not (_last_reanchor is None
                          or (REANCHOR_REARM > 0
                              and current_timestep - _last_reanchor >= REANCHOR_REARM)):
                    rc_blk_rearm += 1
                elif not (conflict_found_at is not None
                          and current_timestep > conflict_found_at):
                    rc_blk_fresh += 1
                elif not (t_diverge is None or t_diverge < conflict_time - 1):
                    rc_blk_tdiv += 1
        if REANCHOR_DEBUG and REANCHOR and conflict_time is not None:
            print(f"  [REANCHOR?] wall_tau={wall_tau} reanchor_for={reanchor_for}"
                  f" last={_last_reanchor} found_at={conflict_found_at}"
                  f" t_diverge={t_diverge} conflict={conflict_time}")
        _rearm_ok = (_last_reanchor is None
                     or (REANCHOR_REARM > 0
                         and current_timestep - _last_reanchor >= REANCHOR_REARM))
        if (REANCHOR and wall_tau is not None and conflict_time is not None
                and reanchor_for is None
                and _rearm_ok
                and conflict_found_at is not None
                and current_timestep > conflict_found_at
                and (t_diverge is None or t_diverge < conflict_time - 1)):
            reanchor_for = conflict_time
            rsoa_events.append(['reanchor_fire', wall_tau, conflict_time,
                                current_timestep - conflict_found_at,
                                _last_reanchor is not None])
            if _last_reanchor is not None:
                reanchor_rearms += 1
            reanchor_done[conflict_time] = current_timestep
            reanchor_resets += 1
            reanchor_gain_steps += current_timestep - conflict_found_at
            reanchor_retry_taus = {k for k, v in mpc_buffer.items()
                                   if v is None and k > current_timestep}
            # Rewind the scan pointer only. The filed boxes stay where they are:
            # they are still sound, just loose, so P1 can keep building at t+1
            # while the re-walk is in flight, and concrete() overwrites each one
            # as it passes. conflict_time deliberately stays set -- it is the only
            # thing stopping P1 from building a backup that starts at a node
            # nothing has cleared (past_wall), and it caps _max_valid_diverge and
            # the passthrough assert. It is dropped only once the re-walk has
            # actually proved that node clear.
            concrete_until = current_timestep
            print(f"{BLUE}[REANCHOR] wall at tau={wall_tau}, t_diverge={t_diverge}"
                  f" < conflict_time={conflict_time} — re-walking the lookahead"
                  f" from t={current_timestep}"
                  f" ({current_timestep - conflict_found_at} timesteps fresher than"
                  f" the chain that found it){RESET}")

        # ---- P4: symbolic refinement (keeps going after a confirmed conflict) ----
        if (pending_job is not None and ESTCTRL not in ('B', 'J')
                and budget.can_afford('symbolic', 1)):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    # Conflict confirmed with tighter bounds: drop infeasible
                    # entries so P3 retries them (valid entries stay).
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    # Pull mpc_horizon_until back to t_diverge so P3 re-scans the
                    # purged gap with tightened bounds on the next iteration.
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    # Reset job for continuous refinement
                    pending_job       = RefinementTask(current_timestep, conflict_time)
                else:
                    # Deconflicted: clear the conflict but keep valid backups.
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    rsoa_events.append(['conflict_deconflicted', conflict_time])
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION: nominal or activate MPC backup
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            # PSF OK: nominal is safe. Record t+1 as a divergence point (only ever
            # raising t_diverge) so the clock never overtakes it; otherwise a
            # later activation could join a plan partway through.
            if t_diverge is None or t_diverge < t_next:
                t_diverge = t_next
            u_diffs.append(0.0)
            _unn = _get_nn_control(tester, current_timestep)
            nn_over += int(_unn is not None and abs(float(_unn[0])) > u_max)
            u_diffs_clip.append(0.0 if _unn is None else
                               max(0.0, abs(float(_unn[0])) - u_max))
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

        elif (PASSTHROUGH and t_diverge is not None
                and t_diverge > current_timestep
                and psf_valid(mpc_buffer, t_diverge)):
            # Passthrough: no backup at t+1, but a valid one exists at t_diverge.
            # Flying nominal until then is safe only because the concrete scan
            # showed every step before conflict_time is obstacle-free.
            assert conflict_time is None or t_diverge < conflict_time, (
                f"PASSTHROUGH gap [{current_timestep},{t_diverge}] spans the "
                f"conflict at t={conflict_time}: nominal is not concrete-certified "
                f"over this gap")
            # Log why there is no backup at t+1.
            if t_next in mpc_reason:
                _why = f"no backup at t+1: {mpc_reason[t_next]}"
            elif t_next not in mpc_buffer:
                _why = "no backup at t+1: never built (budget or bounds missing)"
            else:
                _why = "no backup at t+1: reason unrecorded"

            # Log what makes nominal safe this step.
            _basis = (f"nominal concrete-clear through t={concrete_until} (no conflict)"
                      if conflict_time is None
                      else f"nominal concrete-clear to conflict t={conflict_time}")

            print(f"  [PSF PASSTHROUGH] t={current_timestep} — {_why}; "
                  f"t_diverge={t_diverge} valid ahead; {_basis}"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            passthrough_used += 1
            u_diffs.append(0.0)
            _unn = _get_nn_control(tester, current_timestep)
            nn_over += int(_unn is not None and abs(float(_unn[0])) > u_max)
            u_diffs_clip.append(0.0 if _unn is None else
                               max(0.0, abs(float(_unn[0])) - u_max))
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

        else:
            # PSF failed: switch to a backup plan.
            # A plan only describes the robot from the step it was built for
            # (controls[k] assumes controls[0..k-1] were applied), so use the
            # backup for this exact step and refuse a stale t_diverge.
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
                    eng_steps += 1
                    nn_over += int(u_nn is not None and abs(float(u_nn[0])) > u_max)
                    u_diffs_clip.append(
                        float(np.abs(ctrl[0] - np.clip(u_nn[0], -u_max, u_max)))
                        if u_nn is not None else 0.0)
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
                        eng_steps += 1
                        nn_over += int(u_nn is not None and abs(float(u_nn[0])) > u_max)
                        u_diffs_clip.append(
                            float(np.abs(ctrl[0] - np.clip(u_nn[0], -u_max, u_max)))
                            if u_nn is not None else 0.0)
                        tester.real_state_mpc(current_timestep, ctrl)
                        current_timestep += 1
                    else:
                        print(f"[PSF] Cannot activate — queue still empty. Nominal fallback."
                              f"  (seed={seed}, t={current_timestep})")
                        psf_queue_empty += 1
                        u_diffs.append(0.0)
                        _unn = _get_nn_control(tester, current_timestep)
                        nn_over += int(_unn is not None and abs(float(_unn[0])) > u_max)
                        u_diffs_clip.append(0.0 if _unn is None else
                                           max(0.0, abs(float(_unn[0])) - u_max))
                        tester.real_state_empirical(current_timestep, t_next)
                        current_timestep += 1
            else:
                # No usable backup at all (should not happen in a healthy run).
                _bnext   = tester.horizons[t_next].get_tight_bound() if t_next in tester.horizons else None
                _in_D    = (_bnext is not None and collides_danger(_bnext, obstacles, turning_radius))
                _in_raw  = (_bnext is not None and collides_raw(_bnext, obstacles))
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
                      f"  bnext_in_D={_in_D}  bnext_in_raw={_in_raw}"
                      f"  conflict={conflict_time}  mpc_hz={mpc_horizon_until}"
                      f"  (seed={seed}){RESET}")
                # Terminal safe controller (see PSF_LOITER). Uses the CURRENT RSOA;
                # the next-step RSOA assumes NN control, not the loiter.
                _bcur = (tester.horizons[current_timestep].get_tight_bound()
                         if current_timestep in tester.horizons else None)
                if _msfa.TERMINAL == 'loiter':
                    # Certified: a side whose loiter from EVERY state in the
                    # current set clears every obstacle forever. The plan that
                    # just ran out ended in such a set (its terminal test).
                    _side = (None if _bcur is None
                             else mpc_sf._loiter_side(_bcur, prefer=loiter_side))
                    _cur_clears_D = _side is not None
                else:
                    _side = 1
                    _cur_clears_D = (_bcur is not None
                                     and not collides_danger(_bcur, obstacles,
                                                             turning_radius))
                if PSF_LOITER and _cur_clears_D:
                    loiter_side = _side
                    psf_loiter_neg += int(_side < 0)
                    _u_loiter = np.array([u_max * _side])
                    print(f"{RED}[PSF LOITER] t={current_timestep} — no backup;"
                          f" current RSOA clears D, holding max-turn loiter"
                          f" u={_u_loiter}  (seed={seed}){RESET}")
                    psf_loiter_used += 1
                    u_nn = _get_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(_u_loiter[0] - u_nn[0]))
                                   if u_nn is not None else 0.0)
                    eng_steps += 1
                    nn_over += int(u_nn is not None and abs(float(u_nn[0])) > u_max)
                    u_diffs_clip.append(
                        float(np.abs(_u_loiter[0] - np.clip(u_nn[0], -u_max, u_max)))
                        if u_nn is not None else 0.0)
                    tester.real_state_mpc(current_timestep, _u_loiter)
                    current_timestep += 1
                    continue

                if not _cur_clears_D:
                    psf_unrecoverable += 1

                psf_no_diverge += 1
                _nd_key = ('never_built' if _buf_t1 == 'absent'
                           else 'rejected:' + str(mpc_reason.get(t_next, 'unrecorded')).split('(')[0]
                           if t_next in mpc_reason
                           else 'built_none' if _buf_t1 is None
                           else 'unrecorded')
                psf_no_diverge_reasons[_nd_key] = psf_no_diverge_reasons.get(_nd_key, 0) + 1
                u_diffs.append(0.0)
                _unn = _get_nn_control(tester, current_timestep)
                nn_over += int(_unn is not None and abs(float(_unn[0])) > u_max)
                u_diffs_clip.append(0.0 if _unn is None else
                                   max(0.0, abs(float(_unn[0])) - u_max))
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
                       'turning_radius': turning_radius,
                       'obstacles': [np.asarray(o).flatten().tolist() for o in obstacles],
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

    collision_timesteps = []
    for i, s in enumerate(state_history):
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx) ** 2 + (s_flat[1] - cy) ** 2 <= r ** 2:
                collision_timesteps.append(i)
                break

    had_collision = len(collision_timesteps) > 0
    print(f"\n{'=' * 60}")
    print(f"Simulation complete at timestep {current_timestep}")
    if had_collision:
        print(f"[SAFETY] COLLISION at real-state timesteps: {collision_timesteps}")
    else:
        print(f"[SAFETY] No real-state collision detected")
    print(f"  mpc_calls={mpc_calls}  psf_no_diverge={psf_no_diverge}"
          f"  psf_queue_empty={psf_queue_empty}"
          f"  psf_stale_refused={psf_stale_refused}"
          f"  passthrough_found={passthrough_found}"
          f"  passthrough_used={passthrough_used}"
          f"  reanchor={reanchor_resets}/{reanchor_cleared}/{reanchor_reconfirmed}"
          f"/{reanchor_retry_valid}"
          f"  reanchor_rearms={reanchor_rearms}"
          f"  retest={conflict_retest_cleared}/{conflict_retest_gain}"
          f"  cfcensus_nom={rc_cfclear_nom}/{rc_cfnobox_nom}/{rc_cflive_nom}"
          f"  cfcensus_mpc={rc_cfclear_mpc}/{rc_cfnobox_mpc}/{rc_cflive_mpc}"
          f"  wallcensus={rc_wall_steps}/{rc_conflict_steps}"
          f" blk={rc_blk_rearm}/{rc_blk_fresh}/{rc_blk_tdiv}/{rc_blk_inflight}"
          f"  max_u_diff={max(u_diffs, default=0):.3f}")
    print(f"  p1_bootstrap_resim={p1_bootstrap_resim}"
          f"  p1_bootstrap_infeasible={p1_bootstrap_infeasible}"
          f"  p1_t0_infeasible={p1_t0_infeasible}"
          f"  psf_loiter_used={psf_loiter_used}")
    print(f"  [LOOKAHEAD] jsplit={JSPLIT if ESTCTRL == 'J' else '-'}"
          f"  ebox={est_stats['ebox']}"
          f"  ebox_tightened={est_stats['ebox_tightened']}"
          f"  ebox_repair_fb={est_stats['ebox_repair_fallback']}"
          f"  ebox_hw_th_max={est_stats['ebox_hw_theta_max']:.4f}"
          f"  look_checked={est_stats['look_checked']}"
          f"  look_truth_out={est_stats['look_truth_out']}"
          f"  look_skipped_mpc={est_stats['look_skipped_mpc']}"
          f"  joint_wall_max={est_stats['joint_wall_max']:.4f}s")
    print(f"  [ESTCTRL] mode={est_stats['mode']}"
          f"  known={est_stats['known_steps']}"
          f"  lookahead={est_stats['lookahead_steps']}"
          f"  refresh_in={est_stats['refresh_contained']}"
          f"  refresh_out={est_stats['refresh_uncontained']}"
          f"  invalidated={est_stats['buffer_invalidated']}"
          f"  sym_restored={est_stats['sym_next_restored']}"
          f"  gap_max={est_stats['ctrl_gap_max']:.4f}"
          f"  gap_mean={est_stats['ctrl_gap_sum'] / max(est_stats['ctrl_gap_n'], 1):.4f}"
          f"  nn_w_max={est_stats['nn_width_max']:.4f}")
    print(f"  [SETMEM] on={int(SETMEM)}  reset={int(KF_RESET)}  vmax={VMAX_SIGMA:.1f}sigma"
          f"  meas_w={setmem_stats['meas_width']:.3f}"
          f"  steps={setmem_stats['steps']}"
          f"  truth_out_tight={setmem_stats['truth_out_tight']}"
          f"  truth_out_meas={setmem_stats['truth_out_meas']}"
          f"  empty={setmem_stats['empty_intersect']}"
          f"  repaired={setmem_stats['empty_repaired']}"
          f"  unrepairable={setmem_stats['empty_unrepairable']}"
          f"  meas_sub={setmem_stats['meas_substituted']}"
          f"  mpc_steps={setmem_stats['mpc_interval_steps']}"
          f"  no_parent={setmem_stats['no_parent_bound']}"
          f"  mb_stash={setmem_stats['mb_stash']}"
          f"  mb_fallback={setmem_stats['mb_fallback']}"
          f"  post_add={setmem_stats['post_add']}"
          f"  post_set={setmem_stats['post_set']}"
          f"  truth_out_post={setmem_stats['truth_out_post']}"
          f"  tight_w_max={setmem_stats['tight_width_max']:.4f}")
    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update({
        'setmem':                   int(SETMEM),
        'setmem_kf_reset':          int(KF_RESET),
        'setmem_steps':             setmem_stats['steps'],
        'setmem_truth_out_tight':   setmem_stats['truth_out_tight'],
        'setmem_truth_out_meas':    setmem_stats['truth_out_meas'],
        'setmem_empty_intersect':   setmem_stats['empty_intersect'],
        'setmem_empty_repaired':    setmem_stats['empty_repaired'],
        'setmem_empty_unrepairable': setmem_stats['empty_unrepairable'],
        'setmem_meas_substituted':  setmem_stats['meas_substituted'],
        'setmem_mpc_interval_steps': setmem_stats['mpc_interval_steps'],
        'setmem_no_parent_bound':   setmem_stats['no_parent_bound'],
        'setmem_mb_stash':          setmem_stats['mb_stash'],
        'setmem_mb_fallback':       setmem_stats['mb_fallback'],
        'setmem_post_add':          setmem_stats['post_add'],
        'setmem_post_set':          setmem_stats['post_set'],
        'setmem_truth_out_post':    setmem_stats['truth_out_post'],
        'setmem_tight_width_max':   float(setmem_stats['tight_width_max']),
        'setmem_meas_width':        float(setmem_stats['meas_width']),
        'setmem_meas_delay':        setmem_stats['meas_delay'],
        'setmem_delay_first':       setmem_stats['delay_first'],
        'setmem_delay_empty':       setmem_stats['delay_empty'],
        'est_mode':                 est_stats['mode'],
        'est_known_steps':          est_stats['known_steps'],
        'est_lookahead_steps':      est_stats['lookahead_steps'],
        'est_refresh_contained':    est_stats['refresh_contained'],
        'est_refresh_uncontained':  est_stats['refresh_uncontained'],
        'est_buffer_invalidated':   est_stats['buffer_invalidated'],
        'est_sym_next_restored':    est_stats['sym_next_restored'],
        'est_ctrl_gap_max':         float(est_stats['ctrl_gap_max']),
        'est_ctrl_gap_mean':        float(est_stats['ctrl_gap_sum']
                                          / max(est_stats['ctrl_gap_n'], 1)),
        'est_nn_width_max':         float(est_stats['nn_width_max']),
        'est_jsplit':               ','.join(map(str, JSPLIT)) if ESTCTRL == 'J' else '',
        'est_look_checked':         est_stats['look_checked'],
        'est_look_truth_out':       est_stats['look_truth_out'],
        'est_look_skipped_mpc':     est_stats['look_skipped_mpc'],
        'est_joint_wall_max':       float(est_stats['joint_wall_max']),
        'ebox':                     est_stats['ebox'],
        'est_ebox_tightened':       est_stats['ebox_tightened'],
        'est_ebox_repair_fb':       est_stats['ebox_repair_fallback'],
        'est_ebox_hw_theta_max':    float(est_stats['ebox_hw_theta_max']),
        'usat':                     int(USAT),
        'est_sat_steps':            est_stats['sat_steps'],
        'est_u_raw_max':            float(est_stats['u_raw_max']),
        'receding':                 int(receding),
        'passthrough':              int(PASSTHROUGH),
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
        # mpc_cost_clamped=1: this process's calibration was clamped. A trial
        # with mpc_calls==0 and mpc_skipped_no_budget>0 flew unfiltered and
        # should not be scored.
        'mpc_cost_clamped':         mpc_cost_clamped,
        'mpc_skipped_no_budget':    mpc_skipped_no_budget,
        'mpc_build_yielded':        mpc_build_yielded_to_resolve,
        # Unfiltered steps where the filter wanted to intervene (should be 0), by cause.
        'psf_no_diverge_reasons':   dict(psf_no_diverge_reasons),
        'psf_stale_refused':        psf_stale_refused,
        'reanchor':                 int(REANCHOR),
        'reanchor_resets':          reanchor_resets,
        'reanchor_cleared':         reanchor_cleared,
        'reanchor_reconfirmed':     reanchor_reconfirmed,
        'reanchor_gain_steps':      reanchor_gain_steps,
        'reanchor_retry_valid':     reanchor_retry_valid,
        'reanchor_rearm':           REANCHOR_REARM,
        'reanchor_rearms':          reanchor_rearms,
        'conflict_retest':          int(CONFLICT_RETEST),
        'conflict_retest_cleared':  conflict_retest_cleared,
        'conflict_retest_gain':     conflict_retest_gain,
        'rc_cflive_nom':            rc_cflive_nom,
        'rc_cfclear_nom':           rc_cfclear_nom,
        'rc_cfnobox_nom':           rc_cfnobox_nom,
        'rc_cflive_mpc':            rc_cflive_mpc,
        'rc_cfclear_mpc':           rc_cfclear_mpc,
        'rc_cfnobox_mpc':           rc_cfnobox_mpc,
        'rc_conflict_steps':        rc_conflict_steps,
        'rc_wall_steps':            rc_wall_steps,
        'rc_blk_rearm':             rc_blk_rearm,
        'rc_blk_fresh':             rc_blk_fresh,
        'rc_blk_tdiv':              rc_blk_tdiv,
        'rc_blk_inflight':          rc_blk_inflight,
        'buffer_dropped_uncontained': buffer_dropped_uncontained,
        'p1_skipped_no_budget':     p1_skipped_no_budget,
        'p1_skipped_no_bounds':     p1_skipped_no_bounds,
        'psf_loiter_used':          psf_loiter_used,
        'psf_loiter_neg':           psf_loiter_neg,
        'terminal':                 _msfa.TERMINAL,
        'p1_bootstrap_resim':       p1_bootstrap_resim,
        'p1_bootstrap_infeasible':  p1_bootstrap_infeasible,
        'p1_t0_infeasible':         p1_t0_infeasible,
        'bootstrap_resim':          int(BOOTSTRAP_RESIM),
        'psf_unrecoverable':        psf_unrecoverable,
        'psf_loiter':               int(PSF_LOITER),
        'resolve_reserve':          int(RESOLVE_RESERVE),
        'mpc_cost_s':               float(budget.mpc_cost),
        'mpc_cost_cal_s':           float(budget._mpc_cost_cal),
        'concrete_cost_s':          float(budget.concrete_cost),
        'concrete_cost_cal_s':      float(budget._concrete_cost_cal),
        'mpc_pricing':              int(mpc_pricing),
        'oneshot_rti':              int(oneshot_rti),
        'mpc_deadline_hits':        int(mpc_sf.deadline_hits),
        'mpc_truncated_solves':     int(mpc_sf.truncated_solves),
        'mpc_second_start_skipped': int(mpc_sf.second_start_skipped),
        'ub_single_solves':         int(getattr(mpc_sf, 'single_ub_solves', 0)),
        'ub_single_over':           int(getattr(mpc_sf, 'single_ub_over', 0)),
        'ub_single_peak':           float(getattr(mpc_sf, 'single_ub_peak', 0.0)),
        'ub_acc_resolve':           int(getattr(mpc_sf, 'acc_ub_resolve', 0)),
        'ub_acc_extend':            int(getattr(mpc_sf, 'acc_ub_extend', 0)),
        'ub_acc_peak':              float(getattr(mpc_sf, 'acc_ub_peak', 0.0)),
        'concrete_deadline_hits':   int(concrete_deadline_hits),
        # Size of the intervention |u_mpc - u_nn| on active steps.
        'u_diff_max':               float(max(u_diffs, default=0.0)),
        'u_diff_mean_active':       float(
            sum(d for d in u_diffs if d > 0.0)
            / max(sum(1 for d in u_diffs if d > 0.0), 1)),
        'u_diff_n_active':          int(sum(1 for d in u_diffs if d > 0.0)),
        # Same thresholds as alg29's 'gp_u_diff_gt', for a like-for-like
        # intervention rate (bookkeeping only).
        'eng_steps':                int(eng_steps),
        'nn_over':                  int(nn_over),
        'u_diff_clip_gt':           {str(t): int(sum(1 for d in u_diffs_clip if d > t))
                                     for t in (0.0, 0.01, 0.05, 0.1)},
        'u_diff_clip_len':          int(len(u_diffs_clip)),
        'u_diff_gt':                {str(t): int(sum(1 for d in u_diffs if d > t))
                                     for t in (0.0, 0.01, 0.05, 0.1)},
        'u_diff_len':               len(u_diffs),
        **{f'snd_{k}': (float(v) if isinstance(v, float) else int(v))
           for k, v in sound_stats.items()},
        'noise':                    NOISE,
        'trunc_ratio':              float(TRUNC_RATIO),
        'vmax_sigma':               VMAX_SIGMA,
        **{f'fail_{k}': v for k, v in mpc_fail_reason.items()},
        **{f'rej_{k}': v for k, v in mpc_reject_detail.items()},
    })
    print(f"  [SOUND_MPC] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in sound_stats.items())
          + f"  noise={NOISE}")
    print(f"  [alg22] receding={'on' if receding else 'off'}"
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
    obs = None
    if len(sys.argv) > 2:
        # Parse obstacles: x1,y1,r1 x2,y2,r2 ...
        obs = [np.array([float(v) for v in a.split(',')]) for a in sys.argv[2:]]
    test(seed=seed, obstacles=obs)

