"""
alg25 (DI) — alg14_di's phase portrait with the estimator box removed from the
certified bounds.

Why
---
In alg14_di both box stacks in di_phase_portrait.png are state ESTIMATES, not
reachable sets. A horizon's tight_bound is the intersection of everything filed
at it (ReachableSetHorizon.add_calculation), and real_state_empirical /
real_state_mpc file a Gaussian 3-sigma EKF box at every executed step, so each
drawn box has been conditioned on that step's measurement. run_nominal's
docstring says so outright and warns that the two stacks are different
quantities. A Gaussian confidence region is not a guaranteed over-approximation,
so intersecting it in is unsound -- measured on the unicycle (alg24): the true
state was outside the intersected bound in 2491/2500 trials.

This module removes that intersection, so both stacks become genuine forward
reachable sets of the same initial set, under the same closed-loop dynamics.
They then ARE the same quantity and can be read against each other.

What it changes (install() patches a live ReachabilityTester)
------------------------------------------------------------
1. EMPIRICAL calculations no longer tighten tight_bound. The true state and the
   EKF box are still filed -- real_state_mpc finds its parent through them and
   the filter still runs -- they just do not enter the certified bound.
2. An MPC step is bounded by _interval_step, the exact interval hull of the
   box's image under x' = A x + B u. Not the constant-size traj_bounds box:
   that is a fixed half-diagonal around the planned point and is not an
   over-approximation once the box grows.
3. A step whose horizon has no bound is filled by one concrete (CROWN) step.
   This is what the EKF box was silently doing in the unfiltered arm, which
   files nothing else; without it run_nominal would draw no boxes at all.

Soundness of 2: the double integrator has At = [[1, dt], [0, 1]] and
bt = [[dt^2/2], [dt]] (DoubleIntegrator.__init__), both entrywise non-negative,
so the map is monotone and the image box corners are exactly A @ lo + B u and
A @ hi + B u. Unlike the unicycle's cos/sin enclosure this loses nothing.
"""

import numpy as np
import torch

from REAL_integrated_sim import CalculationType


def _interval_step(b, u, dt):
    """Interval hull of the image of box b under x' = A x + B u.

    Exact for the double integrator: A >= 0 entrywise, so the hull corners are
    the images of b's own corners.
    """
    b = np.asarray(b, dtype=float)
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([0.5 * dt * dt, dt])
    u = float(np.asarray(u, dtype=float).flatten()[0])
    lo = A @ b[:, 0] + B * u
    hi = A @ b[:, 1] + B * u
    return np.stack([lo, hi], axis=1)


def install(tester, dt=None):
    """Drop the estimator box from tight_bound. Returns a live stats dict."""
    stats = {'steps': 0, 'truth_out_tight': 0, 'mpc_interval_steps': 0,
             'fill_steps': 0, 'fill_failed': 0, 'no_parent_bound': 0,
             'width_max': 0.0}
    if dt is None:
        dt = float(tester.analyzer.cl_system.dynamics.dt)
    dt = float(dt)

    # ReachabilityTester.__init__ files the INITIAL SET as an EMPIRICAL
    # calculation (REAL_integrated_sim.py, 'Initial set'). The patch below
    # restores tight_bound after every EMPIRICAL calc, so installing it before
    # construction would erase X0 and leave every later box unanchored. Both
    # callers install afterwards; assert it rather than rely on the ordering.
    assert 0 in tester.horizons and tester.horizons[0].get_tight_bound() is not None, \
        'install() must run AFTER ReachabilityTester.__init__ has filed X0'

    horizon_cls = None
    for h in tester.horizons.values():
        horizon_cls = type(h)
        break
    if horizon_cls is None:
        from REAL_integrated_sim import ReachableSetHorizon as horizon_cls

    if not hasattr(horizon_cls, '_nokf_orig_add'):
        horizon_cls._nokf_orig_add = horizon_cls.add_calculation
    orig_add = horizon_cls._nokf_orig_add
    orig_emp, orig_mpc = tester.real_state_empirical, tester.real_state_mpc

    def _set_bound(h, b):
        h.tight_bound = None if b is None else np.asarray(b, dtype=float).copy()
        if h.tight_bound is not None:
            h.reachable_set.set_range(torch.tensor(h.tight_bound,
                                                   dtype=torch.float32,
                                                   device=h.device))

    def add_calculation(self, *a, **kw):
        calc_type = kw.get('calc_type', a[1] if len(a) > 1 else None)
        keep = None if self.tight_bound is None else self.tight_bound.copy()
        out = orig_add(self, *a, **kw)
        if calc_type == CalculationType.EMPIRICAL:
            _set_bound(self, keep)
        return out
    horizon_cls.add_calculation = add_calculation

    def _check(t):
        h = tester.horizons.get(t)
        if h is None:
            return
        b = h.get_tight_bound()
        if b is None:
            return
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

    def _fill(t):
        h = tester.horizons.get(t)
        if h is not None and h.get_tight_bound() is None:
            stats['fill_steps'] += 1
            tester.concrete(t - 1, t)
            if tester.horizons[t].get_tight_bound() is None:
                stats['fill_failed'] += 1

    def real_state_mpc(start, control):
        b_prev = (tester.horizons[start].get_tight_bound()
                  if start in tester.horizons else None)
        res = orig_mpc(start, control)
        h = tester.horizons.get(start + 1)
        if h is not None:
            if b_prev is None:
                stats['no_parent_bound'] += 1
                _set_bound(h, None)
            else:
                nb = _interval_step(b_prev, control, dt)
                _set_bound(h, nb)
                stats['mpc_interval_steps'] += 1
                stats['width_max'] = max(stats['width_max'],
                                         float(np.max(nb[:, 1] - nb[:, 0])))
        _check(start + 1)
        return res

    def real_state_empirical(start, end):
        # Fill BEFORE the step as well: orig_emp prints the tight volume of
        # `end` (REAL_integrated_sim.py:626) and crashes on a missing bound,
        # which is every step of the unfiltered arm now the EKF box is gone.
        for t in range(start + 1, end + 1):
            h = tester.horizons.get(t)
            if h is None or h.get_tight_bound() is None:
                stats['fill_steps'] += 1
                tester.concrete(t - 1, t)
                if tester.horizons.get(t) is None or \
                        tester.horizons[t].get_tight_bound() is None:
                    stats['fill_failed'] += 1
        res = orig_emp(start, end)
        _fill(end)
        _check(end)
        return res

    tester.real_state_empirical = real_state_empirical
    tester.real_state_mpc = real_state_mpc
    tester._nokf_stats = stats
    return stats


def patch_module(A):
    """Wrap alg14_di's test() and run_nominal() so both arms run without the
    estimator box. The wrappers hook ReachabilityTester.__init__ for the
    duration of the call, because both build their tester internally."""
    import REAL_integrated_sim as R

    if getattr(A, '_nokf_patched', False):
        return A
    orig_test, orig_nom = A.test, A.run_nominal

    def _wrap(fn):
        def inner(*a, **kw):
            orig_init = R.ReachabilityTester.__init__
            made = []

            # install() patches add_calculation on the horizon CLASS. A tester
            # built later in the same process (the second arm, the next seed)
            # would file its X0 through that patch and lose it, so construct
            # every tester against the original method.
            hcls = R.ReachableSetHorizon

            def __init__(self, *ia, **ikw):
                if hasattr(hcls, '_nokf_orig_add'):
                    hcls.add_calculation = hcls._nokf_orig_add
                orig_init(self, *ia, **ikw)
                made.append(self)
                install(self)
            R.ReachabilityTester.__init__ = __init__
            try:
                out = fn(*a, **kw)
            finally:
                R.ReachabilityTester.__init__ = orig_init
                if hasattr(hcls, '_nokf_orig_add'):
                    hcls.add_calculation = hcls._nokf_orig_add
            if made and isinstance(out, dict):
                out['nokf'] = dict(made[0]._nokf_stats)
            return out
        return inner

    A.test, A.run_nominal = _wrap(orig_test), _wrap(orig_nom)
    A._nokf_patched = True
    return A
