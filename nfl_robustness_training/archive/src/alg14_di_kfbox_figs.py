"""
alg14_di figures drawing the raw Kalman 3-sigma box at each step.

alg14_di_figs draws h.get_tight_bound(), which is the Kalman box INTERSECTED
with every reachable set filed at that step. This script runs exactly the same
simulations (same seeds, same code, nothing patched in the loop) and only
changes what is drawn: the EMPIRICAL calculation's own 'bounds' -- the box the
estimator produced -- with no intersection. The intersected boxes are kept in
'tight_history' for comparison. Output goes to results_di_kfbox/.

    python nfl_robustness_training/src/alg14_di_kfbox_figs.py [n_seeds]
    python nfl_robustness_training/src/alg14_di_kfbox_figs.py replot
    python nfl_robustness_training/src/alg14_di_kfbox_figs.py figrun <seed>
        (one seed, written to results_di_kfbox/seed_<seed>/)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alg14_di_figs as F
import REAL_integrated_sim as R
from REAL_integrated_sim import CalculationType

OUT = os.path.join('nfl_robustness_training', 'results_di_kfbox')
os.makedirs(OUT, exist_ok=True)
F.OUT = OUT
F.LOG = os.path.join(OUT, 'run.log')


def _kf_boxes(tester, state_history):
    """The estimator box filed at each executed step, in time order.

    Picks the EMPIRICAL calculation that carries real_state (the one the
    simulator filed for the executed step); t=0 is the initial set. Checked
    against state_history so the boxes line up with the drawn trajectory.
    """
    boxes, states = [], []
    for t in sorted(tester.horizons.keys()):
        kf = None
        for calc in tester.horizons[t].calculations.values():
            if calc['calc_type'] == CalculationType.EMPIRICAL and 'real_state' in calc:
                kf = calc
        if kf is not None:
            boxes.append(np.asarray(kf['bounds'], dtype=float).copy())
            states.append(np.asarray(kf['real_state'], dtype=float).flatten())
    assert len(states) == len(state_history) and all(
        np.allclose(a, b) for a, b in zip(states, state_history)), \
        'Kalman boxes do not line up with state_history'
    return boxes


def _wrap(fn):
    def inner(*a, **kw):
        orig_init = R.ReachabilityTester.__init__
        made = []

        def __init__(self, *ia, **ikw):
            orig_init(self, *ia, **ikw)
            made.append(self)
        R.ReachabilityTester.__init__ = __init__
        try:
            out = fn(*a, **kw)
        finally:
            R.ReachabilityTester.__init__ = orig_init
        # test() also builds throwaway testers for mpc_cost calibration
        # (alg14_di:438); use the one whose trajectory is the one returned.
        boxes = None
        for tester in reversed(made):
            try:
                boxes = _kf_boxes(tester, out['state_history'])
                break
            except AssertionError:
                continue
        assert boxes is not None, 'no tester matches the returned trajectory'
        out['tight_history'] = out['bound_history']
        out['bound_history'] = boxes
        return out
    return inner


F.A.test = _wrap(F.A.test)
F.A.run_nominal = _wrap(F.A.run_nominal)

# Relabel the one place the legend text enters the figure.
_orig_boxes = F._boxes


def _boxes(a, hist, color, alpha, label=None, lw=0.8, ls="-"):
    if isinstance(label, str):
        label = label.replace('uncertainty', 'Kalman 3σ box')
    return _orig_boxes(a, hist, color, alpha, label, lw, ls)


F._boxes = _boxes


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'figrun':
        F.OUT = os.path.join(OUT, f'seed_{int(sys.argv[2])}')
        os.makedirs(F.OUT, exist_ok=True)
        F.LOG = os.path.join(F.OUT, 'run.log')
        F.figrun(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'replot':
        F.replot()
    else:
        F.main()
