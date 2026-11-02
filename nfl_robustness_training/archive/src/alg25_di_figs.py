"""
alg25 (DI) figures — alg14_di_figs.py with the estimator box removed.

Same seeds, same trajectories, same drawing code. The only difference is what
the boxes ARE: alg14_di_figs draws Kalman-conditioned state estimates, this
draws forward reachable sets (see alg25_di_nokf). Output goes to
results_di_nokf/ so the published alg14 figures are untouched.

    python nfl_robustness_training/src/alg25_di_figs.py [n_seeds]
    python nfl_robustness_training/src/alg25_di_figs.py replot
    python nfl_robustness_training/src/alg25_di_figs.py figrun <seed>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alg14_di_figs as F
import alg25_di_nokf as N

OUT = os.path.join('nfl_robustness_training', 'results_di_nokf')
os.makedirs(OUT, exist_ok=True)
F.OUT = OUT
F.LOG = os.path.join(OUT, 'run.log')

# Both arms lose the estimator box.
N.patch_module(F.A)

# The boxes are no longer conditioned on measurements, so they are no longer
# "uncertainty". Relabel at the one place the legend text enters the figure,
# rather than editing alg14_di_figs.
_orig_boxes = F._boxes
def _boxes(a, hist, color, alpha, label, lw, ls):
    if isinstance(label, str):
        label = label.replace('uncertainty', 'reachable set')
    return _orig_boxes(a, hist, color, alpha, label, lw, ls)


F._boxes = _boxes


# The timeseries panel names the same quantity in its title (alg14_di_figs:315).
_orig_ts = F.timeseries


def timeseries(psf, nom, path):
    from matplotlib.axes import Axes
    orig_set_title = Axes.set_title

    def set_title(self, label, *a, **kw):
        if isinstance(label, str):
            label = label.replace('Velocity uncertainty band',
                                  'Velocity reachable-set band')
        return orig_set_title(self, label, *a, **kw)
    Axes.set_title = set_title
    try:
        return _orig_ts(psf, nom, path)
    finally:
        Axes.set_title = orig_set_title


F.timeseries = timeseries


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'figrun':
        F.figrun(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'replot':
        F.replot()
    else:
        F.main()
