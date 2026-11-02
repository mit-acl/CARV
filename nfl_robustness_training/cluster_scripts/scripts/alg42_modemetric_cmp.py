"""alg42: how much of R2's intervention excess over ours is a measurement
artifact? Prints, per arm, the threshold metric and the threshold-free
mode-based metric, then paired bootstrap CIs of (R2 - ours) on each.

usage: python alg42_modemetric_cmp.py FP_OURS FP_R2 [FP_R2_EXACT ...]
Each FP is a run_trials fingerprint json.
"""
import json
import sys

import numpy as np

RNG = np.random.default_rng(20260824)


def load(path):
    return json.load(open(path))['trials']


def per_trial(trials):
    """(steps, raw>0.05, clip>0.05, engaged) arrays, one entry per scene."""
    out = {k: [] for k in ('n', 'raw', 'clip', 'eng', 'over', 'pinf', 'pint')}
    for t in trials:
        gp = 'gp_u_diff_len' in t
        p = 'gp_' if gp else ''
        n = t[p + 'u_diff_len']
        out['n'].append(n)
        out['raw'].append(t[p + 'u_diff_gt']['0.05'])
        out['clip'].append(t[p + 'u_diff_clip_gt']['0.05'])
        out['over'].append(t[p + 'nn_over'])
        out['eng'].append(t['gp_eng_steps'] if gp else t['eng_steps'])
        # Solve-level cross-check for the baselines: a refused pin IS an
        # override. Excludes the t=0 step, where no plan has landed yet.
        out['pint'].append(t.get('gp_pin_tries', 0) if gp else 0)
        out['pinf'].append((t['gp_pin_tries'] - t['gp_pin_ok']) if gp else 0)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def rate(d, key):
    return 100.0 * d[key].sum() / d['n'].sum()


def paired(a, b, key, n_boot=10000):
    """b - a, as a percentage-of-steps difference, bootstrapped over scenes."""
    m = min(len(a['n']), len(b['n']))
    ra = a[key][:m] / a['n'][:m] * 100.0
    rb = b[key][:m] / b['n'][:m] * 100.0
    d = rb - ra
    idx = RNG.integers(0, m, size=(n_boot, m))
    boot = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), float((d > 0).mean())


def main(paths):
    arms = {}
    for p in paths:
        name = p.split('/')[-1].replace('fp_', '').replace('.json', '')
        arms[name] = per_trial(load(p))
    w = max(len(k) for k in arms) + 1
    print(f"{'arm':{w}s} {'scenes':>6s} {'steps':>7s} {'>0.05':>7s} "
          f"{'clip>0.05':>9s} {'engaged':>8s} {'pinrefused':>10s} {'|NN|>1':>7s}")
    for k, d in arms.items():
        print(f"{k:{w}s} {len(d['n']):6d} {int(d['n'].sum()):7d} "
              f"{rate(d, 'raw'):7.2f} {rate(d, 'clip'):9.2f} "
              f"{rate(d, 'eng'):8.2f} "
              f"{(100*d['pinf'].sum()/d['pint'].sum() if d['pint'].sum() else float('nan')):10.2f} "
              f"{rate(d, 'over'):7.2f}")
    ours = next((k for k in arms if 'ours' in k), None)
    if ours is None:
        return
    print(f"\npaired vs {ours} (positive = the baseline is higher), "
          f"bootstrap 95% CI over scenes")
    for k, d in arms.items():
        if k == ours:
            continue
        for key, label in (('raw', '>0.05'), ('clip', 'clip>0.05'), ('eng', 'engaged')):
            m, lo, hi, frac = paired(arms[ours], d, key)
            print(f"  {k:{w}s} {label:10s} {m:+6.2f} pp [{lo:+5.2f}, {hi:+5.2f}]"
                  f"  higher in {100*frac:4.0f}% of scenes")


if __name__ == '__main__':
    main(sys.argv[1:])
