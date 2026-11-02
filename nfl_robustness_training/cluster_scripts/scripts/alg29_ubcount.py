#!/usr/bin/env python3
"""|omega| > 1 overshoot counts per arm (TTTCARV_SINGLE_UBOUND=count run).

usage: alg29_ubcount.py RUN_DIR [REF_RUN_DIR]
REF_RUN_DIR: an earlier run of the same arms; checks that collisions match
(fixed-delay arms are deterministic, so they must be identical).
"""
import glob
import json
import os
import sys


def rows(p):
    return [r for r in json.load(open(p))['trials'] if not r.get('error')]


def coll(r):
    return bool(r.get('collision', r.get('violation')))


run = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else None
print('arm\tn\tcoll\tref_coll\tsingle_solves\tsingle_over\tsingle_peak\t'
      'acc_over(plan|resolve+extend)\tacc_trials\tflown_plan\tflown_conv\tflown_fb\tacc_peak')
for p in sorted(glob.glob(os.path.join(run, 'fp_*.json'))):
    arm = os.path.basename(p)[3:-5]
    rs = rows(p)
    g = lambda k: sum((r.get(k) or 0) for r in rs)
    mx = lambda k: max(((r.get(k) or 0) for r in rs), default=0)
    rc = '-'
    if ref and os.path.exists(os.path.join(ref, os.path.basename(p))):
        rc = sum(coll(r) for r in rows(os.path.join(ref, os.path.basename(p))))
    if rs and 'gp_plan_ub_over' in rs[0]:
        vals = (g('gp_single_ub_solves'), g('gp_single_ub_over'), mx('gp_single_ub_peak'),
                g('gp_plan_ub_over'), sum((r.get('gp_plan_ub_over') or 0) > 0 for r in rs),
                g('gp_land_ub_over'), g('gp_land_ub_over_conv'), g('gp_fb_ub_over'),
                mx('gp_plan_ub_peak'))
    else:
        acc = lambda r: (r.get('ub_acc_resolve') or 0) + (r.get('ub_acc_extend') or 0)
        vals = (g('ub_single_solves'), g('ub_single_over'), mx('ub_single_peak'),
                sum(acc(r) for r in rs), sum(acc(r) > 0 for r in rs), '-', '-', '-',
                mx('ub_acc_peak'))
    fmt = lambda v: f'{v:.2f}' if isinstance(v, float) else str(v)
    print('\t'.join([arm, str(len(rs)), str(sum(coll(r) for r in rs)), str(rc)]
                    + [fmt(v) for v in vals]))
