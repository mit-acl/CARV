#!/usr/bin/env python3
"""Summarize alg29 generic-PSF arms (and alg28 / alg22_di arms) from fp_*.json.

usage: alg29_summarize.py RUN_DIR [RUN_DIR ...] [--csv out.csv]

Collision columns
  node  a tick state inside an obstacle / past a half-plane (what the trial
        runners report; the discrete-time safety claim)
  cont  anywhere on the step (unicycle: straight segment; DI: exact min p).
        Only alg29 records it; alg28 rows show '-'.
"""
import glob
import json
import math
import os
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(path):
    fp = json.load(open(path))
    rows = [r for r in fp['trials'] if not r.get('error')]
    return fp, rows, len(fp['trials']) - len(rows)


def summarize(path):
    fp, rows, n_err = load(path)
    arm = os.path.basename(path)[3:-5]
    n = len(rows)
    node = sum(bool(r.get('collision', r.get('violation'))) for r in rows)
    has_gp = rows and 'gp_collided_seg' in rows[0]
    cont = sum(bool(r.get('gp_collided_seg')) for r in rows) if has_gp else None
    lo, hi = wilson(node, n)
    out = dict(arm=arm, n=n, errors=n_err, node=node, node_pct=100 * node / max(n, 1),
               ci=f'[{100*lo:.1f},{100*hi:.1f}]', cont='-' if cont is None else cont)
    if has_gp:
        g = lambda k: sum(r.get(k, 0) for r in rows)
        coll = [r for r in rows if r.get('gp_collided_seg')]
        offs = [r['gp_offplan_at_collision'] for r in coll if r.get('gp_offplan_at_collision', -1) >= 0]
        out.update(fallback=g('gp_fallback_steps'), loiter=g('gp_loiter_steps'),
                   fail=g('gp_solve_fail'), solves=g('gp_solves'),
                   act=g('gp_activations'), truth_out=g('gp_truth_out'),
                   dl_miss=g('gp_deadline_miss'),
                   cert_inv=g('gp_cert_invalid'), cert_chk=g('gp_cert_checks'),
                   cert_inv_trials=sum(r.get('gp_cert_invalid', 0) > 0 for r in rows),
                   fb_inv=g('gp_fb_invalid'), fb_chk=g('gp_fb_checks'),
                   tube_out=g('gp_tube_out'), tube_chk=g('gp_tube_checks'),
                   tube_out_trials=sum((r.get('gp_tube_out') or 0) > 0 for r in rows),
                   ub_flown=g('gp_land_ub_over') + g('gp_land_ub_over_conv') + g('gp_fb_ub_over'),
                   filt=g('gp_filtered_steps'),
                   off_coll_med=(sorted(offs)[len(offs) // 2] if offs else float('nan')),
                   wall_med=sorted(r['gp_solve_wall_med'] for r in rows)[n // 2] if n else 0)
    else:
        g = lambda k: sum(r.get(k, 0) or 0 for r in rows)
        out.update(act=g('mpc_calls'),
                   audits=sum(g(k) for k in ('est_look_truth_out', 'snd_audit_miss',
                                            'snd_node_truth_out', 'setmem_truth_out_tight',
                                            'setmem_truth_out_meas', 'setmem_truth_out_post')),
                   wall_ts_med=sorted(r.get('wall_per_timestep_s', 0) for r in rows)[n // 2] if n else 0,
                   delay=rows[0].get('setmem_meas_delay', '-') if rows else '-')
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
    if csv in args:
        args.remove(csv)
    res = []
    for d in args:
        for p in sorted(glob.glob(os.path.join(d, 'fp_*.json'))):
            if 'smoke' in os.path.basename(p):
                continue
            res.append(summarize(p))
    keys = []
    for r in res:
        keys += [k for k in r if k not in keys]
    print('\t'.join(keys))
    for r in res:
        print('\t'.join(f'{r[k]:.3f}' if isinstance(r.get(k), float) else str(r.get(k, ''))
                        for k in keys))
    if csv:
        with open(csv, 'w') as fh:
            fh.write(','.join(keys) + '\n')
            for r in res:
                fh.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    main()
