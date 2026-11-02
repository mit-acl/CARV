"""alg45: corridor longer than the MPC horizon, too narrow for a loiter inside.

Prediction (stated before the run): ours as is and R2 both refuse to enter and
override for most of the episode; ours+NOMINAL_PASS flies the NN through
(pt_used > 0) and overrides much less, with 0 collisions on every arm.
"""
import json, os, sys
import numpy as np

D = os.path.expanduser(f'~/runs/alg45_corridor/{sys.argv[1] if len(sys.argv) > 1 else "r1"}')
EXIT_X = -4.917   # nominal NN path leaves the corridor here (k = 24)


def load(n):
    p = f'{D}/fp_{n}.json'
    if not os.path.exists(p):
        return None
    return [t for t in json.load(open(p))['trials']]


for n in ('o_cor', 'o_cor_np', 'b_r2_cor'):
    T = load(n)
    if T is None:
        print(f'{n:10s} missing'); continue
    err = [t for t in T if 'error' in t]
    T = [t for t in T if 'error' not in t]
    S = lambda k: sum((t.get(k) or 0) for t in T)
    if 'gp_u_diff_len' in T[0]:
        L, k = S('gp_u_diff_len'), sum(t['gp_u_diff_gt_pred']['0.05'] for t in T)
    else:
        L, k = S('u_diff_len'), sum(t['u_diff_gt']['0.05'] for t in T)
    fin = np.array([t['final'][0] if 'final' in t else np.nan for t in T])
    w = [max(np.atleast_1d(t['wall_per_timestep_s'])) for t in T if t.get('wall_per_timestep_s') is not None]
    print(f'{n:10s} n={len(T)} err={len(err)} coll={S("collision")}  interv {100 * k / L:5.2f}%'
          f'  pt_used {S("pt_used")}  pt_found {S("pt_found")}  nominal_pass_tries {S("nominal_pass_tries")}'
          f'  wall/step mean max {max(w) if w else float("nan"):.3f}')
    print(f'           audits: snd_audit_miss {S("snd_audit_miss")}  setmem_truth_out_post {S("setmem_truth_out_post")}'
          f'  look_truth_out {S("est_look_truth_out")}  fb_uncert {S("mpc_fallback_uncertified")}')
