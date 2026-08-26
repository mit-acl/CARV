"""Re-run the 20 alg15 failures from the 1000-trial run at a chosen worker count.

Point of the script: the 1000-trial run found these under 64-way contention, but
one of them (seed 52642855) came out safe when run alone. If the failures are a
wall-clock/load effect rather than a geometry effect, they should disappear as
the worker count drops. Same seeds, same obstacles, only the contention changes.
"""
import numpy as np, multiprocessing as mp, os, sys

# run_trials_random_obs.py gets src/ onto sys.path for free by living in it. This
# script sits at the repo root (cwd must stay the root -- utils/nn.py reads
# os.getcwd()), so src/ has to be added explicitly, in the workers too.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'nfl_robustness_training', 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CASES_FILE = os.environ.get('TTTCARV_CASES', '/home/ryosei_t/alg15_repro_cases.txt')
N_WORKERS  = min(int(os.environ.get('TTTCARV_WORKERS', 4)),
                 len(os.sched_getaffinity(0)))
ALG_MODULE = os.environ.get('TTTCARV_ALG', 'alg15_split_terminal')
REPEATS    = int(os.environ.get('TTTCARV_REPEATS', 1))


def load_cases():
    cases = []
    for line in open(CASES_FILE):
        line = line.strip()
        if not line:
            continue
        seed, obs_s = line.split('|')
        obs = [np.array([float(v) for v in grp.split(',')])
               for grp in obs_s.split(';')]
        cases.append((int(seed), obs))
    return cases


def point_collision(traj, obstacles):
    for s in traj:
        f = np.asarray(s).flatten()
        for cx, cy, r in obstacles:
            if (f[0] - cx) ** 2 + (f[1] - cy) ** 2 <= r ** 2:
                return True
    return False


def _worker_init():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    import torch
    torch.set_num_threads(1)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _run(args):
    i, seed, obstacles = args
    from REAL_integrated_sim import setup_analyzer
    from importlib import import_module
    test = import_module(ALG_MODULE).test
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    (traj, _, _, mpc_calls, no_diverge, queue_empty,
     pt_found, pt_used) = test(seed=seed, analyzer=analyzer, obstacles=obstacles)
    return (i, seed, mpc_calls, no_diverge, pt_found, pt_used,
            point_collision(traj, obstacles))


if __name__ == "__main__":
    cases = load_cases()
    args = [(i, s, o) for r in range(REPEATS)
            for i, (s, o) in enumerate(cases)]
    print(f"Re-running {len(cases)} known-unsafe cases x{REPEATS} "
          f"at {N_WORKERS} workers ({len(os.sched_getaffinity(0))} CPUs)", flush=True)
    ctx = mp.get_context('spawn')
    out = []
    with ctx.Pool(processes=min(N_WORKERS, len(args)),
                  initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_run, args):
            out.append(r)
            i, seed, mpc, nd, pf, pu, unsafe = r
            print(f"  seed={seed:<11} {'UNSAFE' if unsafe else 'safe  '}  "
                  f"mpc={mpc:<3} no_diverge={nd:<3} pt_found={pf} pt_used={pu}",
                  flush=True)
    n_unsafe = sum(r[-1] for r in out)
    print(f"\n{'='*60}")
    print(f"{N_WORKERS} workers: {n_unsafe}/{len(out)} still UNSAFE"
          f"   ({len(out)-n_unsafe} now safe)")
    print(f"{'='*60}")
