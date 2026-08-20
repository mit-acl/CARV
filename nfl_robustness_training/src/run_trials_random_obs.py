"""
Run alg14 N times with random initial states and randomly generated obstacles.
Obstacles:
  - Number per trial: 1-4 (uniform)
  - Center x: uniform in [-7, -1]
  - Center y: uniform in [-1,  2]
  - Radius:   0.5

"""

import numpy as np
import multiprocessing as mp
import os, sys

N_TRIALS  = 500
N_WORKERS = 1

# Override from the environment so a cluster run needs no source edit:
#   TTTCARV_WORKERS=64 python run_trials_random_obs.py
# Clamped to the CPUs actually granted to this process. Under SLURM the cgroup
# grants a subset of the node, so sizing off the node's total overcommits, and
# the wall-clock timestep budget then silently skips verification it can afford.
_n_cpus = len(os.sched_getaffinity(0))
N_WORKERS = min(int(os.environ.get('TTTCARV_WORKERS', N_WORKERS)), _n_cpus)
OBS_RADIUS = 0.5
X_RANGE = (-7.0, -1.0)
Y_RANGE = (-1.0,  2.0)


def _gen_obstacles(rng):
    n = int(rng.integers(1, 6))  # 1–5
    obs = []
    for _ in range(n):
        cx = float(rng.uniform(*X_RANGE))
        cy = float(rng.uniform(*Y_RANGE))
        obs.append(np.array([cx, cy, OBS_RADIUS]))
    return obs


def point_collision(traj, obstacles):
    for s in traj:
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx)**2 + (s_flat[1] - cy)**2 <= r**2:
                return True
    return False


def _worker_init():
    # One BLAS/torch thread per worker. Without this each worker spawns
    # torch.get_num_threads() (10 here) intra-op threads, and N workers
    # oversubscribe the machine ~N-fold. That inflates the ops the 0.20s
    # timestep budget is measured against — concrete/step went 0.0142s -> 0.172s
    # at 16 workers — so the algorithm silently skips verification it thinks it
    # cannot afford. Must run before the worker imports torch.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    import torch
    torch.set_num_threads(1)

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _run_trial(args):
    i, seed, obstacles = args
    from REAL_integrated_sim import setup_analyzer
    from alg14_split_terminal import test
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    (traj, _, u_diffs, mpc_calls, no_diverge, queue_empty,
     pt_found, pt_used) = test(seed=seed, analyzer=analyzer, obstacles=obstacles)
    had_collision = point_collision(traj, obstacles)
    return (i, seed, obstacles, len(traj), mpc_calls, no_diverge, queue_empty,
            pt_found, pt_used, had_collision)


if __name__ == "__main__":
    rng = np.random.default_rng()

    args = []
    for i in range(N_TRIALS):
        seed = int(rng.integers(0, 2**31))
        obstacles = _gen_obstacles(rng)
        args.append((i, seed, obstacles))

    # set up workers to run simulation in parallel
    n_workers = min(N_WORKERS, N_TRIALS)
    print(f"Running {N_TRIALS} trials across {n_workers} workers  "
          f"({_n_cpus} CPUs available)  "
          f"(1-5 random obstacles, r={OBS_RADIUS}, "
          f"x∈{X_RANGE}, y∈{Y_RANGE})...")


    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results_raw = []
        for result in pool.imap_unordered(_run_trial, args):
            results_raw.append(result)
            (i, seed, obstacles, n_states, mpc_calls, no_diverge, queue_empty,
             pt_found, pt_used, had_collision) = result
            status = "UNSAFE" if had_collision else "safe"
            obs_str = "  ".join(f"({o[0]:.2f},{o[1]:.2f})" for o in obstacles)
            print(f"  Trial {i+1:>3}  seed={seed}  [{status}]  "
                  f"states={n_states}  mpc={mpc_calls}  "
                  f"pt_found={pt_found}  pt_used={pt_used}  "
                  f"n_obs={len(obstacles)}  centers=[{obs_str}]",
                  flush=True)

    results_raw.sort(key=lambda r: r[0])

    safety_record = []
    unsafe_info = []
    total_no_diverge  = 0
    total_queue_empty = 0
    total_pt_found    = 0
    total_pt_used     = 0
    no_diverge_trials  = []   # (trial#, seed, count)
    queue_empty_trials = []
    pt_trials          = []   # trials where passthrough fired (found or used)
    for (i, seed, obstacles, n_states, mpc_calls, no_diverge, queue_empty,
         pt_found, pt_used, had_collision) in results_raw:
        safety_record.append(not had_collision)
        total_no_diverge  += no_diverge
        total_queue_empty += queue_empty
        total_pt_found    += pt_found
        total_pt_used     += pt_used
        if pt_found or pt_used:
            obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
            pt_trials.append((i + 1, seed, pt_found, pt_used, obs_str))
        if no_diverge:
            obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
            no_diverge_trials.append((i + 1, seed, no_diverge, obs_str))
        if queue_empty:
            obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
            queue_empty_trials.append((i + 1, seed, queue_empty, obs_str))
        if had_collision:
            obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
            unsafe_info.append((i + 1, seed, obs_str))

    n_safe = sum(safety_record)
    print(f"\n{'='*60}")
    print(f"Safety record: {n_safe}/{N_TRIALS} safe  ({100*n_safe/N_TRIALS:.1f}%)")
    print(f"PSF no-diverge fallbacks:  {total_no_diverge} timesteps"
          f" across {len(no_diverge_trials)} trials")
    for trial_num, seed, count, obs_str in no_diverge_trials:
        print(f"    Trial {trial_num:>3}  seed={seed}  ×{count}  obs=[{obs_str}]")
    print(f"PSF queue-empty fallbacks: {total_queue_empty} timesteps"
          f" across {len(queue_empty_trials)} trials")
    for trial_num, seed, count, obs_str in queue_empty_trials:
        print(f"    Trial {trial_num:>3}  seed={seed}  ×{count}  obs=[{obs_str}]")
    print(f"Passthrough found: {total_pt_found}   used: {total_pt_used}"
          f"   across {len(pt_trials)} trials")
    for trial_num, seed, pf, pu, obs_str in pt_trials:
        print(f"    Trial {trial_num:>3}  seed={seed}  found={pf}  used={pu}"
              f"  obs=[{obs_str}]")
    if unsafe_info:
        print("Unsafe trials:")
        for trial_num, seed, obs_str in unsafe_info:
            print(f"  Trial {trial_num:>3}  seed={seed}  obstacles: {obs_str}")
    else:
        print("No unsafe trials.")
    print(f"{'='*60}")
