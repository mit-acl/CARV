"""
Run alg12 N times with random initial states and randomly generated obstacles.
Obstacles:
  - Number per trial: 1-4 (uniform)
  - Center x: uniform in [-7, -1]
  - Center y: uniform in [-1,  2]
  - Radius:   0.5

"""

import numpy as np
import multiprocessing as mp
import os, sys

N_TRIALS  = 200
N_WORKERS = 8
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
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _run_trial(args):
    i, seed, obstacles = args
    from REAL_integrated_sim import setup_analyzer
    from alg12_mpc_every_timestep import test
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    traj, _, u_diffs, mpc_calls, mpc_over = test(seed=seed, analyzer=analyzer, obstacles=obstacles)
    had_collision = point_collision(traj, obstacles)
    return i, seed, obstacles, len(traj), mpc_calls, mpc_over, had_collision


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
          f"(1-5 random obstacles, r={OBS_RADIUS}, "
          f"x∈{X_RANGE}, y∈{Y_RANGE})...")


    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results_raw = []
        for result in pool.imap_unordered(_run_trial, args):
            results_raw.append(result)
            i, seed, obstacles, n_states, mpc_calls, mpc_over, had_collision = result
            status = "UNSAFE" if had_collision else "safe"
            obs_str = "  ".join(f"({o[0]:.2f},{o[1]:.2f})" for o in obstacles)
            print(f"  Trial {i+1:>3}  seed={seed}  [{status}]  "
                  f"states={n_states}  n_obs={len(obstacles)}  centers=[{obs_str}]",
                  flush=True)

    results_raw.sort(key=lambda r: r[0])

    safety_record = []
    unsafe_info = []
    for i, seed, obstacles, n_states, mpc_calls, mpc_over, had_collision in results_raw:
        safety_record.append(not had_collision)
        if had_collision:
            obs_str = "  ".join(f"({o[0]:.3f},{o[1]:.3f},r={o[2]})" for o in obstacles)
            unsafe_info.append((i + 1, seed, obs_str))

    n_safe = sum(safety_record)
    print(f"\n{'='*60}")
    print(f"Safety record: {n_safe}/{N_TRIALS} safe  ({100*n_safe/N_TRIALS:.1f}%)")
    if unsafe_info:
        print("Unsafe trials:")
        for trial_num, seed, obs_str in unsafe_info:
            print(f"  Trial {trial_num:>3}  seed={seed}  obstacles: {obs_str}")
    else:
        print("No unsafe trials.")
    print(f"{'='*60}")
