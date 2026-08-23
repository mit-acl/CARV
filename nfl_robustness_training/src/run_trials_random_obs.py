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

N_TRIALS  = 100
N_WORKERS = 12

# Override from the environment so a cluster run needs no source edit:
#   TTTCARV_WORKERS=64 python run_trials_random_obs.py
# Clamped to the CPUs actually granted to this process. Under SLURM the cgroup
# grants a subset of the node, so sizing off the node's total overcommits, and
# the wall-clock timestep budget then silently skips verification it can afford.
_n_cpus = len(os.sched_getaffinity(0))
N_WORKERS = min(int(os.environ.get('TTTCARV_WORKERS', N_WORKERS)), _n_cpus)
# Trial count is the other knob a cluster run needs without a source edit.
N_TRIALS  = int(os.environ.get('TTTCARV_TRIALS', N_TRIALS))
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


# ── Regime fingerprint ──────────────────────────────────────────────────
# The filter's decisions are a function of how much work fits inside its fixed
# 0.20 s wall-clock timestep budget, so identical code on identical seeds takes
# different trajectories at different compute. Seed 3168 collides 15/15 run
# serially and 4/16 run 16-way concurrent on the same box. Outcomes are
# therefore NOT comparable across machines or worker counts unless the achieved
# compute is reported with them.
#
# _probe times a fixed unit of work inside the worker, under exactly the
# contention that trial saw. It is the per-trial regime coordinate: bucket
# collisions by probe_s, not by hostname. Bigger probe_s = more starved.

def _probe(reps=3):
    """Median seconds for a fixed unit of work under current contention."""
    import numpy as _np, time as _t
    a = _np.random.default_rng(0).standard_normal((256, 256))
    ts = []
    for _ in range(reps):
        t0 = _t.perf_counter()
        a @ a
        ts.append(_t.perf_counter() - t0)
    return float(_np.median(ts))


def _machine():
    import platform, socket
    model = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"host": socket.gethostname(), "cpu": model,
            "cpus_granted": _n_cpus, "workers": N_WORKERS,
            "python": platform.python_version(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus": os.environ.get("SLURM_CPUS_ON_NODE")}


def _run_trial(args):
    i, seed, obstacles = args
    from REAL_integrated_sim import setup_analyzer
    from alg14_split_terminal import test
    import time as _t
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    probe_before = _probe()
    _t0 = _t.perf_counter()
    (traj, _, u_diffs, mpc_calls, no_diverge, queue_empty,
     pt_found, pt_used) = test(seed=seed, analyzer=analyzer, obstacles=obstacles)
    wall = _t.perf_counter() - _t0
    probe_after = _probe()
    had_collision = point_collision(traj, obstacles)
    # Probe both sides: a trial that starts unloaded and ends contended (or the
    # reverse, as the pool drains) has no single regime, and averaging the two
    # would hide that. Reported separately so such trials can be excluded.
    fp = {"probe_before_s": probe_before, "probe_after_s": probe_after,
          "wall_s": wall,
          "wall_per_timestep_s": wall / max(len(traj), 1),
          "pid": os.getpid()}
    return (i, seed, obstacles, len(traj), mpc_calls, no_diverge, queue_empty,
            pt_found, pt_used, had_collision, fp)


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
             pt_found, pt_used, had_collision, fp) = result
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
         pt_found, pt_used, had_collision, fp) in results_raw:
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

    # ── Regime fingerprint report ───────────────────────────────────────
    import json as _json
    mach = _machine()
    rows = [{"seed": r[1], "timesteps": r[3], "collision": bool(r[9]), **r[10]}
            for r in results_raw]
    with open("trials_fingerprint.json", "w") as f:
        _json.dump({"machine": mach, "n_trials": N_TRIALS, "trials": rows}, f, indent=1)

    print(f"\nRegime fingerprint  ->  trials_fingerprint.json")
    print(f"  host={mach['host']}  cpu={mach['cpu']}")
    print(f"  workers={mach['workers']}  cpus_granted={mach['cpus_granted']}"
          f"  slurm_job={mach['slurm_job']}")
    probes = sorted(r["probe_after_s"] for r in rows)
    if probes:
        def _q(p):
            return probes[min(len(probes) - 1, int(p * len(probes)))]
        print(f"  probe_s (fixed unit of work under load):"
              f"  min={probes[0]:.5f}  median={_q(.5):.5f}  p90={_q(.9):.5f}"
              f"  max={probes[-1]:.5f}")
        print(f"  contention spread (max/min) = {probes[-1]/max(probes[0],1e-9):.1f}x"
              f"   -- 1.0x means every trial saw the same compute")
        # Collisions bucketed by achieved compute. This is the comparison that
        # transfers between machines; the raw safe/unsafe count does not.
        med = _q(.5)
        for lab, sel in (("faster half (less starved)", lambda v: v <= med),
                         ("slower half (more starved)", lambda v: v > med)):
            g = [r for r in rows if sel(r["probe_after_s"])]
            nc = sum(1 for r in g if r["collision"])
            if g:
                print(f"  {lab:<28} n={len(g):<5} collisions={nc}"
                      f"  ({100*nc/len(g):.2f}%)")

