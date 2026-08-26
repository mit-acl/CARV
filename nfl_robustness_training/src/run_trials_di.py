"""
Run alg13 (DI / double-integrator) N times with random initial states
and plot all trajectories in (position, velocity) phase space.

Half-plane constraints:  pos >= pos_min (0.0)  and  vel >= vel_min (-1.0)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display on a compute node
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import multiprocessing as mp
import os, sys

# Same environment knobs as run_trials_random_obs.py so a cluster run needs no
# source edit:  TTTCARV_TRIALS / TTTCARV_WORKERS / TTTCARV_SEED / TTTCARV_PLOT.
N_TRIALS  = 200
N_WORKERS = 16
# Clamped to the CPUs actually granted: under SLURM the cgroup grants a subset
# of the node, and oversubscribing inflates the ops the 0.20s timestep budget
# is measured against, so the algorithm skips verification it could afford.
_n_cpus   = len(os.sched_getaffinity(0))
N_WORKERS = min(int(os.environ.get('TTTCARV_WORKERS', N_WORKERS)), _n_cpus)
N_TRIALS  = int(os.environ.get('TTTCARV_TRIALS', N_TRIALS))
# Plotting 10k phase-space trajectories is neither useful nor cheap, so it is
# opt-in above a few hundred trials.
PLOT      = os.environ.get('TTTCARV_PLOT', '1' if N_TRIALS <= 500 else '0') == '1'
# cwd has to stay the repo root (utils/nn.py resolves controller_models off it),
# so the figure needs its own path rather than a chdir.
PLOT_OUT  = os.environ.get('TTTCARV_PLOT_OUT', 'trials_di.png')

POS_MIN = 0.0
VEL_MIN = -1.0


# ── Worker process ──

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
# alg13 gates its work on the same 0.20s wall-clock TimeBudget as the unicycle
# algorithms, so its outcomes are a function of achieved compute and are not
# comparable across machines or worker counts unless compute is reported too.
# _probe times a fixed unit of work inside the worker under exactly the
# contention that trial saw. Bucket violations by probe_s, not by hostname.

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
            "trial_seed": os.environ.get("TTTCARV_SEED"),
            "cpus_granted": _n_cpus, "workers": N_WORKERS,
            "python": platform.python_version(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus": os.environ.get("SLURM_CPUS_ON_NODE")}


def _run_trial(args):
    i, seed = args
    from REAL_integrated_sim import setup_analyzer
    from alg13_di import test
    import time as _t
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    probe_before = _probe()
    _t0 = _t.perf_counter()
    state_history, had_violation, u_diffs, mpc_calls, mpc_over = test(
        seed=seed, analyzer=analyzer, pos_min=POS_MIN, vel_min=VEL_MIN,
    )
    wall = _t.perf_counter() - _t0
    probe_after = _probe()
    # Probe both sides: a trial that starts unloaded and ends contended (or the
    # reverse, as the pool drains) has no single regime.
    fp = {"probe_before_s": probe_before, "probe_after_s": probe_after,
          "wall_s": wall,
          "wall_per_timestep_s": wall / max(len(state_history), 1),
          "pid": os.getpid()}
    return (i, seed, state_history, u_diffs, mpc_calls, mpc_over,
            had_violation, fp)


def plot_trials(all_trajectories, all_u_diffs, trial_seeds, danger):
    all_diffs = [d for diffs in all_u_diffs for d in diffs]
    vmax = max(all_diffs) if all_diffs else 1.0
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw constraint boundaries
    ax.axvline(x=POS_MIN, color='red', linewidth=1.5, linestyle='--', label='pos_min')
    ax.axhline(y=VEL_MIN, color='darkred', linewidth=1.5, linestyle='--', label='vel_min')
    ax.fill_betweenx([VEL_MIN - 1, VEL_MIN], -2, 10, color='salmon', alpha=0.3)
    ax.fill_between([POS_MIN - 2, POS_MIN], VEL_MIN - 1, 5, color='salmon', alpha=0.3)

    all_xs, all_ys = [POS_MIN], [VEL_MIN]

    for traj, u_diffs in zip(all_trajectories, all_u_diffs):
        if len(traj) < 2:
            continue
        states = [np.asarray(s).flatten() for s in traj]
        xs = np.array([s[0] for s in states])  # position
        ys = np.array([s[1] for s in states])  # velocity

        points   = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n_seg    = len(segments)
        diffs    = np.array(u_diffs[:n_seg]) if len(u_diffs) >= n_seg else \
                   np.pad(u_diffs, (0, n_seg - len(u_diffs)))

        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidth=1.2, alpha=0.6)
        lc.set_array(diffs)
        ax.add_collection(lc)

        ax.plot(xs[0], ys[0], 'go', markersize=4)
        ax.plot(xs[-1], ys[-1], 'rs', markersize=4)
        all_xs += list(xs)
        all_ys += list(ys)

    mg = 0.3
    ax.set_xlim(min(all_xs) - mg, max(all_xs) + mg)
    ax.set_ylim(min(all_ys) - mg, max(all_ys) + mg)
    ax.set_aspect('auto')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_title(f'DI Phase-Space Trajectories over {len(all_trajectories)} random trials')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('|u_mpc - u_nn|  (MPC override magnitude)')

    ax.legend(handles=[
        patches.Patch(color='salmon', label='Unsafe region'),
        plt.Line2D([0], [0], color='red',    linestyle='--', label=f'pos_min={POS_MIN}'),
        plt.Line2D([0], [0], color='darkred', linestyle='--', label=f'vel_min={VEL_MIN}'),
        plt.Line2D([0], [0], marker='o', color='g', label='Start', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='r', label='End',   linestyle='None'),
    ])
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=400)
    print(f"Saved {PLOT_OUT}")


# ── Main ──

if __name__ == "__main__":
    # Seed the trial list or the run is not comparable to any other run: an
    # unseeded rng gives every invocation different initial states, so two runs
    # differ in BOTH compute and problem and neither can be attributed.
    _trial_seed = os.environ.get('TTTCARV_SEED')
    rng = np.random.default_rng(int(_trial_seed) if _trial_seed else None)
    if _trial_seed is None:
        print("WARNING: TTTCARV_SEED unset -- initial states are freshly random,"
              " so this run is NOT comparable to any other run.")
    args = [(i, int(rng.integers(0, 2**31))) for i in range(N_TRIALS)]

    n_workers = min(N_WORKERS, N_TRIALS)
    print(f"Running {N_TRIALS} DI trials across {n_workers} workers  "
          f"({_n_cpus} CPUs available)...")
    print(f"machine: {_machine()}", flush=True)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results_raw = []
        for result in pool.imap_unordered(_run_trial, args):
            results_raw.append(result)
            i, seed, traj, u_diffs, mpc_calls, mpc_over, had_violation, fp = result
            status = "UNSAFE" if had_violation else "safe"
            pct = f"  ({100*mpc_over/mpc_calls:.0f}%)" if mpc_calls else ""
            print(f"  Trial {i+1:>3}  seed={seed}  [{status}]  "
                  f"states={len(traj)}  mpc={mpc_calls}  over_budget={mpc_over}{pct}",
                  flush=True)

    results_raw.sort(key=lambda r: r[0])

    all_trajectories, all_u_diffs, trial_seeds, safety_record = [], [], [], []
    total_mpc_calls = total_mpc_over = 0
    danger = set()

    probes = []
    for i, seed, traj, u_diffs, mpc_calls, mpc_over, had_violation, fp in results_raw:
        probes.append(fp["probe_before_s"])
        trial_seeds.append(seed)
        all_trajectories.append(traj)
        all_u_diffs.append(u_diffs)
        safety_record.append(not had_violation)
        total_mpc_calls += mpc_calls
        total_mpc_over  += mpc_over
        if had_violation:
            danger.add(i)

    n_safe = sum(safety_record)
    unsafe_seeds = [trial_seeds[i] for i, s in enumerate(safety_record) if not s]
    print(f"\n{'='*50}")
    print(f"Safety record:  {n_safe}/{N_TRIALS} safe  ({100*n_safe/N_TRIALS:.1f}%)")
    print(f"Unsafe trials:  {[i+1 for i, s in enumerate(safety_record) if not s]}")
    print(f"Unsafe seeds:   {unsafe_seeds}")
    print(f"MPC calls:      {total_mpc_calls}  over-budget: {total_mpc_over}"
          f"  ({100*total_mpc_over/max(total_mpc_calls,1):.1f}%)")
    _p = np.array(probes)
    print(f"probe_s: median={np.median(_p):.5f}  min={_p.min():.5f}"
          f"  max={_p.max():.5f}  contention spread={_p.max()/max(_p.min(),1e-9):.1f}x")
    print(f"{'='*50}", flush=True)

    if PLOT:
        plot_trials(all_trajectories, all_u_diffs, trial_seeds, danger)
