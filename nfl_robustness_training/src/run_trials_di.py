"""
Run alg13 (DI / double-integrator) N times with random initial states
and plot all trajectories in (position, velocity) phase space.

Half-plane constraints:  pos >= pos_min (0.0)  and  vel >= vel_min (-1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import multiprocessing as mp
import os, sys

N_TRIALS  = 200
N_WORKERS = 16

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


def _run_trial(args):
    i, seed = args
    from REAL_integrated_sim import setup_analyzer
    from alg13_di import test
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    state_history, had_violation, u_diffs, mpc_calls, mpc_over = test(
        seed=seed, analyzer=analyzer, pos_min=POS_MIN, vel_min=VEL_MIN,
    )
    return i, seed, state_history, u_diffs, mpc_calls, mpc_over, had_violation


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
    plt.savefig('trials_di.png', dpi=400)
    print("Saved trials_di.png")
    plt.show()


# ── Main ──

if __name__ == "__main__":
    rng  = np.random.default_rng()
    args = [(i, int(rng.integers(0, 2**31))) for i in range(N_TRIALS)]

    n_workers = min(N_WORKERS, N_TRIALS)
    print(f"Running {N_TRIALS} DI trials across {n_workers} workers...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results_raw = []
        for result in pool.imap_unordered(_run_trial, args):
            results_raw.append(result)
            i, seed, traj, u_diffs, mpc_calls, mpc_over, had_violation = result
            status = "UNSAFE" if had_violation else "safe"
            pct = f"  ({100*mpc_over/mpc_calls:.0f}%)" if mpc_calls else ""
            print(f"  Trial {i+1:>3}  seed={seed}  [{status}]  "
                  f"states={len(traj)}  mpc={mpc_calls}  over_budget={mpc_over}{pct}",
                  flush=True)

    results_raw.sort(key=lambda r: r[0])

    all_trajectories, all_u_diffs, trial_seeds, safety_record = [], [], [], []
    total_mpc_calls = total_mpc_over = 0
    danger = set()

    for i, seed, traj, u_diffs, mpc_calls, mpc_over, had_violation in results_raw:
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
    print(f"{'='*50}")

    plot_trials(all_trajectories, all_u_diffs, trial_seeds, danger)
