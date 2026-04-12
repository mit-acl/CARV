"""
Run alg8 N times with random initial states and plot all trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import multiprocessing as mp
import os, sys

N_TRIALS  = 20
N_WORKERS = 16


#error inducing obstacle. seed 1266063356
# obstacles = [
#     np.array([-6.5,  2.02,  0.5]),
#     np.array([-3.2,  1.21,  0.5]),
#     np.array([-1.9,  -0.87, 0.5]),
# ]

obstacles = [
        np.array([-6.5, 2.02,  0.5]),
        np.array([-3.2,  1.21,  0.5]),
        np.array([-1.5,  -0.85, 0.45]),
    ]


def point_collision(traj, obstacles):
    for s in traj:
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx)**2 + (s_flat[1] - cy)**2 <= r**2:
                return True
    return False


# ── Worker process ──

def _worker_init():
    """Called once per worker process at pool startup."""
    # Suppress verbose output inside workers
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _run_trial(args):
    """Worker function: run one trial with a fresh analyzer, return results."""
    i, seed = args
    from REAL_integrated_sim import setup_analyzer
    from alg8_for_run_trial import test
    analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')
    traj, _, u_diffs, mpc_calls, mpc_over = test(seed=seed, analyzer=analyzer)
    had_collision = point_collision(traj, obstacles)
    return i, seed, traj, u_diffs, mpc_calls, mpc_over, had_collision


def plot_trials(all_trajectories, all_u_diffs, obstacles, trial_seeds, danger):
    all_diffs = [d for diffs in all_u_diffs for d in diffs]
    vmax = max(all_diffs) if all_diffs else 1.0
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 6))

    for obs in obstacles:
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        ax.add_patch(patches.Circle(
            (cx, cy), r,
            linewidth=1, edgecolor='red', facecolor='salmon', alpha=0.6,
        ))

    all_xs, all_ys = [], []
    for obs in obstacles:
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        all_xs += [cx - r, cx + r]
        all_ys += [cy - r, cy + r]

    for num, (traj, u_diffs) in enumerate(zip(all_trajectories, all_u_diffs)):
        if len(traj) < 2:
            continue
        xs = np.array([np.asarray(s).flatten()[0] for s in traj])
        ys = np.array([np.asarray(s).flatten()[1] for s in traj])

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

    mg = 0.5
    ax.set_xlim(min(all_xs) - mg, max(all_xs) + mg)
    ax.set_ylim(min(all_ys) - mg, max(all_ys) + mg)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Trajectories over {len(all_trajectories)} random trials')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('|u_mpc - u_nn|  (MPC override magnitude)')

    ax.legend(handles=[
        patches.Patch(color='salmon', label='Obstacle'),
        plt.Line2D([0], [0], marker='o', color='g', label='Start', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='r', label='End',   linestyle='None'),
    ])
    plt.tight_layout()
    plt.savefig('trials.png', dpi=400)
    print("Saved trials.png")
    plt.show()


# ── Main ──

if __name__ == "__main__":
    rng  = np.random.default_rng()
    args = [(i, int(rng.integers(0, 2**31))) for i in range(N_TRIALS)]

    n_workers = min(N_WORKERS, N_TRIALS)
    print(f"Running {N_TRIALS} trials across {n_workers} workers...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results_raw = []
        for result in pool.imap_unordered(_run_trial, args):
            results_raw.append(result)
            i, seed, traj, u_diffs, mpc_calls, mpc_over, had_collision = result
            status = "UNSAFE" if had_collision else "safe"
            pct = f"  ({100*mpc_over/mpc_calls:.0f}%)" if mpc_calls else ""
            print(f"  Trial {i+1:>3}  seed={seed}  [{status}]  "
                  f"states={len(traj)}  mpc={mpc_calls}  over_budget={mpc_over}{pct}",
                  flush=True)

    # Sort back into trial order
    results_raw.sort(key=lambda r: r[0])

    all_trajectories, all_u_diffs, trial_seeds, safety_record = [], [], [], []
    total_mpc_calls = total_mpc_over = 0
    danger = set()

    for i, seed, traj, u_diffs, mpc_calls, mpc_over, had_collision in results_raw:
        trial_seeds.append(seed)
        all_trajectories.append(traj)
        all_u_diffs.append(u_diffs)
        safety_record.append(not had_collision)
        total_mpc_calls += mpc_calls
        total_mpc_over  += mpc_over
        if had_collision:
            danger.add(i)

    n_safe = sum(safety_record)
    unsafe_seeds = [trial_seeds[i] for i, s in enumerate(safety_record) if not s]
    print(f"\n{'='*50}")
    print(f"Safety record:         {n_safe}/{N_TRIALS} safe  ({100*n_safe/N_TRIALS:.1f}%)")
    print(f"Unsafe trials:         {[i+1 for i, s in enumerate(safety_record) if not s]}")
    print(f"Unsafe seeds:          {unsafe_seeds}")
    # print(f"MPC calls total:       {total_mpc_calls}")
    # print(f"MPC over budget:       {total_mpc_over}/{total_mpc_calls}"
    #       + (f"  ({100*total_mpc_over/total_mpc_calls:.1f}%)" if total_mpc_calls else ""))
    # print(f"MPC over budget/trial: {total_mpc_over/N_TRIALS:.2f} avg")
    print(f"{'='*50}")

    plot_trials(all_trajectories, all_u_diffs, obstacles, trial_seeds, danger)
