"""
CARV Algorithm 1 Animation — time-budget aware with carry-over symbolic.
Mirrors alg1.py exactly. Plots x1 vs x2 with obstacles.

Usage:
    python anim_alg1.py

Saves: carv_rsoa_x1x2.gif
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester, CalculationType
from time_budget import TimeBudget
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import math

# ── Config (matches alg1.py) ──
MAX_TIME = 40
MIN_LOOKAHEAD = 4
MAX_SYMBOLIC_HORIZON = 10

obstacles = [
    np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),  # v = -1 wall
    np.array([[-np.inf, 0.0],   [-np.inf, np.inf]]), # x = 0 wall
]

# ── Setup ──
print("Setting up analyzer...")
analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
tester = ReachabilityTester(analyzer, obstacles)
tester_calibrate = ReachabilityTester(analyzer)
budget = TimeBudget(timestep_budget=0.45)
print("Calibrating time budget...")
budget.calibrate(tester_calibrate, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON, max_backward_horizon=0)
print(f"  concrete_cost = {budget.concrete_cost:.5f}s/step")
print(f"  symbolic_costs = {budget.symbolic_costs}")


def snapshot_rsoa(tester):
    """Read all current tight bounds from tester.horizons.
    Returns dict: {t: {x1, x2, type}} where type is the latest calc_type."""
    snap = {}
    for t, horizon in tester.horizons.items():
        tb = horizon.get_tight_bound()
        if tb is not None:
            calc_type = "empirical"
            for calc_id in sorted(horizon.calculations.keys()):
                calc_type = horizon.calculations[calc_id]["calc_type"].value
            snap[t] = {
                "x1": (float(tb[0, 0]), float(tb[0, 1])),
                "x2": (float(tb[1, 0]), float(tb[1, 1])),
                "type": calc_type,
            }
    return snap


def extend_validation(tester, validated_until, max_time, budget, frames, current_timestep):
    """Mirrors alg1.py extend_validation exactly."""
    if validated_until >= max_time:
        return validated_until

    if not budget.can_afford('concrete'):
        frames.append((
            snapshot_rsoa(tester), current_timestep,
            f"t={current_timestep}  skip ext — budget exhausted"
        ))
        return validated_until

    extension_result = tester.concrete(validated_until, max_time)
    frames.append((
        snapshot_rsoa(tester), current_timestep,
        f"t={current_timestep}  ext concrete [{validated_until}→{max_time}]"
    ))

    if not isinstance(extension_result, dict):
        return max_time

    if extension_result["collision"]:
        ext_conflict_time = extension_result["collision_timestep"]

        # find_nearest_symbolic fallback: search_from - 10
        symbolic_start = max(0, ext_conflict_time - 10)
        horizon = ext_conflict_time - symbolic_start

        if not budget.can_afford('symbolic', horizon):
            frames.append((
                snapshot_rsoa(tester), current_timestep,
                f"t={current_timestep}  skip ext symbolic — budget"
            ))
            return ext_conflict_time - 1

        symbolic_result = tester.symbolic(symbolic_start, ext_conflict_time)
        frames.append((
            snapshot_rsoa(tester), current_timestep,
            f"t={current_timestep}  ext symbolic [{symbolic_start}→{ext_conflict_time}]"
        ))

        if symbolic_result["collision"]:
            return validated_until
        else:
            return ext_conflict_time
    else:
        return max_time


# ══════════════════════════════════════════════
# Run CARV algorithm (alg1), capturing frames
# ══════════════════════════════════════════════
print("Running CARV algorithm (alg1)...")
frames = []  # list of (rsoa_snapshot_dict, current_t, label_str)

current_timestep = 0
validated_until = 0
pending_symbolic_start = None  # carry-over: where to resume symbolic
pending_conflict_time  = None  # carry-over: conflict being verified

frames.append((snapshot_rsoa(tester), 0, "t=0  initial"))

while current_timestep < MAX_TIME:
    budget.start_timestep()
    frames.append((
        snapshot_rsoa(tester), current_timestep,
        f"t={current_timestep}  vu={validated_until}"
    ))

    # ── Carry-over: resume pending symbolic from previous timestep ──
    if pending_symbolic_start is not None:
        chunk_size = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
        verify_end = min(pending_conflict_time, pending_symbolic_start + chunk_size)

        symbolic_reach_set = tester.symbolic(pending_symbolic_start, verify_end)
        frames.append((
            snapshot_rsoa(tester), current_timestep,
            f"t={current_timestep}  [carry] symbolic [{pending_symbolic_start}→{verify_end}]"
        ))

        if verify_end == pending_conflict_time:
            # Finished carry-over
            original_conflict_time = pending_conflict_time
            pending_symbolic_start = None
            pending_conflict_time  = None

            if symbolic_reach_set["collision"]:
                confirmed_t = symbolic_reach_set['collision_timestep']
                validated_until = confirmed_t - 1
                frames.append((
                    snapshot_rsoa(tester), current_timestep,
                    f"STOP — collision confirmed@{confirmed_t}"
                ))
                if confirmed_t - current_timestep < MIN_LOOKAHEAD:
                    break
            else:
                validated_until = original_conflict_time
                frames.append((
                    snapshot_rsoa(tester), current_timestep,
                    f"t={current_timestep}  [carry] deconflicted vu={validated_until}"
                ))
                if budget.remaining > 0:
                    validated_until = extend_validation(
                        tester, validated_until, MAX_TIME, budget, frames, current_timestep
                    )
        else:
            # Still incomplete — fall through to empirical at bottom
            pending_symbolic_start = verify_end
            frames.append((
                snapshot_rsoa(tester), current_timestep,
                f"t={current_timestep}  [carry] still at {verify_end}, continuing next"
            ))

    else:
        # ── Normal: concrete lookahead ──
        explore_from = max(validated_until, current_timestep)
        end_check_time = min(explore_from + budget.max_affordable_concrete(), MAX_TIME)

        concrete_reach_set = tester.concrete(explore_from, end_check_time)
        frames.append((
            snapshot_rsoa(tester), current_timestep,
            f"t={current_timestep}  concrete [{explore_from}→{end_check_time}]"
        ))

        if concrete_reach_set["collision"]:
            conflict_time = concrete_reach_set["collision_timestep"]
            frames.append((
                snapshot_rsoa(tester), current_timestep,
                f"t={current_timestep}  ⚠ collision@{conflict_time}"
            ))

            chunk_size = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
            verify_until = min(conflict_time, current_timestep + chunk_size)

            symbolic_reach_set = tester.symbolic(current_timestep, verify_until)
            frames.append((
                snapshot_rsoa(tester), current_timestep,
                f"t={current_timestep}  symbolic [{current_timestep}→{verify_until}]"
            ))

            if verify_until == conflict_time:
                # Full verification done this timestep
                if symbolic_reach_set["collision"]:
                    confirmed_t = symbolic_reach_set['collision_timestep']
                    validated_until = confirmed_t - 1
                    frames.append((
                        snapshot_rsoa(tester), current_timestep,
                        f"STOP — collision confirmed@{confirmed_t}"
                    ))
                    if conflict_time - current_timestep < MIN_LOOKAHEAD:
                        break
                else:
                    validated_until = conflict_time
                    frames.append((
                        snapshot_rsoa(tester), current_timestep,
                        f"t={current_timestep}  ✓ deconflicted vu={validated_until}"
                    ))
                    if budget.remaining > 0:
                        validated_until = extend_validation(
                            tester, validated_until, MAX_TIME, budget, frames, current_timestep
                        )
            else:
                # Partial verification — carry remainder to next timestep
                validated_until = verify_until
                pending_symbolic_start = verify_until
                pending_conflict_time  = conflict_time
                frames.append((
                    snapshot_rsoa(tester), current_timestep,
                    f"t={current_timestep}  budget exhausted@{verify_until}, defer to next"
                ))
        else:
            validated_until = end_check_time
            frames.append((
                snapshot_rsoa(tester), current_timestep,
                f"t={current_timestep}  ✓ clear vu={validated_until}"
            ))

    if validated_until >= MAX_TIME:
        frames.append((snapshot_rsoa(tester), current_timestep, "Done ✓"))
        break

    tester.real_state_empirical(current_timestep, current_timestep + 1)
    current_timestep += 1
    frames.append((
        snapshot_rsoa(tester), current_timestep,
        f"t={current_timestep}  empirical step  vu={validated_until}"
    ))

print(f"total time: {tester.get_time():.2f} seconds")
print(f"\nGenerated {len(frames)} frames")


# ══════════════════════════════════════════════
# Matplotlib animation: x1 vs x2
# ══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.set_xlabel("x₁", color="#ccc", fontsize=13)
ax.set_ylabel("x₂", color="#ccc", fontsize=13)
title = ax.set_title("", color="#eee", fontsize=12, fontfamily="monospace", pad=12)

# Auto axis limits from finite RSOA values and finite obstacle bounds
all_x1, all_x2 = [], []
for snap, _, _ in frames:
    for t, entry in snap.items():
        for v in entry["x1"]:
            if np.isfinite(v):
                all_x1.append(v)
        for v in entry["x2"]:
            if np.isfinite(v):
                all_x2.append(v)
for obs in obstacles:
    for v in [obs[0, 0], obs[0, 1]]:
        if np.isfinite(v):
            all_x1.append(v)
    for v in [obs[1, 0], obs[1, 1]]:
        if np.isfinite(v):
            all_x2.append(v)

margin = 0.3
x1_lo = (min(all_x1) - margin) if all_x1 else -2
x1_hi = (max(all_x1) + margin) if all_x1 else 2
x2_lo = (min(all_x2) - margin) if all_x2 else -2
x2_hi = (max(all_x2) + margin) if all_x2 else 2
ax.set_xlim(x1_lo, x1_hi)
ax.set_ylim(x2_lo, x2_hi)

# Static: obstacles (clip infinite bounds to plot limits)
for i, obs in enumerate(obstacles):
    lo_x = obs[0, 0] if np.isfinite(obs[0, 0]) else x1_lo
    hi_x = obs[0, 1] if np.isfinite(obs[0, 1]) else x1_hi
    lo_y = obs[1, 0] if np.isfinite(obs[1, 0]) else x2_lo
    hi_y = obs[1, 1] if np.isfinite(obs[1, 1]) else x2_hi
    rect = patches.Rectangle(
        (lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
        linewidth=1.5, edgecolor="#ef4444", facecolor="#ef4444", alpha=0.3, zorder=5,
        label="Obstacle" if i == 0 else None
    )
    ax.add_patch(rect)

COLORS = {
    "concrete":  {"fc": "#f59e0b", "fa": 0.18, "ec": "#f59e0b"},
    "symbolic":  {"fc": "#8b5cf6", "fa": 0.25, "ec": "#8b5cf6"},
    "sampled":   {"fc": "#06b6d4", "fa": 0.18, "ec": "#06b6d4"},
    "empirical": {"fc": "#10b981", "fa": 0.18, "ec": "#10b981"},
}

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    rsoa_map, ct, label = frames[frame_idx]
    title.set_text(label)

    for t, entry in sorted(rsoa_map.items()):
        c = COLORS.get(entry["type"], COLORS["concrete"])
        x1_lo_e, x1_hi_e = entry["x1"]
        x2_lo_e, x2_hi_e = entry["x2"]

        # Clip to finite plot bounds
        x1_lo_e = max(x1_lo_e, x1_lo) if np.isfinite(x1_lo_e) else x1_lo
        x1_hi_e = min(x1_hi_e, x1_hi) if np.isfinite(x1_hi_e) else x1_hi
        x2_lo_e = max(x2_lo_e, x2_lo) if np.isfinite(x2_lo_e) else x2_lo
        x2_hi_e = min(x2_hi_e, x2_hi) if np.isfinite(x2_hi_e) else x2_hi

        if x1_hi_e <= x1_lo_e or x2_hi_e <= x2_lo_e:
            continue

        r = patches.Rectangle(
            (x1_lo_e, x2_lo_e), x1_hi_e - x1_lo_e, x2_hi_e - x2_lo_e,
            linewidth=0.8, edgecolor=c["ec"], facecolor=c["fc"],
            alpha=c["fa"], zorder=2
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

        cx = (x1_lo_e + x1_hi_e) / 2
        cy = (x2_lo_e + x2_hi_e) / 2
        txt = ax.text(cx, cy, str(t), fontsize=6, color=c["ec"], alpha=0.8,
                      ha="center", va="center", zorder=3, fontfamily="monospace")
        dynamic_artists.append(txt)

    # Blue dot at current timestep center
    if ct in rsoa_map:
        entry = rsoa_map[ct]
        cx = sum(entry["x1"]) / 2
        cy = sum(entry["x2"]) / 2
        dot = ax.plot(cx, cy, "o", color="#3b82f6", markersize=8, zorder=6)[0]
        dynamic_artists.append(dot)
        lbl = ax.text(cx + 0.05, cy + 0.04, f"t={ct}", color="#3b82f6",
                      fontsize=9, fontfamily="monospace", zorder=6)
        dynamic_artists.append(lbl)

    return dynamic_artists


# Legend
legend_els = [
    Patch(facecolor="#f59e0b", alpha=0.3, edgecolor="#f59e0b", label="Concrete RSOA"),
    Patch(facecolor="#8b5cf6", alpha=0.3, edgecolor="#8b5cf6", label="Symbolic RSOA"),
    Patch(facecolor="#10b981", alpha=0.3, edgecolor="#10b981", label="Empirical (Kalman)"),
    Patch(facecolor="#ef4444", alpha=0.4, edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o", color="#3b82f6", ls="", markersize=6, label="Current t"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="#1a1e28", edgecolor="#333", labelcolor="#aaa")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              interval=500, blit=False, repeat=True)

out_path = "carv_rsoa_x1x2.gif"
ani.save(out_path, writer="pillow", fps=1.5, dpi=130)
print(f"Saved to {out_path}")
plt.close()
