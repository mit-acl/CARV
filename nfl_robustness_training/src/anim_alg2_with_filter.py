"""
Animation file version of alg2_with_filter.py. run on the DoubleIntegrator model
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from alg2_with_filter import concrete_scan, symbolic_step, VerificationTask
from time_budget import TimeBudget

# ── Config ──
MAX_TIME             = 40
MAX_SYMBOLIC_HORIZON = 10

obstacles = [
    np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),
    np.array([[-np.inf, 1.0],    [-np.inf, np.inf]]),
]

# ── Setup ──
print("Setting up analyzer...")
analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
tester          = ReachabilityTester(analyzer, obstacles)
tester_calibrate = ReachabilityTester(analyzer, obstacles)
budget = TimeBudget(timestep_budget=0.40)
print("Calibrating time budget...")
budget.calibrate(tester_calibrate, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON, max_backward_horizon=0)


def snapshot_rsoa(tester):
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


def try_extend(tester, validated_until, max_time, budget, max_symbolic_horizon, current_timestep):
    while validated_until < max_time:
        if not budget.can_afford('concrete'):
            break

        end_check_time = min(validated_until + budget.max_affordable_concrete(), max_time)
        collision, conflict_time = concrete_scan(tester, validated_until, end_check_time)
        frames.append((
            snapshot_rsoa(tester), current_timestep, None,
            f"Step {current_timestep}: extending — concrete scan {validated_until}→{end_check_time}",
        ))

        if not collision:
            validated_until = end_check_time
            break

        if not budget.can_afford('symbolic', 1):
            validated_until = conflict_time - 1
            return validated_until, VerificationTask(symbolic_start=conflict_time - 1, conflict_time=conflict_time)

        chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        sym_from = job.symbolic_start
        sym_end  = min(sym_from + chunk_size, conflict_time)
        job, result = symbolic_step(tester, job, chunk_size)
        frames.append((
            snapshot_rsoa(tester), current_timestep, None,
            f"Step {current_timestep}: extending — symbolic verify {sym_from}→{sym_end}",
        ))

        if result is None:
            validated_until = job.symbolic_start
            return validated_until, job

        validated_until = conflict_time - 1 if result["collision"] else conflict_time
        frames.append((
            snapshot_rsoa(tester), current_timestep, None,
            f"Step {current_timestep}: extending — {'conflict confirmed ' + str(conflict_time) + ', stopping extension' if result['collision'] else 'false alarm, safe through ' + str(validated_until)}",
        ))
        if result["collision"]:
            return validated_until, None

    return validated_until, None


# ══════════════════════════════════════════════
# Run algorithm, capturing frames
# frames: (rsoa_snapshot, current_t, sf_traj_or_None, label)
#   sf_traj: list of (bounds, kind) where kind = 'nominal'|'backup'|'collision'
# ══════════════════════════════════════════════
print("Running CARV + safety filter...")
frames = []

current_timestep = 0
validated_until  = 0
pending_job      = None

frames.append((snapshot_rsoa(tester), 0, None, "Step 0: start"))

while current_timestep < MAX_TIME:
    budget.start_timestep()
    frames.append((snapshot_rsoa(tester), current_timestep, None,
                   f"Step {current_timestep}: begin"))
    print(f"\nCURRENT TIMESTEP ==== {current_timestep}")

    # ── Phase 1: carry-over symbolic ──────────────────────────────────
    if pending_job is not None:
        sym_from = pending_job.symbolic_start
        sym_to   = pending_job.conflict_time
        chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
        pending_job, result = symbolic_step(tester, pending_job, chunk_size)

        frames.append((
            snapshot_rsoa(tester), current_timestep, None,
            f"Step {current_timestep}: symbolic verify {sym_from}→{min(sym_from + chunk_size, sym_to)} (carry-over)",
        ))

        if result is not None:
            conflict_time   = pending_job.conflict_time
            validated_until = conflict_time - 1 if result["collision"] else conflict_time
            frames.append((
                snapshot_rsoa(tester), current_timestep, None,
                f"Step {current_timestep}: {'conflict confirmed t=' + str(conflict_time) + ' — deferring to SF' if result['collision'] else 'deconflicted — safe through t=' + str(validated_until)}",
            ))
            if not result["collision"] and budget.remaining > 0:
                validated_until, pending_job = try_extend(
                    tester, validated_until, MAX_TIME, budget,
                    MAX_SYMBOLIC_HORIZON, current_timestep
                )
                frames.append((
                    snapshot_rsoa(tester), current_timestep, None,
                    f"Step {current_timestep}: done extending, safe through t={validated_until}",
                ))
            else:
                pending_job = None

    # ── Phase 2: normal concrete scan ─────────────────────────────────
    else:
        explore_from   = max(validated_until, current_timestep)
        end_check_time = min(explore_from + budget.max_affordable_concrete(), MAX_TIME)
        collision, conflict_time = concrete_scan(tester, explore_from, end_check_time)

        frames.append((
            snapshot_rsoa(tester), current_timestep, None,
            f"Step {current_timestep}: concrete scan {explore_from}→{end_check_time}",
        ))

        if not collision:
            validated_until = end_check_time
            frames.append((
                snapshot_rsoa(tester), current_timestep, None,
                f"Step {current_timestep}: all clear, safe through t={validated_until}",
            ))
        else:
            frames.append((
                snapshot_rsoa(tester), current_timestep, None,
                f"Step {current_timestep}: concrete conflict at t={conflict_time}, verifying...",
            ))
            chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
            pending_job = VerificationTask(symbolic_start=current_timestep, conflict_time=conflict_time)
            sym_end     = min(current_timestep + chunk_size, conflict_time)
            pending_job, result = symbolic_step(tester, pending_job, chunk_size)

            frames.append((
                snapshot_rsoa(tester), current_timestep, None,
                f"Step {current_timestep}: symbolic verify {current_timestep}→{sym_end}",
            ))

            if result is not None:
                validated_until = conflict_time - 1 if result["collision"] else conflict_time
                pending_job = None
                frames.append((
                    snapshot_rsoa(tester), current_timestep, None,
                    f"Step {current_timestep}: {'conflict confirmed t=' + str(conflict_time) + ' — deferring to SF' if result['collision'] else 'deconflicted — safe through t=' + str(validated_until)}",
                ))
                if not result["collision"] and budget.remaining > 0:
                    validated_until, pending_job = try_extend(
                        tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep
                    )
                    frames.append((
                        snapshot_rsoa(tester), current_timestep, None,
                        f"Step {current_timestep}: done extending, safe through t={validated_until}",
                    ))
            else:
                validated_until = pending_job.symbolic_start
                print(f"Budget exhausted at t={validated_until}, deferring")

    if validated_until >= MAX_TIME:
        frames.append((snapshot_rsoa(tester), current_timestep, None, "Done — fully verified"))
        break

    # ── Safety filter check ────────────────────────────────────────────
    sf = tester.safety_filter
    current_bounds = tester.horizons[current_timestep].get_tight_bound()
    sf_result = sf.filter(current_bounds)

    # Build annotated trajectory for drawing: list of (bounds, kind)
    sf_traj = []
    for i, b in enumerate(sf_result['trajectory']):
        if i == 0:
            continue  # skip current bounds (already drawn as RSOA)
        if sf_result['collision_at'] is not None and i == sf_result['collision_at']:
            kind = 'collision'
        elif i == 1:
            kind = 'nominal'
        else:
            kind = 'backup'
        sf_traj.append((b, kind))

    if sf_result['intervened']:
        label = f"Step {current_timestep}: SF INTERVENED at step {sf_result['collision_at']} — {sf_result['reason']}"
    else:
        label = f"Step {current_timestep}: SF clear — nominal safe"

    frames.append((snapshot_rsoa(tester), current_timestep, sf_traj, label))

    if sf_result['intervened']:
        print(f"[SafetyFilter t={current_timestep}] INTERVENED — stopping ")
        break

    tester.real_state_empirical(current_timestep, current_timestep + 1)
    print(f"Safety margin: {validated_until - current_timestep} steps ahead")
    current_timestep += 1
    frames.append((
        snapshot_rsoa(tester), current_timestep, None,
        f"Step {current_timestep}: advance to next timestep",
    ))

print(f"total time: {tester.get_time():.2f}s")
print(f"Generated {len(frames)} frames")


# ══════════════════════════════════════════════
# Matplotlib animation
# ══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.set_xlabel("x₁ (position)", color="#ccc", fontsize=13)
ax.set_ylabel("x₂ (velocity)", color="#ccc", fontsize=13)
title = ax.set_title("", color="#eee", fontsize=11, fontfamily="monospace", pad=12)

# Axis limits
all_x1, all_x2 = [], []
for snap, _, sf_traj, _ in frames:
    for t, entry in snap.items():
        all_x1.extend(entry["x1"])
        all_x2.extend(entry["x2"])
    if sf_traj:
        for b, _ in sf_traj:
            if np.all(np.isfinite(b)):
                all_x1.extend([float(b[0, 0]), float(b[0, 1])])
                all_x2.extend([float(b[1, 0]), float(b[1, 1])])
for obs in obstacles:
    all_x1.extend([float(v) for v in [obs[0, 0], obs[0, 1]] if np.isfinite(v)])
    all_x2.extend([float(v) for v in [obs[1, 0], obs[1, 1]] if np.isfinite(v)])

margin = 0.3
x1_min = min(all_x1) - margin;  x1_max = max(all_x1) + margin
x2_min = min(all_x2) - margin;  x2_max = max(all_x2) + margin
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

# Static obstacles
for i, obs in enumerate(obstacles):
    x_lo = float(np.clip(obs[0, 0], x1_min, x1_max))
    x_hi = float(np.clip(obs[0, 1], x1_min, x1_max))
    y_lo = float(np.clip(obs[1, 0], x2_min, x2_max))
    y_hi = float(np.clip(obs[1, 1], x2_min, x2_max))
    ax.add_patch(patches.Rectangle(
        (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
        linewidth=1.5, edgecolor="#ef4444", facecolor="#ef4444", alpha=0.5, zorder=5,
        label="Obstacle" if i == 0 else None
    ))

RSOA_COLORS = {
    "concrete":  {"fc": "#f59e0b", "fa": 0.18, "ec": "#f59e0b"},
    "symbolic":  {"fc": "#8b5cf6", "fa": 0.25, "ec": "#8b5cf6"},
    "sampled":   {"fc": "#06b6d4", "fa": 0.18, "ec": "#06b6d4"},
    "empirical": {"fc": "#10b981", "fa": 0.18, "ec": "#10b981"},
}
SF_COLORS = {
    "nominal":   {"ec": "#f97316", "lw": 1.8, "ls": "--", "fa": 0.10},  # orange dashed
    "backup":    {"ec": "#22d3ee", "lw": 1.2, "ls": ":",  "fa": 0.08},  # cyan dotted
    "collision": {"ec": "#ef4444", "lw": 2.0, "ls": "-",  "fa": 0.20},  # red solid
}

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    snap, ct, sf_traj, label = frames[frame_idx]
    title.set_text(label)

    # RSOA boxes
    for t, entry in sorted(snap.items()):
        c = RSOA_COLORS.get(entry["type"], RSOA_COLORS["concrete"])
        x1_lo, x1_hi = entry["x1"]
        x2_lo, x2_hi = entry["x2"]
        r = patches.Rectangle(
            (x1_lo, x2_lo), x1_hi - x1_lo, x2_hi - x2_lo,
            linewidth=0.8, edgecolor=c["ec"], facecolor=c["fc"],
            alpha=c["fa"], zorder=2
        )
        ax.add_patch(r)
        dynamic_artists.append(r)
        cx, cy = (x1_lo + x1_hi) / 2, (x2_lo + x2_hi) / 2
        txt = ax.text(cx, cy, str(t), fontsize=6, color=c["ec"], alpha=0.8,
                      ha="center", va="center", zorder=3, fontfamily="monospace")
        dynamic_artists.append(txt)

    # Safety filter trajectory
    if sf_traj:
        for step_idx, (b, kind) in enumerate(sf_traj):
            if not np.all(np.isfinite(b)):
                continue
            c = SF_COLORS[kind]
            r = patches.Rectangle(
                (b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                linewidth=c["lw"], edgecolor=c["ec"], facecolor=c["ec"],
                alpha=c["fa"], linestyle=c["ls"], zorder=4
            )
            ax.add_patch(r)
            dynamic_artists.append(r)
            # Step number
            cx, cy = (b[0, 0] + b[0, 1]) / 2, (b[1, 0] + b[1, 1]) / 2
            lbl = ax.text(cx, cy, f"+{step_idx + 1}", fontsize=5, color=c["ec"],
                          alpha=0.9, ha="center", va="center", zorder=5,
                          fontfamily="monospace")
            dynamic_artists.append(lbl)

    # Blue dot at current timestep
    if ct in snap:
        entry = snap[ct]
        cx = sum(entry["x1"]) / 2
        cy = sum(entry["x2"]) / 2
        dot = ax.plot(cx, cy, "o", color="#3b82f6", markersize=8, zorder=7)[0]
        dynamic_artists.append(dot)
        lbl = ax.text(cx + 0.05, cy + 0.04, f"t={ct}", color="#3b82f6",
                      fontsize=9, fontfamily="monospace", zorder=7)
        dynamic_artists.append(lbl)

    return dynamic_artists


# Legend
legend_els = [
    Patch(facecolor="#f59e0b", alpha=0.3, edgecolor="#f59e0b", label="Concrete RSOA"),
    Patch(facecolor="#8b5cf6", alpha=0.3, edgecolor="#8b5cf6", label="Symbolic RSOA"),
    Patch(facecolor="#10b981", alpha=0.3, edgecolor="#10b981", label="Empirical (Kalman)"),
    Patch(facecolor="#ef4444", alpha=0.5, edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o", color="#3b82f6", ls="", markersize=6, label="Current t"),
    Line2D([0], [0], color="#f003ec", lw=1.8, ls="--", label="SF nominal step"),
    Line2D([0], [0], color="#0dcfe4", lw=1.2, ls=":",  label="SF backup steps"),
    Line2D([0], [0], color="#ef4444", lw=2.0, ls="-",  label="SF collision step"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="#1a1e28", edgecolor="#333", labelcolor="#aaa")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              interval=600, blit=False, repeat=True)

out_path = "alg2_filter.gif"
ani.save(out_path, writer="pillow", fps=0.75, dpi=130)
print(f"Saved to {out_path}")
plt.close()
