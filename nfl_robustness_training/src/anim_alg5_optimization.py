"""
Animation for alg5_optimization_stop_concrete_scan.

Same visual style as anim_alg4. Key behavioural difference:
  - try_extend_anim is capped at current_timestep + MIN_SAFE_HORIZON
  - Baseline concrete scan is also capped at the same ceiling
  - Optimizer is called immediately when baseline scan reaches the ceiling

Frame tuple: (rsoa_snapshot, current_t, label, origin)
  origin: "baseline" | "optimizer" | "info"
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
from multi_obj_opt.strategies import ExtensionOptimizer
from alg5_optimization_stop_concrete_scan import concrete_scan, symbolic_step, VerificationTask, _get_volume
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional

# ── Config ──
MAX_TIME             = 40
MIN_LOOKAHEAD        = 4
MIN_SAFE_HORIZON     = 6
MAX_SYMBOLIC_HORIZON = 10

obstacles = [
    np.array([[-np.inf, np.inf], [-np.inf, -1.0]]),
    np.array([[-np.inf, 0.0],    [-np.inf, np.inf]]),
]

# ── Setup ──
print("Setting up analyzer...")
analyzer         = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
tester           = ReachabilityTester(analyzer, obstacles)
tester_calibrate = ReachabilityTester(analyzer)
budget           = TimeBudget(timestep_budget=0.40)
print("Calibrating time budget...")
budget.calibrate(tester_calibrate, max_symbolic_horizon=MAX_SYMBOLIC_HORIZON,
                 max_backward_horizon=0)

ext_optimizer = ExtensionOptimizer()
ext_optimizer.set_timing_params({
    "concrete_slope":     0.006311738129818,
    "concrete_intercept": 0.0035703865687052,
    "ratio_slope":        0.5341915550605524,
    "ratio_intercept":   -0.0578267576662421,
})


# ══════════════════════════════════════════════
# Frame helpers
# ══════════════════════════════════════════════

def snapshot_rsoa(tester):
    snap = {}
    for t, horizon in tester.horizons.items():
        tb = horizon.get_tight_bound()
        if tb is not None:
            calc_type = "empirical"
            for calc_id in sorted(horizon.calculations.keys()):
                calc_type = horizon.calculations[calc_id]["calc_type"].value
            snap[t] = {
                "x1":  (float(tb[0, 0]), float(tb[0, 1])),
                "x2":  (float(tb[1, 0]), float(tb[1, 1])),
                "type": calc_type,
            }
    return snap


def push(frames, tester, current_t, label, origin="info"):
    """Append a frame. origin: 'baseline' | 'optimizer' | 'info'"""
    frames.append((snapshot_rsoa(tester), current_t, label, origin))


# ══════════════════════════════════════════════
# Instrumented algorithm helpers
# ══════════════════════════════════════════════

def try_extend_anim(frames, tester, validated_until, max_time, budget,
                    max_symbolic_horizon, current_timestep, min_lookahead,
                    origin="baseline", safe_horizon_ceiling=None):
    """try_extend with frame captures. origin propagated to all frames.
    safe_horizon_ceiling caps the scan so the optimizer can take over from there."""
    scan_ceiling = safe_horizon_ceiling if safe_horizon_ceiling is not None else max_time
    validated_until_inner = validated_until
    while validated_until_inner < scan_ceiling:
        if not budget.can_afford('concrete'):
            push(frames, tester, current_timestep,
                 f"t={current_timestep}  [extend] budget exhausted", origin)
            break

        end_check_time = min(validated_until_inner + budget.max_affordable_concrete(),
                             scan_ceiling, max_time)
        result = tester.concrete(validated_until_inner, end_check_time)
        collision = result["collision"]
        conflict_time = result.get("collision_timestep")
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [extend] concrete {validated_until_inner}→{end_check_time}"
             + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
             origin)

        if not collision:
            validated_until_inner = end_check_time
            break

        if not budget.can_afford('symbolic', 1):
            validated_until_inner = conflict_time - 1
            return validated_until_inner, VerificationTask(
                symbolic_start=conflict_time - 1, conflict_time=conflict_time)

        chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon)
        job = VerificationTask(symbolic_start=validated_until_inner,
                               conflict_time=conflict_time)
        sym_start = job.symbolic_start
        job, result_s = symbolic_step(tester, job, chunk_size)
        sym_end = job.symbolic_start
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [extend] symbolic {sym_start}→{sym_end}",
             origin)

        if result_s is None:
            validated_until_inner = job.symbolic_start
            return validated_until_inner, job

        validated_until_inner = conflict_time - 1 if result_s["collision"] else conflict_time
        force_stop = result_s["collision"] and \
                     (conflict_time - current_timestep) < min_lookahead
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [extend] "
             + (f"conflict confirmed@{conflict_time}"
                + (" FORCED STOP" if force_stop else " — deferring")
                if result_s["collision"]
                else f"✓ deconflicted vu={validated_until_inner}"),
             origin)
        if force_stop or result_s["collision"]:
            return validated_until_inner, None

    return validated_until_inner, None


def optimized_step_anim(frames, tester, validated_until, max_time, budget,
                        max_symbolic_horizon, current_timestep, min_lookahead,
                        ext_optimizer):
    """optimized_step with frame captures. All frames tagged 'optimizer'."""
    target = validated_until + 1
    if target > max_time:
        return validated_until, None

    current_vol  = _get_volume(tester, current_timestep)
    verified_vol = _get_volume(tester, validated_until)

    method = ext_optimizer.get_strategy(
        current_timestep=current_timestep,
        verified_until=validated_until,
        current_vol=current_vol,
        verified_vol=verified_vol,
        time_budget=budget.remaining,
    )

    push(frames, tester, current_timestep,
         f"t={current_timestep}  [OPT] optimizer → {method}  "
         f"cur_vol={current_vol:.3f}  ver_vol={verified_vol:.3f}",
         "optimizer")

    # ── CONCRETE branch ──────────────────────────────────────────────
    if method == "concrete":
        result = tester.concrete(validated_until, target)
        collision    = result["collision"]
        conflict_time = result.get("collision_timestep")
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [OPT] concrete {validated_until}→{target}"
             + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
             "optimizer")

        if not collision:
            return target, None

        if not budget.can_afford('symbolic', 1):
            return conflict_time - 1, VerificationTask(
                symbolic_start=conflict_time - 1, conflict_time=conflict_time)

        chunk_size = min(budget.max_affordable_symbolic(), max_symbolic_horizon,
                         conflict_time - validated_until)
        job = VerificationTask(symbolic_start=validated_until, conflict_time=conflict_time)
        sym_start = job.symbolic_start
        job, result_s = symbolic_step(tester, job, chunk_size)
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [OPT] symbolic verify {sym_start}→{job.symbolic_start}",
             "optimizer")

        if result_s is None:
            return job.symbolic_start, job

        if result_s["collision"]:
            force_stop = (conflict_time - current_timestep) < min_lookahead
            push(frames, tester, current_timestep,
                 f"t={current_timestep}  [OPT] conflict confirmed@{conflict_time}"
                 + (" FORCED STOP" if force_stop else " — deferring"),
                 "optimizer")
            return conflict_time - 1, None

        push(frames, tester, current_timestep,
             f"t={current_timestep}  [OPT] deconflicted vu={conflict_time}", "optimizer")
        return conflict_time, None

    # ── SYMBOLIC branch ──────────────────────────────────────────────
    else:
        k = target - current_timestep
        actual_k      = min(k, max_symbolic_horizon, budget.max_affordable_symbolic())
        actual_target = current_timestep + actual_k

        job = VerificationTask(symbolic_start=current_timestep, conflict_time=actual_target)
        job, result_s = symbolic_step(tester, job, actual_k)
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [OPT] symbolic {current_timestep}→{actual_target}"
             + (f"  ⚠ conflict" if (result_s and result_s["collision"]) else "  ✓ clean"),
             "optimizer")

        if result_s is None:
            return job.symbolic_start, job

        if result_s["collision"]:
            force_stop = (actual_target - current_timestep) < min_lookahead
            push(frames, tester, current_timestep,
                 f"t={current_timestep}  [OPT] conflict before t={actual_target}"
                 + (" FORCED STOP" if force_stop else " — deferring"),
                 "optimizer")
            return actual_target - 1, None

        push(frames, tester, current_timestep,
             f"t={current_timestep}  [OPT] symbolic clean vu={actual_target}", "optimizer")
        return actual_target, None


# ══════════════════════════════════════════════
# Run alg5_optimization_stop_concrete_scan, capturing frames
# ══════════════════════════════════════════════
print("Running alg5_optimization_stop_concrete_scan...")
frames = []

current_timestep = 0
validated_until  = 0
pending_job: Optional[VerificationTask] = None

push(frames, tester, 0, "t=0  initial", "info")

while current_timestep < MAX_TIME:
    budget.start_timestep()

    safety_margin = validated_until - current_timestep
    mode = "OPTIMIZED" if safety_margin >= MIN_SAFE_HORIZON else "BASELINE"
    push(frames, tester, current_timestep,
         f"t={current_timestep}  vu={validated_until}  margin={safety_margin}  [{mode}]",
         "info")

    # ── Phase 1: carry-over ────────────────────────────────────────────
    if pending_job is not None:
        sym_from   = pending_job.symbolic_start
        chunk_size = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
        pending_job, result = symbolic_step(tester, pending_job, chunk_size)
        push(frames, tester, current_timestep,
             f"t={current_timestep}  [carry] symbolic {sym_from}→{pending_job.symbolic_start if pending_job else '?'}",
             "baseline")

        if result is not None:
            conflict_time   = pending_job.conflict_time
            validated_until = conflict_time - 1 if result["collision"] else conflict_time
            force_stop      = result["collision"] and \
                              (conflict_time - current_timestep) < MIN_LOOKAHEAD
            push(frames, tester, current_timestep,
                 f"t={current_timestep}  [carry] "
                 + (f"conflict confirmed@{conflict_time}"
                    + (" FORCED STOP" if force_stop else " — deferring")
                    if result["collision"]
                    else f"✓ deconflicted vu={validated_until}"),
                 "baseline")
            if force_stop:
                break
            if not result["collision"] and budget.remaining > 0:
                validated_until, pending_job = try_extend_anim(
                    frames, tester, validated_until, MAX_TIME, budget,
                    MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                    origin="baseline",
                    safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                )
            else:
                pending_job = None

    # ── Phase 2: optimized or baseline ────────────────────────────────
    else:
        safety_margin = validated_until - current_timestep

        if safety_margin >= MIN_SAFE_HORIZON:
            # ── OPTIMIZED ────────────────────────────────────────────
            validated_until, pending_job = optimized_step_anim(
                frames, tester, validated_until, MAX_TIME, budget,
                MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                ext_optimizer
            )
            new_margin = validated_until - current_timestep
            if new_margin < MIN_SAFE_HORIZON and pending_job is None \
                    and budget.remaining > 0:
                push(frames, tester, current_timestep,
                     f"t={current_timestep}  [OPT] margin dropped to {new_margin} "
                     f"— recovering with baseline",
                     "info")
                validated_until, pending_job = try_extend_anim(
                    frames, tester, validated_until, MAX_TIME, budget,
                    MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                    origin="baseline",
                    safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                )

        else:
            # ── BASELINE ─────────────────────────────────────────────
            explore_from   = max(validated_until, current_timestep)
            end_check_time = min(
                explore_from + budget.max_affordable_concrete(),
                current_timestep + MIN_SAFE_HORIZON,   # never scan past ceiling
                MAX_TIME
            )
            result = tester.concrete(explore_from, end_check_time)
            collision     = result["collision"]
            conflict_time = result.get("collision_timestep")
            push(frames, tester, current_timestep,
                 f"t={current_timestep}  [BASE] concrete {explore_from}→{end_check_time}"
                 + (f"  ⚠ conflict@{conflict_time}" if collision else "  ✓ clear"),
                 "baseline")

            if not collision:
                validated_until = end_check_time
                # If we just hit the ceiling, hand off to optimizer
                new_margin = validated_until - current_timestep
                if new_margin >= MIN_SAFE_HORIZON and budget.remaining > 0:
                    push(frames, tester, current_timestep,
                         f"t={current_timestep}  [BASE→OPT] reached MIN_SAFE_HORIZON "
                         f"— handing off to optimizer",
                         "info")
                    validated_until, pending_job = optimized_step_anim(
                        frames, tester, validated_until, MAX_TIME, budget,
                        MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                        ext_optimizer
                    )
            else:
                chunk_size  = min(budget.max_affordable_symbolic(), MAX_SYMBOLIC_HORIZON)
                pending_job = VerificationTask(symbolic_start=current_timestep,
                                               conflict_time=conflict_time)
                sym_start   = pending_job.symbolic_start
                pending_job, result_s = symbolic_step(tester, pending_job, chunk_size)
                push(frames, tester, current_timestep,
                     f"t={current_timestep}  [BASE] symbolic {sym_start}→"
                     f"{pending_job.symbolic_start if pending_job else conflict_time}",
                     "baseline")

                if result_s is not None:
                    validated_until = conflict_time - 1 if result_s["collision"] \
                                      else conflict_time
                    force_stop = result_s["collision"] and \
                                 (conflict_time - current_timestep) < MIN_LOOKAHEAD
                    push(frames, tester, current_timestep,
                         f"t={current_timestep}  [BASE] "
                         + (f"conflict confirmed@{conflict_time}"
                            + (" FORCED STOP" if force_stop else " — deferring")
                            if result_s["collision"]
                            else f"✓ deconflicted vu={validated_until}"),
                         "baseline")
                    pending_job = None
                    if force_stop:
                        break
                    if not result_s["collision"] and budget.remaining > 0:
                        new_margin = validated_until - current_timestep
                        if new_margin >= MIN_SAFE_HORIZON:
                            push(frames, tester, current_timestep,
                                 f"t={current_timestep}  [BASE→OPT] deconflicted past "
                                 f"MIN_SAFE_HORIZON — handing off to optimizer",
                                 "info")
                            validated_until, pending_job = optimized_step_anim(
                                frames, tester, validated_until, MAX_TIME, budget,
                                MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                                ext_optimizer
                            )
                        else:
                            validated_until, pending_job = try_extend_anim(
                                frames, tester, validated_until, MAX_TIME, budget,
                                MAX_SYMBOLIC_HORIZON, current_timestep, MIN_LOOKAHEAD,
                                origin="baseline",
                                safe_horizon_ceiling=current_timestep + MIN_SAFE_HORIZON
                            )
                else:
                    validated_until = pending_job.symbolic_start
                    push(frames, tester, current_timestep,
                         f"t={current_timestep}  [BASE] budget exhausted@{validated_until}, defer",
                         "baseline")

    if validated_until >= MAX_TIME:
        push(frames, tester, current_timestep, "Done ✓", "info")
        break

    tester.real_state_empirical(current_timestep, current_timestep + 1)
    print(f"Safety margin: {validated_until - current_timestep} | "
          f"[Budget] {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
    current_timestep += 1
    push(frames, tester, current_timestep,
         f"t={current_timestep}  empirical step  vu={validated_until}", "info")

print(f"Total computation time: {tester.get_time():.2f}s")
print(f"Generated {len(frames)} frames")


# ══════════════════════════════════════════════
# Visual scheme
#
# RSOA boxes are rendered with different BORDER styles depending on origin:
#   baseline  → thin dashed border  (same fill colours as before)
#   optimizer → thick solid border + hatching  (visually prominent)
#   info      → no extra decoration (title-only frames)
#
# The fill colour still encodes calc type (concrete/symbolic/empirical)
# so you can read both what was computed AND how it was decided at a glance.
# ══════════════════════════════════════════════

# Per-type fill colours (same palette as original animation files)
RSOA_FILL = {
    "concrete":  {"fc": "#f59e0b", "fa": 0.18, "ec": "#f59e0b"},
    "symbolic":  {"fc": "#8b5cf6", "fa": 0.25, "ec": "#8b5cf6"},
    "sampled":   {"fc": "#06b6d4", "fa": 0.18, "ec": "#06b6d4"},
    "empirical": {"fc": "#10b981", "fa": 0.18, "ec": "#10b981"},
}

# Per-origin border overrides applied on top of fill
ORIGIN_BORDER = {
    "baseline":  {"lw": 0.8,  "ls": "--", "hatch": None,  "alpha_boost": 0.0},
    "optimizer": {"lw": 2.0,  "ls": "-",  "hatch": "//",  "alpha_boost": 0.12},
    "info":      {"lw": 0.8,  "ls": "-",  "hatch": None,  "alpha_boost": 0.0},
}

# Track which timesteps were last touched by which origin so we can border correctly
# We store origin in snap per-timestep by augmenting snapshot with current frame origin
# Instead: we maintain a parallel dict {t: origin} updated each frame

t_origin: dict = {}   # timestep -> last origin that computed it

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.set_xlabel("x₁ (position)", color="#ccc", fontsize=13)
ax.set_ylabel("x₂ (velocity)", color="#ccc", fontsize=13)
title = ax.set_title("", color="#eee", fontsize=11, fontfamily="monospace", pad=12)

# ── Axis limits ──
all_x1, all_x2 = [], []
for snap, _, _, _ in frames:
    for t, entry in snap.items():
        for v in entry["x1"]:
            if np.isfinite(v): all_x1.append(v)
        for v in entry["x2"]:
            if np.isfinite(v): all_x2.append(v)
for obs in obstacles:
    for v in [obs[0, 0], obs[0, 1]]:
        if np.isfinite(v): all_x1.append(v)
    for v in [obs[1, 0], obs[1, 1]]:
        if np.isfinite(v): all_x2.append(v)

margin = 0.3
x1_lo = min(all_x1) - margin;  x1_hi = max(all_x1) + margin
x2_lo = min(all_x2) - margin;  x2_hi = max(all_x2) + margin
ax.set_xlim(x1_lo, x1_hi)
ax.set_ylim(x2_lo, x2_hi)

# ── Static obstacles ──
for i, obs in enumerate(obstacles):
    lx = obs[0, 0] if np.isfinite(obs[0, 0]) else x1_lo
    hx = obs[0, 1] if np.isfinite(obs[0, 1]) else x1_hi
    ly = obs[1, 0] if np.isfinite(obs[1, 0]) else x2_lo
    hy = obs[1, 1] if np.isfinite(obs[1, 1]) else x2_hi
    ax.add_patch(patches.Rectangle(
        (lx, ly), hx - lx, hy - ly,
        linewidth=1.5, edgecolor="#ef4444", facecolor="#ef4444",
        alpha=0.35, zorder=5, label="Obstacle" if i == 0 else None
    ))

dynamic_artists = []


def update(frame_idx):
    global dynamic_artists, t_origin
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    snap, ct, label, origin = frames[frame_idx]
    title.set_text(label)

    # Update t_origin: any timestep present in snap that is newer than what
    # we've seen was (re)computed with the current frame's origin.
    # Heuristic: only tag non-info frames so info frames don't override.
    if origin != "info":
        for t in snap:
            if t not in t_origin:
                t_origin[t] = origin

    for t, entry in sorted(snap.items()):
        fill  = RSOA_FILL.get(entry["type"], RSOA_FILL["concrete"])
        orign = t_origin.get(t, "baseline")
        bdr   = ORIGIN_BORDER[orign]

        x1_lo_e = max(entry["x1"][0], x1_lo) if np.isfinite(entry["x1"][0]) else x1_lo
        x1_hi_e = min(entry["x1"][1], x1_hi) if np.isfinite(entry["x1"][1]) else x1_hi
        x2_lo_e = max(entry["x2"][0], x2_lo) if np.isfinite(entry["x2"][0]) else x2_lo
        x2_hi_e = min(entry["x2"][1], x2_hi) if np.isfinite(entry["x2"][1]) else x2_hi
        if x1_hi_e <= x1_lo_e or x2_hi_e <= x2_lo_e:
            continue

        r = patches.Rectangle(
            (x1_lo_e, x2_lo_e), x1_hi_e - x1_lo_e, x2_hi_e - x2_lo_e,
            linewidth=bdr["lw"],
            linestyle=bdr["ls"],
            edgecolor=fill["ec"],
            facecolor=fill["fc"],
            alpha=fill["fa"] + bdr["alpha_boost"],
            hatch=bdr["hatch"],
            zorder=2
        )
        ax.add_patch(r)
        dynamic_artists.append(r)

        cx  = (x1_lo_e + x1_hi_e) / 2
        cy  = (x2_lo_e + x2_hi_e) / 2
        txt = ax.text(cx, cy, str(t), fontsize=6, color=fill["ec"], alpha=0.85,
                      ha="center", va="center", zorder=3, fontfamily="monospace",
                      fontweight="bold" if orign == "optimizer" else "normal")
        dynamic_artists.append(txt)

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

    # Mode badge — three states: BASE, OPT, BASE→OPT handoff
    if origin == "optimizer":
        badge_color, badge_text = "#22c55e", "● OPT"
    elif label and "BASE→OPT" in label:
        badge_color, badge_text = "#f59e0b", "⇒ BASE→OPT"
    else:
        badge_color, badge_text = "#64748b", "● BASE"
    badge = ax.text(
        0.01, 0.97, badge_text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        color=badge_color, va="top", zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#0e1117", ec=badge_color, lw=1.2)
    )
    dynamic_artists.append(badge)

    return dynamic_artists


# ── Legend ──
legend_els = [
    Patch(facecolor="#f59e0b", alpha=0.3, edgecolor="#f59e0b", label="Concrete RSOA"),
    Patch(facecolor="#8b5cf6", alpha=0.3, edgecolor="#8b5cf6", label="Symbolic RSOA"),
    Patch(facecolor="#10b981", alpha=0.3, edgecolor="#10b981", label="Empirical (Kalman)"),
    Patch(facecolor="#ef4444", alpha=0.4, edgecolor="#ef4444", label="Obstacle"),
    Line2D([0], [0], marker="o", color="#3b82f6", ls="", markersize=6, label="Current t"),
    # Origin indicators
    Patch(facecolor="#888", alpha=0.2, edgecolor="#888",
          linewidth=0.8, linestyle="--", label="Baseline calc"),
    Patch(facecolor="#888", alpha=0.3, edgecolor="#22c55e",
          linewidth=2.0, hatch="//", label="Optimizer calc"),
    Patch(facecolor="#f59e0b", alpha=0.15, edgecolor="#f59e0b",
          linewidth=1.2, label="BASE→OPT handoff"),
]
ax.legend(handles=legend_els, loc="upper right", fontsize=8,
          facecolor="#1a1e28", edgecolor="#333", labelcolor="#aaa")

plt.tight_layout()

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=600, blit=False, repeat=True)

out_path = "alg5_stop_concrete.gif"
ani.save(out_path, writer="pillow", fps=1.2, dpi=130)
print(f"Saved to {out_path}")
plt.close()