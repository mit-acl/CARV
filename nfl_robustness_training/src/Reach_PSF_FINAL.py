"""
Algorithm 21 — PSF with Forward MPC Buffer, D removed as a passthrough gate

Ablation on alg20. D (the danger radius sqrt(r^2 + 2rR)) is KEPT where it
carries the safety argument -- build_mpc_backup still requires the backup's
TERMINAL node to clear D, so the max-turn loiter stays control-invariant and
persistent feasibility is unchanged. What is removed is D used as a
PRECONDITION for passthrough, in the three places alg20 tested it:
  - scan_window no longer requires the candidate tau to be outside D,
  - P1 is no longer skipped when the RSOA at t+1 overlaps D,
  - P3b no longer requires a D-region ahead before scanning past a wall.
Passthrough's licence was never the D geometry in the first place -- it is the
concrete raw-clearance scan over the gap, as the PASSTHROUGH branch states --
so these three gates were restricting the search without backing the guarantee.
collides_danger() survives as diagnostics only.

States:
  NOMINAL      — no conflict found. PSF buffer maintained as 1 step lookahead.
  PRE_CONFLICT — RSOA collides with obstacle. PSF buffer filled towards it. nominal applied
                 while mpc_buffer[t+1] is valid.
  MPC_ACTIVE   — PSF condition failed; executing precomputed backup;
                 revert when mpc_buffer[t+1] becomes valid again.

Changes from alg12:
The MPC intermediate steps can be inside danger area as long as its outside obstacle. Only the terminal rsoa of must be outside.

Budget priorities:
  P1 — Ensure mpc_buffer[t+1] exists (PSF check).
  P2 — Concrete scan forward (feeds MPC buffer with real bounds).
  P3 — Propagate MPC buffer forward toward conflict_time.
  P4 — Symbolic verification (continuous; deconflict or tighten bounds).

Obstacles are in the form: [center_x, center_y, radius]

To call with a specific np seed:
    python alg12_mpc_every_timestep.py <seed_number>
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget, calibrated_costs
import numpy as np
import os
import time
from typing import Optional
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter

RED     = "\033[31m"
GREEN   = "\033[32m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
RESET   = "\033[0m"


# ── alg22: receding-horizon active phase ───────────────────────────────────
# This file is alg21_remove_D_req.py with the MPC ACTIVE phase changed from
# open-loop (fly one committed plan for the whole phase) to receding horizon
# (re-solve from the current bound each timestep, apply u[0]).
# TTTCARV_RECEDING=0 restores the alg21 behaviour exactly, so the two can be
# compared on identical seeds. Everything outside the control-selection step
# is alg21 verbatim.

# Per-trial counters that do not fit test()'s shared return tuple (changing
# that tuple would break every other algN module the driver can load).
# run_trials_random_obs merges this into the fingerprint row after each trial;
# workers are processes running trials sequentially, so it cannot be read
# across trials.
LAST_RUN_STATS: dict = {}

# budget.mpc_cost is calibrated on build_mpc_backup, which runs
# _run_mpc_from_bounds with multi_start=True. The per-timestep re-solve uses a
# single warm start, so gating it on the full mpc_cost would demand more budget
# than it needs and skip the re-solve as "no_budget" on timesteps that could
# easily afford it, which would understate the receding path in an A/B.
#
# CORRECTION (2026-09-11). The 0.25 default was written when
# _run_mpc_multi_start ran FOUR warm starts (NN rollout, omega=+1, -1, 0). It
# no longer does: it runs the NN warm start, returns immediately if that
# already clears D, and otherwise tries ONE bearing-based swerve -- two solves
# max, often one (see _run_mpc_multi_start own docstring). So the calibration
# measures 1-2 solves and a single-start re-solve costs between 1/2 and 1/1 of
# mpc_cost, not 1/4. 0.25 UNDER-reserves: it admits re-solves on timesteps that
# cannot actually afford them, overrunning the 0.20s timestep budget. It errs
# toward doing more work rather than skipping it, which is why no A/B caught
# it. 0.5 is the worst-case-correct value; the env knob exists to measure what
# changing it costs before changing the default.
SINGLE_START_COST_FRAC = float(os.environ.get('TTTCARV_SINGLE_START_FRAC', '0.25'))

# Reserve budget for the active-phase re-solve before the revert-check rebuild
# loop is allowed to spend it (2026-09-11).
#
# The two consumers are not peers. The rebuild loop serves minimal intervention
# -- "can we hand control back to the NN?" -- and losing it costs one extra
# timestep on MPC, recoverable next timestep because the loop runs again. The
# re-solve determines the certificate behind the input applied THIS timestep,
# and losing it is not recoverable: the input goes to the plant either way, it
# just goes uncertified. Letting the first starve the second is a priority
# inversion.
#
# MEASURED: fail_no_budget as a share of active timesteps runs 0.02%-0.92%
# across the 16/32/64-worker load sweep but 15.8% (1050 events) on node4001,
# the most contended node measured. It scales with contention, which is the
# worst possible property for a safety path.
#
# Reserve rather than REORDER. The revert check can `continue` out of the
# timestep before the re-solve is ever reached, so running it first saves a
# whole solve on every timestep where the filter successfully reverts. Hoisting
# the re-solve would burn a solve exactly where it is provably unnecessary.
# MEASURED AND REJECTED (2026-09-11). Default OFF: the reserve is inert by
# construction, because the two consumers never compete. The rebuild loop's
# FIRST build is the revert test at t_next; if it comes back VALID the revert
# fires and `continue`s before the re-solve is reached, and if INFEASIBLE the
# loop breaks after that one build. So every build the reserve could take is
# lookahead on a timestep that reverts, and the budget resets each timestep, so
# it cannot starve the next one either. The purge at the top of the active phase
# then discards lookahead past t+1 on the following active timestep anyway.
#
# Confirmed on 10,000 trials, one node, alternating blocks, paired seeds
# (runs/alg22_reserve2): the reserve yielded 2032 builds and moved active
# timesteps -0.6% / +0.3%, fail_no_budget 5 -> 5, u_diff_n 0.2%. Nothing.
#
# The knob stays so the result can be re-derived, and because the exemption
# would matter if the revert test ever stopped being the loop's first build.
RESOLVE_RESERVE = os.environ.get('TTTCARV_RESOLVE_RESERVE', '0') != '0'

# Terminal safe controller (Algorithm 1 branch 3), used ONLY at the no-backup
# site. When there is no plan to apply, the choice is between the raw NN
# control -- unfiltered, at the moment the filter said it was unsafe -- and a
# max-turn loiter.
#
# What actually certifies it is ONE-STEP MARGIN, not circle invariance. The
# gate requires the current RSOA to clear D = sqrt(r^2 + 2rR) = 1.118; one
# plant step moves at most v*dt = 0.2, so the next state is >= 0.918 from every
# obstacle centre, clear of r = 0.5 whatever the control. The filter then
# re-plans on the next timestep. D's circle-invariance argument holds only when
# the turn is chosen AWAY from the obstacle: with the fixed u=+u_max below,
# ~35% of headings on the D boundary give a left circle that passes through the
# obstacle, so repeated loiter steps are not covered. Measured: at most one
# loiter per trial, psf_unrecoverable = 0, and 0 loiters in 12,500 trials once
# BOOTSTRAP_RESIM is on (job 22638543).
PSF_LOITER = os.environ.get('TTTCARV_PSF_LOITER', '1') != '0'

# Force the reliable (re-simulation) solver for the ONE build that has no
# safety net. Mid-flight, a rejected build is absorbed: t_diverge still points
# at a valid backup banked earlier, psf_valid passes, and the step is covered.
# At t=0 the buffer is structurally empty -- td=never_set -- so a single
# rejection goes straight to the no-backup branch. Every psf_no_diverge event
# on record in a healthy run sits at t=0 for exactly this reason.
#
# TTTCARV_ONESHOT trades convergence for cost: the one-shot path solves the OCP
# once so the terminal-D constraint binds on the endpoint actually returned,
# but when SQP_RTI stops short the plant rollout drifts off the solver's
# iterate and the real endpoint lands inside D (rejected:term_D). ~11% of
# one-shot solves return a non-OK acados status. That is an acceptable price
# where a fallback exists and a bad one where none does.
#
# So: cheap solver everywhere it is safe to be cheap, reliable solver at the
# one site where failure is unrecoverable. Inert when TTTCARV_ONESHOT=0.
BOOTSTRAP_RESIM = os.environ.get('TTTCARV_BOOTSTRAP_RESIM', '1') != '0'

# Deadband for the turn-direction flip probes. Controls near zero have no
# meaningful sign -- a plan drifting through omega=0 is not changing which
# side of an obstacle it passes -- so pairs where either control is inside
# the deadband are not counted as a pair at all, rather than as non-flips.
FLIP_DEADBAND = 0.05



# ── Shared helpers ─────────────────────────────────────────────────────────


def _get_nn_control(tester, timestep):
    """Query NN nominal control at the real state stored in horizons[timestep]."""
    h = tester.horizons.get(timestep)
    if h is None:
        return None
    calc = next((c for c in h.calculations.values() if 'real_state' in c), None)
    if calc is None:
        return None
    state  = np.asarray(calc['real_state']).reshape(1, -1)
    cl_sys = tester.analyzer.cl_system
    return np.asarray(cl_sys.dynamics.control_nn(state, cl_sys.controller.cpu())).flatten()


class RefinementTask:
    """Represents an in-progress symbolic verification toward conflict_time."""
    def __init__(self, symbolic_start: int, conflict_time: int):
        self.symbolic_start = symbolic_start
        self.conflict_time  = conflict_time
        self.time_invested  = 0

    def done(self):
        return self.symbolic_start >= self.conflict_time


# Receives seconds per concrete step after every propagation; set by test()
# under TTTCARV_MPC_PRICING so concrete admission is priced on measured cost.
_CONCRETE_OBSERVER = None
# (time_left, step_price) callables for the per-step concrete deadline; set by
# test() under TTTCARV_MPC_PRICING and used only by deadline scans (P2).
_CONCRETE_DEADLINE = None

def concrete_scan(tester, from_t, to_t):
    """Run concrete propagation; return (collision: bool, collision_t: int|None)."""
    collision, ct, _ = concrete_scan_deadline(tester, from_t, to_t, deadline=False)
    return collision, ct


def concrete_scan_deadline(tester, from_t, to_t, deadline=True):
    """Run concrete propagation, stopping between steps at the budget deadline.

    Returns (collision, collision_t, reached): reached is the last timestep with
    concrete bounds. Without a deadline (or with pricing off) it runs to to_t.
    """
    if from_t >= to_t:
        return False, None, from_t
    kw = {}
    if deadline and _CONCRETE_DEADLINE is not None:
        time_left, step_price = _CONCRETE_DEADLINE
        kw = dict(time_left=time_left, step_price=step_price())
    t0 = time.perf_counter()
    result = tester.concrete(from_t, to_t, **kw)
    wall = time.perf_counter() - t0
    if not result:
        return False, None, from_t
    done = result.get('completed_steps', to_t - from_t)
    # Price on the wall time of the whole call over the steps actually run: it
    # includes per-step overhead, and a collision stop runs fewer steps.
    if _CONCRETE_OBSERVER is not None and done > 0:
        _CONCRETE_OBSERVER(wall / done)
    if not result["collision"]:
        return False, None, from_t + done
    return True, result["collision_timestep"], result["collision_timestep"]


def symbolic_step(tester, job: RefinementTask, chunk_size: int, budget):
    """
    Advance job by one chunk within budget.
    Returns (updated_job, result_dict|None).
    result_dict is non-None only when the job completes this call.
    """
    steps = min(chunk_size, job.conflict_time - job.symbolic_start)
    cost  = budget.symbolic_costs.get(steps, float('inf'))
    if cost - job.time_invested <= budget.remaining:
        verify_end         = min(job.conflict_time, job.symbolic_start + chunk_size)
        result             = tester.symbolic(job.symbolic_start, verify_end)
        job.symbolic_start = verify_end
        job.time_invested  = 0
        if job.done():
            return job, result
        return job, None
    else:
        job.time_invested += budget.remaining
        return job, None


def extend_mpc_sequence(mpc_sf, mpc_state: dict, min_safe_horizon: int,
                        max_time: int, ctrl_idx: int = 0, budget=None):
    """
    Extend the MPC control queue from the end of the current trajectory.
    Falls back along the existing trajectory if extension from endpoint fails.
    Same as alg11.
    """
    committed_at = mpc_state['committed_at']
    controls     = mpc_state['controls']
    traj_bounds  = mpc_state['traj_bounds']
    n_controls   = len(controls)
    mpc_end      = committed_at + n_controls

    if mpc_end >= max_time:
        print(f"  [MPC extend] Reached MAX_TIME, done")
        return

    max_lb = max(min(n_controls - ctrl_idx - 1, mpc_sf.n_horizon), 0)

    for lb in range(max_lb + 1):
        if budget is not None and budget.remaining < budget.mpc_cost:
            print(f"  [MPC extend] Budget exhausted, will retry next timestep")
            return

        try_idx = len(traj_bounds) - 1 - lb
        if try_idx < 1:
            break

        try_bounds = traj_bounds[try_idx]
        center     = (try_bounds[:, 0] + try_bounds[:, 1]) / 2.0

        try:
            _t0 = time.time()
            new_traj_bounds, _, new_controls = mpc_sf._run_mpc_from_bounds(
                try_bounds, center, extra_inflation=0.0)
            print(f"  [MPC extend] from t={committed_at}  lookback={lb}  took {time.time()-_t0:.3f}s")
        except Exception as e:
            print(f"  [MPC extend] Failed at lookback={lb}: {e}")
            continue

        # Split-terminal: the extension continues the committed backup PATH, so
        # intermediate nodes need only clear the RAW radius (they may enter D).
        # But the appended segment must TERMINATE outside D (the danger set),
        # or persistent feasibility breaks — Lemma 2's induction needs every
        # committed plan to end where the loiter is control-invariant. So:
        #   (1) raw-collision-free prefix length  → how far the path stays ∉ O,
        #   (2) within that prefix, the last node that is also outside D
        #       → the only admissible terminal for the appended segment.
        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides_raw(b):
                break
            safe_count += 1

        # Latest index in 1..safe_count whose bound clears D (valid terminal).
        term_count = 0
        for k in range(1, safe_count + 1):
            if not mpc_sf._collides_D(new_traj_bounds[k]):
                term_count = k

        if term_count <= lb:
            print(f"  [MPC extend] lookback={lb}: raw-safe={safe_count} but no "
                  f"D-terminal beyond lb={lb}, going further back")
            continue

        keep = n_controls - lb
        mpc_state['controls']    = list(controls[:keep]) + list(new_controls[:term_count])
        mpc_state['traj_bounds'] = (list(traj_bounds[:try_idx + 1])
                                    + list(new_traj_bounds[1:term_count + 1]))

        net_gain  = term_count - lb
        new_total = len(mpc_state['controls'])
        new_end   = committed_at + new_total
        if lb > 0:
            print(f"  [MPC extend] lookback={lb}: +{term_count} (D-terminal), "
                  f"-{lb} replaced, net +{net_gain}")
        print(f"  [MPC extend] +{net_gain} controls → {new_total} total "
              f"(t={committed_at} to t={new_end}, terminal ∉ D)")
        return

    print(f"  [MPC extend] No safe extension found after {max_lb + 1} lookback attempts")


# PSF helpers

def psf_valid(mpc_buffer: dict, t: int):
    """ Returns true if mpc_buffer at time t exists and is collision free."""
    return mpc_buffer.get(t) is not None


def _rollout_verdict(mpc_sf, traj_bounds):
    """
    The acceptance test for an MPC rollout, in one place, reporting WHICH check
    failed and WHERE. Returns (ok, kind, node, n_nodes).

      kind  None | 'empty' | 'path_raw' | 'term_D'
      node  1-based index of the offending PATH node; n_nodes for the terminal

    Split-terminal semantics: every PATH node (stages 1..N-1) must clear the
    raw obstacle radius (the plan may pass THROUGH the danger region D), and
    the TERMINAL node must clear D so the max-turn loiter is control-invariant
    from there.

    build_mpc_backup and solve_plan_at both go through this, so a plan the
    receding controller applies and a backup the PSF certifies mean the same
    thing. If they were allowed to drift, an A/B against alg21 would be
    measuring the difference between two acceptance tests, not the control law.
    """
    if traj_bounds is None or len(traj_bounds) < 2:
        return False, 'empty', -1, 0
    *path_bounds, terminal_bound = traj_bounds[1:]
    n_nodes = len(path_bounds) + 1
    for i, b in enumerate(path_bounds, start=1):
        if mpc_sf._collides_raw(b):
            return False, 'path_raw', i, n_nodes
    if mpc_sf._collides_D(terminal_bound):
        return False, 'term_D', n_nodes, n_nodes
    return True, None, -1, n_nodes


def _rollout_ok(mpc_sf, traj_bounds) -> bool:
    """Bare verdict, for callers that only need pass/fail."""
    return _rollout_verdict(mpc_sf, traj_bounds)[0]


def _bounds_contain(outer, inner, tol: float = 1e-9) -> bool:
    """True if box `inner` lies inside box `outer`."""
    if outer is None or inner is None:
        return False
    o = np.asarray(outer, dtype=float)
    i = np.asarray(inner, dtype=float)
    if o.shape != i.shape:
        return False
    return bool(np.all(i[:, 0] >= o[:, 0] - tol)
                and np.all(i[:, 1] <= o[:, 1] + tol))


def solve_plan_at(mpc_sf, tester, tau: int, budget=None, multi_start: bool = False):
    """
    Solve an MPC plan from the bounds at tau and return it WITHOUT touching
    mpc_buffer. Returns (ok, controls, traj_bounds, label).

    Deliberately kept out of mpc_buffer: that dict is the PSF's record of where
    the robot may diverge from nominal, and psf_valid() is a presence test.
    Writing a plan for the current timestep into it would read as a revert
    certificate and change the PSF's decisions, which is not what this is for.

    multi_start defaults False because this runs on every active timestep,
    where build_mpc_backup runs once per tau; the NN warm start alone is a
    quarter of the cost. The caller retries with multi_start=True before
    falling back to a plan built from a stale bound.
    """
    if budget is not None:
        need = budget.mpc_cost * (1.0 if (multi_start or budget.pricing)
                                  else SINGLE_START_COST_FRAC)
        if budget.remaining < need:
            return False, None, None, "no_budget"
    bounds = _bounds_at(tester, tau)
    if bounds is None:
        return False, None, None, "no_bounds"
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
            bounds, center, multi_start=multi_start)
    except Exception as e:
        return False, None, None, f"solver_error({e})"
    if controls is None or len(controls) == 0:
        return False, None, None, "empty_plan"
    ok, kind, node, n_nodes = _rollout_verdict(mpc_sf, traj_bounds)
    if not ok:
        # Coarse bucket before "(", detail inside — so the existing
        # split('(')[0] reason counter keeps working unchanged.
        return False, None, None, f"rollout_rejected({kind}@{node}/{n_nodes})"
    return True, list(controls), list(traj_bounds), "VALID"

def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int, reasons: dict = None):
    """
    Solve MPC backup starting from concrete bounds at timestep tau.
    Stores (controls, traj_bounds) in mpc_buffer[tau] if collision-free,
    else None. Returns True if a valid backup was found.

    reasons: optional dict[tau -> str]. When a build is rejected the diagnostic
    label ('path_raw[...]' / 'term_D ...') is recorded there so callers can
    report WHY there is no backup instead of guessing. Kept out of mpc_buffer
    on purpose: psf_valid() tests `is not None`, so a non-None reason stored in
    the buffer would read as a valid backup.

    VALID (split-terminal semantics):
      - every PATH node (stages 1..N-1) must clear the raw obstacle radius r
        (the backup may pass THROUGH the danger region D), and
      - the TERMINAL node (stage N) must clear D, so the max-turn loiter is
        control-invariant from there → persistent feasibility.
    """
    h = tester.horizons.get(tau)
    if h is None:
        if reasons is not None:
            reasons[tau] = "no_bounds(concrete RSOA not propagated to tau)"
        return False
    bounds = h.get_tight_bound()
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        _t0 = time.time()
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(
            bounds, center, multi_start=True)
        elapsed = time.time() - _t0
        *path_bounds, terminal_bound = traj_bounds[1:]

        # Diagnostic: identify which check fired and the offending bound.
        # path_raw — some intermediate node hit the raw obstacle radius
        # term_D   — terminal node hit the danger inflation D
        path_hit = next(
            ((i, b) for i, b in enumerate(path_bounds, start=1)
             if mpc_sf._collides_raw(b)),
            None
        )
        term_hit = mpc_sf._collides_D(terminal_bound)
        # path_hit / term_hit stay for the diagnostic label below; the verdict
        # itself comes from the shared test so it cannot drift from solve_plan_at.
        collision = not _rollout_ok(mpc_sf, traj_bounds)

        if collision:
            if path_hit is not None:
                idx, b = path_hit
                cx_arr = (b[0, 0] + b[0, 1]) / 2.0
                cy_arr = (b[1, 0] + b[1, 1]) / 2.0
                # closest obstacle (raw center) + box-to-center distance
                best = min(
                    mpc_sf.raw_obstacles,
                    key=lambda o: (o[0] - cx_arr) ** 2 + (o[1] - cy_arr) ** 2
                )
                dx = best[0] - np.clip(best[0], b[0, 0], b[0, 1])
                dy = best[1] - np.clip(best[1], b[1, 0], b[1, 1])
                box_dist = float(np.sqrt(dx * dx + dy * dy))
                label = (f"path_raw[node={idx}/{len(path_bounds)}]"
                         f" obs=({best[0]:.2f},{best[1]:.2f},r={best[2]:.2f})"
                         f" box_dist={box_dist:.3f}")
            else:
                b = terminal_bound
                cx_arr = (b[0, 0] + b[0, 1]) / 2.0
                cy_arr = (b[1, 0] + b[1, 1]) / 2.0
                # closest obstacle, and required D radius vs actual distance
                pairs = []
                for o_raw, o_D in zip(mpc_sf.raw_obstacles, mpc_sf.safety_obstacles):
                    dx = o_raw[0] - np.clip(o_raw[0], b[0, 0], b[0, 1])
                    dy = o_raw[1] - np.clip(o_raw[1], b[1, 0], b[1, 1])
                    pairs.append((float(np.sqrt(dx * dx + dy * dy)), o_raw, o_D[2]))
                pairs.sort(key=lambda p: p[0])
                box_dist, best, D_req = pairs[0]
                term_xyz = [(b[0, 0] + b[0, 1]) / 2.0,
                            (b[1, 0] + b[1, 1]) / 2.0]
                label = (f"term_D obs=({best[0]:.2f},{best[1]:.2f},r={best[2]:.2f})"
                         f" D_req={D_req:.3f}  box_dist={box_dist:.3f}"
                         f"  term_center=({term_xyz[0]:.2f},{term_xyz[1]:.2f})")
            status = f"INFEASIBLE({label})"
        else:
            status = "VALID"
        print(f"  [MPC build] τ={tau}  {status}  took {elapsed:.3f}s")
        if not collision:
            mpc_buffer[tau] = (list(controls), list(traj_bounds))
            if reasons is not None:
                reasons.pop(tau, None)
            return True
        mpc_buffer[tau] = None
        if reasons is not None:
            reasons[tau] = label
        return False
    except Exception as e:
        print(f"  [PSF buf] MPC solve failed at τ={tau}: {e}")
        mpc_buffer[tau] = None
        if reasons is not None:
            reasons[tau] = f"solver_error({e})"
        return False

#  Helpers

def _box_circle_overlap(bounds: np.ndarray, cx: float, cy: float, r: float) -> bool:
    dx = cx - np.clip(cx, bounds[0, 0], bounds[0, 1]) #center of circle to bound distance
    dy = cy - np.clip(cy, bounds[1, 0], bounds[1, 1])
    return dx * dx + dy * dy <= r * r


def _bounds_at(tester, t):
    """Return tight bound at timestep t, or None if missing."""
    h = tester.horizons.get(t)
    return h.get_tight_bound() if h is not None else None


def collides_danger(bounds: np.ndarray, obstacles: list,
                    R: float = 1.0) -> bool:
    """Overlap against the danger / uncertainty region D = sqrt(r^2 + 2rR).
    D is the inflated obstacle radius where MPC can't guarantee avoidance;
    outside D is the invariant safe region. r = obstacle radius,
    R = curvature radius. R = 1 for current dynamics.
    """
    return any(_box_circle_overlap(bounds, obs[0], obs[1],
                                   np.sqrt(obs[2]**2 + 2 * obs[2] * R))
               for obs in obstacles)


def collides_raw(bounds: np.ndarray, obstacles: list) -> bool:
    """Overlap against raw obstacle radius."""
    return any(_box_circle_overlap(bounds, obs[0], obs[1], obs[2])
               for obs in obstacles)


def scan_window(tester, mpc_sf, mpc_buffer: dict, obstacles: list,
                     wall_tau: int, scan_limit: int, budget,
                     R: float = 1.0, reasons: dict = None) -> Optional[int]:
    """
    Scan beyond an MPC INFEASIBLE "wall" for a tau with a valid MPC backup.
    Returns that tau as the new t_diverge, else None.

    alg21: D-membership is NO LONGER a precondition. alg20 attempted a build at
    tau only when the RSOA there was outside D, on the theory that a build from
    inside D must fail. But D constrains the MPC's TERMINAL node, not its launch
    state: a backup launched from inside D can still route out and terminate
    outside it. Gating on D therefore skipped taus that were in fact feasible,
    and dropping the gate costs at most a build attempt that build_mpc_backup
    would reject anyway -- budget, not soundness. The two real preconditions are
    unchanged and still enforced:
      - raw-clearance at tau (checked below; a tau already in collision is not a
        place to fly to), and
      - build_mpc_backup returning VALID, which still requires the terminal node
        to clear D.

    R is retained in the signature for call-site compatibility; it no longer
    affects the scan.

    scan_limit is inclusive: the earlier of the last pre-collision step and the last step with bounds.
    """
    for tau in range(wall_tau + 1, scan_limit + 1):
        if tau not in tester.horizons:
            break
        bounds = tester.horizons[tau].get_tight_bound()
        if bounds is None:
            break
        if collides_raw(bounds, obstacles):
            print(f"  [Scan window] tau={tau} collides with raw obstacle — abort")
            return None
        if tau in mpc_buffer:
            if mpc_buffer[tau] is not None:
                print(f"  [Scan window] tau={tau} already VALID in buffer")
                return tau
            continue
        if budget.remaining < budget.mpc_cost:
            print(f"  [Scan window] Budget exhausted at tau={tau}")
            return None
        valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau, reasons)
        if valid:
            print(f"  [Scan window] tau={tau} VALID — passthrough t_diverge found")
            return tau
        else:
            print(f"  [Scan window] tau={tau} raw-clear but MPC INFEASIBLE — continue")
    return None

def _purge_infeasible(mpc_buffer: dict, after_t: int):
    """Delete INFEASIBLE (None) entries with key > after_t."""
    for k in list(mpc_buffer):
        if k > after_t and mpc_buffer[k] is None:
            del mpc_buffer[k]

def _max_valid_diverge(mpc_buffer: dict, from_t: int,
                       below: Optional[int] = None) -> Optional[int]:
    """Return the largest key >= from_t with a valid (non-None) buffer entry.

    below: exclusive ceiling, normally conflict_time. concrete_scan only
    certifies raw-clearance for t < conflict_time, so a t_diverge at or beyond
    it would let PASSTHROUGH fly nominal across a step already known to
    collide. Entries above the ceiling are usually stale: built before the
    conflict was discovered and never re-examined.
    """
    keys = [k for k, v in mpc_buffer.items()
            if k >= from_t and v is not None
            and (below is None or k < below)]
    return max(keys) if keys else None


# ── Simulation loop ────────────────────────────────────────────────────────

def test(seed=None, analyzer=None, obstacles=None):
    # Free the previous trial's objects before this trial builds its acados
    # solver. The solver is generated and dlopen'ed under a per-pid name, so a
    # reference cycle that keeps the old AcadosOcpSolver alive (the budget
    # ledger's charged() wrappers create one) makes this trial load the stale
    # library with the previous trial's obstacles baked in. Confirmed
    # 2026-09-13 in runs/alg22_ledger_bisect; gc here restored exact parity.
    if os.environ.get('TTTCARV_GC_PER_TRIAL', '1') != '0':
        import gc
        gc.collect()
    if analyzer is None:
        analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    if seed is None:
        seed = 1401830092

    if obstacles is None:
        obstacles = [
            np.array([-4.3,  1,   0.5]),
            np.array([-2,   -2,   0.5]),
            np.array([-1,    0.5, 0.5]),
            np.array([-3.5,  0,   0.5]),
        ]

    tester = ReachabilityTester(analyzer, obstacles, seed=seed)
    # TTTCARV_FIRST_STEP_COST=1 charges the min-intervention cost only at
    # stage 0 instead of over the whole horizon. Independent of TTTCARV_RECEDING
    # so the two can be separated in an A/B, though it is the receding loop that
    # makes it principled: only u[0] is ever applied, so only u[0] needs to stay
    # close to nominal, and the freed tail gives the solver more room to find a
    # safe continuation.
    first_step_cost = os.environ.get('TTTCARV_FIRST_STEP_COST', '0') != '0'
    # Tunable because the first measured run at 1e-3 showed 3/5 re-solves
    # failing with SQP_RTI QP errors on a seed that had 0 failures without the
    # flag — the near-singular tail Hessian the constant is there to prevent.
    # Sweep it rather than guess.
    tail_cost_weight = float(os.environ.get('TTTCARV_TAIL_COST_WEIGHT', '1e-3'))
    # TTTCARV_ONESHOT=1 makes each backup ONE OCP solve whose own predicted
    # trajectory is returned, instead of n_horizon solves re-simulated forward
    # one step at a time. The re-simulation never lets any solve's hard
    # terminal-D constraint land on the endpoint it returns, which is what
    # produced the terminal-D rejections; the one-shot plan is constrained by
    # construction. It is also ~n_horizon times cheaper. Combined with
    # TTTCARV_RECEDING=1 this is the standard PSF: solve once per timestep,
    # apply u[0], re-solve.
    oneshot        = os.environ.get('TTTCARV_ONESHOT', '0') != '0'
    oneshot_refine = int(os.environ.get('TTTCARV_ONESHOT_REFINE', '2'))
    # Measures |opt_x_num - plant re-simulation| so the model-fidelity claim
    # the one-shot path rests on is checked rather than assumed. Costs one
    # extra plant rollout per solve; leave off for timing runs.
    oneshot_verify = os.environ.get('TTTCARV_ONESHOT_VERIFY', '0') != '0'
    # SQP_RTI iterations per one-shot refine pass (default 20 = unchanged).
    oneshot_rti    = int(os.environ.get('TTTCARV_ONESHOT_RTI', '20'))
    mpc_sf = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=12,
                                    verify_input_bounds=True,
                                    verify_heading_bounds=True,
                                    use_safety_radius=True, split_terminal_D=True,
                                    first_step_cost=first_step_cost,
                                    tail_cost_weight=tail_cost_weight,
                                    oneshot=oneshot,
                                    oneshot_refine=oneshot_refine,
                                    oneshot_verify=oneshot_verify,
                                    oneshot_rti=oneshot_rti)

    v_nom = tester.analyzer.cl_system.dynamics.vt
    u_max = 1.0
    turning_radius = v_nom / u_max

    MAX_SYMBOLIC_HORIZON = 5
    MAX_TIME             = 60

    budget = TimeBudget(timestep_budget=0.20)
    sym_costs, concrete_cost, mpc_cost = calibrated_costs(
        lambda: ReachabilityTester(analyzer, obstacles, seed=0),
        lambda t: (lambda: build_mpc_backup(mpc_sf, t, {}, 1)),
        MAX_SYMBOLIC_HORIZON,
    )
    budget.symbolic_costs = dict(sym_costs)
    budget.concrete_cost  = concrete_cost
    budget.mpc_cost       = mpc_cost

    # ── Calibration sanity guard (2026-09-11) ─────────────────────────────
    # mpc_cost is measured once per process, during the startup stampede when
    # 64 workers initialise acados and TF at the same instant. Measured spread
    # on node5000: healthy median 0.066s, max 0.174s, and one worker 0.295s.
    # An estimate above the WHOLE timestep budget is not a steady-state cost,
    # it is a startup transient — but the gate `remaining >= mpc_cost` reads it
    # as "an MPC build can never be afforded" and is then false forever, since
    # remaining starts at timestep_budget. That silently disables the safety
    # filter for the entire life of the process.
    #
    # Clamping is the conservative direction here, which is not the obvious
    # one. An optimistic mpc_cost risks overrunning a single timestep — bounded,
    # and visible in wall_per_timestep_s. A pessimistic one disables safety
    # outright and is invisible. So prefer the bounded failure.
    #
    # THRESHOLD CORRECTED 2026-09-12. Testing `mpc_cost >= timestep_budget` is
    # too loose, and the old clamp TARGET (0.9 * budget = 0.18s) fails its own
    # test. P1 must first propagate concrete bounds to t+1 -- it cannot build a
    # backup without them -- so by the time it checks `remaining >= mpc_cost`,
    # concrete_cost is already spent. The real requirement is
    #     mpc_cost + concrete_cost <= timestep_budget.
    # Measured case (runs/alg22_wdbg2, worker 1783205): mpc=0.1782s,
    # concrete=0.0235s. 0.1782 < 0.20 so the old guard passed it, then P1 spent
    # 0.0259s on bounds and tested 0.174 >= 0.1782 -> false. That worker could
    # never build at t=0 and produced 7 of the 11 psf_no_diverge events in its
    # run. 0.18 + 0.0235 = 0.2035 > 0.20, so the old clamp could put a worker
    # INTO this state while claiming to rescue it.
    mpc_cost_clamped = 0
    _headroom = budget.timestep_budget - concrete_cost
    if mpc_cost >= _headroom:
        clamped = 0.9 * _headroom
        print(f"[calibrate] WARNING mpc_cost={mpc_cost:.4f}s >= timestep_budget-"
              f"concrete_cost={_headroom:.4f}s — P1 spends concrete_cost="
              f"{concrete_cost:.4f}s on bounds before it tests the gate, so the "
              f"build at t+1 could never be afforded and the filter would be OFF "
              f"for this whole process. Clamping to {clamped:.4f}s.")
        budget.mpc_cost  = clamped
        mpc_cost_clamped = 1

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    # ── PSF buffer state ───────────────────────────────────────────────────
    mpc_buffer:        dict                      = {}   # tau -> (controls, traj_bounds) | None
    mpc_reason:        dict                      = {}   # tau -> why the build at tau was rejected
    concrete_until:    int                       = 0
    mpc_horizon_until: int                       = -1   # last tau MPC was attempted
    conflict_time:     Optional[int]             = None
    t_diverge:         Optional[int]             = None
    pending_job:       Optional[RefinementTask] = None
    wall_tau:          Optional[int]             = None   # first INFEASIBLE tau hit in P3; reset each iteration

    # ── MPC execution state (mirrors alg11) ───────────────────────────────
    mpc_state = {
        'committed_at':  None,
        'conflict_time': None,
        'controls':      [],
        'traj_bounds':   [],
        'needed':        False,
    }
    mpc_started      = False

    current_timestep = 0
    u_diffs          = []   # |u_mpc - u_nn| per timestep; 0.0 when nominal
    mpc_calls        = 0    # number of PSF activations
    psf_no_diverge   = 0    # PSF failed, no valid t_diverge at all
    # WHY there was no backup. The counter alone cannot distinguish "the build
    # was never attempted because the budget was spent" from "the build ran and
    # the acceptance test rejected it" -- and those need opposite fixes. The
    # printed diagnostic goes to the worker's stdout, which is /dev/null unless
    # TTTCARV_WORKER_LOG_DIR is set, so on the alg22 10k the reason for the one
    # event was lost. Recorded in the fingerprint from here on.
    psf_no_diverge_reasons: dict = {}   # short key -> count
    # P1's budget gate is the only path that leaves mpc_buffer[t+1] *absent*
    # with no reason recorded, which is exactly what every observed
    # psf_no_diverge diagnostic shows (buf[t+1]=absent, td=never_set).
    # It was uncounted: mpc_skipped_no_budget counts the step-2 rebuild loop,
    # not this. Counted separately so the two are never conflated again.
    p1_skipped_no_budget = 0   # P1 build refused by the budget gate
    concrete_deadline_hits = 0  # P2 scans stopped between steps at the deadline
    p1_skipped_no_bounds = 0   # P1 build impossible, concrete bounds missing
    psf_loiter_used      = 0   # no-backup step covered by the max-turn loiter
    p1_bootstrap_resim   = 0   # no-fallback build forced onto the re-sim solver
    p1_bootstrap_infeasible = 0  # no-fallback build that still came back INFEASIBLE
    p1_t0_infeasible     = 0   # the t=0 build specifically -- the one with an empty buffer
    psf_unrecoverable    = 0   # no backup AND current RSOA already inside D
    psf_queue_empty  = 0    # PSF failed, t_diverge valid but queue empty after extend
    psf_stale_refused = 0   # PSF failed, only a stale (ctrl_idx>0) plan existed
    passthrough_found = 0   # P3b scan_window returned a raw-clear τ with a valid backup
    passthrough_used  = 0   # PSF PASSTHROUGH branch executed (nominal, no backup at t+1)
    passthrough_found = 0   # P3b scan_window returned a τ outside D after a wall
    passthrough_used  = 0   # PSF PASSTHROUGH branch executed (nominal while in D)

    # ── Receding-horizon active phase (alg22) ─────────────────────────────
    # TTTCARV_RECEDING=0 restores alg21's open-loop active phase, which makes
    # this file a behavioural no-op against alg21 — the regression check that
    # the copy did not drift anywhere outside the control-selection step.
    receding = os.environ.get('TTTCARV_RECEDING', '1') != '0'
    # TTTCARV_PSF_TAIL=1 makes the failure path branch 2 of Algorithm 1 in the
    # PSF literature. A plan committed at k_bar certifies exactly n_horizon
    # controls, and the terminal constraint is what makes its shifted tail a
    # feasible solution to the shrunk problem — so on a failed re-solve the
    # stored u*[k - k_bar] needs no new solve to be safe, but ONLY while
    # k - k_bar < N. alg21 instead calls extend_mpc_sequence and rides on
    # indefinitely, which manufactures plan past the horizon the terminal
    # constraint ever covered, from a predicted rather than a measured bound.
    # Off by default so the receding arm stays byte-comparable to its last run.
    psf_tail = os.environ.get('TTTCARV_PSF_TAIL', '0') != '0'
    mpc_resolved             = 0  # active timesteps whose input came from a fresh solve
    mpc_resolve_failed       = 0  # active timesteps that fell back to the committed plan
    mpc_fallback_uncertified = 0  # fallbacks where the current RSOA left traj_bounds[idx]
    mpc_tail_exhausted       = 0  # k - k_bar reached N: certificate spent, branch 3 due
    # Why a re-solve failed. "the solver failed" and "the solver returned a plan
    # that violates the terminal constraint" are different problems with
    # different fixes, and mpc_resolve_failed alone cannot tell them apart.
    mpc_fail_reason: dict = {}
    # Which acceptance check rejected the plan, and at which node. Separates
    # "the plan clipped an obstacle early" from "the plan never got out of D by
    # the end of the horizon" — different problems with different fixes.
    mpc_reject_detail: dict = {}
    # Homotopy-jump probe. The per-step re-solve warm-starts from a fresh NN
    # rollout (warm_start_omega=None), not from the previous solution, so
    # nothing structurally carries the previous step left/right choice
    # forward. If the solver is flipping between the two ways around an
    # obstacle, the applied turn direction alternates. Two views:
    #   applied_u_* -- sign of the input actually applied, on consecutive
    #                  active timesteps. This is what the robot does.
    #   plan_flips  -- sign of the mean of a fresh plan vs the mean of the
    #                  tail of the plan it replaces. This is what the solver
    #                  decided, and it fires even when u[0] happens to agree.
    # *_pairs is the denominator: comparisons where both sides cleared the
    # deadband. A flip RATE needs it; a flip count alone is uninterpretable.
    applied_u_flips    = 0
    applied_u_pairs    = 0
    plan_flips         = 0
    plan_pairs         = 0
    _prev_applied_u    = None
    _prev_applied_t    = None
    # Times the backup-build loop stopped because the timestep budget could not
    # cover one more MPC build. Uncounted until 2026-09-11, and that is the whole
    # reason the dead-worker mode survived five sessions: mpc_cost is calibrated
    # ONCE per process, so when that single measurement lands above
    # timestep_budget the gate is false on every timestep of every trial the
    # worker ever runs. The filter never builds a backup, the process flies
    # unfiltered nominal, and the results look ordinary. Nothing incremented,
    # nothing printed, nothing in the fingerprint.
    mpc_skipped_no_budget = 0
    # Subset of the above: builds skipped purely to honour resolve_reserve.
    mpc_build_yielded_to_resolve = 0
    print(f"[alg22] receding-horizon active phase: {'ON' if receding else 'OFF (alg21 parity)'}"
          f"   first-step-only cost: {'ON' if first_step_cost else 'OFF'}"
          f"{f' (tail_w={tail_cost_weight:g})' if first_step_cost else ''}"
          f"   bounded tail (Alg1 branch 2): {'ON' if psf_tail else 'OFF'}")

    # ── RSOA certificate dump (off unless TTTCARV_RSOA_DUMP names a path) ──
    # Every executed timestep leaves through real_state_empirical or
    # real_state_mpc, so wrapping those two is the only placement that cannot
    # miss a branch — the six call sites in the loop are covered by
    # construction rather than by my having found all of them. The record is
    # taken *before* the step so it holds the bounds as they stood when the
    # step was authorized, not as they were later re-propagated.
    _rsoa_path = os.environ.get('TTTCARV_RSOA_DUMP')
    rsoa_log   = []
    if _rsoa_path:
        def _bnd(t):
            h = tester.horizons.get(t)
            if h is None:
                return None
            b = h.get_tight_bound()
            return None if b is None else np.asarray(b).tolist()

        def _real(t):
            h = tester.horizons.get(t)
            if h is None:
                return None
            for c in h.calculations.values():
                if 'real_state' in c:
                    return np.asarray(c['real_state']).flatten().tolist()
            return None

        def _wrap(fn, kind):
            def inner(t, arg):
                rec = {'t': t, 'kind': kind,
                       'conflict_time': conflict_time, 't_diverge': t_diverge,
                       'mpc_started': mpc_started,
                       'committed_at': mpc_state['committed_at'],
                       'bound_t': _bnd(t), 'real_t': _real(t)}
                if kind == 'mpc' and mpc_state['committed_at'] is not None:
                    # traj_bounds[k] is the backup's bound at committed_at + k.
                    # extend_mpc_sequence splices this list, so the index can go
                    # stale; the checker reports a missing entry rather than
                    # assuming one.
                    tb, i = mpc_state['traj_bounds'], t - mpc_state['committed_at']
                    rec['backup_idx']      = i
                    rec['backup_len']      = len(tb)
                    rec['backup_bound_t']  = (np.asarray(tb[i]).tolist()
                                              if 0 <= i < len(tb) else None)
                    rec['backup_bound_t1'] = (np.asarray(tb[i + 1]).tolist()
                                              if 0 <= i + 1 < len(tb) else None)
                out = fn(t, arg)
                rec['bound_t1'] = _bnd(t + 1)
                rec['real_t1']  = _real(t + 1)
                rsoa_log.append(rec)
                return out
            return inner

        tester.real_state_empirical = _wrap(tester.real_state_empirical, 'empirical')
        tester.real_state_mpc       = _wrap(tester.real_state_mpc,       'mpc')

    # ── Per-timestep budget ledger (opt-in via TTTCARV_BUDGET_DUMP) ──────
    # Charges the four kinds of work the budget actually prices, so a timestep
    # can be scored both as-is and with the deferrable symbolic refinement
    # removed. Wrapping the bound methods catches every call site; the residual
    # (elapsed minus the charges) is Python/geometry overhead.
    _budget_path = os.environ.get('TTTCARV_BUDGET_DUMP')
    if _budget_path:
        budget.ledger_enabled       = True
        tester.symbolic             = budget.charged(tester.symbolic,   'symbolic')
        tester.concrete             = budget.charged(tester.concrete,   'concrete')
        tester.real_state_empirical = budget.charged(tester.real_state_empirical, 'step')
        tester.real_state_mpc       = budget.charged(tester.real_state_mpc,       'step')
        mpc_sf._run_mpc_from_bounds = budget.charged(
            mpc_sf._run_mpc_from_bounds, 'mpc')

    # ── Per-solve MPC pricing + deadline (opt-in, TTTCARV_MPC_PRICING=1) ──
    # Off: MPC admission tests remaining >= mpc_cost, one calibration probe's
    # median, and the single-start re-solve pays SINGLE_START_FRAC of it. On:
    # mpc_cost reads as the p99.5 of this process's measured single-solve times
    # plus TTTCARV_BUDGET_MARGIN; every solve, including the second multi-start
    # one, must fit before it starts; and the RTI loop / re-sim rollout stop at
    # the timestep deadline instead of running past it.
    mpc_pricing = os.environ.get('TTTCARV_MPC_PRICING', '0') != '0'
    if mpc_pricing:
        import time_budget as _tbm
        budget.pricing        = True
        budget.margin         = float(os.environ.get('TTTCARV_BUDGET_MARGIN', '0.003'))
        mpc_sf.solve_observer = _tbm.record_solve
        mpc_sf.solve_gate     = lambda: budget.remaining >= budget.mpc_cost
        mpc_sf.time_left      = lambda: budget.remaining - budget.margin
    # Concrete propagation is priced the same way (budget.concrete_cost becomes
    # the p95 measured seconds per step, and the margin is held back).
    globals()['_CONCRETE_OBSERVER'] = _tbm.record_concrete if mpc_pricing else None
    # P2 stops between concrete steps when the next one may not fit.
    globals()['_CONCRETE_DEADLINE'] = ((lambda: budget.remaining - budget.margin,
                                        lambda: budget.concrete_cost)
                                       if mpc_pricing else None)

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next   = current_timestep + 1
        wall_tau = None  # reset each iteration; P3 sets it if it hits INFEASIBLE

        # ══════════════════════════════════════════════════════════════════
        #  MPC ACTIVE PHASE
        # ══════════════════════════════════════════════════════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}  conflict={conflict_time}"
                  f"  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']

            # Clear future horizons; then interleave: one concrete step -> one MPC build, repeat while budget allows.
            # This is to check if we can revert to nominal.
            # If we apply concrete now and mpc build is valid in the next timestep, we are safe to revert

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
                    # INVARIANT: a certificate in mpc_buffer is only meaningful
                    # while the horizon it was derived from is live. The robot
                    # has deviated from the nominal plan, so every future
                    # horizon is stale -- and so is every backup built from one.
                    # Dropping them here is fail-safe by construction: The rebuild loop below runs
                    # AFTER this purge, so backups re-derived from the actually
                    # flown state survive untouched.

                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)

            # Update mpc_horizon_until to the current timestep to match the purge point, so we don't create a gap in buffer
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            # Held back from the rebuild loop so the re-solve below can always
            # pay for itself. Zero when there is no re-solve to protect.
            resolve_reserve = (budget.mpc_cost * (1.0 if budget.pricing else SINGLE_START_COST_FRAC)
                               if (receding and RESOLVE_RESERVE) else 0.0)
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                # One concrete step
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                # One MPC build for this timestep.
                #
                # The build at t_next is NOT speculative: it is the revert test
                # itself, the exact entry psf_valid() reads below. It must never
                # be traded for the re-solve, and it cannot usefully be, because
                # the two are mutually exclusive -- if it comes back VALID the
                # revert fires and `continue`s before the re-solve is reached,
                # and if it comes back INFEASIBLE the loop breaks here after one
                # build. Every LATER build is lookahead buffer for after the
                # revert, and that is the only work the reserve should touch.
                #
                # Reserving against the first build instead (as the version
                # measured in runs/alg22_reserve did) blocks the revert
                # decision, so the filter cannot hand control back: active
                # timesteps went 2496 -> 5834 on 2500 trials while
                # mpc_fallback_uncertified barely moved, 425 -> 404.
                need = budget.mpc_cost + (0.0 if concrete_until == t_next
                                          else resolve_reserve)
                if concrete_until in tester.horizons and budget.remaining >= need:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until, mpc_reason)
                    mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                    if valid:
                        t_diverge = concrete_until
                    print(f"  [MPC PSF] τ={concrete_until} {'VALID' if valid else 'INFEASIBLE'}"
                          f"  t_diverge={t_diverge}")
                    if not valid:
                        break  # cannot revert to nominal, so stop building backups
                else:
                    # Only a budget shortfall is worth counting; falling off the
                    # end of tester.horizons is the loop's normal exit.
                    if concrete_until in tester.horizons:
                        mpc_skipped_no_budget += 1
                        # A speculative lookahead build that was affordable on
                        # its own and was given up only to keep the re-solve
                        # payable. Separating this from a genuine shortfall is
                        # what makes the reserve measurable: it is the price
                        # paid, and it should show up against a fall in
                        # fail_no_budget. The revert test at t_next is exempt
                        # above and so can never land here.
                        if budget.remaining >= budget.mpc_cost:
                            mpc_build_yielded_to_resolve += 1
                    break  # no MPC budget remaining

            
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # PSF reversion: if backup from t+1 (from the concrete RSOA) is valid, return to nominal.
            # Build forward buffer with remaining budget so we have lookahead buffer if we can revert
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None

                # Reset pending_job to start from the new post-MPC position.
                # The mpc changed trajectory so prev symbolic propagation is now inaccurate
                # conflict_time is kept. P4 will re-verify from here.
                pending_job = (RefinementTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
                # were built in section above and remain valid backup buffers
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep,
                                                       below=conflict_time)
                if t_diverge is None or t_diverge < t_next:
                    # This branch is entered *because* psf_valid(mpc_buffer, t_next),
                    # and that backup was built this timestep from the concrete RSOA
                    # re-propagated after the maneuver. conflict_time, by contrast,
                    # was computed under the pre-maneuver trajectory and is stale
                    # here by construction (that is why pending_job re-verifies from
                    # t_next). Letting a stale ceiling veto a just-verified backup
                    # leaves t_diverge=None, and the filter then falls through to
                    # unconditional nominal for the remainder of the run.
                    print(f"  [PSF REVERT] ceiling conflict_time={conflict_time}"
                          f" excluded the just-verified backup at τ={t_next}"
                          f" (would give t_diverge={t_diverge}) — using τ={t_next}")
                    t_diverge = t_next
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            # ── Receding-horizon control selection (alg22) ────────────────
            # alg21 applies queue[t - committed_at]: one plan, flown open loop
            # for the whole active phase. The certificate behind that input is
            # traj_bounds[ctrl_idx], computed from the RSOA at committed_at —
            # but the set the robot is actually in is the Kalman bound written
            # by real_state_mpc each step, which is not guaranteed to stay
            # inside it. Re-solving from the current bound and applying only
            # u[0] keeps the applied input backed by the set the robot is in.
            resolved = False
            if receding:
                ok, r_ctrls, r_traj, r_label = solve_plan_at(
                    mpc_sf, tester, current_timestep, budget)
                if (not ok and r_label != "no_budget"
                        and budget.remaining >= budget.mpc_cost):
                    # One retry from the wider warm-start set before accepting a
                    # plan built from a stale bound.
                    ok, r_ctrls, r_traj, r_label = solve_plan_at(
                        mpc_sf, tester, current_timestep, budget, multi_start=True)
                if ok:
                    # Compare the new plan against the tail of the one it
                    # replaces BEFORE overwriting mpc_state.
                    _old_tail = [float(np.asarray(c).flatten()[0])
                                 for c in (mpc_state.get("controls") or [])[ctrl_idx:]]
                    _new_all  = [float(np.asarray(c).flatten()[0]) for c in r_ctrls]
                    if _old_tail and _new_all:
                        _a = float(np.mean(_old_tail))
                        _b = float(np.mean(_new_all))
                        if abs(_a) > FLIP_DEADBAND and abs(_b) > FLIP_DEADBAND:
                            plan_pairs += 1
                            if _a * _b < 0.0:
                                plan_flips += 1
                                print(f"{RED}[HOMOTOPY] t={current_timestep} fresh"
                                      f" plan reverses turn direction: old tail"
                                      f" mean {_a:+.3f} -> new plan mean"
                                      f" {_b:+.3f}{RESET}")
                    # Re-committing resets committed_at, so ctrl_idx restarts at
                    # 0 and can never index past a control that was not applied.
                    mpc_state['committed_at'] = current_timestep
                    mpc_state['controls']     = r_ctrls
                    mpc_state['traj_bounds']  = r_traj
                    ctrl_idx = 0
                    queue    = mpc_state['controls']
                    resolved = True
                    mpc_resolved += 1
                    print(f"{MAGENTA}[MPC RESOLVE] t={current_timestep} re-solved from"
                          f" current bound  |plan|={len(queue)}{RESET}")
                else:
                    mpc_resolve_failed += 1
                    _rk = str(r_label).split('(')[0]
                    mpc_fail_reason[_rk] = mpc_fail_reason.get(_rk, 0) + 1
                    if _rk == 'rollout_rejected' and '(' in str(r_label):
                        _d = str(r_label).split('(', 1)[1].rstrip(')')
                        _kind, _, _pos = _d.partition('@')
                        _node = _pos.split('/')[0]
                        for _key in (_kind, f"{_kind}@{_node}"):
                            mpc_reject_detail[_key] = mpc_reject_detail.get(_key, 0) + 1
                    print(f"{RED}[MPC RESOLVE] t={current_timestep} FAILED ({r_label})"
                          f" — falling back to plan from"
                          f" t={mpc_state['committed_at']}  ctrl_idx={ctrl_idx}{RESET}")

            if not resolved:
                # Fallback, and alg21's only path: ride the committed plan.
                # Extend whenever <= n_horizon controls remain. Placed after the
                # revert check: no point extending a queue we're about to abandon.
                #
                # Under psf_tail the extend is withheld while the committed
                # plan's own certificate still covers this index. ctrl_idx is
                # exactly k - k_bar, so the bound of Algorithm 1's branch 2 is
                # ctrl_idx < n_horizon; inside it the stored control is the
                # shifted tail and is safe without a solve. At the bound the
                # certificate is spent and branch 3 (the terminal controller)
                # is what the algorithm calls for. That is not implemented, so
                # the old extend still runs — but as a counted, named departure
                # from the algorithm rather than the silent default.
                tail_ok = (receding and psf_tail
                           and ctrl_idx < mpc_sf.n_horizon
                           and ctrl_idx < len(queue))
                if receding and psf_tail and not tail_ok:
                    mpc_tail_exhausted += 1
                    print(f"{RED}[PSF TAIL] t={current_timestep} certificate spent"
                          f" (k-k_bar={ctrl_idx}, N={mpc_sf.n_horizon},"
                          f" |plan|={len(queue)}) — branch 3 (terminal controller)"
                          f" not implemented, extending as alg21 does"
                          f"  (seed={seed}){RESET}")
                if not tail_ok:
                    if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                        extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                            ctrl_idx=ctrl_idx, budget=budget)
                        queue = mpc_state['controls']

                    # Apply queued MPC control
                    if ctrl_idx >= len(queue):
                        #shouldnt run if everything is working
                        print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                        break

                # The fallback input is certified only while the bound the robot
                # is actually in sits inside the bound the plan predicted for
                # this index. Outside that, traj_bounds[ctrl_idx] describes a set
                # the robot is not in and its clearance check is void. Still the
                # best available action, so it is applied — but counted, never
                # silently treated as verified.
                if receding and ctrl_idx > 0:
                    _tb  = mpc_state['traj_bounds']
                    _now = _bounds_at(tester, current_timestep)
                    if (ctrl_idx >= len(_tb)
                            or not _bounds_contain(_tb[ctrl_idx], _now)):
                        mpc_fallback_uncertified += 1
                        print(f"{RED}[MPC RESOLVE] fallback at idx={ctrl_idx}:"
                              f" current RSOA is NOT inside traj_bounds[{ctrl_idx}]"
                              f" — applied input is uncertified"
                              f"  (seed={seed}, t={current_timestep}){RESET}")

            ctrl = queue[ctrl_idx]
            # Only consecutive active timesteps are comparable: a gap means the
            # filter went back to nominal in between, and a sign change across
            # that gap is the NN doing, not the solver.
            _u0 = float(np.asarray(ctrl).flatten()[0])
            if (_prev_applied_u is not None
                    and _prev_applied_t == current_timestep - 1
                    and abs(_u0) > FLIP_DEADBAND
                    and abs(_prev_applied_u) > FLIP_DEADBAND):
                applied_u_pairs += 1
                if _u0 * _prev_applied_u < 0.0:
                    applied_u_flips += 1
            _prev_applied_u = _u0
            _prev_applied_t = current_timestep
            print(f"{MAGENTA}[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                  f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s{RESET}")
            u_nn = _get_nn_control(tester, current_timestep)
            u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
            continue

        # ═════════════════════════════════════════════════════════════════
        #  NOMINAL / PRE_CONFLICT PHASE
        # ═════════════════════════════════════════════════════════════════

        # Stale conflict check — if we've already navigated past conflict_time
        # safely, clear it so we don't stay in PRE_CONFLICT forever.
        if conflict_time is not None and current_timestep > conflict_time:
            print(f"{GREEN}[STALE] conflict_time={conflict_time} is in the past"
                  f" (t={current_timestep}) — clearing{RESET}")
            conflict_time = None
            pending_job   = None

        phase = "PRE_CONFLICT" if conflict_time is not None else "NOMINAL"
        color = BLUE if phase == "PRE_CONFLICT" else GREEN
        print(f"\n{color}CURRENT TIMESTEP ==== {current_timestep}"
              f"  concrete_until={concrete_until}  mpc_hz={mpc_horizon_until}"
              f"  conflict={conflict_time}  t_diverge={t_diverge}  [{phase}]{RESET}")

        # -- P1: Ensure mpc_buffer[t+1] exists (Safety Filter) --

        # alg21: P1 always runs. alg20 skipped the t+1 build whenever the robot
        # was in passthrough AND the RSOA at t+1 overlapped D, to avoid paying
        # mpc_cost for a build it assumed was doomed. That assumption came from
        # the same D-as-precondition reasoning dropped in scan_window, and it
        # cost more than it saved: skipping the build is what KEEPS passthrough
        # alive, because the only way to leave passthrough early is to find a
        # backup at t+1. Running P1 unconditionally can only shorten the stretch
        # flown without a backup in hand. The budget guard below still applies,
        # so the extra work is bounded rather than unconditional.

        # Wall at conflict_time. concrete_scan certified that RSOA(conflict_time)
        # intersects a raw obstacle, so at tau >= conflict_time the robot may
        # already be in collision and "a backup starting from tau" has no valid
        # answer. build_mpc_backup would not notice: it unpacks traj_bounds[1:],
        # dropping the launch bound before the raw check, so a maneuver departing
        # from inside an obstacle still reads VALID. Refuse to build instead.
        past_wall = conflict_time is not None and t_next >= conflict_time
        if past_wall and t_next not in mpc_reason:
            mpc_reason[t_next] = (f"conflict_wall(t+1={t_next} >= conflict_time="
                                  f"{conflict_time}; RSOA there is not raw-clear)")

        # Normal P1 operations
        if t_next not in mpc_buffer and not past_wall:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            # Two distinct signatures reach the no-backup branch. `buf[t+1]=absent
            # mpc_hz=-1` means the budget gate below refused the build -- seen
            # only on starved workers, so it is handled by running uncontended
            # rather than by overrunning the budget. `buf[t+1]=None` means the
            # build RAN and its plan was rejected -- the healthy-run case,
            # addressed by BOOTSTRAP_RESIM. Both read `td=never_set`: the buffer
            # is empty only at t=0, which is why every recorded event sits there.
            _have_fallback = (t_diverge is not None
                              and t_diverge > current_timestep
                              and psf_valid(mpc_buffer, t_diverge))
            _afford = budget.remaining >= budget.mpc_cost
            if t_next not in tester.horizons:
                p1_skipped_no_bounds += 1
            elif _afford:
                # No fallback ahead: this build is load-bearing, so pay for the
                # solver that converges rather than the one that is cheap.
                # Restored in `finally` -- an exception here must not leave the
                # filter permanently on the expensive path.
                _os_saved = mpc_sf.oneshot
                _resim_forced = (BOOTSTRAP_RESIM and not _have_fallback
                                 and _os_saved)
                if _resim_forced:
                    mpc_sf.oneshot = False
                    p1_bootstrap_resim += 1
                try:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next, mpc_reason)
                finally:
                    mpc_sf.oneshot = _os_saved
                # Counted in every arm, whatever solver ran: this is the rate
                # the solver choice is meant to move, and unlike psf_no_diverge
                # it is sampled on every trial rather than once per ~10k.
                if not _have_fallback and not valid:
                    p1_bootstrap_infeasible += 1
                # _have_fallback is false far more often than at t=0 -- every
                # time the robot catches up to t_diverge -- and most of those
                # failures are harmless (nominal passthrough covers them). Only
                # t=0 has a structurally empty buffer, so count it on its own.
                if current_timestep == 0 and not valid:
                    p1_t0_infeasible += 1
                mpc_horizon_until = max(mpc_horizon_until, t_next)
                if valid and (t_diverge is None or t_next > t_diverge):
                    t_diverge = t_next
                print(f"  [P1] mpc_buffer[{t_next}] = {'VALID' if valid else 'INFEASIBLE'}"
                      f"  t_diverge={t_diverge}"
                      f"{'  [RESIM: no fallback]' if _resim_forced else ''}")
            else:
                p1_skipped_no_budget += 1

        # ---- P2: Concrete scan forward (stop at conflict_time) -----
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(
                concrete_until + budget.max_affordable_concrete(),
                scan_ceil, MAX_TIME
            )
            collision, ct, reached = concrete_scan_deadline(tester, concrete_until, end)
            if collision:
                conflict_time = ct
                print(f"  [P2] Conflict detected at t={conflict_time}")
                # A t_diverge established before this conflict was known may now
                # sit at or beyond it. Nominal is only concrete-certified for
                # t < conflict_time, so re-derive t_diverge under the new
                # ceiling. Neither the P1 wall nor P3/P3b's bounds can catch
                # this: t_diverge was already set when conflict_time moved.
                if t_diverge is not None and t_diverge >= conflict_time:
                    _stale = t_diverge
                    t_diverge = _max_valid_diverge(mpc_buffer, current_timestep,
                                                   below=conflict_time)
                    print(f"  [P2] t_diverge={_stale} is at/beyond the new conflict"
                          f" — clamped to {t_diverge}")
                if pending_job is None:
                    pending_job = RefinementTask(current_timestep, conflict_time)
                concrete_until = ct
                break
            concrete_until = reached
            if reached < end:
                concrete_deadline_hits += 1
                break

        # ---- P3: Build MPC buffer forward toward conflict_time -------
        if conflict_time is not None:
            mpc_horizon_until = max(mpc_horizon_until, current_timestep)
            while (mpc_horizon_until < conflict_time - 1
                   and budget.remaining >= budget.mpc_cost):
                tau = mpc_horizon_until + 1
                if tau not in tester.horizons:
                    break
                if mpc_buffer.get(mpc_horizon_until) is None:
                    wall_tau = mpc_horizon_until
                    break
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau, mpc_reason)
                mpc_horizon_until = tau
                if valid:
                    t_diverge = tau
                    print(f"  [P3] mpc_buffer[{tau}] VALID  t_diverge → {t_diverge}")
                else:
                    print(f"  [P3] mpc_buffer[{tau}] INFEASIBLE — wall at tau={tau}")
                    wall_tau = tau
                    break

        #  P3b: Passthrough scan beyond infeasible wall
        # alg21: invoked after ANY wall. alg20 first required a D-region ahead
        # (some bound overlapping D but not the raw obstacle), which made a wall
        # with no D-region ahead unrecoverable even when a perfectly good backup
        # sat one tau further on -- an INFEASIBLE build has causes other than D
        # (path_raw, solver error, a vacuous heading bound), and none of them
        # imply the next tau is bad. scan_window is already budget-guarded and
        # aborts on the first raw collision, so the unconditional call is bounded.
        if wall_tau is not None and budget.remaining >= budget.mpc_cost:
            scan_limit = (min(conflict_time - 1, concrete_until)
                          if conflict_time is not None
                          else concrete_until)
            passthrough_td = scan_window(
                tester, mpc_sf, mpc_buffer, obstacles,
                wall_tau, scan_limit, budget, R=turning_radius,
                reasons=mpc_reason)
            if passthrough_td is not None:
                t_diverge = passthrough_td
                mpc_horizon_until = passthrough_td
                passthrough_found += 1
                print(f"  [P3b] Passthrough t_diverge -> {t_diverge}")

        # ── P4: Symbolic. Still run even after confirmation
        if (pending_job is not None
                and budget.can_afford('symbolic', 1)):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    # Symbolic tightened bounds so drop INFEASIBLE entries so
                    # P3 retries them with tighter bounds (may extend t_diverge).
                    # VALID entries remain safe
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    # Pull mpc_horizon_until back to t_diverge so P3 re-scans the
                    # purged gap with tightened bounds on the next iteration.
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    # Reset job for continuous refinement
                    pending_job       = RefinementTask(current_timestep, conflict_time)
                else:
                    # Deconflicted -  clear conflict state but keep VALID buffer
                    # entries; they're still collision-free and serve as backups
                    # if a new conflict appears immediately after.
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION: nominal or activate MPC backup
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            # PSF satisfied —> nominal control is safe
            #
            # Record t_next as the divergence point. This branch is entered
            # *because* mpc_buffer[t_next] is valid, so t_next IS a certified
            # place to diverge; not writing it down is what lets the clock
            # overtake t_diverge. P1 normally raises it (same rule, same
            # place), but P1 is skipped whenever buffer[t_next] was already
            # built on an earlier timestep — and that is precisely when this
            # branch fires without it. Once current_timestep > t_diverge the
            # activation site computes ctrl_idx > 0 and joins a plan whose
            # first controls were never applied, so traj_bounds[ctrl_idx]
            # describes a trajectory the robot is not on.
            #
            # Only ever raises t_diverge, exactly as P1 does, so a pending
            # passthrough target further ahead is left alone. After the
            # increment below the invariant t_diverge >= current_timestep
            # holds, which makes ctrl_idx > 0 unreachable rather than merely
            # unobserved.
            if t_diverge is None or t_diverge < t_next:
                t_diverge = t_next
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

        elif (t_diverge is not None and t_diverge > current_timestep
                and psf_valid(mpc_buffer, t_diverge)):
            # Passthrough: no backup at t+1, but a valid one exists at t_diverge.
            #
            # Tripwire for the invariant this branch rests on. Flying nominal to
            # t_diverge is only certified because concrete_scan proved every step
            # below conflict_time is raw-clear. P1's wall, P3's loop bound, P3b's
            # scan_limit and P2's clamp each maintain it; none of them is checked
            # here, so a fifth path to t_diverge must fail loudly, not silently.
            assert conflict_time is None or t_diverge < conflict_time, (
                f"PASSTHROUGH gap [{current_timestep},{t_diverge}] spans the "
                f"conflict at t={conflict_time}: nominal is not concrete-certified "
                f"over this gap")
            #
            # Report the REAL reason there is no backup at t+1. alg20 had a
            # further arm here for the skip_p1 route ("inside D"); with P1 no
            # longer skipped on D membership, every no-backup case now reaches
            # this branch having actually attempted a build, so mpc_reason holds
            # the true cause (path_raw, term_D, solver_error, conflict_wall).
            if t_next in mpc_reason:
                _why = f"no backup at t+1: {mpc_reason[t_next]}"
            elif t_next not in mpc_buffer:
                _why = "no backup at t+1: never built (budget or bounds missing)"
            else:
                _why = "no backup at t+1: reason unrecorded"

            # State the basis for flying nominal this step. The guarantee is
            # concrete raw-clearance over the gap, NOT the D geometry.
            _basis = (f"nominal concrete-clear through t={concrete_until} (no conflict)"
                      if conflict_time is None
                      else f"nominal concrete-clear to conflict t={conflict_time}")

            print(f"  [PSF PASSTHROUGH] t={current_timestep} — {_why}; "
                  f"t_diverge={t_diverge} valid ahead; {_basis}"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            passthrough_used += 1
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

        else:
            # PSF fails — need MPC backup from t_diverge
            #
            # A backup is a plan from a specific state at a specific time:
            # controls[k] and traj_bounds[k] both assume controls[0..k-1] were
            # the inputs actually applied. Joining a plan at ctrl_idx > 0 means
            # those inputs were NOT applied — the filter flew nominal over
            # those steps — so traj_bounds[ctrl_idx] describes a state the
            # robot is not in, and every clearance check made against it is
            # void. t_diverge persists across loop iterations, so it can point
            # at a plan committed on an earlier timestep; that is a stale
            # pointer, not a backup.
            #
            # Prefer the plan committed at this exact timestep. If there isn't
            # one, refuse rather than fly a plan whose bounds do not describe
            # the robot: fall through to the no-backup branch, which at least
            # reports the condition instead of hiding it.
            if (t_diverge is not None and t_diverge != current_timestep
                    and psf_valid(mpc_buffer, current_timestep)):
                print(f"{RED}[PSF ACTIVATE] stale t_diverge={t_diverge} at"
                      f" t={current_timestep} (ctrl_idx would be"
                      f" {current_timestep - t_diverge}) — using the plan"
                      f" committed at t={current_timestep} instead{RESET}")
                t_diverge = current_timestep
            if t_diverge is not None and t_diverge != current_timestep:
                print(f"{RED}[PSF ACTIVATE] REFUSED stale backup:"
                      f" t_diverge={t_diverge} != t={current_timestep}"
                      f" (would join at ctrl_idx={current_timestep - t_diverge});"
                      f" no plan committed at t={current_timestep}"
                      f"  (seed={seed}){RESET}")
                psf_stale_refused += 1
                t_diverge = None

            if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
                controls, traj_bounds = mpc_buffer[t_diverge]
                mpc_state['committed_at']  = t_diverge
                mpc_state['conflict_time'] = conflict_time
                mpc_state['controls']      = list(controls)
                mpc_state['traj_bounds']   = list(traj_bounds)
                mpc_state['needed']        = True
                mpc_started                = True
                mpc_calls                 += 1

                ctrl_idx = current_timestep - t_diverge
                queue    = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    print(f"{RED}[PSF ACTIVATE] t={current_timestep}  PSF failed —"
                          f" backup from t_diverge={t_diverge}"
                          f"  ctrl_idx={ctrl_idx}  u={np.round(ctrl, 4)}{RESET}")
                    u_nn = _get_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    # ctrl_idx out of range — extend first
                    print(f"{RED}[PSF ACTIVATE] ctrl_idx={ctrl_idx} beyond queue"
                          f" ({len(queue)}) — extending{RESET}")
                    extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                        ctrl_idx=ctrl_idx, budget=budget)
                    queue = mpc_state['controls']
                    if ctrl_idx < len(queue):
                        ctrl = queue[ctrl_idx]
                        u_nn = _get_nn_control(tester, current_timestep)
                        u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                        tester.real_state_mpc(current_timestep, ctrl)
                        current_timestep += 1
                    else:
                        print(f"[PSF] Cannot activate — queue still empty. Nominal fallback."
                              f"  (seed={seed}, t={current_timestep})")
                        psf_queue_empty += 1
                        u_diffs.append(0.0)
                        tester.real_state_empirical(current_timestep, t_next)
                        current_timestep += 1
            else:
                # No valid backup yet — mpc_buffer[t+1] not ready (P1 stalled)
                # Apply nominal and warn; P1 will catch up next timestep.
                # SHOULD NOT RUN IF WORKING!!!!!
                _bnext   = tester.horizons[t_next].get_tight_bound() if t_next in tester.horizons else None
                _in_D    = (_bnext is not None and collides_danger(_bnext, obstacles, turning_radius))
                _in_raw  = (_bnext is not None and collides_raw(_bnext, obstacles))
                _buf_t1  = mpc_buffer.get(t_next, 'absent')
                _td_state = (
                    'never_set'        if t_diverge is None
                    else f'stale({t_diverge}<=t)'   if t_diverge <= current_timestep
                    else f'invalid({t_diverge})'   if not psf_valid(mpc_buffer, t_diverge)
                    else f'ok({t_diverge})'
                )
                print(f"{RED}[PSF] No backup at t={current_timestep}"
                      f"  buf[{t_next}]={'None' if _buf_t1 is None else ('absent' if _buf_t1=='absent' else 'VALID')}"
                      f"  td={_td_state}"
                      f"  bnext_in_D={_in_D}  bnext_in_raw={_in_raw}"
                      f"  conflict={conflict_time}  mpc_hz={mpc_horizon_until}"
                      f"  (seed={seed}){RESET}")
                # TERMINAL SAFE CONTROLLER.
                #
                # Predicate is the CURRENT bound, not _bnext. _bnext is the RSOA
                # under the nominal NN control; applying the loiter instead sends
                # the robot somewhere that bound does not describe, so it cannot
                # certify this step. The current RSOA does: if it clears D, a
                # max-turn circle from anywhere inside it stays outside the raw
                # obstacle indefinitely, which is the invariance the split
                # terminal constraint is built on.
                _bcur = (tester.horizons[current_timestep].get_tight_bound()
                         if current_timestep in tester.horizons else None)
                _cur_clears_D = (_bcur is not None
                                 and not collides_danger(_bcur, obstacles,
                                                         turning_radius))
                if PSF_LOITER and _cur_clears_D:
                    _u_loiter = np.array([u_max])
                    print(f"{RED}[PSF LOITER] t={current_timestep} — no backup;"
                          f" current RSOA clears D, holding max-turn loiter"
                          f" u={_u_loiter}  (seed={seed}){RESET}")
                    psf_loiter_used += 1
                    u_nn = _get_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(_u_loiter[0] - u_nn[0]))
                                   if u_nn is not None else 0.0)
                    tester.real_state_mpc(current_timestep, _u_loiter)
                    current_timestep += 1
                    continue

                if not _cur_clears_D:
                    psf_unrecoverable += 1

                psf_no_diverge += 1
                _nd_key = ('never_built' if _buf_t1 == 'absent'
                           else 'rejected:' + str(mpc_reason.get(t_next, 'unrecorded')).split('(')[0]
                           if t_next in mpc_reason
                           else 'built_none' if _buf_t1 is None
                           else 'unrecorded')
                psf_no_diverge_reasons[_nd_key] = psf_no_diverge_reasons.get(_nd_key, 0) + 1
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1



    budget.close_timestep()
    if _budget_path:
        import json
        with open(_budget_path, 'w') as _f:
            json.dump({'seed': seed,
                       'timestep_budget': budget.timestep_budget,
                       'calibrated_costs': {
                           'symbolic': dict(budget.symbolic_costs),
                           'concrete': budget.concrete_cost,
                           'mpc':      budget.mpc_cost},
                       'ticks': budget.ledger}, _f)
        print(f"[BUDGET] wrote {len(budget.ledger)} timestep records to {_budget_path}")

    if _rsoa_path:
        import json
        with open(_rsoa_path, 'w') as _f:
            json.dump({'seed': seed, 'dt': tester.analyzer.cl_system.dynamics.dt,
                       'turning_radius': turning_radius,
                       'obstacles': [np.asarray(o).flatten().tolist() for o in obstacles],
                       'steps': rsoa_log}, _f)
        print(f"[RSOA] wrote {len(rsoa_log)} step certificates to {_rsoa_path}")

    # ── Collect state history and check real-state collisions ─────────────
    state_history = []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(calc['real_state'].copy())
                break

    collision_timesteps = []
    for i, s in enumerate(state_history):
        s_flat = np.asarray(s).flatten()
        for obs in obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            if (s_flat[0] - cx) ** 2 + (s_flat[1] - cy) ** 2 <= r ** 2:
                collision_timesteps.append(i)
                break

    had_collision = len(collision_timesteps) > 0
    print(f"\n{'=' * 60}")
    print(f"Simulation complete at timestep {current_timestep}")
    if had_collision:
        print(f"[SAFETY] COLLISION at real-state timesteps: {collision_timesteps}")
    else:
        print(f"[SAFETY] No real-state collision detected")
    print(f"  mpc_calls={mpc_calls}  psf_no_diverge={psf_no_diverge}"
          f"  psf_queue_empty={psf_queue_empty}"
          f"  psf_stale_refused={psf_stale_refused}"
          f"  passthrough_found={passthrough_found}"
          f"  passthrough_used={passthrough_used}"
          f"  max_u_diff={max(u_diffs, default=0):.3f}")
    print(f"  p1_bootstrap_resim={p1_bootstrap_resim}"
          f"  p1_bootstrap_infeasible={p1_bootstrap_infeasible}"
          f"  p1_t0_infeasible={p1_t0_infeasible}"
          f"  psf_loiter_used={psf_loiter_used}")
    LAST_RUN_STATS.clear()
    LAST_RUN_STATS.update({
        'receding':                 int(receding),
        'oneshot':                  int(oneshot),
        'oneshot_refine':           oneshot_refine,
        'oneshot_mismatch':         float(getattr(mpc_sf, 'oneshot_model_mismatch', 0.0)),
        'oneshot_mismatch_acc':     float(getattr(mpc_sf, 'oneshot_mismatch_accepted', 0.0)),
        'oneshot_solves':           int(getattr(mpc_sf, 'oneshot_solves', 0)),
        'oneshot_bad_status':       int(getattr(mpc_sf, 'oneshot_bad_status', 0)),
        'first_step_cost':          int(first_step_cost),
        'tail_cost_weight':         tail_cost_weight,
        'mpc_resolved':             mpc_resolved,
        'mpc_resolve_failed':       mpc_resolve_failed,
        'mpc_fallback_uncertified': mpc_fallback_uncertified,
        'psf_tail':                 int(psf_tail),
        'mpc_tail_exhausted':       mpc_tail_exhausted,
        'single_start_frac':        SINGLE_START_COST_FRAC,
        'applied_u_flips':          applied_u_flips,
        'applied_u_pairs':          applied_u_pairs,
        'plan_flips':               plan_flips,
        'plan_pairs':               plan_pairs,
        # Dead-worker detector. mpc_cost_clamped==1 means this process's
        # once-per-process calibration exceeded the timestep budget and would
        # have disabled the filter outright before the guard existed;
        # mpc_skipped_no_budget counts every build the budget gate refused.
        # A trial with mpc_calls==0 and mpc_skipped_no_budget>0 was flown
        # unfiltered and must not be scored as a filter result.
        'mpc_cost_clamped':         mpc_cost_clamped,
        'mpc_skipped_no_budget':    mpc_skipped_no_budget,
        'mpc_build_yielded':        mpc_build_yielded_to_resolve,
        # An unfiltered step taken at a moment the filter had decided it needed
        # to intervene. Zero is the only acceptable value; the reason breakdown
        # is what says whether the cause is budget or the acceptance test.
        'psf_no_diverge_reasons':   dict(psf_no_diverge_reasons),
        'psf_stale_refused':        psf_stale_refused,
        'p1_skipped_no_budget':     p1_skipped_no_budget,
        'p1_skipped_no_bounds':     p1_skipped_no_bounds,
        'psf_loiter_used':          psf_loiter_used,
        'p1_bootstrap_resim':       p1_bootstrap_resim,
        'p1_bootstrap_infeasible':  p1_bootstrap_infeasible,
        'p1_t0_infeasible':         p1_t0_infeasible,
        'bootstrap_resim':          int(BOOTSTRAP_RESIM),
        'psf_unrecoverable':        psf_unrecoverable,
        'psf_loiter':               int(PSF_LOITER),
        'resolve_reserve':          int(RESOLVE_RESERVE),
        'mpc_cost_s':               float(budget.mpc_cost),
        'mpc_cost_cal_s':           float(budget._mpc_cost_cal),
        'concrete_cost_s':          float(budget.concrete_cost),
        'concrete_cost_cal_s':      float(budget._concrete_cost_cal),
        'mpc_pricing':              int(mpc_pricing),
        'oneshot_rti':              int(oneshot_rti),
        'mpc_deadline_hits':        int(mpc_sf.deadline_hits),
        'mpc_truncated_solves':     int(mpc_sf.truncated_solves),
        'mpc_second_start_skipped': int(mpc_sf.second_start_skipped),
        'concrete_deadline_hits':   int(concrete_deadline_hits),
        # If first_step_cost turns the MPC into a nominal passthrough — the
        # optimizer parking omega_0 at omega_nom and deferring avoidance to the
        # free tail — then the applied control collapses onto the NN control and
        # these go to ~0 while MPC is still nominally "active".
        'u_diff_max':               float(max(u_diffs, default=0.0)),
        'u_diff_mean_active':       float(
            sum(d for d in u_diffs if d > 0.0)
            / max(sum(1 for d in u_diffs if d > 0.0), 1)),
        'u_diff_n_active':          int(sum(1 for d in u_diffs if d > 0.0)),
        **{f'fail_{k}': v for k, v in mpc_fail_reason.items()},
        **{f'rej_{k}': v for k, v in mpc_reject_detail.items()},
    })
    print(f"  [alg22] receding={'on' if receding else 'off'}"
          f"  mpc_resolved={mpc_resolved}"
          f"  mpc_resolve_failed={mpc_resolve_failed}"
          f"  mpc_fallback_uncertified={mpc_fallback_uncertified}"
          f"  mpc_tail_exhausted={mpc_tail_exhausted}"
          f"  fail_reasons={mpc_fail_reason or '{}'}"
          f"  rejects={mpc_reject_detail or '{}'}"
          f"  u_flips={applied_u_flips}/{applied_u_pairs}"
          f"  plan_flips={plan_flips}/{plan_pairs}"
          f"  mpc_skipped_no_budget={mpc_skipped_no_budget}"
          f"{'  [MPC COST CLAMPED]' if mpc_cost_clamped else ''}")
    return (state_history, had_collision, u_diffs, mpc_calls,
            psf_no_diverge, psf_queue_empty,
            passthrough_found, passthrough_used)

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    obs = None
    if len(sys.argv) > 2:
        # Parse obstacles: x1,y1,r1 x2,y2,r2 ...
        obs = [np.array([float(v) for v in a.split(',')]) for a in sys.argv[2:]]
    test(seed=seed, obstacles=obs)

