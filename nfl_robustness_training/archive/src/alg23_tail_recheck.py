"""
alg23 -- Reach_PSF_FINAL_NL plus a tail re-check on the fallback path.

When a re-solve fails, Reach_PSF_FINAL keeps flying the committed plan and
calls it uncertified unless the current RSOA is nested inside the plan's box
for this step. Both boxes have almost the same width, so that nesting test
nearly always fails. alg23 instead re-runs the acceptance test on the plan's
remaining controls, starting from the current RSOA (TTTCARV_TAIL_RECHECK=1).
If that fails and TTTCARV_TAIL_RECHECK_ACT=1, the robot loiters (constant
max turn) when the whole loiter circle from the current RSOA clears the raw
obstacles, trying the turn away from the nearest obstacle first. This works
inside D as long as the robot is not pointed at the obstacle. Otherwise the
plan is applied anyway and counted.
Everything else is identical to Reach_PSF_FINAL_NL.py.

Reachability-based predictive safety filter (PSF) for the unicycle with a
neural-network (NN) controller.

Every 0.2 s timestep the filter decides whether the NN's control is safe to
apply. It is safe when a verified MPC backup plan exists from the next state
(mpc_buffer[t+1]): the robot can then always switch to that plan later.

Reachable sets (RSOA) come in two kinds:
  concrete  cheap, looser bounds, propagated one step at a time
  symbolic  expensive, tighter bounds, computed in chunks that may take
            several timesteps of budget

States:
  NOMINAL       no conflict ahead; keep a backup ready at t+1.
  PRE_CONFLICT  the concrete RSOA hits an obstacle at conflict_time. Build
                backups forward toward it and try to rule it out symbolically.
  MPC_ACTIVE    no backup at t+1, so the backup plan flies the robot
                (re-solved every step) until a backup at t+1 exists again.

Work each timestep, in priority order, each only if the budget allows:
  P1   make sure mpc_buffer[t+1] exists (the safety check itself)
  P2   concrete scan forward, up to conflict_time or MAX_TIME
  P3   build backups forward toward conflict_time - 1
  P3b  if P3 hit an infeasible build (a "wall"), look past it for a valid one
  P4   symbolic refinement of the conflict

A backup is accepted when every intermediate node clears the raw obstacle
radius r and the terminal node clears the danger radius D = sqrt(r^2 + 2rR),
from where a max-turn loiter can keep clear of the obstacle.

Obstacles are [center_x, center_y, radius].

Usage (from the repo root):
    python nfl_robustness_training/src/alg23_tail_recheck.py <seed> x,y,r ...
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


# ── Configuration ──────────────────────────────────────────────────────────
# Settings come from TTTCARV_* environment variables. Not every default is the
# final configuration: the final run (results_alg22/alg22_results.md) sets
# TTTCARV_ONESHOT=1, TTTCARV_ONESHOT_RTI=10, TTTCARV_MPC_PRICING=1,
# TTTCARV_BUDGET_MARGIN=0.003, TTTCARV_PSF_TAIL=1, TTTCARV_ONESHOT_VERIFY=1 and
# TTTCARV_SINGLE_START_FRAC=0.50.

# Extra per-trial stats. test() returns a fixed tuple shared with the other
# algorithms, so everything else goes here; the trial runner merges it into the
# results row after each trial.
LAST_RUN_STATS: dict = {}

# Share of mpc_cost charged for a single-start re-solve when pricing is OFF.
# mpc_cost is calibrated on a multi-start build (1-2 solves), so the true share
# is 0.5-1.0 and 0.25 under-reserves. Ignored with TTTCARV_MPC_PRICING=1, which
# charges every solve its measured price.
SINGLE_START_COST_FRAC = float(os.environ.get('TTTCARV_SINGLE_START_FRAC', '0.25'))

# Hold back budget for the active-phase re-solve so the rebuild loop cannot
# spend it. Off by default: it measured as having no effect, because the loop's
# first build is the revert test, which either reverts (so no re-solve is
# needed) or fails and ends the loop.
RESOLVE_RESERVE = os.environ.get('TTTCARV_RESOLVE_RESERVE', '0') != '0'

# Last resort when there is no backup at all: a max-turn loiter instead of the
# raw NN control, used only if the current RSOA clears D. One step moves at most
# v*dt = 0.2 and D - 0.2 = 0.918 > r = 0.5, so that step cannot collide whatever
# the control; the filter re-plans on the next step. Repeated loiter steps are
# not covered by this argument.
PSF_LOITER = os.environ.get('TTTCARV_PSF_LOITER', '1') != '0'

# When there is no earlier backup to fall back on (e.g. t=0, empty buffer),
# build the t+1 backup with the slower re-simulation solver instead of the
# one-shot solver. The one-shot solver is cheaper but sometimes stops short and
# its plan ends inside D. Elsewhere that rejection is covered by an existing
# backup; here it would leave none. No effect when TTTCARV_ONESHOT=0.
BOOTSTRAP_RESIM = os.environ.get('TTTCARV_BOOTSTRAP_RESIM', '1') != '0'

# Turn-direction flip diagnostics ignore controls with |u| below this; a
# near-zero turn has no meaningful direction.
FLIP_DEADBAND = 0.05

# alg23: on a failed re-solve, re-run the acceptance test on the rest of the
# committed plan from the current RSOA (see recheck_tail). With ACT=1 a failed
# re-check loiters if the loiter circle clears the obstacles (see
# loiter_circle_ok); with ACT=0 it is only counted.
TAIL_RECHECK     = os.environ.get('TTTCARV_TAIL_RECHECK', '1') != '0'
TAIL_RECHECK_ACT = os.environ.get('TTTCARV_TAIL_RECHECK_ACT', '1') != '0'



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


# Hooks set by test() when TTTCARV_MPC_PRICING=1 (None otherwise):
#   _CONCRETE_OBSERVER(sec_per_step)  records the measured concrete cost
#   _CONCRETE_DEADLINE                (time_left, step_price) callables that let
#                                     P2 stop between steps before budget runs out
_CONCRETE_OBSERVER = None
_CONCRETE_DEADLINE = None

def concrete_scan(tester, from_t, to_t):
    """Concrete propagation with no deadline. Returns (collision, collision_t)."""
    collision, ct, _ = concrete_scan_deadline(tester, from_t, to_t, deadline=False)
    return collision, ct


def concrete_scan_deadline(tester, from_t, to_t, deadline=True):
    """Concrete propagation that may stop early at the budget deadline.

    Returns (collision, collision_t, reached); reached is the last timestep that
    now has concrete bounds. Runs all the way to to_t if deadline=False or
    pricing is off.
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
    # Cost = wall time of the whole call / steps completed (includes overhead;
    # a collision stop completes fewer steps).
    if _CONCRETE_OBSERVER is not None and done > 0:
        _CONCRETE_OBSERVER(wall / done)
    if not result["collision"]:
        return False, None, from_t + done
    return True, result["collision_timestep"], result["collision_timestep"]


def symbolic_step(tester, job: RefinementTask, chunk_size: int, budget):
    """
    Advance the symbolic job by one chunk, if the budget allows.

    A chunk usually costs more than one timestep's budget. If it doesn't fit,
    this timestep's remaining budget is banked in job.time_invested instead, and
    the chunk runs once banked time + the current budget covers it.
    Returns (job, result); result is set only when the job finishes.
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
    Append controls to the committed MPC plan (fallback when a re-solve fails).
    Solves from the end of the plan; if that fails, from progressively earlier
    points on it (lookback), replacing the tail.
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

        # The new segment may pass through D but not the raw obstacle, and like
        # any backup it must end outside D. Find:
        #   safe_count  how many new nodes stay clear of the raw obstacle
        #   term_count  the last of those that is also outside D (the new end)
        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides_raw(b):
                break
            safe_count += 1

        # Last node in the raw-safe prefix that clears D.
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
    """True if a valid backup starting at t is stored."""
    return mpc_buffer.get(t) is not None


def _rollout_verdict(mpc_sf, traj_bounds):
    """
    Acceptance test for an MPC plan, shared by backup builds and re-solves.
    Returns (ok, kind, node, n_nodes).

      kind  None | 'empty' | 'path_raw' | 'term_D'
      node  1-based index of the failing node (n_nodes for the terminal)

    Intermediate nodes must clear the raw obstacle radius (they may pass
    through D); the terminal node must clear D.
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


def recheck_tail(mpc_sf, tester, controls, now_bounds):
    """
    Certify the rest of a committed plan from where the robot actually is.

    Runs `controls` through the plant from the center of now_bounds, carries
    now_bounds' half-widths along the path (as a backup build does), and runs
    the result through the acceptance test. Returns (ok, kind, node, n_nodes).
    """
    if now_bounds is None or len(controls) == 0:
        return False, 'empty', -1, 0
    return _rollout_verdict(mpc_sf, _carry_box(tester, controls, now_bounds))


def _carry_box(tester, controls, now_bounds):
    """Run controls through the plant from the box center, keeping the box size."""
    now = np.asarray(now_bounds, dtype=float)
    hw  = (now[:, 1] - now[:, 0]) / 2.0
    xs  = ((now[:, 0] + now[:, 1]) / 2.0).reshape(1, -1)
    dyn = tester.analyzer.cl_system.dynamics
    traj = [now.copy()]
    for u in controls:
        xs = dyn.dynamics_step(xs, np.asarray(u, dtype=float).reshape(1, -1))
        if hasattr(xs, 'numpy'):
            xs = xs.numpy()
        xs = np.array(xs, dtype=float).reshape(1, -1)
        c  = xs.flatten()
        traj.append(np.stack([c - hw, c + hw], axis=1))
    return traj


def loiter_circle_ok(mpc_sf, tester, now_bounds, u):
    """
    True if one full constant-turn circle at control u, starting from now_bounds,
    keeps the box clear of every raw obstacle. The circle closes on itself, so
    repeating it stays clear too.
    """
    dt = float(tester.analyzer.cl_system.dynamics.dt)
    n  = int(np.ceil(2 * np.pi / (abs(u) * dt))) + 1
    traj = _carry_box(tester, [np.array([u])] * n, now_bounds)
    return not any(mpc_sf._collides_raw(b) for b in traj[1:])


def loiter_turn_order(now_bounds, obstacles, u_max):
    """Both turn directions, turning away from the nearest obstacle first."""
    now = np.asarray(now_bounds, dtype=float)
    x, y, th = (now[:, 0] + now[:, 1]) / 2.0
    o = min(obstacles, key=lambda o: (o[0] - x) ** 2 + (o[1] - y) ** 2)
    left = np.cos(th) * (o[1] - y) - np.sin(th) * (o[0] - x) > 0  # obstacle on the left
    return (-u_max, u_max) if left else (u_max, -u_max)


def solve_plan_at(mpc_sf, tester, tau: int, budget=None, multi_start: bool = False):
    """
    Solve an MPC plan from the bounds at tau without storing it in mpc_buffer.
    Returns (ok, controls, traj_bounds, label).

    Used by the active-phase re-solve. It stays out of mpc_buffer because any
    entry there counts as a backup and would change PSF decisions.
    Single-start by default (cheaper); the caller retries with multi_start=True.
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
        # Coarse reason before "(", details inside, so counters can split on "(".
        return False, None, None, f"rollout_rejected({kind}@{node}/{n_nodes})"
    return True, list(controls), list(traj_bounds), "VALID"

def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int, reasons: dict = None):
    """
    Solve an MPC backup from the concrete bounds at tau and store it:
    mpc_buffer[tau] = (controls, traj_bounds) if it passes the acceptance test,
    else None. Returns True if valid.

    reasons (optional dict tau -> str) records why a build was rejected. It is
    separate from mpc_buffer because any non-None buffer entry counts as valid.
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

        # Find which check failed, for the diagnostic label:
        #   path_raw  an intermediate node hits the raw obstacle
        #   term_D    the terminal node is inside D
        path_hit = next(
            ((i, b) for i, b in enumerate(path_bounds, start=1)
             if mpc_sf._collides_raw(b)),
            None
        )
        term_hit = mpc_sf._collides_D(terminal_bound)
        # The verdict itself comes from the shared acceptance test.
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
    """True if the box overlaps any obstacle's danger radius D = sqrt(r^2 + 2rR).
    r is the obstacle radius and R the turning radius (1 here). From outside D a
    max-turn loiter can keep clear of the obstacle.
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
    Passthrough search: look past an infeasible build (a "wall") for a tau that
    has a valid backup. Returns that tau as the new t_diverge, or None.

    For each tau from wall_tau+1 to scan_limit (inclusive): give up if the RSOA
    there hits a raw obstacle, otherwise use an existing valid backup or try to
    build one. tau itself may be inside D; only a backup's terminal node must
    clear D. R is unused (kept for the call signature).
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
    """Largest key >= from_t with a valid backup, or None.

    below (exclusive) is normally conflict_time: nominal is only verified safe
    before conflict_time, so a backup at or past it is not a passthrough target.
    """
    keys = [k for k, v in mpc_buffer.items()
            if k >= from_t and v is not None
            and (below is None or k < below)]
    return max(keys) if keys else None


# ── Simulation loop ────────────────────────────────────────────────────────

def test(seed=None, analyzer=None, obstacles=None):
    # Free the previous trial's acados solver first. Its library is loaded under
    # a per-process name, so a leftover reference cycle would make this trial
    # reuse the old solver with the previous trial's obstacles.
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
    # TTTCARV_FIRST_STEP_COST=1: the stay-close-to-NN cost applies only to the
    # first control (the only one applied under receding horizon).
    first_step_cost = os.environ.get('TTTCARV_FIRST_STEP_COST', '0') != '0'
    # Small cost on the remaining controls when FIRST_STEP_COST is on, to keep
    # the QP well conditioned.
    tail_cost_weight = float(os.environ.get('TTTCARV_TAIL_COST_WEIGHT', '1e-3'))
    # TTTCARV_ONESHOT=1: a backup is one OCP solve that returns the solver's own
    # trajectory, instead of n_horizon solves re-simulated one step at a time.
    # Much cheaper, and the terminal-D constraint applies to the returned end.
    oneshot        = os.environ.get('TTTCARV_ONESHOT', '0') != '0'
    oneshot_refine = int(os.environ.get('TTTCARV_ONESHOT_REFINE', '2'))
    # Compare the one-shot trajectory with a plant re-simulation (diagnostic;
    # costs one extra rollout per solve).
    oneshot_verify = os.environ.get('TTTCARV_ONESHOT_VERIFY', '0') != '0'
    # SQP_RTI iterations per one-shot refine pass (final run: 10).
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

    # ── Calibration guard ─────────────────────────────────────────────────
    # mpc_cost is calibrated once per process and can be inflated by startup
    # contention. P1 spends concrete_cost on bounds before it checks
    # remaining >= mpc_cost, so if mpc_cost >= timestep_budget - concrete_cost
    # the t+1 build could never run and the filter would silently be off.
    # Clamp it below that.
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

    # ── MPC execution state ───────────────────────────────────────────────
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
    # psf_no_diverge broken down by cause (never built vs rejected build).
    psf_no_diverge_reasons: dict = {}   # short key -> count
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
    passthrough_found = 0   # P3b found a valid backup past a wall
    passthrough_used  = 0   # steps flown nominal with no backup at t+1 (one exists ahead)

    # ── Active-phase options ──────────────────────────────────────────────
    # TTTCARV_RECEDING=1 (default): in MPC_ACTIVE, re-solve from the current
    # bound every step and apply u[0]. 0 flies the committed plan open loop.
    receding = os.environ.get('TTTCARV_RECEDING', '1') != '0'
    # TTTCARV_PSF_TAIL=1 (final run): if a re-solve fails, fly the committed
    # plan without extending it while it is within its N-step horizon, the part
    # its terminal constraint covers. Off by default.
    psf_tail = os.environ.get('TTTCARV_PSF_TAIL', '0') != '0'
    mpc_resolved             = 0  # active timesteps whose input came from a fresh solve
    mpc_resolve_failed       = 0  # active timesteps that fell back to the committed plan
    mpc_fallback_uncertified = 0  # fallbacks where the current RSOA left traj_bounds[idx]
    mpc_tail_exhausted       = 0  # k - k_bar reached N: certificate spent, branch 3 due
    tail_recheck_pass        = 0  # failed re-solve; rest of plan re-certified from current RSOA
    tail_recheck_fail        = 0  # failed re-solve; re-check failed too
    tail_recheck_loiter      = 0  # failed re-check -> loiter applied
    tail_loiter_available    = 0  # failed re-check where a loiter circle clears the obstacles
    tail_recheck_applied     = 0  # failed re-check -> plan applied anyway
    tail_recheck_reject: dict = {}  # failed re-checks by kind (path_raw / term_D / empty)
    # Why re-solves failed (e.g. no_budget, rollout_rejected, solver_error).
    mpc_fail_reason: dict = {}
    # Which acceptance check rejected a re-solved plan, and at which node.
    mpc_reject_detail: dict = {}
    # Turn-direction flip diagnostics: does the controller keep switching which
    # way it goes around an obstacle?
    #   applied_u_*  sign of the applied input on consecutive active steps
    #   plan_*       sign of a new plan's mean vs the old plan's remaining tail
    # *_pairs counts comparisons where both values are outside FLIP_DEADBAND.
    applied_u_flips    = 0
    applied_u_pairs    = 0
    plan_flips         = 0
    plan_pairs         = 0
    _prev_applied_u    = None
    _prev_applied_t    = None
    # Active-phase backup builds skipped for lack of budget. Also flags a worker
    # whose mpc_cost calibration disabled the filter (see LAST_RUN_STATS below).
    mpc_skipped_no_budget = 0
    # Subset skipped only to keep RESOLVE_RESERVE.
    mpc_build_yielded_to_resolve = 0
    print(f"[alg22] receding-horizon active phase: {'ON' if receding else 'OFF (alg21 parity)'}"
          f"   first-step-only cost: {'ON' if first_step_cost else 'OFF'}"
          f"{f' (tail_w={tail_cost_weight:g})' if first_step_cost else ''}"
          f"   bounded tail (Alg1 branch 2): {'ON' if psf_tail else 'OFF'}")

    # ── RSOA certificate dump (only if TTTCARV_RSOA_DUMP is a path) ─────────
    # Every step goes through real_state_empirical or real_state_mpc, so wrapping
    # those two logs every step. Bounds are recorded before the step is applied.
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
                    # traj_bounds[k] is the plan's bound at committed_at + k; it
                    # may be missing if the plan was extended.
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

    # ── Per-timestep budget ledger (only if TTTCARV_BUDGET_DUMP is a path) ──
    # Records time spent on symbolic, concrete, plant-step and MPC work each
    # timestep, so steps can be scored with symbolic time taken out.
    _budget_path = os.environ.get('TTTCARV_BUDGET_DUMP')
    if _budget_path:
        budget.ledger_enabled       = True
        tester.symbolic             = budget.charged(tester.symbolic,   'symbolic')
        tester.concrete             = budget.charged(tester.concrete,   'concrete')
        tester.real_state_empirical = budget.charged(tester.real_state_empirical, 'step')
        tester.real_state_mpc       = budget.charged(tester.real_state_mpc,       'step')
        mpc_sf._run_mpc_from_bounds = budget.charged(
            mpc_sf._run_mpc_from_bounds, 'mpc')

    # ── Measured-cost budget (TTTCARV_MPC_PRICING=1, final run) ───────────
    # Off: an MPC solve runs if remaining >= mpc_cost from the one-time
    # calibration. On:
    #   - mpc_cost      = p99.5 of this process's recent solve times + margin
    #   - concrete_cost = p95 of measured seconds per concrete step + margin
    #   - every solve (including the second multi-start one) must fit before it starts
    #   - the RTI loop, re-sim rollout and P2 concrete scan stop at the deadline
    mpc_pricing = os.environ.get('TTTCARV_MPC_PRICING', '0') != '0'
    if mpc_pricing:
        import time_budget as _tbm
        budget.pricing        = True
        budget.margin         = float(os.environ.get('TTTCARV_BUDGET_MARGIN', '0.003'))
        mpc_sf.solve_observer = _tbm.record_solve
        mpc_sf.solve_gate     = lambda: budget.remaining >= budget.mpc_cost
        mpc_sf.time_left      = lambda: budget.remaining - budget.margin
    globals()['_CONCRETE_OBSERVER'] = _tbm.record_concrete if mpc_pricing else None
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

            # The robot has left the nominal trajectory, so future bounds and the
            # backups built from them are stale: drop them. Then alternate one
            # concrete step and one backup build while budget allows; a valid
            # build at t+1 means we can revert to nominal.

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)

            # Keep the horizon counters consistent with the purge.
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            # Budget held back for the re-solve (only with RESOLVE_RESERVE).
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
                # One backup build. The first one (at t+1) is the revert test,
                # so resolve_reserve never holds it back; later builds are
                # lookahead for after the revert.
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
                    # Running out of bounds is the normal exit; only count budget shortfalls.
                    if concrete_until in tester.horizons:
                        mpc_skipped_no_budget += 1
                        # Affordable on its own, skipped only to keep resolve_reserve.
                        if budget.remaining >= budget.mpc_cost:
                            mpc_build_yielded_to_resolve += 1
                    break  # no MPC budget remaining

            
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # Revert: a valid backup at t+1 means nominal is safe again.
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None

                # The trajectory changed, so restart symbolic refinement from t+1
                # (conflict_time is kept).
                pending_job = (RefinementTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                # Drop infeasible entries past t+1; valid ones stay as lookahead.
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep,
                                                       below=conflict_time)
                if t_diverge is None or t_diverge < t_next:
                    # conflict_time was computed for the old trajectory, so it must
                    # not exclude the backup at t+1 that was just verified.
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

            # ── Receding horizon: re-solve from the current bound, apply u[0] ──
            # The committed plan's bounds come from an older RSOA; re-solving keeps
            # the applied input backed by the set the robot is actually in.
            resolved = False
            if receding:
                ok, r_ctrls, r_traj, r_label = solve_plan_at(
                    mpc_sf, tester, current_timestep, budget)
                if (not ok and r_label != "no_budget"
                        and budget.remaining >= budget.mpc_cost):
                    # Single-start failed (not for budget): retry with multi-start.
                    ok, r_ctrls, r_traj, r_label = solve_plan_at(
                        mpc_sf, tester, current_timestep, budget, multi_start=True)
                if ok:
                    # Compare with the old plan's tail before overwriting it.
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
                    # Re-commit at the current step, so ctrl_idx is 0.
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
                # No fresh plan: keep flying the committed plan, extending it when
                # <= n_horizon controls remain. With PSF_TAIL, don't extend while
                # ctrl_idx < n_horizon (still covered by the plan's own terminal
                # constraint); past that, extend anyway and count it as
                # mpc_tail_exhausted.
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
                        # Should not happen.
                        print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                        break

                # The fallback input is only certified if the current RSOA lies
                # inside the bound the plan predicted for this step. Apply it
                # anyway, but count it.
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

                # alg23: re-certify the rest of the plan from where the robot is.
                if receding and TAIL_RECHECK:
                    _now = _bounds_at(tester, current_timestep)
                    _ok, _kind, _node, _n = recheck_tail(
                        mpc_sf, tester, queue[ctrl_idx:], _now)
                    if _ok:
                        tail_recheck_pass += 1
                    else:
                        tail_recheck_fail += 1
                        tail_recheck_reject[_kind] = tail_recheck_reject.get(_kind, 0) + 1
                        _u_loiter = None
                        if PSF_LOITER and _now is not None:
                            for _u in loiter_turn_order(_now, obstacles, u_max):
                                if loiter_circle_ok(mpc_sf, tester, _now, _u):
                                    _u_loiter = np.array([_u])
                                    break
                        if _u_loiter is not None:
                            tail_loiter_available += 1
                        _clears_D = (_now is not None
                                     and not collides_danger(_now, obstacles, turning_radius))
                        print(f"{RED}[TAIL RECHECK] t={current_timestep} rest of plan"
                              f" from idx={ctrl_idx} fails from current RSOA"
                              f" ({_kind}@{_node}/{_n})  clears_D={_clears_D}"
                              f"  loiter={None if _u_loiter is None else float(_u_loiter[0])}"
                              f"  (seed={seed}){RESET}")
                        if TAIL_RECHECK_ACT and _u_loiter is not None:
                            tail_recheck_loiter += 1
                            u_nn = _get_nn_control(tester, current_timestep)
                            u_diffs.append(float(np.abs(_u_loiter[0] - u_nn[0]))
                                           if u_nn is not None else 0.0)
                            tester.real_state_mpc(current_timestep, _u_loiter)
                            current_timestep += 1
                            continue
                        tail_recheck_applied += 1

            ctrl = queue[ctrl_idx]
            # Count flips only across consecutive active steps; a gap means
            # nominal ran in between.
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

        # Already past conflict_time without incident: clear the stale conflict.
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

        # ---- P1: make sure mpc_buffer[t+1] exists (the safety check) ----

        # No backup can start at or past conflict_time: the RSOA there may already
        # overlap the obstacle, and build_mpc_backup doesn't check the start node.
        # Record the reason instead of building.
        past_wall = conflict_time is not None and t_next >= conflict_time
        if past_wall and t_next not in mpc_reason:
            mpc_reason[t_next] = (f"conflict_wall(t+1={t_next} >= conflict_time="
                                  f"{conflict_time}; RSOA there is not raw-clear)")

        # Build the backup at t+1 if it is missing.
        if t_next not in mpc_buffer and not past_wall:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            # Is there a later valid backup to fall back on if this build fails?
            _have_fallback = (t_diverge is not None
                              and t_diverge > current_timestep
                              and psf_valid(mpc_buffer, t_diverge))
            _afford = budget.remaining >= budget.mpc_cost
            if t_next not in tester.horizons:
                p1_skipped_no_bounds += 1
            elif _afford:
                # No fallback: use the slower re-sim solver, which fails less often
                # (BOOTSTRAP_RESIM). Restore the setting even if the build raises.
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
                # Diagnostics: failed builds with no fallback, and at t=0.
                if not _have_fallback and not valid:
                    p1_bootstrap_infeasible += 1
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

        # ---- P2: concrete scan forward, up to conflict_time ----
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
                # A t_diverge set before this conflict was known may be at or
                # past it. Nominal is only verified before conflict_time, so
                # pull t_diverge back below it.
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
            if reached < end:  # stopped early at the deadline
                concrete_deadline_hits += 1
                break

        # ---- P3: build backups forward toward conflict_time ----
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

        # ---- P3b: passthrough scan past an infeasible build (a "wall") ----
        # One infeasible tau says little about the next, so look further ahead.
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

        # ---- P4: symbolic refinement (keeps going after a confirmed conflict) ----
        if (pending_job is not None
                and budget.can_afford('symbolic', 1)):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    # Conflict confirmed with tighter bounds: drop infeasible
                    # entries so P3 retries them (valid entries stay).
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    # Pull mpc_horizon_until back to t_diverge so P3 re-scans the
                    # purged gap with tightened bounds on the next iteration.
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    # Reset job for continuous refinement
                    pending_job       = RefinementTask(current_timestep, conflict_time)
                else:
                    # Deconflicted: clear the conflict but keep valid backups.
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION: nominal or activate MPC backup
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            # PSF OK: nominal is safe. Record t+1 as a divergence point (only ever
            # raising t_diverge) so the clock never overtakes it; otherwise a
            # later activation could join a plan partway through.
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
            # Flying nominal until then is safe only because the concrete scan
            # showed every step before conflict_time is obstacle-free.
            assert conflict_time is None or t_diverge < conflict_time, (
                f"PASSTHROUGH gap [{current_timestep},{t_diverge}] spans the "
                f"conflict at t={conflict_time}: nominal is not concrete-certified "
                f"over this gap")
            # Log why there is no backup at t+1.
            if t_next in mpc_reason:
                _why = f"no backup at t+1: {mpc_reason[t_next]}"
            elif t_next not in mpc_buffer:
                _why = "no backup at t+1: never built (budget or bounds missing)"
            else:
                _why = "no backup at t+1: reason unrecorded"

            # Log what makes nominal safe this step.
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
            # PSF failed: switch to a backup plan.
            # A plan only describes the robot from the step it was built for
            # (controls[k] assumes controls[0..k-1] were applied), so use the
            # backup for this exact step and refuse a stale t_diverge.
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
                # No usable backup at all (should not happen in a healthy run).
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
                # Terminal safe controller (see PSF_LOITER). Uses the CURRENT RSOA;
                # the next-step RSOA assumes NN control, not the loiter.
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
        'tail_recheck':             int(TAIL_RECHECK),
        'tail_recheck_act':         int(TAIL_RECHECK_ACT),
        'tail_recheck_pass':        tail_recheck_pass,
        'tail_recheck_fail':        tail_recheck_fail,
        'tail_recheck_loiter':      tail_recheck_loiter,
        'tail_loiter_available':    tail_loiter_available,
        'tail_recheck_applied':     tail_recheck_applied,
        **{f'trej_{k}': v for k, v in tail_recheck_reject.items()},
        'single_start_frac':        SINGLE_START_COST_FRAC,
        'applied_u_flips':          applied_u_flips,
        'applied_u_pairs':          applied_u_pairs,
        'plan_flips':               plan_flips,
        'plan_pairs':               plan_pairs,
        # mpc_cost_clamped=1: this process's calibration was clamped. A trial
        # with mpc_calls==0 and mpc_skipped_no_budget>0 flew unfiltered and
        # should not be scored.
        'mpc_cost_clamped':         mpc_cost_clamped,
        'mpc_skipped_no_budget':    mpc_skipped_no_budget,
        'mpc_build_yielded':        mpc_build_yielded_to_resolve,
        # Unfiltered steps where the filter wanted to intervene (should be 0), by cause.
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
        # Size of the intervention |u_mpc - u_nn| on active steps.
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

