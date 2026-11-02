"""
Algorithm 14 — Predictive Safety Filter (PSF) with Forward MPC Buffer

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


def concrete_scan(tester, from_t, to_t):
    """Run concrete propagation; return (collision: bool, collision_t: int|None)."""
    if from_t >= to_t:
        return False, None
    result = tester.concrete(from_t, to_t)
    if not result or not result["collision"]:
        return False, None
    return True, result["collision_timestep"]


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
        collision = (path_hit is not None) or term_hit

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
    Scan beyond an MPC INFEASIBLE "wall" to find a window outside the danger region D.
    Returns new t_diverge if a valid MPC backup is found outside D, else None.

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
        if not collides_danger(bounds, obstacles, R):
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
                print(f"  [Scan window] tau={tau} outside D but MPC INFEASIBLE — continue")
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
    mpc_sf = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=12,
                                    verify_input_bounds=True,
                                    use_safety_radius=True, split_terminal_D=True)

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
    psf_queue_empty  = 0    # PSF failed, t_diverge valid but queue empty after extend
    psf_stale_refused = 0   # PSF failed, only a stale (ctrl_idx>0) plan existed
    passthrough_found = 0   # P3b scan_window returned a τ outside D after a wall
    passthrough_used  = 0   # PSF PASSTHROUGH branch executed (nominal while in D)

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
                    # Dropping them here is fail-safe by construction: psf_valid
                    # is a presence test, so a missing entry reads as "no
                    # backup", never as a valid one. The rebuild loop below runs
                    # AFTER this purge, so backups re-derived from the actually
                    # flown state survive untouched.
                    mpc_buffer.pop(_t, None)
                    mpc_reason.pop(_t, None)
            # mpc_horizon_until marks how far the buffer was filled. Leaving it
            # past the purge point would make the P2 forward-fill start from a
            # tau we just erased and read the hole as an infeasible wall.
            mpc_horizon_until = min(mpc_horizon_until, current_timestep)
            concrete_until = current_timestep
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                # One concrete step
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                # One MPC build for this timestep
                if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until, mpc_reason)
                    mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                    if valid:
                        t_diverge = concrete_until
                    print(f"  [MPC PSF] τ={concrete_until} {'VALID' if valid else 'INFEASIBLE'}"
                          f"  t_diverge={t_diverge}")
                    if not valid:
                        break  # infeasible wall. no point going further
                else:
                    break  # no MPC budget remaining

            # Build forward buffer with remaining budget so we have lookahead buffer if we revert
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # PSF reversion: if backup from t+1 (from the concrete RSOA) is valid, return to nominal.
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

            # Extend queue whenever <= n_horizon controls remain.
            # Placed after revert check: no point extending a queue we're about to abandon.
            if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']

            # Apply queued MPC control
            if ctrl_idx >= len(queue):
                #shouldnt run if everything is working
                print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                break

            ctrl = queue[ctrl_idx]
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

        # Skip P1 when we are inside the D-region passthrough.
        # MPC will fail anyway if we are inside D (D is a hard constraint)
        # Skip condition: only skip when t_diverge is ahead (passthrough active) AND
        # bounds at t+1 are still inside D.  Once the robot exits D,
        # P1 resumes so the RSOA check stays tight.
        in_passthrough = (t_diverge is not None and t_diverge > t_next)
        skip_p1 = False
        if in_passthrough and t_next in tester.horizons:
            _bound = tester.horizons[t_next].get_tight_bound()
            if _bound is not None and collides_danger(_bound, obstacles, turning_radius):
                skip_p1 = True

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
        if t_next not in mpc_buffer and not skip_p1 and not past_wall:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next, mpc_reason)
                mpc_horizon_until = max(mpc_horizon_until, t_next)
                if valid and (t_diverge is None or t_next > t_diverge):
                    t_diverge = t_next
                print(f"  [P1] mpc_buffer[{t_next}] = {'VALID' if valid else 'INFEASIBLE'}"
                      f"  t_diverge={t_diverge}")

        # ---- P2: Concrete scan forward (stop at conflict_time) -----
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(
                concrete_until + budget.max_affordable_concrete(),
                scan_ceil, MAX_TIME
            )
            collision, ct = concrete_scan(tester, concrete_until, end)
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
            concrete_until = end

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
        # Only invoke when a D-region exists ahead (bounds overlap D but not raw
        # obstacle)
        if wall_tau is not None and budget.remaining >= budget.mpc_cost:
            scan_limit = (min(conflict_time - 1, concrete_until)
                          if conflict_time is not None
                          else concrete_until)
            _has_D_ahead = any(
                b is not None
                and collides_danger(b, obstacles, turning_radius)
                and not collides_raw(b, obstacles)
                for t in range(wall_tau, scan_limit + 1)
                for b in [_bounds_at(tester, t)]
            )
            if _has_D_ahead:
                passthrough_td = scan_window(
                    tester, mpc_sf, mpc_buffer, obstacles,
                    wall_tau, scan_limit, budget, R=turning_radius,
                    reasons=mpc_reason)
                if passthrough_td is not None:
                    t_diverge = passthrough_td
                    mpc_horizon_until = passthrough_td
                    passthrough_found += 1
                    print(f"  [P3b] Passthrough t_diverge -> {t_diverge}")
            else:
                print(f"  [P3b] No D-region ahead of wall_tau={wall_tau} — skipping scan")

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
            # Report the REAL reason there is no backup at t+1 rather than
            # asserting "inside D". Only the skip_p1 route actually tests D
            # membership (collides_danger on the RSOA at t+1); a build that came
            # back INFEASIBLE(path_raw ...) is a raw-radius violation on the
            # backup's swept path and has nothing to do with D.
            if skip_p1:
                _why = "inside D (P1 skipped: RSOA at t+1 overlaps D)"
            elif t_next in mpc_reason:
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
                psf_no_diverge += 1
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1



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
