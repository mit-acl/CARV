"""
Algorithm 12 — Predictive Safety Filter (PSF) with Forward MPC Buffer

States:
  NOMINAL      — no conflict found. PSF buffer maintained as 1 step lookahead.
  PRE_CONFLICT — RSOA collides with obstacle. PSF buffer filled towards it. nominal applied
                 while mpc_buffer[t+1] is valid.
  MPC_ACTIVE   — PSF condition failed; executing precomputed backup;
                 revert when mpc_buffer[t+1] becomes valid again.

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
from time_budget import TimeBudget
import numpy as np
import time
from typing import Optional
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter

RED     = "\033[31m"
GREEN   = "\033[32m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
RESET   = "\033[0m"


# ── Shared helpers (identical to alg11) ────────────────────────────────────


class VerificationTask:
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


def symbolic_step(tester, job: VerificationTask, chunk_size: int, budget):
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

        safe_count = 0
        for b in new_traj_bounds[1:]:
            if mpc_sf._collides(b):
                break
            safe_count += 1

        if safe_count <= lb:
            print(f"  [MPC extend] lookback={lb}: only {safe_count} safe, going further back")
            continue

        keep = n_controls - lb
        mpc_state['controls']    = list(controls[:keep]) + list(new_controls[:safe_count])
        mpc_state['traj_bounds'] = (list(traj_bounds[:try_idx + 1])
                                    + list(new_traj_bounds[1:safe_count + 1]))

        net_gain  = safe_count - lb
        new_total = len(mpc_state['controls'])
        new_end   = committed_at + new_total
        if lb > 0:
            print(f"  [MPC extend] lookback={lb}: +{safe_count} safe, -{lb} replaced, net +{net_gain}")
        print(f"  [MPC extend] +{net_gain} controls → {new_total} total "
              f"(t={committed_at} to t={new_end})")
        return

    print(f"  [MPC extend] No safe extension found after {max_lb + 1} lookback attempts")


# PSF helpers

def psf_valid(mpc_buffer: dict, t: int):
    """ Returns true if mpc_buffer at time t exists and is collision free."""
    return mpc_buffer.get(t) is not None


def build_mpc_backup(mpc_sf, tester, mpc_buffer: dict, tau: int):
    """
    Solve MPC backup starting from concrete bounds at timestep tau.
    Stores (controls, traj_bounds) in mpc_buffer[tau] if collision-free,
    else None. Returns True if a valid backup was found.
    """
    h = tester.horizons.get(tau)
    if h is None:
        return False
    bounds = h.get_tight_bound()
    center = (bounds[:, 0] + bounds[:, 1]) / 2.0
    try:
        _t0 = time.time()
        traj_bounds, _, controls = mpc_sf._run_mpc_from_bounds(bounds, center)
        elapsed = time.time() - _t0
        collision = any(mpc_sf._collides(b) for b in traj_bounds[1:])
        status = "INFEASIBLE(collision)" if collision else "VALID"
        print(f"  [MPC build] τ={tau}  {status}  took {elapsed:.3f}s")
        if not collision:
            mpc_buffer[tau] = (list(controls), list(traj_bounds))
            return True
        mpc_buffer[tau] = None
        return False
    except Exception as e:
        print(f"  [PSF buf] MPC solve failed at τ={tau}: {e}")
        mpc_buffer[tau] = None
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


def collides_invariant(bounds: np.ndarray, obstacles: list,
                       R: float = 1.0) -> bool:
    """Overlap against the invariant safe set S = sqrt(r^2 + 2rR).
    r is radius of obstacle, R is curvature radius. R = 1 for current dynamics
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
                     R: float = 1.0) -> Optional[int]:
    """
    Scan beyond an MPC INFEASIBLE "wall" to find a window otuside of invariant set S.
    Returns new t_diverge if a valid MPC backup is found outside S, else None.

    scan limit is either
    conflict_time: no point scanning past the collision timestep
    furthest concrete rsoa + 1: RSOA bounds dont exist beyond this
    """
    for tau in range(wall_tau + 1, scan_limit):
        if tau not in tester.horizons:
            break
        bounds = tester.horizons[tau].get_tight_bound()
        if bounds is None:
            break
        if collides_raw(bounds, obstacles):
            print(f"  [Scan window] tau={tau} collides with raw obstacle — abort")
            return None
        if not collides_invariant(bounds, obstacles, R):
            if tau in mpc_buffer:
                if mpc_buffer[tau] is not None:
                    print(f"  [Scan window] tau={tau} already VALID in buffer")
                    return tau
                continue
            if budget.remaining < budget.mpc_cost:
                print(f"  [Scan window] Budget exhausted at tau={tau}")
                return None
            valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau)
            if valid:
                print(f"  [Scan window] tau={tau} VALID — passthrough t_diverge found")
                return tau
            else:
                print(f"  [Scan window] tau={tau} outside S but MPC INFEASIBLE — continue")
    return None

def _purge_infeasible(mpc_buffer: dict, after_t: int):
    """Delete INFEASIBLE (None) entries with key > after_t."""
    for k in list(mpc_buffer):
        if k > after_t and mpc_buffer[k] is None:
            del mpc_buffer[k]

def _max_valid_diverge(mpc_buffer: dict, from_t: int) -> Optional[int]:
    """Return the largest key >= from_t with a valid (non-None) buffer entry."""
    keys = [k for k, v in mpc_buffer.items() if k >= from_t and v is not None]
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
    mpc_sf = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=12, use_safety_radius=True)

    v_nom = tester.analyzer.cl_system.dynamics.vt
    u_max = 1.0
    turning_radius = v_nom / u_max

    MAX_SYMBOLIC_HORIZON = 5
    MAX_TIME             = 60

    budget = TimeBudget(timestep_budget=0.20)
    budget.symbolic_costs = {
        1:  0.05942702293395996,  2: 0.0532071590423584,
        3:  0.12308859825134277,  4: 0.2227306365966797,
        5:  0.3548123836517334,   6: 0.5160810947418213,
        7:  0.7076215744018555,   8: 1.046485185623169,
        9:  1.189185619354248,   10: 1.4745268821716309,
    }
    budget.concrete_cost = 0.0142

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    # ── PSF buffer state ───────────────────────────────────────────────────
    mpc_buffer:        dict                      = {}   # tau -> (controls, traj_bounds) | None
    concrete_until:    int                       = 0
    mpc_horizon_until: int                       = -1   # last tau MPC was attempted
    conflict_time:     Optional[int]             = None
    t_diverge:         Optional[int]             = None
    pending_job:       Optional[VerificationTask] = None
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

            # Extend queue whenever <= n_horizon mpc controls remain (mirrors alg11)
            if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']

            # Clear future horizons; then interleave: one concrete step -> one MPC build, repeat while budget allows.
            # This is to check if we can revert to nominal.
            # If we apply concrete now and mpc build is valid in the next timestep, we are safe to revert

            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
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
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until)
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
                pending_job = (VerificationTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
                # were built in section above and remain valid backup buffers
                _purge_infeasible(mpc_buffer, t_next)
                t_diverge         = _max_valid_diverge(mpc_buffer, current_timestep)
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            # Apply queued MPC control
            if ctrl_idx >= len(queue):
                #shouldnt run if everything is working
                print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                break

            ctrl = queue[ctrl_idx]
            print(f"{MAGENTA}[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                  f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s{RESET}")
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

        # Skip P1 when we are inside the S-region passthrough.
        # MPC will fail anyway if we are inside S (S is a hard constraint)
        # Skip condition: only skip when t_diverge is ahead (passthrough active) AND
        # bounds at t+1 are still inside S.  Once the robot exits S,
        # P1 resumes so the RSOA check stays tight.
        in_passthrough = (t_diverge is not None and t_diverge > t_next)
        skip_p1 = False
        if in_passthrough and t_next in tester.horizons:
            _bound = tester.horizons[t_next].get_tight_bound()
            if _bound is not None and collides_invariant(_bound, obstacles, turning_radius):
                skip_p1 = True

        # Normal P1 operations
        if t_next not in mpc_buffer and not skip_p1:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next)
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
                if conflict_time is None or ct < conflict_time:
                    conflict_time = ct
                    scan_ceil     = ct
                    print(f"  [P2] Conflict detected at t={conflict_time}")
                    if pending_job is None:
                        pending_job = VerificationTask(current_timestep, conflict_time)
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
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau)
                mpc_horizon_until = tau
                if valid:
                    t_diverge = tau
                    print(f"  [P3] mpc_buffer[{tau}] VALID  t_diverge → {t_diverge}")
                else:
                    print(f"  [P3] mpc_buffer[{tau}] INFEASIBLE — wall at tau={tau}")
                    wall_tau = tau
                    break

        #  P3b: Passthrough scan beyond infeasible wall
        # Only invoke when an S-region exists ahead (bounds overlap S but not raw
        # obstacle)
        if wall_tau is not None and budget.remaining >= budget.mpc_cost:
            scan_limit = (min(conflict_time, concrete_until + 1)
                          if conflict_time is not None
                          else concrete_until + 1)
            _has_S_ahead = any(
                b is not None
                and collides_invariant(b, obstacles, turning_radius)
                and not collides_raw(b, obstacles)
                for t in range(wall_tau, scan_limit)
                for b in [_bounds_at(tester, t)]
            )
            if _has_S_ahead:
                passthrough_td = scan_window(
                    tester, mpc_sf, mpc_buffer, obstacles,
                    wall_tau, scan_limit, budget, R=turning_radius)
                if passthrough_td is not None:
                    t_diverge = passthrough_td
                    mpc_horizon_until = passthrough_td
                    print(f"  [P3b] Passthrough t_diverge -> {t_diverge}")
            else:
                print(f"  [P3b] No S-region ahead of wall_tau={wall_tau} — skipping scan")

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
                    pending_job       = VerificationTask(current_timestep, conflict_time)
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
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

        elif (t_diverge is not None and t_diverge > current_timestep
                and psf_valid(mpc_buffer, t_diverge)):
            # Passthrough: t_diverge is ahead — we're inside S but
            # scan_window already verified no raw collision in the gap.
            print(f"  [PSF PASSTHROUGH] t={current_timestep} inside S, "
                  f"t_diverge={t_diverge} ahead — nominal safe"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            tester.real_state_empirical(current_timestep, t_next)
            current_timestep += 1

        else:
            # PSF fails — need MPC backup from t_diverge
            if t_diverge is not None and psf_valid(mpc_buffer, t_diverge):
                controls, traj_bounds = mpc_buffer[t_diverge]
                mpc_state['committed_at']  = t_diverge
                mpc_state['conflict_time'] = conflict_time
                mpc_state['controls']      = list(controls)
                mpc_state['traj_bounds']   = list(traj_bounds)
                mpc_state['needed']        = True
                mpc_started                = True

                ctrl_idx = current_timestep - t_diverge
                queue    = mpc_state['controls']
                if ctrl_idx < len(queue):
                    ctrl = queue[ctrl_idx]
                    print(f"{RED}[PSF ACTIVATE] t={current_timestep}  PSF failed —"
                          f" backup from t_diverge={t_diverge}"
                          f"  ctrl_idx={ctrl_idx}  u={np.round(ctrl, 4)}{RESET}")
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
                        tester.real_state_mpc(current_timestep, ctrl)
                        current_timestep += 1
                    else:
                        print(f"[PSF] Cannot activate — queue still empty. Nominal fallback.")
                        tester.real_state_empirical(current_timestep, t_next)
                        current_timestep += 1
            else:
                # No valid backup yet — mpc_buffer[t+1] not ready (P1 stalled)
                # Apply nominal and warn; P1 will catch up next timestep.
                # SHOULD NOT RUN IF WORKING!!!!!
                print(f"{RED}[PSF] No backup available at t={current_timestep}"
                      f" (t_diverge={t_diverge}) — nominal fallback{RESET}")
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1


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
    return state_history, had_collision

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=seed)
