"""
Algorithm 12 — Predictive Safety Filter (PSF) with Forward MPC Buffer

Safety principle (PSF): apply nominal control at time t if and only if
mpc_buffer[t+1] contains a feasible, collision-free MPC trajectory.
Otherwise, activate the precomputed backup from t_diverge — the last τ
where the buffer was valid.

Key differences from alg11:
  - No backward search (find_stopping_timestep removed).
  - Forward buffer: build_mpc_backup fills mpc_buffer[τ] proactively.
  - t_diverge updated lazily as the forward buffer loop succeeds/fails.
  - Concrete propagation stops at conflict_time (no point going further).
  - Symbolic runs continuously even after confirmation — tighter RSOA
    bounds may push t_diverge later or deconflict entirely.
  - Reversion: re-check PSF condition from MPC-propagated state each step.

States:
  NOMINAL      — no conflict; PSF buffer maintained as 1-step lookahead.
  PRE_CONFLICT — conflict known; buffer fills toward it; nominal applied
                 while mpc_buffer[t+1] is valid.
  MPC_ACTIVE   — PSF condition failed; executing precomputed backup;
                 revert when mpc_buffer[t+1] becomes valid again.

Budget priorities (each sim timestep, outside MPC_ACTIVE):
  P1 — Ensure mpc_buffer[t+1] exists (safety-critical PSF check).
  P2 — Concrete scan forward (feeds MPC buffer with real bounds).
  P3 — MPC buffer forward toward conflict_time (lazily find t_diverge).
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

def _get_nn_control(tester, timestep):
    """Query NN nominal control at the real state stored in horizons[timestep]."""
    h = tester.horizons.get(timestep)
    if h is None:
        return None
    for calc_data in h.calculations.values():
        if 'real_state' in calc_data:
            state  = np.asarray(calc_data['real_state']).reshape(1, -1)
            cl_sys = tester.analyzer.cl_system
            u      = cl_sys.dynamics.control_nn(state, cl_sys.controller.cpu())
            return np.asarray(u).flatten()
    return None


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
    if result is False:
        return False, None
    if result["collision"]:
        return True, result["collision_timestep"]
    return False, None


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
    Identical to alg11.
    """
    committed_at = mpc_state['committed_at']
    controls     = mpc_state['controls']
    traj_bounds  = mpc_state['traj_bounds']
    n_controls   = len(controls)
    mpc_end      = committed_at + n_controls

    if mpc_end >= max_time:
        print(f"  [MPC extend] Reached MAX_TIME, done")
        return

    max_lb = min(n_controls - ctrl_idx - 1, mpc_sf.n_horizon)
    max_lb = max(max_lb, 0)

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


# ── PSF-specific helpers ───────────────────────────────────────────────────

def psf_valid(mpc_buffer: dict, t: int):
    """ Returns true if mpc_buffer at time t exists and is collision free."""
    return t in mpc_buffer and mpc_buffer[t] is not None


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

# ── Helpers ────────────────────────────────────────────────────

def collides_invariant(bounds: np.ndarray, obstacles: list) -> bool:
    """
    Circle-box overlap test against the invariant safe set.
    Calculate Invariant safe set from obstacle list. Assume v = 1, umax = 1.
    """


    for obs in obstacles:
        r = obs[2]
        inf_r = np.sqrt(r**2 + 2*r*1)

        cx, cy, r = obs[0], obs[1], inf_r
        closest_x = np.clip(cx, bounds[0, 0], bounds[0, 1])
        closest_y = np.clip(cy, bounds[1, 0], bounds[1, 1])
        if (cx - closest_x) ** 2 + (cy - closest_y) ** 2 <= r ** 2:
            return True
    return False

# ── Simulation loop ────────────────────────────────────────────────────────

def test(seed=None, analyzer=None, obstacles=None):
    if analyzer is None:
        analyzer = setup_analyzer('Unicycle_NL', 'natural_none_default')

    if seed is None:
        seed = 1401830092

    if obstacles is None:
        obstacles = [
            np.array([-6.5,  2.02,  0.5]),
            np.array([-3.2,  1.21,  0.5]),
            np.array([-2.0, -0.3,  0.45]),
            np.array([-2.0, -1.3,  0.5]),
        ]

    tester = ReachabilityTester(analyzer, obstacles, seed=seed)
    mpc_sf = make_mpc_safety_filter(tester, obstacles_list=obstacles, n_horizon=10, use_safety_radius=True)

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
    mpc_buffer:        dict                      = {}   # τ → (controls, traj_bounds) | None
    concrete_until:    int                       = 0
    mpc_horizon_until: int                       = -1   # last τ MPC was attempted
    conflict_time:     Optional[int]             = None
    t_diverge:         Optional[int]             = None
    pending_job:       Optional[VerificationTask] = None
    mpc_wall_found:    bool                      = False  # True once P3 hits first INFEASIBLE

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
    u_diffs          = []
    mpc_calls        = 0   # number of PSF activations
    mpc_over_budget  = 0   # timesteps where elapsed > timestep_budget

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next = current_timestep + 1

        # ══════════════════════════════════════════════════════════════════
        #  MPC ACTIVE PHASE
        # ══════════════════════════════════════════════════════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}  conflict={conflict_time}"
                  f"  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']

            # Extend queue whenever ≤ n_horizon controls remain (mirrors alg11)
            if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']

            # Clear future horizons; then interleave: one concrete step →
            # one MPC build, repeat while budget allows.
            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
            concrete_until = current_timestep
            _first = True
            while _first or budget.can_afford('concrete'):
                _first = False
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
                        break  # infeasible wall — no point going further
                else:
                    break  # no MPC budget remaining

            # Build forward buffer with remaining budget so reversion has lookahead
            mpc_horizon_until = max(mpc_horizon_until, t_next)
            # PSF reversion: if backup from t+1 is now valid, return to nominal.
            # Keep mpc_buffer — entries built from MPC-propagated bounds are still
            # valid and give immediate lookahead after reversion.
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None
                # Keep conflict_time and pending_job — conflict is still real;
                # only P4 symbolic deconfliction should clear them.
                # Only clear INFEASIBLE (None) entries beyond t_next. VALID entries
                # were built from nominal-trajectory concrete bounds (concrete_scan
                # always uses nominal dynamics) so they remain valid safety backups
                # now that we are reverting to nominal control.
                for _k in list(mpc_buffer.keys()):
                    if _k > t_next and mpc_buffer[_k] is None:
                        del mpc_buffer[_k]
                valid_keys = [k for k, v in mpc_buffer.items()
                              if k >= current_timestep and v is not None]
                t_diverge         = max(valid_keys) if valid_keys else None
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            # Apply queued MPC control (extend above guarantees ctrl_idx < len(queue))
            if ctrl_idx >= len(queue):
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

        # ══════════════════════════════════════════════════════════════════
        #  NOMINAL / PRE_CONFLICT PHASE
        # ══════════════════════════════════════════════════════════════════

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

        # ── P1: Ensure mpc_buffer[t+1] exists (critical for PSF check) ───
        if t_next not in mpc_buffer:
            # Ensure concrete bounds exist at t+1
            if t_next not in tester.horizons:
                concrete_scan(tester, concrete_until, t_next)
                concrete_until = max(concrete_until, t_next)
            if t_next in tester.horizons and budget.remaining >= budget.mpc_cost:
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, t_next)
                mpc_horizon_until = max(mpc_horizon_until, t_next)
                if valid:
                    t_diverge = t_next
                print(f"  [P1] mpc_buffer[{t_next}] = {'VALID' if valid else 'INFEASIBLE'}"
                      f"  t_diverge={t_diverge}")

        # ── P2: Concrete scan forward (stop at conflict_time) ─────────────
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

        # ── P3: MPC buffer forward toward conflict_time ───────────────────
        if conflict_time is not None:
            # Ensure mpc_horizon_until is at least current_timestep so we
            # don't waste budget re-building stale past entries.
            mpc_horizon_until = max(mpc_horizon_until, current_timestep)
            while (mpc_horizon_until < conflict_time - 1
                   and budget.remaining >= budget.mpc_cost):
                tau = mpc_horizon_until + 1
                if tau not in tester.horizons:
                    break
                if mpc_buffer.get(mpc_horizon_until) is None:
                    break  # last τ was infeasible — wall reached, don't go further
                valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, tau)
                mpc_horizon_until = tau
                if valid:
                    t_diverge = tau
                    print(f"  [P3] mpc_buffer[{tau}] VALID  t_diverge → {t_diverge}")
                else:
                    print(f"  [P3] mpc_buffer[{tau}] INFEASIBLE — stopping buffer build")
                    break

        # ── P4: Symbolic — continuous even after confirmation ─────────────
        if (pending_job is not None
                and budget.can_afford('symbolic', 1)):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    # Confirmed or re-confirmed — refresh buffer with tighter bounds
                    # Symbolic tightened bounds — drop only INFEASIBLE entries so
                    # P3 retries them with tighter bounds (may extend t_diverge).
                    # VALID entries remain safe; no need to throw them away.
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    for k in list(mpc_buffer.keys()):
                        if k > current_timestep and mpc_buffer[k] is None:
                            del mpc_buffer[k]
                    # Recalculate t_diverge from remaining valid entries.
                    # Include k == current_timestep: backup can still activate NOW (ctrl_idx=0).
                    valid_keys = [k for k, v in mpc_buffer.items() if k >= current_timestep and v is not None]
                    t_diverge = max(valid_keys) if valid_keys else None
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    # Reset job for continuous refinement
                    pending_job = VerificationTask(current_timestep, conflict_time)
                else:
                    # Deconflicted — clear conflict state but keep VALID buffer
                    # entries; they're still collision-free and serve as backups
                    # if a new conflict appears immediately after.
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    for k in list(mpc_buffer.keys()):
                        if k > current_timestep and mpc_buffer[k] is None:
                            del mpc_buffer[k]
                    valid_keys = [k for k, v in mpc_buffer.items() if k >= current_timestep and v is not None]
                    t_diverge = max(valid_keys) if valid_keys else None
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION: nominal or activate MPC backup
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            # PSF satisfied — nominal control is safe
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
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
                        u_diffs.append(
                            float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                        tester.real_state_mpc(current_timestep, ctrl)
                        current_timestep += 1
                    else:
                        print(f"[PSF] Cannot activate — queue still empty. Nominal fallback.")
                        u_diffs.append(0.0)
                        tester.real_state_empirical(current_timestep, t_next)
                        current_timestep += 1
            else:
                # No valid backup yet — mpc_buffer[t+1] not ready (P1 stalled)
                # Apply nominal and warn; P1 will catch up next timestep.
                print(f"{RED}[PSF] No backup available at t={current_timestep}"
                      f" (t_diverge={t_diverge}) — nominal fallback{RESET}")
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1

        if budget.elapsed > budget.timestep_budget:
            mpc_over_budget += 1

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
    return state_history, had_collision, u_diffs, mpc_calls, mpc_over_budget


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=seed)
