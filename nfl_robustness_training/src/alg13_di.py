"""
Algorithm 13

This is a DI-equivalent of alg12_mpc_every_timestep.py:

Changes
  - Use a parabola p >= 0.5 * max(0, v^2) as the danger / uncertainty region S
    (outside S is the invariant safe region)
  - Circular-obstacle / S-region / passthrough machinery is REMOVED;
    half-planes have no S-region to traverse. once robot is inside S region, it cannot get out

"""

import time
import numpy as np
from typing import Optional, List

from REAL_integrated_sim import setup_analyzer, ReachabilityTester
from time_budget import TimeBudget
from mpc_safety_filter_acados import make_mpc_safety_filter_acados as make_mpc_safety_filter

# Reuse the system-agnostic PSF helpers from alg12 verbatim.
from alg12_mpc_every_timestep import (
    VerificationTask, concrete_scan, symbolic_step,
    extend_mpc_sequence, psf_valid, build_mpc_backup,
    _purge_infeasible,
)

RED, GREEN, BLUE, MAGENTA, CYAN, RESET = (
    "\033[31m", "\033[32m", "\033[34m", "\033[35m", "\033[36m", "\033[0m"
)


class HalfPlaneConstraints:
    """Replacement for Obstacles for half-plane state constraints."""

    POS_SENTINEL = np.array([0.0, 0.0, -1.0])   # tag for pos<=pos_min violation
    VEL_SENTINEL = np.array([0.0, 0.0, -2.0])   # tag for vel<=vel_min violation

    def __init__(self, pos_min: float = 0.0, vel_min: float = -1.0):
        self.pos_min = pos_min
        self.vel_min = vel_min
        # Required by some code paths in REAL_integrated_sim
        self.obstacle_list: List[np.ndarray] = []

    def check_collision(self, state: np.ndarray):
        """`state` is a (2,2) bounds array [[p_min,p_max],[v_min,v_max]]."""
        collisions = []
        if state[0, 0] <= self.pos_min:
            collisions.append(self.POS_SENTINEL)
        if state[1, 0] <= self.vel_min:
            collisions.append(self.VEL_SENTINEL)
        if collisions:
            return collisions, []
        return None, None


# ── DI-specific collision helper for the post-loop safety check ───────────

def real_state_violates(state: np.ndarray,
                        pos_min: float = 0.0,
                        vel_min: float = -1.0) -> bool:
    s = np.asarray(state).flatten()
    return bool(s[0] <= pos_min or s[1] <= vel_min)


# ── Simulation loop ───────────────────────────────────────────────────────

def test(seed=None, analyzer=None,
         pos_min: float = 0.0, vel_min: float = -1.0,
         buffer: float = 0.05,
         init_range=None):

    if analyzer is None:
        analyzer = setup_analyzer(
            'DoubleIntegrator',
            'constraint_default_more_data_5hz',
            init_range=init_range,
        )

    if seed is None:
        seed = 1401830092

    # ReachabilityTester needs *some* obstacle list to construct; pass [] and
    # then swap in the half-plane shim.
    tester = ReachabilityTester(analyzer, [], seed=seed)
    tester.obstacles = HalfPlaneConstraints(pos_min=pos_min, vel_min=vel_min)

    mpc_sf = make_mpc_safety_filter(
        tester,
        obstacles_list=[],
        n_horizon=12,
        use_half_plane=True,
        pos_min=pos_min,
        vel_min=vel_min,
        buffer=buffer,
    )

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

    # ── PSF buffer state ─────────────────────────────────────────────────
    mpc_buffer:        dict                       = {}
    concrete_until:    int                        = 0
    mpc_horizon_until: int                        = -1
    conflict_time:     Optional[int]              = None
    t_diverge:         Optional[int]              = None
    pending_job:       Optional[VerificationTask] = None
    wall_tau:          Optional[int]              = None

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
    mpc_calls        = 0
    mpc_over_budget  = 0

    while current_timestep < MAX_TIME:
        budget.start_timestep()
        t_next   = current_timestep + 1
        wall_tau = None

        # ══════════════════════════════════════════════════════════════════
        #  MPC ACTIVE PHASE
        # ══════════════════════════════════════════════════════════════════
        if mpc_started:
            print(f"{MAGENTA}MPC ACTIVE ==== t={current_timestep}"
                  f"  committed_at={mpc_state['committed_at']}  conflict={conflict_time}"
                  f"  t_diverge={t_diverge}{RESET}")
            ctrl_idx = current_timestep - mpc_state['committed_at']
            queue    = mpc_state['controls']

            if len(queue) - ctrl_idx <= mpc_sf.n_horizon:
                extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                    ctrl_idx=ctrl_idx, budget=budget)
                queue = mpc_state['controls']

            # Try to revert: clear forward horizons, then interleave one
            # concrete step + one MPC build until budget is gone.
            for _t in list(tester.horizons.keys()):
                if _t > current_timestep:
                    del tester.horizons[_t]
            concrete_until = current_timestep
            while budget.can_afford('concrete'):
                if concrete_until >= MAX_TIME:
                    break
                collision, _ = concrete_scan(tester, concrete_until, concrete_until + 1)
                concrete_until += 1
                if collision:
                    break
                if concrete_until in tester.horizons and budget.remaining >= budget.mpc_cost:
                    valid = build_mpc_backup(mpc_sf, tester, mpc_buffer, concrete_until)
                    mpc_horizon_until = max(mpc_horizon_until, concrete_until)
                    if valid:
                        t_diverge = concrete_until
                    print(f"  [MPC PSF] τ={concrete_until} {'VALID' if valid else 'INFEASIBLE'}"
                          f"  t_diverge={t_diverge}")
                    if not valid:
                        break
                else:
                    break

            mpc_horizon_until = max(mpc_horizon_until, t_next)
            if psf_valid(mpc_buffer, t_next):
                print(f"{CYAN}[PSF REVERT] t={current_timestep}  mpc_buffer[{t_next}] valid"
                      f"  mpc_hz={mpc_horizon_until} — reverting to nominal{RESET}")
                mpc_started                = False
                mpc_state['needed']        = False
                mpc_state['committed_at']  = None
                mpc_state['conflict_time'] = None

                pending_job = (VerificationTask(t_next, conflict_time)
                               if conflict_time is not None else None)

                _purge_infeasible(mpc_buffer, t_next)
                mpc_horizon_until = t_diverge if t_diverge is not None else t_next
                concrete_until    = t_next
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                print(f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
                current_timestep += 1
                continue

            if ctrl_idx >= len(queue):
                print(f"[MPC] ERROR: queue still exhausted after extend — ABORT")
                break

            ctrl = queue[ctrl_idx]
            print(f"{MAGENTA}[MPC] t={current_timestep}  idx={ctrl_idx}/{len(queue)-1}"
                  f"  u={np.round(ctrl, 4)}  (plan from t={mpc_state['committed_at']})"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s{RESET}")
            u_nn = _get_di_nn_control(tester, current_timestep)
            u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
            tester.real_state_mpc(current_timestep, ctrl)
            current_timestep += 1
            continue

        # ═════════════════════════════════════════════════════════════════
        #  NOMINAL / PRE_CONFLICT PHASE
        # ═════════════════════════════════════════════════════════════════

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

        # ---- P1: ensure mpc_buffer[t+1] exists ---------------------------
        # No passthrough concept for half-planes — always run P1.
        if t_next not in mpc_buffer:
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

        # ---- P2: concrete scan forward (stop at conflict_time) -----------
        scan_ceil = conflict_time if conflict_time is not None else MAX_TIME
        while concrete_until < scan_ceil and budget.can_afford('concrete'):
            end = min(
                concrete_until + budget.max_affordable_concrete(),
                scan_ceil, MAX_TIME,
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

        # ---- P3: build MPC buffer forward toward conflict_time ----------
        # No P3b passthrough scan: half-planes have no S-region gap to bridge.
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

        # ---- P4: symbolic verification ----------------------------------
        if pending_job is not None and budget.can_afford('symbolic', 1):
            print(f"  [P4] Symbolic: t={pending_job.symbolic_start} → t={pending_job.conflict_time}")
            pending_job, result = symbolic_step(
                tester, pending_job, MAX_SYMBOLIC_HORIZON, budget)

            if result is not None:
                if result["collision"]:
                    print(f"  [P4] Conflict confirmed at t={conflict_time} — retrying INFEASIBLE entries")
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep
                    pending_job       = VerificationTask(current_timestep, conflict_time)
                else:
                    print(f"  [P4] Deconflicted! Clearing conflict state")
                    conflict_time = None
                    pending_job   = None
                    _purge_infeasible(mpc_buffer, current_timestep)
                    mpc_horizon_until = t_diverge if t_diverge is not None else current_timestep

        # ══════════════════════════════════════════════════════════════════
        #  PSF DECISION
        # ══════════════════════════════════════════════════════════════════
        if psf_valid(mpc_buffer, t_next):
            u_diffs.append(0.0)
            tester.real_state_empirical(current_timestep, t_next)
            print(f"  [PSF NOMINAL] t={current_timestep}  PSF OK (buffer[{t_next}] valid)"
                  f"  Budget: {budget.elapsed:.3f}s / {budget.timestep_budget:.3f}s")
            current_timestep += 1

        else:
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
                    u_nn = _get_di_nn_control(tester, current_timestep)
                    u_diffs.append(float(np.abs(ctrl[0] - u_nn[0])) if u_nn is not None else 0.0)
                    tester.real_state_mpc(current_timestep, ctrl)
                    current_timestep += 1
                else:
                    print(f"{RED}[PSF ACTIVATE] ctrl_idx={ctrl_idx} beyond queue"
                          f" ({len(queue)}) — extending{RESET}")
                    extend_mpc_sequence(mpc_sf, mpc_state, mpc_sf.n_horizon, MAX_TIME,
                                        ctrl_idx=ctrl_idx, budget=budget)
                    queue = mpc_state['controls']
                    if ctrl_idx < len(queue):
                        ctrl = queue[ctrl_idx]
                        u_nn = _get_di_nn_control(tester, current_timestep)
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
                # SHOULD NOT RUN IF WORKING
                print(f"{RED}[PSF] No backup available at t={current_timestep}"
                      f" (t_diverge={t_diverge}) — nominal fallback{RESET}")
                u_diffs.append(0.0)
                tester.real_state_empirical(current_timestep, t_next)
                current_timestep += 1

        if budget.elapsed > budget.timestep_budget:
            mpc_over_budget += 1

    # ── Collect state history and check half-plane violations ────────────
    state_history = []
    for t in sorted(tester.horizons.keys()):
        h = tester.horizons[t]
        for calc in h.calculations.values():
            if 'real_state' in calc:
                state_history.append(calc['real_state'].copy())
                break

    violation_timesteps = [
        i for i, s in enumerate(state_history)
        if real_state_violates(s, pos_min=pos_min, vel_min=vel_min)
    ]

    had_violation = len(violation_timesteps) > 0
    print(f"\n{'=' * 60}")
    print(f"Simulation complete at timestep {current_timestep}")
    if had_violation:
        print(f"[SAFETY] CONSTRAINT VIOLATION at real-state timesteps: {violation_timesteps}")
    else:
        print(f"[SAFETY] No real-state constraint violation detected")
    return state_history, had_violation, u_diffs, mpc_calls, mpc_over_budget


# ── DI-specific NN-control query ──────────────────────────────────────────
# alg12's _get_nn_control queries cl_sys.dynamics.control_nn, which works for
# DI too — but we duplicate here for clarity and to keep the import surface
# small in case unicycle and DI diverge later.

def _get_di_nn_control(tester, timestep):
    h = tester.horizons.get(timestep)
    if h is None:
        return None
    calc = next((c for c in h.calculations.values() if 'real_state' in c), None)
    if calc is None:
        return None
    state  = np.asarray(calc['real_state']).reshape(1, -1)
    cl_sys = tester.analyzer.cl_system
    return np.asarray(cl_sys.dynamics.control_nn(state, cl_sys.controller.cpu())).flatten()


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed=seed)
