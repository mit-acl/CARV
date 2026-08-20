"""
Time budget for TTT-CARV.
Calibrate at startup, then query what you can afford each timestep.
"""
import os
import time
import math
import numpy as np


class TimeBudget:
    def __init__(self, timestep_budget=0.4):
        self.timestep_budget = timestep_budget
        self.symbolic_costs = {}   # horizon -> seconds
        self.backward_costs = {}   # horizon -> seconds
        self.concrete_cost = 0.0   # per-step cost (seconds per concrete step)
        self.mpc_cost = 0.040      # conservative estimate per MPC solve (seconds)
        self._start = None
        self._log = []             # [(operation_name, elapsed)] for current timestep
        self._excluded = 0.0       # time excluded from budget (non-algorithmic overhead)

    def calibrate(self, tester, max_symbolic_horizon=10, max_backward_horizon=10,
                  num_repeats=1, mpc_probe=None):
        """Run once at startup with a throwaway tester.

        Costs must be measured on the machine and under the load the run will
        actually see — a table calibrated on an idle box makes can_afford() lie
        by up to 12x when workers contend (see concrete_cost in particular).

        max_backward_horizon=0 skips backward calibration, which is the right
        choice for callers that never query 'backward' (e.g. alg12).
        mpc_probe: optional zero-arg callable performing one representative MPC
        solve. Without it mpc_cost keeps its conservative default, which gates
        six affordability checks in alg12.
        """
        max_needed = max(max_symbolic_horizon, max_backward_horizon) + 5
        for t in range(max_needed):
            tester.real_state_empirical(t, t + 1)

        def _median_of(fn):
            # First call absorbs lazy init/JIT warmup; measuring it inflates the
            # cost by ~4x at h=1 and makes the algorithm skip work it can afford.
            fn()
            return float(np.median([fn() for _ in range(num_repeats)]))

        # Symbolic forward
        for h in range(1, max_symbolic_horizon + 1):
            self.symbolic_costs[h] = _median_of(lambda: tester.symbolic(0, h)['time'])

        # Backward
        # We need a collision to backproject from, so we create a dummy target set
        # Just measure the backward_from_set call with a small synthetic set
        dummy_set = np.array([[0.5, 0.6], [-0.3, -0.2]])
        for h in range(1, max_backward_horizon + 1):
            self.backward_costs[h] = _median_of(
                lambda: tester.backward_from_set(
                    target_set=dummy_set,
                    target_timestep=max_needed - 1,
                    num_steps=h,
                )['time']
            )

        # Concrete
        self.concrete_cost = _median_of(lambda: tester.concrete(0, 15)['time']) / 15

        # MPC solve
        if mpc_probe is not None:
            def _timed_mpc():
                t0 = time.perf_counter()
                mpc_probe()
                return time.perf_counter() - t0
            self.mpc_cost = _median_of(_timed_mpc)

    # ─── Timestep tracking ───────────────────────────────────────────

    def exclude_elapsed(self, seconds: float):
        """Exclude time from budget — for non-algorithmic overhead (e.g. frame capture)."""
        self._excluded += seconds

    def start_timestep(self):
        self._start = time.perf_counter()
        self._excluded = 0.0
        self._log = []

    def record(self, name):
        """Call right after an operation to log its time."""
        self._log.append((name, time.perf_counter() - self._start if not self._log
                          else time.perf_counter() - sum(e for _, e in self._log) - self._start))

    @property
    def elapsed(self):
        if self._start is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._start - self._excluded)

    @property
    def remaining(self):
        return max(0.0, self.timestep_budget - self.elapsed)

    # ─── Queries ─────────────────────────────────────────────────────

    def can_afford(self, operation, horizon=None):
        """Check if an operation fits in remaining budget.
        operation: 'symbolic', 'backward', 'concrete'
        """
        return self._cost(operation, horizon) <= self.remaining

    def max_affordable_concrete(self):
        """Number of concrete steps affordable in remaining budget."""
        return int(self.remaining // self.concrete_cost)


    def max_affordable_symbolic(self):
        """Largest symbolic horizon that fits in remaining budget."""
        best = 0
        for h, cost in sorted(self.symbolic_costs.items()):
            if cost <= self.remaining:
                best = h
        return best

    def max_affordable_backward(self):
        """Largest backward horizon that fits in remaining budget."""
        best = 0
        for h, cost in sorted(self.backward_costs.items()):
            if cost <= self.remaining:
                best = h
        return best

    def chunked_cost(self, total_horizon, chunk_size):
        """Estimate cost of running symbolic in chunks of chunk_size."""
        num_chunks = math.ceil(total_horizon / chunk_size)
        cost = 0.0
        remaining = total_horizon
        for _ in range(num_chunks):
            c = min(chunk_size, remaining)
            cost += self._cost('symbolic', c)
            remaining -= c
        return cost

    # ─── Options display ─────────────────────────────────────────────

    def get_options(self, conflict_distance=None, extension_distance=None):
        """
        Return list of (name, cost, affordable) tuples for current situation.
        Pass conflict_distance if a conflict was detected.
        Pass extension_distance if extension is needed after deconfliction.
        """
        opts = []
        r = self.remaining

        opts.append(("concrete lookahead", self.concrete_cost, self.concrete_cost <= r))

        if conflict_distance is not None:
            # Full symbolic to conflict
            h = min(conflict_distance, max(self.symbolic_costs.keys()))
            c = self._cost('symbolic', h)
            opts.append((f"symbolic full ({h} steps)", c, c <= r))

            # Chunked options
            for cs in [3, 5]:
                if cs < conflict_distance:
                    cc = self.chunked_cost(conflict_distance, cs)
                    opts.append((f"symbolic chunked ({cs}-step chunks)", cc, cc <= r))

            # Backward (if we'd need it after symbolic confirms collision)
            bh = min(conflict_distance, max(self.backward_costs.keys()))
            bc = self._cost('backward', bh)
            opts.append((f"backward ({bh} steps)", bc, bc <= r))

            # Symbolic + backward combo
            combo = c + bc
            opts.append((f"symbolic + backward", combo, combo <= r))

        if extension_distance is not None:
            ec = self.concrete_cost
            opts.append((f"concrete extension ({extension_distance} steps)", ec, ec <= r))

            eh = min(extension_distance, max(self.symbolic_costs.keys()))
            esc = self._cost('symbolic', eh)
            opts.append((f"symbolic extension ({eh} steps)", esc, esc <= r))

        return opts

    # def print_options(self, conflict_distance=None, extension_distance=None):
    #     """Print available options with affordability."""
    #     print(f"\n  Budget: {self.remaining:.3f}s / {self.timestep_budget:.3f}s remaining")
    #     opts = self.get_options(conflict_distance, extension_distance)
    #     for name, cost, ok in opts:
    #         status = "✓" if ok else "✗"
    #         print(f"    {status} {name}: {cost:.4f}s")

    # ─── Internal ────────────────────────────────────────────────────

    def _cost(self, operation, horizon=None):
        if operation == 'concrete':
            return self.concrete_cost
        elif operation == 'symbolic':
            return self.symbolic_costs.get(horizon, float('inf'))
        elif operation == 'backward':
            return self.backward_costs.get(horizon, float('inf'))
        return float('inf')


# ─── Per-process calibration ─────────────────────────────────────────────

# Fallback table, measured once on an idle machine. Accurate solo, but under
# 16-way parallelism the real costs are 2-12x these, so can_afford() overcommits
# and blows the whole timestep budget in a single concrete scan.
FALLBACK_SYMBOLIC_COSTS = {
    1:  0.05942702293395996,  2: 0.0532071590423584,
    3:  0.12308859825134277,  4: 0.2227306365966797,
    5:  0.3548123836517334,   6: 0.5160810947418213,
    7:  0.7076215744018555,   8: 1.046485185623169,
    9:  1.189185619354248,   10: 1.4745268821716309,
}
FALLBACK_CONCRETE_COST = 0.0142
FALLBACK_MPC_COST      = 0.040

_CALIBRATION = None


def calibrated_costs(make_tester, make_mpc_probe=None, max_symbolic_horizon=5):
    """Measure op costs once per process and cache them.

    Costs are only meaningful when measured on the machine and under the load
    the run will actually see, so this runs inside each pool worker rather than
    shipping a static table. The result is cached because it costs a few seconds
    and amortizes over every trial the worker goes on to run.

    make_tester()      -> a throwaway tester; calibration advances real_state,
                          which would corrupt the trial if run on the live one.
    make_mpc_probe(t)  -> zero-arg callable doing one representative MPC solve
                          on that tester. Reuse the caller's existing filter:
                          acados codegen dirs are keyed on the pid, so a second
                          filter in this process clobbers the first one's C code.

    Returns (symbolic_costs, concrete_cost, mpc_cost).
    """
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION

    fallback = (dict(FALLBACK_SYMBOLIC_COSTS), FALLBACK_CONCRETE_COST, FALLBACK_MPC_COST)
    if os.environ.get('TTTCARV_CALIBRATE', '1') == '0':
        _CALIBRATION = fallback
        return _CALIBRATION

    t0 = time.time()
    try:
        probe_tester = make_tester()
        cal = TimeBudget()
        cal.calibrate(
            probe_tester,
            max_symbolic_horizon=max_symbolic_horizon,
            max_backward_horizon=0,   # forward algorithms never query 'backward'
            num_repeats=2,
            mpc_probe=make_mpc_probe(probe_tester) if make_mpc_probe else None,
        )
        _CALIBRATION = (cal.symbolic_costs, cal.concrete_cost, cal.mpc_cost)
        print(f"[calibrate] {time.time()-t0:.1f}s  concrete={cal.concrete_cost:.4f}s"
              f"  mpc={cal.mpc_cost:.4f}s"
              f"  symbolic={ {h: round(c, 4) for h, c in cal.symbolic_costs.items()} }")
    except Exception as e:
        print(f"[calibrate] failed ({e}) — falling back to the static table")
        _CALIBRATION = fallback
    return _CALIBRATION
