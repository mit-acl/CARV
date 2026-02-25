"""
Time budget for TTT-CARV.
Calibrate at startup, then query what you can afford each timestep.
"""
import time
import math
import numpy as np


class TimeBudget:
    def __init__(self, timestep_budget=0.4):
        self.timestep_budget = timestep_budget
        self.symbolic_costs = {}   # horizon -> seconds
        self.backward_costs = {}   # horizon -> seconds
        self.concrete_cost = 0.0   # per-step cost (seconds per concrete step)
        self._start = None
        self._log = []             # [(operation_name, elapsed)] for current timestep

    def calibrate(self, tester, max_symbolic_horizon=10, max_backward_horizon=10, num_repeats=1):
        """Run once at startup with a throwaway tester."""
        max_needed = max(max_symbolic_horizon, max_backward_horizon) + 5
        for t in range(max_needed):
            tester.real_state_empirical(t, t + 1)

        # Symbolic forward
        for h in range(1, max_symbolic_horizon + 1):
            times = []
            for _ in range(num_repeats):
                result = tester.symbolic(0, h)
                times.append(result['time'])
            self.symbolic_costs[h] = np.median(times)

        # Backward
        # We need a collision to backproject from, so we create a dummy target set
        # Just measure the backward_from_set call with a small synthetic set
        dummy_set = np.array([[0.5, 0.6], [-0.3, -0.2]])
        for h in range(1, max_backward_horizon + 1):
            times = []
            for _ in range(num_repeats):
                result = tester.backward_from_set(
                    target_set=dummy_set,
                    target_timestep=max_needed - 1,
                    num_steps=h
                )
                times.append(result['time'])
            self.backward_costs[h] = np.median(times)

        # Concrete
        times = []
        for _ in range(num_repeats):
            result = tester.concrete(0, 15)
            times.append(result['time'])
        self.concrete_cost = np.median(times) / 15  # per-step cost

    # ─── Timestep tracking ───────────────────────────────────────────

    def start_timestep(self):
        self._start = time.perf_counter()
        self._log = []

    def record(self, name):
        """Call right after an operation to log its time."""
        self._log.append((name, time.perf_counter() - self._start if not self._log
                          else time.perf_counter() - sum(e for _, e in self._log) - self._start))

    @property
    def elapsed(self):
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

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
