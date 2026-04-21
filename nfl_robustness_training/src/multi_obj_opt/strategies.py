import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REAL_integrated_sim import setup_analyzer, setup_backward_analyzer, ReachabilityTester, CalculationType, Obstacles
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from scipy import stats as scipy_stats
import time as _time

import numpy as np

class TimeEstimator:
    _DEFAULT_TIMING = {
        "concrete_slope":     0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope":        0.5341915550605524,
        "ratio_intercept":   -0.0578267576662421,
    }

    def __init__(self):
        self._timing_params = dict(self._DEFAULT_TIMING)
        self.set_timing_params(self._timing_params)

    def set_timing_params(self, p):
        self.concrete_slope     = p["concrete_slope"]
        self.concrete_intercept = p["concrete_intercept"]
        self.ratio_slope        = p["ratio_slope"]
        self.ratio_intercept    = p["ratio_intercept"]
        
    def concrete_time_from_horizon(self, h_c):
        return self.concrete_slope * h_c + self.concrete_intercept

    def symbolic_time_from_horizon(self, h_s):
        base_concrete_k = self.concrete_slope * h_s + self.concrete_intercept
        ratio           = self.ratio_slope * h_s + self.ratio_intercept
        comp_time       = ratio * base_concrete_k
        return comp_time

    def concrete_horizon_from_time(self, t_c):
        return (t_c - self.concrete_intercept) / self.concrete_slope

    def symbolic_horizon_from_time(self, t_s):
        a = self.concrete_slope
        b = self.concrete_intercept
        alpha = self.ratio_slope
        beta  = self.ratio_intercept

        coeffs = [
            alpha * a,                  # h^2
            alpha * b + beta * a,      # h
            beta * b - t_s             # constant
        ]

        raw_roots = np.roots(coeffs)

        # Keep real, positive roots
        roots = raw_roots[np.isreal(raw_roots)].real
        roots = roots[roots >= 0]

        if len(roots) == 0:
            raise ValueError(f"No valid horizon found ({raw_roots=}, {t_s=})")

        return roots.max()

class ExtensionOptimizer:
    """
    Chooses between symbolic and concrete for computing the reachable set at T+1.

    symbolic cost = w_time * time_symbolic(k) + w_vol * vol_symbolic(k)
    concrete cost = w_time * time_concrete(1) + w_vol * vol_concrete(1)

    where:
        k               = verified_until + 1 - current_timestep  (symbolic span)
        time_symbolic   = ratio(k) * time_concrete(k)            (regression model)
        time_concrete   = slope * h + intercept                  (regression model)
        vol_symbolic(k) = current_vol * a * k^b                  (growth from current set)
        vol_concrete(1) = verified_vol * a * 1^b = verified_vol * a  (growth from T)
    """

    _DEFAULT_TIMING = {
        "concrete_slope":     0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope":        0.5341915550605524,
        "ratio_intercept":   -0.0578267576662421,
    }
    _DEFAULT_VOLUME = {
        "vol_growth_a": 1.05,
        "vol_growth_b": 0.3,
    }

    def __init__(self, algorithm=None):
        self.algorithm      = MixedVariableGA(pop_size=20) if algorithm is None else algorithm
        self._timing_params = dict(self._DEFAULT_TIMING)
        self._volume_params = dict(self._DEFAULT_VOLUME)
        self._build_problem()
        self.set_timing_params(self._timing_params)
        self.set_volume_params(self._volume_params)

    # ─── Problem ─────────────────────────────────────────────────────────

    def _build_problem(self):

        class ExtensionProblem(ElementwiseProblem):
            def __init__(self, **kwargs):
                vars = {"method": Choice(options=["concrete", "symbolic"])}
                # Decision inputs — set before each call to minimize()
                self.k            = 1.0    # symbolic span = verified_until+1 - current_t
                self.current_vol  = 1.0    # volume at current_timestep
                self.verified_vol = 1.0    # volume at verified_until (T)
                self.w_time       = 1.0
                self.w_vol        = 1.0
                self.time_budget  = 0.4
                # Timing params
                self.concrete_slope     = 0.006311738129818
                self.concrete_intercept = 0.0035703865687052
                self.ratio_slope        = 0.5341915550605524
                self.ratio_intercept    = -0.0578267576662421
                # Volume growth params
                self.vol_growth_a = 1.05
                self.vol_growth_b = 0.3
                super().__init__(vars=vars, n_obj=1, n_ieq_constr=1, **kwargs)

            def set_inputs(self, k, current_vol, verified_vol,
                           w_time, w_vol, pow_time, pow_vol, time_budget):
                self.k            = float(k)
                self.current_vol  = current_vol
                self.verified_vol = verified_vol
                self.w_time       = w_time
                self.w_vol        = w_vol
                self.pow_time       = pow_time
                self.pow_vol        = pow_vol
                self.time_budget  = time_budget

            def set_timing_params(self, p):
                self.concrete_slope     = p["concrete_slope"]
                self.concrete_intercept = p["concrete_intercept"]
                self.ratio_slope        = p["ratio_slope"]
                self.ratio_intercept    = p["ratio_intercept"]

            def set_volume_params(self, p):
                self.vol_growth_a = p["vol_growth_a"]
                self.vol_growth_b = p["vol_growth_b"]

            def _concrete_vol_growth(self, h):
                """Predicted concrete volume growth over h steps."""
                return self.vol_growth_a * (float(h) ** self.vol_growth_b)

            def _symbolic_vol_growth(self, h):
                """Predicted symbolic volume growth over h steps."""
                return 1

            def _evaluate(self, X, out, *args, **kwargs):
                method = X["method"]

                if method == "concrete":
                    comp_time = self.concrete_slope * 1 + self.concrete_intercept
                    final_vol = self.verified_vol * self._concrete_vol_growth(1)

                else:  # symbolic
                    base_concrete_k = self.concrete_slope * self.k + self.concrete_intercept
                    ratio           = self.ratio_slope * self.k + self.ratio_intercept
                    comp_time       = ratio * base_concrete_k
                    final_vol       = self.current_vol * self._symbolic_vol_growth(self.k)

                out["F"] = self.w_time * comp_time ** self.pow_time + self.w_vol * final_vol ** self.pow_vol
                out["G"] = [comp_time - self.time_budget]

        self.problem = ExtensionProblem()

    # ─── Parameter management ─────────────────────────────────────────────

    def set_timing_params(self, param_dict):
        self._timing_params = dict(param_dict)
        self.problem.set_timing_params(param_dict)

    def set_volume_params(self, param_dict):
        self._volume_params = dict(param_dict)
        self.problem.set_volume_params(param_dict)

    # ─── Main query ───────────────────────────────────────────────────────

    def get_strategy(self, current_timestep, verified_until,
                     current_vol, verified_vol,
                     time_budget, w_time=1.0, w_vol=1.0,
                     pow_time=1, pow_vol=1):
        """
        Args:
            current_timestep : current real-world timestep
            verified_until   : T — last verified timestep
            current_vol      : tight-bound volume at current_timestep
            verified_vol     : tight-bound volume at verified_until
            time_budget      : remaining seconds this timestep
            w_time           : weight on computation time in objective
            w_vol            : weight on predicted final volume in objective

        Returns: "concrete" or "symbolic"
        """
        k = verified_until + 1 - current_timestep

        self.problem.set_inputs(
            k            = k,
            current_vol  = current_vol,
            verified_vol = verified_vol,
            w_time       = w_time,
            w_vol        = w_vol,
            pow_time     = pow_time,
            pow_vol      = pow_vol,
            time_budget  = time_budget,
        )

        # res    = minimize(self.problem, self.algorithm,
        #                   termination=('n_gen', 40), seed=1, verbose=False)
        # method = res.X["method"]

        # print(f"[ExtOpt] t={current_timestep}  T={verified_until}  k={k}  "
        #       f"cur_vol={current_vol:.4f}  ver_vol={verified_vol:.4f}  "
        #       f"budget={time_budget:.3f}s  "
        #       f"→ {method}  "
        #       f"(F={res.F[0]:.4f}, {'OK' if res.G[0] <= 0 else 'BUDGET VIOLATED'})")
        # return method

        _t0 = _time.perf_counter()
        res = minimize(self.problem, self.algorithm,
               termination=('n_gen', 40), seed=1, verbose=False)
        _strategy_time = _time.perf_counter() - _t0

        # res.X is None when pymoo finds no feasible solution (budget constraint
        # eliminates both options). Fall back to concrete — it's cheaper and
        # the budget check in optimized_step will catch overruns.
        if res.X is None:
            method = "concrete"
            print(f"[ExtOpt] t={current_timestep}  T={verified_until}  k={k}  "
                f"no feasible solution (budget too tight) — defaulting to concrete  "
                f"[strategy_time={_strategy_time:.4f}s]")
        else:
            method = res.X["method"]
            print(f"[ExtOpt] t={current_timestep}  T={verified_until}  k={k}  "
                f"cur_vol={current_vol:.4f}  ver_vol={verified_vol:.4f}  "
                f"budget={time_budget:.3f}s  "
                f"→ {method}  "
                f"(F={res.F[0]:.4f}, {'OK' if res.G[0] <= 0 else 'BUDGET VIOLATED'})  "
                f"[strategy_time={_strategy_time:.4f}s]")

        return method

    def get_strategy_opt_free(self, current_timestep, verified_until,
                            current_vol, verified_vol,
                            time_budget, w_time=1.0, w_vol=1.0,
                            pow_time=1, pow_vol=1):
        _t0 = _time.perf_counter()
        k = verified_until + 1 - current_timestep

        concrete_time, concrete_cost = self._eval_concrete(verified_vol, w_time, w_vol, pow_time, pow_vol)
        symbolic_time, symbolic_cost = self._eval_symbolic(k, current_vol, w_time, w_vol, pow_time, pow_vol)

        concrete_feasible = concrete_time <= time_budget
        symbolic_feasible = symbolic_time <= time_budget

        if not concrete_feasible and not symbolic_feasible:
            method = "concrete"
            print(f"[ExtOpt] t={current_timestep}  T={verified_until}  k={k}  "
                f"no feasible solution (budget too tight) — defaulting to concrete")
        elif not symbolic_feasible:
            method = "concrete"
        elif not concrete_feasible:
            method = "symbolic"
        else:
            method = "concrete" if concrete_cost <= symbolic_cost else "symbolic"

        _strategy_time = _time.perf_counter() - _t0

        print(f"[ExtOpt] t={current_timestep}  T={verified_until}  k={k}  "
            f"cur_vol={current_vol:.4f}  ver_vol={verified_vol:.4f}  "
            f"budget={time_budget:.3f}s  → {method}  "
            f"(concrete F={concrete_cost:.4f}/t={concrete_time:.4f}  "
            f"symbolic F={symbolic_cost:.4f}/t={symbolic_time:.4f})"
            f"[strategy_time={_strategy_time:.4f}s]")
        return method

    def _cost_fn(self, w_time, comp_time, pow_time, w_vol, final_vol, pow_vol):
        return w_time * comp_time ** pow_time + w_vol * final_vol ** pow_vol

    def _cost_fn_log(self, w_time, comp_time, pow_time, w_vol, final_vol, pow_vol):
        return w_time * np.log(1 + comp_time) + w_vol * (final_vol) ** pow_vol

    def _eval_concrete(self, verified_vol, w_time, w_vol, pow_time, pow_vol):
        p = self._timing_params
        comp_time = p["concrete_slope"] * 1 + p["concrete_intercept"]
        final_vol = verified_vol * self.problem._concrete_vol_growth(1)
        # cost = w_time * comp_time ** pow_time + w_vol * final_vol ** pow_vol
        cost = self._cost_fn_log(w_time, comp_time, pow_time, w_vol, final_vol, pow_vol)
        return comp_time, cost


    def _eval_symbolic(self, k, current_vol, w_time, w_vol, pow_time, pow_vol):
        p = self._timing_params
        base_concrete_k = p["concrete_slope"] * k + p["concrete_intercept"]
        ratio           = p["ratio_slope"] * k + p["ratio_intercept"]
        comp_time       = ratio * base_concrete_k
        final_vol       = current_vol * self.problem._symbolic_vol_growth(k)
        # cost = w_time * comp_time ** pow_time + w_vol * final_vol ** pow_vol
        cost = self._cost_fn_log(w_time, comp_time, pow_time, w_vol, final_vol, pow_vol)
        return comp_time, cost