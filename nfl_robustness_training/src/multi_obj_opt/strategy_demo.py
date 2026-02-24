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

import numpy as np

# TODO: Goals
# Estimate the time required to perform a symbolic, concrete, etc. calculation
    # Do this by fitting the curve from timing_analysis and finding the relationship between 

class StrategyOptimizer:
    def __init__(
        self, 
        analyzer, 
        backward_analyzer, 
        reachability_tester, 
        # problem: ElementwiseProblem, 
        algorithm = None, 
        problem = None,
        # termination = None,
    ):
        self.analyzer = analyzer
        self.backward_analyzer = backward_analyzer
        self.reachability_tester = reachability_tester
        self.algorithm = MixedVariableGA(pop_size=20) if algorithm is None else algorithm
        self.set_problem(problem)

    def set_problem(self, problem = None, default_horizon = 5):
        if problem is not None:
            self.problem = problem
            return
        class MixedVariableProblem(ElementwiseProblem):
            def __init__(self, horizon = 1, w_vol = 1, time_budget = 20, **kwargs):
                vars = {
                    "method": Choice(options=["symbolic", "concrete"]),
                }
                self.horizon = float(horizon)
                self.w_vol = w_vol
                self.time_budget = time_budget
                super().__init__(vars=vars, n_obj=1, n_ieq_constr=1, **kwargs)

            def set_horizon(self, h):
                self.horizon = h

            def set_vol_weight(self, w):
                self.w_vol = w

            def set_time_budget(self, T):
                self.time_budget = T

            def _evaluate(self, X, out, *args, **kwargs):
                method = X["method"]

                concrete_time = self.concrete_slope * self.horizon + self.concrete_intercept
                symbolic_time = (self.ratio_slope * self.horizon + self.ratio_intercept) * concrete_time

                concrete_vol = 1.0 * self.horizon + 2.0
                symbolic_vol = 0.8 * (self.horizon ** 0.5) + 1.0

                if method == "symbolic":
                    comp_time = symbolic_time
                    final_vol = symbolic_vol
                else:
                    comp_time = concrete_time
                    final_vol = concrete_vol

                cost = comp_time + self.w_vol * final_vol

                out["F"] = cost
                out["G"] = [comp_time - self.time_budget] # G > 0 indicates violated constraint

            def set_params(self, param_dict = None):
                if param_dict is None:
                    self.concrete_slope = 0.006311738129818,
                    self.concrete_intercept = 0.0035703865687052,
                    self.ratio_slope = 0.5341915550605524,
                    self.ratio_intercept = -0.0578267576662421,
                else:
                    self.concrete_slope = param_dict["concrete_slope"]
                    self.concrete_intercept = param_dict["concrete_intercept"]
                    self.ratio_slope = param_dict["ratio_slope"]
                    self.ratio_intercept = param_dict["ratio_intercept"]

        self.problem = MixedVariableProblem(default_horizon)

    def calibrate(self, t_start, t_end):
        """
        use all calculations from self.analyzer within the interval [t_start, t_end] to calculate self.param_dict
        """

        concrete_horizons = []
        concrete_times = []
        ratio_horizons = []
        ratios = []

        # Walk through timesteps in [t_start, t_end]
        for t in range(t_start, t_end + 1):
            if t not in self.reachability_tester.horizons:
                continue

            horizon_obj = self.reachability_tester.horizons[t]

            # Accumulate concrete and symbolic times grouped by horizon length
            concrete_total = {}  # horizon_length -> list of times
            symbolic_total = {}  # horizon_length -> list of times

            for calc_id, calc_info in horizon_obj.calculations.items():
                calc_type = calc_info['calc_type']
                step_size = calc_info.get('step_size', None)
                elapsed = calc_info.get('time', None)

                if step_size is None or elapsed is None:
                    continue

                if calc_type.name == 'CONCRETE':
                    concrete_total.setdefault(step_size, []).append(elapsed)
                elif calc_type.name == 'SYMBOLIC':
                    symbolic_total.setdefault(step_size, []).append(elapsed)

            # For each horizon length with concrete data, record it
            for h, times in concrete_total.items():
                mean_time = sum(times) / len(times)
                concrete_horizons.append(h)
                concrete_times.append(mean_time)

                # If we also have symbolic data for this horizon, compute ratio
                if h in symbolic_total:
                    sym_mean = sum(symbolic_total[h]) / len(symbolic_total[h])
                    ratio = sym_mean / mean_time
                    ratio_horizons.append(h)
                    ratios.append(ratio)

        param_dict = {}

        # Fit concrete time vs horizon
        if len(concrete_horizons) >= 2:
            slope, intercept, _, _, _ = scipy_stats.linregress(concrete_horizons, concrete_times)
            param_dict['concrete_slope'] = slope
            param_dict['concrete_intercept'] = intercept
        else:
            # Fall back to defaults
            param_dict['concrete_slope'] = 0.006311738129818
            param_dict['concrete_intercept'] = 0.0035703865687052
            print("Warning: insufficient concrete data for calibration, using defaults.")

        # Fit ratio vs horizon
        if len(ratio_horizons) >= 2:
            slope, intercept, _, _, _ = scipy_stats.linregress(ratio_horizons, ratios)
            param_dict['ratio_slope'] = slope
            param_dict['ratio_intercept'] = intercept
        else:
            param_dict['ratio_slope'] = 0.5341915550605524
            param_dict['ratio_intercept'] = -0.0578267576662421
            print("Warning: insufficient symbolic data for calibration, using defaults.")

        self.set_params(param_dict)
        return param_dict

    def set_params(self, param_dict = None):
        self.problem.set_params(param_dict)

    def get_strategy(self, horizon, volume_weight = None, time_budget = None):
        self.problem.set_horizon(horizon)
        if volume_weight is not None:
            self.problem.set_vol_weight(volume_weight) 
        if time_budget is not None:
            self.problem.set_time_budget(time_budget)

        res = minimize(
            self.problem,
            self.algorithm,
            termination=('n_gen', 40),
            seed=1,
            verbose=True
        )

        best_x = res.X
        best_f = res.F[0]
        best_g = res.G[0]

        print(f"Total computation time = {best_g + self.problem.time_budget}/Total time budget = {self.problem.time_budget}")

        return best_x["method"]
    

if __name__ == "__main__":
    # from load_regression_results import load_regression_results

    # results = load_regression_results("run_12_di", use_weighted=False)
    # concrete_slope = results['concrete']['slope']
    # concrete_intercept = results['concrete']['intercept']
    # ratio_slope = results['ratio']['slope']
    # ratio_intercept = results['ratio']['intercept']
    # param_dict = {
    #     "concrete_slope": concrete_slope,
    #     "concrete_intercept": concrete_intercept,
    #     "ratio_slope": ratio_slope,
    #     "ratio_intercept": ratio_intercept,
    # }


    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    backward_analyzer = setup_backward_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    reachability_tester = ReachabilityTester(analyzer, backward_analyzer=backward_analyzer)

    algorithm = None
    param_dict = {
        "concrete_slope": 0.006311738129818,
        "concrete_intercept": 0.0035703865687052,
        "ratio_slope": 0.5341915550605524,
        "ratio_intercept": -0.0578267576662421,
    }

    strategy_optimizer = StrategyOptimizer(analyzer, backward_analyzer, reachability_tester, algorithm = algorithm)
    strategy_optimizer.set_params(param_dict)

    horizon = 5
    time_budget = .15 # seconds
    method = strategy_optimizer.get_strategy(horizon, time_budget=time_budget)
    print(f"the best strategy for horizon = {horizon} is {method}")
    