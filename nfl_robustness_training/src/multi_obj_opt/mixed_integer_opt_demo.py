# from pymoo.core.problem import ElementwiseProblem
# from pymoo.core.variable import Real, Integer, Choice, Binary

# class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

#     def __init__(self, **kwargs):
#         vars = {
#             "b": Binary(),
#             "x": Choice(options=["nothing", "multiply"]),
#             "y": Integer(bounds=(-2, 2)),
#             "z": Real(bounds=(-5, 5)),
#         }
#         super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

#     def _evaluate(self, X, out, *args, **kwargs):
#         b, x, z, y = X["b"], X["x"], X["z"], X["y"]

#         f1 = z ** 2 + y ** 2
#         f2 = (z+2) ** 2 + (y-1) ** 2

#         if b:
#             f2 = 100 * f2

#         if x == "multiply":
#             f2 = 10 * f2

#         out["F"] = [f1, f2]

# from pymoo.visualization.scatter import Scatter
# from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
# from pymoo.core.mixed import MixedVariableGA
# from pymoo.optimize import minimize
# import matplotlib.pyplot as plt

# problem = MultiObjectiveMixedVariableProblem()

# algorithm = MixedVariableGA(pop_size=20, survival=RankAndCrowdingSurvival())

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 50),
#                seed=1,
#                verbose=False)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")

# import os
# output_dir = "./nfl_robustness_training/src/multi_obj_opt"
# os.makedirs(output_dir, exist_ok=True)

# plot.save(f"{output_dir}/mixed_integer_opt_objective_pareto_front.png")
# plt.show()


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REAL_integrated_sim import setup_analyzer, setup_backward_analyzer, ReachabilityTester, CalculationType, Obstacles
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import numpy as np

from load_regression_results import load_regression_results

FIXED_HORIZON = 15
A_TIME = 1.0     # weight on computation time
B_VOL  = 0.5     # weight on reachable set volume

# Load results (weighted by default - recommended)
results = load_regression_results("run_12_di", use_weighted=False)
# print(f'results: {results}')

# Access coefficients directly
concrete_slope = results['concrete']['slope']
concrete_intercept = results['concrete']['intercept']

ratio_slope = results['ratio']['slope']
ratio_intercept = results['ratio']['intercept']

class MixedVariableProblem(ElementwiseProblem):
    def __init__(self, horizon, **kwargs):
        vars = {
            "method": Choice(options=["symbolic", "concrete"]),
        }
        self.horizon = float(horizon)
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def set_horizon(self, h):
        self.horizon = h

    def _evaluate(self, X, out, *args, **kwargs):
        method = X["method"]
        h = self.horizon

        concrete_time = concrete_slope * h + concrete_intercept
        symbolic_time = (ratio_slope * h + ratio_intercept) * concrete_time

        concrete_vol = 1.0 * h + 2.0
        symbolic_vol = 0.8 * (h ** 0.5) + 1.0

        if method == "symbolic":
            comp_time = symbolic_time
            final_vol = symbolic_vol
        else:
            comp_time = concrete_time
            final_vol = concrete_vol

        cost = A_TIME * comp_time + B_VOL * final_vol

        out["F"] = cost

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import os

problem = MixedVariableProblem(FIXED_HORIZON)

algorithm = MixedVariableGA(
    pop_size=20
)

res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 40),
    seed=1,
    verbose=True
)

best_x = res.X
# best_f = res.F
best_f = res.F[0]

print(
    f"Best method @ h={FIXED_HORIZON}: {best_x['method']}\n"
    f"Weighted cost = {best_f:.4f}\n"
    f"(a={A_TIME}, b={B_VOL})"
)

import numpy as np

results = []

for b in np.linspace(0.0, 2.0, 15):
    B_VOL = b

    problem = MixedVariableProblem(FIXED_HORIZON)
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 30),
        seed=1,
        verbose=False
    )

    results.append((b, res.X["method"], res.F[0]))

for b, method, cost in results:
    print(f"b={b:.2f} -> method={method:9s}, cost={cost:.3f}")

print(f"{concrete_slope = }")
print(f"{concrete_intercept = }")
print(f"{ratio_slope = }")
print(f"{ratio_intercept = }")