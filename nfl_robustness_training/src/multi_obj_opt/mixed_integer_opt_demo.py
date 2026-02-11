from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "b": Binary(),
            "x": Choice(options=["nothing", "multiply"]),
            "y": Integer(bounds=(-2, 2)),
            "z": Real(bounds=(-5, 5)),
        }
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        b, x, z, y = X["b"], X["x"], X["z"], X["y"]

        f1 = z ** 2 + y ** 2
        f2 = (z+2) ** 2 + (y-1) ** 2

        if b:
            f2 = 100 * f2

        if x == "multiply":
            f2 = 10 * f2

        out["F"] = [f1, f2]

from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

problem = MultiObjectiveMixedVariableProblem()

algorithm = MixedVariableGA(pop_size=20, survival=RankAndCrowdingSurvival())

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")

import os
output_dir = "./nfl_robustness_training/src/multi_obj_opt"
os.makedirs(output_dir, exist_ok=True)

plot.save(f"{output_dir}/mixed_integer_opt_objective_pareto_front.png")
plt.show()