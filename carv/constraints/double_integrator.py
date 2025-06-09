import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from constraints.constraint import Constraint

class DoubleIntegratorTwoConstraint(Constraint):
    def __init__(self, perturbation=0.0):
        super().__init__(perturbation)
        self.delta = perturbation
        self.obstacles = [
                        {'x': -6, 'y': -0.5, 'r': 2.4+self.delta},
                        {'x': -1.25, 'y': 1.75, 'r': 1.6+self.delta}]

    def is_safe(self, input_range):
        return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + self.delta
    
    def plot(self, ax):
        constraint_color = '#262626'
        ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c=constraint_color, linewidth=2)
        rect = Rectangle(np.array([0.0, -1.25]), 3.25, 0.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
        ax.add_patch(rect)

        ax.plot(np.array([0, 0]), np.array([-1.25, 1.]), c=constraint_color, linewidth=2)
        rect = Rectangle(np.array([-1.5, -1.25]), 1.5, 2.25, linewidth=1, edgecolor=constraint_color, facecolor=constraint_color, alpha=0.2)
        ax.add_patch(rect)