import torch
import numpy as np
import matplotlib.pyplot as plt
from constraints.constraint import Constraint

class UnicycleConstraint(Constraint):
    def __init__(self, perturbation=0.0):
        super().__init__(perturbation)
        self.delta = perturbation
        self.obstacles = [
                        {'x': -6, 'y': -0.5, 'r': 2.4+self.delta},
                        {'x': -1.25, 'y': 1.75, 'r': 1.6+self.delta}]

    def is_safe(self, input_range):
        obstacles = self.obstacles
        
        rx, ry = input_range[[0, 1], 0]
        width, height = input_range[[0, 1], 1] - input_range[[0, 1], 0]

        for obs in obstacles:
            cx, cy = obs['x'], obs['y']
            testX = torch.tensor(cx)
            testY = torch.tensor(cy)

            if (cx < rx):
                testX = rx
            elif (cx > rx + width):
                testX = rx + width


            if (cy < ry):
                testY = ry
            elif (cy > ry + height):
                testY = ry + height
            
            dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
            if dist < obs['r']:
                return False
            
        return True
    
    def plot(self, ax):
        for obstacle in self.obstacles:
            color = '#262626'
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor='none')
            ax.add_patch(circle)
            circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
            ax.add_patch(circle)