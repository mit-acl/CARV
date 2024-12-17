import torch
import numpy as np
from constraints.constraint import Constraint

class GatesConstraint(Constraint):
    def __init__(self, perturbation=0.0):
        super().__init__(perturbation)
        delta = self.perturbation
        self.yoffset1 = 1
        self.zoffset1 = 3
        self.yoffset2 = -1.5
        self.zoffset2 = 1
        self.little_radius = 1.25*0.4 + delta
        self.yoffset3 = 0
        self.zoffset3 = 0
        self.obstacles = [
                    {'x': -6, 'y': -2. + self.yoffset1, 'r': self.little_radius, 'gate_number': 1},
                    {'x': -6, 'y': 2. + self.yoffset1, 'r': self.little_radius, 'gate_number': 1},
                    {'x': -6, 'z': -1. + self.zoffset1, 'r': self.little_radius, 'gate_number': 1},
                    {'x': -6, 'z': 3. + self.zoffset1, 'r': self.little_radius, 'gate_number': 1},

        
                    {'x': -3, 'y': -2. + self.yoffset2, 'r': self.little_radius, 'gate_number': 2},
                    {'x': -3, 'y': 2. + self.yoffset2, 'r': self.little_radius, 'gate_number': 2},
                    {'x': -3, 'z': -1. + self.zoffset2, 'r': self.little_radius, 'gate_number': 2},
                    {'x': -3, 'z': 3. + self.zoffset2, 'r': self.little_radius, 'gate_number': 2},

                    
                    {'x': 0, 'y': -2. + self.yoffset3, 'r': self.little_radius, 'gate_number': 3},
                    {'x': 0, 'y': 2. + self.yoffset3, 'r': self.little_radius, 'gate_number': 3},
                    {'x': 0, 'z': -1. + self.zoffset3, 'r': self.little_radius, 'gate_number': 3},
                    {'x': 0, 'z': 3. + self.zoffset3, 'r': self.little_radius, 'gate_number': 3},]
        

    def is_safe(self, input_range):
        obstacles = self.obstacles
        
        rx, ry, rz = input_range[[0, 1, 2], 0]
        length, width, height = input_range[[0, 1, 2], 1] - input_range[[0, 1, 2], 0]

        for obs in obstacles:
            if 'y' in obs:
                cx, cy = obs['x'], obs['y']
                testX = torch.tensor(cx)
                testY = torch.tensor(cy)

                if (cx < rx):
                    testX = rx
                elif (cx > rx + length):
                    testX = rx + length

                if (cy < ry):
                    testY = ry
                elif (cy > ry + width):
                    testY = ry + width
                
                dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
                if dist < obs['r']:
                    return False
            if 'z' in obs:
                cx, cz = obs['x'], obs['z']
                testX = torch.tensor(cx)
                testZ = torch.tensor(cz)

                if (cx < rx):
                    testX = rx
                elif (cx > rx + length):
                    testX = rx + length

                if (cz < rz):
                    testZ = rz
                elif (cz > rz + height):
                    testZ = rz + height
                
                dist = torch.sqrt((cx-testX)**2 + (cz - testZ)**2)
                if dist < obs['r']:
                    return False
        
        return True
    
    def plot(self, ax):
        for obstacle in self.obstacles:
            color = '#262626'
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = obstacle['r'] * np.cos(v) + obstacle['z']
            elif 'y' in obstacle and obstacle['r'] == self.little_radius:
                if obstacle['gate_number'] == 1:
                    offset = self.zoffset1 + 1
                elif obstacle['gate_number'] == 2:
                    offset = self.zoffset2 + 1
                elif obstacle['gate_number'] == 3:
                    offset = self.zoffset3 + 1

                height = (4 + self.little_radius)/2
                num_points = 64
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height + offset, height + offset, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                z = np.outer(np.ones(np.size(u)), v)

            elif 'z' in obstacle and obstacle['r'] == self.little_radius:
                if obstacle['gate_number'] == 1:
                    offset = self.yoffset1
                elif obstacle['gate_number'] == 2:
                    offset = self.yoffset2
                elif obstacle['gate_number'] == 3:
                    offset = self.yoffset3
                
                height = (4 + self.little_radius)/2
                num_points = 64
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height + offset, height + offset, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
                y = np.outer(np.ones(np.size(u)), v)
            ax.plot_surface(x, y, z, color=color, alpha=0.1)

        # for obstacle in obstacles:
        #     color = '#262626'
        #     if 'z' in obstacle and 'y' in obstacle:
        #         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        #         x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
        #         y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
        #         z = obstacle['r'] * np.cos(v) + obstacle['z']
        #     elif 'y' in obstacle and obstacle['r'] == little_radius:
        #         if obstacle['gate_number'] == 1:
        #             offset = zoffset1 + 1
        #         elif obstacle['gate_number'] == 2:
        #             offset = zoffset2 + 1
        #         elif obstacle['gate_number'] == 3:
        #             offset = zoffset3 + 1

        #         height = (4 + little_radius)/2
        #         num_points = 64
        #         u = np.linspace(0, 2 * np.pi, num_points)
        #         v = np.linspace(-height + offset, height + offset, num_points)
        #         x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
        #         y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
        #         z = np.outer(np.ones(np.size(u)), v)

        #     elif 'z' in obstacle and obstacle['r'] == little_radius:
        #         if obstacle['gate_number'] == 1:
        #             offset = yoffset1
        #         elif obstacle['gate_number'] == 2:
        #             offset = yoffset2
        #         elif obstacle['gate_number'] == 3:
        #             offset = yoffset3
                
        #         height = (4 + little_radius)/2
        #         num_points = 64
        #         u = np.linspace(0, 2 * np.pi, num_points)
        #         v = np.linspace(-height + offset, height + offset, num_points)
        #         x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
        #         z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
        #         y = np.outer(np.ones(np.size(u)), v)
        #     ax.plot_surface(x, y, z, color=color, alpha=0.1)