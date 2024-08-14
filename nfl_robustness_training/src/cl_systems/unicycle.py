import torch
import torch.nn as nn
import torch.nn.functional as F
from .cl_dynamics import ClosedLoopDynamics

class Unicycle_NL(ClosedLoopDynamics):
    def __init__(self, controller, dynamics, num_steps=1, device='cpu') -> None:
        super().__init__(controller, dynamics, num_steps, device)
    
    def forward(self, xt):
        num_steps = self.num_steps
        xts = [xt]
        for i in range(num_steps):
            ut = self.controller(xts[-1])

            xt_0 = torch.matmul(xts[-1], torch.Tensor([[1], [0], [0]]))
            xt_1 = torch.matmul(xts[-1], torch.Tensor([[0], [1], [0]]))
            xt_2 = torch.matmul(xts[-1], torch.Tensor([[0], [0], [1]]))

            xt1_0 = xt_0 + self.dynamics.dt*ut[:, 0]*torch.cos(xt_2)
            xt1_1 = xt_1 + self.dynamics.dt*ut[:, 0]*torch.sin(xt_2)
            xt1_2 = xt_2 + self.dynamics.dt*ut[:, 1]

            xt1 = torch.cat([xt1_0, xt1_1, xt1_2], 1)
            xts.append(xt1)


        return xts[-1]

class Normalizer(nn.Module):
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std
        super().__init__()

    def forward(self, xt):
        return (xt - self.mean)/self.std

class unicycle_4layer_controller(nn.Module):
    def __init__(self, neurons_per_layer, mean, std):
        super(unicycle_4layer_controller, self).__init__()
        self.norm = Normalizer(mean, std)
        self.fc1 = nn.Linear(3, neurons_per_layer[0])
        self.fc2 = nn.Linear(neurons_per_layer[0], neurons_per_layer[1])
        self.fc3 = nn.Linear(neurons_per_layer[1], neurons_per_layer[2])
        self.fc4 = nn.Linear(neurons_per_layer[2], 2)
        

    def forward(self, xt):
        ut = self.norm(xt)
        ut = F.relu(self.fc1(xt))
        ut = F.relu(self.fc2(ut))
        ut = F.relu(self.fc3(ut))
        ut = self.fc4(ut)
        return ut
    
