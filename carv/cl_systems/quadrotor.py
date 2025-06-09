import torch
import torch.nn as nn
import torch.nn.functional as F
from .cl_dynamics import ClosedLoopDynamics

class Quadrotor(ClosedLoopDynamics):
    def __init__(self, controller, dynamics, num_steps=1, device='cpu') -> None:
        super().__init__(controller, dynamics, num_steps, device)
        self.g = 9.81
    
    def forward(self, xt):
        g = self.g
        num_steps = self.num_steps
        xts = [xt]

        for i in range(num_steps):
            ut = self.controller(xts[-1])
            # ut_bounded = -F.relu(F.relu(ut + torch.pi/4)) + torch.pi/4
            # ut[:, 0] = torch.clip(ut[:, 0], -torch.pi/4, torch.pi/4)
            # ut[:, 1] = torch.clip(ut[:, 1], -torch.pi/4, torch.pi/4)
            # ut[:, 2] = torch.clip(ut[:, 2], -10, 10)
            # ut = -F.relu(-ut + torch.pi/4) + torch.pi/4
            ut_bounded = F.relu(-F.relu(-ut + torch.pi/6) + torch.pi/3) - torch.pi/6
            ut_bounded2 = F.relu(-F.relu(-ut + 12) + 12 - 8) + 8
            # ut2 = F.relu(ut + torch.pi/4) - torch.pi/4
            # ut
            # ut[:, 0] = -torch.relu(-ut[:, 0] + torch.pi/4) + torch.pi/4
            # ut = torch.clip()

            xt_0 = torch.matmul(xts[-1], torch.Tensor([[1], [0], [0], [0], [0], [0]]))
            xt_1 = torch.matmul(xts[-1], torch.Tensor([[0], [1], [0], [0], [0], [0]]))
            xt_2 = torch.matmul(xts[-1], torch.Tensor([[0], [0], [1], [0], [0], [0]]))
            xt_3 = torch.matmul(xts[-1], torch.Tensor([[0], [0], [0], [1], [0], [0]]))
            xt_4 = torch.matmul(xts[-1], torch.Tensor([[0], [0], [0], [0], [1], [0]]))
            xt_5 = torch.matmul(xts[-1], torch.Tensor([[0], [0], [0], [0], [0], [1]]))

            xt1_0 = xt_0 + self.dynamics.dt*xt_3
            xt1_1 = xt_1 + self.dynamics.dt*xt_4
            xt1_2 = xt_2 + self.dynamics.dt*xt_5

            
            xt1_3 = xt_3 + self.dynamics.dt*g*torch.tan(ut_bounded[:, 0])
            xt1_4 = xt_4 - self.dynamics.dt*g*torch.tan(ut_bounded[:, 1])
            xt1_5 = xt_5 + self.dynamics.dt*(ut_bounded2[:, 2] - g)

            xt1 = torch.cat([xt1_0, xt1_1, xt1_2, xt1_3, xt1_4, xt1_5], 1)
            xts.append(xt1)


        return xts[-1]

class Normalizer(nn.Module):
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std
        super().__init__()

    def forward(self, xt):
        return (xt - self.mean)/self.std

class quadrotor_4layer_controller(nn.Module):
    def __init__(self, neurons_per_layer, mean, std):
        super(quadrotor_4layer_controller, self).__init__()
        self.norm = Normalizer(mean, std)
        self.fc1 = nn.Linear(6, neurons_per_layer[0])
        self.fc2 = nn.Linear(neurons_per_layer[0], neurons_per_layer[1])
        self.fc3 = nn.Linear(neurons_per_layer[1], neurons_per_layer[2])
        self.fc4 = nn.Linear(neurons_per_layer[2], 3)
        

    def forward(self, xt):
        ut = self.norm(xt)
        ut = F.relu(self.fc1(xt))
        ut = F.relu(self.fc2(ut))
        ut = F.relu(self.fc3(ut))
        ut = self.fc4(ut)
        return ut