import numpy as np
from nfl_veripy.utils.mpc import control_mpc
from scipy.linalg import solve_discrete_are
import torch.nn as nn
import torch.nn.functional as F
import torch

from .Dynamics import DiscreteTimeDynamics


class DoubleIntegrator(DiscreteTimeDynamics):
    def __init__(self, dt=1):
        self.continuous_time = False

        # dt = 0.0625
        # self.dt = dt

        At = np.array([[1, dt], [0, 1]])
        bt = np.array([[0.5 * dt * dt], [dt]])
        ct = np.array([0.0, 0.0]).T

        self.dynamics_module = DoubleIntegratorDynamics()
        self.controller_module = Controller()

        # u_limits = None
        u_limits = np.array([[-1.0, 1.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=ct, dt=dt, u_limits=u_limits)

        self.cmap_name = "tab10"

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(2)
        if not hasattr(self, "R"):
            self.R = 1
        if not hasattr(self, "Pinf"):
            self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

        return control_mpc(
            x0,
            self.At,
            self.bt,
            self.ct,
            self.Q,
            self.R,
            self.Pinf,
            self.u_limits[:, 0],
            self.u_limits[:, 1],
            n_mpc=10,
            debug=False,
        )


# Define computation as a nn.Module.
class DoubleIntegratorDynamics(nn.Module):
    def __init__(self, dt=0.2) -> None:
        At = np.array([[1, dt], [0, 1]])
        bt = np.array([[0.5 * dt * dt], [dt]])
        ct = np.array([0.0, 0.0]).T
        self.At = torch.tensor(At, dtype=torch.float32).transpose(0, 1)
        self.bt = torch.tensor(bt, dtype=torch.float32).transpose(0, 1)
        self.ct = torch.tensor(ct, dtype=torch.float32)
        super().__init__()

    def forward(self, xt, ut):
        # got this from pg 15 of: https://arxiv.org/pdf/2108.01220.pdf
        # updated to values from page 21
        
        xt1 = torch.matmul(xt, self.At) + torch.matmul(ut, self.bt) + self.ct
        return xt1


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, xt):
        # ut = F.relu(torch.matmul(xt, torch.Tensor([[1], [0]])))
        output = F.relu(self.fc1(xt))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)

        return output