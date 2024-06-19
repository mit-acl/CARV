import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import PerturbationLpNorm, BoundedParameter
from nfl_veripy.dynamics import DoubleIntegrator

class ClosedLoopDynamics(nn.Module):
    def __init__(self, controller, dynamics) -> None:
        super().__init__()
        self.controller = controller
        self.dynamics = dynamics
    
    def forward(self, xt):
        ut = self.controller(xt)
        xt1 =  self.dynamics.dynamics_step_torch(xt, ut)
        return xt1