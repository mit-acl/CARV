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
        self.At = torch.tensor(dynamics.At, dtype=torch.float32).transpose(0, 1)
        self.bt = torch.tensor(dynamics.bt, dtype=torch.float32).transpose(0, 1)
        self.ct = torch.tensor(dynamics.ct, dtype=torch.float32)

    
    def forward(self, xt):
        ut = self.controller(xt)
        # At = torch.tensor([[1., 1.], [0., 1.]]).transpose(0, 1)
        # bt = torch.tensor([[0.5], [1.]]).transpose(0, 1)
        
        # xt1 = torch.matmul(torch.eye(1), ut.transpose(0, 1)).transpose(1, 0)
        # xt1 =  self.dynamics.dynamics_step_torch(xt, ut)
        xt1 = torch.matmul(xt, self.At) + torch.matmul(ut, self.bt) + self.ct
        # import pdb; pdb.set_trace()
        return xt1