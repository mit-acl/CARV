import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import PerturbationLpNorm, BoundedParameter
from nfl_veripy.dynamics import DoubleIntegrator

class di_2layer_controller(nn.Module):
    def __init__(self, neurons_per_layer):
        super(di_2layer_controller, self).__init__()
        self.fc1 = nn.Linear(2, neurons_per_layer[0])
        self.fc2 = nn.Linear(neurons_per_layer[0], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class di_3layer_controller(nn.Module):
    def __init__(self, neurons_per_layer):
        super(di_3layer_controller, self).__init__()
        self.fc1 = nn.Linear(2, neurons_per_layer[0])
        self.fc2 = nn.Linear(neurons_per_layer[0], neurons_per_layer[1])
        self.fc3 = nn.Linear(neurons_per_layer[1], 1)
        

    def forward(self, xt):
        ut = F.relu(self.fc1(xt))
        ut = F.relu(self.fc2(ut))
        ut = self.fc3(ut)
        return ut
    

class di_4layer_controller(nn.Module):
    def __init__(self, neurons_per_layer):
        super(di_3layer_controller, self).__init__()
        self.fc1 = nn.Linear(2, neurons_per_layer[0])
        self.fc2 = nn.Linear(neurons_per_layer[0], neurons_per_layer[1])
        self.fc3 = nn.Linear(neurons_per_layer[1], neurons_per_layer[2])
        self.fc4 = nn.Linear(neurons_per_layer[2], 1)
        

    def forward(self, xt):
        ut = F.relu(self.fc1(xt))
        ut = F.relu(self.fc2(ut))
        ut = F.relu(self.fc3(ut))
        ut = self.fc3(ut)
        return ut