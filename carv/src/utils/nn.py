import numpy as np
import os
import pickle
import torch
import pdb

import cl_systems

import torch.nn as nn

PATH = os.getcwd()

def load_controller(system, controller_name = 'default', dagger=False, device='cpu'):
    if system == 'DoubleIntegrator':
        neurons_per_layer = [30, 20, 10]
        controller = cl_systems.Controllers["di_4layer"](neurons_per_layer)

        if dagger:
            controller_path = PATH + '/carv/controller_models/double_integrator/daggers/di_4layer/' + controller_name + '.pth'
        else:
            controller_path = PATH + '/carv/controller_models/double_integrator/di_4layer/' + controller_name + '.pth'

    elif system == "Unicycle_NL":
        neurons_per_layer = [40, 20, 10]
        mean = torch.tensor([-7.5, 2.5, 0], device=device)
        std = torch.tensor([7.5, 2.5, torch.pi/6], device=device)
        controller = cl_systems.Controllers["unicycle_nl_4layer"](neurons_per_layer, mean, std)
        
        controller_path = PATH + '/carv/controller_models/Unicycle_NL/unicycle_nl_4layer/' + controller_name + '.pth'

    elif system == "Quadrotor_NL":
        neurons_per_layer = [40, 20, 10]
        mean = torch.tensor([-5.25, 1.75, 1, 1, 0, 0], device=device)
        std = torch.tensor([5.25, 1.75, 0.25, 0.25, 1, 1], device=device)
        controller = cl_systems.Controllers["quadrotor_4layer"](neurons_per_layer, mean, std)
        
        controller_path = PATH + '/carv/controller_models/Quadrotor/quadrotor_4layer/' + controller_name + '.pth'
        
    else:
        raise NotImplementedError
    
    state_dict = torch.load(controller_path)['state_dict']
    controller.load_state_dict(state_dict)
    if device == 'cuda':
        controller = controller.cuda()

    return controller

def controller2sequential(controller):
    model = nn.Sequential(
        controller.fc1,
        nn.ReLU(),
        controller.fc2,
        nn.ReLU(),
        controller.fc3,
        nn.ReLU(),
        controller.fc4
    )
    return model