import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import pdb

import cl_systems

PATH = os.getcwd()

def visualize_nn_controller(system = 'double_integrator', data = 'default'):
    if system == 'double_integrator':
        with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/xs.pkl', 'rb') as f:
            xs = pickle.load(f)
        
        with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/us.pkl', 'rb') as f:
            us = pickle.load(f)
        
        neurons_per_layer = [10, 5]
        controller = cl_systems.Controllers["di_3layer"](neurons_per_layer)

        controller_path = PATH + '/nfl_robustness_training/src/controller_models/double_integrator/natural_default.pth'
        state_dict = torch.load(controller_path)['state_dict']
        controller.load_state_dict(state_dict)

        
        us_controller = controller(torch.tensor(xs, dtype=torch.float32))


        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], us.flatten(), c='b')
        ax.scatter(xs[:, 0], xs[:, 1], us_controller.detach().numpy().flatten(), c='r')
        fig_path = PATH + '/nfl_robustness_training/src/plots/default_di_controller.png'
        fig.savefig(fig_path)
        # plt.show()

    else:
        raise NotImplementedError
    

    
if __name__ == '__main__':
    visualize_nn_controller()