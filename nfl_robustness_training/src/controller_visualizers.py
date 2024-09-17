import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

import cl_systems


PATH = os.getcwd()

def visualize_nn_controller(system = 'double_integrator', data = 'default'):
    if system == 'double_integrator':
        if data == "default":
            with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/default/xs.pkl', 'rb') as f:
                xs = pickle.load(f)
            
            with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/default/us.pkl', 'rb') as f:
                us = pickle.load(f)
        elif data == "expanded":
            with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/expanded/dataset.pkl', 'rb') as f:
                xs, us, ts = pickle.load(f)
        elif data == "expanded_5hz":
            with open(PATH + '/nfl_robustness_training/src/_static/datasets/double_integrator/expanded_5hz/dataset.pkl', 'rb') as f:
                xs, us = pickle.load(f)
        else:
            raise NotImplementedError
        
        neurons_per_layer = [30, 20, 10]
        controller = cl_systems.Controllers["di_4layer"](neurons_per_layer)

        controller_path = PATH + '/nfl_robustness_training/src/controller_models/double_integrator/di_4layer/natural_expanded_5hz.pth'
        state_dict = torch.load(controller_path)['state_dict']
        controller.load_state_dict(state_dict)

        
        us_controller = controller(torch.tensor(xs, dtype=torch.float32))

        # import pdb; pdb.set_trace()

        fig = plt.figure(figsize=(12, 12))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], us.flatten(), c='b')
        ax.scatter(xs[:, 0], xs[:, 1], us_controller.detach().numpy().flatten(), c='r')
        ax.set_xlabel('x1', fontsize=20)
        ax.set_ylabel('x2', fontsize=20)
        ax.set_zlabel('u', fontsize=20)
        fig_path = PATH + '/nfl_robustness_training/src/plots/controllers/{}_di_controller.png'.format(data)
        fig.savefig(fig_path)
        plt.show()

    else:
        raise NotImplementedError
    

    
if __name__ == '__main__':
    visualize_nn_controller(data="expanded_5hz")