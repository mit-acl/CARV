import numpy as np
import os
import pickle
import torch
import pdb

import cl_systems

import torch.nn as nn

PATH = os.getcwd()

def load_controller(system, controller_name = 'default', dagger=False):
    if system == 'DoubleIntegrator':
        neurons_per_layer = [30, 20, 10]
        # neurons_per_layer = [15, 10, 5]
        controller = cl_systems.Controllers["di_4layer"](neurons_per_layer)

        if dagger:
            controller_path = PATH + '/nfl_robustness_training/src/controller_models/double_integrator/daggers/di_4layer/' + controller_name + '.pth'
        else:
            controller_path = PATH + '/nfl_robustness_training/src/controller_models/double_integrator/di_4layer/' + controller_name + '.pth'
        state_dict = torch.load(controller_path)['state_dict']
        controller.load_state_dict(state_dict)
    
    else:
        raise NotImplementedError

    return controller

def controller2sequential(controller):
    # import pdb; pdb.set_trace()
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


# def load_torch_controller(
#     system: str = "DoubleIntegrator",
#     model_name: str = "default",
#     model_type: str = "torch",
# ) -> Sequential:
#     if type(model_name) is not str:
#         # TODO: update type in signature to properly reflect other options
#         return model_name
#     system = system.replace(
#         "OutputFeedback", ""
#     )  # remove OutputFeedback suffix if applicable
#     # path = "{}/../_static/models/{}/{}".format(dir_path, system, model_name)
#     model_file = "/home/nick/code/nfl_robustness_training/nfl_robustness_training/src/controller_models/double_integrator/natural_default.pth"

#     neurons_per_layer = [10, 5]
#     # keras_model = create_model(neurons_per_layer, input_shape=(2,), output_shape=(1,))
#     # torch_model = keras2torch(keras_model, "torch_model")
    
#     torch_model = model_mlp_any(2, neurons_per_layer, 1)
#     checkpoint = torch.load(model_file)
#     if isinstance(checkpoint["state_dict"], list):
#         checkpoint["state_dict"] = checkpoint["state_dict"][0]
#     new_state_dict = {}
#     for k in checkpoint["state_dict"].keys():
#         if "prev" in k:
#             pass
#         else:
#             new_state_dict[k] = checkpoint["state_dict"][k]
#     checkpoint["state_dict"] = new_state_dict
    
#     """
#     state_dict = m.state_dict()
#     state_dict.update(checkpoint["state_dict"])
#     m.load_state_dict(state_dict)
#     print(checkpoint["state_dict"]["__mask_layer.weight"])
#     """

#     torch_model.load_state_dict(checkpoint["state_dict"])

#     # torch_model.load_state_dict(torch.load(file_path))
#     # torch_model.load_state_dict(torch.load(path + '/default'))


#     weights = []
#     for name, param in torch_model.named_parameters():
#         weights.append(param.detach().numpy())

#     return torch_model