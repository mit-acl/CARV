import numpy as np
import os
import pickle
from nfl_veripy.utils.create_training_dataset import generate_dataset
import nfl_veripy.dynamics as dynamics
import nfl_veripy.constraints as constraints

dir_path = os.getcwd()

def generate_new_data(dyn, input_constraint, dataset_name, t_max=5, num_samples=12000) -> None:
    xs, us = generate_dataset(dyn, input_constraint, dataset_name, t_max, num_samples=num_samples)

    if dyn.name == "DoubleIntegrator":
        system = "double_integrator"
    else:
        raise NotImplementedError

    path = "{}/nfl_robustness_training/src/_static/datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path + "/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)


if __name__ == "__main__":
    init_state_range = np.array(
        [
            [2.5, 3],
            [-0.25, 0.25]
        ]
    )
    dyn = dynamics.DoubleIntegrator(dt=0.2)
    input_constraint = constraints.LpConstraint(
        range=init_state_range, p=np.inf
    )
    dataset_name = "default_more_data_5hz"
    t_max = 25

    generate_new_data(dyn, input_constraint, dataset_name, t_max, num_samples=6000)

