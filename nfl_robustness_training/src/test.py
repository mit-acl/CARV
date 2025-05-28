"""Runs a closed-loop reachability experiment according to a param file."""

from nfl_veripy.utils.utils import suppress_unecessary_logs

suppress_unecessary_logs()  # needs to happen before other imports

import argparse  # noqa: E402
import ast  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import tracemalloc

import nfl_veripy.analyzers as analyzers  # noqa: E402
import nfl_veripy.constraints as constraints  # noqa: E402
import nfl_veripy.dynamics as dynamics  # noqa: E402
from nfl_veripy.utils.nn import load_controller as load_controller_old  # noqa: E402

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import nfl_veripy.dynamics as dynamics
import cl_systems
import constraints
from utils.robust_training_utils import Analyzer, ReachableSet
from experiment_plotter import official_plotter, official_3D_plotter


from utils.nn import *
from utils.utils import get_plot_filename  # noqa: E402

dir_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main_forward_nick(params: dict) -> Tuple[Dict, Dict]:
    """
    Main function to perform forward analysis for different system types.
    Args:
        params (dict): Dictionary containing parameters for the analysis. 
            Expected keys include:
            - "system": Dictionary with keys:
            - "type": Type of the system ('DoubleIntegrator', 'Unicycle_NL', 'Quadrotor_NL').
            - "controller": Controller type.
            - "dagger": Dagger parameter.
            - "method": Method to be used for analysis ('carv', 'part', 'symb', 'unif', 'pseudo').
    Returns:
        Tuple[Dict, Dict]: A tuple containing:
            - reach_set_dict (Dict): Dictionary of reachable sets.
            - info (Dict): Dictionary containing additional information about the analysis.
    Raises:
        ValueError: If an unknown method is specified in params.
    """
    device = 'cpu'
    torch.no_grad()

    # Set up the system, controller, and analysis parameters for different system types
    if params["system"]["type"] == 'DoubleIntegrator':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"], device=device)
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)
        
        init_range = np.array([
            [2.5, 3.],
            [-0.25, 0.25]
        ])
        time_horizon = 30

        # Initialize the analyzer and constraint for the system
        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=30, device=device)
        constraint = constraints.DoubleIntegratorTwoConstraint()

        # Perform forward reachability analysis using the specified method
        tstart = time.time()
        if params['method'] == 'carv':
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'part':
            analyzer.set_partition_strategy(0, np.array([10,10]))
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'symb':
            analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=time_horizon, device=device)
            reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=constraint.is_safe)
        elif params['method'] == 'unif':
            reach_set_dict, info = analyzer.hybr(condition=constraint.is_safe)
        elif params['method'] == 'pseudo':
            reach_set_dict, info = analyzer.pseudo(condition=constraint.is_safe)
        else:
            raise ValueError(f"Unknown method: {params['method']}")
        tend = time.time()

        print('Calculation Time: {}'.format(tend-tstart))
        official_plotter(info, cl_dyn, save_animation=False, name='DI', constraint=constraint)


    if params["system"]["type"] == 'Unicycle_NL':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"], device=device)
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        init_range = np.array([
            [-9.55, -9.45],
            [3.45, 3.55],
            [-np.pi/24, np.pi/24]
        ])
        time_horizon = 52

        # Initialize the analyzer and constraint for the system
        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=10, device=device, save_info=True)
        constraint = constraints.UnicycleConstraint()

        # Perform forward reachability analysis using the specified method
        tstart = time.time()
        if params['method'] == 'carv':
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'part':
            analyzer.set_partition_strategy(0, np.array([6,6,18]))
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'symb':
            analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=time_horizon, device=device)
            reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=constraint.is_safe)
        elif params['method'] == 'unif':
            reach_set_dict, info = analyzer.hybr(condition=constraint.is_safe)
        elif params['method'] == 'pseudo':
            reach_set_dict, info = analyzer.pseudo(condition=constraint.is_safe)
        else:
            raise ValueError(f"Unknown method: {params['method']}")
        tend = time.time()

        print('Calculation Time: {}'.format(tend-tstart))
        official_plotter(info, cl_dyn, save_animation=False, name='CARV10', constraint=constraint)
    

    if params["system"]["type"] == 'Quadrotor_NL':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"], device=device)
        ol_dyn = dynamics.Quadrotor_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Quadrotor(controller, ol_dyn, device=device)

        init_range = np.array([
            [-10.55, -10.45],
            [-0.05, 0.05],
            [0.95, 1.05],
            [0.99, 1.01],
            [-0.01, 0.01],
            [-0.01, 0.01],
        ])
        time_horizon = 51

        # Initialize the analyzer and constraint for the system
        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=15, device=device, save_info=True)
        constraint = constraints.GatesConstraint()

        # Perform forward reachability analysis using the specified method
        tstart = time.time()
        if params['method'] == 'carv':
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'part':
            analyzer.set_partition_strategy(0, np.array([1,16,1,1,10,1]))
            reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=constraint.is_safe)
        elif params['method'] == 'symb':
            reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=constraint.is_safe)
        elif params['method'] == 'unif':
            reach_set_dict, info = analyzer.hybr(condition=constraint.is_safe)
        elif params['method'] == 'pseudo':
            reach_set_dict, info = analyzer.pseudo(condition=constraint.is_safe)
        else:
            raise ValueError(f"Unknown method: {params['method']}")
        tend = time.time()

        print('Calculation Time: {}'.format(tend-tstart))
        official_3D_plotter(info, cl_dyn, save_animation=False, name='CARV15', constraint=constraint)

def setup_parser() -> dict:
    """
    Parses command-line arguments and loads a YAML configuration file with experiment parameters.
    Returns:
        dict: A dictionary containing the experiment parameters loaded from the YAML file,
              with an additional key 'method' indicating the chosen method for the experiment.
    Command-line Arguments:
        --config (str): Absolute or relative path to the YAML file describing the experiment configuration.
                        If the path starts with 'configs/', the configuration files in the installed package
                        will be used (ignoring the current working directory).
        --method (str): Method to use for the experiment. Choices are 'carv', 'part', 'symb', 'unif', or 'pseudo'.
    Raises:
        FileNotFoundError: If the specified YAML configuration file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    """Load yaml config file with experiment params."""
    parser = argparse.ArgumentParser(
        description="Analyze a closed loop system w/ NN controller."
    )

    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Absolute or relative path to yaml file describing experiment"
            " configuration. Note: if this arg starts with 'example_configs/',"
            " the configs in the installed package will be used (ignoring the"
            " pwd)."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['carv', 'part', 'symb', 'unif', 'pseudo'],
        help=(
            "Method to use for the experiment. Choose from 'carv', 'part', 'symb', 'unif', or 'pseudo'."
        ),
    )

    args = parser.parse_args()

    if args.config.startswith("configs/"):
        # Use the config files in the pip-installed package
        param_filename = f"{dir_path}/_static/{args.config}"
    else:
        # Use the absolute/relative path provided in args.config
        param_filename = f"{args.config}"

    with open(param_filename, mode="r", encoding="utf-8") as file:
        params = yaml.load(file, yaml.Loader)
    
    params['method'] = args.method

    return params


if __name__ == "__main__":
    experiment_params = setup_parser()
    main_forward_nick(experiment_params)
