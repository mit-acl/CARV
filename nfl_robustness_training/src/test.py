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
from utils.robust_training_utils import Analyzer, ReachableSet


from utils.nn import *
from utils.utils import get_plot_filename  # noqa: E402

dir_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main_forward_nick(params: dict) -> Tuple[Dict, Dict]:
    device = 'cpu'
    torch.no_grad()

    def di_condition(input_range):
            delta = 0.0
            return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + delta
        
    def unicycle_condition(input_range):
        delta = 0.0
        obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
                     {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
        
        rx, ry = input_range[[0, 1], 0]
        width, height = input_range[[0, 1], 1] - input_range[[0, 1], 0]

        for obs in obstacles:
            cx, cy = obs['x'], obs['y']
            testX = torch.tensor(cx)
            testY = torch.tensor(cy)

            if (cx < rx):
                testX = rx
            elif (cx > rx + width):
                testX = rx + width


            if (cy < ry):
                testY = ry
            elif (cy > ry + height):
                testY = ry + height
            
            dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
            if dist < obs['r']:
                return False
            
        return True

    def quadrotor_condition(input_range):
        delta = 0.0
        # obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4+delta},
        #              {'x': -1.25, 'y': 1.75, 'r': 1.6+delta}]
        yoffset1 = 2
        zoffset1 = 2
        yoffset2 = -1.5
        zoffset2 = 0.25
        little_radius = 1.25 * 0.5
        big_radius = 2.5
        obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
                     {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
        
                     {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
                     {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},]
        
        rx, ry, rz = input_range[[0, 1, 2], 0]
        length, width, height = input_range[[0, 1, 2], 1] - input_range[[0, 1, 2], 0]

        for obs in obstacles:
            if 'y' in obs:
                cx, cy = obs['x'], obs['y']
                testX = torch.tensor(cx)
                testY = torch.tensor(cy)

                if (cx < rx):
                    testX = rx
                elif (cx > rx + width):
                    testX = rx + width


                if (cy < ry):
                    testY = ry
                elif (cy > ry + height):
                    testY = ry + height
                
                dist = torch.sqrt((cx-testX)**2 + (cy - testY)**2)
                if dist < obs['r']:
                    return False
            if 'z' in obs:
                cx, cz = obs['x'], obs['z']
                testX = torch.tensor(cx)
                testZ = torch.tensor(cz)

                if (cx < rx):
                    testX = rx
                elif (cx > rx + width):
                    testX = rx + width


                if (cz < rz):
                    testZ = rz
                elif (cz > rz + height):
                    testZ = rz + height
                
                dist = torch.sqrt((cx-testX)**2 + (cz - testZ)**2)
                if dist < obs['r']:
                    return False
            
        return True

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

        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=30, device=device)

        tstart = time.time()
        reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=di_condition) # 3, 4, 5, 6, 7
        # reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, visualize=False, condition=di_condition)
        # reach_set_dict = analyzer.calculate_N_step_reachable_sets(indices=[3, 4, 5, 6, 7, 8]) # 3, 4, 5, 6, 7
        # reach_set_dict, info = analyzer.pseudo(condition = di_condition)
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print('Calculation Time: {}'.format(tend-tstart))
        # print('Memory Usage: {}'.format(tracemalloc.get_traced_memory()))


        analyzer.plot_reachable_sets()

        # analyzer.animate_reachability_calculation(info)


        for i in [0, 4, 9, -1]:
            analyzer.another_plotter(info, [i])  

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


        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=24, device=device, save_info=True)

        # analyzer.set_partition_strategy(0, np.array([6,6,18]))
        # tracemalloc.start()
        tstart = time.time()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=unicycle_condition) # 3, 4, 5, 6, 7
        # reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, condition=unicycle_condition, visualize=False)
        # reach_set_dict, info = analyzer.hybr(condition = unicycle_condition)
        reach_set_dict, info = analyzer.pseudo(condition = unicycle_condition)
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print('Calculation Time: {}'.format(tend-tstart))
        # print('Memory Usage: {}'.format(tracemalloc.get_traced_memory()))

        num_trajectories = 500
        xs = analyzer.reachable_sets[0].sample_from_reachable_set(analyzer.cl_system, num_steps=time_horizon, num_trajectories=num_trajectories, sample_corners=False)
        # xs_sorted = xs.reshape((time_horizon+1, num_trajectories, 3))
        

        
        analyzer.plot_reachable_sets()
        
        # for i in [34, 58, -1]:
        for i in [-1]:
            analyzer.another_plotter(info, [i])

        # analyzer.animate_reachability_calculation(info)
    

    if params["system"]["type"] == 'Quadrotor_NL':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"], device=device)
        ol_dyn = dynamics.Quadrotor_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Quadrotor(controller, ol_dyn, device=device)

        # init_range = np.array([
        #     [-10.5, -9.5],
        #     [2.5, 3.5],
        #     [0.75, 1.22],
        #     [0.75, 1.25],
        #     [-0.25, 0.25],
        #     [-0.25, 0.25],
        # ])
        # init_range = np.array([
        #     [-10.05, -9.95],
        #     [2.95, 3.05],
        #     [0.99, 1.01],
        #     [0.99, 1.01],
        #     [-0.01, 0.01],
        #     [-0.01, 0.01],
        # ])
        init_range = np.array([
            [-10.501, -10.499],
            [-0.01, 0.01],
            [0.999, 1.001],
            [0.999, 1.001],
            [-0.001, 0.001],
            [-0.01, 0.01],
        ])


        time_horizon = 53
        # time_horizon = 33


        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=20, device=device, save_info=True)

        # analyzer.set_partition_strategy(0, np.array([1,16,1,1,10,1]))
        # tracemalloc.start()
        tstart = time.time()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=unicycle_condition) # 3, 4, 5, 6, 7
        reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, condition=quadrotor_condition, visualize=False)
        # reach_set_dict, info = analyzer.hybr(condition = unicycle_condition)
        # reach_set_dict, info = analyzer.pseudo(condition = unicycle_condition)
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print('Calculation Time: {}'.format(tend-tstart))
        # print('Memory Usage: {}'.format(tracemalloc.get_traced_memory()))

        num_trajectories = 500
        # import pdb; pdb.set_trace()
        xs = analyzer.reachable_sets[0].sample_from_reachable_set(analyzer.cl_system, num_steps=time_horizon, num_trajectories=num_trajectories, sample_corners=False)
        # fig, ax = plt.subplots()
        # ax.plot(xs[:, 0], xs[:, 1], 'k.', markersize=1)

        # delta = 0.
        # obstacles = [{'x': -6, 'y': -0.5, 'r': 2.4 + delta},
        #             {'x': -1.25, 'y': 1.75, 'r': 1.6 + delta}]
        
        # # for obstacle in obstacles:
        # #     color = '#262626'
        # #     circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor='none')
        # #     ax.add_patch(circle)
        # #     circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor=color, facecolor=color, alpha=0.2)
        # #     ax.add_patch(circle)

        # ax.set_xlim([-10, 1])
        # ax.set_ylim([-1, 4])
        # ax.set_aspect('equal')

        # ax.set_xlabel('$x$ [m]')
        # ax.set_ylabel('$y$ [m]')
        # ax.set_title('x-y quadrotor with obstacles')
        # plt.show()
        yoffset1 = 2
        zoffset1 = 2
        yoffset2 = -1.5
        zoffset2 = 0.25
        little_radius = 1.25
        big_radius = 2.5
        obstacles = [{'x': -6, 'y': -2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': 2. + yoffset1, 'r': little_radius},
                     {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
                     {'x': -6, 'z': -1. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': 3. + zoffset1, 'r': little_radius},
                     {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
                     {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
        
                     {'x': -3, 'y': -2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': 2. + yoffset2, 'r': little_radius},
                     {'x': -3, 'y': -4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'y': 4.5 + yoffset2, 'r': big_radius},
                     {'x': -3, 'z': -1. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': 3. + zoffset2, 'r': little_radius},
                     {'x': -3, 'z': -3.5 + zoffset2, 'r': big_radius},
                     {'x': -3, 'z': 5.5 + zoffset2, 'r': big_radius},]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for obstacle in obstacles:
            if 'z' in obstacle and 'y' in obstacle:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = obstacle['r'] * np.cos(u) * np.sin(v) + obstacle['x']
                y = obstacle['r'] * np.sin(u) * np.sin(v) + obstacle['y']
                z = obstacle['r'] * np.cos(v) + obstacle['z']
            elif 'y' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                y = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['y']
                z = np.outer(np.ones(np.size(u)), v)

            elif 'z' in obstacle and obstacle['r'] == 1.25:
                height = 5
                num_points = 8
                u = np.linspace(0, 2 * np.pi, num_points)
                v = np.linspace(-height, height, num_points)
                x = obstacle['r'] * np.outer(np.cos(u), np.ones(np.size(v))) + obstacle['x']
                z = obstacle['r'] * np.outer(np.sin(u), np.ones(np.size(v))) + obstacle['z']
                y = np.outer(np.ones(np.size(u)), v)
            ax.plot_surface(x, y, z, color='b', alpha=0.1)
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], 'k.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        # xs_sorted = xs.reshape((time_horizon+1, num_trajectories, 3))
        

        
        # analyzer.plot_reachable_sets()
        
        # for i in [34, 58, -1]:
        for i in [-1]:
            analyzer.another_plotter(info, [i])

        for i in [-1]:
            analyzer.three_dimensional_plotter(info, [i])

        # analyzer.animate_reachability_calculation(info)

def setup_parser() -> dict:
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

    args = parser.parse_args()

    if args.config.startswith("configs/"):
        # Use the config files in the pip-installed package
        param_filename = f"{dir_path}/_static/{args.config}"
    else:
        # Use the absolute/relative path provided in args.config
        param_filename = f"{args.config}"

    with open(param_filename, mode="r", encoding="utf-8") as file:
        params = yaml.load(file, yaml.Loader)

    return params


if __name__ == "__main__":
    experiment_params = setup_parser()
    main_forward_nick(experiment_params)
