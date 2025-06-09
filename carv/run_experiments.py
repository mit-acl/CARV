"""Runs a closed-loop reachability experiment according to a param file."""

from nfl_veripy.utils.utils import suppress_unecessary_logs

suppress_unecessary_logs()  # needs to happen before other imports

import argparse  # noqa: E402
import ast  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Tuple  # noqa: E402
import pickle

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import nfl_veripy.analyzers as analyzers  # noqa: E402
import nfl_veripy.constraints as constraints  # noqa: E402
import nfl_veripy.dynamics as dynamics  # noqa: E402
from nfl_veripy.utils.nn import load_controller as load_controller_old  # noqa: E402


from utils.nn import *
from utils.utils import get_plot_filename  # noqa: E402

dir_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def di_condition(input_range, delta=0):
    return input_range[1, 0] >= -1 and input_range[0, 0] >= 0 + delta


def unicycle_condition(input_range, delta=0):
    # obstacles = [{'x': -10, 'y': -1, 'r': 3},
    #                 {'x': -3, 'y': 2.5, 'r': 2}]
    obstacles = [
        {"x": -6, "y": -0.5, "r": 2.4 + delta},
        {"x": -1.25, "y": 1.75, "r": 1.6 + delta},
    ]

    rx, ry = input_range[[0, 1], 0]
    width, height = input_range[[0, 1], 1] - input_range[[0, 1], 0]

    for obs in obstacles:
        cx, cy = obs["x"], obs["y"]
        testX = torch.tensor(cx)
        testY = torch.tensor(cy)

        if cx < rx:
            testX = rx
        elif cx > rx + width:
            testX = rx + width

        if cy < ry:
            testY = ry
        elif cy > ry + height:
            testY = ry + height

        dist = torch.sqrt((cx - testX) ** 2 + (cy - testY) ** 2)
        if dist < obs["r"]:
            return False

    return True


def quadrotor_condition(input_range):
    delta = 0.0
    yoffset1 = 1
    zoffset1 = 3
    yoffset2 = -1.5
    zoffset2 = 1
    little_radius = 1.25 * 0.4
    big_radius = 0.03
    yoffset3 = 0
    zoffset3 = 0
    obstacles = [
        {"x": -6, "y": -2.0 + yoffset1, "r": little_radius},
        {"x": -6, "y": 2.0 + yoffset1, "r": little_radius},
        #  {'x': -6, 'y': -4.5 + yoffset1, 'r': big_radius},
        #  {'x': -6, 'y': 4.5 + yoffset1, 'r': big_radius},
        {"x": -6, "z": -1.0 + zoffset1, "r": little_radius},
        {"x": -6, "z": 3.0 + zoffset1, "r": little_radius},
        #  {'x': -6, 'z': -3.5 + zoffset1, 'r': big_radius},
        #  {'x': -6, 'z': 5.5 + zoffset1, 'r': big_radius},
        {"x": -3, "y": -2.0 + yoffset2, "r": little_radius},
        {"x": -3, "y": 2.0 + yoffset2, "r": little_radius},
        {"x": -3, "y": -4.5 + yoffset2, "r": big_radius},
        {"x": -3, "y": 4.5 + yoffset2, "r": big_radius},
        {"x": -3, "z": -1.0 + zoffset2, "r": little_radius},
        {"x": -3, "z": 3.0 + zoffset2, "r": little_radius},
        {"x": -3, "z": -3.5 + zoffset2, "r": big_radius},
        {"x": -3, "z": 5.5 + zoffset2, "r": big_radius},
        {"x": 0, "y": -2.0 + yoffset3, "r": little_radius},
        {"x": 0, "y": 2.0 + yoffset3, "r": little_radius},
        {"x": 0, "z": -1.0 + zoffset3, "r": little_radius},
        {"x": 0, "z": 3.0 + zoffset3, "r": little_radius},
    ]

    rx, ry, rz = input_range[[0, 1, 2], 0]
    length, width, height = input_range[[0, 1, 2], 1] - input_range[[0, 1, 2], 0]

    for obs in obstacles:
        if "y" in obs:
            cx, cy = obs["x"], obs["y"]
            testX = torch.tensor(cx)
            testY = torch.tensor(cy)

            if cx < rx:
                testX = rx
            elif cx > rx + length:
                testX = rx + length

            if cy < ry:
                testY = ry
            elif cy > ry + width:
                testY = ry + width

            dist = torch.sqrt((cx - testX) ** 2 + (cy - testY) ** 2)
            if dist < obs["r"]:
                return False
        if "z" in obs:
            cx, cz = obs["x"], obs["z"]
            testX = torch.tensor(cx)
            testZ = torch.tensor(cz)

            if cx < rx:
                testX = rx
            elif cx > rx + length:
                testX = rx + length

            if cz < rz:
                testZ = rz
            elif cz > rz + height:
                testZ = rz + height

            dist = torch.sqrt((cx - testX) ** 2 + (cz - testZ) ** 2)
            if dist < obs["r"]:
                return False

    return True


def main_forward_nick(params: dict) -> Tuple[Dict, Dict]:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import nfl_veripy.dynamics as dynamics
    import cl_systems
    from auto_LiRPA import BoundedModule, BoundedTensor
    from utils.robust_training_utils import (
        calculate_reachable_sets,
        partition_init_set,
        plot_reachable_sets,
    )
    from utils.robust_training_utils import (
        calculate_next_reachable_set,
        partition_set,
        calculate_reachable_sets_old,
    )
    from utils.robust_training_utils import Analyzer, ReachableSet

    device = "cpu"
    torch.no_grad()

    if params["system"]["type"] == "DoubleIntegrator":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)

        init_range = np.array([[2.5, 3.0], [-0.25, 0.25]])

        time_horizon = 30

        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(
            cl_dyn, time_horizon, init_range, max_diff=15, device=device
        )
        tstart = time.time()
        # reach_set_dict = analyzer.calculate_hybrid_symbolic_reachable_sets()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None) # 3, 4, 5, 6, 7
        reach_set_dict, snapshots = analyzer.calculate_reachable_sets(
            training=False, autorefine=True, visualize=False, condition=di_condition
        )
        # reach_set_dict = analyzer.calculate_N_step_reachable_sets(indices=[3, 4, 5, 6, 7, 8]) # 3, 4, 5, 6, 7
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print("Calculation Time: {}".format(tend - tstart))

        analyzer.switch_sets_on_off(di_condition)
        # import pdb; pdb.set_trace()

        analyzer.plot_reachable_sets()

        # analyzer.animate_reachability_calculation(snapshots)

        # analyzer.plot_all_subsets()

        with open(dir_path + "/experimental_data/double_integrator.pkl", "wb") as f:
            pickle.dump(snapshots, f)

        for i in [4, 9, -1]:
            analyzer.another_plotter(snapshots, [i])

    if params["system"]["type"] == "Unicycle_NL":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        dummy_input = torch.tensor([[-14.5, 4.5, 0.0]], device=device)

        # init_range = np.array([
        #     [-17.5, -16.5],
        #     [4.5, 5.5],
        #     [-np.pi/6, np.pi/6]
        # ])
        init_range = np.array([[-9.55, -9.45], [3.45, 3.55], [-np.pi / 24, np.pi / 24]])

        time_horizon = 52
        # time_horizon = 19

        # def condition(input_range):
        #     return input_range[1, 0] >= -1

        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(
            cl_dyn, time_horizon, init_range, max_diff=10, device=device
        )

        analyzer.set_partition_strategy(0, np.array([6, 6, 20]))
        # analyzer.set_partition_strategy(0, np.array([3,3,3]))
        tstart = time.time()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=unicycle_condition) # 3, 4, 5, 6, 7
        reach_set_dict, info = analyzer.calculate_reachable_sets(
            training=False,
            autorefine=False,
            condition=unicycle_condition,
            visualize=False,
        )
        # reach_set_dict, info = analyzer.hybr(condition = unicycle_condition)
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print("Calculation Time: {}".format(tend - tstart))

        # analyzer.switch_sets_on_off(condition)

        analyzer.plot_reachable_sets()

        with open(dir_path + "/experimental_data/unicycle_part_more.pkl", "wb") as f:
            pickle.dump(info, f)

        # for i in [34, 58, -1]:
        #     analyzer.another_plotter(info, [i])

        # import pdb; pdb.set_trace()
        # analyzer.plot_all_subsets()
        # analyzer.animate_reachability_calculation(info)

    if params["system"]["type"] == "Quadrotor_NL":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.Quadrotor_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Quadrotor(controller, ol_dyn, device=device)

        init_range = np.array(
            [
                [-10.55, -10.45],
                [-0.05, 0.05],
                [0.95, 1.05],
                [0.99, 1.01],
                [-0.01, 0.01],
                [-0.01, 0.01],
            ]
        )

        time_horizon = 51

        # def condition(input_range):
        #     return input_range[1, 0] >= -1

        init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
        analyzer = Analyzer(
            cl_dyn, time_horizon, init_range, max_diff=15, device=device, save_info=True
        )

        # analyzer.set_partition_strategy(0, np.array([1,16,1,1,10,1]))
        # tracemalloc.start()
        tstart = time.time()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=unicycle_condition) # 3, 4, 5, 6, 7
        reach_set_dict, info = analyzer.calculate_reachable_sets(
            training=False,
            autorefine=True,
            condition=quadrotor_condition,
            visualize=False,
        )
        # reach_set_dict, info = analyzer.hybr(condition = unicycle_condition)
        # reach_set_dict, info = analyzer.pseudo(condition = unicycle_condition)
        tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        print("Calculation Time: {}".format(tend - tstart))

        # analyzer.switch_sets_on_off(condition)

        analyzer.plot_reachable_sets()

        with open(dir_path + "/experimental_data/quadrotor_gates.pkl", "wb") as f:
            pickle.dump(info, f)

        # for i in [34, 58, -1]:
        #     analyzer.another_plotter(info, [i])

        # import pdb; pdb.set_trace()
        # analyzer.plot_all_subsets()
        # analyzer.animate_reachability_calculation(info)


def run_multiple_experiments(params: dict) -> Tuple[Dict, Dict]:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import nfl_veripy.dynamics as dynamics
    import cl_systems
    from auto_LiRPA import BoundedModule, BoundedTensor
    from utils.robust_training_utils import (
        calculate_reachable_sets,
        partition_init_set,
        plot_reachable_sets,
    )
    from utils.robust_training_utils import (
        calculate_next_reachable_set,
        partition_set,
        calculate_reachable_sets_old,
    )
    from utils.robust_training_utils import Analyzer, ReachableSet

    device = "cpu"
    torch.no_grad()

    if params["system"]["type"] == "DoubleIntegrator":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)

        # init_range = np.array([
        #     [2.5, 3.],
        #     [-0.25, 0.25]
        # ])

        time_horizon = 30

        # method = "symb"
        # method = "part"
        # method = "ttt"
        # method = "carv"

        for method in ["part"]:
            max_diff = 15
            if method == "symb":
                max_diff = 2 * time_horizon

            num_trials = 10
            time_data = []
            for i in range(num_trials):

                cl_dyn = cl_systems.ClosedLoopDynamics(
                    controller, ol_dyn, device=device
                )

                init_range = np.array([[2.5, 3.0], [-0.25, 0.25]])
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn,
                    time_horizon,
                    init_range,
                    max_diff=max_diff,
                    device=device,
                    save_info=False,
                )

                if method == "symb":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(
                        indices=None, condition=di_condition
                    )
                    tend = time.time()
                elif method == "part":
                    analyzer.set_partition_strategy(0, np.array([10, 10]))
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=False,
                        visualize=False,
                        condition=di_condition,
                    )
                    tend = time.time()
                elif method == "ttt":
                    tstart = time.time()
                    analyzer.hybr(condition=di_condition)
                    tend = time.time()
                elif method == "carv":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=True,
                        visualize=False,
                        condition=di_condition,
                    )
                    tend = time.time()

                t_total = tend - tstart
                print(t_total)
                time_data.append(t_total)

            print(method + " mean: {}".format(np.mean(time_data)))
            print(method + " std: {}".format(np.std(time_data)))

        # print('Calculation Time: {}'.format(tend-tstart))

        # analyzer.switch_sets_on_off(di_condition)
        # import pdb; pdb.set_trace()

        # analyzer.plot_reachable_sets()

        # analyzer.animate_reachability_calculation(snapshots)

        # analyzer.plot_all_subsets()

        # with open(dir_path + '/experimental_data/double_integrator.pkl', 'wb') as f:
        #     pickle.dump(snapshots, f)

        for i in [4, 9, -1]:
            analyzer.another_plotter(info, [i])

    if params["system"]["type"] == "Unicycle_NL":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        dummy_input = torch.tensor([[-14.5, 4.5, 0.0]], device=device)

        # init_range = np.array([
        #     [-17.5, -16.5],
        #     [4.5, 5.5],
        #     [-np.pi/6, np.pi/6]
        # ])

        time_horizon = 52
        # time_horizon = 19

        # def condition(input_range):
        #     return input_range[1, 0] >= -1

        for method in ["part"]:
            max_diff = 10
            if method == "symb":
                max_diff = 19

            num_trials = 10
            time_data = []
            for i in range(num_trials):
                cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)
                init_range = np.array(
                    [[-9.55, -9.45], [3.45, 3.55], [-np.pi / 24, np.pi / 24]]
                )
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn, time_horizon, init_range, max_diff=10, device=device
                )

                if method == "symb":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(
                        indices=None, condition=unicycle_condition
                    )
                    tend = time.time()
                elif method == "part":
                    # analyzer.set_partition_strategy(0, np.array([10,10]))
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=False,
                        visualize=False,
                        condition=unicycle_condition,
                    )
                    tend = time.time()
                elif method == "ttt":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.hybr(condition=unicycle_condition)
                    tend = time.time()
                elif method == "carv":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=True,
                        visualize=False,
                        condition=unicycle_condition,
                    )
                    tend = time.time()

                t_total = tend - tstart
                print(t_total)
                time_data.append(t_total)

            print(method + " mean: {}".format(np.mean(time_data)))
            print(method + " std: {}".format(np.std(time_data)))
        # analyzer.set_partition_strategy(0, np.array([6,6,18]))
        # analyzer.set_partition_strategy(0, np.array([3,3,3]))
        # tstart = time.time()
        # reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(indices=None, condition=unicycle_condition) # 3, 4, 5, 6, 7
        # reach_set_dict, info = analyzer.calculate_reachable_sets(training=False, autorefine=True, condition=unicycle_condition, visualize=False)
        # reach_set_dict, info = analyzer.ttt(condition = unicycle_condition)
        # tend = time.time()
        # analyzer.switch_sets_on_off(condition)
        # print('Calculation Time: {}'.format(tend-tstart))

        # analyzer.switch_sets_on_off(condition)

        # analyzer.plot_reachable_sets()

        # with open(dir_path + '/experimental_data/unicycle_ttt.pkl', 'wb') as f:
        #     pickle.dump(info, f)

        # for i in [34, 58, -1]:
        #     analyzer.another_plotter(info, [i])

        # import pdb; pdb.set_trace()
        # analyzer.plot_all_subsets()
        # analyzer.animate_reachability_calculation(info)


def sweep_k(params: dict) -> Tuple[Dict, Dict]:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import nfl_veripy.dynamics as dynamics
    import cl_systems
    from auto_LiRPA import BoundedModule, BoundedTensor
    from utils.robust_training_utils import (
        calculate_reachable_sets,
        partition_init_set,
        plot_reachable_sets,
    )
    from utils.robust_training_utils import (
        calculate_next_reachable_set,
        partition_set,
        calculate_reachable_sets_old,
    )
    from utils.robust_training_utils import Analyzer, ReachableSet

    device = "cpu"
    torch.no_grad()

    if params["system"]["type"] == "DoubleIntegrator":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)

        # init_range = np.array([
        #     [2.5, 3.],
        #     [-0.25, 0.25]
        # ])

        time_horizon = 30

        # method = "symb"
        # method = "part"
        # method = "ttt"
        # method = "carv"

        for method in ["ttt"]:
            max_diff = 15
            if method == "symb":
                max_diff = 2 * time_horizon

            num_trials = 10
            time_data = []
            for i in range(num_trials):

                cl_dyn = cl_systems.ClosedLoopDynamics(
                    controller, ol_dyn, device=device
                )

                init_range = np.array([[2.5, 3.0], [-0.25, 0.25]])
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn,
                    time_horizon,
                    init_range,
                    max_diff=max_diff,
                    device=device,
                    save_info=False,
                )

                if method == "symb":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(
                        indices=None, condition=di_condition
                    )
                    tend = time.time()
                elif method == "part":
                    analyzer.set_partition_strategy(0, np.array([10, 10]))
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=False,
                        visualize=False,
                        condition=di_condition,
                    )
                    tend = time.time()
                elif method == "ttt":
                    tstart = time.time()
                    analyzer.hybr(condition=di_condition)
                    tend = time.time()
                elif method == "carv":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=True,
                        visualize=False,
                        condition=di_condition,
                    )
                    tend = time.time()

                t_total = tend - tstart
                print(t_total)
                time_data.append(t_total)

            print(method + " mean: {}".format(np.mean(time_data)))
            print(method + " std: {}".format(np.std(time_data)))

        # print('Calculation Time: {}'.format(tend-tstart))

        # analyzer.switch_sets_on_off(di_condition)
        # import pdb; pdb.set_trace()

        # analyzer.plot_reachable_sets()

        # analyzer.animate_reachability_calculation(snapshots)

        # analyzer.plot_all_subsets()

        # with open(dir_path + '/experimental_data/double_integrator.pkl', 'wb') as f:
        #     pickle.dump(snapshots, f)

        for i in [4, 9, -1]:
            analyzer.another_plotter(info, [i])

    if params["system"]["type"] == "Unicycle_NL":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        dummy_input = torch.tensor([[-14.5, 4.5, 0.0]], device=device)

        # init_range = np.array([
        #     [-17.5, -16.5],
        #     [4.5, 5.5],
        #     [-np.pi/6, np.pi/6]
        # ])

        time_horizon = 52
        # time_horizon = 19

        # def condition(input_range):
        #     return input_range[1, 0] >= -1
        ttt_verif_results = []
        ttt_k_max = []
        ttt_time = []
        ttt_error = []

        carv_verif_results = []
        carv_k_max = []
        carv_time = []
        carv_error = []

        for method in ["ttt", "carv"]:
            for k in range(6, 25):
                cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)
                init_range = np.array(
                    [[-9.55, -9.45], [3.45, 3.55], [-np.pi / 24, np.pi / 24]]
                )
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn, time_horizon, init_range, max_diff=k, device=device
                )

                if method == "symb":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_N_step_reachable_sets(
                        indices=None, condition=unicycle_condition
                    )
                    tend = time.time()
                elif method == "part":
                    analyzer.set_partition_strategy(0, np.array([10, 10]))
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=False,
                        visualize=False,
                        condition=unicycle_condition,
                    )
                    tend = time.time()
                elif method == "ttt":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.hybr(condition=unicycle_condition)
                    tend = time.time()
                elif method == "carv":
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=True,
                        visualize=False,
                        condition=unicycle_condition,
                    )
                    tend = time.time()

                num_trajectories = 500
                xs = analyzer.reachable_sets[0].sample_from_reachable_set(
                    analyzer.cl_system,
                    num_steps=time_horizon,
                    num_trajectories=num_trajectories,
                    sample_corners=False,
                )
                xs_sorted = xs.reshape((time_horizon + 1, num_trajectories, 3))

                t_total = tend - tstart
                print(t_total)

                numerator_volume = 0
                denominator_volume = 0

                if method == "ttt":
                    safe = True
                    for i, reachable_set in reach_set_dict.items():
                        if not unicycle_condition(reachable_set.full_set):
                            safe = False

                        if i > 0:
                            state_range = reachable_set.full_set.detach().numpy()
                            reach_set_volume = np.prod(
                                state_range[:, 1] - state_range[:, 0]
                            )
                            numerator_volume += reach_set_volume

                            underapprox_state_range = np.vstack(
                                (
                                    np.min(xs_sorted[i], axis=0),
                                    np.max(xs_sorted[i], axis=0),
                                )
                            ).T
                            underapprox_volume = np.prod(
                                underapprox_state_range[:, 1]
                                - underapprox_state_range[:, 0]
                            )
                            denominator_volume += underapprox_volume

                    ttt_verif_results.append(safe)
                    ttt_k_max.append(k)
                    ttt_time.append(t_total)
                    ttt_error.append(numerator_volume / denominator_volume)

                if method == "carv":
                    safe = True
                    for i, reachable_set in reach_set_dict.items():
                        if not unicycle_condition(reachable_set.full_set):
                            safe = False

                        if i > 0:
                            state_range = reachable_set.full_set.detach().numpy()
                            reach_set_volume = np.prod(
                                state_range[:, 1] - state_range[:, 0]
                            )
                            numerator_volume += reach_set_volume

                            underapprox_state_range = np.vstack(
                                (
                                    np.min(xs_sorted[i], axis=0),
                                    np.max(xs_sorted[i], axis=0),
                                )
                            ).T
                            underapprox_volume = np.prod(
                                underapprox_state_range[:, 1]
                                - underapprox_state_range[:, 0]
                            )
                            denominator_volume += underapprox_volume

                    carv_verif_results.append(safe)
                    carv_k_max.append(k)
                    carv_time.append(t_total)
                    carv_error.append(numerator_volume / denominator_volume)

        results_dict = {
            "carv_verif_results": carv_verif_results,
            "carv_k_max": carv_k_max,
            "carv_time": carv_time,
            "carv_error": carv_error,
            "ttt_verif_results": ttt_verif_results,
            "ttt_k_max": ttt_k_max,
            "ttt_time": ttt_time,
            "ttt_error": ttt_error,
        }

        with open(dir_path + "/experimental_data/sweep_k_amended.pkl", "wb") as f:
            pickle.dump(results_dict, f)


def sweep_constraints(params: dict) -> Tuple[Dict, Dict]:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import nfl_veripy.dynamics as dynamics
    import cl_systems
    from auto_LiRPA import BoundedModule, BoundedTensor
    from utils.robust_training_utils import (
        calculate_reachable_sets,
        partition_init_set,
        plot_reachable_sets,
    )
    from utils.robust_training_utils import (
        calculate_next_reachable_set,
        partition_set,
        calculate_reachable_sets_old,
    )
    from utils.robust_training_utils import Analyzer, ReachableSet

    device = "cpu"
    torch.no_grad()

    if params["system"]["type"] == "DoubleIntegrator":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)

        # init_range = np.array([
        #     [2.5, 3.],
        #     [-0.25, 0.25]
        # ])

        time_horizon = 30

        # method = "symb"
        # method = "part"
        # method = "ttt"
        # method = "carv"

        carv15_verif_results = []
        carv15_time = []
        carv15_delta = []

        carv30_verif_results = []
        carv30_time = []
        carv30_delta = []

        deltas = np.arange(0, 0.19, 0.01)
        # deltas = np.array([0.0, 0.2])

        for k in [30]:
            for delta in deltas:
                cl_dyn = cl_systems.ClosedLoopDynamics(
                    controller, ol_dyn, device=device
                )
                init_range = np.array([[2.5, 3.0], [-0.25, 0.25]])
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn, time_horizon, init_range, max_diff=k, device=device
                )
                unicycle_condition_delta = lambda inp: di_condition(inp, delta=delta)

                # if (k == 10 and delta <= 0.2) or k == 14:
                tstart = time.time()
                reach_set_dict, info = analyzer.calculate_reachable_sets(
                    training=False,
                    autorefine=True,
                    visualize=False,
                    condition=unicycle_condition_delta,
                )
                tend = time.time()
                t_total = tend - tstart
                print(t_total)

                safe = True
                for i, reachable_set in reach_set_dict.items():
                    if not unicycle_condition(reachable_set.full_set):
                        safe = False

                if k == 15:
                    carv15_verif_results.append(safe)
                    carv15_time.append(t_total)
                    carv15_delta.append(delta)

                if k == 30:
                    carv30_verif_results.append(safe)
                    carv30_time.append(t_total)
                    carv30_delta.append(delta)

        results_dict = {
            # "carv10_verif_results": carv15_verif_results,
            # "carv10_time": carv15_time,
            # "carv10_delta": carv15_delta,
            "carv14_verif_results": carv30_verif_results,
            "carv14_time": carv30_time,
            "carv14_delta": carv30_delta,
        }

        with open(
            dir_path + "/experimental_data/sweep_double_integrator_constraint.pkl", "wb"
        ) as f:
            pickle.dump(results_dict, f)

        # print('Calculation Time: {}'.format(tend-tstart))

        # analyzer.switch_sets_on_off(di_condition)
        # import pdb; pdb.set_trace()

        # analyzer.plot_reachable_sets()

        # analyzer.animate_reachability_calculation(snapshots)

        # analyzer.plot_all_subsets()

        # with open(dir_path + '/experimental_data/double_integrator.pkl', 'wb') as f:
        #     pickle.dump(snapshots, f)

        for i in [4, 9, -1]:
            analyzer.another_plotter(info, [i])

    if params["system"]["type"] == "Unicycle_NL":
        controller = load_controller(
            params["system"]["type"],
            params["system"]["controller"],
            params["system"]["dagger"],
            device=device,
        )
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        time_horizon = 52

        carv10_verif_results = []
        carv10_time = []
        carv10_delta = []

        carv14_verif_results = []
        carv14_time = []
        carv14_delta = []

        deltas = np.hstack((np.arange(0, 0.29, 0.02), 0.29))
        # import pdb; pdb.set_trace()
        # deltas = np.array([0.0, 0.2])

        for k in [12]:
            for delta in deltas:
                cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)
                init_range = np.array(
                    [[-9.55, -9.45], [3.45, 3.55], [-np.pi / 24, np.pi / 24]]
                )
                init_range = torch.from_numpy(init_range).type(torch.float32).to(device)
                analyzer = Analyzer(
                    cl_dyn, time_horizon, init_range, max_diff=k, device=device
                )
                unicycle_condition_delta = lambda inp: unicycle_condition(
                    inp, delta=delta
                )

                if (k == 12 and delta <= 0.24) or k == 24:
                    tstart = time.time()
                    reach_set_dict, info = analyzer.calculate_reachable_sets(
                        training=False,
                        autorefine=True,
                        visualize=False,
                        condition=unicycle_condition_delta,
                    )
                    tend = time.time()
                    t_total = tend - tstart
                    print(t_total)

                    safe = True
                    for i, reachable_set in reach_set_dict.items():
                        if not unicycle_condition(reachable_set.full_set):
                            safe = False

                    if k == 12:
                        carv10_verif_results.append(safe)
                        carv10_time.append(t_total)
                        carv10_delta.append(delta)

                    if k == 24:
                        carv14_verif_results.append(safe)
                        carv14_time.append(t_total)
                        carv14_delta.append(delta)

        results_dict = {
            "carv10_verif_results": carv10_verif_results,
            "carv10_time": carv10_time,
            "carv10_delta": carv10_delta,
            "carv14_verif_results": carv14_verif_results,
            "carv14_time": carv14_time,
            "carv14_delta": carv14_delta,
        }

        with open(
            dir_path + "/experimental_data/sweep_unicycle_constraint.pkl", "wb"
        ) as f:
            pickle.dump(results_dict, f)


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

    # run_multiple_experiments(experiment_params)
    # sweep_k(experiment_params)
    # sweep_constraints(experiment_params)

    save_data = True
    if save_data:
        main_forward_nick(experiment_params)

# if __name__ == "__main__":
#     controller = load_controller('double_integrator', 'natural_default')
#     import pdb; pdb.set_trace()
