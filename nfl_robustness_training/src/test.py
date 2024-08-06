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

import nfl_veripy.analyzers as analyzers  # noqa: E402
import nfl_veripy.constraints as constraints  # noqa: E402
import nfl_veripy.dynamics as dynamics  # noqa: E402
from nfl_veripy.utils.nn import load_controller as load_controller_old  # noqa: E402


from utils.nn import *
from utils.utils import get_plot_filename  # noqa: E402

dir_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def main_forward(params: dict) -> Tuple[Dict, Dict]:
    """Runs a forward reachability analysis experiment according to params."""
    np.random.seed(seed=0)
    stats = {}

    # dyn = dynamics.get_dynamics_instance(
    #     params["system"]["type"], params["system"]["feedback"]
    # )
    dyn = dynamics.DoubleIntegrator(dt=0.2)
    # controller_old = load_controller_old(
    #     system=dyn.__class__.__name__,
    #     model_name=params["system"]["controller"],
    # )

    controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"])
    controller = controller2sequential(controller)

    # Set up analyzer (+ parititoner + propagator + visualizer)
    analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
    analyzer.partitioner = params["analysis"]["partitioner"]
    analyzer.propagator = params["analysis"]["propagator"]
    analyzer.visualizer = params["visualization"]

    initial_state_range = np.array(
        ast.literal_eval(params["analysis"]["initial_state_range"])
    )
    initial_state_set = constraints.state_range_to_constraint(
        initial_state_range, params["analysis"]["propagator"]["boundary_type"]
    )

    # Run the analyzer to get reachable set estimates
    if params["analysis"]["estimate_runtime"]:
        # Run the analyzer N times to compute an estimated runtime
        times = np.empty(params["analysis"]["num_calls"])
        final_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        avg_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_reachable_sets = np.empty(
            params["analysis"]["num_calls"], dtype=object
        )
        for num in range(params["analysis"]["num_calls"]):
            logger.info(f"call: {num}")
            t_start = time.time()
            reachable_sets, analyzer_info = analyzer.get_reachable_set(
                initial_state_set, t_max=params["analysis"]["t_max"]
            )
            t_end = time.time()
            times[num] = t_end - t_start

            if num == 0:
                final_error, avg_error, all_error = analyzer.get_error(
                    initial_state_set,
                    reachable_sets,
                    t_max=params["analysis"]["t_max"],
                )
                final_errors[num] = final_error
                avg_errors[num] = avg_error
                all_errors[num] = all_error
                all_reachable_sets[num] = reachable_sets

        stats["runtimes"] = times
        stats["final_step_errors"] = final_errors
        stats["avg_errors"] = avg_errors
        stats["all_errors"] = all_errors
        stats["reachable_sets"] = all_reachable_sets

        logger.info(f"All times: {times}")
        logger.info(f"Avg time: {times.mean()} +/- {times.std()}")
    else:
        # Run analysis once
        t_start = time.time()
        reachable_sets, analyzer_info = analyzer.get_reachable_set(
            initial_state_set, t_max=params["analysis"]["t_max"]
        )
        t_end = time.time()
        logger.info(f"Runtime: {t_end - t_start} sec.")
        stats["reachable_sets"] = reachable_sets

    # Calculate error of estimated reachable sets vs. true ones
    if params["analysis"]["estimate_error"]:
        final_error, avg_error, errors = analyzer.get_error(
            initial_state_set,
            reachable_sets,
            t_max=params["analysis"]["t_max"],
        )
        logger.info(f"Final step approximation error: {final_error}")
        logger.info(f"Avg errors: {avg_error}")
        logger.info(f"All errors: {errors}")

    # Visualize the reachable sets and MC samples
    if (
        params["visualization"]["show_plot"]
        or params["visualization"]["save_plot"]
    ):
        analyzer.visualizer.plot_filename = get_plot_filename(params)
        analyzer.visualize(
            initial_state_set,
            reachable_sets,
            analyzer.propagator.network,
            **analyzer_info,
        )

    return stats, analyzer_info


def main_forward_nick(params: dict) -> Tuple[Dict, Dict]:
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    import nfl_veripy.dynamics as dynamics
    import cl_systems
    from auto_LiRPA import BoundedModule, BoundedTensor
    from utils.robust_training_utils import calculate_reachable_sets, partition_init_set, plot_reachable_sets
    from utils.robust_training_utils import calculate_next_reachable_set, partition_set, calculate_reachable_sets_old
    from utils.robust_training_utils import Analyzer, ReachableSet

    if params["system"]["type"] == 'DoubleIntegrator':
        controller = load_controller(params['system']['type'], params['system']['controller'], params["system"]["dagger"])
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to('cpu')
        ol_dyn.bt_torch = ol_dyn.bt_torch.to('cpu')
        ol_dyn.ct_torch = ol_dyn.ct_torch.to('cpu')
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn)
        

        dummy_input = torch.tensor([[2.75, 0.]], device='cpu')
        
        bounded_cl_sys = BoundedModule(cl_dyn, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device='cpu')
        init_range = np.array([
            [2.5, 3.],
            [-0.25, 0.25]
        ])

        
        # init_ranges = partition_init_set(init_range, params["analysis"]["partitioner"]["num_partitions"])
        # time_reachable_sets = torch.zeros((len(init_ranges), 25, 2, 2))
        # reachable_set, reachable_sets = calculate_next_reachable_set(bounded_cl_sys, init_range, params["analysis"]["partitioner"]["num_partitions"])
        # time_reachable_sets[:, 0, :, :] = reachable_sets
        # time_reachable_sets[0, 1, :, :] = reachable_set
        # reach_sets_np = time_reachable_sets.detach().numpy()

        # plot_reachable_sets(cl_dyn, partition_set(init_range, params["analysis"]["partitioner"]["num_partitions"]), reach_sets_np)
        # print(reach_sets_np)

        time_horizon = 25
        # init_ranges = partition_init_set(init_range, params["analysis"]["partitioner"]["num_partitions"])
        # reach_sets = torch.zeros((len(init_ranges), time_horizon, 2, 2))
        
        # for i, ir in enumerate(init_ranges):
        #     reach_sets[i] = calculate_reachable_sets_old(cl_dyn, ir, time_horizon)

        # reach_sets_np = reach_sets.detach().numpy()
        # plot_reachable_sets(cl_dyn, init_range, reach_sets_np, time_horizon)

        # import pdb; pdb.set_trace()
        # reach_sets = torch.zeros((len(init_ranges), time_horizon, 2, 2))
        # partition_schedule = np.ones((time_horizon, 2), dtype=int)
        # reach_sets, subsets = calculate_reachable_sets(bounded_cl_sys, init_range, partition_schedule)

        
        def condition(input_range):
            return input_range[1, 0] < -1

        
        # reach_sets_np = subsets
        # plot_reachable_sets(cl_dyn, init_range, reach_sets_np, time_horizon)
        init_range = torch.from_numpy(init_range).type(torch.float32)
        analyzer = Analyzer(cl_dyn, time_horizon, init_range)
        # analyzer.set_partition_strategy(0, np.array([5,5]))
        analyzer.set_partition_strategy(0, np.array([3,3]))
        analyzer.set_partition_strategy(8, np.array([2,2]))
        analyzer.set_partition_strategy(12, np.array([1,1]))
        tstart = time.time()
        # reach_set_dict = analyzer.calculate_hybrid_symbolic_reachable_sets()
        # reach_set_dict = analyzer.calculate_N_step_reachable_sets(indices=None) # 3, 4, 5, 6, 7
        reach_set_dict = analyzer.calculate_reachable_sets()
        tend = time.time()
        print('Calculation Time: {}'.format(tend-tstart))

        # analyzer.switch_sets_on_off(condition)
        # import pdb; pdb.set_trace()

        analyzer.plot_reachable_sets()
        analyzer.plot_all_subsets()

        
        # num_trajectories = 50
        # x0 = np.random.uniform(
        #     low=init_range[:, 0],
        #     high=init_range[:, 1],
        #     size=(num_trajectories, cl_dyn.At.shape[0]),
        # )
        # x0_torch = torch.from_numpy(x0).type(torch.float32)
        # xs = [x0]
        # for i in range(time_horizon):
        #     cl_dyn.set_num_steps(i+1)
        #     xt1_torch = cl_dyn.forward(x0_torch)
        #     xs.append(xt1_torch.detach().numpy())
        
        # xs = np.array(xs)
        # fig, ax = plt.subplots()
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "Helvetica"
        # })
        # ax.scatter(xs[:, :, 0], xs[:, :, 1], s=1, c='k')
        # ax.set_xlabel('x1', fontsize=20)
        # ax.set_ylabel('x2', fontsize=20)

            
        # plt.show()

        # tstart = time.time()
        # reach_set_dict = analyzer.calculate_N_step_reachable_sets()
        # tend = time.time()
        # print('N-step Calculation Time: {}'.format(tend-tstart))
        # analyzer.plot_reachable_sets()
        # analyzer.plot_all_subsets()        






        

def main_backward(params: dict) -> tuple[dict, dict]:
    """Runs a backward reachability analysis experiment according to params."""

    np.random.seed(seed=0)
    stats = {}

    if params["system"]["feedback"] != "FullState":
        raise ValueError(
            "Currently only support state feedback for backward reachability."
        )

    dyn = dynamics.get_dynamics_instance(
        params["system"]["type"], params["system"]["feedback"]
    )

    controller = load_controller(
        system=dyn.__class__.__name__,
        model_name=params["system"]["controller"],
    )

    # Set up analyzer (+ parititoner + propagator)
    analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, dyn)
    analyzer.partitioner = params["analysis"]["partitioner"]
    analyzer.propagator = params["analysis"]["propagator"]
    analyzer.visualizer = params["visualization"]

    final_state_range = np.array(
        ast.literal_eval(params["analysis"]["final_state_range"])
    )
    target_set = constraints.state_range_to_constraint(
        final_state_range, params["analysis"]["propagator"]["boundary_type"]
    )

    if params["analysis"]["estimate_runtime"]:
        # Run the analyzer N times to compute an estimated runtime

        times = np.empty(params["analysis"]["num_calls"])
        final_errors = np.empty(params["analysis"]["num_calls"])
        avg_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_backprojection_sets = np.empty(
            params["analysis"]["num_calls"], dtype=object
        )
        target_sets = np.empty(params["analysis"]["num_calls"], dtype=object)
        for num in range(params["analysis"]["num_calls"]):
            logger.info(f"call: {num}")
            t_start = time.time()
            (
                backprojection_sets,
                analyzer_info,
            ) = analyzer.get_backprojection_set(
                target_set,
                t_max=params["analysis"]["t_max"],
                overapprox=params["analysis"]["overapprox"],
            )
            t_end = time.time()
            times[num] = t_end - t_start

            if num == 0:
                (
                    final_error,
                    avg_error,
                    all_error,
                ) = analyzer.get_backprojection_error(
                    target_set,
                    backprojection_sets,
                    t_max=params["analysis"]["t_max"],
                )

                final_errors[num] = final_error
                avg_errors[num] = avg_error
                all_errors[num] = all_error
                all_backprojection_sets[num] = backprojection_sets
                target_sets[num] = target_sets

        stats["runtimes"] = times
        stats["final_step_errors"] = final_errors
        stats["avg_errors"] = avg_errors
        stats["all_errors"] = all_errors
        stats["all_backprojection_sets"] = all_backprojection_sets
        stats["target_sets"] = target_sets
        stats["avg_runtime"] = times.mean()

        logger.info(f"All times: {times}")
        logger.info(f"Avg time: {times.mean()} +/- {times.std()}")
        logger.info(f"Final Error: {final_errors[-1]}")
        logger.info(f"Avg Error: {avg_errors[-1]}")
    else:
        # Run analysis once
        # Run analysis & generate a plot
        backprojection_sets, analyzer_info = analyzer.get_backprojection_set(
            target_set,
            t_max=params["analysis"]["t_max"],
            overapprox=params["analysis"]["overapprox"],
        )
        stats["backprojection_sets"] = backprojection_sets

    # Visualize the reachable sets and MC samples
    if (
        params["visualization"]["show_plot"]
        or params["visualization"]["save_plot"]
    ):
        analyzer.visualizer.plot_filename = get_plot_filename(params)
        analyzer.visualize(
            target_set,
            backprojection_sets,
            analyzer.propagator.network,
            **analyzer_info,
        )

    return stats, analyzer_info


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

    
    if experiment_params["analysis"]["reachability_direction"] == "forward":
        main_forward_nick(experiment_params)
    if experiment_params["analysis"]["reachability_direction"] == "backward":
        main_backward(experiment_params)

# if __name__ == "__main__":
#     controller = load_controller('double_integrator', 'natural_default')
#     import pdb; pdb.set_trace()