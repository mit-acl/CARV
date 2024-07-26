import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ast import literal_eval
from itertools import product

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

def partition_init_set(initial_set, num_partitions):
    if isinstance(num_partitions, str):
        num_partitions = np.array(literal_eval(num_partitions))
    input_shape = initial_set.shape[:-1]
    
    slope = np.divide(
        (initial_set[..., 1] - initial_set[..., 0]), num_partitions
    )

    ranges = []
    output_range = None

    for element in product(
        *[range(num) for num in num_partitions.flatten()]
    ):
        element_ = np.array(element).reshape(input_shape)
        input_range_ = np.empty_like(initial_set)
        input_range_[..., 0] = initial_set[..., 0] + np.multiply(
            element_, slope
        )
        input_range_[..., 1] = initial_set[..., 0] + np.multiply(
            element_ + 1, slope
        )

        ranges.append(input_range_,)
    
    return np.array(ranges)

def calculate_reachable_sets_old(cl_system, initial_set, time_horizon):
    # lb, ub = cl_system.compute_bounds(x=(initial_set,), method="backward")
    num_states = initial_set.shape[0]
    reachable_sets = torch.zeros((time_horizon, num_states, 2))
    x = torch.from_numpy(np.mean(initial_set, axis=1).reshape(-1,2)).type(torch.float32)
    eps = torch.from_numpy((initial_set[:, 1] - initial_set[:, 0])/2).type(torch.float32)
    ptb = PerturbationLpNorm(eps = eps)
    
    for i in range(time_horizon):
        prev_set = BoundedTensor(x, ptb)
        lb, ub = cl_system.compute_bounds(x=(prev_set,), method="backward", IBP=True)

        reach_set = torch.hstack((lb.T, ub.T))
        reachable_sets[i] = reach_set

        x = torch.mean(reach_set, axis=1).reshape(-1,2)
        eps = (reach_set[:, 1] - reach_set[:, 0])/2
        ptb = PerturbationLpNorm(eps = eps) 

    return reachable_sets


def calculate_reachable_sets(cl_system, initial_set, partition_schedule):
    # lb, ub = cl_system.compute_bounds(x=(initial_set,), method="backward")
    num_states = initial_set.shape[0]
    time_horizon = len(partition_schedule)

    reachable_sets = torch.zeros((time_horizon, num_states, 2))
    all_subsets = []
    
    prev_range = initial_set

    for i in range(time_horizon):
        num_partitions = partition_schedule[i]
        reachable_set, reachable_set_subsets = calculate_next_reachable_set(cl_system, prev_range, num_partitions)

        # lb, ub = cl_system.compute_bounds(x=(prev_set,), method="backward")

        # reach_set = torch.hstack((lb.T, ub.T))
        reachable_sets[i] = reachable_set
        
        for rss in reachable_set_subsets:
            all_subsets.append(rss.detach().numpy())

        prev_range = reachable_set.detach().numpy()
        # x = torch.mean(reach_set, axis=1).reshape(-1,2)
        # eps = (reach_set[:, 1] - reach_set[:, 0])/2

    return reachable_sets, np.array(all_subsets).reshape((-1, 1, num_states, 2))







def partition_set(prev_set, num_partitions):
    if isinstance(num_partitions, str):
        num_partitions = np.array(literal_eval(num_partitions))
    input_shape = prev_set.shape[:-1]
    
    slope = np.divide(
        (prev_set[..., 1] - prev_set[..., 0]), num_partitions
    )

    ranges = []
    output_range = None

    for element in product(
        *[range(num) for num in num_partitions.flatten()]
    ):
        element_ = np.array(element).reshape(input_shape)
        input_range_ = np.empty_like(prev_set)
        input_range_[..., 0] = prev_set[..., 0] + np.multiply(
            element_, slope
        )
        input_range_[..., 1] = prev_set[..., 0] + np.multiply(
            element_ + 1, slope
        )

        ranges.append(input_range_,)
    
    return np.array(ranges)

def calculate_next_reachable_set(cl_system, prev_range, num_partitions):
    # lb, ub = cl_system.compute_bounds(x=(initial_set,), method="backward")
    num_states = prev_range.shape[0]

    prev_ranges = partition_set(prev_range, num_partitions)
    reachable_set_subsets = torch.zeros((len(prev_ranges), num_states, 2))

    for i, part in enumerate(prev_ranges):
        x = torch.from_numpy(np.mean(part, axis=1).reshape(-1,2)).type(torch.float32)
        eps = torch.from_numpy((part[:, 1] - part[:, 0])/2).type(torch.float32)
        ptb = PerturbationLpNorm(eps = eps)

        prev_set = BoundedTensor(x, ptb)
        lb, ub = cl_system.compute_bounds(x=(prev_set,), method="backward", IBP=True)

        reach_set = torch.hstack((lb.T, ub.T))
        reachable_set_subsets[i] = reach_set

        x = torch.mean(reach_set, axis=1).reshape(-1,2)
        eps = (reach_set[:, 1] - reach_set[:, 0])/2

    lb, _ = torch.min(reachable_set_subsets[:, :, 0], dim=0)
    ub, _ = torch.max(reachable_set_subsets[:, :, 1], dim=0)
    reachable_set = torch.vstack((lb, ub)).T

    return reachable_set, reachable_set_subsets








# def plot_reachable_sets(cl_system, total_initial_set, total_reachable_sets, num_trajectories=50):
#     fig, ax = plt.subplots()
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Helvetica"
#     })
    
#     for i, reachable_sets in enumerate(total_reachable_sets):
#         initial_set = total_initial_set[i]
    
#         num_steps = len(reachable_sets)

#         np.random.seed(0)
#         x0s = np.random.uniform(
#             low=initial_set[:, 0],
#             high=initial_set[:, 1],
#             size=(num_trajectories, cl_system.At.shape[0]),
#         )
#         xt = x0s
#         xs = x0s
#         for _ in range(num_steps):
#             u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller)
#             xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
#             xt = xt1

#             xs = np.vstack((xs, xt1))
        
        
#         ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

#         xy = initial_set[:, 0]
#         width, height = initial_set[:, 1] - initial_set[:, 0]
#         rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
#         ax.add_patch(rect)

#         for set_range in reachable_sets:
#             xy = set_range[:, 0]
#             width, height = set_range[:, 1] - set_range[:, 0]
#             rect = Rectangle(xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
#             ax.add_patch(rect)

#     ax.set_xlabel('x1', fontsize=20)
#     ax.set_ylabel('x2', fontsize=20)

        
#     plt.show()

def plot_reachable_sets(cl_system, initial_set, total_reachable_sets, time_horizon, num_trajectories=50):
    fig, ax = plt.subplots()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    num_steps = time_horizon
    np.random.seed(0)
    x0s = np.random.uniform(
        low=initial_set[:, 0],
        high=initial_set[:, 1],
        size=(num_trajectories, cl_system.At.shape[0]),
    )
    xt = x0s
    xs = x0s
    for _ in range(num_steps):
        u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller)
        xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
        xt = xt1

        xs = np.vstack((xs, xt1))
    
    ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

    xy = initial_set[:, 0]
    width, height = initial_set[:, 1] - initial_set[:, 0]
    rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    for i, reachable_sets in enumerate(total_reachable_sets):
        for set_range in reachable_sets:
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)

        
    plt.show()