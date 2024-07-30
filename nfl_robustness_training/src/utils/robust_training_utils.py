import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ast import literal_eval
from itertools import product

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

class ReachableSet:
    def __init__(self, t, ranges = None, partition_strategy = 'maintain', thread = 0) -> None:
        self.t = t
        
        self.full_set = ranges
        self.subsets = {}
        self.partition_strategy = partition_strategy
        self.get_partitions()
        self.thread = thread
        self.recalculate = True

    def set_range(self, ranges):
        self.full_set = ranges

    def add_subset(self, ranges, index):
        self.subsets[index] = ReachableSet(self.t, ranges, thread=index)
    
    def set_partition_strategy(self, partition_strategy):
        if partition_strategy in ['maintain', 'consolidate'] or isinstance(partition_strategy, np.ndarray):
            self.partition_strategy = partition_strategy
        else:
            raise NotImplementedError
        
    def calculate_full_set(self):
        num_subsets = len(self.subsets)
        num_states = self.subsets[0].full_set.shape[0]
        subset_tensor = torch.zeros((num_subsets, num_states, 2))

        for i, subset in self.subsets.items():
            subset_tensor[i] = subset.full_set
        
        lb, _ = torch.min(subset_tensor[:, :, 0], dim=0)
        ub, _ = torch.max(subset_tensor[:, :, 1], dim=0)
        self.full_set = torch.vstack((lb, ub)).T
        
    
    def get_partitions(self):
        num_partitions = self.partition_strategy
        
        if self.partition_strategy == 'maintain' or self.partition_strategy == 'consolidate':
            pass
        else:
            # num_partitions = np.array(literal_eval(self.partition_strategy))
            self.subsets = {}
            prev_set = self.full_set

            input_shape = self.full_set.shape[:-1]

            slope = torch.divide(
                (prev_set[..., 1] - prev_set[..., 0]), torch.from_numpy(num_partitions).type(torch.float32)
            )

            ranges = []
            output_range = None

            for element in product(
                *[range(num) for num in num_partitions.flatten()]
            ):
                element_ = torch.tensor(element).reshape(input_shape)
                input_range_ = torch.empty_like(prev_set)
                input_range_[..., 0] = prev_set[..., 0] + torch.multiply(
                    element_, slope
                )
                input_range_[..., 1] = prev_set[..., 0] + torch.multiply(
                    element_ + 1, slope
                )

                ranges.append(input_range_,)

            for i, partition in enumerate(ranges):
                self.subsets[i] = ReachableSet(self.t, torch.tensor(partition), thread = i)

    def consolidate(self):
        if self.partition_strategy != 'consolidate':
            pass
        else:
            self.calculate_full_set()
            self.subsets = {0: ReachableSet(self.t, self.full_set, thread=self.thread)}
        
    
    def populate_next_reachable_set(self, bounded_cl_system, next_reachable_set):
        if self.subsets == {} and next_reachable_set.recalculate:
            x = torch.mean(self.full_set, axis=1).reshape(-1,2)
            eps = (self.full_set[:, 1] - self.full_set[:, 0])/2
            ptb = PerturbationLpNorm(eps = eps)
            range_tensor = BoundedTensor(x, ptb)

            lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward", IBP=True)

            reach_set_range = torch.hstack((lb.T, ub.T))
            next_reachable_set.add_subset(reach_set_range, self.thread)
        else:
            for i, subset in self.subsets.items():
                subset.populate_next_reachable_set(bounded_cl_system, next_reachable_set)

        next_reachable_set.calculate_full_set()

    def switch_on_off(self, condition):
        self.recalculate = condition(self.full_set)
    
    def plot_reachable_set(self, ax):
        if self.subsets == {}:
            set_range = self.full_set.detach().numpy()
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        else:
            for i, subset in self.subsets.items():
                subset.plot_reachable_set(ax)
            


class Analyzer:
    def __init__(self, cl_system,  num_steps, initial_range, device='cpu') -> None:
        self.num_steps = num_steps
        self.cl_system = cl_system
        self.device = device

        dummy_input = torch.tensor([[2.75, 0.]], device=device)
        self.bounded_cl_system = BoundedModule(cl_system, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device=device)
        
        self.reachable_sets = {0: ReachableSet(0, initial_range, partition_strategy = 'maintain')}
        for i in range(num_steps):
            self.reachable_sets[i+1] = ReachableSet(i+1)
    
    def set_partition_strategy(self, t, partition_strategy):
        self.reachable_sets[t].set_partition_strategy(partition_strategy)

    def calculate_reachable_sets(self, training = False):
        initial_range = self.reachable_sets[0].full_set
        num_states = initial_range.shape[0]
        reach_set_range = initial_range
        

        for i in range(self.num_steps):
            # prev_set = self.reachable_sets[i].full_set

            # partitions = self.reachable_sets[i].get_partitions()


            # x = torch.mean(reach_set_range, axis=1).reshape(-1,2)
            # eps = (reach_set_range[:, 1] - reach_set_range[:, 0])/2
            # ptb = PerturbationLpNorm(eps = eps) 
            # prev_set = BoundedTensor(x, ptb)

            # lb, ub = self.bounded_cl_system.compute_bounds(x=(prev_set,), method="backward", IBP=True)

            # reach_set_range = torch.hstack((lb.T, ub.T))
            
            # self.reachable_sets[i+1] = ReachableSet(i+1, reach_set_range)
            # import pdb; pdb.set_trace()
            self.reachable_sets[i].get_partitions()
            self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_system, self.reachable_sets[i+1])
            self.reachable_sets[i+1].consolidate()
            # import pdb; pdb.set_trace()
            
        return self.reachable_sets
    
    def calculate_N_step_reachable_sets(self, training = False):
        from copy import deepcopy
        initial_range = self.reachable_sets[0].full_set
        num_states = initial_range.shape[0]
        reach_set_range = initial_range

        for i in range(self.num_steps):
            print("Calculating set {}".format(i))
            self.cl_system.set_num_steps(i+1)
            cl_system = deepcopy(self.cl_system)
            cl_system.set_num_steps(i+1)
            dummy_input = torch.tensor([[2.75, 0.]], device=self.device)
            bounded_cl_system = BoundedModule(cl_system, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device=self.device)

            self.reachable_sets[i].get_partitions()
            self.reachable_sets[0].populate_next_reachable_set(bounded_cl_system, self.reachable_sets[i+1])
        
        return self.reachable_sets


    def get_all_ranges(self):
        all_ranges = []
        for i, reachable_set in self.reachable_sets.items():
            if reachable_set.subsets == {}:
                all_ranges.append(reachable_set.full_set)
            else:
                for _, reachable_subset in reachable_set.subsets.items():
                    all_ranges.append(reachable_subset.full_set)
                     
        return torch.stack(all_ranges, dim=0)
    

    def get_all_reachable_sets(self):
        all_sets = []
        for i, reachable_set in self.reachable_sets.items():
            if reachable_set.subsets == {}:
                all_sets.append(reachable_set)
            else:
                for _, reachable_subset in reachable_set.subsets.items():
                    all_sets.append(reachable_subset)
                     
        return all_sets
    

    def switch_sets_on_off(self, condition):
        all_sets = self.get_all_reachable_sets()
        for reachable_set in all_sets:
            reachable_set.switch_on_off(condition)


    def plot_reachable_sets(self, num_trajectories=50):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        initial_set = self.reachable_sets[0].full_set
        total_reachable_sets = self.reachable_sets


        
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

        # import pdb; pdb.set_trace()
        for i, reachable_sets in total_reachable_sets.items():
            # import pdb; pdb.set_trace()
            # for _, set_range in reachable_sets:
            set_range = reachable_sets.full_set.detach().numpy()
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        ax.set_xlabel('x1', fontsize=20)
        ax.set_ylabel('x2', fontsize=20)

            
        plt.show()

    
    def plot_all_subsets(self, num_trajectories=50):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        initial_set = self.reachable_sets[0].full_set
        total_reachable_sets = self.reachable_sets


        
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

        # xy = initial_set[:, 0]
        # width, height = initial_set[:, 1] - initial_set[:, 0]
        # rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect)

        # import pdb; pdb.set_trace()
        for i, reachable_sets in total_reachable_sets.items():
            reachable_sets.plot_reachable_set(ax)

        ax.set_xlabel('x1', fontsize=20)
        ax.set_ylabel('x2', fontsize=20)

            
        plt.show()


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