import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from ast import literal_eval
from itertools import product
from copy import deepcopy

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import cl_systems
import time

class ReachableSet:
    """
    A class to represent hyper-rectangular reachable set over-approximations for NFLs.
    There is some complexity related to partitioning that will be removed in the future
    as it is not necessary fo CARV or to generate the results in the L-CSS paper.
    
    Attributes:
    -----------
    t : int
        The current time step of the reachable set.
    full_set : torch.Tensor
        The full set of reachable states at the current time step.
    subsets : dict
        A dictionary of subsets of the reachable set.
    partition_strategy : str or np.ndarray
        The strategy for partitioning the reachable set. In practice only use this to partition the
        intial set, but can be used to partition mid-way through the time horizon.
    thread : int
        An index variable to track a partitioned subset's parent set.
    recalculate : bool
        A flag indicating whether to recalculate the reachable set.
    device : str
        The device to use for tensor computations.
    t_parent : int
        The parent time step.
    symbolic : bool
        A flag indicating whether the reachable set is symbolic.
    populated : bool
        A flag indicating whether the reachable set is populated.
    Methods:
    --------
    set_range(ranges):
        Sets the range of the full set.
    add_subset(ranges, index):
        Adds a subset to the reachable set.
    get_thread(thread):
        Gets the reachable set for a specific thread.
    set_partition_strategy(partition_strategy):
        Sets the partition strategy.
    calculate_full_set():
        Calculates the full set from the subsets.
    get_partitions():
        Partitions the full set according to the partition strategy.
    consolidate():
        Consolidates the subsets into the full set.
    populate_next_reachable_set(bounded_cl_system, next_reachable_set, training=False):
        Populates the next reachable set.
    switch_on_off(condition, thread=0):
        Switches the recalculation flag based on a condition.
    plot_reachable_set(ax, plot_partitions=True, edgecolor=None, facecolor='none', alpha=0.1):
        Plots the reachable set.
    sample_from_reachable_set(cl_system, num_steps=1, num_trajectories=100, sample_corners=False):
        Samples trajectories from the reachable set.
    """
    def __init__(self, t, ranges, device='cpu') -> None:
        self.t = t
        self.subsets = ranges.unsqueeze(0)
        self.full_set = ranges
        self.recalculate = True
        self.device = device
        self.symbolic = True
        self.populated = False

    def update_full_set(self, ranges):
        self.full_set = ranges

    def update(self, subsets):
        self.subsets = subsets

        self.full_set = torch.vstack(
            [torch.min(subsets[:, :, 0], dim=0)[0], torch.max(subsets[:, :, 1], dim=0)[0]]
        ).T.to(self.device)
        
    # def calculate_full_set(self):
    #     num_subsets = len(self.subsets)
    #     num_states = self.subsets[0].full_set.shape[0]
    #     subset_tensor = torch.zeros((num_subsets, num_states, 2), device=self.device)

    #     for i, subset in self.subsets.items():
    #         subset_tensor[i] = subset.full_set
        
    #     lb, _ = torch.min(subset_tensor[:, :, 0], dim=0)
    #     ub, _ = torch.max(subset_tensor[:, :, 1], dim=0)
    #     self.full_set = torch.vstack((lb, ub)).T.to(self.device)
        
    
    def partition(self, num_partitions=None):
        
        if num_partitions is None:
            pass
        elif isinstance(num_partitions, np.ndarray) and len(num_partitions) == self.full_set.shape[:-1][0]:
            # num_partitions = np.array(literal_eval(self.partition_strategy))
            prev_set = self.full_set

            input_shape = self.full_set.shape[:-1]

            slope = torch.divide(
                (prev_set[..., 1] - prev_set[..., 0]), torch.from_numpy(num_partitions).type(torch.float32).to(self.device)
            )

            self.subsets = torch.zeros((np.prod(num_partitions), *input_shape, 2), dtype=torch.float32, device=self.device)

            for i, element in enumerate(product(
                *[range(num) for num in num_partitions.flatten()])
            ):
                element_ = torch.tensor(element).reshape(input_shape).to(self.device)
                input_range_ = torch.empty_like(prev_set)
                input_range_[..., 0] = prev_set[..., 0] + torch.multiply(
                    element_, slope
                )
                input_range_[..., 1] = prev_set[..., 0] + torch.multiply(
                    element_ + 1, slope
                )

                self.subsets[i] = input_range_

            # for i, partition in enumerate(ranges):
            #     self.subsets[i] = torch.tensor(partition).to(self.device)

    def consolidate(self):
        if self.partition_strategy != 'consolidate':
            pass
        else:
            self.calculate_full_set()
            self.subsets = {0: ReachableSet(self.t, self.full_set, thread=self.thread, device=self.device)}
        
    
    def populate_next_reachable_set(self, bounded_cl_system, next_reachable_set, training=False):
        """
        Populate a reachable set based on the current state and bounded closed-loop system.
        Args:
            bounded_cl_system: The bounded closed-loop system used to compute bounds.
            next_reachable_set: The set to be populated with the next reachable states.
            training (bool, optional): If True, additional timing information and debug prints are enabled. Additionally, 
            a more conservative calculation is used. Defaults to False.
        Returns:
            None
        """
        if next_reachable_set.recalculate and len(self.subsets) == 1:
            x = torch.mean(self.subsets, axis=2)
            eps = (self.subsets[0, :, 1] - self.subsets[0, :, 0])/2
            ptb = PerturbationLpNorm(eps = eps)
            range_tensor = BoundedTensor(x, ptb)
            import pdb; pdb.set_trace()
            if training:
                print("crown fast")
                tstart = time.time()
                lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method=None, IBP=True)
                tend = time.time()
                print("first calc: {}".format(tend-tstart))
                tstart = time.time()
                lb, _ = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward", IBP=False,  bound_upper=False)
                tend = time.time()
                print("second calc: {}".format(tend-tstart))
                # lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward")
            else:
                lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward")
            
            # if next_reachable_set.populated:
            #     lb_ = torch.hstack((lb.T, next_reachable_set.full_set))
            #     ub_ = torch.hstack((ub.T, next_reachable_set.full_set))
            #     lb = torch.max(lb_[:,[0, 1]], axis = 1)[0].reshape((1, -1))
            #     ub = torch.min(ub_[:,[0, 2]], axis = 1)[0].reshape((1, -1))
                
            # import pdb; pdb.set_trace()
            reach_set_subsets = torch.transpose(torch.transpose(torch.stack((lb, ub)), 0, 1), 1, 2)
            next_reachable_set.update(reach_set_subsets)
        elif len(self.subsets) > 1:
            # If the set is partitioned, we need to calculate the bounds for each subset
            reach_set_subsets = torch.zeros((len(self.subsets), *self.full_set.shape), device=self.device)
            for i, subset in enumerate(self.subsets):
                x = torch.mean(subset, axis=1).unsqueeze(0)
                eps = (subset[:, 1] - subset[:, 0])/2
                ptb = PerturbationLpNorm(eps = eps)
                range_tensor = BoundedTensor(x, ptb)
                # import pdb; pdb.set_trace()
                if training:
                    print("crown fast")
                    tstart = time.time()
                    lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method=None, IBP=True)
                    tend = time.time()
                    print("first calc: {}".format(tend-tstart))
                    tstart = time.time()
                    lb, _ = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward", IBP=False, bound_upper=False)
                    tend = time.time()
                    print("second calc: {}".format(tend-tstart))
                else:
                    lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward")
                
                # import pdb; pdb.set_trace()
                reach_set_subsets[i] = torch.transpose(torch.transpose(torch.stack((lb, ub)), 0, 1), 1, 2)[0]

            next_reachable_set.update(reach_set_subsets)

        # next_reachable_set.calculate_full_set()
        # next_reachable_set.t_parent = self.t
        if next_reachable_set.t - self.t == 1:
            next_reachable_set.symbolic = False
        next_reachable_set.populated = True

    def switch_on_off(self, condition, thread=0):
        self.recalculate = not condition(self.full_set)
        if self.recalculate:
            if self.full_set[1, 0] < -1:
                print("Recalculating")
                print(self.full_set)
    
    def plot_reachable_set(self, ax, plot_partitions = True, edgecolor = None, facecolor = 'none', alpha = 0.1):
        if self.subsets == {}:
            set_range = self.full_set.cpu().detach().numpy()
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            if edgecolor is None:
                if self.recalculate:
                    edgecolor = 'orange'
                else:
                    edgecolor = 'b'
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
            ax.add_patch(rect)
        else:
            for i, subset in self.subsets.items():
                subset.plot_reachable_set(ax, plot_partitions=plot_partitions)

    def sample_from_reachable_set(self, cl_system, num_steps=1, num_trajectories=100, sample_corners=False):
        np.random.seed(0)
        num_states = cl_system.At.shape[0]
        if sample_corners:
            set_range = self.full_set.cpu().detach().numpy()
            test = np.meshgrid(*[set_range.T[:, i] for i in range(set_range.shape[0])])  
            corners = np.array(np.meshgrid([set_range.T[:, i] for i in range(set_range.shape[0])])).T.reshape(-1, num_states)
            # corners = np.array(np.meshgrid(set_range.T[:,0], set_range.T[:,1])).T.reshape(-1, num_states)
            num_trajectories -= len(corners)
            np.meshgrid()
        

        x0s = np.random.uniform(
            low=self.full_set[:, 0].cpu().detach().numpy(),
            high=self.full_set[:, 1].cpu().detach().numpy(),
            size=(num_trajectories, num_states),
        )
        
        if sample_corners:
            xs = np.vstack((corners, x0s))
        else:
            xs = x0s
        
        xt = xs
        for _ in range(num_steps):
            u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller.cpu())
            xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
            xt = xt1

            xs = np.vstack((xs, xt1))
        
        return xs
            


class Analyzer:
    """
    Analyzer class for calculating reachable sets of a closed-loop system.
    Attributes:
        num_steps (int): Number of steps for reachability analysis.
        cl_system (ClosedLoopDynamics): Closed-loop system to analyze.
        device (str): Device to use for computation ('cpu' or 'cuda').
        max_diff (int): Maximum difference for symbolic reachability (i.e., maximum symbolic horizon).
        save_info (bool): Flag to save snapshots of reachable sets.
        reachable_sets (dict): Dictionary of reachable sets for each time step.
        bounded_cl_systems (dict): Dictionary of bounded modules for each time step shorter than max_diff.
    Methods:
        RSOA Calculation Approaches:
            calculate_reachable_sets(training=False, autorefine=False, visualize=False, condition=None): Calculate reachable sets for the closed-loop system.
            refine(reachable_set, condition, snapshots, t, force=False): Refine a reachable set based on a condition.
            calculate_N_step_reachable_sets(training=False, indices=None, condition=None): Calculate N-step reachable sets for the closed-loop system.
            hybr(visualize=False, condition=None): Hybrid-symbolic reachability analysis with a pre-defined schedule.
            pseudo(visualize=False, condition=None, num_trajectories=100): Pseudo-reachability analysis.
        Helper Functions:
            set_partition_strategy(t, partition_strategy): Set the partition strategy for a reachable set at time t.
            get_parent_set(reachable_set): Get the parent set of a given reachable set.
            get_all_ranges(): Get all ranges of reachable sets.
            get_all_reachable_sets(): Get all reachable sets.
            is_safe(constraint_list): Check if the system is safe based on a list of constraints.
    """
    def __init__(self, cl_system,  num_steps, initial_range, max_diff=10, device='cpu', save_info=True) -> None:
        self.num_steps = num_steps
        self.cl_system = cl_system
        self.device = device
        self.max_diff = max_diff
        self.save_info = save_info


        if cl_system.dynamics.name == 'DoubleIntegrator':
            dummy_input = torch.tensor([[2.75, 0.]], device=device)
        elif cl_system.dynamics.name == 'Unicycle_NL':
            dummy_input = torch.tensor([[-12.5, 3.5, -0.5]], device=device)
        elif cl_system.dynamics.name == 'Quadrotor_NL':
            dummy_input = torch.tensor([[-10, 3, 1., 1., 0., 0.,]], device=device)
        bound_opts = {
            'relu': "CROWN-IBP",
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False,
            'sparse_features_alpha': False,
            'sparse_spec_alpha': False,
            # 'zero-lb': True,
            # 'same-slope': False,
        }
        self.reachable_sets = {0: ReachableSet(0, initial_range, device=device)}
        self.bounded_cl_systems = {0: BoundedModule(cl_system, dummy_input, bound_opts=bound_opts, device=device)}
        for i in range(num_steps):
            self.reachable_sets[i+1] = ReachableSet(i+1, torch.zeros((6, 2), device=device), device=device)
            cl_system.set_num_steps(i+2)
            if i < self.max_diff:
                self.bounded_cl_systems[i+1] = BoundedModule(cl_system, dummy_input, bound_opts=bound_opts, device=device)


    def calculate_reachable_sets(self, num_partitions = None, training = False, autorefine = False, visualize = False, condition = None):
        """
        Calculate the reachable sets for a given system over a number of steps.
        Args:
            training: (bool, optional) If True, the function will operate in training mode. Default is False.
            autorefine: (bool, optional) If True, CARV is used; the function will automatically refine the reachable sets. Default is False.
            visualize: (bool, optional) If True, the function will visualize the reachable sets during the refinement process. Default is False.
            condition: (callable, optional) A function that takes a state range and returns a boolean indicating whether the states satisfy safety. Default is None.
        Returns:
            reachable_sets: A list of reachable sets calculated over the number of steps.
            snapshots: A list of snapshots containing information about the reachable sets at each step.
        Raises:
            RuntimeError: If there is an error in calculating the reachable set at any time step.
        """
        initial_range = self.reachable_sets[0].full_set
        snapshots = []
        if self.save_info:
            current_snapshot = {}
            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic) for _, reachable_set in self.reachable_sets.items()])
            current_snapshot['time'] = 1
            current_snapshot['child_idx'] = 0
            current_snapshot['parent_idx'] = 0
        
        self.reachable_sets[0].partition(num_partitions)
        for i in range(self.num_steps):
            current_snapshot = {}
            # self.reachable_sets[i].get_partitions()
            # Error handling to catch issues with bounding tan function in auto_LiRPA; if bounds are too big, use refinement
            try:
                tstart = time.time()
                # By default, calculate concrete reachable sets
                self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[0], self.reachable_sets[i+1], training)
                tend = time.time()
                # self.reachable_sets[i+1].consolidate()
                if self.save_info:
                    current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                    current_snapshot['time'] = tend - tstart
                    current_snapshot['child_idx'] = i + 1
                    current_snapshot['parent_idx'] = i
                    snapshots.append(current_snapshot)
            except RuntimeError:
                # If bounds on prev set are too big, refine it first, then calculate next step concretely
                print("Error in calculating set at time {}".format(i+1))
                self.refine(self.reachable_sets[i], condition, snapshots, i, force=True)
                tstart = time.time()
                self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[0], self.reachable_sets[i+1], training)
                tend = time.time()
                # self.reachable_sets[i+1].consolidate()
                if self.save_info:
                    current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                    current_snapshot['time'] = tend - tstart
                    current_snapshot['child_idx'] = i + 1
                    current_snapshot['parent_idx'] = i
                    snapshots.append(current_snapshot)

            if autorefine:
                if visualize:
                    self.plot_reachable_sets()
                
                # If using CARV, call refine function to automatically refine reachable sets
                refined = self.refine(self.reachable_sets[i+1], condition, snapshots, i)
                    
                if visualize:
                    self.plot_reachable_sets()
            
        return self.reachable_sets, snapshots
    
    
    def refine(self, reachable_set, condition, snapshots, t, force=False):
        """
        Refines the reachable set based on a given condition and updates snapshots.
        Args:
            reachable_set (ReachableSet): The current reachable set to be refined.
            condition (callable): A function that takes a reachable set and returns a boolean indicating if the condition is met.
            snapshots (list): A list to store snapshots of the reachable sets during the refinement process.
            t (int): The current time step.
            force (bool, optional): If True, forces the refinement process even if the condition is met. Default is False.
        Returns:
            refined: (bool) True if the reachable set was refined, False otherwise.
        """
        refined = not condition(reachable_set.full_set) or force
        tf = reachable_set.t
        min_diff = 2
        max_diff = self.max_diff

        if force and not reachable_set.symbolic:
            marching_back = False
            while not marching_back:
                next_idx = max(tf - self.max_diff, 0)
                print("marching back from set {} to set {}".format(tf, next_idx))
                if self.reachable_sets[next_idx].symbolic:

                    print("{} is symbolic, marching back".format(next_idx))
                    marching_back = True
                    if self.save_info:
                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['child_idx'] = tf
                        current_snapshot['parent_idx'] = next_idx

                    try:
                        tstart = time.time()
                        self.reachable_sets[next_idx].populate_next_reachable_set(self.bounded_cl_systems[tf - next_idx - 1], reachable_set)
                        reachable_set.symbolic = True
                        tend = time.time()

                        if self.save_info:
                            current_snapshot['time'] = tend - tstart
                            snapshots.append(current_snapshot)

                        if self.save_info:
                            current_snapshot = {}
                            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                            current_snapshot['time'] = 0
                            current_snapshot['child_idx'] = next_idx + 1
                            current_snapshot['parent_idx'] = next_idx
                            snapshots.append(current_snapshot)
                    except:
                        print("Error in recalculating set {} from time {}".format(tf, next_idx))
                        pass

                else:
                    self.refine(self.reachable_sets[next_idx], condition, snapshots, next_idx, force=True)
        else:
            final_idx = max(tf - self.max_diff, 0)
            i = tf - 2
            if not condition(reachable_set.full_set):
                print("Collision detected at t = {}".format(tf))
            while i >= final_idx and not condition(reachable_set.full_set):
                diff = tf - i
                if self.reachable_sets[i].symbolic and (diff >= min_diff or i == 0):
                    print("recalculating set {} from time {}".format(tf, i))
                    if self.save_info:
                        current_snapshot = {}
                        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                        current_snapshot['child_idx'] = tf
                        current_snapshot['parent_idx'] = i

                    try:
                        tstart = time.time()
                        self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[tf - i - 1], reachable_set)
                        reachable_set.symbolic = True
                        tend = time.time()
                        if self.save_info:
                            current_snapshot['time'] = tend - tstart
                            snapshots.append(current_snapshot)

                            current_snapshot = {}
                            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                            current_snapshot['time'] = 0
                            current_snapshot['child_idx'] = i+1
                            current_snapshot['parent_idx'] = i
                            snapshots.append(current_snapshot)
                    except:
                        print("Error in recalculating set {} from time {}".format(tf, i))
                        pass

                elif diff == max_diff:
                    print("cannot do full symbolic from tf = {}, starting march".format(tf))

                    if i >=  1:
                        self.refine(self.reachable_sets[i], condition, snapshots, i, force=True)
                        i = i + 1
                
                if self.save_info:
                    current_snapshot = {}
                    current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                    current_snapshot['time'] = 0
                    current_snapshot['child_idx'] = tf
                    current_snapshot['parent_idx'] = i
                    snapshots.append(current_snapshot)

                i -= 1

        return refined
    

    def calculate_N_step_reachable_sets(self, training = False, indices = None, condition = None):
        if indices is None:
            indices = list(range(self.num_steps))

        snapshots = []

        for i in indices:
            current_snapshot = {}
            print("Calculating set {}".format(i))
            tstart = time.time()
            self.reachable_sets[0].populate_next_reachable_set(self.bounded_cl_systems[i], self.reachable_sets[i+1])
            print(self.reachable_sets[i+1].full_set)
            tend = time.time()
            print('Calculation Time: {}'.format(tend-tstart))
            if self.save_info:
                current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), True, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                current_snapshot['time'] = tend - tstart
                current_snapshot['child_idx'] = i + 1
                current_snapshot['parent_idx'] = 0
                snapshots.append(current_snapshot)

        return self.reachable_sets, snapshots
    
    


    def hybr(self, visualize = False, condition = None):
        initial_range = self.reachable_sets[0].full_set
        snapshots = []
        current_snapshot = {}
        current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic) for _, reachable_set in self.reachable_sets.items()])
        current_snapshot['time'] = 1
        current_snapshot['child_idx'] = 0
        current_snapshot['parent_idx'] = 0
        
        for i in range(self.num_steps):
            tf = self.reachable_sets[i+1].t
            parent_idx = i
            if (i + 1) % self.max_diff != 0:
                tstart = time.time()
                self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[0], self.reachable_sets[i+1])
                tend = time.time()
            else:
                parent_idx = max(tf - self.max_diff, 0)
                tstart = time.time()
                self.reachable_sets[parent_idx].populate_next_reachable_set(self.bounded_cl_systems[tf - parent_idx - 1], self.reachable_sets[i+1])
                tend = time.time()

            current_snapshot = {}
            current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), reachable_set.symbolic, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
            current_snapshot['time'] = tend - tstart
            current_snapshot['child_idx'] = i + 1
            current_snapshot['parent_idx'] = parent_idx
            snapshots.append(current_snapshot)
        
            if visualize:
                self.plot_reachable_sets()

            
        return self.reachable_sets, snapshots

    def pseudo(self, visualize = False, condition = None, num_trajectories = 100):
        snapshots = []
        xs = self.reachable_sets[0].sample_from_reachable_set(self.cl_system, num_steps=self.num_steps, sample_corners=False)
        xs_sorted = xs.reshape((self.num_steps+1, num_trajectories, self.cl_system.dynamics.num_states))
        for i in range(self.num_steps):
            current_snapshot = {}
            underapprox_state_range = np.vstack((np.min(xs_sorted[i], axis = 0), np.max(xs_sorted[i], axis = 0))).T
            pseudo_reachable_set = ReachableSet(i, ranges=torch.tensor(underapprox_state_range, dtype = torch.float32))
            tstart = time.time()
            pseudo_reachable_set.populate_next_reachable_set(self.bounded_cl_systems[0], self.reachable_sets[i+1], training=True)
            tend = time.time()

            if self.save_info:
                current_snapshot['reachable_sets'] = deepcopy([(reachable_set.full_set.cpu().detach().numpy(), True, not condition(reachable_set.full_set)) for _, reachable_set in self.reachable_sets.items()])
                current_snapshot['time'] = tend - tstart
                current_snapshot['child_idx'] = i + 1
                current_snapshot['parent_idx'] = 0
                snapshots.append(current_snapshot)
        # import pdb; pdb.set_trace()
        return self.reachable_sets, snapshots
        t_curr = tstart_ttt + bsteps
        if t_est >= self.max_diff:
            return self.max_diff, 'jump'
        else:
            return t_curr+1, 'search'


    def set_partition_strategy(self, t, partition_strategy):
        self.reachable_sets[t].set_partition_strategy(partition_strategy)

    def get_parent_set(self, reachable_set):
        if reachable_set.t == 0:
            return IndexError
        
        return self.reachable_sets[reachable_set.t - 1].get_thread(reachable_set.thread)
    

    def get_all_ranges(self):
        all_ranges = []
        for i, reachable_set in self.reachable_sets.items():
            if reachable_set.subsets == {}:
                if reachable_set.full_set is None:
                    pass
                else:
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
    

    def is_safe(self, constraint_list):
        for constraint in constraint_list:
            for i, reachable_set in self.reachable_sets.items():
                if not constraint(reachable_set.full_set):
                    return False
                
        return True