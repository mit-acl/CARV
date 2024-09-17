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
    def __init__(self, t, ranges = None, partition_strategy = 'maintain', thread = 0, device='cpu') -> None:
        self.t = t
        
        if ranges is None:
            ranges = torch.tensor([[0, 0], [0, 0]], device=device)
        self.full_set = ranges
        self.subsets = {}
        self.partition_strategy = partition_strategy
        self.get_partitions()
        self.thread = thread
        self.recalculate = True
        self.device = device
        self.t_parent = -1
        self.symbolic = True

    def set_range(self, ranges):
        self.full_set = ranges

    def add_subset(self, ranges, index):
        self.subsets[index] = ReachableSet(self.t, ranges, thread=index)

    def get_thread(self, thread):
        if thread == 0 and self.subsets == {}:
            return self
        else:
            return self.subsets[thread]
        
    
    def set_partition_strategy(self, partition_strategy):
        if partition_strategy in ['maintain', 'consolidate'] or isinstance(partition_strategy, np.ndarray):
            self.partition_strategy = partition_strategy
        else:
            raise NotImplementedError
        
    def calculate_full_set(self):
        num_subsets = len(self.subsets)
        num_states = self.subsets[0].full_set.shape[0]
        subset_tensor = torch.zeros((num_subsets, num_states, 2), device=self.device)

        for i, subset in self.subsets.items():
            subset_tensor[i] = subset.full_set
        
        lb, _ = torch.min(subset_tensor[:, :, 0], dim=0)
        ub, _ = torch.max(subset_tensor[:, :, 1], dim=0)
        self.full_set = torch.vstack((lb, ub)).T.to(self.device)
        
    
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
                (prev_set[..., 1] - prev_set[..., 0]), torch.from_numpy(num_partitions).type(torch.float32).to(self.device)
            )

            ranges = []
            output_range = None

            for element in product(
                *[range(num) for num in num_partitions.flatten()]
            ):
                element_ = torch.tensor(element).reshape(input_shape).to(self.device)
                input_range_ = torch.empty_like(prev_set)
                input_range_[..., 0] = prev_set[..., 0] + torch.multiply(
                    element_, slope
                )
                input_range_[..., 1] = prev_set[..., 0] + torch.multiply(
                    element_ + 1, slope
                )

                ranges.append(input_range_,)

            for i, partition in enumerate(ranges):
                self.subsets[i] = ReachableSet(self.t, torch.tensor(partition).to(self.device), thread = i, device = self.device)

    def consolidate(self):
        if self.partition_strategy != 'consolidate':
            pass
        else:
            self.calculate_full_set()
            self.subsets = {0: ReachableSet(self.t, self.full_set, thread=self.thread, device=self.device)}
        
    
    def populate_next_reachable_set(self, bounded_cl_system, next_reachable_set, training=False):
        if self.subsets == {} and next_reachable_set.recalculate:
            x = torch.mean(self.full_set, axis=1).reshape((1, -1))
            eps = (self.full_set[:, 1] - self.full_set[:, 0])/2
            ptb = PerturbationLpNorm(eps = eps)
            range_tensor = BoundedTensor(x, ptb)
            # import pdb; pdb.set_trace()
            if training:
                # print("crown fast")
                # tstart = time.time()
                # lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method=None, IBP=True)
                # tend = time.time()
                # # print("first calc: {}".format(tend-tstart))
                # tstart = time.time()
                # lb, _ = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward", IBP=False,  bound_upper=False)
                # tend = time.time()
                # print("second calc: {}".format(tend-tstart))
                lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward")
            else:
                lb, ub = bounded_cl_system.compute_bounds(x=(range_tensor,), method="backward")
            

            reach_set_range = torch.hstack((lb.T, ub.T))
            next_reachable_set.add_subset(reach_set_range, self.thread)
        else:
            for i, subset in self.subsets.items():
                subset.populate_next_reachable_set(bounded_cl_system, next_reachable_set, training)

        next_reachable_set.calculate_full_set()
        next_reachable_set.t_parent = self.t
        if next_reachable_set.t - self.t == 1:
            next_reachable_set.symbolic = False

    def switch_on_off(self, condition, thread=0):
        self.recalculate = not condition(self.full_set)
        if self.recalculate:
            if self.full_set[1, 0] < -1:
                print("Recalculating")
                print(self.full_set)
    
    def plot_reachable_set(self, ax, plot_partitions = True):
        if self.subsets == {}:
            set_range = self.full_set.cpu().detach().numpy()
            xy = set_range[:, 0]
            width, height = set_range[:, 1] - set_range[:, 0]
            if self.recalculate:
                edgecolor = 'orange'
            else:
                edgecolor = 'b'
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor=edgecolor, facecolor='none')
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
    def __init__(self, cl_system,  num_steps, initial_range, device='cpu') -> None:
        self.num_steps = num_steps
        self.cl_system = cl_system
        self.device = device
        self.max_diff = 12

        if cl_system.dynamics.name == 'DoubleIntegrator':
            dummy_input = torch.tensor([[2.75, 0.]], device=device)
        elif cl_system.dynamics.name == 'Unicycle_NL':
            dummy_input = torch.tensor([[-12.5, 3.5, -0.5]], device=device)
        self.bounded_cl_system = BoundedModule(cl_system, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device=device)
        
        self.reachable_sets = {0: ReachableSet(0, initial_range, partition_strategy = 'maintain', device=device)}
        self.bounded_cl_systems = {0: BoundedModule(cl_system, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device=device)}
        for i in range(num_steps):
            self.reachable_sets[i+1] = ReachableSet(i+1, device=device)
            cl_system.set_num_steps(i+2)
            if i < self.max_diff:
                self.bounded_cl_systems[i+1] = BoundedModule(cl_system, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device=device)

    def set_partition_strategy(self, t, partition_strategy):
        self.reachable_sets[t].set_partition_strategy(partition_strategy)

    def get_parent_set(self, reachable_set):
        if reachable_set.t == 0:
            return IndexError
        
        return self.reachable_sets[reachable_set.t - 1].get_thread(reachable_set.thread)


    def calculate_reachable_sets(self, training = False, autorefine = False, visualize = False, condition = None):
        initial_range = self.reachable_sets[0].full_set
        num_states = initial_range.shape[0]
        reach_set_range = initial_range
        info = {}
        

        for i in range(self.num_steps):
            info[i] = {}
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
            tstart = time.time()
            self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_system, self.reachable_sets[i+1], training)
            tend = time.time()
            self.reachable_sets[i+1].consolidate()
            info[i]['unrefined'] = deepcopy(self.reachable_sets[i+1].full_set.cpu().detach().numpy())
            info[i]['time'] = tend - tstart

            # if autorefine:
            #     self.refine(self.reachable_sets[i+1], condition)
            if autorefine:
                if visualize:
                    self.plot_reachable_sets()
                
                refined = self.refine(self.reachable_sets[i+1], condition, info, i)
                    
                if visualize:
                    self.plot_reachable_sets()

            
            # if self.reachable_sets[i+1].full_set[1, 0] < -1 or self.reachable_sets[i+1].full_set[0, 0] < 0 and autorefine:
            #     tstart = time.time()
            #     # self.refine(self.reachable_sets[i+1])
            #     self.reachable_sets[0].populate_next_reachable_set(self.bounded_cl_systems[i], self.reachable_sets[i+1])
            #     tend = time.time()
            #     info[i]['refined'] = self.reachable_sets[i+1].full_set.cpu().detach().numpy()
            #     info[i]['recalc_time'] = tend - tstart
                
            #     if visualize:
            #         print("recalculating set at time {}".format(i+1 ))
            #         self.plot_reachable_sets()
            
        return self.reachable_sets, info
    
    def calculate_N_step_reachable_sets(self, training = False, indices = None):
        # from copy import deepcopy
        # from cl_systems import ClosedLoopDynamics
        # initial_range = self.reachable_sets[0].full_set
        # num_states = initial_range.shape[0]
        # reach_set_range = initial_range
        if indices is None:
            indices = list(range(self.num_steps))
        import time

        info = {}

        for i in indices:
            info[i] = {}
            print("Calculating set {}".format(i))
            tstart = time.time()
            self.reachable_sets[0].populate_next_reachable_set(self.bounded_cl_systems[i], self.reachable_sets[i+1])
            print(self.reachable_sets[i+1].full_set)
            tend = time.time()
            print('Calculation Time: {}'.format(tend-tstart))
            info[i]['unrefined'] = deepcopy(self.reachable_sets[i+1].full_set.cpu().detach().numpy())
            info[i]['time'] = tend - tstart
        # self.cl_system.set_num_steps(7)
        # dummy_input = torch.tensor([[2.75, 0.]], device=self.device)
        # bounded_cl_system = BoundedModule(self.cl_system, dummy_input, device=self.device)
        # self.reachable_sets[0].get_partitions()
        # tstart = time.time()
        # self.reachable_sets[0].populate_next_reachable_set(self.bounded_cl_system, self.reachable_sets[7])
        # print(self.reachable_sets[7].full_set)
        # tend = time.time()
        # print('Calculation Time: {}'.format(tend-tstart))
        # initial_range = self.reachable_sets[0].full_set
        # num_states = initial_range.shape[0]
        # reach_set_range = initial_range
        

        # for i in indices:
        #     # prev_set = self.reachable_sets[i].full_set

        #     # partitions = self.reachable_sets[i].get_partitions()


        #     # x = torch.mean(reach_set_range, axis=1).reshape(-1,2)
        #     # eps = (reach_set_range[:, 1] - reach_set_range[:, 0])/2
        #     # ptb = PerturbationLpNorm(eps = eps) 
        #     # prev_set = BoundedTensor(x, ptb)

        #     # lb, ub = self.bounded_cl_system.compute_bounds(x=(prev_set,), method="backward", IBP=True)

        #     # reach_set_range = torch.hstack((lb.T, ub.T))
            
        #     # self.reachable_sets[i+1] = ReachableSet(i+1, reach_set_range)
        #     # import pdb; pdb.set_trace()
        #     self.reachable_sets[0].get_partitions()
        #     self.reachable_sets[0].populate_next_reachable_set(self.bounded_cl_systems[i], self.reachable_sets[i+1])
        #     # self.reachable_sets[0+1].consolidate()   


        return self.reachable_sets, info
    
    def calculate_hybrid_symbolic_reachable_sets(self, concretization_rate = 5, training = False):
        from copy import deepcopy
        from cl_systems import ClosedLoopDynamics
        initial_range = self.reachable_sets[0].full_set
        num_states = initial_range.shape[0]
        reach_set_range = initial_range
        import time

        idx = 0

        for i in range(self.num_steps):
            print("Calculating set {}".format(i))
            cl_system = ClosedLoopDynamics(self.cl_system.controller, self.cl_system.dynamics, i%concretization_rate+1)
            dummy_input = torch.tensor([[2.75, 0.]], device=self.device)
            bounded_cl_system = BoundedModule(cl_system, dummy_input, device=self.device)

            if i % concretization_rate == 0:
                idx = i
            tstart = time.time()
            self.reachable_sets[idx].populate_next_reachable_set(bounded_cl_system, self.reachable_sets[i+1])
            print(self.reachable_sets[i+1].full_set)
            tend = time.time()
            print('Calculation Time: {}'.format(tend-tstart))

        return self.reachable_sets
    
    def refine(self, reachable_set, condition, info, t, force=False):
        refined = not condition(reachable_set.full_set) or force
        tf = reachable_set.t
        min_diff = 0
        max_diff = self.max_diff

        # if tf == 35:
        #     import pdb; pdb.set_trace()
        if force and not reachable_set.symbolic:
            next_idx = max(tf - self.max_diff, 0)
            print("marching back from set {} to set {}".format(tf, next_idx))
            if self.reachable_sets[next_idx].symbolic:
                print("{} is symbolic, marching back".format(next_idx))
                tstart = time.time()
                self.reachable_sets[next_idx].populate_next_reachable_set(self.bounded_cl_systems[tf - next_idx - 1], reachable_set)
                reachable_set.symbolic = True
                tend = time.time()
                info[tf-1]['refined'] = reachable_set.full_set.cpu().detach().numpy()
                info[tf-1]['recalc_time'] = tend - tstart
            else:
                self.refine(self.reachable_sets[next_idx], condition, info, next_idx, force=True)
        else:
            final_idx = max(tf - self.max_diff, 0)
            i = tf - 2
            if not condition(reachable_set.full_set):
                print("Collision detected at t = {}".format(tf))
            while i >= final_idx and not condition(reachable_set.full_set):
                diff = tf - i
                if self.reachable_sets[i].symbolic and (diff >= min_diff or i == 0):
                    print("recalculating set {} from time {}".format(tf, i))
                    # import pdb; pdb.set_trace()
                    tstart = time.time()
                    self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[tf - i - 1], reachable_set)
                    reachable_set.symbolic = True
                    tend = time.time()
                    info[tf-1]['refined'] = reachable_set.full_set.cpu().detach().numpy()
                    info[tf-1]['recalc_time'] = tend - tstart
                elif diff == max_diff:
                    print("cannot do full symbolic from tf = {}, starting march".format(tf))
                    # if i == 41:
                    #     import pdb; pdb.set_trace()
                    if i > 1:
                        self.refine(self.reachable_sets[i], condition, info, i, force=True)
                        i = tf - 2

                i -= 1

        # i = tf - 2
        # while i >= 0 and (not condition(reachable_set.full_set) or force):
        #     diff = tf - i
        #     if ((self.reachable_sets[i].symbolic and not force) or (force)) and ((diff >= min_diff or i == 0) and diff <= max_diff):
        #         print("recalculating set {} from time {}".format(tf, i))
        #         # import pdb; pdb.set_trace()
        #         tstart = time.time()
        #         self.reachable_sets[i].populate_next_reachable_set(self.bounded_cl_systems[tf - i - 1], reachable_set)
        #         reachable_set.symbolic = True
        #         tend = time.time()
        #         info[tf-1]['refined'] = reachable_set.full_set.cpu().detach().numpy()
        #         info[tf-1]['recalc_time'] = tend - tstart
        #     elif diff == max_diff:
        #         print("here, i = {}".format(i))
        #         # if i == 41:
        #         #     import pdb; pdb.set_trace()
        #         if i > 1:
        #             self.refine(self.reachable_sets[i], condition, info, i, force=True)
        #             i = tf - 2

        #     i -= 1

        return refined
        

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
    

    def switch_sets_on_off(self, constraint):

        all_sets = self.get_all_reachable_sets()

        y = lambda y: False
        if y.__code__.co_code == constraint.__code__.co_code:
            for reachable_set in all_sets:
                reachable_set.recalculate = True
        else:
            for reachable_set in all_sets: # find colliding sets
                reachable_set.switch_on_off(constraint)

            
            # determine how far back we should step
            t_violation = self.num_steps # start with no partitioning
            for i, reachable_set in self.reachable_sets.items(): # find reachable sets that violate constraint
                if not constraint(reachable_set.full_set):
                    t_violation = min(t_violation, i)
                    print("Collision at t = {}".format(t_violation))
                    break

            walk_back = True # going backwards from violating set
            t = t_violation
            steps_back = 0
            print("violation: {}".format(t_violation))
            while(walk_back):
                t -= 1
                steps_back += 1
                print("stepping back to {}".format(t))
                xs = self.reachable_sets[t].sample_from_reachable_set(self.cl_system, steps_back, sample_corners=True)
                sample_range = np.vstack((np.min(xs, axis=0), np.max(xs, axis=0))).T
                if constraint(sample_range) or t == 0:
                    walk_back = False
                    print("Need to partition t = {}".format(t))

            for reachable_set in all_sets:
                if reachable_set.recalculate:
                    next_set = reachable_set
                    for i in range(t_violation, t, -1):
                        next_set = self.get_parent_set(next_set)
                        next_set.recalculate = True
    



    def constraint_aware_partition(self, constraint):
        t_violation = self.num_steps # start with no partitioning
        for i, reachable_set in self.reachable_sets.items(): # find reachable sets that violate constraint
            if not constraint(reachable_set.full_set):
                t_violation = min(t_violation, i)
                print("Collision at t = {}".format(t_violation))
                break
        
        walk_back = True # going backwards from violating set
        t = t_violation
        steps_back = 0
        while(walk_back):
            t -= 1
            steps_back += 1
            xs = self.reachable_sets[t].sample_from_reachable_set(self.cl_system, steps_back, sample_corners=True)
            sample_range = np.vstack((np.min(xs, axis=0), np.max(xs, axis=0))).T
            if constraint(sample_range) or t == 0:
                walk_back = False
                print("Need to partition t = {}".format(t))
        
        if not isinstance(self.reachable_sets[t].partition_strategy, np.ndarray):
            self.set_partition_strategy(t, np.array([2, 2]))
        else:
            num_parts = self.reachable_sets[t].partition_strategy[0]
            self.set_partition_strategy(t, np.array([num_parts+1, num_parts+1]))


    def is_safe(self, constraint_list):
        for constraint in constraint_list:
            for i, reachable_set in self.reachable_sets.items():
                if not constraint(reachable_set.full_set):
                    return False
                
        return True



    def plot_reachable_sets(self, num_trajectories=50):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        initial_set = self.reachable_sets[0].full_set.cpu()
        total_reachable_sets = self.reachable_sets


        
        fig, ax = plt.subplots()
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

        # num_steps = time_horizon
        # np.random.seed(0)
        # x0s = np.random.uniform(
        #     low=initial_set[:, 0],
        #     high=initial_set[:, 1],
        #     size=(num_trajectories, cl_system.At.shape[0]),
        # )
        # xt = x0s
        # xs = x0s
        # for _ in range(num_steps):
        #     u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller)
        #     xt1 = cl_system.dynamics.dynamics_step(xt, u_nn)
        #     xt = xt1

        #     xs = np.vstack((xs, xt1))
        xs = self.reachable_sets[0].sample_from_reachable_set(self.cl_system, num_steps=time_horizon, num_trajectories=num_trajectories, sample_corners=False)
        
        ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

        xy = initial_set[[0, 1], 0]
        width, height = initial_set[[0, 1], 1] - initial_set[[0, 1], 0]
        rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # import pdb; pdb.set_trace()
        for i, reachable_sets in total_reachable_sets.items():
            # import pdb; pdb.set_trace()
            # for _, set_range in reachable_sets:
            if reachable_sets.full_set is not None:
                set_range = reachable_sets.full_set.cpu().detach().numpy()
                xy = set_range[[0, 1], 0]
                width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
                rect = Rectangle(xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

        if self.cl_system.dynamics.name == "DoubleIntegrator":
            ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c='r', linewidth=2)
            rect = Rectangle(np.array([-1.5, -1.25]), 4.75, 0.25, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
            ax.add_patch(rect)

            ax.plot(np.array([0, 0]), np.array([-1.25, 1.]), c='r', linewidth=2)
            rect = Rectangle(np.array([-1.5, -1.25]), 1.5, 2.25, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
            ax.add_patch(rect)

            ax.set_xlim([-1.5, 3.25])
            ax.set_ylim([-1.25, 1.])
        elif self.cl_system.dynamics.name == "Unicycle_NL":
            # obstacles = [{'x': -10, 'y': -1, 'r': 3},
            #              {'x': -3, 'y': 2.5, 'r': 2 }]
            obstacles = [{'x': -6, 'y': -0.5, 'r': 2.2},
                         {'x': -1.25, 'y': 1.75, 'r': 1.6}]
            for obstacle in obstacles:
                circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor='r', facecolor='none')
                ax.add_patch(circle)
                circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], edgecolor='r', facecolor='r', alpha=0.1)
                ax.add_patch(circle)

            ax.set_xlim([-10, 1])
            ax.set_ylim([-3, 5])
            ax.set_aspect('equal')

        ax.set_xlabel('x1', fontsize=20)
        ax.set_ylabel('x2', fontsize=20)

            
        plt.show()

    
    def plot_all_subsets(self, num_trajectories=100):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        initial_set = self.reachable_sets[0].full_set.cpu()
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
            u_nn = cl_system.dynamics.control_nn(xt, cl_system.controller.cpu())
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


        ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c='r', linewidth=2)
        rect = Rectangle(np.array([-1.5, -1.25]), 4.75, 0.25, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
        ax.add_patch(rect)

        ax.set_xlim([-1.5, 3.25])
        ax.set_ylim([-1.25, 1.])

        ax.set_xlabel('x1', fontsize=20)
        ax.set_ylabel('x2', fontsize=20)

            
        plt.show()
    
    def animate_reachability_calculation(self, info, plot_partitions=False):
        cl_system = self.cl_system
        time_horizon = self.num_steps
        initial_set = self.reachable_sets[0].full_set.cpu()
        total_reachable_sets = self.reachable_sets


        
        fig, ax = plt.subplots()
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

        reachable_sets = []
        colors = []
        remove = []
        num_times = []
        for i, time_step_dict in info.items():
            num_times.append(max(int(5*time_step_dict['time']), 1))
            # num_times.append(2)
            reachable_sets.append(time_step_dict['unrefined'])
            colors.append('b')
            remove.append(False)

            if 'refined' in time_step_dict.keys():
                colors[-1] = 'r'
                remove[-1] = True
                num_times.append(max(5+int(5*time_step_dict['recalc_time']), 1))
                reachable_sets.append(time_step_dict['refined'])
                colors.append('b')
                remove.append(True)
    

        num_times.append(5)

        
        reachable_sets_extended = []
        colors_extended = []
        remove_extended = []
        for i in range(len(reachable_sets)):
            for _ in range(num_times[i+1]):
                reachable_sets_extended.append(reachable_sets[i])
                colors_extended.append(colors[i])
                remove_extended.append(remove[i])
        
        import pdb; pdb.set_trace()
        def animate(i):    
            ax.clear()
            xs = self.reachable_sets[0].sample_from_reachable_set(self.cl_system, num_steps=time_horizon, sample_corners=False)
            
            ax.scatter(xs[:, 0], xs[:, 1], s=1, c='k')

            xy = initial_set[[0, 1], 0]
            width, height = initial_set[[0, 1], 1] - initial_set[[0, 1], 0]
            rect = Rectangle(xy, width, height, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)
            
            if self.cl_system.dynamics.name == "DoubleIntegrator":
                ax.plot(np.array([-1.5, 3.25]), np.array([-1, -1]), c='r', linewidth=2)
                rect = Rectangle(np.array([-1.5, -1.25]), 4.75, 0.25, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
                ax.add_patch(rect)

                ax.plot(np.array([0, 0]), np.array([-1.25, 1.]), c='r', linewidth=2)
                rect = Rectangle(np.array([-1.5, -1.25]), 1.5, 2.25, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1)
                ax.add_patch(rect)

                ax.set_xlim([-1.5, 3.25])
                ax.set_ylim([-1.25, 1.])
            elif self.cl_system.dynamics.name == "Unicycle_NL":
                # obstacles = [{'x': -10, 'y': -1, 'r': 3},
                #              {'x': -3, 'y': 2.5, 'r': 2 }]
                obstacles = [{'x': -6, 'y': -0.5, 'r': 2.2},
                            {'x': -1.25, 'y': 1.75, 'r': 1.6}]
                for obstacle in obstacles:
                    circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['r'], color='blue')
                    ax.add_patch(circle)

                ax.set_xlim([-10, 1])
                ax.set_ylim([-3, 5])
                ax.set_aspect('equal')
            

            ax.set_xlabel('x1', fontsize=20)
            ax.set_ylabel('x2', fontsize=20)

            
            for j in range(i):
                set_range = reachable_sets_extended[j]
                xy = set_range[[0, 1], 0]
                width, height = set_range[[0, 1], 1] - set_range[[0, 1], 0]
                rect = Rectangle(xy, width, height, linewidth=1, edgecolor=colors_extended[j], facecolor='none')



                # import pdb; pdb.set_trace()
                if j == i - 1 or not remove_extended[j]:
                    ax.add_patch(rect)
                

        ani = FuncAnimation(fig, animate, frames=sum(num_times), repeat=True)
        ani.save("double_integrator.gif", dpi=300, writer=PillowWriter(fps=10))            









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

def calculate_reachable_sets_old(cl_dyn, initial_set, time_horizon):
    # lb, ub = cl_system.compute_bounds(x=(initial_set,), method="backward")
    num_states = initial_set.shape[0]
    reachable_sets = torch.zeros((time_horizon, num_states, 2))
    x = torch.from_numpy(np.mean(initial_set, axis=1).reshape(-1,2)).type(torch.float32)
    eps = torch.from_numpy((initial_set[:, 1] - initial_set[:, 0])/2).type(torch.float32)
    ptb = PerturbationLpNorm(eps = eps)
    prev_set = BoundedTensor(x, ptb)
    
    for i in range(time_horizon):
        dummy_input = torch.tensor([[2.75, 0.]], device='cpu')
        cl_dyn_t = cl_systems.ClosedLoopDynamics(cl_dyn.controller, cl_dyn.dynamics, i+1)
        cl_system = BoundedModule(cl_dyn_t, dummy_input, bound_opts={'relu': "CROWN-IBP"}, device='cpu')
        
        lb, ub = cl_system.compute_bounds(x=(prev_set,), method="backward", IBP=True)

        reach_set = torch.hstack((lb.T, ub.T))
        reachable_sets[i] = reach_set

        # x = torch.mean(reach_set, axis=1).reshape(-1,2)
        # eps = (reach_set[:, 1] - reach_set[:, 0])/2
        # ptb = PerturbationLpNorm(eps = eps) 

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
            all_subsets.append(rss.cpu().detach().numpy())

        prev_range = reachable_set.cpu().detach().numpy()
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