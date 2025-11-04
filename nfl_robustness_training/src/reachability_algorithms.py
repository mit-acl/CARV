import reachable_set_sim as rss
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

class ReachabilityAlgorithm:
    def __init__(self):
        pass
    @abstractmethod
    def calculate_reachability(self, time_horizon):
        pass

class ReachabilityPruning(ReachabilityAlgorithm):
    def __init__(self, num_partitions: int = 3):
        super().__init__()
        # self.num_partitions = num_partitions

        # For a 2D system, convert single int to array
        if isinstance(num_partitions, int):
            # Option 1: Make it a 1D partition
            partition_array = np.array([num_partitions, num_partitions])
            # Option 2: Try to make a square grid (uncomment if preferred)
            # partition_array = np.array([num_partitions, num_partitions])
        else:
            partition_array = np.array(num_partitions)
        
        self.num_partitions = int(np.prod(partition_array))  # Total number of partitions
        self.partition_structure = partition_array
        # self.state_estimates = {} # dict of timestep -> state_estimate_bounds
        self.state_analyzer = rss.setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
        self.state_tester = rss.ReachabilityTester(self.state_analyzer, dynamic_plot=False)
        self.partition_analyzers = {}
        self.partition_testers = {}

        initial_partitions = self.state_tester.get_partitions(0, num_partitions=self.partition_structure)

        for i in range(self.num_partitions):
            analyzer = rss.setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz', init_range=initial_partitions[i])
            tester = rss.ReachabilityTester(analyzer, dynamic_plot=False)
            self.partition_analyzers[i] = analyzer
            self.partition_testers[i] = tester

        self.partition_calc_sets = {i:{0:0} for i in range(self.num_partitions)} # dict of partition_index -> dict of timestep -> partition_ids
        self.state_calc_sets = {0:0} # dict of timestep -> state_estimate_ids
        # self.consolidated_calc_sets = {} # dict of timestep -> consolidated set ids

    # def bounds_intersect(self, bounds_a: np.ndarray, bounds_b: np.ndarray):
    #     """
    #     Check if two N-dimensional bounding boxes intersect.
        
    #     Each bounds array should have shape (n_dims, 2),
    #     where bounds[i, 0] = lower bound, bounds[i, 1] = upper bound.
    #     """
    #     if bounds_a.shape != bounds_b.shape:
    #         raise ValueError("Bounds must have the same shape.")

    #     # Ensure lower <= upper for both (in case inputs are unsorted)
    #     a_min, a_max = np.minimum(bounds_a[:, 0], bounds_a[:, 1]), np.maximum(bounds_a[:, 0], bounds_a[:, 1])
    #     b_min, b_max = np.minimum(bounds_b[:, 0], bounds_b[:, 1]), np.maximum(bounds_b[:, 0], bounds_b[:, 1])

    #     # Check overlap along each dimension
    #     overlap = np.all((a_min <= b_max) & (b_min <= a_max))
    #     return overlap

    def bounds_intersect(self, bounds_a: np.ndarray, bounds_b: np.ndarray, eps: float = 1e-8):
        """
        Check if two N-dimensional bounding boxes intersect (any overlap, not containment).
        Each bounds array should have shape (n_dims, 2).
        """
        if bounds_a.shape != bounds_b.shape:
            raise ValueError(f"Bounds must have the same shape, got {bounds_a.shape} vs {bounds_b.shape}")

        # Normalize lower/upper
        a_min, a_max = np.minimum(bounds_a[:, 0], bounds_a[:, 1]), np.maximum(bounds_a[:, 0], bounds_a[:, 1])
        b_min, b_max = np.minimum(bounds_b[:, 0], bounds_b[:, 1]), np.maximum(bounds_b[:, 0], bounds_b[:, 1])

        # Check overlap (allowing epsilon tolerance)
        for dim, (amin, amax, bmin, bmax) in enumerate(zip(a_min, a_max, b_min, b_max)):
            if amax < bmin - eps or bmax < amin - eps:
                return False  # separated along this dimension

        return True

    def calculate_reachability(self, timestep: int, time_horizon: int):
        """
        Calculates reachability using pruning strategy given a parent node ID and time horizon.
        
        Steps:
        - get initial set
            - compare reachable set partitions to current state estimation
            - prune non-overlapping partitions

        - for each step in time horizon:
            - propogate reachable set partitions
        
        """
        # current_state_estimate = self.get_state_estimate(timestep)
        # self.state_estimates[timestep] = current_state_estimate
        # self.prune_partitions(timestep)

        debug_print(f"DEBUG: empirical bounds at 0 are {self.state_tester.get_bounds(calc_id=0, calc_type=rss.CalculationType.EMPIRICAL)}")
        debug_print(f"DEBUG: concrete bounds at 0 are{[self.partition_testers[i].get_bounds(calc_id=0, calc_type=rss.CalculationType.CONCRETE) for i in range(self.num_partitions)]}")
        for i in range(time_horizon):
            current_timestep = timestep + i
            self.propogate_partitions(current_timestep)
            self.prune_partitions(current_timestep + 1)

    def prune_partitions(self, timestep):
        """
        Prunes the partitions at a given timestep based on overlap with the state estimate.
        
        Args:
            timestep (int): The timestep for which to prune partitions.
        Returns:
            dict: Dictionary of pruned partition IDs that overlap with the state estimate.
        """
        print(f"DEBUG: Pruning partitions at timestep {timestep}")
        self.state_estimate_bounds = self.get_state_estimate(timestep)
        print(f"DEBUG: State estimate bounds at t={timestep} are {self.state_estimate_bounds}")
        for i in range(self.num_partitions):
            partition_id = self.partition_calc_sets[i][timestep]
            print(f"DEBUG: Checking partition {i} at t={timestep} with ID {partition_id}")
            if partition_id == "pruned" or isinstance(partition_id, tuple):
                continue
            partition_bounds = self.partition_testers[i].get_bounds(calc_id=partition_id, calc_type=rss.CalculationType.CONCRETE)
            print(f"DEBUG: Partition {i} at t={timestep} has bounds {partition_bounds}")
            if self.bounds_intersect(partition_bounds, self.state_estimate_bounds):
                print(f"DEBUG: Partition {i} at t={timestep} with bounds {partition_bounds} intersects state estimate.")
                self.partition_calc_sets[i][timestep] = partition_id
            else:
                print(f"DEBUG: Partition {i} at t={timestep} with bounds {partition_bounds} does NOT intersect state estimate. Pruning.")
                self.partition_calc_sets[i][timestep] = ("pruned", partition_id)

    def propogate_partitions(self, timestep: int):
        """
        Propogates the given ORDERED partitions via concrete steps.
        
        Args:
            partition_ids (dict): Dictionary of partition IDs to propogate.
        Returns:
            dict: Dictionary of new partition IDs after propogation.
        """

        for i in range(self.num_partitions):
            partition_id = self.partition_calc_sets[i][timestep]
            if isinstance(partition_id, tuple) or partition_id == "pruned":
                self.partition_calc_sets[i][timestep + 1] = "pruned"
            else:
                new_id = self.partition_testers[i].concrete(partition_id, visualize=True)
                self.partition_calc_sets[i][timestep + 1] = new_id

    def propogate_state_estimate(self, timestep: int, horizon: int):
        """
        Propogates the state estimate via empirical steps.
        
        Args:
            timestep (int): The starting timestep for propagation.
            horizon (int): The number of steps to propagate.
        Returns:
            np.ndarray: The bounds of the propagated state estimate.
        """
        for i in range(horizon):
            # self.state_calc_sets[timestep+i] = self.state_tester.empirical(timestep, timestep + 1, num_samples=1000, visualize=True)
            self.state_calc_sets[timestep + i + 1] = self.state_tester.empirical(timestep + i, timestep + i + 1, num_samples=1000, visualize=True)
        # return self.get_state_estimate(timestep + horizon)

    def get_state_estimate(self, timestep: int):
        """
        Retrieves the empirical bounds for the state estimate at a given timestep.
        
        Args:
            timestep (int): The timestep for which to retrieve the state estimate.
        Returns:
            np.ndarray: The bounds of the state estimate.
        """
        state_calc_id = self.state_calc_sets[timestep]
        return self.state_tester.get_bounds(calc_id=state_calc_id, calc_type=rss.CalculationType.EMPIRICAL)
        



    def animate_partitions(self, filename="partitions_animation.gif", fps=2):
        """
        Animate partitions as 2D rectangles over time.
        Each frame shows all partitions that exist at a given timestep.
        Plot bounds are kept consistent across all frames.
        """

        print("\n" + "="*80)
        print("CREATING PARTITION ANIMATION")
        print("="*80)

        # Gather all timesteps available in partition_calc_sets
        all_timesteps = sorted({
            t for partition_dict in self.partition_calc_sets.values()
            for t in partition_dict.keys()
        })

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle("Reachability Animation: Concrete Evaluation Pruning Originating from t=0", fontsize=16, fontweight="bold")

        # Helper to plot a rectangle
        def plot_rectangle(ax, bounds, color='blue', alpha=0.3, linewidth=1.5):
            xy = bounds[:2, 0]
            width = bounds[0, 1] - bounds[0, 0]
            height = bounds[1, 1] - bounds[1, 0]
            rect = Rectangle(
                xy, width, height,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                linewidth=linewidth
            )
            ax.add_patch(rect)

        # Compute global plot limits once, based on all bounds across all timesteps
        all_bounds = []
        for i in range(self.num_partitions):
            for timestep in all_timesteps:
                part_id = self.partition_calc_sets[i].get(timestep)
                if part_id is None or part_id == "pruned" or isinstance(part_id, tuple):
                    continue
                bounds = self.partition_testers[i].get_bounds(
                    calc_id=part_id, calc_type=rss.CalculationType.CONCRETE
                )
                all_bounds.append(bounds)

        for timestep in self.state_calc_sets:
            state_bounds = self.state_tester.get_bounds(
                calc_id=self.state_calc_sets[timestep],
                calc_type=rss.CalculationType.EMPIRICAL
            )
            all_bounds.append(state_bounds)

        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min = np.min(all_bounds_array[:, 0, 0])
            x_max = np.max(all_bounds_array[:, 0, 1])
            y_min = np.min(all_bounds_array[:, 1, 0])
            y_max = np.max(all_bounds_array[:, 1, 1])

            x_range, y_range = x_max - x_min, y_max - y_min
            padding_x = 0.1 * x_range if x_range > 0 else 0.1
            padding_y = 0.1 * y_range if y_range > 0 else 0.1

            global_xlim = (x_min - padding_x, x_max + padding_x)
            global_ylim = (y_min - padding_y, y_max + padding_y)
        else:
            global_xlim = (-1, 1)
            global_ylim = (-1, 1)

        # Animation function
        def animate_frame(frame_idx):
            ax.clear()
            timestep = all_timesteps[frame_idx]

            for i in range(self.num_partitions):
                part_id = self.partition_calc_sets[i].get(timestep)
                if part_id is None or part_id == "pruned" or isinstance(part_id, tuple):
                    continue
                bounds = self.partition_testers[i].get_bounds(
                    calc_id=part_id, calc_type=rss.CalculationType.CONCRETE
                )
                plot_rectangle(ax, bounds, color='blue', alpha=0.3)

            # Plot state estimate
            if timestep in self.state_calc_sets:
                state_bounds = self.state_tester.get_bounds(
                    calc_id=self.state_calc_sets[timestep],
                    calc_type=rss.CalculationType.EMPIRICAL
                )
                plot_rectangle(ax, state_bounds, color='red', alpha=0.3)

            # Keep global limits fixed
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_xlabel("State 1")
            ax.set_ylabel("State 2")
            ax.set_title(f"Timestep {timestep}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            legend_elements = [
                Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='Concrete (Partition Reachability)'),  # <--
                Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Empirical (State Estimate)')            # <--
            ]
            ax.legend(handles=legend_elements, loc='upper right')  # <-- adds the legend

            return ax

        # Build animation
        anim = FuncAnimation(
            fig,
            animate_frame,
            frames=len(all_timesteps),
            interval=1000/fps,
            repeat=True
        )

        print(f"Saving animation to {filename}...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close(fig)
        print(f"Saved animation to {filename}")



def debug_print(*args):
    print("\033[31m[DEBUG]\033[0m", *args)

def run():
    max_horizon = 20
    print (f"{('-'*5)} Testing ReachabilityPruning Algorithm {('-'*5)}")
    # print("Initializing analyzer...")
    # analyzer = rss.setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    # print("\n2. Creating interactive tester...")
    # tester = rss.ReachabilityTester(analyzer, dynamic_plot=True)
    print("Initializing reachability algorithm...")
    reach_algorithm = ReachabilityPruning(num_partitions=3)
    print("Generating empirical data for state estimation...")
    reach_algorithm.propogate_state_estimate(timestep=0, horizon=max_horizon+1)

    print("Calculating reachability with pruning...")
    reach_algorithm.calculate_reachability(timestep=0, time_horizon=max_horizon)

    reach_algorithm.animate_partitions(filename="partition_animation.gif", fps=2)




if __name__ == "__main__":
    run()
    
