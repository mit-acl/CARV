# reachability_algorithms.py

import integrated_reachable_sim as rss
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
    def __init__(self, num_partitions: int = 3, process_noise_std=0.01, measurement_noise_std=0.3):
        super().__init__()
        
        # Convert single int to array for 2D system
        if isinstance(num_partitions, int):
            partition_array = np.array([num_partitions, num_partitions])
        else:
            partition_array = np.array(num_partitions)
        
        self.num_partitions = int(np.prod(partition_array))
        self.partition_structure = partition_array
        
        # Create state tester with Kalman filtering
        self.state_analyzer = rss.setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
        self.state_tester = rss.ReachabilityTester(
            self.state_analyzer, 
            dynamic_plot=False,
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std
        )
        
        # Get initial partitions from t=0
        init_bounds = self.state_tester.horizons[0].get_tight_bound()
        initial_partitions = self._partition_bounds(init_bounds, partition_array)
        
        # Create partition analyzers and testers
        self.partition_analyzers = {}
        self.partition_testers = {}
        
        for i in range(self.num_partitions):
            analyzer = rss.setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz', 
                                         init_range=initial_partitions[i])
            tester = rss.ReachabilityTester(analyzer, dynamic_plot=False)
            self.partition_analyzers[i] = analyzer
            self.partition_testers[i] = tester
        
        # Track which partitions are active at each timestep
        # Format: {partition_idx: {timestep: 'active' or 'pruned'}}
        self.partition_status = {i: {0: 'active'} for i in range(self.num_partitions)}
        
    def _partition_bounds(self, bounds: np.ndarray, partition_array: np.ndarray):
        """
        Partition bounds into grid.
        
        Args:
            bounds: (n_dims, 2) array
            partition_array: (n_dims,) array specifying partitions per dimension
        
        Returns:
            dict: {partition_idx: bounds}
        """
        n_dims = bounds.shape[0]
        partition_bounds = {}
        
        # Create partition edges for each dimension
        edges = []
        for dim in range(n_dims):
            dim_edges = np.linspace(bounds[dim, 0], bounds[dim, 1], 
                                   partition_array[dim] + 1)
            edges.append(dim_edges)
        
        # Create all partition combinations
        partition_idx = 0
        for indices in np.ndindex(*partition_array):
            partition_bound = np.zeros((n_dims, 2))
            for dim, idx in enumerate(indices):
                partition_bound[dim, 0] = edges[dim][idx]
                partition_bound[dim, 1] = edges[dim][idx + 1]
            partition_bounds[partition_idx] = partition_bound
            partition_idx += 1
        
        return partition_bounds

    def bounds_intersect(self, bounds_a: np.ndarray, bounds_b: np.ndarray, eps: float = 1e-8):
        """Check if two N-dimensional bounding boxes intersect."""
        if bounds_a.shape != bounds_b.shape:
            raise ValueError(f"Bounds must have the same shape")

        a_min = np.minimum(bounds_a[:, 0], bounds_a[:, 1])
        a_max = np.maximum(bounds_a[:, 0], bounds_a[:, 1])
        b_min = np.minimum(bounds_b[:, 0], bounds_b[:, 1])
        b_max = np.maximum(bounds_b[:, 0], bounds_b[:, 1])

        for amin, amax, bmin, bmax in zip(a_min, a_max, b_min, b_max):
            if amax < bmin - eps or bmax < amin - eps:
                return False

        return True

    def calculate_reachability(self, timestep: int, time_horizon: int):
        """
        Calculate reachability using pruning strategy.
        
        Args:
            timestep: Starting timestep
            time_horizon: Number of steps to propagate
        """
        print(f"\n{'='*60}")
        print(f"CALCULATING REACHABILITY WITH PRUNING")
        print(f"{'='*60}")
        
        # Propagate state estimate with Kalman filtering
        print(f"\nPropagating state estimate for {time_horizon} steps...")
        for i in range(time_horizon):
            current_timestep = timestep + i
            self.state_tester.real_state_empirical(
                current_timestep, 
                current_timestep + 1, 
                visualize=False
            )
        
        # Propagate and prune partitions
        print(f"\nPropagating and pruning partitions...")
        for i in range(time_horizon):
            current_timestep = timestep + i
            print(f"\n--- Timestep {current_timestep} -> {current_timestep + 1} ---")
            
            self.propagate_partitions(current_timestep)
            self.prune_partitions(current_timestep + 1)
            
            # Print status
            active_count = sum(1 for p in range(self.num_partitions) 
                             if self.partition_status[p][current_timestep + 1] == 'active')
            print(f"Active partitions at t={current_timestep + 1}: {active_count}/{self.num_partitions}")

    def prune_partitions(self, timestep):
        """Prune partitions that don't intersect with state estimate."""
        # Get state estimate bounds from Kalman filter
        state_estimate_bounds = self.state_tester.horizons[timestep].get_tight_bound()
        
        for i in range(self.num_partitions):
            # Skip if already pruned
            if self.partition_status[i].get(timestep - 1) == 'pruned':
                self.partition_status[i][timestep] = 'pruned'
                continue
            
            # Get partition bounds at this timestep
            partition_horizon = self.partition_testers[i].horizons.get(timestep)
            if partition_horizon is None:
                self.partition_status[i][timestep] = 'pruned'
                continue
            
            partition_bounds = partition_horizon.get_tight_bound()
            
            if self.bounds_intersect(partition_bounds, state_estimate_bounds):
                # Keep active
                self.partition_status[i][timestep] = 'active'
            else:
                # Prune this partition
                self.partition_status[i][timestep] = 'pruned'
                print(f"  Pruned partition {i}")

    def propagate_partitions(self, timestep: int):
        """Propagate active partitions one step forward."""
        for i in range(self.num_partitions):
            current_status = self.partition_status[i].get(timestep)
            
            if current_status == 'pruned':
                self.partition_status[i][timestep + 1] = 'pruned'
                continue
            
            # Propagate with concrete step
            self.partition_testers[i].concrete(timestep, timestep + 1, visualize=False)
            self.partition_status[i][timestep + 1] = 'active'

    def animate_partitions(self, filename="partitions_animation.gif", fps=2):
        """Animate partitions as 2D rectangles over time."""
        print("\n" + "="*80)
        print("CREATING PARTITION ANIMATION")
        print("="*80)

        # Gather all timesteps
        all_timesteps = sorted(self.state_tester.horizons.keys())

        fig, ax = plt.subplots(figsize=(10, 8))
        
        process_noise = self.state_tester.process_noise_std
        measurement_noise = self.state_tester.measurement_noise_std
        
        fig.suptitle(
            f"Reachability Animation: Concrete Evaluation Pruning\n"
            f"Process Noise σ={process_noise:.3f} | Measurement Noise σ={measurement_noise:.3f}",
            fontsize=14, fontweight="bold"
        )

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

        # Compute global plot limits
        all_bounds = []
        for i in range(self.num_partitions):
            for timestep in all_timesteps:
                if self.partition_status[i].get(timestep) == 'pruned':
                    continue
                horizon = self.partition_testers[i].horizons.get(timestep)
                if horizon is not None:
                    bounds = horizon.get_tight_bound()
                    if bounds is not None:
                        all_bounds.append(bounds)

        for timestep in all_timesteps:
            horizon = self.state_tester.horizons.get(timestep)
            if horizon is not None:
                bounds = horizon.get_tight_bound()
                if bounds is not None:
                    all_bounds.append(bounds)

        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min, x_max = np.min(all_bounds_array[:, 0, 0]), np.max(all_bounds_array[:, 0, 1])
            y_min, y_max = np.min(all_bounds_array[:, 1, 0]), np.max(all_bounds_array[:, 1, 1])

            x_range, y_range = x_max - x_min, y_max - y_min
            padding_x = 0.1 * x_range if x_range > 0 else 0.1
            padding_y = 0.1 * y_range if y_range > 0 else 0.1

            global_xlim = (x_min - padding_x, x_max + padding_x)
            global_ylim = (y_min - padding_y, y_max + padding_y)
        else:
            global_xlim, global_ylim = (-1, 1), (-1, 1)

        def animate_frame(frame_idx):
            ax.clear()
            timestep = all_timesteps[frame_idx]

            # Plot active partitions
            for i in range(self.num_partitions):
                if self.partition_status[i].get(timestep) == 'pruned':
                    continue
                horizon = self.partition_testers[i].horizons.get(timestep)
                if horizon is not None:
                    bounds = horizon.get_tight_bound()
                    if bounds is not None:
                        plot_rectangle(ax, bounds, color='blue', alpha=0.3)

            # Plot state estimate bounds
            state_horizon = self.state_tester.horizons.get(timestep)
            if state_horizon is not None:
                state_bounds = state_horizon.get_tight_bound()
                if state_bounds is not None:
                    plot_rectangle(ax, state_bounds, color='red', alpha=0.3)
                
                # Plot true state as a dot
                for calc_data in state_horizon.calculations.values():
                    if (calc_data['calc_type'] == rss.CalculationType.EMPIRICAL and 
                        'real_state' in calc_data):
                        true_state = calc_data['real_state']
                        ax.plot(true_state[0], true_state[1], 
                               marker='o', markersize=10, 
                               color='black', markeredgecolor='white', 
                               markeredgewidth=2, zorder=10)
                        break

            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_xlabel("State 1", fontsize=12)
            ax.set_ylabel("State 2", fontsize=12)
            ax.set_title(f"Timestep {timestep}", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            legend_elements = [
                Patch(facecolor='blue', alpha=0.3, label='Concrete (Partition Reachability)'),
                Patch(facecolor='red', alpha=0.3, label='Empirical (State Estimate)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                          markeredgecolor='white', markeredgewidth=2, markersize=10, 
                          label='True State')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

            return ax

        anim = FuncAnimation(fig, animate_frame, frames=len(all_timesteps), 
                           interval=1000/fps, repeat=True)

        print(f"Saving animation to {filename}...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)
        plt.close(fig)
        print(f"✓ Saved animation to {filename}")


def run():
    max_horizon = 20
    print(f"{'-'*5} Testing ReachabilityPruning Algorithm {'-'*5}")
    print("Initializing reachability algorithm...")
    reach_algorithm = ReachabilityPruning(num_partitions=3)
    
    print("Calculating reachability with pruning...")
    reach_algorithm.calculate_reachability(timestep=0, time_horizon=max_horizon)

    reach_algorithm.animate_partitions(filename="partition_animation.gif", fps=1)


if __name__ == "__main__":
    run()