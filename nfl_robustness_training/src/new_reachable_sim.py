"""
Reachable Set Simulator - Corrected Version
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from ast import literal_eval
from itertools import product
from copy import deepcopy
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum
import nfl_veripy.dynamics as dynamics

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import cl_systems

from utils.nn import load_controller
from utils.robust_training_utils import ReachableSet
from utils.robust_training_utils import Analyzer


class CalculationType(Enum):
    CONCRETE = "concrete"
    SYMBOLIC = "symbolic"
    EMPIRICAL = "empirical"


class ReachableSetHorizon:
    """
    Manages all reachable set calculations at a specific timestep.
    """
    def __init__(self, global_timestep: int, device='cpu'):
        self.global_timestep = global_timestep
        self.device = device

        # Main reachable set that gets updated with tightest bounds
        self.reachable_set = ReachableSet(t=global_timestep, device=device)
        self.reachable_set.recalculate = True

        # Record all individual calculation informaiton for this timestep
        self.calculations = {}  # maps calc_id -> dict with metadata
        self.calc_counter = 0

        # Track tightest bounds separately
        self.tight_bound = None

    def __repr__(self):
        vol = self.get_tight_volume() if self.tight_bound is not None else 0
        return (f"ReachableSetHorizon(t={self.global_timestep}, "
                f"num_calcs={len(self.calculations)}, "
                f"tight_vol={vol:.6f})")

    def add_calculation(self, bounds: np.ndarray, calc_type: CalculationType,
                       parent_id: Optional[int] = None, origin_timestep: int = 0,
                       computation_time: float = 0.0, step_size: int = 1,
                       num_samples: Optional[int] = None, notes: str = ""):

        # calc_id = self.calc_counter
        # self.calc_counter += 1

        # Store calculation metadata
        self.calculations[self.calc_counter] = {
            # 'calc_id': calc_id,
            'bounds': bounds.copy(),
            'calc_type': calc_type,
            # 'parent_id': parent_id,
            'origin_timestep': origin_timestep,
            'computation_time': computation_time,
            'step_size': step_size,
            'num_samples': num_samples,
            'volume': np.prod(bounds[:, 1] - bounds[:, 0]),
            'notes': notes
        }
        self.calc_counter += 1

        # Update tight bounds (intersection of all calculations)
        if self.tight_bound is None:
            self.tight_bound = bounds.copy()
        else:
            # Intersection: max of lower bounds, min of upper bounds
            self.tight_bound[:, 0] = np.maximum(self.tight_bound[:, 0], bounds[:, 0])
            self.tight_bound[:, 1] = np.minimum(self.tight_bound[:, 1], bounds[:, 1])

            # Check if intersection is empty
            if np.any(self.tight_bound[:, 0] > self.tight_bound[:, 1]):
                print(f"Warning: Empty intersection at t={self.global_timestep}")

        # Update the core ReachableSet with tightest bounds
        self.reachable_set.set_range(torch.tensor(self.tight_bound, dtype=torch.float32, device=self.device))

        # return calc_id
        return None

    def get_tight_bound(self):
        """Return the tightest bound"""
        return self.tight_bound

    def get_tight_volume(self):
        """Return volume of tightest bound"""
        if self.tight_bound is None:
            return None
        return np.prod(self.tight_bound[:, 1] - self.tight_bound[:, 0])

    def get_calculation(self, calc_id: int):
        """Get metadata for a specific calculation"""
        return self.calculations.get(calc_id, None)

    def list_calculations(self):
        """Print all calculations at this timestep"""
        print(f"\nCalculations at t={self.global_timestep}:")
        print(f"{'ID':<5} {'Type':<12} {'Volume':<12} {'Time':<8} {'Notes'}")
        print("-" * 60)
        for calc_id, calc in self.calculations.items():
            print(f"{calc_id:<5} {calc['calc_type'].value:<12} {calc['volume']:<12.6f} "
                  f"{calc['computation_time']:<8.4f} {calc['notes']}")



class ReachabilityTester:
    """
    Manual testing framework for reachability calculations using ReachableSetHorizon.
    """
    def __init__(self, analyzer, dynamic_plot=True):
        self.analyzer = analyzer
        self.dynamic_plot = dynamic_plot

        # Track horizons by timestep
        self.horizons: Dict[int, ReachableSetHorizon] = {}

        # Active calculation ID for each timestep (for visualization)
        self.counter = 0

        # Initialize horizon at t=0
        init_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
        self.horizons[0] = ReachableSetHorizon(0, device=analyzer.device)

        self.horizons[0].add_calculation(
            bounds=init_bounds,
            calc_type=CalculationType.EMPIRICAL,
            parent_id=None,
            origin_timestep=0,
            computation_time=0.0,
            step_size=0,
            notes='Initial set'
        )

        if self.dynamic_plot:
            plt.ion()
            self.fig, self.axes = self._setup_plot()

    def concrete(self, start_timestep: int, end: Optional[int] = None, visualize=True):
        """
        Compute concrete reachable set
        Args: start timestep, end timeste[]
        """

        # Get parent horizon
        if start_timestep not in self.horizons:
            print(f"Error: No horizon exists at timestep {start_timestep}")
            return False

        # Set end timestep
        if end is None:
            end = start_timestep + 1

        num_steps = end - start_timestep
        if num_steps <= 0:
            print(f"Error: Invalid step count (start={start_timestep}, end={end})")
            return False

        # Track total computation time
        total_time = 0.0
        current_parent_timestep = start_timestep

        print(f"Starting concrete propagation: t={start_timestep} → t={end} ({num_steps} steps)")

        # Loop through each iteration/timestep
        for step in range(num_steps):
            current_timestep = current_parent_timestep + 1

            # Get the parent timestep horizon
            parent_horizon = self.horizons[current_parent_timestep]

            # Create horizon at current timestep if it doesn't exist
            if current_timestep not in self.horizons:
                self.horizons[current_timestep] = ReachableSetHorizon(current_timestep, device=self.analyzer.device)

            # Get the parent horizon's reachable set
            parent_reachset = parent_horizon.reachable_set

            # Create temporary reachable set for this step's result
            temp_reachset = ReachableSet(t=current_timestep, device=self.analyzer.device)
            temp_reachset.recalculate = True

            # Perform single-step concrete propagation
            t_start = time.time()
            parent_reachset.populate_next_reachable_set(
                self.analyzer.bounded_cl_system,
                temp_reachset,
                training=False
            )
            t_elapsed = time.time() - t_start
            total_time += t_elapsed

            # Extract bounds
            bounds = temp_reachset.full_set.detach().cpu().numpy()

            # Add calculation to current horizon
            self.horizons[current_timestep].add_calculation(
                bounds=bounds,
                calc_type=CalculationType.CONCRETE,
                origin_timestep=start_timestep,
                computation_time=t_elapsed,
                step_size=1,  # Each iteration is a single step
                notes=f'Concrete step {step+1}/{num_steps} from t={start_timestep}'
            )

            # Update parent reference for next iteration
            current_parent_timestep = current_timestep

        print(f"Concrete propagation of {num_steps} steps: "
              f"total time={total_time:.4f}s | "
              f"final vol @t={current_timestep}: {np.prod(bounds[:, 1] - bounds[:, 0]):.6f}")

        if visualize and self.dynamic_plot:
            self.visualize()

        return t_elapsed

    def empirical(self, start: int, end: int, num_samples: int = 10000, visualize=True):
        """
        Use dynamics to calculate actual reachset
        Args:
            parent_timestep: Timestep to sample from
            end: Target timestep
            num_samples: Number of trajectories to sample
        """

        # Get parent horizon
        if start not in self.horizons:
            print(f"Error: No horizon exists at timestep {start}")
            return False

        parent_horizon = self.horizons[start]

        # Get tightest bounds from parent horizon
        init_bounds = parent_horizon.get_tight_bound()
        if init_bounds is None:
            print(f"Error: No bounds available at timestep {start}")
            return False

        # Create horizon at target timestep if it doesn't exist
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        num_states = init_bounds.shape[0]

        t_start = time.time()

        # Sample initial states uniformly from parent bounds
        np.random.seed(None)

        # Run actual dynamics forward
        x0 = np.random.uniform(
            low=init_bounds[:, 0],
            high=init_bounds[:, 1],
            size=(num_samples, num_states)
        )
        xt = x0
        for step in range(start, end):
            u_nn = self.analyzer.cl_system.dynamics.control_nn(
                xt, self.analyzer.cl_system.controller.cpu()
            )
            xt1 = self.analyzer.cl_system.dynamics.dynamics_step(xt, u_nn)
            xt = xt1

        # Compute empirical bounds
        empirical_bounds = np.stack([
            np.min(xt, axis=0),
            np.max(xt, axis=0)
        ], axis = 1)

        t_elapsed = time.time() - t_start


        # Add calculation to new horizon
        self.horizons[end].add_calculation(
            bounds=empirical_bounds,
            calc_type=CalculationType.EMPIRICAL,
            origin_timestep=end,  # Empirical starts new origin
            computation_time=t_elapsed,
            step_size=end - start,
            num_samples=num_samples,
            notes=f'Empirical from t={start}, {num_samples} samples'
        )

        # Print detailed info
        print("=" * 20 + " Empirical " + "=" * 20)
        print(f"  From t={start} to t={end}")
        print(f"  Samples: {num_samples}")
        print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {np.prod(empirical_bounds[:, 1] - empirical_bounds[:, 0]):.6f}")
        print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")

        if visualize and self.dynamic_plot:
            self.visualize()

        return t_elapsed

    def symbolic(self, start: int, end: int, visualize=True):
        """
        Compute symbolic reachable set
        Args: start timestep, end timestep
        Returns: computation time
        """

        k = end - start
        if k > self.analyzer.max_diff:
            print(f"Error: Symbolic propagation limited to {self.analyzer.max_diff} steps")
            return False

        # Get k-step bounded system
        bounded_sys_k = self.analyzer.bounded_cl_systems.get(k - 1)
        if bounded_sys_k is None:
            print(f"No {k}-step bounded system available")
            print(f"Available: 1 to {len(self.analyzer.bounded_cl_systems)} steps")
            return None

        # Get parent horizon
        if start not in self.horizons:
            print(f"Error: No horizon exists at timestep {start}")
            return False

        parent_horizon = self.horizons[start]

        # Create horizon at target timestep if it doesn't exist
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        # Use analyzer's symbolic propagation
        t_start = time.time()

        # Set the parent's bounds as the starting point
        parent_reachset = parent_horizon.reachable_set

        # Create temporary reachable set for result
        temp_reachset = ReachableSet(t=end, device=self.analyzer.device)

        # Propagate
        t_start = time.time()
        parent_reachset.populate_next_reachable_set(
            bounded_sys_k,
            temp_reachset,
            training=False
        )
        t_elapsed = time.time() - t_start

        # Get bounds
        bounds = temp_reachset.full_set.detach().cpu().numpy()

        # Create or get horizon at target timestep
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        # Add calculation to horizon
        self.horizons[end].add_calculation(
            bounds=bounds,
            calc_type=CalculationType.SYMBOLIC,
            origin_timestep=start,
            computation_time=t_elapsed,
            step_size=k,
            notes=f"Symbolic {k}-step from t={start}"
        )

        # Print info
        print("=" * 20 + " Symbolic " + "=" * 20)
        print(f"  Parent Volume: {parent_horizon.get_tight_volume()}")
        print(f"  From t={start} to t={end} (k={k} steps)")
        print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {np.prod(bounds[:, 1] - bounds[:, 0]):.6f}")
        print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")

        if visualize and self.dynamic_plot:
            self.visualize()

        return t_elapsed

    def _setup_plot(self):
        """Setup the plot with GridSpec layout"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

        axes = {
            'main': fig.add_subplot(gs[0, :]),
            'info': fig.add_subplot(gs[1, :])
        }

        fig.suptitle('Reachability Analysis Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig, axes

    def visualize(self):
        """Update visualization with current state"""
        if not self.dynamic_plot:
            return

        # Clear axes
        self.axes['main'].clear()
        self.axes['info'].clear()

        # Plot all horizons and their tightest bounds
        all_bounds = []

        for t in sorted(self.horizons.keys()):
            horizon = self.horizons.get(t)
            if horizon is None:
                continue

            # Get tightest bounds for this timestep
            bounds = horizon.get_tight_bound()
            if bounds is None:
                continue

            all_bounds.append(bounds)

            # Color based on timestep (gradient from blue to red)
            if t == 0:
                color = 'black'
                alpha = 0.5
            else:
                # Create gradient color
                color_val = min(t / max(self.horizons.keys()), 1.0) if self.horizons.keys() else 0
                color = (color_val, 0, 1 - color_val)  # Blue to red gradient
                alpha = 0.3

            self._plot_rectangle(self.axes['main'], bounds,
                                edgecolor=color, facecolor=color, alpha=alpha,
                                linewidth=2 if t == max(self.horizons.keys()) else 1)

        # Set axis limits
        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min = np.min(all_bounds_array[:, 0, 0])
            x_max = np.max(all_bounds_array[:, 0, 1])
            y_min = np.min(all_bounds_array[:, 1, 0])
            y_max = np.max(all_bounds_array[:, 1, 1])

            x_range = x_max - x_min
            y_range = y_max - y_min
            padding_x = x_range * 0.1 if x_range > 0 else 0.1
            padding_y = y_range * 0.1 if y_range > 0 else 0.1

            self.axes['main'].set_xlim(x_min - padding_x, x_max + padding_x)
            self.axes['main'].set_ylim(y_min - padding_y, y_max + padding_y)

        self.axes['main'].set_xlabel('State 1', fontsize=12)
        self.axes['main'].set_ylabel('State 2', fontsize=12)
        self.axes['main'].set_title('Active Reachable Sets', fontsize=14, fontweight='bold')
        self.axes['main'].grid(True, alpha=0.3)
        self.axes['main'].set_aspect('equal', adjustable='datalim')

        self._plot_info_panel()

        plt.draw()
        plt.pause(0.01)

    def _plot_rectangle(self, ax, bounds, **kwargs):
        """Helper to plot rectangle"""
        xy = bounds[:2, 0]
        width = bounds[0, 1] - bounds[0, 0]
        height = bounds[1, 1] - bounds[1, 0]
        rect = Rectangle(xy, width, height, **kwargs)
        ax.add_patch(rect)

    def _plot_info_panel(self):
        """Plot info panel"""
        ax = self.axes['info']
        ax.axis('off')

        ax.text(0.5, 0.95, 'Calculation Summary',
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)

        y = 0.85
        total_horizons = len(self.horizons)
        total_calcs = sum(len(h.calculations) for h in self.horizons.values())
        stats_text = f"Total Horizons: {total_horizons}\n"
        stats_text += f"Total Calculations: {total_calcs}\n"
        stats_text += f"Timesteps: 0-{max(self.horizons.keys())}\n"

        ax.text(0.1, y, stats_text, va='top', fontsize=10,
                transform=ax.transAxes, family='monospace')

        y = 0.65
        ax.text(0.1, y, 'Recent Horizons:', va='top', fontsize=11,
                fontweight='bold', transform=ax.transAxes)

        y = 0.60
        recent_timesteps = sorted(self.horizons.keys(), reverse=True)[:8]
        for t in recent_timesteps:
            horizon = self.horizons[t]
            text = f"t={t:2d} | {len(horizon.calculations)} calc(s) | vol={horizon.get_tight_volume():.4f}"
            ax.text(0.1, y, text, va='top', fontsize=9,
                   transform=ax.transAxes, family='monospace')
            y -= 0.04

        y = 0.15
        ax.text(0.1, y, 'Legend:', va='top', fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        y -= 0.05
        ax.text(0.1, y, '🔵 Concrete  🟢 Symbolic  🟣 Empirical',
                va='top', fontsize=9, transform=ax.transAxes)


# =================== Setup and Testing Functions ===================#

def setup_analyzer(system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz'):
    """Setup analyzer for simulation testing"""
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), 'nfl_robustness_training/src'))

    device = 'cpu'

    if system_type == 'DoubleIntegrator':
        controller = load_controller('DoubleIntegrator', controller_name, False, device=device)
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)

        init_range = torch.tensor([[2.5, 3.0], [-0.25, 0.25]], device=device)
        time_horizon = 30
        max_diff = 10

    elif system_type == 'Unicycle_NL':
        controller = load_controller('Unicycle_NL', controller_name, False, device=device)
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        init_range = torch.tensor([
            [-9.55, -9.45],
            [3.45, 3.55],
            [-np.pi/24, np.pi/24]
        ], device=device)
        time_horizon = 52
        max_diff = 10

    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Create analyzer
    analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=max_diff, device=device)

    print(f"  Created analyzer for {system_type}")
    print(f"  Time horizon: {time_horizon}, Max symbolic steps: {max_diff}")

    return analyzer


def test():
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer, dynamic_plot=True)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])
    t = 0
    while t<15:
        tester.concrete(t,t+3)
        tester.symbolic(t, t+3)
        tester.horizons[t+3].list_calculations()
        tester.empirical(t,t+1)
        t+=1

def animate():
    """Create animation of reachability propagation following test pattern"""
    print("=" * 80)
    print("CREATING ANIMATION")
    print("=" * 80)

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    # Create tester without dynamic plot (we'll save frames instead)
    tester = ReachabilityTester(analyzer, dynamic_plot=False)

    print("\nGenerating animation frames...")

    frames = []
    colors_map = {
        CalculationType.CONCRETE: 'blue',
        CalculationType.SYMBOLIC: 'green',
        CalculationType.EMPIRICAL: 'purple'
    }

    # Generate frames following the same pattern as test
    t = 0
    max_t = 12  # Stop at 12 to avoid going past 15 with t+3

    while t <= max_t:
        print(f"Generating frame for t={t}")

        # Compute reachable sets - ONLY CONCRETE
        if t + 3 <= 15:
            tester.concrete(t, t+3)

        tester.empirical(t, t+1)

        # Create NEW figure for each frame
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        ax.set_title(f'Reachability Analysis - Time t={t}', fontsize=14, fontweight='bold')
        ax.set_xlabel('State 1', fontsize=12)
        ax.set_ylabel('State 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot all horizons
        all_bounds = []

        # For current timestep, show individual calculation bounds
        if t + 3 in tester.horizons:
            horizon_t3 = tester.horizons[t + 3]

            # Draw individual concrete propagation steps (t+1, t+2, t+3)
            for calc_id, calc_info in horizon_t3.calculations.items():
                if calc_info['calc_type'] == CalculationType.CONCRETE:
                    bounds = calc_info['bounds']
                    all_bounds.append(bounds)
                    step_num = calc_info['step_size']

                    # Color based on step
                    if step_num == 1:
                        color = 'lightblue'
                        label = f'Concrete t+1'
                        alpha = 0.3
                    elif step_num == 2:
                        color = 'cornflowerblue'
                        label = f'Concrete t+2'
                        alpha = 0.4
                    elif step_num == 3:
                        color = 'blue'
                        label = f'Concrete t+3'
                        alpha = 0.5
                    else:
                        continue

                    rect = Rectangle(
                        bounds[:2, 0],
                        bounds[0, 1] - bounds[0, 0],
                        bounds[1, 1] - bounds[1, 0],
                        edgecolor=color,
                        facecolor='none',
                        alpha=alpha,
                        linewidth=2,
                        linestyle='--',
                        label=label
                    )
                    ax.add_patch(rect)

        # Draw empirical bound at t+1
        if t + 1 in tester.horizons:
            horizon_t1 = tester.horizons[t + 1]
            for calc_id, calc_info in horizon_t1.calculations.items():
                if calc_info['calc_type'] == CalculationType.EMPIRICAL:
                    bounds = calc_info['bounds']
                    all_bounds.append(bounds)

                    rect = Rectangle(
                        bounds[:2, 0],
                        bounds[0, 1] - bounds[0, 0],
                        bounds[1, 1] - bounds[1, 0],
                        edgecolor='purple',
                        facecolor='none',
                        alpha=0.6,
                        linewidth=2,
                        linestyle=':',
                        label='Empirical t+1'
                    )
                    ax.add_patch(rect)

        # Draw tightest bounds for all timesteps (past, current, and lookahead)
        for timestep in sorted(tester.horizons.keys()):
            horizon = tester.horizons[timestep]
            bounds = horizon.get_tight_bound()

            if bounds is None:
                continue

            all_bounds.append(bounds)

            # Color based on timestep relative to current
            if timestep == 0:
                color = 'black'
                alpha = 0.7
                label = 'Initial (tightest)'
                linewidth = 3
            elif timestep < t:
                # Past timesteps - show tightest
                color = 'gray'
                alpha = 0.4
                label = f'Tightest t={timestep}' if timestep == t-1 else None
                linewidth = 2
            elif timestep == t:
                # Current timestep
                color = 'orange'
                alpha = 0.8
                label = f'Tightest t={t}'
                linewidth = 3
            elif timestep == t + 3:
                # Lookahead tightest
                color = 'red'
                alpha = 0.9
                label = f'Tightest t+3'
                linewidth = 3
            else:
                # Other future timesteps
                color = 'orange'
                alpha = 0.3
                label = None
                linewidth = 1.5

            # Draw filled rectangle for tightest bounds
            rect = Rectangle(
                bounds[:2, 0],
                bounds[0, 1] - bounds[0, 0],
                bounds[1, 1] - bounds[1, 0],
                edgecolor=color,
                facecolor=color,
                alpha=alpha * 0.3,
                linewidth=linewidth,
                label=label
            )
            ax.add_patch(rect)

        # Set axis limits
        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min = np.min(all_bounds_array[:, 0, 0])
            x_max = np.max(all_bounds_array[:, 0, 1])
            y_min = np.min(all_bounds_array[:, 1, 0])
            y_max = np.max(all_bounds_array[:, 1, 1])

            x_range = x_max - x_min
            y_range = y_max - y_min
            padding_x = x_range * 0.15
            padding_y = y_range * 0.15

            ax.set_xlim(x_min - padding_x, x_max + padding_x)
            ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Add text info
        info_text = f"Current t: {t}\n"
        info_text += f"Showing:\n"

        # Count calculation types at t+3
        if t + 3 in tester.horizons:
            h = tester.horizons[t + 3]
            concrete_count = sum(1 for c in h.calculations.values() if c['calc_type'] == CalculationType.CONCRETE)
            info_text += f"  • {concrete_count} Concrete steps (t→t+3)\n"
            info_text += f"  • Tightest vol @t+3: {h.get_tight_volume():.4f}\n"

        # Count empirical at t+1
        if t + 1 in tester.horizons:
            h = tester.horizons[t + 1]
            empirical_count = sum(1 for c in h.calculations.values() if c['calc_type'] == CalculationType.EMPIRICAL)
            if empirical_count > 0:
                info_text += f"  • {empirical_count} Empirical (t→t+1)\n"
                info_text += f"  • Tightest vol @t+1: {h.get_tight_volume():.4f}\n"

        info_text += f"\nTotal timesteps: {len(tester.horizons)}"

        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Save frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Drop alpha channel to get RGB

        # Debug: print frame info
        print(f"  Frame {len(frames)}: shape={image.shape}, size={image.nbytes} bytes")

        frames.append(image)

        # Close this figure before moving to next
        plt.close(fig)

        # Move to next timestep
        t += 1

    # Save animation
    print("\n" + "="*80)
    print("SAVING ANIMATION")
    print("="*80)
    print(f"Total frames collected: {len(frames)}")

    if len(frames) == 0:
        print("ERROR: No frames were generated!")
        return

    # Print frame statistics
    frame_sizes = [f.nbytes for f in frames]
    print(f"Frame shape: {frames[0].shape}")
    print(f"Frame dtype: {frames[0].dtype}")
    print(f"Average frame size: {np.mean(frame_sizes)/1024:.1f} KB")
    print(f"Total data size: {sum(frame_sizes)/1024:.1f} KB")

    # Save as GIF using imageio (more reliable than matplotlib animation)
    import os
    try:
        import imageio
        use_imageio = True
        print("Using imageio for GIF creation")
    except ImportError:
        print("imageio not available, falling back to matplotlib animation")
        use_imageio = False

    output_dir = './animation_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reachability_animation.gif')

    if use_imageio:
        # Use imageio for better GIF support with higher quality settings
        print(f"Saving {len(frames)} frames to GIF...")

        # Try with higher quality settings
        try:
            # quantizer=0 means no color quantization, subrectangles=False prevents optimization
            imageio.mimsave(output_file, frames, duration=500, loop=0, quantizer=0, subrectangles=False)
        except TypeError:
            # Fallback if quantizer parameter not supported
            print("  (using default quality settings)")
            imageio.mimsave(output_file, frames, duration=500, loop=0)

        # Check file size
        file_size = os.path.getsize(output_file)
        print(f"✓ Animation saved to: {output_file}")
        print(f"  File size: {file_size/1024:.1f} KB ({file_size/1024/1024:.2f} MB)")

        # Also save as MP4 for better quality/compression
        mp4_file = os.path.join(output_dir, 'reachability_animation.mp4')
        print(f"\nAlso saving as MP4 for better quality...")
        try:
            imageio.mimsave(mp4_file, frames, fps=2, codec='libx264', quality=8)
            mp4_size = os.path.getsize(mp4_file)
            print(f"✓ MP4 saved to: {mp4_file}")
            print(f"  File size: {mp4_size/1024:.1f} KB ({mp4_size/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"  Could not create MP4: {e}")
            print(f"  (Try: pip install imageio-ffmpeg)")
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')

        im = ax.imshow(frames[0])
        ax.axis('off')

        def update(frame_num):
            im.set_data(frames[frame_num])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)

        writer = PillowWriter(fps=2)
        anim.save(output_file, writer=writer)
        print(f"✓ Animation saved to: {output_file}")
        plt.close(fig)

    # Also save key frames as separate images
    for i in [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]:
        if i < len(frames):
            frame_file = os.path.join(output_dir, f'frame_{i:03d}.png')
            plt.imsave(frame_file, frames[i])
            print(f"✓ Frame {i} saved to: {frame_file}")

    print("\n✓ Animation complete!")
    print(f"  Total frames: {len(frames)}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        animate()
    else:
        test()
