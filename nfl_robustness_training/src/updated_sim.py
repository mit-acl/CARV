"""
Reachable Set Simulator
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

        # Track all individual calculations for this timestep
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

        calc_id = self.calc_counter
        self.calc_counter += 1

        # Store calculation metadata
        self.calculations[calc_id] = {
            'calc_id': calc_id,
            'bounds': bounds.copy(),
            'calc_type': calc_type,
            'parent_id': parent_id,
            'origin_timestep': origin_timestep,
            'computation_time': computation_time,
            'step_size': step_size,
            'num_samples': num_samples,
            'volume': np.prod(bounds[:, 1] - bounds[:, 0]),
            'notes': notes
        }

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

        return calc_id

    def get_tight_bound(self):
        """Return the tightest bound (intersection of all calculations)"""
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
        self.active_calc_ids: Dict[int, int] = {}
        self.counter = 0

        # Initialize horizon at t=0
        init_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
        self.horizons[0] = ReachableSetHorizon(0, device=analyzer.device)
        init_calc_id = self.horizons[0].add_calculation(
            bounds=init_bounds,
            calc_type=CalculationType.EMPIRICAL,
            parent_id=None,
            origin_timestep=0,
            computation_time=0.0,
            step_size=0,
            notes='Initial set'
        )
        self.active_calc_ids[0] = init_calc_id

        if self.dynamic_plot:
            plt.ion()
            self.fig, self.axes = self._setup_plot()

    def concrete(self, parent_timestep: int, parent_calc_id: Optional[int] = None,
                 end: Optional[int] = None, visualize=True):
        """
        Compute concrete reachable set using the analyzer's built-in method.

        Args:
            parent_timestep: Timestep to start from
            parent_calc_id: Optional specific calculation ID at parent_timestep to use
            end: End timestep (defaults to parent_timestep + 1)
            visualize: Whether to update visualization

        Returns:
            calc_id of the new calculation or None if failed
        """

        # Get parent horizon
        if parent_timestep not in self.horizons:
            print(f"Error: No horizon exists at timestep {parent_timestep}")
            return None

        parent_horizon = self.horizons[parent_timestep]

        # Use specified calc_id or get the active one
        if parent_calc_id is None:
            parent_calc_id = self.active_calc_ids.get(parent_timestep)
            if parent_calc_id is None:
                print(f"Error: No active calculation at timestep {parent_timestep}")
                return None

        # Get parent calculation
        parent_calc = parent_horizon.get_calculation(parent_calc_id)
        if parent_calc is None:
            print(f"Error: Calculation {parent_calc_id} not found at timestep {parent_timestep}")
            return None

        # Set end timestep
        if end is None:
            end = parent_timestep + 1

        num_steps = end - parent_timestep
        if num_steps <= 0:
            print(f"Error: Invalid step count (start={parent_timestep}, end={end})")
            return None

        # Create horizon at target timestep if it doesn't exist
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        # Use parent's reachable set for propagation
        parent_reachset = parent_horizon.reachable_set

        # Create temporary reachable set for the result
        temp_reachset = ReachableSet(t=end, device=self.analyzer.device)
        temp_reachset.recalculate = True

        # Perform concrete propagation
        t_start = time.time()
        parent_reachset.populate_next_reachable_set(
            self.analyzer.bounded_cl_system,
            temp_reachset,
            training=False
        )
        t_elapsed = time.time() - t_start

        # Extract bounds
        bounds = temp_reachset.full_set.detach().cpu().numpy()

        # Add calculation to target horizon
        calc_id = self.horizons[end].add_calculation(
            bounds=bounds,
            calc_type=CalculationType.CONCRETE,
            parent_id=parent_calc_id,
            origin_timestep=parent_calc['origin_timestep'],
            computation_time=t_elapsed,
            step_size=num_steps,
            notes=f'Concrete from t={parent_timestep}'
        )

        # Update active calculation for this timestep
        self.active_calc_ids[end] = calc_id

        print(f"✓ Concrete: t={parent_timestep} → t={end} | "
              f"vol={np.prod(bounds[:, 1] - bounds[:, 0]):.6f} | "
              f"time={t_elapsed:.4f}s")

        if visualize and self.dynamic_plot:
            self.visualize()

        return calc_id


    def symbolic(self, parent_timestep: int, end: int, parent_calc_id: Optional[int] = None,
                 visualize=True):

        if parent_timestep not in self.horizons:
            print(f"No horizon at timestep {parent_timestep}")
            return None

        parent_horizon = self.horizons[parent_timestep]
        k = end - parent_timestep

        if k <= 0:
            print("Can't propagate backwards")
            return None

        if k > self.analyzer.max_diff:
            print(f"Warning: {k} steps exceeds max_diff={self.analyzer.max_diff}")

        # Get k-step bounded system
        bounded_sys_k = self.analyzer.bounded_cl_systems.get(k - 1)
        if bounded_sys_k is None:
            print(f"No {k}-step bounded system available")
            print(f"Available: 1 to {len(self.analyzer.bounded_cl_systems)} steps")
            return None

        # Get parent calculation info
        if parent_calc_id is not None:
            parent_calc = parent_horizon.get_calculation(parent_calc_id)
            if parent_calc is None:
                print(f"No calculation {parent_calc_id} at t={parent_timestep}")
                return None
            origin_timestep = parent_calc['origin_timestep']
        else:
            parent_calc_id = self.active_calc_ids.get(parent_timestep, 0)
            parent_calc = parent_horizon.get_calculation(parent_calc_id)
            origin_timestep = parent_calc['origin_timestep'] if parent_calc else parent_timestep

        # Use parent horizon's ReachableSet
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
        calc_id = self.horizons[end].add_calculation(
            bounds=bounds,
            calc_type=CalculationType.SYMBOLIC,
            parent_id=parent_calc_id,
            origin_timestep=origin_timestep,
            computation_time=t_elapsed,
            step_size=k,
            notes=f"Symbolic {k}-step from t={parent_timestep}"
        )

        # Set as active
        self.active_calc_ids[end] = calc_id

        # Print info
        print("=" * 20 + " Symbolic " + "=" * 20)
        print(f"  From t={parent_timestep} to t={end} (k={k} steps)")
        print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {self.horizons[end].calculations[calc_id]['volume']:.6f}")
        print(f"  Calculation ID: {calc_id}")
        print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")

        if visualize and self.dynamic_plot:
            self.plot()

        return calc_id

    def empirical(self, parent_timestep: int, end: int, num_samples=10000,
                  parent_calc_id: Optional[int] = None, visualize=True):
        """
        Sample actual dynamics to get actual reachable set.

        Args:
            parent_timestep: Timestep to sample from
            end: Target timestep
            num_samples: Number of trajectories to sample
            parent_calc_id: Specific calculation ID to use (if None, uses tight bounds)
            visualize: Update plot

        Returns:
            calc_id of new empirical calculation
        """
        if parent_timestep not in self.horizons:
            print(f"No horizon at timestep {parent_timestep}")
            return None

        parent_horizon = self.horizons[parent_timestep]

        # Get parent bounds
        if parent_calc_id is not None:
            parent_calc = parent_horizon.get_calculation(parent_calc_id)
            if parent_calc is None:
                print(f"No calculation {parent_calc_id} at t={parent_timestep}")
                return None
            init_bounds = parent_calc['bounds']
        else:
            init_bounds = parent_horizon.get_tight_bound()
            if init_bounds is None:
                print(f"No tight bounds at t={parent_timestep}")
                return None
            parent_calc_id = self.active_calc_ids.get(parent_timestep, 0)

        num_states = init_bounds.shape[0]

        t_start = time.time()

        # Sample initial states
        np.random.seed(None)
        x0s = np.random.uniform(
            low=init_bounds[:, 0],
            high=init_bounds[:, 1],
            size=(num_samples, num_states)
        )

        # Run actual dynamics forward
        xt = x0s
        for step in range(parent_timestep, end):
            u_nn = self.analyzer.cl_system.dynamics.control_nn(
                xt, self.analyzer.cl_system.controller.cpu()
            )
            xt1 = self.analyzer.cl_system.dynamics.dynamics_step(xt, u_nn)
            xt = xt1

        # Compute empirical bounds
        empirical_bounds = np.stack([
            np.min(xt, axis=0),
            np.max(xt, axis=0)
        ], axis=1)

        t_elapsed = time.time() - t_start

        # Create or get horizon at target timestep
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        # Add calculation to horizon
        calc_id = self.horizons[end].add_calculation(
            bounds=empirical_bounds,
            calc_type=CalculationType.EMPIRICAL,
            parent_id=parent_calc_id,
            origin_timestep=end,  # Empirical starts a new origin
            computation_time=t_elapsed,
            step_size=end - parent_timestep,
            num_samples=num_samples,
            notes=f"Empirical from t={parent_timestep}, {num_samples} samples"
        )

        # Set as active
        self.active_calc_ids[end] = calc_id

        # Print info
        print("=" * 20 + " Empirical " + "=" * 20)
        print(f"  From t={parent_timestep} to t={end}")
        print(f"  Samples: {num_samples}")
        print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {self.horizons[end].calculations[calc_id]['volume']:.6f}")
        print(f"  Calculation ID: {calc_id}")
        print(f"  Horizon: {self.horizons[end].calc_counter} calculation(s)")
        print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")

        if visualize and self.dynamic_plot:
            self.plot()

        return calc_id

    # =================== Utility Methods ===============#

    def get_horizon(self, timestep: int):
        """Get the ReachableSetHorizon object for a timestep"""
        return self.horizons.get(timestep, None)

    def get_tight_bound(self, timestep: int):
        """Get tightest bound at a timestep"""
        horizon = self.horizons.get(timestep)
        if horizon is None:
            return None
        return horizon.get_tight_bound()

    def set_active(self, timestep: int, calc_id: int):
        """Set which calculation is active for visualization"""
        horizon = self.horizons.get(timestep)
        if horizon is None:
            print(f"No horizon at t={timestep}")
            return

        if calc_id not in horizon.calculations:
            print(f"Calculation ID {calc_id} not at t={timestep}")
            return

        self.active_calc_ids[timestep] = calc_id
        print(f"✓ Set calculation ID {calc_id} as active at t={timestep}")

        if self.dynamic_plot:
            self.plot()

    def list_calculations(self, timestep: Optional[int] = None):
        """List calculations at a specific timestep or all timesteps"""
        if timestep is not None:
            horizon = self.horizons.get(timestep)
            if horizon is None:
                print(f"No horizon at t={timestep}")
                return
            horizon.list_calculations()
        else:
            print("\nAll Calculations:")
            for t in sorted(self.horizons.keys()):
                self.horizons[t].list_calculations()

    def list_horizons(self):
        """List all horizons with their statistics"""
        print(f"\n{'=' * 80}")
        print("ALL HORIZONS")
        print(f"{'=' * 80}")
        print(f"{'t':<5} {'# Calcs':<10} {'Tight Volume':<15}")
        print("-" * 80)
        for t in sorted(self.horizons.keys()):
            horizon = self.horizons[t]
            vol_str = f"{horizon.get_tight_volume():.6f}" if horizon.tight_bound is not None else "N/A"
            print(f"{t:<5} {len(horizon.calculations):<10} {vol_str:<15}")
        print(f"{'=' * 80}\n")

    def compare(self, timestep: int):
        """Compare all calculations at a timestep"""
        horizon = self.horizons.get(timestep)
        if horizon is None:
            print(f"No horizon at t={timestep}")
            return

        print(f"\n{'=' * 80}")
        print(f"HORIZON at t={timestep}")
        print(f"{'=' * 80}")
        print(f"Total calculations: {len(horizon.calculations)}")
        if horizon.tight_bound is not None:
            print(f"Tightest bound volume: {horizon.get_tight_volume():.6f}")
            print(f"Tightest bounds:\n{horizon.tight_bound.T}")
        print(f"{'=' * 80}\n")

        for calc_id, calc in horizon.calculations.items():
            active = "★" if self.active_calc_ids.get(timestep) == calc_id else " "
            parent_str = f"from calc {calc['parent_id']}" if calc['parent_id'] is not None else "initial"

            print(f"\n{active} Calculation ID {calc_id}:")
            print(f"  Type: {calc['calc_type'].value}")
            print(f"  Origin t={calc['origin_timestep']}")
            print(f"  Parent: {parent_str}")
            print(f"  Steps: {calc['step_size']}")
            if calc['num_samples'] is not None:
                print(f"  Samples: {calc['num_samples']}")
            print(f"  Computation Time: {calc['computation_time']:.4f}s")
            print(f"  Volume: {calc['volume']:.6f}")
            print(f"  Bounds:\n{calc['bounds'].T}")

    # ================== Plotting Methods ==================#

    def _setup_plot(self):
        """Setup interactive matplotlib figure"""
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)

        ax_main = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])

        return fig, {'main': ax_main, 'info': ax_info}

    def plot(self):
        """Update visualization"""
        if not self.dynamic_plot:
            return

        self.axes['main'].clear()
        self.axes['info'].clear()

        # Collect all bounds for axis scaling
        all_bounds = []

        # Plot active calculations
        for t in sorted(self.active_calc_ids.keys()):
            calc_id = self.active_calc_ids[t]
            horizon = self.horizons.get(t)
            if horizon is None:
                continue

            calc = horizon.get_calculation(calc_id)
            if calc is None:
                continue

            bounds = calc['bounds']
            all_bounds.append(bounds)

            # Color by type
            if t == 0:
                color, alpha = 'black', 0.3
            elif calc['calc_type'] == CalculationType.CONCRETE:
                color, alpha = 'blue', 0.3
            elif calc['calc_type'] == CalculationType.SYMBOLIC:
                color, alpha = 'green', 0.4
            else:  # EMPIRICAL
                color, alpha = 'purple', 0.4

            self._plot_rectangle(self.axes['main'], bounds,
                                edgecolor=color, facecolor=color, alpha=alpha)

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

    print(f"✓ Created analyzer for {system_type}")
    print(f"  Time horizon: {time_horizon}, Max symbolic steps: {max_diff}")

    return analyzer


def test():
    """Example using analyzer with actual propagation"""
    print("=" * 80)
    print("HORIZON-BASED REACHABILITY TESTING")
    print("=" * 80)

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    print("\nCreating interactive tester...")
    tester = ReachabilityTester(analyzer, dynamic_plot=True)

    print("\n" + "=" * 20 + " Initial State " + "=" * 20)
    print(tester.horizons[0])

    timestep =0
    while timestep<20:
        tester.conrete()

        tester.emprical(timestep, timestep+1)
        timestep+=1

def animate():
    """Create animations of reachability propagation"""
    print("=" * 80)
    print("CREATING ANIMATIONS")
    print("=" * 80)

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    tester = ReachabilityTester(analyzer, dynamic_plot=False)

    print("\nGenerating empirical propagation with concrete lookahead...")

    # For animation, we'll compute empirical at each step and show concrete lookahead
    max_timesteps = 15
    lookahead_steps = 3

    frames_data = []

    for t in range(max_timesteps):
        print(f"Frame {t + 1}/{max_timesteps}")

        # Compute empirical
        emp_calc_id = tester.empirical(t, t + 1, num_samples=1000, visualize=False)

        # Compute concrete lookahead
        lookahead_calc_ids = []
        current_t = t + 1
        for step in range(lookahead_steps):
            if current_t + step < max_timesteps + lookahead_steps:
                look_id = tester.concrete(current_t + step - 1, end=current_t + step, visualize=False)
                if look_id is not None:
                    lookahead_calc_ids.append((current_t + step, look_id))

        # Store frame data
        frame_data = {
            'timestep': t + 1,
            'empirical_calcs': [(i, tester.active_calc_ids[i]) for i in range(t + 2) if i in tester.horizons],
            'lookahead_calcs': lookahead_calc_ids
        }
        frames_data.append(frame_data)

    print(f"\n✓ Animation data prepared: {len(frames_data)} frames")
    print("Note: Full animation rendering not implemented in this version")
    print("Use the test() function to see interactive visualization")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        animate()
    else:
        test()
