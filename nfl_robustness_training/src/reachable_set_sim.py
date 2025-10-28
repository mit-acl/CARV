"""
Reachable Set Simulator with Logging and Visualization

This script simulates forward reachability calculations, logging each step
and allowing inspection of the reachable sets at any timestep.
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

class CalculationRecord:
    """Log entry for a calculation step"""
    def __init__(self, global_timestep:int, bounds:np.ndarray, count_child_steps: int,
                  calc_id:int, parent_calc_id: int, origin_timestep:int, calculation_type:CalculationType, computation_time: float,
                    step_size:int, num_samples: int=None, num_partitions:int = 1, notes:str =""):

        self.global_timestep = global_timestep
        self.bounds = bounds
        self.calc_id = calc_id                              #Used as the Identifier
        self.parent_calc_id = parent_calc_id                #Immediate predecessor's calculation ID
        self.count_child_steps = count_child_steps          #Number of calculations done from origin (The reachable set based on actual state. Eg: If concrete steps were taken from t=3, their origin would be the t=3 empirical reach-set)
        self.origin_timestep = origin_timestep              #From which empirical data calculations started from
        self.calculation_type = calculation_type            #empirical, symbolic, concrete
        self.computation_time = computation_time
        self.step_size = step_size                          #step size for calculations -> concrete=1,
        self.volume = np.prod(bounds[:,1]- bounds[:,0])
        self.num_samples = num_samples                      # Used for empirical
        self.num_partitions = num_partitions
        self.notes = notes

    def __repr__(self):
        return (f"ReachableSetLog(t={self.global_timestep}), "
                f"origin_t={self.origin_timestep}, "
                f"id={self.calc_id},"
                f"type={self.calculation_type}, "
                f"steps={self.step_size}, "
                f"volume={self.volume}"
                )

class ReachabilityTester:
    """
    Manual testing framework for reachability calculations.
    """
    def __init__(self, analyzer, dynamic_plot = True):
        self.analyzer = analyzer
        self.dynamic_plot = dynamic_plot

        # Track all calculations (can have multiple per timestep)
        self.calculations: Dict[int, CalculationRecord] = {}    # maps calc_id -> CalculationRecord
        self.calcs_by_timestep: Dict[int, List[int]] = {}       # maps timestep -> [calc_ids]
        self.calc_counter = 0                                   # Used for assdigning calc_id

        # Active calculation for each timestep (for visualization)
        self.active_calcs: Dict[int, int] = {0: 0}  # maps timestep -> calc_id
        # Map calc_id to its ReachableSet in analyzer
        self.calc_id_to_reachset: Dict[int, int] = {}  # maps calc_id -> "timestep" in analyzer



        init_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
        init_record = CalculationRecord(
            global_timestep=0,
            origin_timestep=0,
            calc_id=0,
            parent_calc_id = None,
            count_child_steps=0 ,
            calculation_type=CalculationType.EMPIRICAL,
            computation_time=0.0,
            step_size=0,
            bounds=init_bounds,
            notes ='Initial set'
        )

        self.calculations[0]=init_record
        self.calcs_by_timestep[0] = [0]
        self.active_calcs[0]=0
        self.calc_id_to_reachset[0] = 0
        self.calc_counter = 1


        if self.dynamic_plot:
            plt.ion()
            self.fig, self.axes = self._setup_plot()

    def concrete(self, parent_id: int, end: Optional[int]= None, visualize = True):
        """
        Do 1 step concrete propagation
        Takes as args:
            parent_calc_id: ID of parent calculation to sample from
            end: Target timestep
            num_samples: Number of trajectories to sample
            visualize: Update plot

        Returns:
            calc_id of new concrete calculation
        """

        if parent_id not in self.calculations:
            print(f"no calculation found with id {parent_id}")
            return
        parent_calc = self.calculations[parent_id]
        start = parent_calc.global_timestep
        if end is None:
            end = start+1

        # Get parent's reachable set
        parent_reachset_t = self.calc_id_to_reachset[parent_id]

        # Create temporary reachable set at target time
        temp_reachset_t = max(self.analyzer.reachable_sets.keys()) + 1
        self.analyzer.reachable_sets[temp_reachset_t] = ReachableSet(
            t=temp_reachset_t,
            device=self.analyzer.device
        )
        self.analyzer.reachable_sets[temp_reachset_t].recalculate = True

        #calculate next reachable set
        t_start = time.time()
        self.analyzer.reachable_sets[parent_reachset_t].populate_next_reachable_set(
            self.analyzer.bounded_cl_system,
            self.analyzer.reachable_sets[temp_reachset_t],
            training = False
        )

        t_elapsed = time.time()-t_start

        #get bounds
        bounds = self.analyzer.reachable_sets[temp_reachset_t].full_set.detach().cpu().numpy()

        #record calculation
        calc_id = self.calc_counter
        self.calc_counter+=1

        record = CalculationRecord(
            global_timestep=end,
            bounds = bounds,
            calc_id=calc_id,
            parent_calc_id=parent_id,
            origin_timestep=parent_calc.origin_timestep,
            calculation_type=CalculationType.CONCRETE,
            computation_time=t_elapsed,
            step_size=1,
            count_child_steps=parent_calc.count_child_steps+1
        )

        self.calculations[calc_id] = record
        self.active_calcs[end] = calc_id
        self.calc_id_to_reachset[calc_id] = temp_reachset_t
        if end not in self.calcs_by_timestep:
            self.calcs_by_timestep[end] = []
        self.calcs_by_timestep[end].append(calc_id)


        # Print info
        print("="*20 + "Concrete" + "="*20)
        print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {record.volume:.6f}")
        print(f"  New Calculation ID: {calc_id}")

        if visualize and self.dynamic_plot:
            self.plot()

        return calc_id

    def symbolic(self, parent_id:int, end:int, visualize:True):
        """
        Perform symbolic (k-step) propagation.
        Takes as Args:
            parent_calc_id: ID of parent calculation to sample from
            end: Target global-timestep

        Returns:
            calc_id of new symbolic calculation
        """

        if parent_id not in self.calculations:
            print(f"no calculation found with id {parent_id}")
            return
        parent_calc = self.calculations[parent_id]
        start = parent_calc.global_timestep
        k = end - start

        if k<=0:
            print("can't propagate backwards")
            return

        if k > self.analyzer.max_diff:
            print(f"Warning: {k} steps exceeds max_diff={self.analyzer.max_diff}")

        #get k-step bounded system
        bounded_sys_k = self.analyzer.bounded_cl_systems.get(k - 1)
        if bounded_sys_k is None:
            print(f"No {k}-step bounded system available")
            print(f"Available: 1 to {len(self.analyzer.bounded_cl_systems)} steps")
            return None

        #get parent's reachable set
        parent_reachset_t = self.calc_id_to_reachset[parent_id]

        #get key of temporary timestep in analyzer
        temp_reachset_t = max(self.analyzer.reachable_sets.keys()) + 1
        self.analyzer.reachable_sets[temp_reachset_t] = ReachableSet(
            t=temp_reachset_t, device=self.analyzer.device
        )

        #propagate
        t_start = time.time()
        self.analyzer.reachable_sets[parent_reachset_t].populate_next_reachable_set(
            bounded_sys_k,
            self.analyzer.reachable_sets[temp_reachset_t],
            training = False
            )
        t_elapsed = time.time()- t_start

        #get bounds
        bounds = self.analyzer.reachable_sets[temp_reachset_t].full_set.detach().cpu().numpy()

        #record calc
        calc_id = self.calc_counter
        self.calc_counter+=1

        # Mark as symbolic
        self.analyzer.reachable_sets[temp_reachset_t].symbolic = True

        #Record
        record = CalculationRecord(
            global_timestep=end,
            bounds=bounds,
            calc_id=calc_id,
            count_child_steps=parent_calc.count_child_steps+1,
            parent_calc_id = parent_id,
            origin_timestep=parent_calc.origin_timestep,
            calculation_type=CalculationType.SYMBOLIC,
            computation_time=t_elapsed,
            step_size=k,
            num_partitions=1,
            notes=f"Symbolic {k}-step from t={start} to t{end}"
            )

        self.calculations[calc_id]= record
        self.active_calcs[end] = calc_id
        self.calc_id_to_reachset[calc_id] = temp_reachset_t
        if end not in self.calcs_by_timestep:
            self.calcs_by_timestep[end] = []
        self.calcs_by_timestep[end].append(calc_id)

        print("="*20 + "Symbolic" + "="*20)
        print(f"computed in {t_elapsed:.4f}s")
        print(f"timestep: {end}")
        print(f"Volume:{record.volume}")
        print(f"calc ID: {calc_id}")

        if visualize and self.dynamic_plot:
            self.plot()
        return calc_id

    def empirical(self, parent_id, end, num_samples=10000, visualize= True):
        """
        Sample actual dynamics to get actual reachable set.

        Takes as args:
            parent_calc_id: ID of parent calculation to sample from
            end: Target global timestep
            num_samples: Number of trajectories to sample
            visualize: Update plot

        Returns:
            calc_id of new empirical calculation
        """
        if parent_id not in self.calculations:
            print(f" No calculation with ID {parent_id}")
            return None

        parent_calc = self.calculations[parent_id]
        start = parent_calc.global_timestep
        #get parent bounds
        init_bounds = parent_calc.bounds
        num_states = init_bounds.shape[0]

        t_start = time.time()
        # Sample initial states
        np.random.seed(None)

        # Run actual dynamics forward
        x0s = np.random.uniform(
            low=init_bounds[:, 0],
            high=init_bounds[:, 1],
            size=(num_samples, num_states)
        )
        xt = x0s
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

        temp_reachset_t = max(self.analyzer.reachable_sets.keys()) + 1
        self.analyzer.reachable_sets[temp_reachset_t] = ReachableSet(
            temp_reachset_t, device=self.analyzer.device
        )
        self.analyzer.reachable_sets[temp_reachset_t].full_set = torch.tensor(
            empirical_bounds, dtype = torch.float32, device = self.analyzer.device
        )
        self.analyzer.reachable_sets[temp_reachset_t].symbolic = False

        calc_id = self.calc_counter
        self.calc_counter+=1

        #Record
        record = CalculationRecord(
            global_timestep=end,
            bounds=empirical_bounds,
            count_child_steps=0,
            calc_id=calc_id,
            parent_calc_id=parent_id,
            origin_timestep=end,
            computation_time=t_elapsed,
            step_size=end - start,
            calculation_type=CalculationType.EMPIRICAL,
            notes=f"Actual Bounds at t = {end}"
        )

        self.calculations[calc_id] = record
        if end not in self.calcs_by_timestep:
            self.calcs_by_timestep[end] = []
        self.calcs_by_timestep[end].append(calc_id)
        self.active_calcs[end] = calc_id
        self.calc_id_to_reachset[calc_id] = temp_reachset_t
        print("="*20 + " Empirical " + "="*20)
        print(f"  Volume: {record.volume:.6f}")
        print(f"  New Calculation ID: {calc_id}")
        print(num_samples)

        if visualize and self.dynamic_plot:
            self.plot()

        return calc_id


    #=================== Below are for Printing/Visualization ===============#


    def set_active(self, timestep: int, calc_id: int):
        """Set which calculation is active for visualization"""
        if timestep not in self.calcs_by_timestep:
            print(f"No calculations at t={timestep}")
            return

        if calc_id not in self.calcs_by_timestep[timestep]:
            print(f"Calculation ID {calc_id} not at t={timestep}")
            return

        self.active_calcs[timestep] = calc_id
        print(f"✓ Set calculation ID {calc_id} as active at t={timestep}")

        if self.dynamic_plot:
            self.plot()

    def list_calculations(self, timestep: Optional[int] = None):
        if timestep is not None:
            if timestep not in self.calcs_by_timestep:
                print("no calculations at this timestep")
                return
            calc_ids = self.calcs_by_timestep[timestep]

            for id in calc_ids:
                calc = self.calculations[id]
                active = "★" if self.active_calcs.get(timestep) == id else " "
                parent_str = f"from ID {calc.parent_calc_id}" if calc.parent_calc_id is not None else "initial"
                origin_str = f"calculation originate from timestep: {calc.origin_timestep}" if calc.origin_timestep else "initial"
                print(f"{active} ID {id:3d} | {calc.calculation_type.value:10s} | "
                      f"{parent_str:15s} | f{origin_str}| vol={calc.volume:.6f}")
        else:
            for t in sorted(self.calcs_by_timestep.keys()):
                for id in self.calcs_by_timestep[t]:
                    calc = self.calculations[id]
                    parent_str = f"from ID {calc.parent_calc_id}" if calc.parent_calc_id is not None else "initial"
                    origin_str = f"calculation originate from timestep: {calc.origin_timestep}" if calc.origin_timestep else "initial"
                    print(f"ID {id:3d} | {calc.calculation_type.value:10s} | {parent_str} | {origin_str}")

    def _setup_plot(self):
        """Setup interactive matplotlib figure"""
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)

        ax_main = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])

        return fig, {'main': ax_main, 'info': ax_info}

    def compare(self, timestep: int):
        """Compare all calculations at a timestep"""

        if timestep not in self.calcs_by_timestep:
            print(f"No calculations at t={timestep}")
            return

        calc_ids = self.calcs_by_timestep[timestep]

        for calc_id in calc_ids:
            calc = self.calculations[calc_id]
            active = "★" if self.active_calcs.get(timestep) == calc_id else " "
            parent_str = f"ID {calc.parent_calc_id} (t={self.calculations[calc.parent_calc_id].global_timestep})" if calc.parent_calc_id is not None else "initial"

            print(f"\n{active} Calculation ID {calc_id}:")
            print(f"  Type: {calc.calculation_type.value}")
            print(f"  Origin t = {calc.origin_timestep}")
            print(f"  Parent: {parent_str}")
            print(f"  Steps: {calc.step_size}")
            if calc.num_samples is not None:
                print(f"  Samples: {calc.num_samples}")
            print(f"  Computation Time: {calc.computation_time:.4f}s")
            print(f"  Volume: {calc.volume:.6f}")
            print(f"  Bounds:\n{calc.bounds.T}")  # Transpose for readability





    #================== BELOW ARE SOLELY FOR PLOTTING/ANIMATION ==================#
    #=============================================================================#

    def plot(self):
        """Update visualization"""
        if not self.dynamic_plot:
            return

        self.axes['main'].clear()
        self.axes['info'].clear()

        # Collect all bounds for axis scaling
        all_bounds = []

        # Plot active calculations
        for t in sorted(self.active_calcs.keys()):
            calc_id = self.active_calcs[t]
            if calc_id not in self.calculations:
                continue

            calc = self.calculations[calc_id]
            all_bounds.append(calc.bounds)

            # Color by type
            if t == 0:
                color, alpha = 'black', 0.3
            elif calc.calculation_type == CalculationType.CONCRETE:
                color, alpha = 'blue', 0.3
            elif calc.calculation_type == CalculationType.SYMBOLIC:
                color, alpha = 'green', 0.4
            else:  # EMPIRICAL
                color, alpha = 'purple', 0.4

            self._plot_rectangle(self.axes['main'], calc.bounds,
                               edgecolor=color, facecolor=color, alpha=alpha)

        # Set axis limits based on all bounds with padding
        if all_bounds:
            all_bounds_array = np.array(all_bounds)
            x_min = np.min(all_bounds_array[:, 0, 0])
            x_max = np.max(all_bounds_array[:, 0, 1])
            y_min = np.min(all_bounds_array[:, 1, 0])
            y_max = np.max(all_bounds_array[:, 1, 1])

            # Add 10% padding
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

    def _plot_rectangle_on_ax(self, ax, bounds, **kwargs):
        """Helper to plot rectangle on specific axis"""
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
        total_calcs = len(self.calculations)
        stats_text = f"Total Calculations: {total_calcs}\n"
        stats_text += f"Timesteps: 0-{max(self.calcs_by_timestep.keys())}\n"

        ax.text(0.1, y, stats_text, va='top', fontsize=10,
                transform=ax.transAxes, family='monospace')

        y = 0.65
        ax.text(0.1, y, 'Recent Calculations:', va='top', fontsize=11,
                fontweight='bold', transform=ax.transAxes)

        y = 0.60
        recent_ids = sorted(self.calculations.keys(), reverse=True)[:8]
        for calc_id in recent_ids:
            calc = self.calculations[calc_id]
            active = "★" if self.active_calcs.get(calc.global_timestep) == calc_id else " "
            text = f"{active} ID{calc_id:3d} | t={calc.global_timestep:2d} | {calc.calculation_type.value[:4]}"
            ax.text(0.1, y, text, va='top', fontsize=9,
                   transform=ax.transAxes, family='monospace')
            y -= 0.04

        y = 0.15
        ax.text(0.1, y, 'Legend:', va='top', fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        y -= 0.05
        ax.text(0.1, y, '🔵 Concrete  🟢 Symbolic  🟣 Empirical',
                va='top', fontsize=9, transform=ax.transAxes)

    def animate_propagation(self, max_timesteps=20, lookahead_steps=3,
                           num_samples=1000, filename='reachability_animation.gif',
                           fps=2):
        """
        Create animation showing empirical propagation with concrete lookahead.

        Args:
            max_timesteps: How many empirical steps to show
            lookahead_steps: How many concrete steps to show ahead of empirical
            num_samples: Number of samples for empirical calculation
            filename: Output filename for animation
            fps: Frames per second
        """
        print(f"\n{'='*80}")
        print(f"CREATING ANIMATION: {max_timesteps} steps, {lookahead_steps}-step lookahead")
        print(f"{'='*80}")

        # Prepare figure for animation
        fig, ax = plt.subplots(figsize=(12, 8))

        # Storage for animation frames
        frames_data = []

        # Initial empirical ID
        current_empirical_id = 0

        for t in range(max_timesteps):
            print(f"\nFrame {t+1}/{max_timesteps}: Computing empirical at t={t+1}")

            # Calculate empirical for next step
            next_empirical_id = self.empirical(
                current_empirical_id,
                t + 1,
                num_samples=num_samples,
                visualize=False
            )

            # Calculate concrete lookahead from this empirical
            print(f"  Computing {lookahead_steps}-step concrete lookahead...")
            lookahead_ids = []
            current_id = next_empirical_id

            for step in range(lookahead_steps):
                lookahead_id = self.concrete(
                    current_id,
                    end=t + 2 + step,
                    visualize=False
                )
                if lookahead_id is not None:
                    lookahead_ids.append(lookahead_id)
                    current_id = lookahead_id

            # Store frame data
            frame_data = {
                'timestep': t + 1,
                'empirical_history': list(range(next_empirical_id + 1)),  # All empirical IDs so far
                'lookahead_ids': lookahead_ids,
                'current_empirical_id': next_empirical_id
            }
            frames_data.append(frame_data)

            # Update for next iteration
            current_empirical_id = next_empirical_id

        print(f"\n{'='*80}")
        print(f"RENDERING ANIMATION...")
        print(f"{'='*80}")

        # Create animation
        def animate_frame(frame_idx):
            ax.clear()
            frame = frames_data[frame_idx]

            # Collect all bounds to determine axis limits
            all_bounds = []

            # Plot empirical history (purple, solid)
            for emp_id in frame['empirical_history']:
                if emp_id not in self.calculations:
                    continue
                calc = self.calculations[emp_id]
                all_bounds.append(calc.bounds)
                self._plot_rectangle_on_ax(
                    ax, calc.bounds,
                    edgecolor='purple',
                    facecolor='purple',
                    alpha=0.3,
                    linewidth=2
                )

            # Plot lookahead (blue, dashed)
            for look_id in frame['lookahead_ids']:
                if look_id not in self.calculations:
                    continue
                calc = self.calculations[look_id]
                all_bounds.append(calc.bounds)
                self._plot_rectangle_on_ax(
                    ax, calc.bounds,
                    edgecolor='blue',
                    facecolor='blue',
                    alpha=0.2,
                    linewidth=1.5,
                    linestyle='--'
                )

            # Highlight current empirical (purple, thick)
            current_calc = self.calculations[frame['current_empirical_id']]
            self._plot_rectangle_on_ax(
                ax, current_calc.bounds,
                edgecolor='red',
                facecolor='none',
                linewidth=3
            )

            # Set axis limits based on all bounds with padding
            if all_bounds:
                all_bounds_array = np.array(all_bounds)
                x_min = np.min(all_bounds_array[:, 0, 0])
                x_max = np.max(all_bounds_array[:, 0, 1])
                y_min = np.min(all_bounds_array[:, 1, 0])
                y_max = np.max(all_bounds_array[:, 1, 1])

                # Add 10% padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                padding_x = x_range * 0.1 if x_range > 0 else 0.1
                padding_y = y_range * 0.1 if y_range > 0 else 0.1

                ax.set_xlim(x_min - padding_x, x_max + padding_x)
                ax.set_ylim(y_min - padding_y, y_max + padding_y)

            # Add labels
            ax.set_xlabel('State 1', fontsize=14)
            ax.set_ylabel('State 2', fontsize=14)
            ax.set_title(f'Reachability Propagation - Timestep {frame["timestep"]}',
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='purple', alpha=0.5, label='Empirical (actual)'),
                Patch(facecolor='blue', alpha=0.2, label=f'Concrete lookahead ({lookahead_steps} steps)'),
                Patch(facecolor='none', edgecolor='purple', linewidth=3, label='Current timestep')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

            # Add text info
            info_text = f"t = {frame['timestep']}\n"
            info_text += f"Empirical history: {len(frame['empirical_history'])} steps\n"
            info_text += f"Lookahead: {len(frame['lookahead_ids'])} steps"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            return ax

        # Create animation
        anim = FuncAnimation(
            fig,
            animate_frame,
            frames=len(frames_data),
            interval=1000/fps,  # milliseconds per frame
            repeat=True
        )

        # Save animation
        print(f"Saving animation to {filename}...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)

        print(f"✓ Animation saved to {filename}")
        print(f"  Total frames: {len(frames_data)}")
        print(f"  Duration: {len(frames_data)/fps:.1f} seconds")

        plt.close(fig)

        return filename

    def animate_symbolic_propagation(self, max_timesteps=15, symbolic_steps=4,
                                     num_samples=1000, filename='symbolic_animation.gif',
                                     fps=2):
        """
        Create animation showing empirical propagation with symbolic lookahead.

        Args:
            max_timesteps: How many empirical steps to show
            symbolic_steps: How many steps to jump with symbolic propagation
            num_samples: Number of samples for empirical calculation
            filename: Output filename for animation
            fps: Frames per second
        """
        print(f"\n{'='*80}")
        print(f"CREATING SYMBOLIC ANIMATION: {max_timesteps} steps, {symbolic_steps}-step symbolic lookahead")
        print(f"{'='*80}")

        # Prepare figure for animation
        fig, ax = plt.subplots(figsize=(12, 8))

        # Storage for animation frames
        frames_data = []

        # Initial empirical ID
        current_empirical_id = 0

        for t in range(max_timesteps):
            print(f"\nFrame {t+1}/{max_timesteps}: Computing empirical at t={t+1}")

            # Calculate empirical for next step
            next_empirical_id = self.empirical(
                current_empirical_id,
                t + 1,
                num_samples=num_samples,
                visualize=False
            )

            # Calculate symbolic lookahead from this empirical
            print(f"  Computing {symbolic_steps}-step symbolic lookahead from t={t+1} to t={t+1+symbolic_steps}...")
            symbolic_id = None

            # Do one symbolic jump of 'symbolic_steps' steps
            if t + 1 + symbolic_steps <= max_timesteps + symbolic_steps:
                symbolic_id = self.symbolic(
                    next_empirical_id,
                    end=t + 1 + symbolic_steps,
                    visualize=False
                )

            # Store frame data
            frame_data = {
                'timestep': t + 1,
                'empirical_history': list(range(next_empirical_id + 1)),  # All empirical IDs so far
                'symbolic_id': symbolic_id,
                'current_empirical_id': next_empirical_id
            }
            frames_data.append(frame_data)

            # Update for next iteration
            current_empirical_id = next_empirical_id

        print(f"\n{'='*80}")
        print(f"RENDERING SYMBOLIC ANIMATION...")
        print(f"{'='*80}")

        # Create animation
        def animate_frame(frame_idx):
            ax.clear()
            frame = frames_data[frame_idx]

            # Collect all bounds to determine axis limits
            all_bounds = []

            # Plot empirical history (purple, solid)
            for emp_id in frame['empirical_history']:
                if emp_id not in self.calculations:
                    continue
                calc = self.calculations[emp_id]
                all_bounds.append(calc.bounds)
                self._plot_rectangle_on_ax(
                    ax, calc.bounds,
                    edgecolor='purple',
                    facecolor='purple',
                    alpha=0.5,
                    linewidth=2
                )

            # Plot symbolic lookahead (green, dashed)
            if frame['symbolic_id'] is not None and frame['symbolic_id'] in self.calculations:
                calc = self.calculations[frame['symbolic_id']]
                all_bounds.append(calc.bounds)
                self._plot_rectangle_on_ax(
                    ax, calc.bounds,
                    edgecolor='green',
                    facecolor='green',
                    alpha=0.3,
                    linewidth=2,
                    linestyle='--'
                )

            # Highlight current empirical (purple, thick)
            current_calc = self.calculations[frame['current_empirical_id']]
            self._plot_rectangle_on_ax(
                ax, current_calc.bounds,
                edgecolor='purple',
                facecolor='none',
                linewidth=3
            )

            # Set axis limits based on all bounds with padding
            if all_bounds:
                all_bounds_array = np.array(all_bounds)
                x_min = np.min(all_bounds_array[:, 0, 0])
                x_max = np.max(all_bounds_array[:, 0, 1])
                y_min = np.min(all_bounds_array[:, 1, 0])
                y_max = np.max(all_bounds_array[:, 1, 1])

                # Add 10% padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                padding_x = x_range * 0.1 if x_range > 0 else 0.1
                padding_y = y_range * 0.1 if y_range > 0 else 0.1

                ax.set_xlim(x_min - padding_x, x_max + padding_x)
                ax.set_ylim(y_min - padding_y, y_max + padding_y)

            # Add labels
            ax.set_xlabel('State 1', fontsize=14)
            ax.set_ylabel('State 2', fontsize=14)
            ax.set_title(f'Symbolic Reachability Propagation - Timestep {frame["timestep"]}',
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='purple', alpha=0.5, label='Empirical (actual)'),
                Patch(facecolor='green', alpha=0.3, label=f'{symbolic_steps}-step symbolic lookahead'),
                Patch(facecolor='none', edgecolor='purple', linewidth=3, label='Current timestep')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

            # Add text info
            info_text = f"t = {frame['timestep']}\n"
            info_text += f"Empirical history: {len(frame['empirical_history'])} steps\n"
            if frame['symbolic_id'] is not None:
                info_text += f"Symbolic lookahead to t = {frame['timestep'] + symbolic_steps}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            return ax

        # Create animation
        anim = FuncAnimation(
            fig,
            animate_frame,
            frames=len(frames_data),
            interval=1000/fps,  # milliseconds per frame
            repeat=True
        )

        # Save animation
        print(f"Saving animation to {filename}...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=100)

        print(f"✓ Animation saved to {filename}")
        print(f"  Total frames: {len(frames_data)}")
        print(f"  Duration: {len(frames_data)/fps:.1f} seconds")

        plt.close(fig)

        return filename








def setup_analyzer(system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz'):
    """
    Setup analyzer for simlulation testing
    """
    import sys, os

    # Add path
    sys.path.insert(0, os.path.join(os.getcwd(), 'nfl_robustness_training/src'))

    device = 'cpu'

    if system_type == 'DoubleIntegrator':
        # Setup Double Integrator system
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
        # Setup Unicycle system
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

def animate():
    # Setup analyzer
    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')

    # Create tester (disable live plotting for animation)
    tester = ReachabilityTester(analyzer, dynamic_plot=False)

    # Create animation
    # filename = tester.animate_propagation(
    #     max_timesteps=15,           # Number of steps to animate
    #     lookahead_steps=3,          # How many concrete steps ahead
    #     num_samples=1000,           # Samples for empirical
    #     filename='my_animation3.gif', # Output filename
    #     fps=1                       # Frames per second
    # )

    filename = tester.animate_symbolic_propagation(
        max_timesteps=15,           # Number of steps to animate
        symbolic_steps=4,           # 4-step symbolic lookahead
        num_samples=10000,           # Samples for empirical
        filename='symbolic_animation.gif', # Output filename
        fps=1                       # Frames per second
    )

    print(f"Animation saved to: {filename}")



def test():
    """
    Example using analyzer with actual propagation.
    """
    print("="*80)
    print("REAL USAGE EXAMPLE")
    print("="*80)

    analyzer = setup_analyzer('DoubleIntegrator', 'constraint_default_more_data_5hz')
    # analyzer = setup_real_analyzer("Unicycle_NL", "natural_none_default")
    # Create tester
    print("\n2. Creating interactive tester...")
    tester = ReachabilityTester(analyzer, dynamic_plot=True)

    print("="*20 + "Init State" + "="*20)
    print(tester.calculations[0])

    #calculate actual bounds
    for i in range(20):
        tester.empirical(i,i+1,1000,True)

    id_0 = 0  # Initial set
    id_1 = tester.concrete(id_0, visualize=False)
    id_2 = tester.concrete(id_1, visualize=False)
    id_3 = tester.concrete(id_2, visualize=False)
    id_4 = tester.concrete(id_3, visualize=False)
    id_5_concrete = tester.concrete(id_4, visualize=True)

    t5_symbolic = tester.symbolic(0,5, visualize=True)

    tester.compare(5)

    print("\5. Continue from DIFFERENT parents:")
    print("-"*80)
    print("Option A: Continue from concrete (ID {})".format(id_5_concrete))
    id_6a = tester.concrete(id_5_concrete, visualize=False)

    print("\nOption B: Continue from symbolic (ID {})".format(t5_symbolic))
    id_6b = tester.concrete(t5_symbolic, visualize=False)

    print("\nOption C: Continue from empirical (ID {}) - tightest!".format(5))
    id_6c = tester.concrete(5, visualize=True)

    tester.compare(6)

    tester.list_calculations()

    # From t=6 option C (empirical chain), go to t=10 three ways
    id_10a = tester.symbolic(id_6c, 10, visualize=False)  # 4-step symbolic

    # Concrete chain
    id_7 = tester.concrete(id_6c, visualize=False)
    id_8 = tester.concrete(id_7, visualize=False)
    id_9 = tester.concrete(id_8, visualize=False)
    id_10b = tester.concrete(id_9, visualize=False)

    tester.compare(10)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--real':
        # Run simulation analyzer
        test()
    else:
        #run animations
        animate()
