
"""
Reachable Set Simulator - Includes EKF, Linear Kalman filtering
Use with Double Integrator or Unicycle
"""

import numpy as np
import torch
from ast import literal_eval
from itertools import product
from copy import deepcopy
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum
import nfl_veripy.analyzers as analyzers
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.nn import load_controller as nfl_load_controller

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import cl_systems

from utils.nn import load_controller, controller2sequential
from utils.robust_training_utils import ReachableSet
from utils.robust_training_utils import Analyzer
from state_estimator import LinearKalmanEstimator, ExtendedKalmanEstimator


class CalculationType(Enum):
    CONCRETE = "concrete"
    SYMBOLIC = "symbolic"
    EMPIRICAL = "empirical"
    BACKWARD = "backward"


class ReachableSetHorizon:
    """
    Reachable set calculations at a specific timestep.
    """
    def __init__(self, global_timestep: int, device='cpu'):
        self.global_timestep = global_timestep
        self.device = device

        # Main reachable set that gets updated with tightest bound
        self.reachable_set = ReachableSet(t=global_timestep, device=device)
        self.reachable_set.recalculate = True

        # Record all individual calculation informaiton for this timestep
        self.calculations = {}  # maps calc_id -> dict
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
                    num_samples: Optional[int] = None, notes: str = "",
                    real_state: Optional[np.ndarray] = None):


        # Store calculation metadata
        self.calculations[self.calc_counter] = {
            'bounds': bounds.copy(),
            'calc_type': calc_type,
            'origin_timestep': origin_timestep,
            'computation_time': computation_time,
            'step_size': step_size,
            'num_samples': num_samples,
            'volume': np.prod(bounds[:, 1] - bounds[:, 0]),
            'notes': notes
        }

        # Store real_state if provided (for empirical with state tracking)
        if real_state is not None:
            self.calculations[self.calc_counter]['real_state'] = real_state.copy()

        self.calc_counter += 1

        # Update tight bounds
        if self.tight_bound is None:
            self.tight_bound = bounds.copy()
        else:
            # Tightest bound: max of lower bounds, min of upper bounds
            self.tight_bound[:, 0] = np.maximum(self.tight_bound[:, 0], bounds[:, 0])
            self.tight_bound[:, 1] = np.minimum(self.tight_bound[:, 1], bounds[:, 1])

            # Check if no intersection
            if np.any(self.tight_bound[:, 0] > self.tight_bound[:, 1]):
                print(f"Error: Empty intersection at t={self.global_timestep}\n")
                print(f"added bounds: {bounds}")

        # Update the ReachableSet with tightest bounds
        self.reachable_set.set_range(torch.tensor(self.tight_bound, dtype=torch.float32, device=self.device))

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
        """Get metadata for a calculation
            keyed by the index of the calculation at that timestep
        """

        return self.calculations.get(calc_id, None)

    def list_calculations(self):
        """Print all calculations at this timestep"""
        print(f"\nCalculations at t={self.global_timestep}:")
        print(f"{'ID':<5} {'Type':<12} {'Volume':<12} {'Time':<8} {'Notes'}")
        print("-" * 60)
        for calc_id, calc in self.calculations.items():
            print(f"{calc_id:<5} {calc['calc_type'].value:<12} {calc['volume']:<12.6f} "
                  f"{calc['computation_time']:<8.4f} {calc['notes']}")

        print(f"Tightest Bound: {self.get_tight_volume()}\n")


class ReachabilityTester:
    """
    Reachability calculations using ReachableSetHorizon.
    """
    def __init__(self, analyzer, process_noise_std=0.01, measurement_noise_std=0.05,
                 backward_analyzer=None):
        self.analyzer = analyzer
        self.backward_analyzer = backward_analyzer

        # Track horizons by timestep
        self.horizons: Dict[int, ReachableSetHorizon] = {}

        self.counter = 0

        # Initialize horizon at t=0
        init_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
        self.horizons[0] = ReachableSetHorizon(0, device=analyzer.device)

        # Sample random initial state
        num_states = init_bounds.shape[0]
        np.random.seed(None)
        initial_state = np.random.uniform(
            low=init_bounds[:, 0],
            high=init_bounds[:, 1],
            size=num_states
        )

        self.horizons[0].add_calculation(
            bounds=init_bounds,
            calc_type=CalculationType.EMPIRICAL,
            parent_id=None,
            origin_timestep=0,
            computation_time=0.0,
            step_size=0,
            notes='Initial set',
            real_state=initial_state  # Add real state tracking
        )

        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        self.estimator = self.create_estimator(analyzer, initial_state, init_bounds, process_noise_std, measurement_noise_std)

    def create_estimator(self, analyzer, initial_state, initial_bounds, process_noise_std, measurement_noise_std):
        """Create appropriate filter based on dynamics type"""
        dynamics_obj = analyzer.cl_system.dynamics

        if hasattr(dynamics_obj, "vt"):
            print("Using EKF filter")
            return ExtendedKalmanEstimator(
                initial_state,
                initial_bounds,
                dynamics=dynamics_obj,
                process_noise_std=process_noise_std,
                measurement_noise_std=measurement_noise_std
            )
        else:
            print("Using Linear Kalman Filter")
            # Initialize Kalman Filter
            if isinstance(analyzer.cl_system.dynamics.At, torch.Tensor):
                A = analyzer.cl_system.dynamics.At.cpu().numpy()
                B = analyzer.cl_system.dynamics.bt.cpu().numpy()
            else:
                A = analyzer.cl_system.dynamics.At
                B = analyzer.cl_system.dynamics.bt

            return LinearKalmanEstimator(
                initial_state,
                initial_bounds,
                A=A,
                B=B,
                process_noise_std=process_noise_std,
                measurement_noise_std=measurement_noise_std
            )



    def concrete(self, start_timestep: int, end: Optional[int] = None):
        """
        Get concrete reachable set
        Args: start timestep, end timestep
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

        print("="*20 + " Concrete " + "="*20)
        print(f"Starting concrete propagation: t={start_timestep} -> t={end} ({num_steps} steps)")

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
            print("right after add calc")

            # Update parent reference for next iteration
            current_parent_timestep = current_timestep

        print(f"Concrete propagation of {num_steps} steps: \n"
              f"total time={total_time:.4f}s\n"
              f"bounds= {bounds}\n"
              f"final vol @t={current_timestep}: {np.prod(bounds[:, 1] - bounds[:, 0]):.6f}\n")

        return t_elapsed

    # def empirical(self, start: int, end: int, num_samples: int = 10000):
    #     """
    #     Use dynamics to calculate actual reachset
    #     Args: start timestep, end timestep
    #         num_samples: Number of trajectories to sample
    #     """

    #     # Get parent horizon
    #     if start not in self.horizons:
    #         print(f"Error: No horizon exists at timestep {start}")
    #         return False

    #     parent_horizon = self.horizons[start]

    #     # Get tightest bounds from parent horizon
    #     init_bounds = parent_horizon.get_tight_bound()
    #     if init_bounds is None:
    #         print(f"Error: No bounds available at timestep {start}")
    #         return False

    #     # Create horizon at target timestep if it doesn't exist
    #     if end not in self.horizons:
    #         self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

    #     num_states = init_bounds.shape[0]

    #     t_start = time.time()

    #     np.random.seed(None)
    #     x0 = np.random.uniform(
    #         low=init_bounds[:, 0],
    #         high=init_bounds[:, 1],
    #         size=(num_samples, num_states)
    #     )
    #     xt = x0
    #     for step in range(start, end):
    #         u_nn = self.analyzer.cl_system.dynamics.control_nn(
    #             xt, self.analyzer.cl_system.controller.cpu()
    #         )
    #         xt1 = self.analyzer.cl_system.dynamics.dynamics_step(xt, u_nn)
    #         xt = xt1

    #     # Compute bounds
    #     empirical_bounds = np.stack([
    #         np.min(xt, axis=0),
    #         np.max(xt, axis=0)
    #     ], axis = 1)

    #     t_elapsed = time.time() - t_start


    #     # Add calculation to new horizon
    #     self.horizons[end].add_calculation(
    #         bounds=empirical_bounds,
    #         calc_type=CalculationType.EMPIRICAL,
    #         origin_timestep=end,  # Empirical starts new origin
    #         computation_time=t_elapsed,
    #         step_size=end - start,
    #         num_samples=num_samples,
    #         notes=f'Empirical from t={start}, {num_samples} samples'
    #     )

    #     # Print info
    #     print("=" * 20 + " Empirical " + "=" * 20)
    #     print(f"  From t={start} to t={end}")
    #     print(f"  Samples: {num_samples}")
    #     print(f"  Computed in {t_elapsed:.4f}s")
    #     print(f"  Volume: {np.prod(empirical_bounds[:, 1] - empirical_bounds[:, 0]):.6f}")
    #     print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")


    #     return t_elapsed

    def real_state_empirical(self, start: int, end: int):
        """
        Propagate actual state using dynamics with Kalman filtering.
        Tracks true hidden state while estimating with noisy measurements.

        Args:
            start: Starting timestep
            end: Target timestep

        Returns:
            Computation time
        """
        print("=" * 20 + " Empirical (Kalman) " + "=" * 20)
        if start not in self.horizons:
            print(f"Error: No horizon exists at timestep {start}")
            return False

        parent_horizon = self.horizons[start]

        # Find the empirical calculation with real_state at parent timestep
        parent_real_state = None
        parent_bounds = None
        for calc_data in parent_horizon.calculations.values():
            if (calc_data['calc_type'] == CalculationType.EMPIRICAL and
                'real_state' in calc_data):
                parent_real_state = calc_data['real_state']
                parent_bounds = calc_data['bounds']
                break

        if parent_real_state is None:
            print(f"Error: No empirical calculation with real_state at timestep {start}")
            return False

        num_states = parent_real_state.shape[0]

        # Reset Kalman filter to parent's state and bounds
        self.estimator.reset(parent_real_state, parent_bounds)

        t_start = time.time()

        # Run actual dynamics forward (TRUE hidden state)
        xt_true = parent_real_state.reshape(1, -1)

        for step in range(start, end):
            # === Propagate TRUE state ===
            u_nn_true = self.analyzer.cl_system.dynamics.control_nn(
                xt_true, self.analyzer.cl_system.controller.cpu()
            )
            xt1_true = self.analyzer.cl_system.dynamics.dynamics_step(xt_true, u_nn_true)
            xt_true = xt1_true

            # === Kalman Filter Predict Step ===
            xt_est = self.estimator.state.reshape(1, -1)

            if isinstance(self.estimator, ExtendedKalmanEstimator):
                #Extended kalman filter
                u_nn_est = self.analyzer.cl_system.dynamics.control_nn(
                    xt_est, self.analyzer.cl_system.controller.cpu()
                )
            else:
                # Linear Kalman filter
                xt_est = torch.tensor(self.estimator.state.reshape(1, -1), dtype=torch.float32)
                u_nn_est = self.analyzer.cl_system.dynamics.control_nn(
                    xt_est, self.analyzer.cl_system.controller.cpu()
                )

            # KF prediction (uses linear A, B matrices)
            predicted_state, predicted_bounds = self.estimator.predict(
                dynamics_fn= None, # no longer an argument?
                control_input=u_nn_est
            )

            # === Kalman Filter Update Step ===
            # Convert true state to numpy
            if isinstance(xt_true, torch.Tensor):
                true_state_np = xt_true.squeeze().cpu().numpy()
            else:
                true_state_np = xt_true.squeeze() if isinstance(xt_true, np.ndarray) else xt_true

            # Simulate noisy measurement of TRUE state
            measurement_noise = np.random.normal(0, self.measurement_noise_std, size=num_states)
            noisy_measurement = true_state_np + measurement_noise

            # Update KF estimate with noisy measurement
            self.estimator.update(noisy_measurement)

        # Final states
        if isinstance(xt_true, torch.Tensor):
            real_state = xt_true.squeeze().cpu().numpy()
        else:
            real_state = xt_true.squeeze() if isinstance(xt_true, np.ndarray) else xt_true

        estimated_state = self.estimator.state.copy()
        kf_bounds = self.estimator.bounds.copy()

        t_elapsed = time.time() - t_start

        # Create horizon at target timestep if it doesn't exist
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        # Add calculation with KF bounds AND real state
        self.horizons[end].add_calculation(
            bounds=kf_bounds,
            calc_type=CalculationType.EMPIRICAL,
            origin_timestep=end,
            computation_time=t_elapsed,
            step_size=end - start,
            num_samples=None,
            notes=f'KF estimate from t={start}',
            real_state=real_state  # Store the true state
        )


        print(f"  From t={start} to t={end}")
        print(f"  True State: {real_state}")
        print(f"  KF Estimate: {estimated_state}")
        print(f"  Estimation Error: {np.linalg.norm(real_state - estimated_state):.6f}")
        print(f"  Bounds Volume: {np.prod(kf_bounds[:, 1] - kf_bounds[:, 0]):.6f}")
        print(f"  Tightest overlapped volume at t = {end}: {self.horizons[end].get_tight_volume():.6f}\n")

        return t_elapsed

    def symbolic(self, start: int, end: int):
        """
        Compute symbolic reachable set
        Args: start timestep, end timestep
        Returns: computation time
        """

        k = end - start
        if k > self.analyzer.max_diff:
            print(f"Error: Symbolic propagation max is {self.analyzer.max_diff} steps")
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

        # Create horizon object at target timestep if it doesn't exist
        if end not in self.horizons:
            self.horizons[end] = ReachableSetHorizon(end, device=self.analyzer.device)

        t_start = time.time()

        # Set parent's bounds as the starting point
        parent_reachset = parent_horizon.reachable_set

        # Create temp reachable set for result
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
        # print(f"  From t={start} to t={end} (k={k} steps)")
        # print(f"  Computed in {t_elapsed:.4f}s")
        print(f"  Volume: {np.prod(bounds[:, 1] - bounds[:, 0]):.6f}")
        # print(f"  Tightest volume: {self.horizons[end].get_tight_volume():.6f}")

        return t_elapsed

    # def backward(self, final_state_range: np.ndarray, boundary_type: str = "rectangle", t_max: int = 5, overapprox: bool = True):
    #     # TODO: integrate with horizons
    #     """
    #     Compute backward reachable set (backprojection set)
        
    #     Args:
    #         final_state_range: The final state range to backproject to
    #         boundary_type: The type of boundary (e.g., "rectangle", "ellipsoid")
    #         t_max: The time horizon for backprojection
    #         overapprox: 
    #     """

    #     # if start_timestep not in self.horizons:
    #     #     print(f"Error: No horizon exists at timestep {start_timestep}")
    #     #     return False

    #     # if backproject_timestep is None:
    #     #     backproject_timestep = start_timestep - 1
    #     # elif backproject_timestep >= start_timestep:
    #     #     print(f"Error: backproject_timestep must be less than start_timestep")
    #     #     return False
    #     # elif backproject_timestep < 0:
    #     #     print(f"Error: backproject_timestep must be non-negative")
    #     #     return False

    #     # # Track total computation time
    #     # total_time = 0.0
    #     # current_parent_timestep = start_timestep
    #     # num_steps = start_timestep - backproject_timestep

    #     # print("="*20 + " Backward " + "="*20)
    #     # print(f"Starting backward propagation: t={start_timestep} -> t={backproject_timestep} ({num_steps} steps)")

    #     # perform backprojection
    #     target_set = constraints.state_range_to_constraint(final_state_range, boundary_type)
        
    #     backprojection_sets, analyzer_info = self.backward_analyzer.get_backprojection_set(
    #         target_set,
    #         t_max=t_max,
    #         overapprox=overapprox
    #     )

    #     print("DEBUG: backward reachablity complete")
    #     print(f"Backprojection sets: {backprojection_sets.range}")
    #     return backprojection_sets, analyzer_info

    def backward(self, start_timestep: int, num_steps: int = 5, boundary_type: str = "rectangle", overapprox: bool = True):
        """
        Compute backward reachable set (backprojection set) and store in horizons.
        
        Args:
            start_timestep: The target timestep index (T) to backproject FROM
            num_steps: Number of discrete timestep indices to backproject
            boundary_type: The type of boundary (e.g., "rectangle", "ellipsoid")
            overapprox: Whether to use over-approximation
        
        Returns:
            Computation time
        """
        
        if start_timestep not in self.horizons:
            print(f"Error: No horizon exists at timestep {start_timestep}")
            return False
        
        if self.backward_analyzer is None:
            print("Error: No backward analyzer available")
            return False
        
        earliest_timestep = start_timestep - num_steps
        if earliest_timestep < 0:
            print(f"Error: Backprojection would go to t={earliest_timestep} (negative timestep)")
            return False
        
        # Get dt and convert discrete timestep indices to continuous time
        dt = self.backward_analyzer.propagator.dynamics.dt
        t_max = num_steps * dt  # Convert timestep indices to seconds
        
        # Get target set
        start_horizon = self.horizons[start_timestep]
        final_state_range = start_horizon.get_tight_bound()
        
        if final_state_range is None:
            print(f"Error: No bounds available at timestep {start_timestep}")
            return False
        
        print("=" * 20 + " Backward " + "=" * 20)
        print(f"Computing backprojection from target at timestep index {start_timestep}")
        print(f"Going back {num_steps} timestep indices = {t_max}s simulation time")
        print(f"Target timestep indices: {earliest_timestep} to {start_timestep-1}")
        
        # Perform backprojection
        t_start = time.time()
        target_set = constraints.state_range_to_constraint(final_state_range, boundary_type)
        backprojection_sets, analyzer_info = self.backward_analyzer.get_backprojection_set(
            target_set,
            t_max=t_max,  # Pass continuous time
            overapprox=overapprox
        )
        t_elapsed = time.time() - t_start
        
        num_timesteps = backprojection_sets.get_t_max()
        
        if num_timesteps != num_steps:
            print(f"Warning: Expected {num_steps} timesteps but got {num_timesteps}")
            print(f"  Check that backward analyzer dt matches forward analyzer dt")
        
        print(f"\nStoring {num_timesteps} backprojection sets:")
        
        # Store each timestep's backprojection
        for i in range(num_timesteps):
            current_timestep = earliest_timestep + i
            steps_to_target = start_timestep - current_timestep
            
            constraint = backprojection_sets.get_constraint_at_time_index(i)
            
            if isinstance(constraint, constraints.PolytopeConstraint):
                bounds = constraint.to_range()
            elif isinstance(constraint, constraints.LpConstraint):
                bounds = constraint.range
            else:
                print(f"Warning: Unknown constraint type at index {i}")
                continue
            
            volume = np.prod(bounds[:, 1] - bounds[:, 0])
            
            if current_timestep not in self.horizons:
                self.horizons[current_timestep] = ReachableSetHorizon(
                    current_timestep, 
                    device=self.analyzer.device
                )
            
            self.horizons[current_timestep].add_calculation(
                bounds=bounds,
                calc_type=CalculationType.BACKWARD,
                origin_timestep=start_timestep,
                computation_time=t_elapsed / num_timesteps,
                step_size=steps_to_target,
                notes=f'Backproject from t={start_timestep} ({steps_to_target} steps to target)'
            )
            
            if num_steps <= 10 or i % 5 == 0 or i == num_timesteps - 1:
                print(f"  t={current_timestep} ({steps_to_target} steps to target): volume={volume:.6f}")
        
        print(f"\nBackward propagation complete: {t_elapsed:.4f}s")
        
        return t_elapsed

def setup_analyzer(system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz', init_range=None, max_diff: int = 10):
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

        # Use custom init_range if provided, otherwise use default
        if init_range is None:
            init_range = torch.tensor([[2.5, 3.0], [-0.25, 0.25]], device=device)
        else:
            # Convert numpy array to torch tensor if needed
            if isinstance(init_range, np.ndarray):
                init_range = torch.tensor(init_range, dtype=torch.float32, device=device)
            elif isinstance(init_range, torch.Tensor):
                init_range = init_range.to(device)
            else:
                init_range = torch.tensor(init_range, device=device)

        time_horizon = 30
        # max_diff = 10

    elif system_type == 'Unicycle_NL':
        controller = load_controller('Unicycle_NL', controller_name, False, device=device)
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        # Use custom init_range if provided, otherwise use default
        if init_range is None:
            init_range = torch.tensor([
                [-9.55, -9.45],
                [3.45, 3.55],
                [-np.pi/24, np.pi/24]
            ], device=device)
        else:
            # Convert numpy array to torch tensor if needed
            if isinstance(init_range, np.ndarray):
                init_range = torch.tensor(init_range, dtype=torch.float32, device=device)
            elif isinstance(init_range, torch.Tensor):
                init_range = init_range.to(device)
            else:
                init_range = torch.tensor(init_range, device=device)

        time_horizon = 52
        # max_diff = 10

    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Create analyzer
    analyzer = Analyzer(cl_dyn, time_horizon, init_range, max_diff=max_diff, device=device)

    print(f"  Created analyzer for {system_type}")
    print(f"  Time horizon: {time_horizon}, Max symbolic steps: {max_diff}")

    return analyzer

#TODO: maybe merge this with setup_analyzer above?
def setup_backward_analyzer(system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz', init_range=None, partitioner=None, propogator=None):
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), 'nfl_robustness_training/src'))

    device = 'cpu'

    if system_type == 'DoubleIntegrator':
        controller = load_controller('DoubleIntegrator', controller_name, False, device=device)
        # controller = nfl_load_controller('DoubleIntegrator', controller_name)
        ol_dyn = dynamics.DoubleIntegrator(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.ClosedLoopDynamics(controller, ol_dyn, device=device)

        # Use custom init_range if provided, otherwise use default
        if init_range is None:
            init_range = torch.tensor([[2.5, 3.0], [-0.25, 0.25]], device=device)
        else:
            # Convert numpy array to torch tensor if needed
            if isinstance(init_range, np.ndarray):
                init_range = torch.tensor(init_range, dtype=torch.float32, device=device)
            elif isinstance(init_range, torch.Tensor):
                init_range = init_range.to(device)
            else:
                init_range = torch.tensor(init_range, device=device)

        time_horizon = 30
        max_diff = 10

    elif system_type == 'Unicycle_NL':
        controller = load_controller('Unicycle_NL', controller_name, False, device=device)
        # controller = nfl_load_controller('Unicycle_NL', controller_name)
        ol_dyn = dynamics.Unicycle_NL(dt=0.2)
        ol_dyn.At_torch = ol_dyn.At_torch.to(device)
        ol_dyn.bt_torch = ol_dyn.bt_torch.to(device)
        ol_dyn.ct_torch = ol_dyn.ct_torch.to(device)
        cl_dyn = cl_systems.Unicycle_NL(controller, ol_dyn, device=device)

        # Use custom init_range if provided, otherwise use default
        if init_range is None:
            init_range = torch.tensor([
                [-9.55, -9.45],
                [3.45, 3.55],
                [-np.pi/24, np.pi/24]
            ], device=device)
        else:
            # Convert numpy array to torch tensor if needed
            if isinstance(init_range, np.ndarray):
                init_range = torch.tensor(init_range, dtype=torch.float32, device=device)
            elif isinstance(init_range, torch.Tensor):
                init_range = init_range.to(device)
            else:
                init_range = torch.tensor(init_range, device=device)

        time_horizon = 52
        max_diff = 10

    controller = controller2sequential(controller)
    analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, ol_dyn)

    if partitioner is not None:
        analyzer.partitioner = partitioner
    else:
        print("Setting default partitioner")
        analyzer.partitioner = {"type": "Uniform", "num_partitions": "[4, 4]"}
        print(f"  Using default partitioner for {system_type}")

    if propogator is not None:
        analyzer.propagator = propogator
    else:
        print("Setting custom propagator")
        analyzer.propagator = {"type": "CROWN", "boundary_type": "rectangle", "num_iterations": 1}
        print(f"  Using custom propagator for {system_type}")

    print(f"  Created backward analyzer for {system_type}")

    return analyzer