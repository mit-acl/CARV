"""
Timing Analysis: Concrete vs Symbolic Reachability Computation
Compares sequential concrete steps to single/multi-step symbolic propagation
"""

from REAL_integrated_sim import setup_analyzer, ReachabilityTester, CalculationType
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TimingAnalyzer:
    """Analyze and compare timing of concrete vs symbolic reachability"""
    
    def __init__(self, system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz', num_trials=10, max_diff=10):
        """Initialize analyzer for timing tests
        
        Args:
            system_type: Type of dynamical system
            controller_name: Name of controller to use
            num_trials: Number of trials to run for each test (with random initial states)
            max_diff: Maximum number of symbolic steps allowed in single propagation
        """
        self.system_type = system_type
        self.controller_name = controller_name
        self.analyzer = setup_analyzer(system_type, controller_name, max_diff=max_diff)
        self.num_trials = num_trials
        
        # Get initial bounds for sampling
        self.init_bounds = self.analyzer.reachable_sets[0].full_set.cpu().numpy()
        
        # Storage for results (now storing stats across trials)
        self.results = {
            'concrete_sequential': {},  # horizon -> {'mean': x, 'std': y, 'trials': [...]}
            'symbolic_single': {},      # horizon -> {'mean': x, 'std': y, 'trials': [...]}
            'symbolic_multi': {},       # (horizon, num_steps) -> {'mean': x, 'std': y, 'trials': [...]}
        }
        
        self.volume_results = {
            'concrete_sequential': {},  # horizon -> {'mean': x, 'std': y, 'trials': [...]}
            'symbolic_single': {},      # horizon -> {'mean': x, 'std': y, 'trials': [...]}
            'symbolic_multi': {},       # (horizon, num_steps) -> {'mean': x, 'std': y, 'trials': [...]}
        }
        
    def test_concrete_sequential(self, horizon: int) -> Dict:
        """
        Test sequential concrete propagation over horizon with multiple trials
        
        Args:
            horizon: Number of timesteps to propagate
            
        Returns:
            Dict with 'mean', 'std', 'trials' for both time and volume
        """
        times = []
        volumes = []
        
        for trial in range(self.num_trials):
            # Sample random initial state within bounds
            init_state = np.random.uniform(
                low=self.init_bounds[:, 0],
                high=self.init_bounds[:, 1]
            )
            
            # Create fresh tester with sampled initial state
            # Note: ReachabilityTester always uses the analyzer's default init_bounds,
            # but we can work with the random state through empirical tracking
            tester = ReachabilityTester(self.analyzer)
            
            total_time = 0.0
            
            # Sequential concrete steps
            for t in range(horizon):
                result = tester.concrete(t, t + 1)
                if isinstance(result, dict):
                    total_time += result['time']
                else:
                    # Fallback for old return format
                    total_time += result
            
            # Get final volume
            final_horizon = tester.horizons[horizon]
            final_volume = final_horizon.get_tight_volume()
            
            times.append(total_time)
            volumes.append(final_volume)
        
        return {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_trials': times,
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'volume_trials': volumes
        }
    
    def test_symbolic_single_step(self, horizon: int) -> Dict:
        """
        Test single symbolic step spanning entire horizon with multiple trials
        
        Args:
            horizon: Horizon length (must be <= max_diff)
            
        Returns:
            Dict with statistics, or None if horizon exceeds max_diff
        """
        if horizon > self.analyzer.max_diff:
            print(f"Warning: horizon {horizon} exceeds max_diff {self.analyzer.max_diff}")
            return None
        
        times = []
        volumes = []
        
        for trial in range(self.num_trials):
            # Sample random initial state within bounds
            init_state = np.random.uniform(
                low=self.init_bounds[:, 0],
                high=self.init_bounds[:, 1]
            )
            
            # Create fresh tester
            tester = ReachabilityTester(self.analyzer)
            
            # Single symbolic step
            result = tester.symbolic(0, horizon)
            if isinstance(result, dict):
                step_time = result['time']
            else:
                # Fallback for old return format
                step_time = result
            
            # Get final volume
            final_horizon = tester.horizons[horizon]
            final_volume = final_horizon.get_tight_volume()
            
            times.append(step_time)
            volumes.append(final_volume)
        
        return {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_trials': times,
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'volume_trials': volumes
        }
    
    def test_symbolic_multi_step(self, horizon: int, num_steps: int) -> Dict:
        """
        Test multi-step symbolic propagation with multiple trials
        
        Args:
            horizon: Total horizon length
            num_steps: Number of symbolic steps to use
            
        Returns:
            Dict with statistics, or None if step size exceeds max_diff
        """
        step_size = horizon // num_steps
        if step_size > self.analyzer.max_diff:
            print(f"Warning: step_size {step_size} exceeds max_diff {self.analyzer.max_diff}")
            return None
        
        times = []
        volumes = []
        
        for trial in range(self.num_trials):
            # Sample random initial state within bounds
            init_state = np.random.uniform(
                low=self.init_bounds[:, 0],
                high=self.init_bounds[:, 1]
            )
            
            # Create fresh tester
            tester = ReachabilityTester(self.analyzer)
            
            total_time = 0.0
            current_t = 0
            
            # Multi-step symbolic propagation
            for i in range(num_steps):
                next_t = min(current_t + step_size, horizon)
                if next_t <= current_t:
                    break
                    
                result = tester.symbolic(current_t, next_t)
                if isinstance(result, dict):
                    total_time += result['time']
                else:
                    # Fallback for old return format
                    total_time += result
                current_t = next_t
                
            # Handle remainder if horizon not evenly divisible
            if current_t < horizon:
                result = tester.symbolic(current_t, horizon)
                if isinstance(result, dict):
                    total_time += result['time']
                else:
                    total_time += result
                
            # Get final volume
            final_horizon = tester.horizons[horizon]
            final_volume = final_horizon.get_tight_volume()
            
            times.append(total_time)
            volumes.append(final_volume)
        
        return {
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_trials': times,
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'volume_trials': volumes
        }
    
    def run_comparison(self, horizons: List[int], symbolic_step_counts: List[int] = [1, 2, 3, 5]):
        """
        Run full comparison across different horizons and symbolic step counts
        
        Args:
            horizons: List of horizon lengths to test
            symbolic_step_counts: List of symbolic step counts to test
        """
        print("=" * 80)
        print("TIMING ANALYSIS: CONCRETE vs SYMBOLIC REACHABILITY")
        print(f"Number of trials per test: {self.num_trials}")
        print("=" * 80)
        
        for horizon in horizons:
            print(f"\n{'=' * 40}")
            print(f"Testing Horizon = {horizon}")
            print(f"{'=' * 40}")
            
            # Test concrete sequential
            print(f"  [1/3] Concrete sequential ({horizon} steps, {self.num_trials} trials)...", end=" ")
            concrete_stats = self.test_concrete_sequential(horizon)
            self.results['concrete_sequential'][horizon] = concrete_stats
            self.volume_results['concrete_sequential'][horizon] = concrete_stats
            print(f"{concrete_stats['time_mean']:.4f}±{concrete_stats['time_std']:.4f}s, "
                  f"vol={concrete_stats['volume_mean']:.6f}±{concrete_stats['volume_std']:.6f}")
            
            # Test single symbolic step
            if horizon <= self.analyzer.max_diff:
                print(f"  [2/3] Symbolic single-step (1 step of {horizon}, {self.num_trials} trials)...", end=" ")
                symbolic_stats = self.test_symbolic_single_step(horizon)
                if symbolic_stats is not None:
                    self.results['symbolic_single'][horizon] = symbolic_stats
                    self.volume_results['symbolic_single'][horizon] = symbolic_stats
                    print(f"{symbolic_stats['time_mean']:.4f}±{symbolic_stats['time_std']:.4f}s, "
                          f"vol={symbolic_stats['volume_mean']:.6f}±{symbolic_stats['volume_std']:.6f}")
                    speedup = concrete_stats['time_mean'] / symbolic_stats['time_mean']
                    print(f"        Speedup: {speedup:.2f}x")
                else:
                    print("SKIPPED (exceeds max_diff)")
            else:
                print(f"  [2/3] Symbolic single-step SKIPPED (horizon {horizon} > max_diff {self.analyzer.max_diff})")
            
            # Test multi-step symbolic
            print(f"  [3/3] Symbolic multi-step ({self.num_trials} trials each):")
            for num_steps in symbolic_step_counts:
                if num_steps >= horizon:
                    continue
                    
                step_size = horizon // num_steps
                if step_size > self.analyzer.max_diff:
                    print(f"        {num_steps} steps SKIPPED (step_size {step_size} > max_diff)")
                    continue
                    
                print(f"        {num_steps} steps (each ~{step_size})...", end=" ")
                multi_stats = self.test_symbolic_multi_step(horizon, num_steps)
                
                if multi_stats is not None:
                    self.results['symbolic_multi'][(horizon, num_steps)] = multi_stats
                    self.volume_results['symbolic_multi'][(horizon, num_steps)] = multi_stats
                    print(f"{multi_stats['time_mean']:.4f}±{multi_stats['time_std']:.4f}s, "
                          f"vol={multi_stats['volume_mean']:.6f}±{multi_stats['volume_std']:.6f}")
                    speedup = concrete_stats['time_mean'] / multi_stats['time_mean']
                    print(f"              Speedup vs concrete: {speedup:.2f}x")
                else:
                    print("FAILED")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
    
    def create_plots(self, output_dir='./timing_analysis'):
        """Create comprehensive visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating plots in {output_dir}...")
        
        # Plot 1: Time vs Horizon (all methods)
        self._plot_time_vs_horizon(output_dir)
        
        # Plot 2: Speedup vs Horizon
        self._plot_speedup_vs_horizon(output_dir)
        
        # Plot 3: Volume comparison
        self._plot_volume_comparison(output_dir)
        
        # Plot 4: Efficiency analysis (time per step)
        self._plot_time_per_step(output_dir)
        
        # Plot 5: Multi-step symbolic comparison
        self._plot_multistep_comparison(output_dir)
        
        # Plot 6: Heatmap of speedup
        self._plot_speedup_heatmap(output_dir)
        
        # Plot 7: NEW - Direct symbolic vs concrete time comparison
        self._plot_symbolic_vs_concrete_direct(output_dir)
        
        print(f"All plots saved to {output_dir}/")
    
    def _plot_time_vs_horizon(self, output_dir):
        """Plot computation time vs horizon for all methods with error bars"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete sequential
        horizons = sorted(self.results['concrete_sequential'].keys())
        times = [self.results['concrete_sequential'][h]['time_mean'] for h in horizons]
        errors = [self.results['concrete_sequential'][h]['time_std'] for h in horizons]
        ax.errorbar(horizons, times, yerr=errors, fmt='o-', linewidth=2, markersize=8, 
                    capsize=5, capthick=2, label='Concrete Sequential', color='blue', alpha=0.8)
        
        # Symbolic single-step
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        times_sym = [self.results['symbolic_single'][h]['time_mean'] for h in horizons_sym]
        errors_sym = [self.results['symbolic_single'][h]['time_std'] for h in horizons_sym]
        ax.errorbar(horizons_sym, times_sym, yerr=errors_sym, fmt='s-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Symbolic Single-Step', color='green', alpha=0.8)
        
        # Symbolic multi-step (group by num_steps)
        multi_data = {}
        for (h, n), stats in self.results['symbolic_multi'].items():
            if n not in multi_data:
                multi_data[n] = {'horizons': [], 'times': [], 'errors': []}
            multi_data[n]['horizons'].append(h)
            multi_data[n]['times'].append(stats['time_mean'])
            multi_data[n]['errors'].append(stats['time_std'])
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(multi_data)))
        for i, (num_steps, data) in enumerate(sorted(multi_data.items())):
            sorted_idx = np.argsort(data['horizons'])
            hs_sorted = [data['horizons'][i] for i in sorted_idx]
            ts_sorted = [data['times'][i] for i in sorted_idx]
            es_sorted = [data['errors'][i] for i in sorted_idx]
            ax.errorbar(hs_sorted, ts_sorted, yerr=es_sorted, fmt='^--', linewidth=2, markersize=7,
                        capsize=4, capthick=1.5, label=f'Symbolic {num_steps}-Step', 
                        color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Computation Time vs Horizon ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_vs_horizon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ time_vs_horizon.png")
    
    def _plot_speedup_vs_horizon(self, output_dir):
        """Plot speedup (concrete/symbolic) vs horizon with error propagation"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Single-step speedup with error propagation
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        speedups = []
        speedup_errors = []
        
        for h in horizons_sym:
            if h in self.results['concrete_sequential']:
                concrete_mean = self.results['concrete_sequential'][h]['time_mean']
                concrete_std = self.results['concrete_sequential'][h]['time_std']
                symbolic_mean = self.results['symbolic_single'][h]['time_mean']
                symbolic_std = self.results['symbolic_single'][h]['time_std']
                
                speedup = concrete_mean / symbolic_mean
                # Error propagation for division: σ(a/b) ≈ |a/b| * sqrt((σa/a)² + (σb/b)²)
                rel_error = np.sqrt((concrete_std/concrete_mean)**2 + (symbolic_std/symbolic_mean)**2)
                speedup_error = speedup * rel_error
                
                speedups.append(speedup)
                speedup_errors.append(speedup_error)
            else:
                speedups.append(np.nan)
                speedup_errors.append(0)
        
        ax.errorbar(horizons_sym, speedups, yerr=speedup_errors, fmt='s-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Symbolic Single-Step', color='green', alpha=0.8)
        
        # Multi-step speedup
        multi_data = {}
        for (h, n), stats in self.results['symbolic_multi'].items():
            if h in self.results['concrete_sequential']:
                concrete_mean = self.results['concrete_sequential'][h]['time_mean']
                concrete_std = self.results['concrete_sequential'][h]['time_std']
                symbolic_mean = stats['time_mean']
                symbolic_std = stats['time_std']
                
                speedup = concrete_mean / symbolic_mean
                rel_error = np.sqrt((concrete_std/concrete_mean)**2 + (symbolic_std/symbolic_mean)**2)
                speedup_error = speedup * rel_error
                
                if n not in multi_data:
                    multi_data[n] = {'horizons': [], 'speedups': [], 'errors': []}
                multi_data[n]['horizons'].append(h)
                multi_data[n]['speedups'].append(speedup)
                multi_data[n]['errors'].append(speedup_error)
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(multi_data)))
        for i, (num_steps, data) in enumerate(sorted(multi_data.items())):
            sorted_idx = np.argsort(data['horizons'])
            hs_sorted = [data['horizons'][i] for i in sorted_idx]
            speedups_sorted = [data['speedups'][i] for i in sorted_idx]
            errors_sorted = [data['errors'][i] for i in sorted_idx]
            ax.errorbar(hs_sorted, speedups_sorted, yerr=errors_sorted, fmt='^--', linewidth=2, markersize=7,
                        capsize=4, capthick=1.5, label=f'Symbolic {num_steps}-Step', 
                        color=colors[i], alpha=0.8)
        
        # Reference line at 1x
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='No speedup')
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (Concrete Time / Symbolic Time)', fontsize=12, fontweight='bold')
        ax.set_title(f'Speedup vs Horizon ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_vs_horizon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ speedup_vs_horizon.png")
    
    def _plot_volume_comparison(self, output_dir):
        """Plot volume comparison across methods with error bars"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete
        horizons = sorted(self.volume_results['concrete_sequential'].keys())
        vols = [self.volume_results['concrete_sequential'][h]['volume_mean'] for h in horizons]
        errors = [self.volume_results['concrete_sequential'][h]['volume_std'] for h in horizons]
        ax.errorbar(horizons, vols, yerr=errors, fmt='o-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Concrete Sequential', color='blue', alpha=0.8)
        
        # Symbolic single
        horizons_sym = sorted(self.volume_results['symbolic_single'].keys())
        vols_sym = [self.volume_results['symbolic_single'][h]['volume_mean'] for h in horizons_sym]
        errors_sym = [self.volume_results['symbolic_single'][h]['volume_std'] for h in horizons_sym]
        ax.errorbar(horizons_sym, vols_sym, yerr=errors_sym, fmt='s-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Symbolic Single-Step', color='green', alpha=0.8)
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reachable Set Volume', fontsize=12, fontweight='bold')
        ax.set_title(f'Reachable Set Volume vs Horizon ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/volume_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ volume_comparison.png")
    
    def _plot_time_per_step(self, output_dir):
        """Plot time per step (efficiency) with error bars"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete: time per step
        horizons = sorted(self.results['concrete_sequential'].keys())
        time_per_step = [self.results['concrete_sequential'][h]['time_mean'] / h for h in horizons]
        errors = [self.results['concrete_sequential'][h]['time_std'] / h for h in horizons]
        ax.errorbar(horizons, time_per_step, yerr=errors, fmt='o-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Concrete Sequential', color='blue', alpha=0.8)
        
        # Symbolic single: time per step
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        time_per_step_sym = [self.results['symbolic_single'][h]['time_mean'] / h for h in horizons_sym]
        errors_sym = [self.results['symbolic_single'][h]['time_std'] / h for h in horizons_sym]
        ax.errorbar(horizons_sym, time_per_step_sym, yerr=errors_sym, fmt='s-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Symbolic Single-Step', color='green', alpha=0.8)
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time per Timestep (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Computational Efficiency: Time per Timestep ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_per_step.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ time_per_step.png")
    
    def _plot_multistep_comparison(self, output_dir):
        """Compare different multi-step symbolic strategies with error bars"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by horizon
        horizon_data = {}
        for (h, n), stats in self.results['symbolic_multi'].items():
            if h not in horizon_data:
                horizon_data[h] = {'num_steps': [], 'times': [], 'errors': []}
            horizon_data[h]['num_steps'].append(n)
            horizon_data[h]['times'].append(stats['time_mean'])
            horizon_data[h]['errors'].append(stats['time_std'])
        
        # Plot each horizon as a line
        for h in sorted(horizon_data.keys()):
            data = horizon_data[h]
            sorted_idx = np.argsort(data['num_steps'])
            num_steps_sorted = [data['num_steps'][i] for i in sorted_idx]
            times_sorted = [data['times'][i] for i in sorted_idx]
            errors_sorted = [data['errors'][i] for i in sorted_idx]
            
            # Add concrete time as reference point at x=h
            if h in self.results['concrete_sequential']:
                num_steps_sorted.append(h)
                times_sorted.append(self.results['concrete_sequential'][h]['time_mean'])
                errors_sorted.append(self.results['concrete_sequential'][h]['time_std'])
            
            ax.errorbar(num_steps_sorted, times_sorted, yerr=errors_sorted, fmt='o-', 
                        linewidth=2, markersize=7, capsize=4, capthick=1.5,
                        label=f'Horizon {h}', alpha=0.8)
        
        ax.set_xlabel('Number of Symbolic Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Multi-Step Symbolic Strategy Comparison ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/multistep_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ multistep_comparison.png")
    
    def _plot_speedup_heatmap(self, output_dir):
        """Create heatmap of speedup for multi-step symbolic (using mean values)"""
        # Collect data
        horizons = set()
        num_steps_set = set()
        for (h, n) in self.results['symbolic_multi'].keys():
            horizons.add(h)
            num_steps_set.add(n)
        
        horizons = sorted(horizons)
        num_steps_list = sorted(num_steps_set)
        
        # Create matrix
        speedup_matrix = np.zeros((len(num_steps_list), len(horizons)))
        speedup_matrix[:] = np.nan
        
        for i, n in enumerate(num_steps_list):
            for j, h in enumerate(horizons):
                if (h, n) in self.results['symbolic_multi'] and h in self.results['concrete_sequential']:
                    concrete_mean = self.results['concrete_sequential'][h]['time_mean']
                    symbolic_mean = self.results['symbolic_multi'][(h, n)]['time_mean']
                    speedup = concrete_mean / symbolic_mean
                    speedup_matrix[i, j] = speedup
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', 
                       vmin=0, vmax=max(2, np.nanmax(speedup_matrix)))
        
        # Set ticks
        ax.set_xticks(np.arange(len(horizons)))
        ax.set_yticks(np.arange(len(num_steps_list)))
        ax.set_xticklabels(horizons)
        ax.set_yticklabels(num_steps_list)
        
        # Labels
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Symbolic Steps', fontsize=12, fontweight='bold')
        ax.set_title(f'Speedup Heatmap: Concrete vs Symbolic Multi-Step ({self.system_type}, n={self.num_trials} trials)', 
                     fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup Factor', fontsize=11, fontweight='bold')
        
        # Annotate cells with speedup values
        for i in range(len(num_steps_list)):
            for j in range(len(horizons)):
                if not np.isnan(speedup_matrix[i, j]):
                    text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ speedup_heatmap.png")
    
    def _plot_symbolic_vs_concrete_direct(self, output_dir):
        """
        Plot direct comparison: single symbolic calculation time vs total concrete time
        for the same horizon. This shows the trend/relationship between symbolic and concrete
        computation times.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Collect data for horizons where we have both concrete and symbolic single-step
        horizons = []
        concrete_times = []
        concrete_errors = []
        symbolic_times = []
        symbolic_errors = []
        
        for h in sorted(self.results['concrete_sequential'].keys()):
            if h in self.results['symbolic_single']:
                horizons.append(h)
                concrete_times.append(self.results['concrete_sequential'][h]['time_mean'])
                concrete_errors.append(self.results['concrete_sequential'][h]['time_std'])
                symbolic_times.append(self.results['symbolic_single'][h]['time_mean'])
                symbolic_errors.append(self.results['symbolic_single'][h]['time_std'])
        
        if not horizons:
            print(f"  ⚠ Skipping symbolic_vs_concrete_direct.png (no overlapping data)")
            return
        
        # --- LEFT PLOT: Direct Time Comparison ---
        ax1.errorbar(horizons, concrete_times, yerr=concrete_errors, 
                    fmt='o-', linewidth=2.5, markersize=9, capsize=5, capthick=2,
                    label='Concrete Sequential (sum of all steps)', 
                    color='#2E86AB', alpha=0.85)
        
        ax1.errorbar(horizons, symbolic_times, yerr=symbolic_errors,
                    fmt='s-', linewidth=2.5, markersize=9, capsize=5, capthick=2,
                    label='Symbolic Single-Step (one calculation)', 
                    color='#A23B72', alpha=0.85)
        
        ax1.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Time Comparison: Sequential Concrete vs Single Symbolic', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add annotations showing speedup at each point
        for i, h in enumerate(horizons):
            speedup = concrete_times[i] / symbolic_times[i]
            # Place annotation slightly above the symbolic point
            y_pos = max(symbolic_times[i], concrete_times[i]) * 1.05
            ax1.annotate(f'{speedup:.2f}x', 
                        xy=(h, y_pos), 
                        fontsize=8, 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # --- RIGHT PLOT: Scatter with Trend Line ---
        # This shows the relationship: how does symbolic time scale with concrete time?
        
        ax2.errorbar(concrete_times, symbolic_times, 
                    xerr=concrete_errors, yerr=symbolic_errors,
                    fmt='o', markersize=10, capsize=5, capthick=2,
                    color='#F18F01', alpha=0.7, ecolor='gray')
        
        # Add labels for each point (horizon value)
        for i, h in enumerate(horizons):
            ax2.annotate(f'h={h}', 
                        xy=(concrete_times[i], symbolic_times[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        # Fit linear trend line (y = mx + b)
        if len(concrete_times) > 1:
            # Weighted least squares using inverse variance as weights
            weights = 1.0 / (np.array(concrete_errors)**2 + np.array(symbolic_errors)**2 + 1e-10)
            coeffs = np.polyfit(concrete_times, symbolic_times, 1, w=weights)
            poly_fn = np.poly1d(coeffs)
            
            # Generate smooth line for plotting
            x_trend = np.linspace(min(concrete_times) * 0.9, max(concrete_times) * 1.1, 100)
            y_trend = poly_fn(x_trend)
            
            ax2.plot(x_trend, y_trend, '--', linewidth=2, color='red', alpha=0.6,
                    label=f'Linear fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.4f}')
        
        # Add y=x reference line (where symbolic = concrete)
        max_val = max(max(concrete_times), max(symbolic_times)) * 1.1
        ax2.plot([0, max_val], [0, max_val], ':', linewidth=1.5, color='black', 
                alpha=0.4, label='y=x (equal time)')
        
        ax2.set_xlabel('Concrete Sequential Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Symbolic Single-Step Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Correlation: Symbolic vs Concrete Computation Time', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Force equal aspect ratio for better comparison
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/symbolic_vs_concrete_direct.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ symbolic_vs_concrete_direct.png")
    
    def save_results_table(self, output_dir='./timing_analysis'):
        """Save results as CSV tables with statistics"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Table 1: Concrete vs Single Symbolic
        data = []
        for h in sorted(set(list(self.results['concrete_sequential'].keys()) + 
                           list(self.results['symbolic_single'].keys()))):
            row = {'Horizon': h}
            
            if h in self.results['concrete_sequential']:
                stats = self.results['concrete_sequential'][h]
                row['Concrete_Time_Mean'] = stats['time_mean']
                row['Concrete_Time_Std'] = stats['time_std']
                row['Concrete_Volume_Mean'] = stats['volume_mean']
                row['Concrete_Volume_Std'] = stats['volume_std']
            
            if h in self.results['symbolic_single']:
                stats = self.results['symbolic_single'][h]
                row['Symbolic_Time_Mean'] = stats['time_mean']
                row['Symbolic_Time_Std'] = stats['time_std']
                row['Symbolic_Volume_Mean'] = stats['volume_mean']
                row['Symbolic_Volume_Std'] = stats['volume_std']
                
                if h in self.results['concrete_sequential']:
                    concrete_mean = self.results['concrete_sequential'][h]['time_mean']
                    symbolic_mean = stats['time_mean']
                    row['Speedup'] = concrete_mean / symbolic_mean
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/concrete_vs_symbolic.csv', index=False)
        print(f"\n  ✓ concrete_vs_symbolic.csv")
        
        # Table 2: Multi-step symbolic
        data_multi = []
        for (h, n), stats in sorted(self.results['symbolic_multi'].items()):
            row = {
                'Horizon': h,
                'Num_Steps': n,
                'Time_Mean': stats['time_mean'],
                'Time_Std': stats['time_std'],
                'Volume_Mean': stats['volume_mean'],
                'Volume_Std': stats['volume_std']
            }
            
            if h in self.results['concrete_sequential']:
                concrete_mean = self.results['concrete_sequential'][h]['time_mean']
                row['Speedup_vs_Concrete'] = concrete_mean / stats['time_mean']
            
            if h in self.results['symbolic_single']:
                single_mean = self.results['symbolic_single'][h]['time_mean']
                row['Slowdown_vs_SingleSymbolic'] = stats['time_mean'] / single_mean
            
            data_multi.append(row)
        
        df_multi = pd.DataFrame(data_multi)
        df_multi.to_csv(f'{output_dir}/symbolic_multistep.csv', index=False)
        print(f"  ✓ symbolic_multistep.csv")
        
        # Table 3: Detailed trials data (all individual trial results)
        print(f"\n  Saving detailed trial data...")
        
        # Concrete trials
        concrete_trials = []
        for h, stats in self.results['concrete_sequential'].items():
            for trial_idx, (time_val, vol_val) in enumerate(zip(stats['time_trials'], stats['volume_trials'])):
                concrete_trials.append({
                    'Horizon': h,
                    'Method': 'Concrete',
                    'Trial': trial_idx,
                    'Time': time_val,
                    'Volume': vol_val
                })
        
        # Symbolic single trials
        symbolic_trials = []
        for h, stats in self.results['symbolic_single'].items():
            for trial_idx, (time_val, vol_val) in enumerate(zip(stats['time_trials'], stats['volume_trials'])):
                symbolic_trials.append({
                    'Horizon': h,
                    'Method': 'Symbolic_Single',
                    'Trial': trial_idx,
                    'Time': time_val,
                    'Volume': vol_val
                })
        
        # Multi-step trials
        multi_trials = []
        for (h, n), stats in self.results['symbolic_multi'].items():
            for trial_idx, (time_val, vol_val) in enumerate(zip(stats['time_trials'], stats['volume_trials'])):
                multi_trials.append({
                    'Horizon': h,
                    'Num_Steps': n,
                    'Method': f'Symbolic_{n}Step',
                    'Trial': trial_idx,
                    'Time': time_val,
                    'Volume': vol_val
                })
        
        # Combine and save
        all_trials = concrete_trials + symbolic_trials + multi_trials
        df_trials = pd.DataFrame(all_trials)
        df_trials.to_csv(f'{output_dir}/detailed_trials.csv', index=False)
        print(f"  ✓ detailed_trials.csv (n={self.num_trials} trials per test)")


def run_analysis(system_type='DoubleIntegrator', 
                 controller_name='constraint_default_more_data_5hz',
                 horizons=None,
                 symbolic_step_counts=None,
                 num_trials=10,
                 max_diff=10):
    """
    Main entry point for timing analysis
    
    Args:
        system_type: 'DoubleIntegrator' or 'Unicycle_NL'
        controller_name: Controller to use
        horizons: List of horizons to test (default: [2, 3, 4, 5, 6, 8, 10])
        symbolic_step_counts: List of step counts for multi-step symbolic (default: [1, 2, 3, 5])
        num_trials: Number of trials per test with random initial states (default: 10)
        max_diff: Maximum number of symbolic steps allowed in single propagation (default: 10)
    """
    if horizons is None:
        horizons = [2, 3, 4, 5, 6, 8, 10]
    
    if symbolic_step_counts is None:
        symbolic_step_counts = [1, 2, 3, 5]
    
    # Create analyzer
    analyzer = TimingAnalyzer(system_type, controller_name, num_trials=num_trials, max_diff=max_diff)
    
    # Run comparison
    analyzer.run_comparison(horizons, symbolic_step_counts)
    
    # Create plots
    analyzer.create_plots()
    
    # Save tables
    analyzer.save_results_table()
    
    print("\n" + "=" * 80)
    print("TIMING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: ./timing_analysis/")
    print(f"Number of trials per test: {num_trials}")
    print("\nGenerated files:")
    print("  • time_vs_horizon.png")
    print("  • speedup_vs_horizon.png")
    print("  • volume_comparison.png")
    print("  • time_per_step.png")
    print("  • multistep_comparison.png")
    print("  • speedup_heatmap.png")
    print("  • symbolic_vs_concrete_direct.png  [NEW]")
    print("  • concrete_vs_symbolic.csv")
    print("  • symbolic_multistep.csv")
    print("  • detailed_trials.csv")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        system_type = sys.argv[1]
    else:
        system_type = 'DoubleIntegrator'
    
    if len(sys.argv) > 2:
        controller_name = sys.argv[2]
    else:
        controller_name = 'constraint_default_more_data_5hz'
    
    if len(sys.argv) > 3:
        num_trials = int(sys.argv[3])
    else:
        num_trials = 10
    
    if len(sys.argv) > 4:
        max_diff = int(sys.argv[4])
    else:
        max_diff = 10
    
    # Run analysis
    run_analysis(
        system_type=system_type,
        controller_name=controller_name,
        horizons=[2, 3, 4, 5, 6, 8, 10],
        symbolic_step_counts=[1, 2, 3, 5],
        num_trials=num_trials,
        max_diff=max_diff
    )