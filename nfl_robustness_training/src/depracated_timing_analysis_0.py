"""
Timing Analysis: Concrete vs Symbolic Reachability Computation
Compares sequential concrete steps to single/multi-step symbolic propagation
"""

from clean_integrated_sim import setup_analyzer, ReachabilityTester, CalculationType
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
    
    def __init__(self, system_type='DoubleIntegrator', controller_name='constraint_default_more_data_5hz'):
        """Initialize analyzer for timing tests"""
        self.system_type = system_type
        self.controller_name = controller_name
        self.analyzer = setup_analyzer(system_type, controller_name)
        
        # Storage for results
        self.results = {
            'concrete_sequential': {},  # horizon -> time
            'symbolic_single': {},      # horizon -> time
            'symbolic_multi': {},       # (horizon, num_steps) -> time
        }
        
        self.volume_results = {
            'concrete_sequential': {},  # horizon -> volume
            'symbolic_single': {},      # horizon -> volume
            'symbolic_multi': {},       # (horizon, num_steps) -> volume
        }
        
    def test_concrete_sequential(self, horizon: int) -> Tuple[float, float]:
        """
        Test sequential concrete propagation over horizon
        
        Args:
            horizon: Number of timesteps to propagate
            
        Returns:
            (total_time, final_volume)
        """
        # Create fresh tester
        tester = ReachabilityTester(self.analyzer)
        
        total_time = 0.0
        
        # Sequential concrete steps
        for t in range(horizon):
            step_time = tester.concrete(t, t + 1)
            total_time += step_time
            
        # Get final volume
        final_horizon = tester.horizons[horizon]
        final_volume = final_horizon.get_tight_volume()
        
        return total_time, final_volume
    
    def test_symbolic_single_step(self, horizon: int) -> Tuple[float, float]:
        """
        Test single symbolic step spanning entire horizon
        
        Args:
            horizon: Horizon length (must be <= max_diff)
            
        Returns:
            (time, final_volume)
        """
        if horizon > self.analyzer.max_diff:
            print(f"Warning: horizon {horizon} exceeds max_diff {self.analyzer.max_diff}")
            return None, None
            
        # Create fresh tester
        tester = ReachabilityTester(self.analyzer)
        
        # Single symbolic step
        step_time = tester.symbolic(0, horizon)
        
        # Get final volume
        final_horizon = tester.horizons[horizon]
        final_volume = final_horizon.get_tight_volume()
        
        return step_time, final_volume
    
    def test_symbolic_multi_step(self, horizon: int, num_steps: int) -> Tuple[float, float]:
        """
        Test multi-step symbolic propagation
        
        Args:
            horizon: Total horizon length
            num_steps: Number of symbolic steps to use
            
        Returns:
            (total_time, final_volume)
        """
        step_size = horizon // num_steps
        if step_size > self.analyzer.max_diff:
            print(f"Warning: step_size {step_size} exceeds max_diff {self.analyzer.max_diff}")
            return None, None
            
        # Create fresh tester
        tester = ReachabilityTester(self.analyzer)
        
        total_time = 0.0
        current_t = 0
        
        # Multi-step symbolic propagation
        for i in range(num_steps):
            next_t = min(current_t + step_size, horizon)
            if next_t <= current_t:
                break
                
            step_time = tester.symbolic(current_t, next_t)
            total_time += step_time
            current_t = next_t
            
        # Handle remainder if horizon not evenly divisible
        if current_t < horizon:
            step_time = tester.symbolic(current_t, horizon)
            total_time += step_time
            
        # Get final volume
        final_horizon = tester.horizons[horizon]
        final_volume = final_horizon.get_tight_volume()
        
        return total_time, final_volume
    
    def run_comparison(self, horizons: List[int], symbolic_step_counts: List[int] = [1, 2, 3, 5]):
        """
        Run full comparison across different horizons and symbolic step counts
        
        Args:
            horizons: List of horizon lengths to test
            symbolic_step_counts: List of symbolic step counts to test
        """
        print("=" * 80)
        print("TIMING ANALYSIS: CONCRETE vs SYMBOLIC REACHABILITY")
        print("=" * 80)
        
        for horizon in horizons:
            print(f"\n{'=' * 40}")
            print(f"Testing Horizon = {horizon}")
            print(f"{'=' * 40}")
            
            # Test concrete sequential
            print(f"  [1/3] Concrete sequential ({horizon} steps)...", end=" ")
            concrete_time, concrete_vol = self.test_concrete_sequential(horizon)
            self.results['concrete_sequential'][horizon] = concrete_time
            self.volume_results['concrete_sequential'][horizon] = concrete_vol
            print(f"{concrete_time:.4f}s, vol={concrete_vol:.6f}")
            
            # Test single symbolic step
            if horizon <= self.analyzer.max_diff:
                print(f"  [2/3] Symbolic single-step (1 step of {horizon})...", end=" ")
                symbolic_time, symbolic_vol = self.test_symbolic_single_step(horizon)
                if symbolic_time is not None:
                    self.results['symbolic_single'][horizon] = symbolic_time
                    self.volume_results['symbolic_single'][horizon] = symbolic_vol
                    print(f"{symbolic_time:.4f}s, vol={symbolic_vol:.6f}")
                    print(f"        Speedup: {concrete_time / symbolic_time:.2f}x")
                else:
                    print("SKIPPED (exceeds max_diff)")
            else:
                print(f"  [2/3] Symbolic single-step SKIPPED (horizon {horizon} > max_diff {self.analyzer.max_diff})")
            
            # Test multi-step symbolic
            print(f"  [3/3] Symbolic multi-step:")
            for num_steps in symbolic_step_counts:
                if num_steps >= horizon:
                    continue
                    
                step_size = horizon // num_steps
                if step_size > self.analyzer.max_diff:
                    print(f"        {num_steps} steps SKIPPED (step_size {step_size} > max_diff)")
                    continue
                    
                print(f"        {num_steps} steps (each ~{step_size})...", end=" ")
                multi_time, multi_vol = self.test_symbolic_multi_step(horizon, num_steps)
                
                if multi_time is not None:
                    self.results['symbolic_multi'][(horizon, num_steps)] = multi_time
                    self.volume_results['symbolic_multi'][(horizon, num_steps)] = multi_vol
                    print(f"{multi_time:.4f}s, vol={multi_vol:.6f}")
                    print(f"              Speedup vs concrete: {concrete_time / multi_time:.2f}x")
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
        
        print(f"All plots saved to {output_dir}/")
    
    def _plot_time_vs_horizon(self, output_dir):
        """Plot computation time vs horizon for all methods"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete sequential
        horizons = sorted(self.results['concrete_sequential'].keys())
        times = [self.results['concrete_sequential'][h] for h in horizons]
        ax.plot(horizons, times, 'o-', linewidth=2, markersize=8, 
                label='Concrete Sequential', color='blue')
        
        # Symbolic single-step
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        times_sym = [self.results['symbolic_single'][h] for h in horizons_sym]
        ax.plot(horizons_sym, times_sym, 's-', linewidth=2, markersize=8,
                label='Symbolic Single-Step', color='green')
        
        # Symbolic multi-step (group by num_steps)
        multi_data = {}
        for (h, n), t in self.results['symbolic_multi'].items():
            if n not in multi_data:
                multi_data[n] = ([], [])
            multi_data[n][0].append(h)
            multi_data[n][1].append(t)
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(multi_data)))
        for i, (num_steps, (hs, ts)) in enumerate(sorted(multi_data.items())):
            sorted_idx = np.argsort(hs)
            hs_sorted = [hs[i] for i in sorted_idx]
            ts_sorted = [ts[i] for i in sorted_idx]
            ax.plot(hs_sorted, ts_sorted, '^--', linewidth=2, markersize=7,
                    label=f'Symbolic {num_steps}-Step', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Computation Time vs Horizon ({self.system_type})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_vs_horizon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ time_vs_horizon.png")
    
    def _plot_speedup_vs_horizon(self, output_dir):
        """Plot speedup (concrete/symbolic) vs horizon"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Single-step speedup
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        speedups = []
        for h in horizons_sym:
            if h in self.results['concrete_sequential']:
                speedup = self.results['concrete_sequential'][h] / self.results['symbolic_single'][h]
                speedups.append(speedup)
            else:
                speedups.append(np.nan)
        
        ax.plot(horizons_sym, speedups, 's-', linewidth=2, markersize=8,
                label='Symbolic Single-Step', color='green')
        
        # Multi-step speedup
        multi_data = {}
        for (h, n), t in self.results['symbolic_multi'].items():
            if h in self.results['concrete_sequential']:
                speedup = self.results['concrete_sequential'][h] / t
                if n not in multi_data:
                    multi_data[n] = ([], [])
                multi_data[n][0].append(h)
                multi_data[n][1].append(speedup)
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(multi_data)))
        for i, (num_steps, (hs, speedups)) in enumerate(sorted(multi_data.items())):
            sorted_idx = np.argsort(hs)
            hs_sorted = [hs[i] for i in sorted_idx]
            speedups_sorted = [speedups[i] for i in sorted_idx]
            ax.plot(hs_sorted, speedups_sorted, '^--', linewidth=2, markersize=7,
                    label=f'Symbolic {num_steps}-Step', color=colors[i], alpha=0.8)
        
        # Reference line at 1x
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='No speedup')
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (Concrete Time / Symbolic Time)', fontsize=12, fontweight='bold')
        ax.set_title(f'Speedup vs Horizon ({self.system_type})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_vs_horizon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ speedup_vs_horizon.png")
    
    def _plot_volume_comparison(self, output_dir):
        """Plot volume comparison across methods"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete
        horizons = sorted(self.volume_results['concrete_sequential'].keys())
        vols = [self.volume_results['concrete_sequential'][h] for h in horizons]
        ax.semilogy(horizons, vols, 'o-', linewidth=2, markersize=8,
                    label='Concrete Sequential', color='blue')
        
        # Symbolic single
        horizons_sym = sorted(self.volume_results['symbolic_single'].keys())
        vols_sym = [self.volume_results['symbolic_single'][h] for h in horizons_sym]
        ax.semilogy(horizons_sym, vols_sym, 's-', linewidth=2, markersize=8,
                    label='Symbolic Single-Step', color='green')
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reachable Set Volume (log scale)', fontsize=12, fontweight='bold')
        ax.set_title(f'Reachable Set Volume vs Horizon ({self.system_type})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/volume_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ volume_comparison.png")
    
    def _plot_time_per_step(self, output_dir):
        """Plot time per step (efficiency)"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Concrete: time per step
        horizons = sorted(self.results['concrete_sequential'].keys())
        time_per_step = [self.results['concrete_sequential'][h] / h for h in horizons]
        ax.plot(horizons, time_per_step, 'o-', linewidth=2, markersize=8,
                label='Concrete Sequential', color='blue')
        
        # Symbolic single: time per step
        horizons_sym = sorted(self.results['symbolic_single'].keys())
        time_per_step_sym = [self.results['symbolic_single'][h] / h for h in horizons_sym]
        ax.plot(horizons_sym, time_per_step_sym, 's-', linewidth=2, markersize=8,
                label='Symbolic Single-Step', color='green')
        
        ax.set_xlabel('Horizon (timesteps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time per Timestep (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Computational Efficiency: Time per Timestep ({self.system_type})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_per_step.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ time_per_step.png")
    
    def _plot_multistep_comparison(self, output_dir):
        """Compare different multi-step symbolic strategies"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by horizon
        horizon_data = {}
        for (h, n), t in self.results['symbolic_multi'].items():
            if h not in horizon_data:
                horizon_data[h] = ([], [])
            horizon_data[h][0].append(n)
            horizon_data[h][1].append(t)
        
        # Plot each horizon as a line
        for h in sorted(horizon_data.keys()):
            num_steps, times = horizon_data[h]
            sorted_idx = np.argsort(num_steps)
            num_steps_sorted = [num_steps[i] for i in sorted_idx]
            times_sorted = [times[i] for i in sorted_idx]
            
            # Add concrete time as reference point at x=h
            if h in self.results['concrete_sequential']:
                num_steps_sorted.append(h)
                times_sorted.append(self.results['concrete_sequential'][h])
            
            ax.plot(num_steps_sorted, times_sorted, 'o-', linewidth=2, markersize=7,
                    label=f'Horizon {h}', alpha=0.8)
        
        ax.set_xlabel('Number of Symbolic Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Multi-Step Symbolic Strategy Comparison ({self.system_type})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/multistep_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ multistep_comparison.png")
    
    def _plot_speedup_heatmap(self, output_dir):
        """Create heatmap of speedup for multi-step symbolic"""
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
                    speedup = self.results['concrete_sequential'][h] / self.results['symbolic_multi'][(h, n)]
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
        ax.set_title(f'Speedup Heatmap: Concrete vs Symbolic Multi-Step ({self.system_type})', 
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
    
    def save_results_table(self, output_dir='./timing_analysis'):
        """Save results as CSV tables"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Table 1: Concrete vs Single Symbolic
        data = []
        for h in sorted(set(list(self.results['concrete_sequential'].keys()) + 
                           list(self.results['symbolic_single'].keys()))):
            row = {'Horizon': h}
            
            if h in self.results['concrete_sequential']:
                row['Concrete_Time'] = self.results['concrete_sequential'][h]
                row['Concrete_Volume'] = self.volume_results['concrete_sequential'][h]
            
            if h in self.results['symbolic_single']:
                row['Symbolic_Time'] = self.results['symbolic_single'][h]
                row['Symbolic_Volume'] = self.volume_results['symbolic_single'][h]
                
                if h in self.results['concrete_sequential']:
                    row['Speedup'] = self.results['concrete_sequential'][h] / self.results['symbolic_single'][h]
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/concrete_vs_symbolic.csv', index=False)
        print(f"\n  ✓ concrete_vs_symbolic.csv")
        
        # Table 2: Multi-step symbolic
        data_multi = []
        for (h, n), t in sorted(self.results['symbolic_multi'].items()):
            row = {
                'Horizon': h,
                'Num_Steps': n,
                'Time': t,
                'Volume': self.volume_results['symbolic_multi'][(h, n)]
            }
            
            if h in self.results['concrete_sequential']:
                row['Speedup_vs_Concrete'] = self.results['concrete_sequential'][h] / t
            
            if h in self.results['symbolic_single']:
                row['Slowdown_vs_SingleSymbolic'] = t / self.results['symbolic_single'][h]
            
            data_multi.append(row)
        
        df_multi = pd.DataFrame(data_multi)
        df_multi.to_csv(f'{output_dir}/symbolic_multistep.csv', index=False)
        print(f"  ✓ symbolic_multistep.csv")


def run_analysis(system_type='DoubleIntegrator', 
                 controller_name='constraint_default_more_data_5hz',
                 horizons=None,
                 symbolic_step_counts=None):
    """
    Main entry point for timing analysis
    
    Args:
        system_type: 'DoubleIntegrator' or 'Unicycle_NL'
        controller_name: Controller to use
        horizons: List of horizons to test (default: [2, 3, 4, 5, 6, 8, 10])
        symbolic_step_counts: List of step counts for multi-step symbolic (default: [1, 2, 3, 5])
    """
    if horizons is None:
        horizons = [2, 3, 4, 5, 6, 8, 10]
    
    if symbolic_step_counts is None:
        symbolic_step_counts = [1, 2, 3, 5]
    
    # Create analyzer
    analyzer = TimingAnalyzer(system_type, controller_name)
    
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
    print("\nGenerated files:")
    print("  • time_vs_horizon.png")
    print("  • speedup_vs_horizon.png")
    print("  • volume_comparison.png")
    print("  • time_per_step.png")
    print("  • multistep_comparison.png")
    print("  • speedup_heatmap.png")
    print("  • concrete_vs_symbolic.csv")
    print("  • symbolic_multistep.csv")


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
    
    # Run analysis
    run_analysis(
        system_type=system_type,
        controller_name=controller_name,
        horizons=[2, 3, 4, 5, 6, 8, 10],
        symbolic_step_counts=[1, 2, 3, 5]
    )