"""
Volume-Dependent Timing Analysis

Investigates how symbolic propagation timing depends on volume by:
1. Fitting separate equations for different volume ranges
2. Checking if the k² coefficient varies with volume
3. Testing threshold effects
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'nfl_robustness_training/src'))

from clean_integrated_sim import setup_analyzer, ReachabilityTester, ReachableSetHorizon, CalculationType
from utils.robust_training_utils import ReachableSet


@dataclass
class TimingDataPoint:
    system_type: str
    parent_volume: float
    num_steps: int
    start_timestep: int
    compute_time: float


def collect_timing_at_fixed_volumes(system_type: str, controller_name: str,
                                      volume_targets: List[float],
                                      step_sizes: List[int] = None,
                                      num_repeats: int = 3) -> List[TimingDataPoint]:
    """
    Collect timing data at specific volume levels by scaling initial bounds.
    """
    if step_sizes is None:
        step_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    data_points = []

    print(f"\nCollecting timing at fixed volumes for {system_type}")
    print(f"  Target volumes: {volume_targets}")
    print(f"  Step sizes: {step_sizes}")
    print(f"  Repeats per config: {num_repeats}")

    for target_vol in volume_targets:
        print(f"\n  Target volume: {target_vol}")

        for repeat in range(num_repeats):
            # Create analyzer
            analyzer = setup_analyzer(system_type, controller_name)

            # Get base bounds and compute scale factor
            base_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
            base_volume = np.prod(base_bounds[:, 1] - base_bounds[:, 0])

            # Scale to achieve target volume
            # volume scales as scale^n_dims
            n_dims = base_bounds.shape[0]
            scale = (target_vol / base_volume) ** (1.0 / n_dims)

            center = (base_bounds[:, 0] + base_bounds[:, 1]) / 2
            half_width = (base_bounds[:, 1] - base_bounds[:, 0]) / 2 * scale
            scaled_bounds = np.stack([center - half_width, center + half_width], axis=1)

            # Set scaled bounds
            analyzer.reachable_sets[0].set_range(
                torch.tensor(scaled_bounds, dtype=torch.float32, device=analyzer.device)
            )

            # Verify actual volume
            actual_bounds = analyzer.reachable_sets[0].full_set.cpu().numpy()
            actual_volume = np.prod(actual_bounds[:, 1] - actual_bounds[:, 0])

            current_reachset = analyzer.reachable_sets[0]

            # Measure timing for each k
            for k in step_sizes:
                bounded_sys_k = analyzer.bounded_cl_systems.get(k - 1)
                if bounded_sys_k is None:
                    continue

                output_reachset = ReachableSet(t=k, device=analyzer.device)
                output_reachset.recalculate = True

                t_start = time.perf_counter()
                current_reachset.populate_next_reachable_set(
                    bounded_sys_k,
                    output_reachset,
                    training=False
                )
                elapsed = time.perf_counter() - t_start

                data_points.append(TimingDataPoint(
                    system_type=system_type,
                    parent_volume=actual_volume,
                    num_steps=k,
                    start_timestep=0,
                    compute_time=elapsed
                ))

            if repeat == 0:
                print(f"    Actual volume: {actual_volume:.6f}")

    print(f"\nCollected {len(data_points)} data points")
    return data_points


def fit_by_volume_bins(data_points: List[TimingDataPoint], n_bins: int = 4):
    """
    Fit separate quadratic models for different volume ranges.
    """
    volumes = np.array([dp.parent_volume for dp in data_points])
    steps = np.array([dp.num_steps for dp in data_points])
    times = np.array([dp.compute_time for dp in data_points])

    # Create volume bins
    vol_min, vol_max = volumes.min(), volumes.max()
    bin_edges = np.logspace(np.log10(vol_min), np.log10(vol_max), n_bins + 1)

    print(f"\nFitting by volume bins:")
    print(f"  Volume range: {vol_min:.6f} to {vol_max:.6f}")
    print(f"  Bin edges: {bin_edges}")

    results = []

    for i in range(n_bins):
        mask = (volumes >= bin_edges[i]) & (volumes < bin_edges[i+1])
        if mask.sum() < 10:
            print(f"  Bin {i}: Not enough data ({mask.sum()} points)")
            continue

        bin_steps = steps[mask]
        bin_times = times[mask]
        bin_vols = volumes[mask]

        # Fit: time = a*k² + b*k + c
        X = np.column_stack([bin_steps**2, bin_steps, np.ones_like(bin_steps)])
        model = LinearRegression(fit_intercept=False)
        model.fit(X, bin_times)

        a, b, c = model.coef_
        y_pred = model.predict(X)
        r2 = r2_score(bin_times, y_pred)

        avg_vol = bin_vols.mean()

        print(f"\n  Bin {i}: volume {bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}")
        print(f"    Avg volume: {avg_vol:.6f}")
        print(f"    Points: {mask.sum()}")
        print(f"    Fit: time = {a:.6f}k² + {b:.6f}k + {c:.6f}")
        print(f"    R²: {r2:.4f}")

        results.append({
            'bin': i,
            'vol_low': bin_edges[i],
            'vol_high': bin_edges[i+1],
            'avg_vol': avg_vol,
            'a_k2': a,
            'b_k': b,
            'c': c,
            'r2': r2,
            'n_points': mask.sum()
        })

    return results


def fit_volume_dependent_coefficient(data_points: List[TimingDataPoint]):
    """
    Fit model: time = a*k² + b*k + c
    where each coefficient can depend on volume:
        a = a0 + a1*volume
        b = b0 + b1*volume
        c = c0 + c1*volume

    Full model: time = (a0 + a1*vol)*k² + (b0 + b1*vol)*k + (c0 + c1*vol)
    """
    volumes = np.array([dp.parent_volume for dp in data_points])
    steps = np.array([dp.num_steps for dp in data_points])
    times = np.array([dp.compute_time for dp in data_points])

    # Full model with volume interactions
    # time = a0*k² + a1*vol*k² + b0*k + b1*vol*k + c0 + c1*vol
    X_full = np.column_stack([
        steps**2,           # a0 * k²
        volumes * steps**2, # a1 * vol * k²
        steps,              # b0 * k
        volumes * steps,    # b1 * vol * k
        np.ones_like(steps),# c0
        volumes             # c1 * vol
    ])

    model_full = LinearRegression(fit_intercept=False)
    model_full.fit(X_full, times)
    a0, a1, b0, b1, c0, c1 = model_full.coef_

    y_pred_full = model_full.predict(X_full)
    r2_full = r2_score(times, y_pred_full)

    print(f"\nFull volume-dependent model:")
    print(f"  time = (a0 + a1*vol)*k² + (b0 + b1*vol)*k + (c0 + c1*vol)")
    print(f"\n  k² coefficient:  {a0:.6f} + {a1:.6f}*vol")
    print(f"  k coefficient:   {b0:.6f} + {b1:.6f}*vol")
    print(f"  constant:        {c0:.6f} + {c1:.6f}*vol")
    print(f"  R²: {r2_full:.4f}")

    # Also fit simpler model without volume (for comparison)
    X_simple = np.column_stack([steps**2, steps, np.ones_like(steps)])
    model_simple = LinearRegression(fit_intercept=False)
    model_simple.fit(X_simple, times)
    a_simple, b_simple, c_simple = model_simple.coef_

    y_pred_simple = model_simple.predict(X_simple)
    r2_simple = r2_score(times, y_pred_simple)

    print(f"\nSimple model (no volume dependence):")
    print(f"  time = {a_simple:.6f}*k² + {b_simple:.6f}*k + {c_simple:.6f}")
    print(f"  R²: {r2_simple:.4f}")

    print(f"\nR² improvement from volume: {r2_full - r2_simple:.4f}")

    # Interpret which terms matter
    print(f"\n  Interpretation:")
    print(f"    Base k² coefficient: {a0:.6f}")
    if abs(a1) > 1e-6:
        print(f"    k² changes by {a1:.6f} per unit volume")
    if abs(b0) > 0.001:
        print(f"    Linear k term: {b0:.6f} (per-step overhead)")
    if abs(b1) > 1e-6:
        print(f"    k term changes by {b1:.6f} per unit volume")

    return {
        'a0': a0, 'a1': a1,
        'b0': b0, 'b1': b1,
        'c0': c0, 'c1': c1,
        'r2_full': r2_full,
        'a_simple': a_simple, 'b_simple': b_simple, 'c_simple': c_simple,
        'r2_simple': r2_simple
    }


def plot_volume_analysis(data_points: List[TimingDataPoint], bin_results: List[dict],
                         system_type: str, save_path: str = None):
    """
    Visualize volume-dependent timing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Volume-Dependent Timing Analysis: {system_type}', fontsize=14)

    volumes = np.array([dp.parent_volume for dp in data_points])
    steps = np.array([dp.num_steps for dp in data_points])
    times = np.array([dp.compute_time for dp in data_points])

    # Plot 1: k² coefficient vs volume bin
    ax1 = axes[0, 0]
    if bin_results:
        avg_vols = [r['avg_vol'] for r in bin_results]
        k2_coefs = [r['a_k2'] for r in bin_results]
        ax1.plot(avg_vols, k2_coefs, 'bo-', markersize=10, linewidth=2)
        ax1.set_xlabel('Average Volume in Bin')
        ax1.set_ylabel('k² Coefficient')
        ax1.set_title('How k² Coefficient Varies with Volume')
        ax1.grid(True, alpha=0.3)

        # Add trend line
        if len(avg_vols) > 1:
            z = np.polyfit(avg_vols, k2_coefs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(avg_vols), max(avg_vols), 100)
            ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend: {z[0]:.4f}*vol + {z[1]:.4f}')
            ax1.legend()

    # Plot 2: Time vs k for different volume ranges
    ax2 = axes[0, 1]
    unique_vols = sorted(set(volumes))
    n_vol_groups = min(5, len(unique_vols))
    vol_groups = np.array_split(sorted(unique_vols), n_vol_groups)
    colors = plt.cm.viridis(np.linspace(0, 1, n_vol_groups))

    for i, (vol_group, color) in enumerate(zip(vol_groups, colors)):
        vol_min, vol_max = min(vol_group), max(vol_group)
        mask = (volumes >= vol_min) & (volumes <= vol_max)

        group_steps = steps[mask]
        group_times = times[mask]

        # Average by k
        unique_k = sorted(set(group_steps))
        avg_times = [group_times[group_steps == k].mean() for k in unique_k]

        ax2.plot(unique_k, avg_times, 'o-', color=color,
                 label=f'vol: {vol_min:.3f}-{vol_max:.3f}', markersize=6)

    ax2.set_xlabel('Symbolic Steps (k)')
    ax2.set_ylabel('Compute Time (s)')
    ax2.set_title('Time vs k (by Volume Range)')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time/k² vs volume (should be flat if purely quadratic)
    ax3 = axes[1, 0]
    time_per_k2 = times / (steps ** 2)
    scatter = ax3.scatter(volumes, time_per_k2, c=steps, cmap='viridis', alpha=0.6, s=30)
    ax3.set_xlabel('Parent Volume')
    ax3.set_ylabel('Time / k² (s)')
    ax3.set_title('Normalized Time vs Volume')
    plt.colorbar(scatter, ax=ax3, label='k')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals from pure k² fit vs volume
    ax4 = axes[1, 1]
    # Fit simple k² model
    X_simple = steps.reshape(-1, 1) ** 2
    model_simple = LinearRegression()
    model_simple.fit(X_simple, times)
    pred_simple = model_simple.predict(X_simple)
    residuals = times - pred_simple

    scatter2 = ax4.scatter(volumes, residuals, c=steps, cmap='viridis', alpha=0.6, s=30)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Parent Volume')
    ax4.set_ylabel('Residual from k² fit (s)')
    ax4.set_title('Residuals vs Volume (pattern = volume matters)')
    plt.colorbar(scatter2, ax=ax4, label='k')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig


def run_volume_analysis(system_type: str, controller_name: str):
    """
    Run complete volume-dependent analysis.
    """
    print("=" * 60)
    print(f"VOLUME-DEPENDENT TIMING ANALYSIS: {system_type}")
    print("=" * 60)

    # Test across wide range of volumes
    if system_type == 'DoubleIntegrator':
        # Base volume is ~0.25, test from 0.01 to 10
        volume_targets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    else:  # Unicycle_NL
        # Base volume is ~0.0007, test from 0.0001 to 0.01
        volume_targets = [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007, 0.01]

    # Collect data
    data_points = collect_timing_at_fixed_volumes(
        system_type, controller_name,
        volume_targets=volume_targets,
        step_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        num_repeats=3
    )

    # Analysis 1: Fit by volume bins
    print("\n" + "=" * 40)
    print("ANALYSIS 1: Separate fits per volume bin")
    print("=" * 40)
    bin_results = fit_by_volume_bins(data_points, n_bins=4)

    # Analysis 2: Volume-dependent k² coefficient
    print("\n" + "=" * 40)
    print("ANALYSIS 2: Volume-dependent k² model")
    print("=" * 40)
    coef_results = fit_volume_dependent_coefficient(data_points)

    # Analysis 3: Full quadratic fit for comparison (no volume)
    print("\n" + "=" * 40)
    print("ANALYSIS 3: Simple quadratic fit (ignoring volume)")
    print("=" * 40)

    steps = np.array([dp.num_steps for dp in data_points])
    times = np.array([dp.compute_time for dp in data_points])

    X = np.column_stack([steps**2, steps, np.ones_like(steps)])
    model = LinearRegression(fit_intercept=False)
    model.fit(X, times)
    a, b, c = model.coef_
    r2 = r2_score(times, model.predict(X))

    print(f"  time = {a:.6f}*k² + {b:.6f}*k + {c:.6f}")
    print(f"  R²: {r2:.4f}")

    # Plot
    plot_volume_analysis(data_points, bin_results, system_type,
                         save_path=f'volume_analysis_{system_type}.png')

    # Save results
    results = {
        'system_type': system_type,
        'data_points': data_points,
        'bin_results': bin_results,
        'coef_results': coef_results,
        'simple_fit': {'a': a, 'b': b, 'c': c, 'r2': r2}
    }

    with open(f'volume_analysis_results_{system_type}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Volume-dependent timing analysis')
    parser.add_argument('--system', type=str, default='DoubleIntegrator',
                        choices=['DoubleIntegrator', 'Unicycle_NL', 'both'])

    args = parser.parse_args()

    if args.system == 'both':
        run_volume_analysis('DoubleIntegrator', 'constraint_default_more_data_5hz')
        run_volume_analysis('Unicycle_NL', 'natural_none_default')
    else:
        if args.system == 'DoubleIntegrator':
            controller = 'constraint_default_more_data_5hz'
        else:
            controller = 'natural_none_default'
        run_volume_analysis(args.system, controller)
