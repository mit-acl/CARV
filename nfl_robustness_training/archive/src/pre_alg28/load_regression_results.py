"""
Utility to load regression analysis results from timing_analysis runs.

Usage:
    from load_regression_results import load_regression_results
    
    # Load results from a specific run
    results = load_regression_results("run_12_di")
    
    # Access coefficients
    concrete_slope = results['concrete']['slope']
    concrete_intercept = results['concrete']['intercept']
    
    # Predict concrete time for horizon 20
    predicted_time = results['concrete']['slope'] * 20 + results['concrete']['intercept']
    
    # Get time ratio prediction
    ratio = results['ratio']['slope'] * 20 + results['ratio']['intercept']
"""

import os
import pandas as pd
import numpy as np


def load_regression_results(run_dir, base_dir='./timing_analysis', use_weighted=True):
    """
    Load regression analysis results from a timing analysis run.
    
    Args:
        run_dir: Name of the run directory (e.g., "run_12_di")
        base_dir: Base directory containing all runs (default: './timing_analysis')
        use_weighted: If True, return weighted regression coefficients; 
                      if False, return unweighted (default: True)
    
    Returns:
        Dictionary with structure:
        {
            'concrete': {
                'slope': float,
                'intercept': float,
                'r_squared': float,
                'p_value': float or None (None for weighted),
                'formula': str,
                'predict': function(horizon) -> time
            },
            'ratio': {
                'slope': float,
                'intercept': float,
                'r_squared': float,
                'p_value': float or None,
                'formula': str,
                'predict': function(horizon) -> ratio
            },
            'symbolic': {
                'predict': function(horizon) -> time  (combines concrete and ratio)
            },
            'metadata': {
                'run_dir': str,
                'weighted': bool,
                'csv_path': str,
                'py_path': str
            }
        }
    
    Example:
        >>> results = load_regression_results("run_1_di")
        >>> horizon = 15
        >>> concrete_time = results['concrete']['predict'](horizon)
        >>> symbolic_time = results['symbolic']['predict'](horizon)
        >>> ratio = results['ratio']['predict'](horizon)
    """
    
    # Construct full path to run directory
    full_path = os.path.join(base_dir, run_dir)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Run directory not found: {full_path}")
    
    # Load CSV file
    csv_path = os.path.join(full_path, 'regression_fits.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Regression fits CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Determine which rows to use based on weighted flag
    if use_weighted:
        concrete_row = df[df['Regression'] == 'Concrete_Time_vs_Horizon_Weighted'].iloc[0]
        ratio_row = df[df['Regression'] == 'TimeRatio_vs_Horizon_Weighted'].iloc[0]
        weight_type = 'weighted'
    else:
        concrete_row = df[df['Regression'] == 'Concrete_Time_vs_Horizon_Unweighted'].iloc[0]
        ratio_row = df[df['Regression'] == 'TimeRatio_vs_Horizon_Unweighted'].iloc[0]
        weight_type = 'unweighted'
    
    # Extract concrete regression parameters
    concrete_slope = concrete_row['Slope']
    concrete_intercept = concrete_row['Intercept']
    concrete_r_squared = concrete_row['R_Squared']
    concrete_p_value = concrete_row['P_Value'] if not pd.isna(concrete_row['P_Value']) else None
    concrete_formula = concrete_row['Formula']
    
    # Extract ratio regression parameters
    ratio_slope = ratio_row['Slope']
    ratio_intercept = ratio_row['Intercept']
    ratio_r_squared = ratio_row['R_Squared']
    ratio_p_value = ratio_row['P_Value'] if not pd.isna(ratio_row['P_Value']) else None
    ratio_formula = ratio_row['Formula']
    
    # Check if ratio parameters are valid
    ratio_valid = not (pd.isna(ratio_slope) or pd.isna(ratio_intercept))
    
    # Create prediction functions
    def predict_concrete(horizon):
        """Predict concrete sequential time for given horizon."""
        return concrete_slope * horizon + concrete_intercept
    
    def predict_ratio(horizon):
        """Predict symbolic/concrete time ratio for given horizon."""
        if not ratio_valid:
            return None
        return ratio_slope * horizon + ratio_intercept
    
    def predict_symbolic(horizon):
        """Predict symbolic single-step time for given horizon."""
        if not ratio_valid:
            return None
        ratio = predict_ratio(horizon)
        concrete_time = predict_concrete(horizon)
        return ratio * concrete_time
    
    # Build result dictionary
    results = {
        'concrete': {
            'slope': concrete_slope,
            'intercept': concrete_intercept,
            'r_squared': concrete_r_squared,
            'p_value': concrete_p_value,
            'formula': concrete_formula,
            'predict': predict_concrete
        },
        'ratio': {
            'slope': ratio_slope if ratio_valid else None,
            'intercept': ratio_intercept if ratio_valid else None,
            'r_squared': ratio_r_squared if ratio_valid else None,
            'p_value': ratio_p_value if ratio_valid else None,
            'formula': ratio_formula if ratio_valid else None,
            'predict': predict_ratio
        },
        'symbolic': {
            'predict': predict_symbolic
        },
        'metadata': {
            'run_dir': run_dir,
            'weighted': use_weighted,
            'weight_type': weight_type,
            'csv_path': csv_path,
            'py_path': os.path.join(full_path, 'regression_coefficients.py'),
            'ratio_valid': ratio_valid
        }
    }
    
    return results


def load_all_coefficients(run_dir, base_dir='./timing_analysis'):
    """
    Load ALL regression coefficients (both weighted and unweighted).
    
    Args:
        run_dir: Name of the run directory (e.g., "run_12_di")
        base_dir: Base directory containing all runs (default: './timing_analysis')
    
    Returns:
        Dictionary with structure:
        {
            'unweighted': {
                'concrete': {...},
                'ratio': {...}
            },
            'weighted': {
                'concrete': {...},
                'ratio': {...}
            },
            'metadata': {...}
        }
    """
    
    # Construct full path to run directory
    full_path = os.path.join(base_dir, run_dir)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Run directory not found: {full_path}")
    
    # Load CSV file
    csv_path = os.path.join(full_path, 'regression_fits.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Regression fits CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Extract all four regression types
    concrete_unweighted = df[df['Regression'] == 'Concrete_Time_vs_Horizon_Unweighted'].iloc[0]
    concrete_weighted = df[df['Regression'] == 'Concrete_Time_vs_Horizon_Weighted'].iloc[0]
    ratio_unweighted = df[df['Regression'] == 'TimeRatio_vs_Horizon_Unweighted'].iloc[0]
    ratio_weighted = df[df['Regression'] == 'TimeRatio_vs_Horizon_Weighted'].iloc[0]
    
    def make_predictor(slope, intercept):
        """Create a prediction function."""
        if pd.isna(slope) or pd.isna(intercept):
            return lambda h: None
        return lambda h: slope * h + intercept
    
    results = {
        'unweighted': {
            'concrete': {
                'slope': concrete_unweighted['Slope'],
                'intercept': concrete_unweighted['Intercept'],
                'r_squared': concrete_unweighted['R_Squared'],
                'p_value': concrete_unweighted['P_Value'] if not pd.isna(concrete_unweighted['P_Value']) else None,
                'formula': concrete_unweighted['Formula'],
                'predict': make_predictor(concrete_unweighted['Slope'], concrete_unweighted['Intercept'])
            },
            'ratio': {
                'slope': ratio_unweighted['Slope'] if not pd.isna(ratio_unweighted['Slope']) else None,
                'intercept': ratio_unweighted['Intercept'] if not pd.isna(ratio_unweighted['Intercept']) else None,
                'r_squared': ratio_unweighted['R_Squared'] if not pd.isna(ratio_unweighted['R_Squared']) else None,
                'p_value': ratio_unweighted['P_Value'] if not pd.isna(ratio_unweighted['P_Value']) else None,
                'formula': ratio_unweighted['Formula'],
                'predict': make_predictor(ratio_unweighted['Slope'], ratio_unweighted['Intercept'])
            }
        },
        'weighted': {
            'concrete': {
                'slope': concrete_weighted['Slope'],
                'intercept': concrete_weighted['Intercept'],
                'r_squared': concrete_weighted['R_Squared'],
                'p_value': None,  # p-value not computed for weighted
                'formula': concrete_weighted['Formula'],
                'predict': make_predictor(concrete_weighted['Slope'], concrete_weighted['Intercept'])
            },
            'ratio': {
                'slope': ratio_weighted['Slope'] if not pd.isna(ratio_weighted['Slope']) else None,
                'intercept': ratio_weighted['Intercept'] if not pd.isna(ratio_weighted['Intercept']) else None,
                'r_squared': ratio_weighted['R_Squared'] if not pd.isna(ratio_weighted['R_Squared']) else None,
                'p_value': None,  # p-value not computed for weighted
                'formula': ratio_weighted['Formula'],
                'predict': make_predictor(ratio_weighted['Slope'], ratio_weighted['Intercept'])
            }
        },
        'metadata': {
            'run_dir': run_dir,
            'csv_path': csv_path,
            'py_path': os.path.join(full_path, 'regression_coefficients.py')
        }
    }
    
    return results


def print_regression_summary(results):
    """
    Pretty-print regression results.
    
    Args:
        results: Dictionary returned by load_regression_results() or load_all_coefficients()
    """
    print("=" * 70)
    print("REGRESSION ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Check if this is from load_regression_results or load_all_coefficients
    if 'unweighted' in results:
        # Full results from load_all_coefficients
        print(f"Run: {results['metadata']['run_dir']}")
        print()
        
        for weight_type in ['unweighted', 'weighted']:
            print(f"\n{weight_type.upper()} REGRESSION:")
            print("-" * 70)
            
            concrete = results[weight_type]['concrete']
            print(f"\nConcrete Time vs Horizon:")
            print(f"  Formula:   {concrete['formula']}")
            print(f"  R²:        {concrete['r_squared']:.6f}")
            if concrete['p_value'] is not None:
                print(f"  p-value:   {concrete['p_value']:.6e}")
            
            ratio = results[weight_type]['ratio']
            print(f"\nTime Ratio vs Horizon:")
            if ratio['slope'] is not None:
                print(f"  Formula:   {ratio['formula']}")
                print(f"  R²:        {ratio['r_squared']:.6f}")
                if ratio['p_value'] is not None:
                    print(f"  p-value:   {ratio['p_value']:.6e}")
            else:
                print(f"  No data available")
    
    else:
        # Results from load_regression_results
        print(f"Run: {results['metadata']['run_dir']}")
        print(f"Type: {results['metadata']['weight_type']}")
        print()
        
        concrete = results['concrete']
        print(f"Concrete Time vs Horizon:")
        print(f"  Formula:   {concrete['formula']}")
        print(f"  R²:        {concrete['r_squared']:.6f}")
        if concrete['p_value'] is not None:
            print(f"  p-value:   {concrete['p_value']:.6e}")
        
        ratio = results['ratio']
        print(f"\nTime Ratio vs Horizon:")
        if ratio['slope'] is not None:
            print(f"  Formula:   {ratio['formula']}")
            print(f"  R²:        {ratio['r_squared']:.6f}")
            if ratio['p_value'] is not None:
                print(f"  p-value:   {ratio['p_value']:.6e}")
        else:
            print(f"  No data available")
    
    print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_regression_results.py <run_dir>")
        print("Example: python load_regression_results.py run_1_di")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    # Load and display results
    print("\n--- Loading weighted regression (recommended) ---")
    results = load_regression_results(run_dir, use_weighted=True)
    print_regression_summary(results)
    
    # Example predictions
    test_horizon = 15
    print(f"\nExample predictions for horizon = {test_horizon}:")
    print(f"  Concrete time:  {results['concrete']['predict'](test_horizon):.6f} seconds")
    if results['symbolic']['predict'](test_horizon) is not None:
        print(f"  Symbolic time:  {results['symbolic']['predict'](test_horizon):.6f} seconds")
        print(f"  Time ratio:     {results['ratio']['predict'](test_horizon):.6f}")
    
    print("\n--- Full coefficients (all types) ---")
    all_coeffs = load_all_coefficients(run_dir)
    print(f"\nWeighted concrete slope: {all_coeffs['weighted']['concrete']['slope']:.10f}")
    print(f"Unweighted concrete slope: {all_coeffs['unweighted']['concrete']['slope']:.10f}")