import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def run_calibration_check():
    results_path = 'data/processed/test_with_uncertainty.csv'
    if not os.path.exists(results_path):
        print("Error: test_with_uncertainty.csv not found. Run uncertainty training first.")
        return

    # Load predictions with uncertainty estimates
    df = pd.read_csv(results_path, index_col='datetime', parse_dates=True)
    y_true = df['target']
    y_pred = df['final_prediction']
    y_std = df['gp_uncertainty_std']
    
    # Calculate standardized residuals
    z_scores = (y_true - y_pred) / y_std
    
    # Check interval coverage
    coverage_1sigma = (np.abs(z_scores) <= 1.0).mean()
    coverage_2sigma = (np.abs(z_scores) <= 1.96).mean()
    
    print("\n" + "="*40)
    print("Calibration Results")
    print("="*40)
    print(f"68% Interval Coverage: {coverage_1sigma*100:.2f}% (Target: 68.2%)")
    print(f"95% Interval Coverage: {coverage_2sigma*100:.2f}% (Target: 95.0%)")
    
    # Negative log likelihood
    nll = -norm.logpdf(y_true, loc=y_pred, scale=y_std).mean()
    print(f"Negative Log Likelihood: {nll:.4f}")
    
    # Reliability diagram
    expected_p = np.linspace(0.05, 0.95, 20)
    observed_p = []
    
    for p in expected_p:
        multiplier = norm.ppf((1 + p) / 2)
        coverage = (np.abs(z_scores) <= multiplier).mean()
        observed_p.append(coverage)
        
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(expected_p, observed_p, 'o-', color='orange', label='GP Model')
    plt.fill_between(expected_p, expected_p, observed_p, color='orange', alpha=0.1)
    plt.xlabel("Target Confidence Level")
    plt.ylabel("Observed Coverage")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/05_calibration_reliability.png')
    print("\nPlot saved to figures/05_calibration_reliability.png")
    print("="*40)

if __name__ == "__main__":
    run_calibration_check()