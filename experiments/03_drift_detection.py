import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mean_model import EnergyMeanModel
from models.residual_gp import ResidualGPWrapper

def run_drift_test():
    print("--- Drift Detection Test ---")
    
    # Load data and models
    df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
    base_data = df.loc['2010-02-01':'2010-02-14'].copy()
    
    mean_model = EnergyMeanModel()
    if os.path.exists('models/lgbm_mean_model.pkl'):
        mean_model.load('models/lgbm_mean_model.pkl')
    else:
        print("Mean model not found. Run baseline training first.")
        return

    gp_system = ResidualGPWrapper()
    if os.path.exists('models/residual_gp_model.pth'):
        gp_system.load('models/residual_gp_model.pth')
    else:
        print("GP model not found. Run uncertainty training first.")
        return

    # Simulate sensor failure with chaotic input values
    drift_data = base_data.copy()
    drift_start = '2010-02-07 12:00:00'
    mask = drift_data.index >= drift_start
    
    print("Simulating chaotic drift...")
    possible_means = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    random_means = np.random.choice(possible_means, size=mask.sum())
    noise = np.random.normal(loc=random_means, scale=1.5)
    
    drift_data.loc[mask, 'lag_target_1h'] = noise
    drift_data.loc[mask, 'lag_target_2h'] = noise
    
    # Run inference and check uncertainty response
    pred_mean_lgbm = mean_model.predict(drift_data[features])
    gp_mean, gp_std = gp_system.predict(drift_data[features].values)
    final_pred = pred_mean_lgbm + gp_mean

    # Measure uncertainty increase
    unc_before = gp_std[~mask].mean()
    unc_after = gp_std[mask].mean()
    ratio = unc_after / unc_before
    print(f"   Normal uncertainty: {unc_before:.4f} kW")
    print(f"   Drift uncertainty:  {unc_after:.4f} kW")
    print(f"   Increase factor:    {ratio:.2f}x")

    # Visualize drift detection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top panel: predictions with uncertainty
    ax1.plot(drift_data.index, drift_data['target'], color='gray', alpha=0.5, label='Actual')
    ax1.plot(drift_data.index, final_pred, color='red', linestyle='--', label='Prediction')
    
    upper = final_pred + 1.96 * gp_std
    lower = final_pred - 1.96 * gp_std
    ax1.fill_between(drift_data.index, lower, upper, color='orange', alpha=0.4, label='95% Confidence')
    
    ax1.axvline(pd.to_datetime(drift_start), color='blue', linestyle='--', linewidth=2, label='Drift Start')
    ax1.set_title("Chaotic Sensor Failure Detection")
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: uncertainty signal
    ax2.plot(drift_data.index, gp_std, color='purple', linewidth=2)
    ax2.axvline(pd.to_datetime(drift_start), color='blue', linestyle='--', linewidth=2)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax2.text(drift_data.index[10], unc_before + 0.05, f"Normal\n(Avg: {unc_before:.2f})", bbox=props, fontsize=11)
    ax2.text(drift_data.index[-30], unc_after - 0.1, f"Drift Detected\n(Avg: {unc_after:.2f})", 
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=1.0), fontsize=11, fontweight='bold', color='darkred')

    ax2.set_title(f"Uncertainty Signal (Increase: {ratio:.1f}x)")
    ax2.set_ylabel("Predictive Std Dev (kW)")
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/03_drift_detection_chaotic.png')
    print("   Plot saved to figures/03_drift_detection_chaotic.png")

if __name__ == "__main__":
    run_drift_test()