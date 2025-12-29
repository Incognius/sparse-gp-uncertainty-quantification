"""
Visual Pattern Analysis: Compare actual vs predicted across different time periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mean_model import EnergyMeanModel

print("Loading data and model...")
df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)
features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
target = 'target'

train_df = df[df.index < '2010-01-01']
test_df = df[df.index >= '2010-01-01']

# Load model and make predictions
model = EnergyMeanModel()
model.load('models/lgbm_mean_model.pkl')

train_preds = model.predict(train_df[features])
test_preds = model.predict(test_df[features])

# Select 4 consecutive 7-day periods (168 hours each)
# Training: 2 consecutive weeks
# Testing: 2 consecutive weeks

periods = [
    ('Train: Week 1', '2009-06-01', '2009-06-07', train_df, train_preds),
    ('Train: Week 2', '2009-06-08', '2009-06-16', train_df, train_preds),
    ('Test: Week 1', '2010-03-01', '2010-03-07', test_df, test_preds),
    ('Test: Week 2', '2010-03-08', '2010-03-14', test_df, test_preds)
]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

print("\nGenerating 7-day comparison plots...")

for idx, (title, start_date, end_date, data_df, predictions) in enumerate(periods):
    ax = axes[idx]
    
    # Get the data for this period
    mask = (data_df.index >= start_date) & (data_df.index <= end_date)
    period_df = data_df[mask]
    
    if len(period_df) == 0:
        print(f"Warning: No data found for {title}")
        continue
    
    # Get corresponding predictions
    period_actual = period_df[target].values
    if 'Train' in title:
        # For training periods, recalculate predictions for correct indices
        period_preds = model.predict(period_df[features])
    else:
        # For test periods, use the pre-calculated test predictions
        test_start_idx = np.where(test_df.index == period_df.index[0])[0][0]
        test_end_idx = test_start_idx + len(period_df)
        period_preds = test_preds[test_start_idx:test_end_idx]
    
    # Calculate MAE for this period
    mae = np.mean(np.abs(period_actual - period_preds))
    
    # Create time axis (hours from start)
    hours = np.arange(len(period_df))
    
    # Plot actual vs predicted
    ax.plot(hours, period_actual, 'b-', linewidth=1.5, alpha=0.7, label='Actual', marker='o', markersize=2)
    ax.plot(hours, period_preds, 'r--', linewidth=1.5, alpha=0.7, label='Predicted', marker='x', markersize=2)
    
    # Add vertical lines for day boundaries
    for day in range(1, 8):
        ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # Formatting
    ax.set_xlabel('Hours from Start', fontsize=11)
    ax.set_ylabel('Active Power (kW)', fontsize=11)
    ax.set_title(f'{title}\nMAE: {mae:.4f} kW', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add day labels
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xticklabels(['Day 1\n0h', '24h', 'Day 2\n48h', '72h', 'Day 3\n96h', '120h', 'Day 4\n144h', '168h'], 
                       fontsize=9)
    
    # Calculate and display correlation
    corr = np.corrcoef(period_actual, period_preds)[0, 1]
    
    # Add text box with stats
    textstr = f'Correlation: {corr:.4f}\nSamples: {len(period_df)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    print(f"{title}: MAE={mae:.4f}, Corr={corr:.4f}, Samples={len(period_df)}")

plt.suptitle('Visual Pattern Analysis: Actual vs Predicted Across Different Time Periods', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/diagnostic_pattern_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Pattern analysis plot saved to figures/diagnostic_pattern_analysis.png")
print("\nLook for:")
print("  - Daily repeating patterns (should be visible in both actual and predicted)")
print("  - How closely predictions follow actual values")
print("  - Whether model captures peaks and troughs or just follows the lag")
print("  - Any systematic under/over-prediction")
