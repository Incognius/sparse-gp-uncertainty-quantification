"""
LGBM Model Diagnostic: Data Leakage and Overfitting Detection
Comprehensive analysis to identify potential issues in the forecasting pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mean_model import EnergyMeanModel

print("="*80)
print("LGBM MODEL DIAGNOSTIC: DATA LEAKAGE & OVERFITTING DETECTION")
print("="*80)

# Load data
df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)
features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
target = 'target'

train_df = df[df.index < '2010-01-01']
test_df = df[df.index >= '2010-01-01']

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"\nDataset sizes:")
print(f"Train: {len(train_df)} samples")
print(f"Test:  {len(test_df)} samples")

# =============================================================================
# TEST 1: Check for future information in lag features
# =============================================================================
print("\n" + "="*80)
print("TEST 1: Checking lag features for future information leakage")
print("="*80)

# Verify lag_target_1h doesn't peek into the future
print("\nVerifying lag feature construction...")
for i in range(5, 10):
    actual_lag_1h = df['target'].iloc[i-1]
    recorded_lag_1h = df['lag_target_1h'].iloc[i]
    match = "✓" if abs(actual_lag_1h - recorded_lag_1h) < 1e-6 else "✗ LEAKAGE DETECTED"
    print(f"Row {i}: lag_target_1h={recorded_lag_1h:.4f}, actual t-1={actual_lag_1h:.4f} {match}")

# Check if any test data was used in feature engineering
print("\nChecking if test statistics leaked into training features...")
train_lag_max = train_df['lag_target_1h'].max()
train_target_max = train_df['target'].max()
test_target_max = test_df['target'].max()
print(f"Max lag value in train: {train_lag_max:.4f}")
print(f"Max target in train: {train_target_max:.4f}")
print(f"Max target in test: {test_target_max:.4f}")
if train_lag_max <= train_target_max:
    print("✓ No obvious leakage from test set into train features")
else:
    print("✗ WARNING: Lag features may contain future information!")

# =============================================================================
# TEST 2: Feature importance analysis
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Feature importance analysis (checking lag feature dominance)")
print("="*80)

model = EnergyMeanModel()
model.load('models/lgbm_mean_model.pkl')

# Get feature importance
importance = model.model.feature_importance(importance_type='gain')
feature_names = features
importance_dict = dict(zip(feature_names, importance))
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature importance (by gain):")
total_importance = sum(importance)
for feat, imp in sorted_importance:
    pct = (imp / total_importance) * 100
    print(f"  {feat:20s}: {imp:8.1f} ({pct:5.1f}%)")

lag_importance = sum(imp for feat, imp in importance_dict.items() if 'lag' in feat)
lag_pct = (lag_importance / total_importance) * 100
print(f"\nTotal lag feature importance: {lag_pct:.1f}%")
if lag_pct > 90:
    print("⚠ WARNING: Lag features dominate (>90%) - model may be trivially predicting from lags!")
elif lag_pct > 70:
    print("⚠ CAUTION: Lag features very dominant (>70%) - check if model learns patterns beyond autocorrelation")
else:
    print("✓ Feature importance seems balanced")

# =============================================================================
# TEST 3: Model without lag features
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Training model WITHOUT lag features (testing pattern learning)")
print("="*80)

temporal_features = ['hour_sin', 'hour_cos', 'day_of_week']
X_train_no_lag = train_df[temporal_features]
X_test_no_lag = test_df[temporal_features]

print("\nTraining model with only temporal features (no lags)...")
model_no_lag = EnergyMeanModel()
model_no_lag.train(X_train_no_lag, y_train, X_test_no_lag, y_test)

train_pred_no_lag = model_no_lag.predict(X_train_no_lag)
test_pred_no_lag = model_no_lag.predict(X_test_no_lag)

train_mae_no_lag = mean_absolute_error(y_train, train_pred_no_lag)
test_mae_no_lag = mean_absolute_error(y_test, test_pred_no_lag)

print(f"\nResults without lag features:")
print(f"Train MAE: {train_mae_no_lag:.4f} kW")
print(f"Test MAE:  {test_mae_no_lag:.4f} kW")
print(f"Ratio: {test_mae_no_lag/train_mae_no_lag:.2f}x")

# Compare with full model
test_pred_full = model.predict(X_test)
test_mae_full = mean_absolute_error(y_test, test_pred_full)
improvement = ((test_mae_no_lag - test_mae_full) / test_mae_no_lag) * 100

print(f"\nComparison:")
print(f"Full model MAE: {test_mae_full:.4f} kW")
print(f"No-lag model MAE: {test_mae_no_lag:.4f} kW")
print(f"Improvement from lags: {improvement:.1f}%")

if improvement > 50:
    print("⚠ WARNING: Model heavily dependent on lag features (>50% improvement)")
    print("   This suggests the model is mostly doing autocorrelation, not pattern learning")
else:
    print("✓ Reasonable balance between temporal patterns and autocorrelation")

# =============================================================================
# TEST 4: Naive baseline comparison
# =============================================================================
print("\n" + "="*80)
print("TEST 4: Detailed naive baseline comparison")
print("="*80)

# Persistence (use last hour)
naive_persist = X_test['lag_target_1h'].values
mae_persist = mean_absolute_error(y_test, naive_persist)

# Daily persistence (use same hour yesterday)
naive_daily = X_test['lag_target_24h'].values
mae_daily = mean_absolute_error(y_test, naive_daily)

# Mean prediction
naive_mean = np.full_like(y_test, y_train.mean())
mae_mean = mean_absolute_error(y_test, naive_mean)

print(f"\nNaive baselines on test set:")
print(f"Persistence (lag_1h):     {mae_persist:.4f} kW")
print(f"Daily persistence (lag_24h): {mae_daily:.4f} kW")
print(f"Mean prediction:          {mae_mean:.4f} kW")
print(f"\nLGBM model:               {test_mae_full:.4f} kW")

print(f"\nImprovement over persistence: {((mae_persist - test_mae_full)/mae_persist)*100:.1f}%")
print(f"Improvement over daily:       {((mae_daily - test_mae_full)/mae_daily)*100:.1f}%")

if test_mae_full >= mae_persist * 0.95:
    print("⚠ WARNING: Model barely beats simple persistence!")
    print("   The model may just be learning to output lag_target_1h")

# =============================================================================
# TEST 5: Walk-forward validation
# =============================================================================
print("\n" + "="*80)
print("TEST 5: Walk-forward validation (checking temporal robustness)")
print("="*80)

print("\nSplitting test set into 4 quarters...")
test_quarters = [
    ('Q1 2010', test_df.loc['2010-01':'2010-03']),
    ('Q2 2010', test_df.loc['2010-04':'2010-06']),
    ('Q3 2010', test_df.loc['2010-07':'2010-09']),
    ('Q4 2010', test_df.loc['2010-10':'2010-12'])
]

quarter_maes = []
for name, quarter_df in test_quarters:
    if len(quarter_df) > 0:
        X_q = quarter_df[features]
        y_q = quarter_df[target]
        pred_q = model.predict(X_q)
        mae_q = mean_absolute_error(y_q, pred_q)
        quarter_maes.append(mae_q)
        print(f"{name}: MAE = {mae_q:.4f} kW ({len(quarter_df)} samples)")

mae_std = np.std(quarter_maes)
mae_range = max(quarter_maes) - min(quarter_maes)
print(f"\nQuarterly MAE variability:")
print(f"Std dev: {mae_std:.4f} kW")
print(f"Range:   {mae_range:.4f} kW")

if mae_range > 0.1:
    print("⚠ CAUTION: Large performance variation across time periods")
    print("   Model may not be robust to temporal shifts")

# =============================================================================
# TEST 6: Prediction vs actual lag correlation
# =============================================================================
print("\n" + "="*80)
print("TEST 6: Testing if model just copies lag_target_1h")
print("="*80)

test_pred_full = model.predict(X_test)
correlation = np.corrcoef(test_pred_full, X_test['lag_target_1h'])[0, 1]
print(f"\nCorrelation between predictions and lag_target_1h: {correlation:.4f}")

if correlation > 0.98:
    print("⚠ WARNING: Predictions extremely correlated with lag_target_1h (>0.98)")
    print("   Model may be trivially outputting the lag feature")
elif correlation > 0.95:
    print("⚠ CAUTION: Very high correlation with lag_target_1h (>0.95)")
else:
    print("✓ Predictions show independence from simple lag copying")

# Plot scatter
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test['lag_target_1h'], test_pred_full, alpha=0.3, s=1)
plt.plot([0, 10], [0, 10], 'r--', label='y=x line')
plt.xlabel('lag_target_1h')
plt.ylabel('Model prediction')
plt.title(f'Predictions vs lag_target_1h (corr={correlation:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, test_pred_full, alpha=0.3, s=1)
plt.plot([0, 10], [0, 10], 'r--', label='y=x line')
plt.xlabel('Actual target')
plt.ylabel('Model prediction')
plt.title('Predictions vs actual target')
plt.legend()

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/diagnostic_leakage_check.png', dpi=150)
print("\nScatter plots saved to figures/diagnostic_leakage_check.png")

# =============================================================================
# TEST 7: Residual autocorrelation
# =============================================================================
print("\n" + "="*80)
print("TEST 7: Residual autocorrelation analysis")
print("="*80)

residuals = y_test.values - test_pred_full
residual_lag1_corr = np.corrcoef(residuals[1:], residuals[:-1])[0, 1]
print(f"\nResidual lag-1 autocorrelation: {residual_lag1_corr:.4f}")

if abs(residual_lag1_corr) > 0.3:
    print("⚠ WARNING: Strong residual autocorrelation detected")
    print("   Model not capturing all temporal patterns")
else:
    print("✓ Residuals show low autocorrelation (good)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

issues = []
warnings = []

# Analyze results
lag_feature_pct = (lag_importance / total_importance) * 100
if lag_feature_pct > 90:
    issues.append("Lag features dominate model (>90%)")
    
if test_mae_full >= mae_persist * 0.95:
    issues.append("Model barely improves over persistence baseline")
    
if correlation > 0.98:
    issues.append("Predictions trivially copy lag_target_1h")
    
if improvement > 50:
    warnings.append("Heavy dependence on lag features")

if mae_range > 0.1:
    warnings.append("High temporal variability in performance")

print("\n🔴 CRITICAL ISSUES:")
if issues:
    for issue in issues:
        print(f"  - {issue}")
else:
    print("  None detected")

print("\n🟡 WARNINGS:")
if warnings:
    for warn in warnings:
        print(f"  - {warn}")
else:
    print("  None")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)

if lag_feature_pct > 85 and correlation > 0.95:
    print("""
The model appears to be heavily reliant on lag features, particularly lag_target_1h.
This explains the 'too perfect' fits you're seeing - the model is essentially doing:
    prediction ≈ lag_target_1h + small_correction

This is not necessarily wrong, but it means:
  1. Your features have very strong autocorrelation (common in energy data)
  2. The model isn't learning complex patterns, just exploiting temporal correlation
  3. The 0.95x test/train ratio might occur if test period has stronger autocorrelation

RECOMMENDATIONS:
  - Consider removing or downweighting lag features to force pattern learning
  - Try multi-step ahead forecasting (predict t+2, t+3, etc.) where lags are less useful
  - Add regularization to reduce lag feature importance
  - Test on a period with regime changes or anomalies where lags fail
""")
else:
    print("""
The model shows reasonable behavior with balanced feature usage.
The strong fits are likely due to:
  1. Good feature engineering (temporal + lag features complement each other)
  2. Strong inherent patterns in energy consumption data
  3. LightGBM effectively learning non-linear interactions

The 0.95x ratio is still unusual but may be valid if the test period is more predictable.
""")

print("="*80)
print("Diagnostic complete. Check figures/diagnostic_leakage_check.png for visual analysis.")
print("="*80)
