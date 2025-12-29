import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mean_model import EnergyMeanModel
from sklearn.metrics import mean_absolute_error

# Load preprocessed data
df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)

features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
target = 'target'

# Temporal split: train on 2009, test on 2010
train_df = df[df.index < '2010-01-01']
test_df = df[df.index >= '2010-01-01']

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# Train baseline LightGBM model
model = EnergyMeanModel()
model.train(X_train, y_train, X_test, y_test)
model.save('models/lgbm_mean_model.pkl')

# Evaluate performance
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

print(f"\n--- Baseline Model Performance ---")
print(f"Train MAE: {train_mae:.4f} kW")
print(f"Test MAE:  {test_mae:.4f} kW")
print(f"Ratio (Test/Train): {test_mae/train_mae:.2f}x")

# Compare against naive persistence baseline
naive_preds = X_test['lag_target_1h']
naive_mae = mean_absolute_error(y_test, naive_preds)
print(f"Naive Persistence MAE: {naive_mae:.4f} kW")
print(f"Model Improvement: {(1 - test_mae/naive_mae)*100:.1f}%")

# Save residuals for GP training
test_results = test_df.copy()
test_results['predicted_mean'] = test_preds
test_results['residual'] = test_results[target] - test_results['predicted_mean']
test_results.to_csv('data/processed/test_with_residuals.csv')

# Visualize fit quality
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

ax1.plot(y_train.iloc[-168:].values, label='Actual', color='blue', alpha=0.5)
ax1.plot(train_preds[-168:], label='Predicted', color='red', linestyle='--')
ax1.set_title("Training Set Fit (Last Week)")
ax1.legend()

ax2.plot(y_test.iloc[:168].values, label='Actual', color='blue', alpha=0.5)
ax2.plot(test_preds[:168], label='Predicted', color='red', linestyle='--')
ax2.set_title("Test Set Fit (First Week)")
ax2.legend()

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/01_baseline_diagnostics.png')
print(f"\nDiagnostic plot saved to figures/01_baseline_diagnostics.png")