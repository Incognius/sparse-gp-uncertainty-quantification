import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mean_model import EnergyMeanModel
from models.residual_gp import ResidualGPWrapper
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)
features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
target = 'target'

train_df = df[df.index < '2010-01-01']
test_df = df[df.index >= '2010-01-01']

# Load baseline model and compute training residuals
mean_model = EnergyMeanModel()
mean_model.load('models/lgbm_mean_model.pkl')

train_preds = mean_model.predict(train_df[features])
train_residuals = train_df[target].values - train_preds

# Check if GP model already exists
if os.path.exists('models/residual_gp_model.pth'):
    print("GP model already exists, loading...")
    gp_system = ResidualGPWrapper(num_inducing=500)
    gp_system.load('models/residual_gp_model.pth')
    print("Skipping training, using existing model.")
    history = None
else:
    # Train sparse GP on residuals with tracking for visualization
    print("Training new GP model...")
    gp_system = ResidualGPWrapper(num_inducing=500)
    history = gp_system.fit(train_df[features].values, train_residuals, epochs=100, track_inducing=True)
    gp_system.save('models/residual_gp_model.pth')

# Generate uncertainty-aware predictions
gp_correction, gp_std = gp_system.predict(test_df[features].values)
test_preds_mean = mean_model.predict(test_df[features])
final_preds = test_preds_mean + gp_correction

# Save results with uncertainty estimates
test_df_results = test_df.copy()
test_df_results['lgbm_mean'] = test_preds_mean
test_df_results['gp_uncertainty_std'] = gp_std
test_df_results['final_prediction'] = final_preds
test_df_results.to_csv('data/processed/test_with_uncertainty.csv')

# Visualize predictions with uncertainty bands
plt.figure(figsize=(15, 7))
window_idx = 168
time = test_df.index[:window_idx]
actual = test_df[target].values[:window_idx]
pred = final_preds[:window_idx]
std = gp_std[:window_idx]

plt.plot(time, actual, label='Actual', color='black', alpha=0.4)
plt.plot(time, pred, label='Prediction', color='red', linestyle='--')
plt.fill_between(time, pred - 1.96 * std, pred + 1.96 * std, 
                 color='orange', alpha=0.3, label='95% Confidence')

plt.title("Uncertainty-Aware Energy Forecast")
plt.ylabel("Active Power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/02_uncertainty_forecast.png')
print("\nUncertainty forecast saved to figures/02_uncertainty_forecast.png")

# Visualize GP inducing point evolution (only if we trained)
if history is not None:
    print("\nGenerating GP evolution visualization...")
    X_train = train_df[features].values
    
    # Scale data to match GP's internal scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualize inducing point evolution
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    snapshots = [0, 2, 5, -1]
    titles = ['Initial (Random)', 'Epoch 20', 'Epoch 50', 'Final']
    
    for i, idx in enumerate(snapshots):
        ax = axes[i]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgrey', s=1, alpha=0.1, label='Data Manifold')
        
        if idx < len(history):
            Z_pca = pca.transform(history[idx])
            ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c='red', s=15, edgecolors='black', label='Inducing Points')
        
        ax.set_title(titles[i])
        if i == 0: ax.set_ylabel("PCA Component 2")
        ax.set_xlabel("PCA Component 1")
    
    plt.suptitle("Sparse GP: Inducing Point Evolution")
    plt.tight_layout()
    plt.savefig('figures/02_gp_evolution.png')
    print("GP evolution visualization saved to figures/02_gp_evolution.png")