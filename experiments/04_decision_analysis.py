import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mean_model import EnergyMeanModel
from models.residual_gp import ResidualGPWrapper


def run_decision_analysis():
    """Run and visualize decision analysis."""
    
    print("\n--- Decision Analysis ---")
    
    df = pd.read_csv('data/processed/hourly_energy.csv', index_col='datetime', parse_dates=True)
    features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_target_1h', 'lag_target_2h', 'lag_target_24h']
    
    mean_model = EnergyMeanModel()
    mean_model.load('models/lgbm_mean_model.pkl')
    
    gp_system = ResidualGPWrapper()
    gp_system.load('models/residual_gp_model.pth')
    
    sim_data = df.loc['2010-02-06':'2010-02-09'].copy()
    actual_load = sim_data['target'].values
    
    # Inject drift
    drift_start = '2010-02-07 12:00:00'
    drift_mask = sim_data.index >= drift_start
    random_means = np.random.choice([-8, -7, -6, -5, -4], size=drift_mask.sum())
    noise = np.random.normal(loc=random_means, scale=1.5)
    
    input_data = sim_data.copy()
    input_data.loc[drift_mask, 'lag_target_1h'] = noise
    input_data.loc[drift_mask, 'lag_target_2h'] = noise
    
    # Model inference
    pred_mean = mean_model.predict(input_data[features])
    gp_mean, gp_std = gp_system.predict(input_data[features].values)
    pred_robust = pred_mean + gp_mean
    
    # Decision economics
    REWARD_SUCCESS = 5.0
    COST_FAILURE = 100.0
    BREAKER_LIMIT = 4.0
    ADDED_LOAD = 2.5
    
    bank_naive = 0.0
    bank_robust = 0.0
    history_naive = []
    history_robust = []
    results = []
    
    for i in range(len(sim_data)):
        # Naive: just check if predicted load is safe
        est_load_naive = pred_mean[i] + ADDED_LOAD
        action_naive = 1 if est_load_naive < BREAKER_LIMIT else 0
        
        # Robust: calculate expected value
        mu = pred_robust[i] + ADDED_LOAD
        sigma = gp_std[i]
        z = (BREAKER_LIMIT - mu) / sigma
        prob_fail = 1 - norm.cdf(z)
        prob_success = 1 - prob_fail
        
        expected_value = (prob_success * REWARD_SUCCESS) - (prob_fail * COST_FAILURE)
        action_robust = 1 if expected_value > 0 else 0
        
        # Reality check
        real_load_after_act = actual_load[i] + ADDED_LOAD
        did_trip = real_load_after_act > BREAKER_LIMIT
        
        step_profit_naive = 0
        if action_naive:
            step_profit_naive = -COST_FAILURE if did_trip else REWARD_SUCCESS
        
        step_profit_robust = 0
        if action_robust:
            step_profit_robust = -COST_FAILURE if did_trip else REWARD_SUCCESS
        
        bank_naive += step_profit_naive
        bank_robust += step_profit_robust
        
        history_naive.append(bank_naive)
        history_robust.append(bank_robust)
        
        results.append({
            'time': sim_data.index[i],
            'naive_action': action_naive,
            'robust_action': action_robust,
            'did_trip': did_trip,
            'naive_pnl': step_profit_naive,
            'robust_pnl': step_profit_robust
        })
    
    df_res = pd.DataFrame(results).set_index('time')
    baseline_abs = abs(bank_naive) if abs(bank_naive) > 0 else 1.0
    improvement_pct = ((bank_robust - bank_naive) / baseline_abs) * 100
    
    print("\n" + "=" * 50)
    print("Financial Performance Report")
    print("=" * 50)
    print(f"Naive Strategy Net Profit:   ${bank_naive:,.2f}")
    print(f"Robust Strategy Net Profit:  ${bank_robust:,.2f}")
    print(f"Net Value Generated:         ${bank_robust - bank_naive:,.2f}")
    print(f"Performance Delta:           {improvement_pct:+.1f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(sim_data.index, history_naive, 'r--', label='Naive Agent')
    plt.plot(sim_data.index, history_robust, 'g-', linewidth=2, label='Risk-Aware Agent')
    plt.axvline(pd.to_datetime(drift_start), color='blue', linestyle=':', label='Sensor Failure Starts')
    plt.title("Cumulative Profit/Loss Over Time")
    plt.ylabel("Net Profit ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/04_decision_trajectory.png')
    print("\nChart saved to figures/04_decision_trajectory.png")


if __name__ == "__main__":
    run_decision_analysis()
