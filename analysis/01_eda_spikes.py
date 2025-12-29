import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the newly processed data
data_path = 'data/processed/hourly_energy.csv'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run 'python data/process_data.py' first.")
else:
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Visualize a 3-day window
    window = df.iloc[100:100+72] # 3 days (72 hours)

    plt.figure(figsize=(15, 6))
    
    # Updated to match 'target_mean' and 'peak_power' from process_data.py
    plt.fill_between(window.index, window['target_mean'], window['peak_power'], 
                     color='orange', alpha=0.2, label='Intra-hour Volatility (Peak-Mean Range)')
    
    plt.plot(window.index, window['target_mean'], label='Hourly Mean (Target)', color='red', linewidth=2)
    plt.plot(window.index, window['peak_power'], label='Hourly Peak (Spike)', color='black', linestyle='--', alpha=0.5)

    plt.title("Energy Consumption: Mean vs Peak (Capturing the Spikes)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/eda_spikes_captured.png')
    print("EDA plot saved to figures/eda_spikes_captured.png")