import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid")

def run_eda():
    raw_path = 'data/raw/raw_energy_data.csv'
    if not os.path.exists(raw_path):
        print("Raw data not found. Run 'python load_data.py' first.")
        return

    print("Loading data for EDA...")
    df = pd.read_csv(raw_path, low_memory=False)
    
    # 1. Basic Cleaning for Viz
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    
    # Convert target to numeric
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    
    # 2. Resampling for Comparison
    print("Resampling to Hourly...")
    df_hourly = df['Global_active_power'].resample('H').mean().to_frame()
    
    # 3. Visualization: Raw vs Resampled (Small Slice)
    print("Generating Raw vs Resampled comparison plot...")
    plt.figure(figsize=(15, 6))
    
    # Pick a 3-day window to see the detail
    start_date = df.index[1000]
    end_date = start_date + pd.Timedelta(days=1300)
    
    slice_raw = df.loc[start_date:end_date]
    slice_hourly = df_hourly.loc[start_date:end_date]
    
    plt.plot(slice_raw.index, slice_raw['Global_active_power'], label='Raw (1-min)', alpha=0.4, color='blue')
    plt.step(slice_hourly.index, slice_hourly['Global_active_power'], label='Resampled (Hourly)', color='red', where='post', linewidth=2)
    
    plt.title("Energy Consumption: Raw vs Hourly Resampling (3-Day Window)")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.savefig('figures/raw_vs_hourly.png')
    
    # 4. Distribution Plot
    print("Generating distribution plot...")
    plt.figure(figsize=(10, 5))
    sns.histplot(df_hourly['Global_active_power'].dropna(), kde=True, color='green')
    plt.title("Distribution of Hourly Energy Consumption")
    plt.savefig('figures/target_distribution.png')
    
    # 5. Seasonal Decomposition (Quick check for seasonality)
    print("Checking for missing values visually...")
    plt.figure(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Values Map (Yellow = Missing)")
    plt.savefig('figures/missing_values_map.png')

    print("EDA Complete. Check the 'figures/' folder for plots.")

if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    run_eda()