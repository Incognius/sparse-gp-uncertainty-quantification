import pandas as pd
import numpy as np
import os

def run_pipeline():
    raw_path = 'data/raw/raw_energy_data.csv'
    if not os.path.exists(raw_path):
        print("Raw data not found. Ensure it's in data/raw/raw_energy_data.csv")
        return

    print("Loading raw data...")
    df = pd.read_csv(raw_path, low_memory=False)
    
    # Format datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df = df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce').dropna(subset=['Global_active_power'])
    
    # Resample to hourly mean
    df_hourly = df['Global_active_power'].resample('h').mean().to_frame()
    df_hourly.columns = ['target']
    
    # Better interpolation: use same hour from previous day(s) to avoid flat regions
    # This captures daily patterns without using future data
    print(f"Missing values before interpolation: {df_hourly['target'].isna().sum()}")
    
    # Strategy: Try 24h lag first, then 168h (weekly), then limited ffill as last resort
    missing_mask = df_hourly['target'].isna()
    df_hourly.loc[missing_mask, 'target'] = df_hourly['target'].shift(24).loc[missing_mask]  # Try yesterday same hour
    
    still_missing = df_hourly['target'].isna()
    df_hourly.loc[still_missing, 'target'] = df_hourly['target'].shift(168).loc[still_missing]  # Try last week same hour
    
    # For any remaining gaps (early in dataset), use limited forward fill (max 6 hours)
    df_hourly['target'] = df_hourly['target'].ffill(limit=6)
    
    print(f"Missing values after interpolation: {df_hourly['target'].isna().sum()}")
    
    # Create temporal features
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['day_of_week'] = df_hourly.index.dayofweek
    
    # Create lag features
    df_hourly['lag_target_1h'] = df_hourly['target'].shift(1)
    df_hourly['lag_target_2h'] = df_hourly['target'].shift(2)
    df_hourly['lag_target_24h'] = df_hourly['target'].shift(24)
    
    df_hourly.dropna(inplace=True)
    
    os.makedirs('data/processed', exist_ok=True)
    df_hourly.to_csv('data/processed/hourly_energy.csv')
    print(f"Success! Processed data saved with columns: {list(df_hourly.columns)}")

if __name__ == "__main__":
    run_pipeline()