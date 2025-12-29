import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_raw_data():
    output_path = 'data/raw/raw_energy_data.csv'
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}. Skipping download.")
        return

    print("Fetching dataset from UCI (ID 235)...")
    energy_data = fetch_ucirepo(id=235) 
    
    X = energy_data.data.features 
    y = energy_data.data.targets 
    
    raw_df = pd.concat([X, y], axis=1)
    
    os.makedirs('data/raw', exist_ok=True)
    
    print(f"Saving raw data to {output_path} (this might take a minute)...")
    raw_df.to_csv(output_path, index=False)
    print("Download and save complete.")

if __name__ == "__main__":
    download_raw_data()