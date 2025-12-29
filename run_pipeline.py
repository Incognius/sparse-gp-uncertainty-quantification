"""
Energy Forecasting Pipeline with Uncertainty Quantification
Runs the complete workflow from data loading to decision analysis.
"""

import os
import sys
import argparse
from pathlib import Path

def setup_paths():
    """Ensure all required directories exist."""
    dirs = ['data/raw', 'data/processed', 'models', 'figures']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_data_loading():
    """Download raw energy data from UCI repository."""
    print("\n" + "="*60)
    print("Step 1: Data Loading")
    print("="*60)
    from data.load_data import download_raw_data
    download_raw_data()

def run_data_processing():
    """Process raw data into hourly features."""
    print("\n" + "="*60)
    print("Step 2: Data Processing")
    print("="*60)
    from data.process_data import run_pipeline
    run_pipeline()

def run_baseline_training():
    """Train baseline LightGBM model."""
    print("\n" + "="*60)
    print("Step 3: Baseline Model Training")
    print("="*60)
    os.system(f"{sys.executable} experiments/01_baseline_training.py")

def run_uncertainty_training():
    """Train Gaussian Process for uncertainty estimation."""
    print("\n" + "="*60)
    print("Step 4: Uncertainty Quantification")
    print("="*60)
    os.system(f"{sys.executable} experiments/02_uncertainty_training.py")

def run_calibration():
    """Evaluate uncertainty calibration."""
    print("\n" + "="*60)
    print("Step 5: Calibration Check")
    print("="*60)
    os.system(f"{sys.executable} evaluation/01_calibration.py")

def run_drift_detection():
    """Test drift detection capability."""
    print("\n" + "="*60)
    print("Step 6: Drift Detection")
    print("="*60)
    os.system(f"{sys.executable} experiments/03_drift_detection.py")

def run_decision_analysis():
    """Run decision analysis."""
    print("\n" + "="*60)
    print("Step 7: Decision Analysis")
    print("="*60)
    os.system(f"{sys.executable} experiments/04_decision_analysis.py")

def main():
    parser = argparse.ArgumentParser(description='Run energy forecasting pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['load', 'process', 'baseline', 'uncertainty', 
                               'calibration', 'drift', 'decision', 'all'],
                       default=['all'],
                       help='Pipeline steps to run')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data loading/processing if already done')
    
    args = parser.parse_args()
    
    setup_paths()
    
    steps = args.steps
    if 'all' in steps:
        steps = ['load', 'process', 'baseline', 'uncertainty', 
                'calibration', 'drift', 'decision']
    
    if args.skip_data:
        steps = [s for s in steps if s not in ['load', 'process']]
    
    step_map = {
        'load': run_data_loading,
        'process': run_data_processing,
        'baseline': run_baseline_training,
        'uncertainty': run_uncertainty_training,
        'calibration': run_calibration,
        'drift': run_drift_detection,
        'decision': run_decision_analysis
    }
    
    print("\n" + "="*60)
    print("ENERGY FORECASTING PIPELINE")
    print("="*60)
    print(f"Running steps: {', '.join(steps)}")
    
    for step in steps:
        try:
            step_map[step]()
        except Exception as e:
            print(f"\nError in step '{step}': {e}")
            print("Continuing to next step...")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - Models: models/")
    print("  - Figures: figures/")
    print("  - Processed data: data/processed/")

if __name__ == "__main__":
    main()
