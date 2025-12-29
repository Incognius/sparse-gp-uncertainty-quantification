#!/usr/bin/env python
"""
Quick test to verify pipeline setup is working correctly.
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'torch': 'torch',
        'gpytorch': 'gpytorch',
        'ucimlrepo': 'ucimlrepo'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    return missing

def check_directory_structure():
    """Verify required directories exist."""
    dirs = ['data/raw', 'data/processed', 'models', 'figures', 'experiments', 'evaluation']
    
    print("\nDirectory structure:")
    for d in dirs:
        exists = os.path.exists(d)
        print(f"{'✓' if exists else '✗'} {d}")

def check_files():
    """Verify key files exist."""
    files = [
        'run_pipeline.py',
        'data/load_data.py',
        'data/process_data.py',
        'models/mean_model.py',
        'models/residual_gp.py',
        'experiments/01_baseline_training.py',
        'experiments/02_uncertainty_training.py',
        'experiments/03_drift_detection.py',
        'experiments/04_decision_analysis.py',
        'evaluation/01_calibration.py'
    ]
    
    print("\nKey files:")
    for f in files:
        exists = os.path.exists(f)
        print(f"{'✓' if exists else '✗'} {f}")

if __name__ == "__main__":
    print("="*60)
    print("Pipeline Setup Verification")
    print("="*60)
    
    print("\nChecking dependencies:")
    missing = check_dependencies()
    
    check_directory_structure()
    check_files()
    
    print("\n" + "="*60)
    if missing:
        print("SETUP INCOMPLETE")
        print("\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("SETUP COMPLETE")
        print("\nReady to run:")
        print("python run_pipeline.py")
    print("="*60)
