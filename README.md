# Energy Forecasting with Uncertainty Quantification

A complete pipeline for energy consumption forecasting that combines gradient boosting with Gaussian processes to provide calibrated uncertainty estimates for downstream decision-making.

## Project Structure

```
├── data/
│   ├── load_data.py              # Download raw data from UCI repository
│   ├── process_data.py           # Convert to hourly features with lags
│   ├── raw/                      # Raw energy consumption data (excluded from git)
│   └── processed/                # Processed hourly datasets (excluded from git)
├── models/
│   ├── mean_model.py             # LightGBM baseline forecaster
│   └── residual_gp.py            # Sparse GP for uncertainty estimation
├── experiments/
│   ├── 01_baseline_training.py   # Train baseline model
│   ├── 02_uncertainty_training.py # Train GP on residuals
│   ├── 03_drift_detection.py     # Test OOD detection capabilities
│   └── 04_decision_analysis.py   # Risk-aware decision making
├── evaluation/
│   └── 01_calibration.py         # Validate uncertainty calibration
├── figures/                       # Generated visualizations
├── run_pipeline.py               # Master pipeline runner
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── README.md
```

## Setup

### 1. Verify Environment Setup

```bash
# Check Python version (3.8+ required)
python --version

# Verify pip is available
pip --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test imports
python -c "import torch; import gpytorch; import lightgbm; print('Setup successful!')"
```

## Quick Start

### Run Complete Pipeline

```bash
python run_pipeline.py
```

This executes all 7 steps:
1. Data loading (downloads UCI dataset)
2. Data processing (hourly aggregation + feature engineering)
3. Baseline training (LightGBM model)
4. Uncertainty training (Sparse GP on residuals)
5. Calibration validation (coverage analysis)
6. Drift detection (OOD uncertainty inflation test)
7. Decision analysis (risk-aware vs naive policy comparison)

### Run Specific Steps

```bash
# Run only baseline and uncertainty training
python run_pipeline.py --steps baseline uncertainty

# Run calibration and drift detection
python run_pipeline.py --steps calibration drift
```

### Skip Data Loading

If you already have processed data:

```bash
python run_pipeline.py --skip-data
```

This starts from step 3 (baseline training).

## Pipeline Steps

1. **Data Loading**: Downloads UCI Household Electric Power Consumption dataset (2,075,259 records)
2. **Data Processing**: 
   - Hourly aggregation with temporal-pattern-aware interpolation (24h/168h lag fallback)
   - Feature engineering: cyclical time features, lag variables (1h, 2h, 24h)
3. **Baseline Training**: LightGBM gradient boosting with early stopping
4. **Uncertainty Estimation**: Sparse variational GP (500 inducing points) on residuals
5. **Calibration**: Verify prediction interval coverage (68% and 95% intervals)
6. **Drift Detection**: Test uncertainty inflation under simulated sensor failures
7. **Decision Analysis**: Compare naive vs risk-aware policies using expected value

## Key Features

- **Calibrated Uncertainty**: GP provides well-calibrated confidence intervals (93% coverage on 95% intervals)
- **Drift Detection**: Automatic OOD detection via 2.05x uncertainty amplification
- **Decision Support**: Risk-aware policies generate $3,250 additional value (+98.8%)
- **Scalable GP**: Sparse inducing points reduce complexity from O(N³) to O(NM²)
- **Superior Generalization**: 0.96x test/train ratio with 15.7% improvement over naive baselines

## Requirements

Python 3.8+ with the following packages (see `requirements.txt`):
- PyTorch + GPyTorch (Gaussian processes)
- LightGBM (gradient boosting)
- pandas, numpy, matplotlib, scikit-learn
- ucimlrepo (dataset loader)
- scipy (statistics)

## Model Architecture

1. **Mean Model**: LightGBM gradient boosting with early stopping
   - Features: hour_sin, hour_cos, day_of_week, lag_target_{1h,2h,24h}
   - Optimization: L-BFGS with validation-based early stopping
   
2. **Residual GP**: Sparse variational GP with 500 inducing points
   - Kernel: RBF + Linear (smooth patterns + global trends)
   - Variational inference: ELBO optimization via Adam
   - Noise constraint: σ_n ∈ [1e-4, 0.4] for stability
   
3. **Final Prediction**: μ(x) = f_lgbm(x) + f_gp(x), σ(x) = σ_gp(x)

## Results

**Forecasting Performance:**
- Test MAE: 0.3438 kW
- Train MAE: 0.3576 kW (0.96x ratio)
- Improvement over naive: 15.7%

**Calibration:**
- 68% interval coverage: 73.57% (target: 68.2%)
- 95% interval coverage: 93.03% (target: 95.0%)

**Drift Detection:**
- Normal uncertainty: 0.4720 kW
- Drift uncertainty: 0.9661 kW (2.05x increase)

**Decision Value:**
- Naive policy: -$3,290
- Risk-aware policy: -$40
- Net value: +$3,250 (+98.8% improvement)

## Output Files

The pipeline generates:
- **Models**: `models/lgbm_mean_model.pkl`, `models/residual_gp_model.pth`
- **Figures**: Numbered diagnostic plots in `figures/`
  - `01_baseline_diagnostics.png`
  - `02_uncertainty_forecast.png`
  - `03_drift_detection_chaotic.png`
  - `04_decision_trajectory.png`
  - `05_calibration_reliability.png`
- **Data**: Processed datasets with predictions in `data/processed/`

## Documentation

- **Project Report**: `project_report.tex` - Comprehensive technical report with results
- **Theory Document**: `theory_mathematical_foundations.tex` - Rigorous mathematical derivations from Bayesian inference to sparse variational GPs

Compile LaTeX documents:
```bash
pdflatex project_report.tex
pdflatex theory_mathematical_foundations.tex
```

## Citation

**Dataset**: Household Electric Power Consumption, UCI Machine Learning Repository

**Key References**:
- Rasmussen & Williams (2006) - Gaussian Processes for Machine Learning
- Hensman et al. (2013) - Gaussian Processes for Big Data (Sparse Variational GP)
- Titsias (2009) - Variational Learning of Inducing Variables

## License

This project uses the UCI Household Electric Power Consumption dataset under their terms of use.
