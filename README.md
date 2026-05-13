# PMSM Winding Process — Data-Driven Quality Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![SHAP](https://img.shields.io/badge/SHAP-analysis-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A data-driven process model that predicts winding quality outcomes in PMSM (Permanent Magnet Synchronous Motor) stator manufacturing — using machine process signals to enable inline quality monitoring without destructive testing.

---

## Project Motivation

In motor manufacturing, winding quality is traditionally verified through post-process electrical testing and visual inspection. This introduces latency — defective parts may proceed through several downstream assembly steps before rejection. The goal of this project is to predict quality outcomes **during the winding process itself**, using only the signals already available from the winding machine.

This project mirrors the methodological framework of data-based process modelling in joining technology, applied to the PMSM winding domain.

---

## Dataset

Synthetic but physically realistic dataset of **500 winding experiments** with **30 columns** spanning three signal groups:

| Signal Group | Features | Physical Source |
|---|---|---|
| Tension-displacement | mean, std, peak, slope, area | Wire tension sensor |
| Spindle power | mean, std, peak, rms, energy | Motor power monitor |
| Temperature / pyrometry | max temp, gradient, time above 60°C | Pyrometer |
| Process parameters | speed, torque, wire geometry, slot dimensions | Machine settings |

### Target Variables

| Target | Type | Analogy to Thesis |
|---|---|---|
| `slot_fill_factor_pct` | Regression | Compaction degree |
| `winding_resistance_mOhm` | Regression | Electrical resistance |
| `insulation_integrity_score` | Regression | Quality score |
| `process_stability_index` | Regression | Process consistency |
| `insulation_failure` | Classification | Sample failure rate |

---

## Methodology

```
Raw Data (500 samples, 30 cols)
        ↓
Exploratory Data Analysis
  • Distribution analysis, correlation heatmap
  • Signal group correlation with targets
  • ANOVA by coil former material
        ↓
Feature Engineering (23 → 44 features)
  • 9 physics-motivated derived features
  • Log transformation of skewed signals
  • One-hot encoding of categorical variables
  • StandardScaler (fit on training data only)
        ↓
Train / Val / Test Split (70 / 15 / 15) — stratified
        ↓
Model Training & Comparison
  • Linear / Ridge Regression
  • Random Forest
  • XGBoost
        ↓
Validation
  • 5-Fold Cross-Validation
  • Hold-out test set evaluation
  • Learning curve analysis
        ↓
Feature Importance (SHAP)
  • Per-target feature ranking
  • Signal group contribution analysis
  • Dependence plots
        ↓
Inline Monitoring Concept
  • Real-time prediction script
  • PASS / WARN / FAIL verdict system
```

---

## Results

### Regression Models

| Target | Best Model | Val R² | CV R² |
|---|---|---|---|
| Slot fill factor | Random Forest | 0.984 | 0.985 ± 0.006 |
| Winding resistance | Random Forest | 0.996 | 0.996 ± 0.001 |
| Insulation integrity | Ridge Regression | 0.639 | 0.747 ± 0.041 |
| Process stability | Random Forest | −0.148 | 0.068 ± 0.054 |

### Classification — Insulation Failure Prediction

| Model | Val AUC | CV AUC |
|---|---|---|
| Logistic Regression | 0.673 | 0.664 ± 0.056 |
| Random Forest | 0.643 | 0.657 ± 0.055 |
| XGBoost | 0.651 | 0.623 ± 0.044 |

### Key Findings

- **Slot fill factor and winding resistance are highly predictable** (R² > 0.98) from process signals alone — enabling reliable inline geometric and electrical quality estimation
- **Insulation failure is moderately predictable** (AUC 0.67) — temperature signals are the dominant predictor, suggesting pyrometry is the highest-value sensor for failure prevention
- **Process stability proved unpredictable** (R² < 0) from the available signals — indicating that additional sensors (vibration, acoustic emission) would be needed for this target
- **Feature engineering added significant value** — derived features (slot density index, tension variability ratio) outperformed all raw signal groups in SHAP importance

### SHAP Feature Importance — Top Predictors per Target

| Target | #1 Feature | #2 Feature | #3 Feature |
|---|---|---|---|
| Slot fill factor | log_slot_density_idx | slot_density_idx | tension_mean_N |
| Winding resistance | wire_diameter_mm | num_turns | slot_depth_mm |
| Insulation integrity | temp_max_C | tension_peak_N | tension_mean_N |
| Process stability | log_tension_variability_ratio | tension_variability_ratio | power_instability_idx |
| Insulation failure | temp_max_C | winding_temperature_C | log_temp_time_above_60s |

---

## Repository Structure

```
pmsm-winding-quality-prediction/
├── data/
│   ├── raw/
│   │   └── pmsm_winding_process_data.csv    ← 500 samples, 30 features
│   └── processed/
│       ├── train.csv                         ← 350 samples (70%)
│       ├── val.csv                           ← 75 samples (15%)
│       └── test.csv                          ← 75 samples (15%)
├── notebooks/
│   ├── 01_EDA.ipynb                          ← Exploratory data analysis
│   ├── 02_FeatureEngineering.ipynb           ← Feature pipeline
│   ├── 03_Regression.ipynb                   ← Regression models
│   ├── 04_Classification.ipynb               ← Failure classification
│   └── 05_FeatureImportance.ipynb            ← SHAP analysis
├── src/
│   ├── inline_monitor.py                     ← Real-time monitoring script
│   ├── standard_scaler.pkl                   ← Fitted scaler
│   ├── feature_list.txt                      ← 44 model features
│   ├── best_regression_models.pkl            ← Trained regressors
│   └── best_classifier.pkl                   ← Trained classifier
├── results/
│   └── plots/                                ← All generated figures
├── docs/
│   └── inline_monitoring_concept.md          ← System architecture & concept
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone and set up environment
```bash
git clone https://github.com/pushkarb001/pmsm-winding-quality-prediction.git
cd pmsm-winding-quality-prediction
python -m venv venv
source venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

### 2. Run notebooks in order
Open VS Code, select the venv kernel, and run each notebook:
```
notebooks/01_EDA.ipynb
notebooks/02_FeatureEngineering.ipynb
notebooks/03_Regression.ipynb
notebooks/04_Classification.ipynb
notebooks/05_FeatureImportance.ipynb
```

### 3. Run inline monitor
```bash
cd src
python inline_monitor.py                        # process 10 random samples
python inline_monitor.py --n 20                 # process 20 samples
python inline_monitor.py --sample_id PMSM_0042  # specific sample
```

---

## Relevance to Process Monitoring Research

This project demonstrates the core methodological skills required for data-based process modelling in manufacturing:

- **Signal feature extraction** from tension, power and thermal process signals
- **Physics-motivated feature engineering** that outperforms raw signal features
- **Multi-target regression and classification** with rigorous cross-validation
- **SHAP-based feature importance** identifying the most predictive signal sources
- **Inline applicability concept** with latency analysis and threshold design

The negative R² for process stability and moderate AUC for failure classification are honest results that point toward concrete sensor recommendations — which is the kind of finding that drives real engineering decisions in process monitoring system design.

---

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
xgboost >= 2.0
shap >= 0.44
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.10
joblib >= 1.3
jupyter >= 1.0
```

Install all: `pip install -r requirements.txt`

---

*Project developed as a portfolio demonstration of data-driven process modelling skills for manufacturing quality prediction.*
