# PMSM Winding Process — Data-Driven Quality Prediction

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

The dataset was synthetically generated to simulate realistic PMSM winding process 
conditions, with physically motivated signal ranges and five injected anomaly types 
based on known failure modes in motor manufacturing. All modelling and analysis 
decisions from this point were made independently based on what the data showed.

Wire tension, spindle power, and temperature signals are extracted as statistical 
features per winding cycle. Nine additional features are derived from physical process 
knowledge, for example combining wire cross-sectional area, number of turns, and slot 
geometry into a theoretical slot density index that directly captures the geometric 
limit of how tightly a slot can be packed.

Features are scaled using StandardScaler fitted exclusively on training data to prevent 
leakage, then split 70/15/15 into train, validation and test sets with stratification on 
the failure label. Four regression models predict continuous quality outcomes and a 
classifier predicts binary insulation failure, both evaluated with 5-fold cross-validation 
and a held-out test set. SHAP analysis identifies which signal groups drive each quality 
target, forming the basis of sensor priority recommendations in the inline monitoring concept.


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

The slot fill and resistance models performed well beyond expectation — R² above 0.98 in both cases— largely because the slot density index derived feature encodes the core geometric relationship directly. 
Insulation integrity was harder to predict (R² 0.64), which I initially expected to improve with more complex models, but Ridge Regression outperformed Random Forest here suggesting the relationship is approximately linear. Process stability was unpredictable from the available signals entirely — rather than tuning further I treated this as a finding and recommended additional vibration sensors in the monitoring concept document.

### SHAP Feature Importance — Top Predictors per Target

| Target | #1 Feature | #2 Feature | #3 Feature |
|---|---|---|---|
| Slot fill factor | log_slot_density_idx | slot_density_idx | tension_mean_N |
| Winding resistance | wire_diameter_mm | num_turns | slot_depth_mm |
| Insulation integrity | temp_max_C | tension_peak_N | tension_mean_N |
| Process stability | log_tension_variability_ratio | tension_variability_ratio | power_instability_idx |
| Insulation failure | temp_max_C | winding_temperature_C | log_temp_time_above_60s |

The SHAP results confirmed most of what the correlation analysis suggested, 
with one surprise — derived features dominated over raw signals for slot fill 
factor, with slot_density_idx ranking above all original sensor readings. 
This validated the feature engineering approach more strongly than I expected.

Temperature features appearing as top predictors for both insulation integrity 
and failure prediction is the most actionable finding — it points directly to 
pyrometry as the single most critical sensor for an inline insulation monitoring 
system.
