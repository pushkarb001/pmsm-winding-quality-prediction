# Inline Quality Monitoring Concept
## PMSM Winding Process — Data-Driven Quality Prediction System

---

## 1. Overview

This document describes the concept for deploying the trained process models as an **inline quality monitoring system** for PMSM stator winding. The system predicts winding quality outcomes in real time during the winding process, enabling operators to intervene before defective parts proceed to downstream assembly.

This concept maps directly to the industrial inline monitoring requirements described in the thesis on inductive compaction of RF litz wires — adapted here for the PMSM motor winding domain.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WINDING MACHINE                             │
│                                                                 │
│  [Tension Sensor] → tension_mean, tension_std, tension_peak     │
│  [Power Monitor]  → power_mean, power_std, power_rms            │
│  [Pyrometer]      → temp_max, temp_gradient                     │
│  [Encoder]        → winding_speed, num_turns                    │
└────────────────────────┬────────────────────────────────────────┘
                         │ raw signal stream (10–100 Hz)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SIGNAL PROCESSING LAYER                        │
│                                                                 │
│  • Feature extraction from time-series windows                  │
│  • Derived feature computation (9 engineered features)          │
│  • Log transformation of skewed signals                         │
│  • StandardScaler normalisation (fitted on training data)       │
└────────────────────────┬────────────────────────────────────────┘
                         │ feature vector (44 features)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREDICTION LAYER                               │
│                                                                 │
│  Model 1 (Random Forest): slot_fill_factor_pct    R²=0.984     │
│  Model 2 (Random Forest): winding_resistance_mOhm R²=0.996     │
│  Model 3 (Ridge):         insulation_integrity    R²=0.639     │
│  Model 4 (Classifier):    insulation_failure      AUC=0.673    │
└────────────────────────┬────────────────────────────────────────┘
                         │ quality predictions + confidence
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VERDICT LAYER                                  │
│                                                                 │
│  PASS  → continue to next winding                               │
│  WARN  → flag for manual inspection                             │
│  FAIL  → stop machine, reject part, log event                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              [Operator HMI / Dashboard]
              [Data Logger / MES System]
```

---

## 3. Data Flow & Timing

| Stage | Operation | Latency |
|---|---|---|
| Signal acquisition | Sensor sampling at 50 Hz | 20 ms/sample |
| Feature extraction | Window aggregation + derived features | < 5 ms |
| Normalisation | StandardScaler transform | < 1 ms |
| Model inference | 4 models × predict() | < 10 ms |
| Verdict decision | Threshold comparison | < 1 ms |
| **Total per sample** | **End-to-end** | **< 20 ms** |

The system comfortably operates within a single winding cycle (typically 2–30 seconds per coil), making true inline prediction feasible without slowing production.

---

## 4. Quality Thresholds & Alert Logic

### Regression Targets

| Target | PASS | WARN | FAIL |
|---|---|---|---|
| Slot fill factor (%) | ≥ 35% | 25–35% | < 25% |
| Insulation integrity score | ≥ 50 | 35–50 | < 35 |
| Process stability index | ≥ 0.85 | 0.70–0.85 | < 0.70 |
| Winding resistance (mΩ) | < 5000 | 5000–7000 | > 7000 |

### Classification Target

| Failure probability | Verdict |
|---|---|
| < 35% | PASS |
| 35–60% | WARN |
| > 60% | FAIL |

### Overall Verdict Rule
- Any single FAIL → overall FAIL
- Any single WARN (no FAIL) → overall WARN
- All PASS → overall PASS

---

## 5. Feature Importance Findings & Sensor Priority

Based on SHAP analysis, the following sensors are most critical for each quality dimension:

| Quality Target | Most Important Signal | Signal Group |
|---|---|---|
| Slot fill factor | slot_density_idx (derived) | Geometry |
| Winding resistance | wire_diameter_mm, num_turns | Process params |
| Insulation integrity | temp_max_C | Temperature |
| Process stability | tension_variability_ratio | Tension |
| Insulation failure | temp_max_C, winding_temperature_C | Temperature |

**Key recommendation:** Temperature monitoring (pyrometry) is the single most critical sensing modality. If budget constrains sensor selection, prioritise the pyrometer over additional tension or power sensors.

---

## 6. Model Deployment Architecture

```
src/
├── standard_scaler.pkl          ← fitted normalisation transform
├── feature_list.txt             ← ordered list of 44 feature names
├── best_regression_models.pkl   ← dict of 4 trained regressors
├── best_classifier.pkl          ← insulation failure classifier
└── inline_monitor.py            ← real-time prediction script
```

All models are serialised with `joblib` and load in < 200 ms. No GPU required — inference runs on standard industrial PC hardware.

---

## 7. Limitations & Recommendations

### Current Limitations
- **process_stability_index** could not be reliably predicted (R² = −0.15). Additional sensors capturing vibration or acoustic emission may be needed.
- **insulation_failure AUC = 0.673** indicates moderate classification performance. Direct voltage withstand testing remains necessary as a complementary check.
- Models trained on synthetic data — retraining on real machine data is required before production deployment.

### Recommendations for Production Deployment
1. **Retrain on real winding data** from at least 3 machine configurations
2. **Add vibration sensor** to improve process stability prediction
3. **Implement drift detection** — monitor feature distributions over time and trigger retraining when drift is detected
4. **Set thresholds empirically** based on actual scrap rates from production data
5. **Log all predictions** to enable continuous model improvement

---

## 8. Mapping to Thesis Requirements

| Thesis Requirement | This Implementation |
|---|---|
| Signal-based quality prediction | ✓ Tension, power, temperature → quality targets |
| Feature engineering from process signals | ✓ 9 derived features from 3 signal groups |
| Statistical + ML models | ✓ Ridge, Random Forest, XGBoost |
| Cross-validation + hold-out | ✓ 5-fold CV + 15% hold-out test set |
| Feature importance analysis | ✓ SHAP values across all targets |
| Inline applicability concept | ✓ This document + inline_monitor.py |
| Recommendations for quality monitoring | ✓ Section 7 above |

