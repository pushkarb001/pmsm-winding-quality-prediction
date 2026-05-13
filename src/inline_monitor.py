"""
inline_monitor.py
=================
PMSM Winding Process — Inline Quality Monitoring Simulation
------------------------------------------------------------
Simulates real-time quality prediction as winding samples are
processed one by one. Loads trained models and scaler, reads
each sample, predicts all quality targets, and outputs a
PASS / WARN / FAIL verdict with confidence scores.

Usage:
    python inline_monitor.py
    python inline_monitor.py --input ../data/raw/pmsm_winding_process_data.csv
    python inline_monitor.py --sample_id PMSM_0042
"""

import argparse
import time
import os
import sys
import joblib
import numpy as np
import pandas as pd

# ── Thresholds for quality verdict ──────────────────────────────────────────
THRESHOLDS = {
    'slot_fill_factor_pct': {
        'warn': 35.0,   # below this → WARN
        'fail': 25.0,   # below this → FAIL
        'direction': 'higher_is_better'
    },
    'process_stability_index': {
        'warn': 0.85,
        'fail': 0.70,
        'direction': 'higher_is_better'
    },
    'winding_resistance_mOhm': {
        'warn': 5000.0,  # above this → WARN
        'fail': 7000.0,  # above this → FAIL
        'direction': 'lower_is_better'
    },
}

FAILURE_PROB_THRESHOLDS = {
    'warn': 0.35,
    'fail': 0.60,
}

# ── Console colours ──────────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def colour_verdict(verdict):
    if verdict == 'PASS': return f"{GREEN}{BOLD}PASS{RESET}"
    if verdict == 'WARN': return f"{YELLOW}{BOLD}WARN{RESET}"
    if verdict == 'FAIL': return f"{RED}{BOLD}FAIL{RESET}"
    return verdict

def get_verdict_reg(target, value):
    t = THRESHOLDS[target]
    if t['direction'] == 'higher_is_better':
        if value < t['fail']: return 'FAIL'
        if value < t['warn']: return 'WARN'
        return 'PASS'
    else:
        if value > t['fail']: return 'FAIL'
        if value > t['warn']: return 'WARN'
        return 'PASS'

def overall_verdict(verdicts, failure_prob):
    if 'FAIL' in verdicts or failure_prob >= FAILURE_PROB_THRESHOLDS['fail']:
        return 'FAIL'
    if 'WARN' in verdicts or failure_prob >= FAILURE_PROB_THRESHOLDS['warn']:
        return 'WARN'
    return 'PASS'

def load_models(src_dir='../src'):
    """Load scaler, feature list and trained models."""
    scaler_path   = os.path.join(src_dir, 'standard_scaler.pkl')
    features_path = os.path.join(src_dir, 'feature_list.txt')
    reg_path      = os.path.join(src_dir, 'best_regression_models.pkl')
    clf_path      = os.path.join(src_dir, 'best_classifier.pkl')

    missing = [p for p in [scaler_path, features_path, reg_path, clf_path]
               if not os.path.exists(p)]
    if missing:
        print(f"{RED}ERROR: Missing model files:{RESET}")
        for p in missing:
            print(f"  {p}")
        print("Run notebooks 02–04 first to generate these files.")
        sys.exit(1)

    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        features = [line.strip() for line in f.readlines()]
    reg_models = joblib.load(reg_path)
    clf_model  = joblib.load(clf_path)

    return scaler, features, reg_models, clf_model

def predict_sample(row, scaler, features, reg_models, clf_model):
    """Run full prediction pipeline on a single sample row."""
    X = row[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_df = pd.DataFrame(X_scaled, columns=features)

    predictions = {}
    verdicts    = []

    # Regression targets
    for target, model in reg_models.items():
        val = model.predict(X_df)[0]
        predictions[target] = val
        if target in THRESHOLDS:
            v = get_verdict_reg(target, val)
            verdicts.append(v)

    # Classification
    failure_prob = clf_model.predict_proba(X_df)[0][1]
    predictions['failure_probability'] = failure_prob

    # Overall verdict
    verdict = overall_verdict(verdicts, failure_prob)
    return predictions, verdict, failure_prob

def print_header():
    print(f"\n{BOLD}{CYAN}{'='*62}{RESET}")
    print(f"{BOLD}{CYAN}  PMSM WINDING — INLINE QUALITY MONITOR{RESET}")
    print(f"{BOLD}{CYAN}{'='*62}{RESET}\n")

def print_sample_result(sample_id, preds, verdict, failure_prob, elapsed_ms):
    print(f"{BOLD}Sample : {sample_id}{RESET}")
    print(f"{'─'*50}")
    print(f"  {'Metric':<35} {'Value':>10}  {'Status'}")
    print(f"  {'─'*48}")

    target_labels = {
        'slot_fill_factor_pct'      : 'Slot Fill Factor',
        'winding_resistance_mOhm'   : 'Winding Resistance (mΩ)',
        'process_stability_index'   : 'Process Stability',
    }

    for target, label in target_labels.items():
        val = preds[target]
        if target in THRESHOLDS:
            v = get_verdict_reg(target, val)
            status = colour_verdict(v)
        else:
            status = ''

        if target == 'slot_fill_factor_pct':
            print(f"  {label:<35} {val:>9.2f}%  {status}")
        elif target == 'winding_resistance_mOhm':
            print(f"  {label:<35} {val:>9.1f}   {status}")
        elif target == 'process_stability_index':
            print(f"  {label:<35} {val:>9.4f}   {status}")
        else:
            print(f"  {label:<35} {val:>9.2f}   {status}")

    # Failure probability
    fp_bar = '█' * int(failure_prob * 20) + '░' * (20 - int(failure_prob * 20))
    fp_color = GREEN if failure_prob < 0.35 else YELLOW if failure_prob < 0.60 else RED
    print(f"  {'─'*48}")
    print(f"  {'Failure Probability':<35} {failure_prob:>9.1%}")
    print(f"  [{fp_color}{fp_bar}{RESET}]")
    print(f"  {'─'*48}")
    print(f"  {'OVERALL VERDICT':<35} {colour_verdict(verdict)}")
    print(f"  {'Inference time':<35} {elapsed_ms:>7.1f} ms")
    print(f"{'─'*50}\n")

def run_monitor(data_path, src_dir, n_samples=10, delay=0.8, sample_id=None):
    print_header()
    print(f"Loading models from: {src_dir}")
    scaler, features, reg_models, clf_model = load_models(src_dir)
    print(f"  ✓ Scaler loaded")
    print(f"  ✓ {len(features)} features")
    print(f"  ✓ {len(reg_models)} regression models")
    print(f"  ✓ Classifier loaded\n")

    # Load data
    df = pd.read_csv(data_path)

    # Feature engineering (must match 02_FeatureEngineering.ipynb)
    df['tension_variability_ratio'] = df['tension_std_N'] / (df['tension_mean_N'] + 1e-6)
    df['tension_dynamic_range']     = df['tension_peak_N'] - df['tension_mean_N']
    df['power_per_turn_W']          = df['power_mean_W'] / (df['num_turns'] + 1e-6)
    df['power_instability_idx']     = df['power_std_W'] / (df['power_mean_W'] + 1e-6)
    df['thermal_stress_idx']        = df['temp_max_C'] * df['temp_gradient_C_per_turn']
    slot_area                       = df['slot_width_mm'] * df['slot_depth_mm']
    wire_cs                         = np.pi * (df['wire_diameter_mm'] / 2.0) ** 2
    df['slot_density_idx']          = (df['num_turns'] * wire_cs) / (slot_area + 1e-6)
    df['mech_load_idx']             = df['wire_tension_N'] * df['spindle_torque_Nm']
    df['speed_tension_product']     = df['winding_speed_rpm'] * df['wire_tension_N']
    df['back_tension_ratio']        = df['back_tension_N'] / (df['wire_tension_N'] + 1e-6)

    # Log transforms
    skewed_cols = [c for c in df.columns if df[c].dtype in ['float64','int64']
                   and df[c].skew() > 1.0 and c not in
                   ['insulation_failure','num_turns']]
    for col in skewed_cols:
        df[f'log_{col}'] = np.log1p(np.clip(df[col], 0, None))

    # Material dummies
    mat_dummies = pd.get_dummies(df['coil_former_material'], prefix='material', dtype=int)
    df = pd.concat([df, mat_dummies], axis=1)

    # Filter by sample_id if specified
    if sample_id:
        df = df[df['sample_id'] == sample_id]
        if len(df) == 0:
            print(f"{RED}Sample ID '{sample_id}' not found.{RESET}")
            sys.exit(1)
        rows = df.iterrows()
        n_samples = 1
    else:
        rows = df.sample(min(n_samples, len(df)),
                         random_state=42).iterrows()

    # ── Main monitoring loop ─────────────────────────────────────────────────
    print(f"Starting inline monitoring — processing {n_samples} samples...\n")
    summary = {'PASS': 0, 'WARN': 0, 'FAIL': 0}

    for _, row in rows:
        sid = row.get('sample_id', 'UNKNOWN')
        t0  = time.perf_counter()
        preds, verdict, failure_prob = predict_sample(
            row, scaler, features, reg_models, clf_model)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print_sample_result(sid, preds, verdict, failure_prob, elapsed_ms)
        summary[verdict] += 1
        if n_samples > 1:
            time.sleep(delay)

    # ── Session summary ──────────────────────────────────────────────────────
    total = sum(summary.values())
    print(f"{BOLD}{CYAN}{'='*62}{RESET}")
    print(f"{BOLD}  SESSION SUMMARY — {total} samples processed{RESET}")
    print(f"{'─'*62}")
    print(f"  {colour_verdict('PASS')}  {summary['PASS']:>3}  ({summary['PASS']/total*100:.1f}%)")
    print(f"  {colour_verdict('WARN')}  {summary['WARN']:>3}  ({summary['WARN']/total*100:.1f}%)")
    print(f"  {colour_verdict('FAIL')}  {summary['FAIL']:>3}  ({summary['FAIL']/total*100:.1f}%)")
    print(f"{BOLD}{CYAN}{'='*62}{RESET}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PMSM Winding Inline Quality Monitor')
    parser.add_argument('--input', default='../data/raw/pmsm_winding_process_data.csv',
                        help='Path to input CSV')
    parser.add_argument('--src',   default='../src',
                        help='Path to saved models directory')
    parser.add_argument('--n',     type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between samples in seconds')
    parser.add_argument('--sample_id', default=None,
                        help='Process a specific sample by ID')
    args = parser.parse_args()

    run_monitor(
        data_path=args.input,
        src_dir=args.src,
        n_samples=args.n,
        delay=args.delay,
        sample_id=args.sample_id,
    )
