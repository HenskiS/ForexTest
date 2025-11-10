"""
Train XGBoost with alternative targets.

Usage:
  python train_xgboost_multitarget.py --target target_binary
  python train_xgboost_multitarget.py --target target_5day_return
  python train_xgboost_multitarget.py --target target_10day_return
"""

import pandas as pd
import numpy as np
import pickle
import time
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True,
                    help='Target column name (e.g., target_binary, target_5day_return)')
parser.add_argument('--n_iter', type=int, default=20,
                    help='Number of hyperparameter iterations per window')
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD, AUDUSD, USDJPY)')
args = parser.parse_args()

TARGET_COLUMN = args.target
N_ITER = args.n_iter
CURRENCY_PAIR = args.pair.upper()

# Determine if classification or regression
IS_BINARY = 'binary' in TARGET_COLUMN.lower()

print(f"Training XGBoost with target: {TARGET_COLUMN}")
print(f"Currency Pair: {CURRENCY_PAIR}")
print(f"Task type: {'CLASSIFICATION' if IS_BINARY else 'REGRESSION'}")
print("="*70)

# Load data
print("\nLoading data with alternative targets...")
df = pd.read_csv(f'data/{CURRENCY_PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Drop rows where technical features are NaN
# (Don't drop for macro features since we're not using them anymore)
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]
df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])

print(f"Data loaded: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

FEATURE_COLS = [
    # Technical indicators only (reverted from failed macro experiment)
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# XGBoost hyperparameter space
if IS_BINARY:
    # Classification parameters
    param_grid = {
        'n_estimators': [125, 250],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [8, 10, 12, 14],
        'gamma': [0, 0.001, 0.002],
        'scale_pos_weight': [1.0]  # Adjust if imbalanced
    }
else:
    # Regression parameters (same as before)
    param_grid = {
        'n_estimators': [125, 250],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [8, 10, 12, 14],
        'gamma': [0, 0.001, 0.002]
    }

print(f"\nXGBoost Hyperparameter Grid:")
print(f"  n_estimators: {param_grid['n_estimators']}")
print(f"  learning_rate: {param_grid['learning_rate']}")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  gamma: {param_grid['gamma']}")
print(f"  Total combinations: {len(list(ParameterGrid(param_grid)))}")


def generate_windows(df, window_size, roll_days):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'window_id': len(windows),
            'train_start': start_idx,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': test_end,
            'date_start': df.index[start_idx],
            'date_end': df.index[test_end - 1]
        }

        windows.append(window)
        start_idx += roll_days

    return windows


def prepare_window_data(df, window, feature_cols, target_col):
    """Prepare train, validation, and test data for one window."""
    # Extract data
    train_data = df.iloc[window['train_start']:window['train_end']]
    val_data = df.iloc[window['val_start']:window['val_end']]
    test_data = df.iloc[window['test_start']:window['test_end']]

    # Fit scaler on training data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    X_val = scaler.transform(val_data[feature_cols])
    X_test = scaler.transform(test_data[feature_cols])

    # Get targets
    y_train = train_data[target_col].values
    y_val = val_data[target_col].values
    y_test = test_data[target_col].values

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler
    }


def train_xgboost(X_train, y_train, X_val, y_val, params, is_binary):
    """Train XGBoost model with given parameters."""
    if is_binary:
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            gamma=params['gamma'],
            scale_pos_weight=params.get('scale_pos_weight', 1.0),
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            gamma=params['gamma'],
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Calculate validation metric
    if is_binary:
        # Use log loss for classification
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_metric = -np.mean(y_val * np.log(y_val_pred_proba + 1e-15) +
                              (1 - y_val) * np.log(1 - y_val_pred_proba + 1e-15))
    else:
        # Use MAE for regression
        y_val_pred = model.predict(X_val)
        val_metric = np.mean(np.abs(y_val - y_val_pred))

    return model, val_metric


def randomized_search_xgboost(X_train, y_train, X_val, y_val, n_iter, is_binary):
    """Randomized search over XGBoost hyperparameters."""
    all_params = list(ParameterGrid(param_grid))

    # Randomly sample n_iter combinations
    if len(all_params) > n_iter:
        indices = np.random.choice(len(all_params), n_iter, replace=False)
        param_combinations = [all_params[i] for i in indices]
    else:
        param_combinations = all_params

    best_metric = float('inf')
    best_params = None
    best_model = None

    print(f"Starting randomized search with {len(param_combinations)} iterations...")

    for i, params in enumerate(param_combinations):
        print(f"\nIteration {i+1}/{len(param_combinations)}")
        print(f"  Params: {params}")

        try:
            model, val_metric = train_xgboost(X_train, y_train, X_val, y_val,
                                              params, is_binary)

            metric_name = "Log Loss" if is_binary else "MAE"
            print(f"  Val {metric_name}: {val_metric:.6f}")

            if val_metric < best_metric:
                best_metric = val_metric
                best_params = params
                best_model = model
                print(f"  --> New best!")

        except Exception as e:
            print(f"  Error: {str(e)}")
            continue

    print(f"\n=== Search Complete ===")
    print(f"Best metric: {best_metric:.6f}")
    print(f"Best params: {best_params}")

    return best_model, best_params, best_metric


# Generate windows
print("\nGenerating windows...")
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
print(f"Generated {len(windows)} windows")

# Load existing results if they exist
import os
results_file = f'xgboost_results_{CURRENCY_PAIR}_{TARGET_COLUMN}_checkpoint.pkl'
if os.path.exists(results_file):
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    print(f"\nLoaded {len(all_results)} existing results. Resuming from window {len(all_results) + 1}...")
else:
    all_results = []
    print("\nStarting fresh training...")

# Training loop
n_windows = len(windows)
start_window = len(all_results)

print(f"\nTraining XGBoost on windows {start_window + 1} to {n_windows}...")
print(f"Estimated time: ~{(n_windows - start_window) * N_ITER * 0.5} minutes")
print("="*70)

for window_idx in range(start_window, n_windows):
    window = windows[window_idx]
    print(f"\n{'='*70}")
    print(f"WINDOW {window_idx + 1}/{n_windows}")
    print(f"Date range: {window['date_start']} to {window['date_end']}")
    print(f"{'='*70}")

    start_time = time.time()

    # Prepare data
    window_data = prepare_window_data(df_clean, window, FEATURE_COLS, TARGET_COLUMN)

    X_train = window_data['X_train']
    y_train = window_data['y_train']
    X_val = window_data['X_val']
    y_val = window_data['y_val']
    X_test = window_data['X_test']
    y_test = window_data['y_test']

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # Hyperparameter search
    print(f"\nStarting hyperparameter search ({N_ITER} iterations)...")
    best_model, best_params, best_metric = randomized_search_xgboost(
        X_train, y_train, X_val, y_val, N_ITER, IS_BINARY
    )

    # Make predictions on test set
    if IS_BINARY:
        y_pred = best_model.predict_proba(X_test)[:, 1]  # Probability of class 1
        y_pred_class = best_model.predict(X_test)
        test_accuracy = np.mean(y_test == y_pred_class)
        test_metric = test_accuracy
    else:
        y_pred = best_model.predict(X_test)
        test_metric = np.mean(np.abs(y_test - y_pred))

    # Store results
    result = {
        'window_id': window_idx,
        'date_start': str(window['date_start']),
        'date_end': str(window['date_end']),
        'best_params': best_params,
        'val_metric': best_metric,
        'test_metric': test_metric,
        'predictions': y_pred.tolist(),
        'actuals': y_test.tolist(),
        'target_type': 'binary' if IS_BINARY else 'regression'
    }
    all_results.append(result)

    elapsed = time.time() - start_time
    metric_name = "Accuracy" if IS_BINARY else "MAE"
    print(f"\nWindow {window_idx + 1} complete in {elapsed/60:.1f} minutes")
    print(f"  Val metric:  {best_metric:.6f}")
    print(f"  Test {metric_name}: {test_metric:.6f}")

    # Save checkpoint
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Checkpoint saved: {len(all_results)} windows complete")

print(f"\n{'='*70}")
print(f"ALL WINDOWS COMPLETE")
print(f"{'='*70}")

# Save final results
final_file = f'xgboost_results_{CURRENCY_PAIR}_{TARGET_COLUMN}.pkl'
with open(final_file, 'wb') as f:
    pickle.dump(all_results, f)

# Save summary
import json
results_summary = []
for r in all_results:
    summary = {k: v for k, v in r.items() if k not in ['predictions', 'actuals']}
    results_summary.append(summary)

summary_file = f'xgboost_results_{CURRENCY_PAIR}_{TARGET_COLUMN}_summary.json'
with open(summary_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nResults saved:")
print(f"  {final_file} (full results with predictions)")
print(f"  {summary_file} (summary without predictions)")

# Display average performance
avg_val = np.mean([r['val_metric'] for r in all_results])
avg_test = np.mean([r['test_metric'] for r in all_results])

metric_name = "Accuracy" if IS_BINARY else "MAE"
print(f"\nAverage Performance Across All Windows:")
print(f"  Validation: {avg_val:.6f}")
print(f"  Test {metric_name}: {avg_test:.6f}")
