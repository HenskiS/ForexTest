"""
Train XGBoost model for forex trading using parameters from the paper.

XGBoost Hyperparameters (Daily frequency):
- n_estimators: [125, 250]
- learning_rate: [0, 0.1]
- max_depth: [8, 14]
- gamma: [0, 0.002]
"""

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

# Load data (FIXED - no data leakage!)
print("Loading data (FIXED - features shifted forward)...")
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED.csv', index_col='date', parse_dates=True)
df_clean = df.dropna()

print(f"Data loaded: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

TARGET_COLUMN = 'simple_return'
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# XGBoost hyperparameter space from paper (Table 5, Daily)
param_grid = {
    'n_estimators': [125, 250],
    'learning_rate': [0.01, 0.05, 0.1],  # Range [0, 0.1]
    'max_depth': [8, 10, 12, 14],  # Range [8, 14]
    'gamma': [0, 0.001, 0.002]  # Range [0, 0.002]
}

print(f"\nXGBoost Hyperparameter Grid:")
print(f"  n_estimators: {param_grid['n_estimators']}")
print(f"  learning_rate: {param_grid['learning_rate']}")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  gamma: {param_grid['gamma']}")
print(f"  Total combinations: {len(list(ParameterGrid(param_grid)))}")


def generate_windows(df, window_size, roll_days, min_windows=40):
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

        if len(windows) >= min_windows:
            break

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


def train_xgboost(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model with given parameters."""
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

    # Calculate validation MAE
    y_val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(y_val - y_val_pred))

    return model, val_mae


def randomized_search_xgboost(X_train, y_train, X_val, y_val, n_iter=20):
    """
    Randomized search over XGBoost hyperparameters.
    Sample n_iter random combinations from the grid.
    """
    all_params = list(ParameterGrid(param_grid))

    # Randomly sample n_iter combinations
    if len(all_params) > n_iter:
        indices = np.random.choice(len(all_params), n_iter, replace=False)
        param_combinations = [all_params[i] for i in indices]
    else:
        param_combinations = all_params

    best_mae = float('inf')
    best_params = None
    best_model = None

    print(f"Starting randomized search with {len(param_combinations)} iterations...")

    for i, params in enumerate(param_combinations):
        print(f"\nIteration {i+1}/{len(param_combinations)}")
        print(f"  Params: {params}")

        try:
            model, val_mae = train_xgboost(X_train, y_train, X_val, y_val, params)

            print(f"  Val MAE: {val_mae:.6f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_params = params
                best_model = model
                print(f"  --> New best!")

        except Exception as e:
            print(f"  Error: {str(e)}")
            continue

    print(f"\n=== Search Complete ===")
    print(f"Best MAE: {best_mae:.6f}")
    print(f"Best params: {best_params}")

    return best_model, best_params, best_mae


# Generate windows
print("\nGenerating windows...")
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=40)
print(f"Generated {len(windows)} windows")

# Load existing results if they exist
import os
results_file = 'xgboost_results_FIXED_checkpoint.pkl'
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
print(f"Estimated time: ~{(n_windows - start_window) * 20 * 0.5} minutes (assuming 0.5 min per iteration)")
print("="*60)

for window_idx in range(start_window, n_windows):
    window = windows[window_idx]
    print(f"\n{'='*60}")
    print(f"WINDOW {window_idx + 1}/{n_windows}")
    print(f"Date range: {window['date_start']} to {window['date_end']}")
    print(f"{'='*60}")

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
    print(f"\nStarting hyperparameter search (20 iterations)...")
    best_model, best_params, best_mae = randomized_search_xgboost(
        X_train, y_train, X_val, y_val, n_iter=20
    )

    # Make predictions on test set
    y_pred = best_model.predict(X_test)

    # Calculate test MAE
    test_mae = np.mean(np.abs(y_test - y_pred))

    # Store results
    result = {
        'window_id': window_idx,
        'date_start': str(window['date_start']),
        'date_end': str(window['date_end']),
        'best_params': best_params,
        'val_mae': best_mae,
        'test_mae': test_mae,
        'predictions': y_pred.tolist(),
        'actuals': y_test.tolist()
    }
    all_results.append(result)

    elapsed = time.time() - start_time
    print(f"\nWindow {window_idx + 1} complete in {elapsed/60:.1f} minutes")
    print(f"  Val MAE:  {best_mae:.6f}")
    print(f"  Test MAE: {test_mae:.6f}")

    # Save checkpoint
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Checkpoint saved: {len(all_results)} windows complete")

print(f"\n{'='*60}")
print(f"ALL WINDOWS COMPLETE")
print(f"{'='*60}")

# Save final results
with open('xgboost_results_FIXED.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Save summary
import json
results_summary = []
for r in all_results:
    summary = {k: v for k, v in r.items() if k not in ['predictions', 'actuals']}
    results_summary.append(summary)

with open('xgboost_results_FIXED_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nResults saved:")
print("  xgboost_results_FIXED.pkl (full results with predictions)")
print("  xgboost_results_FIXED_summary.json (summary without predictions)")

# Display average performance
avg_val_mae = np.mean([r['val_mae'] for r in all_results])
avg_test_mae = np.mean([r['test_mae'] for r in all_results])

print(f"\nAverage Performance Across All Windows:")
print(f"  Validation MAE: {avg_val_mae:.6f}")
print(f"  Test MAE:       {avg_test_mae:.6f}")
