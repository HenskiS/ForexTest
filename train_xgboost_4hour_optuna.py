"""
Train XGBoost on 4-hour forex data with Optuna hyperparameter optimization.

This script uses Optuna's Bayesian optimization for smarter hyperparameter search
compared to random/grid search. Optuna learns from previous trials to suggest
better parameters and can prune unpromising trials early.

Usage:
    python train_xgboost_4hour_optuna.py --pair EURUSD --n-trials 100
    python train_xgboost_4hour_optuna.py --pair GBPUSD --n-trials 50 --timeout 3600
"""

import pandas as pd
import numpy as np
import pickle
import time
import argparse
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import optuna
# from optuna.pruners import MedianPruner  # Not needed with simplified approach

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD)')
parser.add_argument('--n-trials', type=int, default=100,
                    help='Number of Optuna trials per window (default: 100)')
parser.add_argument('--timeout', type=int, default=None,
                    help='Timeout in seconds for optimization per window')
parser.add_argument('--output-suffix', type=str, default='optuna',
                    help='Suffix for output files (default: optuna)')
parser.add_argument('--max-windows', type=int, default=None,
                    help='Maximum number of windows to train (None = all windows). Use last N windows.')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()
N_TRIALS = args.n_trials
TIMEOUT = args.timeout
OUTPUT_SUFFIX = args.output_suffix
MAX_WINDOWS = args.max_windows
TARGET_COLUMN = 'target_1day_return'

print(f"Training XGBoost on 4-hour {CURRENCY_PAIR} data with Optuna")
print(f"Target: {TARGET_COLUMN}")
print(f"Trials per window: {N_TRIALS}")
print(f"Timeout per window: {TIMEOUT if TIMEOUT else 'None'}")
print("="*70)

# Load 4-hour data
print("\nLoading 4-hour data...")
df = pd.read_csv(f'data/{CURRENCY_PAIR}_4hour_with_features.csv',
                 index_col='date', parse_dates=True)

# Technical features (same 26 as daily model)
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# Convert target to percentage scale
print(f"\nTarget before scaling - Mean: {df[TARGET_COLUMN].mean():.6f}, Std: {df[TARGET_COLUMN].std():.6f}")
df[TARGET_COLUMN] = df[TARGET_COLUMN] * 100
print(f"Target after scaling (%) - Mean: {df[TARGET_COLUMN].mean():.6f}%, Std: {df[TARGET_COLUMN].std():.6f}%")

# Drop rows with NaN
df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])

print(f"Data loaded: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
print(f"Total 4-hour bars: {len(df_clean)}")

# Configuration - BAR-BASED
TRAIN_BARS = 1800
VAL_BARS = 468
TEST_BARS = 378
ROLL_BARS = 378
WINDOW_SIZE = TRAIN_BARS + VAL_BARS + TEST_BARS

print(f"\nWalk-Forward Configuration (BAR-BASED):")
print(f"  Training bars:   {TRAIN_BARS}")
print(f"  Validation bars: {VAL_BARS}")
print(f"  Test bars:       {TEST_BARS}")
print(f"  Roll forward:    {ROLL_BARS} bars")
print(f"  Total window:    {WINDOW_SIZE} bars")

FEATURE_COLS = technical_features


def generate_windows(df, window_size, roll_bars):
    """Generate rolling walk-forward windows (bar-based)."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_BARS
        val_end = train_end + VAL_BARS
        test_end = val_end + TEST_BARS

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
        start_idx += roll_bars

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


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost hyperparameter optimization."""

    # Suggest hyperparameters (aggressive ranges to prevent underfitting)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.001, 0.02),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    # Train model
    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train, verbose=False)

    # Calculate validation MAE
    y_val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(y_val - y_val_pred))

    return val_mae


def optimize_window(X_train, y_train, X_val, y_val, window_id, n_trials, timeout):
    """Optimize hyperparameters for one window using Optuna."""

    # Create study (pruning disabled due to XGBoost API compatibility)
    study = optuna.create_study(direction='minimize')

    # Optimize
    print(f"\nStarting Optuna optimization (target: {n_trials} trials, timeout: {timeout}s)...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False
    )

    # Get best parameters
    best_params = study.best_params
    best_val_mae = study.best_value

    print(f"\n=== Optuna Results ===")
    print(f"Best validation MAE: {best_val_mae:.6f}")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best params: {best_params}")

    # Train final model with best parameters
    final_params = {**best_params, 'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1}
    best_model = xgb.XGBRegressor(**final_params)
    best_model.fit(X_train, y_train, verbose=False)

    return best_model, best_params, best_val_mae


# Main training loop
print("\n" + "="*70)
print("WALK-FORWARD TRAINING WITH OPTUNA")
print("="*70)

# Generate windows
all_windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_BARS)

# Limit to last N windows if specified
if MAX_WINDOWS is not None and MAX_WINDOWS < len(all_windows):
    windows = all_windows[-MAX_WINDOWS:]
    print(f"\nGenerated {len(all_windows)} total windows, using LAST {len(windows)} windows")
else:
    windows = all_windows
    print(f"\nGenerated {len(windows)} walk-forward windows")

print(f"Each window: {TRAIN_BARS} train + {VAL_BARS} val + {TEST_BARS} test bars")

# Store results
results = []
start_time = time.time()

for window in windows:
    print(f"\n{'='*70}")
    print(f"WINDOW {window['window_id'] + 1}/{len(windows)}")
    print(f"{'='*70}")
    print(f"Train: {window['date_start']} to {df_clean.index[window['train_end']-1]}")
    print(f"Val:   {df_clean.index[window['val_start']]} to {df_clean.index[window['val_end']-1]}")
    print(f"Test:  {df_clean.index[window['test_start']]} to {window['date_end']}")

    # Prepare data
    data = prepare_window_data(df_clean, window, FEATURE_COLS, TARGET_COLUMN)

    # Optimize with Optuna
    best_model, best_params, val_mae = optimize_window(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        window['window_id'],
        N_TRIALS,
        TIMEOUT
    )

    # Test predictions
    y_test_pred = best_model.predict(data['X_test'])
    test_mae = np.mean(np.abs(data['y_test'] - y_test_pred))

    print(f"\n=== Window {window['window_id'] + 1} Final Results ===")
    print(f"Val MAE:  {val_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # Store results
    test_data = df_clean.iloc[window['test_start']:window['test_end']]
    for i, (idx, row) in enumerate(test_data.iterrows()):
        results.append({
            'date': idx,
            'window_id': window['window_id'],
            'actual_return': data['y_test'][i],
            'predicted_return': y_test_pred[i],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            **{f'param_{k}': v for k, v in best_params.items()}
        })

    elapsed = time.time() - start_time
    avg_time = elapsed / (window['window_id'] + 1)
    remaining = avg_time * (len(windows) - window['window_id'] - 1)
    print(f"\nProgress: {window['window_id'] + 1}/{len(windows)} windows")
    print(f"Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")

# Save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('date')

output_file = f'xgboost_results_{CURRENCY_PAIR}_4hour_{TARGET_COLUMN}_{OUTPUT_SUFFIX}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results_df, f)

print(f"\nResults saved to: {output_file}")
print(f"Total predictions: {len(results_df)}")
print(f"Date range: {results_df.index.min()} to {results_df.index.max()}")
print(f"\nTotal training time: {(time.time() - start_time)/60:.1f} minutes")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nNext step: Backtest with:")
print(f"  python backtest_advanced_exits_4hour.py --pair {CURRENCY_PAIR} --target {TARGET_COLUMN} --model-suffix {OUTPUT_SUFFIX}")
