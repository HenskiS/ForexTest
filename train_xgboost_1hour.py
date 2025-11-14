"""
Train XGBoost on 1-hour forex data with bar-based walk-forward validation.

Usage:
    python train_xgboost_1hour.py --pair EURUSD --n_iter 20
    python train_xgboost_1hour.py --pair GBPUSD --n_iter 20
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
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD)')
parser.add_argument('--n_iter', type=int, default=20,
                    help='Number of hyperparameter iterations per window')
parser.add_argument('--max-windows', type=int, default=None,
                    help='Maximum number of windows to train (None = all windows). Use last N windows.')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()
N_ITER = args.n_iter
MAX_WINDOWS = args.max_windows
TARGET_COLUMN = 'target_12hour_return'  # Predict 6 bars ahead (1 day)

print(f"Training XGBoost on 1-hour {CURRENCY_PAIR} data")
print(f"Target: {TARGET_COLUMN}")
print("="*70)

# Load 1-hour data
print("\nLoading 1-hour data...")
df = pd.read_csv(f'data/{CURRENCY_PAIR}_1hour_with_features.csv',
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

# Convert target to percentage scale (critical for XGBoost regularization!)
print(f"\nTarget before scaling - Mean: {df[TARGET_COLUMN].mean():.6f}, Std: {df[TARGET_COLUMN].std():.6f}")
df[TARGET_COLUMN] = df[TARGET_COLUMN] * 100
print(f"Target after scaling (%) - Mean: {df[TARGET_COLUMN].mean():.6f}%, Std: {df[TARGET_COLUMN].std():.6f}%")

# Drop rows with NaN
df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])

print(f"Data loaded: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
print(f"Total 1-hour bars: {len(df_clean)}")

# Configuration - BAR-BASED (not day-based)
# Smaller windows for higher-frequency 1-hour trading (half of daily equivalent)
TRAIN_BARS = 7200  # 300 days × 6 bars/day
VAL_BARS = 1872     # 78 days × 6 bars/day
TEST_BARS = 1512    # 63 days × 6 bars/day
ROLL_BARS = 1512    # Roll forward by 378 bars (63 days)
WINDOW_SIZE = TRAIN_BARS + VAL_BARS + TEST_BARS

print(f"\nWalk-Forward Configuration (BAR-BASED):")
print(f"  Training bars:   {TRAIN_BARS}")
print(f"  Validation bars: {VAL_BARS}")
print(f"  Test bars:       {TEST_BARS}")
print(f"  Roll forward:    {ROLL_BARS} bars")
print(f"  Total window:    {WINDOW_SIZE} bars")

FEATURE_COLS = technical_features

# XGBoost hyperparameter space (adjusted for percentage-scaled targets)
# With std ~0.2%, gamma values should be larger: gamma=0.02 → 10% of std
param_grid = {
    'n_estimators': [125, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [8, 10, 12],
    'gamma': [0, 0.01, 0.02]  # Adjusted for percentage scale
}

print(f"\nXGBoost Hyperparameter Grid:")
print(f"  n_estimators: {param_grid['n_estimators']}")
print(f"  learning_rate: {param_grid['learning_rate']}")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  gamma: {param_grid['gamma']}")
print(f"  Total combinations: {len(list(ParameterGrid(param_grid)))}")


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


def train_xgboost(X_train, y_train, X_val, y_val, params):
    """Train XGBoost regression model with given parameters."""
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
    val_metric = np.mean(np.abs(y_val - y_val_pred))

    return model, val_metric


def randomized_search_xgboost(X_train, y_train, X_val, y_val, n_iter):
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
            model, val_metric = train_xgboost(X_train, y_train, X_val, y_val, params)
            print(f"  Val MAE: {val_metric:.6f}")

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


# Main training loop
print("\n" + "="*70)
print("WALK-FORWARD TRAINING")
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

    # Hyperparameter search
    best_model, best_params, val_metric = randomized_search_xgboost(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        N_ITER
    )

    # Test predictions
    y_test_pred = best_model.predict(data['X_test'])
    test_mae = np.mean(np.abs(data['y_test'] - y_test_pred))

    print(f"\n=== Window {window['window_id'] + 1} Results ===")
    print(f"Val MAE:  {val_metric:.6f}")
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

output_file = f'xgboost_results_{CURRENCY_PAIR}_1hour_target_12hour_return.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results_df, f)

print(f"\nResults saved to: {output_file}")
print(f"Total predictions: {len(results_df)}")
print(f"Date range: {results_df.index.min()} to {results_df.index.max()}")
print(f"\nTotal training time: {(time.time() - start_time)/60:.1f} minutes")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nNext step: Backtest with backtest_advanced_exits_1hour.py")
