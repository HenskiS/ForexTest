"""
Train with rolling daily window for ALL years (full walk-forward).

For each of the 46 walk-forward windows:
- For each day in the 126-day test period:
  - Train on rolling 756-day window ending yesterday
  - Make prediction for today

This is computationally expensive (~6,000 model trainings) but gives
the most realistic estimate of production performance.
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse
import time
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD, USDJPY, AUDUSD)')
parser.add_argument('--target', type=str, default='target_5day_return',
                    help='Target column')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = args.target

print(f"Rolling Daily Training for All Years")
print(f"Currency Pair: {PAIR}")
print(f"Target: {TARGET}")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Technical features
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

df_clean = df.dropna(subset=technical_features + [TARGET])
print(f"Data loaded: {len(df_clean)} days ({df_clean.index.min()} to {df_clean.index.max()})")

# Configuration (must match static window approach)
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
TRAIN_WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS  # 756 days for rolling window

# Get best hyperparameters from static approach
try:
    with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
        static_results = pickle.load(f)
    print(f"\nLoaded {len(static_results)} windows from static results")
except:
    print("\nERROR: Need static results file to get hyperparameters!")
    exit(1)

# Generate same windows as static approach
def generate_windows(df, window_size, roll_days):
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

windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
print(f"Generated {len(windows)} windows")

# Check for existing checkpoint
checkpoint_file = f'rolling_daily_results_{PAIR}_{TARGET}_checkpoint.pkl'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        all_results = pickle.load(f)
    print(f"\nResuming from checkpoint: {len(all_results)} windows complete")
    start_window = len(all_results)
else:
    all_results = []
    start_window = 0
    print(f"\nStarting fresh training...")

# Estimate time
total_test_days = (len(windows) - start_window) * TEST_DAYS
print(f"\nEstimated work:")
print(f"  Remaining windows: {len(windows) - start_window}")
print(f"  Total model trainings: ~{total_test_days}")
print(f"  Estimated time: {total_test_days * 3 / 3600:.1f} - {total_test_days * 5 / 3600:.1f} hours")
print("="*70)

overall_start_time = time.time()

# Process each window
for window_idx in range(start_window, len(windows)):
    window = windows[window_idx]
    window_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"WINDOW {window_idx + 1}/{len(windows)}")
    print(f"Date range: {window['date_start'].date()} to {window['date_end'].date()}")
    print(f"{'='*70}")

    # Get hyperparameters for this window from static results
    best_params = static_results[window_idx]['best_params']
    print(f"Using hyperparameters: {best_params}")

    # Get test period dates
    test_start_idx = window['test_start']
    test_end_idx = window['test_end']
    test_dates = df_clean.index[test_start_idx:test_end_idx]

    print(f"\nTest period: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"Days to predict: {len(test_dates)}")

    predictions = []
    actuals = []
    dates = []

    # For each day in test period, retrain with rolling window
    for i, date in enumerate(test_dates):
        if i % 20 == 0:
            print(f"  Day {i+1}/{len(test_dates)}: {date.date()}")

        current_idx = df_clean.index.get_loc(date)

        # Check if we have enough history for 756-day window
        if current_idx < TRAIN_WINDOW_SIZE:
            print(f"  Skipping {date.date()} - insufficient history")
            continue

        # Get rolling 756-day window ending just before current day
        train_end_idx = current_idx
        train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
        train_data = df_clean.iloc[train_start_idx:train_end_idx]

        # Prepare training data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_data[technical_features])
        y_train = train_data[TARGET].values

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            gamma=best_params['gamma'],
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, verbose=False)

        # Make prediction for today
        X_today = scaler.transform(df_clean.loc[[date]][technical_features])
        y_pred = model.predict(X_today)[0]
        y_actual = df_clean.loc[date, TARGET]

        predictions.append(y_pred)
        actuals.append(y_actual)
        dates.append(date)

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(actuals - predictions))

    # Store results
    result = {
        'window_id': window_idx,
        'date_start': str(window['date_start']),
        'date_end': str(window['date_end']),
        'best_params': best_params,
        'test_metric': mae,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist(),
        'dates': [str(d) for d in dates],
        'method': 'rolling_daily'
    }
    all_results.append(result)

    window_elapsed = time.time() - window_start_time
    overall_elapsed = time.time() - overall_start_time

    print(f"\nWindow {window_idx + 1} complete in {window_elapsed/60:.1f} minutes")
    print(f"  Test MAE: {mae:.6f}")
    print(f"  Predictions: {len(predictions)}")

    # Save checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Checkpoint saved: {len(all_results)}/{len(windows)} windows complete")

    # Time estimate
    windows_remaining = len(windows) - len(all_results)
    if len(all_results) > 0:
        avg_time_per_window = overall_elapsed / len(all_results)
        est_remaining_hours = (windows_remaining * avg_time_per_window) / 3600
        print(f"  Estimated time remaining: {est_remaining_hours:.1f} hours")

print(f"\n{'='*70}")
print(f"ALL WINDOWS COMPLETE")
print(f"Total time: {(time.time() - overall_start_time)/3600:.1f} hours")
print(f"{'='*70}")

# Save final results
final_file = f'rolling_daily_results_{PAIR}_{TARGET}.pkl'
with open(final_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to: {final_file}")

# Calculate average performance
avg_test = np.mean([r['test_metric'] for r in all_results])
print(f"\nAverage Test MAE: {avg_test:.6f}")

print("\nUse analyze_1day_eurusd_annual.py to analyze trading performance!")
