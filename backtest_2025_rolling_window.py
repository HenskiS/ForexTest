"""
Test 2025 performance with rolling window retraining.

Uses a FIXED 756-day training window (train+val size) that slides
forward each day/week, always using the most recent 756 days.

This matches production deployment where you keep a fixed lookback period.
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD, USDJPY, AUDUSD)')
parser.add_argument('--frequency', type=str, default='daily', choices=['daily', 'weekly'],
                    help='Retraining frequency (daily or weekly)')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
FREQUENCY = args.frequency
RETRAIN_DAYS = 1 if FREQUENCY == 'daily' else 7

# Training window size (matches train+val from walk-forward)
TRAIN_WINDOW_SIZE = 756  # 600 train + 156 val

print(f"Testing 2025 with ROLLING WINDOW ({FREQUENCY} retraining)")
print(f"Currency Pair: {PAIR}")
print(f"Training Window: {TRAIN_WINDOW_SIZE} days (fixed)")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Technical features (same as training)
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

# Get 2025 data
df_2025 = df_clean[df_clean.index.year == 2025]
print(f"\n2025 data: {len(df_2025)} days ({df_2025.index.min()} to {df_2025.index.max()})")

# Get the best hyperparameters from the most recent training window
try:
    with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
        all_results = pickle.load(f)
    best_params = all_results[-1]['best_params']
    print(f"\nUsing hyperparameters from window {len(all_results)}:")
    print(f"  {best_params}")
except:
    best_params = {
        'n_estimators': 250,
        'learning_rate': 0.05,
        'max_depth': 10,
        'gamma': 0.001
    }
    print(f"\nUsing default hyperparameters:")
    print(f"  {best_params}")

# Get the index where 2025 starts
start_2025_idx = df_clean.index.get_loc(df_2025.index[0])
print(f"\n2025 starts at index {start_2025_idx}")
print(f"Training window: {TRAIN_WINDOW_SIZE} days")
print(f"Minimum data needed: {TRAIN_WINDOW_SIZE} days before first prediction")

if start_2025_idx < TRAIN_WINDOW_SIZE:
    print(f"\nERROR: Not enough historical data!")
    print(f"Need {TRAIN_WINDOW_SIZE} days, have {start_2025_idx}")
    exit(1)

# Rolling window retraining and prediction
print(f"\nStarting rolling window retraining ({FREQUENCY})...")
print("="*70)

predictions = []
actuals = []
dates = []

model = None
scaler = None
last_train_day = -999

for i, date in enumerate(df_2025.index):
    current_idx = df_clean.index.get_loc(date)

    # Check if we need to retrain
    days_since_train = i - last_train_day
    need_retrain = (model is None) or (days_since_train >= RETRAIN_DAYS)

    if need_retrain:
        if i % 10 == 0 or need_retrain:
            print(f"Day {i+1}/{len(df_2025)}: {date.date()}{' - Retraining' if need_retrain else ''}")

        # Get FIXED window of last 756 days (ending just before current day)
        train_end_idx = current_idx
        train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE

        train_data = df_clean.iloc[train_start_idx:train_end_idx]

        if len(train_data) != TRAIN_WINDOW_SIZE:
            print(f"Warning: Train data size {len(train_data)} != {TRAIN_WINDOW_SIZE}")
            continue

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
        last_train_day = i

    # Make prediction for today
    X_today = scaler.transform(df_clean.loc[[date]][technical_features])
    y_pred = model.predict(X_today)[0]
    y_actual = df_clean.loc[date, TARGET]

    predictions.append(y_pred)
    actuals.append(y_actual)
    dates.append(date)

n_retrains = len([i for i in range(len(df_2025)) if i % RETRAIN_DAYS == 0])
print(f"\nCompleted {len(predictions)} predictions with {n_retrains} model retrainings")
print(f"Average {len(predictions) / max(n_retrains, 1):.1f} predictions per model")

# Save results
results = {
    'dates': [str(d) for d in dates],
    'predictions': predictions,
    'actuals': actuals,
    'pair': PAIR,
    'best_params': best_params,
    'retrain_frequency': RETRAIN_DAYS,
    'train_window_size': TRAIN_WINDOW_SIZE,
    'method': 'rolling_window'
}

output_file = f'rolling_{FREQUENCY}_2025_{PAIR}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: {output_file}")

# Calculate basic metrics
predictions = np.array(predictions)
actuals = np.array(actuals)

mae = np.mean(np.abs(actuals - predictions))
correlation = np.corrcoef(predictions, actuals)[0, 1]

print(f"\n2025 Prediction Quality:")
print(f"  MAE: {mae:.6f}")
print(f"  Correlation: {correlation:.4f}")

# Generate trading signals
long_threshold = np.percentile(predictions, 48)
short_threshold = np.percentile(predictions, 52)

signals = np.zeros(len(predictions))
signals[predictions >= short_threshold] = 1   # Long
signals[predictions <= long_threshold] = -1   # Short

n_long = np.sum(signals == 1)
n_short = np.sum(signals == -1)
n_hold = np.sum(signals == 0)

print(f"\nSignal Distribution:")
print(f"  Long:  {n_long} ({100*n_long/len(signals):.1f}%)")
print(f"  Short: {n_short} ({100*n_short/len(signals):.1f}%)")
print(f"  Hold:  {n_hold} ({100*n_hold/len(signals):.1f}%)")

# Save signals
results['signals'] = signals.tolist()
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'='*70}")
print(f"Rolling window {FREQUENCY} retraining complete!")
print(f"Fixed window size: {TRAIN_WINDOW_SIZE} days")
print(f"Total retrainings: {n_retrains}")
print(f"{'='*70}")
