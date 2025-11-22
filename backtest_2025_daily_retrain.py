"""
Test 2025 performance with daily retraining.

This simulates production deployment where the model is retrained
every day with all historical data up to yesterday.
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
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'

print(f"Testing 2025 with DAILY RETRAINING")
print(f"Currency Pair: {PAIR}")
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
# Load the latest results to get typical good hyperparameters
try:
    with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
        all_results = pickle.load(f)
    # Use hyperparameters from the last window
    best_params = all_results[-1]['best_params']
    print(f"\nUsing hyperparameters from window {len(all_results)}:")
    print(f"  {best_params}")
except:
    # Default good parameters if no results file
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
print(f"\n2025 starts at index {start_2025_idx} in the full dataset")
print(f"Training data available: {start_2025_idx} days before 2025")

# Daily retraining and prediction
print(f"\nStarting daily retraining for {len(df_2025)} days...")
print("="*70)

predictions = []
actuals = []
dates = []

for i, date in enumerate(df_2025.index):
    if i % 10 == 0:
        print(f"Day {i+1}/{len(df_2025)}: {date.date()}")

    # Get all data up to (but not including) this day for training
    current_idx = df_clean.index.get_loc(date)
    train_data = df_clean.iloc[:current_idx]

    # Skip if not enough training data
    if len(train_data) < 200:
        continue

    # Prepare training data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[technical_features])
    y_train = train_data[TARGET].values

    # Train model with best hyperparameters
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

print(f"\nCompleted {len(predictions)} predictions for 2025")

# Save results
results = {
    'dates': [str(d) for d in dates],
    'predictions': predictions,
    'actuals': actuals,
    'pair': PAIR,
    'best_params': best_params
}

output_file = f'daily_retrain_2025_{PAIR}.pkl'
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

# Generate trading signals (using same logic as main backtest)
# We'll use quantile-based thresholds
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

# Save signals for backtesting
results['signals'] = signals.tolist()
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'='*70}")
print("Daily retraining complete!")
print("Use this file with your backtesting script to test trading performance.")
print(f"{'='*70}")
