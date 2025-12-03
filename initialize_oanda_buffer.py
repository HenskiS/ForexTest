"""
Initialize prediction buffer with OANDA data for production readiness.

This script:
1. Fetches historical data from OANDA (if not cached)
2. Generates predictions for last 200 days using rolling daily retraining
3. Saves prediction buffer for production use
4. Shows what the current signal would be with the buffer

This ensures the production trader can use percentile-based thresholds immediately.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
import os
import sys
from oanda_data_fetcher import OandaDataFetcher
from trading.config import TradingConfig
from trading.market_utils import calculate_technical_features

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Asset to initialize')
parser.add_argument('--live', action='store_true', help='Use LIVE account for OANDA data')
parser.add_argument('--buffer-days', type=int, default=TradingConfig.PREDICTION_BUFFER_SIZE,
                    help='Number of days to generate predictions for')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'  # Production uses 1-day forward return
TRAIN_WINDOW_SIZE = TradingConfig.TRAIN_WINDOW_SIZE  # 378 days
BUFFER_SIZE = args.buffer_days

print(f"Initializing Prediction Buffer with OANDA Data - {PAIR}")
print("="*80)
print(f"Fetching from: {'LIVE' if args.live else 'PRACTICE'} account")
print(f"Training window: {TRAIN_WINDOW_SIZE} days")
print(f"Buffer size: {BUFFER_SIZE} days")
print("="*80)

# Step 1: Load OANDA data
print("\nStep 1: Loading OANDA historical data...")
print("-"*80)

raw_file = f'data/{PAIR}_1day_oanda.csv'
if os.path.exists(raw_file):
    print(f"Loading cached data from {raw_file}")
    df_raw = pd.read_csv(raw_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
else:
    print(f"Cached data not found, fetching from OANDA...")
    fetcher = OandaDataFetcher(practice=not args.live)

    # Fetch enough data for buffer initialization (need at least TRAIN_WINDOW_SIZE + BUFFER_SIZE)
    required_candles = TRAIN_WINDOW_SIZE + BUFFER_SIZE + 300  # Extra buffer for indicator warmup
    print(f"Fetching {PAIR} data: {required_candles} D candles...")

    df_raw = fetcher.get_historical_data(PAIR, count=required_candles, granularity='D')

    if df_raw is None or df_raw.empty:
        print(f"ERROR: Failed to fetch data for {PAIR}")
        print("Check that:")
        print("  - Your OANDA API key is valid")
        print("  - Your account has access to this instrument")
        print("  - The instrument name is correct")
        sys.exit(1)

    # Save for future use
    os.makedirs('data', exist_ok=True)
    df_raw.to_csv(raw_file, index=False)
    print(f"Saved data to {raw_file}")

print(f"Loaded {len(df_raw)} days of data")
print(f"Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

# Step 2: Calculate technical features using production code
print("\nStep 2: Calculating technical features...")
print("-"*80)

df = df_raw.copy()
df = df.set_index('date')

# Use production feature calculation
df_with_features = calculate_technical_features(df)

# Calculate target (1-day forward return) - same as production
df_with_features[TARGET] = df_with_features['close'].pct_change(1).shift(-1)

# Clean data - drop rows with NaN in features
df_clean = df_with_features.dropna(subset=TradingConfig.TECHNICAL_FEATURES + [TARGET])

print(f"Clean data: {len(df_clean)} days ({df_clean.index.min()} to {df_clean.index.max()})")

# Save engineered data
engineered_file = f'data/{PAIR}_oanda_engineered.csv'
df_clean.to_csv(engineered_file)
print(f"Engineered data saved to: {engineered_file}")

# Step 3: Load hyperparameters
print("\nStep 3: Loading hyperparameters...")
print("-"*80)

hyperparam_file = f'hyperparams_rolling_daily_{PAIR}.pkl'
if os.path.exists(hyperparam_file):
    with open(hyperparam_file, 'rb') as f:
        best_params = pickle.load(f)
    print(f"Loaded optimized hyperparameters from {hyperparam_file}")
    print(f"  {best_params}")
else:
    # Use centralized config parameters
    best_params = TradingConfig.XGBOOST_PARAMS.copy()
    print(f"Using production hyperparameters from TradingConfig")
    print(f"  {best_params}")

# Step 4: Generate predictions for last N days
print(f"\nStep 4: Generating predictions for last {BUFFER_SIZE} days...")
print("-"*80)
print("This will take a few minutes...")

prediction_buffer = []
prediction_dates = []
actual_returns = []

# Get the last BUFFER_SIZE days
if len(df_clean) < TRAIN_WINDOW_SIZE + BUFFER_SIZE:
    print(f"ERROR: Need at least {TRAIN_WINDOW_SIZE + BUFFER_SIZE} days")
    print(f"Have: {len(df_clean)} days")
    sys.exit(1)

# Generate predictions for the last BUFFER_SIZE days
start_idx = len(df_clean) - BUFFER_SIZE

for i in tqdm(range(start_idx, len(df_clean)), desc="Generating predictions"):
    current_date = df_clean.index[i]

    # Check if we have enough history
    if i < TRAIN_WINDOW_SIZE:
        continue

    # Get rolling TRAIN_WINDOW_SIZE-day window ending just before current day
    train_end_idx = i
    train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
    train_data = df_clean.iloc[train_start_idx:train_end_idx]

    # Prepare training data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[TradingConfig.TECHNICAL_FEATURES])
    y_train = train_data[TARGET].values

    # Train model
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)

    # Make prediction for today
    X_today = scaler.transform(df_clean.iloc[[i]][TradingConfig.TECHNICAL_FEATURES])
    y_pred = model.predict(X_today)[0]
    y_actual = df_clean.iloc[i][TARGET]

    prediction_buffer.append(y_pred)
    prediction_dates.append(current_date)
    actual_returns.append(y_actual)

print(f"\nGenerated {len(prediction_buffer)} predictions")
print(f"Date range: {prediction_dates[0]} to {prediction_dates[-1]}")

# Step 5: Save prediction buffer
print("\nStep 5: Saving prediction buffer...")
print("-"*80)

os.makedirs('data/oanda_cache', exist_ok=True)
buffer_file = f'data/oanda_cache/prediction_buffer_{PAIR}.pkl'

with open(buffer_file, 'wb') as f:
    pickle.dump(prediction_buffer, f)

print(f"Prediction buffer saved to: {buffer_file}")
print(f"Buffer size: {len(prediction_buffer)} predictions")

# Step 6: Show statistics
print("\nStep 6: Buffer Statistics")
print("-"*80)

predictions_array = np.array(prediction_buffer)
actuals_array = np.array(actual_returns)

print(f"Prediction range: {predictions_array.min():.6f} to {predictions_array.max():.6f}")
print(f"Prediction mean: {predictions_array.mean():.6f}")
print(f"Prediction std: {predictions_array.std():.6f}")

# Only calculate correlation if we have valid actuals (not NaN)
valid_mask = ~np.isnan(actuals_array)
if valid_mask.sum() > 0:
    corr = np.corrcoef(predictions_array[valid_mask], actuals_array[valid_mask])[0,1]
    print(f"Correlation with actual returns: {corr:.4f}")

# Calculate percentile thresholds
lower_pct = TradingConfig.PERCENTILE_LOWER
upper_pct = TradingConfig.PERCENTILE_UPPER
lower_threshold = np.percentile(predictions_array, lower_pct)
upper_threshold = np.percentile(predictions_array, upper_pct)

print(f"\nPercentile Thresholds:")
print(f"  {lower_pct}th percentile (SHORT): {lower_threshold:.6f}")
print(f"  {upper_pct}th percentile (LONG): {upper_threshold:.6f}")

# Show signal distribution
signals = np.zeros(len(predictions_array))
signals[predictions_array >= upper_threshold] = 1
signals[predictions_array <= lower_threshold] = -1

n_long = np.sum(signals == 1)
n_short = np.sum(signals == -1)
n_hold = np.sum(signals == 0)

print(f"\nSignal Distribution:")
print(f"  Long: {n_long} ({n_long/len(signals)*100:.1f}%)")
print(f"  Short: {n_short} ({n_short/len(signals)*100:.1f}%)")
print(f"  Hold: {n_hold} ({n_hold/len(signals)*100:.1f}%)")

# Step 7: Show what current signal would be
print("\nStep 7: Current Signal (with buffer)")
print("-"*80)

latest_prediction = prediction_buffer[-1]
print(f"Latest prediction: {latest_prediction:.6f}")

if latest_prediction >= upper_threshold:
    signal = 1
    signal_str = "LONG"
elif latest_prediction <= lower_threshold:
    signal = -1
    signal_str = "SHORT"
else:
    signal = 0
    signal_str = "HOLD"

print(f"Signal: {signal} ({signal_str})")

print("\n" + "="*80)
print("SUCCESS: Buffer initialized successfully!")
print("="*80)
print(f"\nYou can now run the production trader:")
print(f"  venv/Scripts/python.exe oanda_14_asset_trader.py --live --dry-run --yes")
print()
