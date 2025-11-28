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
from datetime import datetime, timedelta
from oanda_data_fetcher import OandaDataFetcher
from trading.config import TradingConfig

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--live', action='store_true', help='Use LIVE account for OANDA data')
parser.add_argument('--buffer-days', type=int, default=200, help='Number of days to generate predictions for')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
TRAIN_WINDOW_SIZE = 756  # 600 train + 156 val
BUFFER_SIZE = args.buffer_days

print(f"Initializing Prediction Buffer with OANDA Data - {PAIR}")
print("="*80)
print(f"Fetching from: {'LIVE' if args.live else 'PRACTICE'} account")
print(f"Buffer size: {BUFFER_SIZE} days")
print("="*80)

# Step 1: Load OANDA data
print("\nStep 1: Loading OANDA historical data...")
print("-"*80)

raw_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(raw_file):
    print(f"ERROR: {raw_file} not found")
    print("Run this command first to fetch OANDA data:")
    print(f"  python oanda_data_fetcher.py --live")
    sys.exit(1)

df_raw = pd.read_csv(raw_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])

print(f"Loaded {len(df_raw)} days of data")
print(f"Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

# Step 2: Calculate technical features
print("\nStep 2: Calculating technical features...")
print("-"*80)

df = df_raw.copy()
df = df.set_index('date')

# Basic features
df['momentum'] = df['close'].pct_change()
df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['range'] = df['high'] - df['low']
df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

# EMAs
for period in [10, 20, 50, 100, 200]:
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# MACD
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# ADX
def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di

df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df['high'], df['low'], df['close'])

# RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['close'])

# Stochastic
def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])

# CCI
def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

df['cci'] = calculate_cci(df['high'], df['low'], df['close'])

# Williams %R
def calculate_williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])

# Bollinger Bands
def calculate_bollinger_bands(close, period=20, std_dev=2):
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = upper - lower
    position = (close - lower) / (upper - lower)
    return upper, middle, lower, width, position

df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'], df['bb_position'] = \
    calculate_bollinger_bands(df['close'])

# ATR
def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

df['atr'] = calculate_atr(df['high'], df['low'], df['close'])

# Target (5-day forward return)
df[TARGET] = df['close'].pct_change(5).shift(-5)

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

    # Get rolling 756-day window ending just before current day
    train_end_idx = i
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
    X_today = scaler.transform(df_clean.iloc[[i]][technical_features])
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
print(f"Correlation with actual returns: {np.corrcoef(predictions_array, actuals_array)[0,1]:.4f}")

# Calculate percentile thresholds
lower_pct = 48
upper_pct = 52
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
print("✓ Buffer initialized successfully!")
print("="*80)
print(f"\nYou can now run the production trader with percentile-based signals:")
print(f"  python oanda_production_trader.py --pair {PAIR} {'--live' if args.live else ''} --dry-run")
print()
