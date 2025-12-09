"""
Generate recent predictions for production buffer updates
Fast version - only processes last ~200 days for buffer
Uses all available data up to latest date
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from trading.market_utils import calculate_technical_features

print("="*100)
print("GENERATING RECENT PREDICTIONS FOR PRODUCTION BUFFERS")
print("="*100)
print()

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

# Optimized hyperparameters
HIDDEN_LAYERS = (13, 20, 31)
ACTIVATION = 'tanh'
SOLVER = 'sgd'
LEARNING_RATE = 0.001
MOMENTUM = 0.4
BATCH_SIZE = 64
MAX_ITER = 20
ALPHA = 0.0001

# Buffer settings
TRAIN_WINDOW = 378
BUFFER_SIZE = 200
OUTPUT_DIR = 'data/oanda_cache'

# Feature set (31 features - optimized)
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'atr', 'volume_sma',
    'close_to_high', 'close_to_low',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_predict_recent(pair):
    """
    Train model and generate predictions for last BUFFER_SIZE days

    Args:
        pair: Currency pair (e.g., 'EURUSD')

    Returns:
        List of predictions (length = BUFFER_SIZE)
    """
    print(f"\n{pair}:")
    print("-" * 100)

    # Load all available data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    print(f"  Total data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Latest date: {df.index[-1].date()} (should be 2025-12-08)")

    # Calculate features
    df = calculate_technical_features(df)

    # Remove NaN rows
    df_clean = df.dropna()
    print(f"  Clean data: {len(df_clean)} days (after removing NaN)")

    # We need TRAIN_WINDOW + BUFFER_SIZE days
    required_days = TRAIN_WINDOW + BUFFER_SIZE
    if len(df_clean) < required_days:
        print(f"  ERROR: Need {required_days} days, only have {len(df_clean)}")
        return None

    # Use only the recent data we need
    df_recent = df_clean.iloc[-required_days:].copy()
    print(f"  Using last {len(df_recent)} days for training + predictions")

    predictions = []

    # Generate predictions for last BUFFER_SIZE days
    # For each prediction day, train on the TRAIN_WINDOW days before it
    for i in range(BUFFER_SIZE):
        # The prediction day index in df_recent
        pred_idx = TRAIN_WINDOW + i

        # Training data: TRAIN_WINDOW days before pred_idx
        train_start = i
        train_end = pred_idx

        df_train = df_recent.iloc[train_start:train_end]

        # Prepare training data
        X_train = df_train[FEATURE_COLS].values
        y_train = df_train['target_1day_return'].values

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation=ACTIVATION,
            solver=SOLVER,
            learning_rate_init=LEARNING_RATE,
            momentum=MOMENTUM,
            batch_size=BATCH_SIZE,
            max_iter=MAX_ITER,
            alpha=ALPHA,
            random_state=42,
            early_stopping=False
        )

        model.fit(X_train_scaled, y_train)

        # Get features for prediction day
        X_pred = df_recent.iloc[pred_idx][FEATURE_COLS].values.reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)

        # Generate prediction
        prediction = model.predict(X_pred_scaled)[0]
        predictions.append(prediction)

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{BUFFER_SIZE} predictions...")

    print(f"  SUCCESS: Generated {len(predictions)} predictions")
    print(f"  Prediction range: [{min(predictions):.6f}, {max(predictions):.6f}]")

    return predictions

# Process all pairs
print("\nProcessing all pairs...")
print("="*100)

buffers = {}

for pair in PAIRS:
    predictions = train_and_predict_recent(pair)

    if predictions is not None:
        buffers[pair] = predictions

        # Save buffer to pickle file
        buffer_file = os.path.join(OUTPUT_DIR, f'prediction_buffer_ann_{pair}.pkl')
        with open(buffer_file, 'wb') as f:
            pickle.dump(predictions, f)

        print(f"  Saved buffer to: {buffer_file}")

# Summary
print("\n" + "="*100)
print("SUMMARY")
print("="*100)

for pair in PAIRS:
    if pair in buffers:
        buffer = buffers[pair]
        print(f"\n{pair}:")
        print(f"  Buffer size: {len(buffer)}")
        print(f"  Min prediction: {min(buffer):.6f}")
        print(f"  Max prediction: {max(buffer):.6f}")
        print(f"  Mean prediction: {np.mean(buffer):.6f}")
        print(f"  Std prediction: {np.std(buffer):.6f}")
        print(f"  File: data/oanda_cache/prediction_buffer_ann_{pair}.pkl")

print("\n" + "="*100)
print("COMPLETE - Ready for production deployment")
print("="*100)
print()
print("Next steps:")
print("1. SCP buffer files to server:")
print("   scp data/oanda_cache/prediction_buffer_ann_*.pkl user@server:~/ForexTest/data/oanda_cache/")
print()
print("2. On server, pull git changes:")
print("   git pull")
print()
print("3. Restart trader service")
print()
