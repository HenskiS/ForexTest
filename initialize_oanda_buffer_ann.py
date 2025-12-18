"""
Initialize ANN prediction buffer for OANDA trading

This script generates the last 200 days of predictions using the ANN model
and saves them to the prediction buffer. This allows the bot to start
generating signals immediately without waiting 50+ days to build up history.

Run this on your desktop (faster) and then copy the buffer files to your server.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from trading import TradingConfig, OandaClient
from trading.trading_models import TradingModel

# Parse command line arguments
parser = argparse.ArgumentParser(description='Initialize ANN prediction buffers')
parser.add_argument('--live', action='store_true', help='Use LIVE account instead of practice')
args = parser.parse_args()

PRACTICE_MODE = not args.live

print("="*100)
print("ANN PREDICTION BUFFER INITIALIZATION")
print("="*100)
print(f"Mode: {'LIVE' if not PRACTICE_MODE else 'PRACTICE'} account")
print()

PAIRS = TradingConfig.DEFAULT_PAIRS  # 8-pair Sleep Well portfolio
BUFFER_SIZE = TradingConfig.PREDICTION_BUFFER_SIZE  # 200
TRAIN_WINDOW = TradingConfig.TRAIN_WINDOW_SIZE  # 378

print(f"Pairs to initialize: {', '.join(PAIRS)}")
print(f"Buffer size: {BUFFER_SIZE} predictions")
print(f"Training window: {TRAIN_WINDOW} days")
print()

# Create output directory
os.makedirs('data/oanda_cache', exist_ok=True)

for pair in PAIRS:
    print(f"\n{'='*100}")
    print(f"INITIALIZING BUFFER FOR {pair}")
    print(f"{'='*100}\n")

    # Initialize client and model (verbose=False for cleaner output during batch generation)
    client = OandaClient(pair, practice=PRACTICE_MODE)
    model = TradingModel(pair, model_type='ann', verbose=False)

    # Fetch enough data for training + buffer
    # We need: TRAIN_WINDOW days for training + BUFFER_SIZE days to generate predictions
    # Plus some extra for feature calculation
    total_days_needed = TRAIN_WINDOW + BUFFER_SIZE + 300

    print(f"Fetching {total_days_needed} days of historical data...")
    df = client.fetch_latest_data(count=total_days_needed)

    if df.empty:
        print(f"ERROR: Failed to fetch data for {pair}")
        continue

    df = df.set_index('date')
    print(f"Fetched {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
    print()

    # Generate predictions for last BUFFER_SIZE days
    predictions = []

    print(f"Generating {BUFFER_SIZE} predictions (rolling window)...")
    print(f"This will take ~{BUFFER_SIZE * 2 // 60} minutes on your machine")
    print()

    # We'll generate predictions for the last BUFFER_SIZE days
    # For each day, we train on previous TRAIN_WINDOW days
    start_idx = len(df) - BUFFER_SIZE

    for i in range(start_idx, len(df)):
        # Get training data up to current day
        train_end = i
        train_start = max(0, train_end - TRAIN_WINDOW - 300)  # Extra for features
        df_train = df.iloc[train_start:train_end+1].copy()

        # Train model
        model_obj, scaler, df_clean = model.train(df_train)

        # Generate prediction for current day
        prediction = model.predict(df_clean)
        predictions.append(prediction)

        # Progress indicator
        progress = ((i - start_idx + 1) / BUFFER_SIZE) * 100
        if (i - start_idx + 1) % 10 == 0:
            print(f"  Progress: {progress:.1f}% ({i - start_idx + 1}/{BUFFER_SIZE})")

    print()
    print(f"Generated {len(predictions)} predictions")

    # Save to buffer
    model.prediction_buffer = predictions
    model.save_prediction_buffer()

    print(f"Buffer saved to: {model.buffer_file}")
    print(f"✓ {pair} initialization complete")

print()
print("="*100)
print("INITIALIZATION COMPLETE")
print("="*100)
print()

print("Buffer files created:")
for pair in PAIRS:
    buffer_file = f'data/oanda_cache/prediction_buffer_ann_{pair}.pkl'
    if os.path.exists(buffer_file):
        print(f"  ✓ {buffer_file}")
    else:
        print(f"  ✗ {buffer_file} (MISSING!)")

print()
print("Next steps:")
print("1. Commit and push code changes to git")
print("2. Copy buffer files to server:")
print("   scp data/oanda_cache/prediction_buffer_ann_*.pkl user@server:~/ForexTest/data/oanda_cache/")
print("3. On server: git pull")
print("4. On server: Test with --dry-run")
print("5. On server: Run live!")
