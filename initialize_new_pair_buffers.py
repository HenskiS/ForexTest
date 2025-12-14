"""
Initialize prediction buffers for the 4 new pairs
Run this on desktop, then copy buffers to server
"""
import os
import pandas as pd
import numpy as np
from trading import TradingConfig, OandaClient
from trading.trading_models import TradingModel

NEW_PAIRS = ['EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE = TradingConfig.PREDICTION_BUFFER_SIZE  # 200
TRAIN_WINDOW = TradingConfig.TRAIN_WINDOW_SIZE  # 378

print("="*80)
print("INITIALIZING BUFFERS FOR NEW PAIRS")
print("="*80)
print(f"Pairs: {', '.join(NEW_PAIRS)}")
print(f"Buffer size: {BUFFER_SIZE}")
print()

os.makedirs('data/oanda_cache', exist_ok=True)

for pair in NEW_PAIRS:
    print(f"\n{'='*80}")
    print(f"INITIALIZING {pair}")
    print(f"{'='*80}\n")

    client = OandaClient(pair, practice=True)
    model = TradingModel(pair, model_type='ann')

    total_days_needed = TRAIN_WINDOW + BUFFER_SIZE + 300
    print(f"Fetching {total_days_needed} days of data...")
    df = client.fetch_latest_data(count=total_days_needed)

    if df.empty:
        print(f"ERROR: Failed to fetch data for {pair}")
        continue

    df = df.set_index('date')
    print(f"Fetched {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    predictions = []
    start_idx = len(df) - BUFFER_SIZE

    print(f"Generating {BUFFER_SIZE} predictions...")
    for i in range(start_idx, len(df)):
        train_end = i
        train_start = max(0, train_end - TRAIN_WINDOW - 300)
        df_train = df.iloc[train_start:train_end+1].copy()

        model_obj, scaler, df_clean = model.train(df_train)
        prediction = model.predict(df_clean)
        predictions.append(prediction)

        progress = ((i - start_idx + 1) / BUFFER_SIZE) * 100
        if (i - start_idx + 1) % 20 == 0:
            print(f"  Progress: {progress:.0f}%")

    model.prediction_buffer = predictions
    model.save_prediction_buffer()
    print(f"✓ {pair} buffer saved ({len(predictions)} predictions)")

print()
print("="*80)
print("COMPLETE! New buffer files:")
for pair in NEW_PAIRS:
    f = f'data/oanda_cache/prediction_buffer_ann_{pair}.pkl'
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} MISSING")
print()
print("Next: Copy these to your server")
