"""
Convert saved ANN backtest predictions to prediction buffer format

This is much faster than regenerating predictions from scratch.
We already have predictions from the backtest, just need to reformat them.
"""
import os
import pickle
from trading import TradingConfig

print("="*100)
print("CONVERTING BACKTEST PREDICTIONS TO BUFFER FORMAT")
print("="*100)
print()

PAIRS = TradingConfig.DEFAULT_PAIRS
PREDICTION_DIR = 'optimized_ann_predictions'
BUFFER_SIZE = TradingConfig.PREDICTION_BUFFER_SIZE  # 200

# Create output directory
os.makedirs('data/oanda_cache', exist_ok=True)

for pair in PAIRS:
    print(f"{pair}:")

    # Load backtest predictions
    pred_file = os.path.join(PREDICTION_DIR, f'predictions_{pair}.pkl')

    if not os.path.exists(pred_file):
        print(f"  ERROR: {pred_file} not found!")
        print(f"  Run train_all_pairs_optimized_hyperparams.py first")
        continue

    with open(pred_file, 'rb') as f:
        data = pickle.load(f)

    predictions = data['predictions']
    print(f"  Loaded {len(predictions)} predictions from backtest")

    # Take last BUFFER_SIZE predictions
    if len(predictions) >= BUFFER_SIZE:
        buffer_predictions = list(predictions[-BUFFER_SIZE:])
        print(f"  Using last {BUFFER_SIZE} predictions for buffer")
    else:
        buffer_predictions = list(predictions)
        print(f"  WARNING: Only {len(predictions)} predictions available (need {BUFFER_SIZE})")
        print(f"  Buffer will be smaller than optimal, but will grow over time")

    # Save to buffer file
    buffer_file = f'data/oanda_cache/prediction_buffer_ann_{pair}.pkl'
    with open(buffer_file, 'wb') as f:
        pickle.dump(buffer_predictions, f)

    print(f"  Saved {len(buffer_predictions)} predictions to {buffer_file}")
    print(f"  [OK] Buffer created")
    print()

print("="*100)
print("CONVERSION COMPLETE")
print("="*100)
print()

print("Buffer files created:")
for pair in PAIRS:
    buffer_file = f'data/oanda_cache/prediction_buffer_ann_{pair}.pkl'
    if os.path.exists(buffer_file):
        # Check size
        with open(buffer_file, 'rb') as f:
            buffer = pickle.load(f)
        print(f"  [OK] {buffer_file} ({len(buffer)} predictions)")
    else:
        print(f"  [X] {buffer_file} (MISSING!)")

print()
print("Next steps:")
print("1. Commit and push code changes")
print("2. Copy buffer files to server:")
print("   scp data/oanda_cache/prediction_buffer_ann_*.pkl user@server:~/ForexTest/data/oanda_cache/")
print("3. On server: git pull")
print("4. On server: Test with --dry-run")
