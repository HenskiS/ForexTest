"""
Train all 8 pairs with optimized hyperparameters
Save predictions for easy testing of different stop/take profit levels
Uses winning config: (13, 20, 31) SGD with LR 0.001, momentum 0.4
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING ALL 8 PAIRS WITH OPTIMIZED HYPERPARAMETERS")
print("="*100)
print()

# Best configuration from validation
BEST_CONFIG = {
    'hidden_layer_sizes': (13, 20, 31),
    'activation': 'tanh',
    'solver': 'sgd',
    'learning_rate_init': 0.001,
    'momentum': 0.4,
    'batch_size': 64,
    'max_iter': 20,
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'random_state': 42,
    'verbose': False,
    'early_stopping': False
}

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
TEST_DAYS = 4500
TRAIN_WINDOW = 378
OUTPUT_DIR = 'optimized_ann_predictions'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  Architecture: {BEST_CONFIG['hidden_layer_sizes']}")
print(f"  Optimizer: {BEST_CONFIG['solver']}")
print(f"  Learning rate: {BEST_CONFIG['learning_rate_init']}")
print(f"  Momentum: {BEST_CONFIG['momentum']}")
print(f"  Batch size: {BEST_CONFIG['batch_size']}")
print(f"  Epochs: {BEST_CONFIG['max_iter']}")
print()
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Test days: {TEST_DAYS}")
print()

def calculate_features(df):
    """Calculate all technical features"""
    df['momentum'] = df['close'].pct_change()
    df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['range'] = df['high'] - df['low']
    df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    for period in [10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    df['stoch_k'] = 100 * (df['close'] - low_20) / (high_20 - low_20)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['atr'] = atr
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    df['target_1day_return'] = df['close'].pct_change(1).shift(-1)

    return df

# Train each pair
for pair_idx, pair in enumerate(PAIRS):
    print(f"\n{'='*100}")
    print(f"TRAINING: {pair} ({pair_idx+1}/{len(PAIRS)})")
    print(f"{'='*100}\n")

    # Load data
    print(f"Loading data...")
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate features
    print("Calculating features...")
    df = calculate_features(df)
    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in ['target_1day_return', 'open', 'high', 'low', 'close', 'volume']]

    # Split data
    train_size = len(df) - TEST_DAYS
    test_start_idx = train_size
    test_indices = list(range(test_start_idx, len(df)))

    print(f"Total samples: {len(df)}")
    print(f"Test samples: {len(test_indices)}")
    print()

    # Train with rolling window
    print(f"Training with rolling window ({len(test_indices)} iterations)...")
    scaler = MinMaxScaler()
    predictions = []

    for i, test_idx in enumerate(test_indices):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(test_indices)} ({i/len(test_indices)*100:.1f}%)")

        # Get training window
        train_end = test_idx
        train_start = max(0, train_end - TRAIN_WINDOW)

        X_window = df[feature_cols].iloc[train_start:train_end]
        y_window = df['target_1day_return'].iloc[train_start:train_end]

        # Scale
        X_window_scaled = scaler.fit_transform(X_window)

        # Train model
        model = MLPRegressor(**BEST_CONFIG)
        model.fit(X_window_scaled, y_window)

        # Predict
        X_test_sample = df[feature_cols].iloc[test_idx:test_idx+1]
        X_test_scaled = scaler.transform(X_test_sample)
        pred = model.predict(X_test_scaled)[0]
        predictions.append(pred)

    print(f"  Progress: {len(test_indices)}/{len(test_indices)} (100.0%)")
    print()
    print("Training complete!")

    # Save predictions
    checkpoint = {
        'pair': pair,
        'predictions': predictions,
        'test_indices': test_indices,
        'config': BEST_CONFIG,
        'feature_cols': feature_cols,
        'dates': df.index[test_indices].tolist()
    }

    checkpoint_file = os.path.join(OUTPUT_DIR, f'predictions_{pair}.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Saved: {checkpoint_file}")

    # Quick stats
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(predictions):.6f}")
    print(f"  Std: {np.std(predictions):.6f}")
    print(f"  Min: {np.min(predictions):.6f}")
    print(f"  Max: {np.max(predictions):.6f}")

print(f"\n\n{'='*100}")
print("ALL PAIRS TRAINED SUCCESSFULLY")
print(f"{'='*100}\n")

print(f"Predictions saved to: {OUTPUT_DIR}/")
print(f"Files created:")
for pair in PAIRS:
    print(f"  - predictions_{pair}.pkl")

print("\nNext steps:")
print("  1. Run multi-pair backtest with different SL/TP combinations")
print("  2. Compare to XGBoost results")
print("  3. Test at different leverage levels")

print(f"\n{'='*100}")
print("TRAINING COMPLETE")
print(f"{'='*100}")
