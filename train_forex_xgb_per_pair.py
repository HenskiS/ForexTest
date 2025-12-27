"""
Train one XGBoost model per forex pair with improved features:
- 3-day prediction horizon (less noise)
- Cross-pair relative strength features
- Volatility regime features
- Binary classification
"""
import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TRAINING PER-PAIR FOREX XGBoost MODELS")
print("="*100)
print()

DATA_DIR = 'data'
OUTPUT_DIR = 'models'
TEST_SPLIT_DATE = '2024-01-01'
PREDICTION_HORIZON = 3  # 3-day forward returns (less noisy than 1-day)

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Configuration:")
print(f"  Model: XGBoost (per-pair)")
print(f"  Prediction horizon: {PREDICTION_HORIZON} days")
print(f"  Train/Test split: {TEST_SPLIT_DATE}")
print()

def calculate_features(df):
    """Calculate technical features"""
    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # EMAs
    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # Volume (if available, otherwise will be NaN)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

    return df

# Load all pairs
print("Loading all pairs...")
all_data = {}

for pair in PAIRS:
    filepath = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')

    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = calculate_features(df)
        df = df.dropna()

        # Calculate target: will price go up in N days?
        df[f'target_return_{PREDICTION_HORIZON}d'] = df['close'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
        df['target'] = (df[f'target_return_{PREDICTION_HORIZON}d'] > 0).astype(int)

        all_data[pair] = df
        print(f"  {pair}: {len(df)} days")

    except Exception as e:
        print(f"  {pair}: ERROR - {str(e)}")
        continue

print()
print(f"Loaded {len(all_data)} pairs")
print()

# Add cross-pair features
print("Calculating cross-pair features...")

# Calculate average return across all pairs (market sentiment)
for date in all_data['EURUSD'].index:
    returns_on_date = []
    for pair, df in all_data.items():
        if date in df.index:
            returns_on_date.append(df.loc[date, 'return_1d'])

    avg_return = np.mean(returns_on_date) if len(returns_on_date) > 0 else 0

    # Add to each pair as "market sentiment"
    for pair, df in all_data.items():
        if date in df.index:
            df.loc[date, 'market_sentiment'] = avg_return

# Calculate relative strength vs other pairs
for pair, df in all_data.items():
    df['relative_strength_vs_market'] = df['return_1d'] - df['market_sentiment']

print("Cross-pair features calculated")
print()

# Define base features
technical_features = [
    'return_1d', 'return_3d', 'return_5d', 'return_10d',
    'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50',
    'macd', 'macd_signal', 'macd_diff',
    'rsi', 'atr_pct',
    'volatility_10d', 'volatility_20d',
    'bb_position',
    'momentum_10', 'momentum_20',
    'market_sentiment',
    'relative_strength_vs_market'
]

# XGBoost hyperparameters (optimized for classification)
XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 20
}

# Train model for each pair
print("="*100)
print("TRAINING MODELS")
print("="*100)
print()

models = {}
results = []

for pair in PAIRS:
    if pair not in all_data:
        continue

    print(f"{pair}:")
    print("-" * 50)

    df = all_data[pair].copy()

    # Remove rows with missing features or target
    df = df.dropna(subset=technical_features + ['target'])

    # Split train/test
    train_df = df[df.index < TEST_SPLIT_DATE]
    test_df = df[df.index >= TEST_SPLIT_DATE]

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    if len(train_df) < 100 or len(test_df) < 10:
        print(f"  SKIPPED: Not enough data")
        print()
        continue

    # Prepare data
    X_train = train_df[technical_features].values
    y_train = train_df['target'].values
    X_test = test_df[technical_features].values
    y_test = test_df['target'].values

    # Check class balance
    train_pos_pct = y_train.mean()
    test_pos_pct = y_test.mean()
    print(f"  Train class balance: {train_pos_pct*100:.1f}% up")
    print(f"  Test class balance: {test_pos_pct*100:.1f}% up")

    # Calculate scale_pos_weight for imbalanced classes
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Train model
    model = XGBClassifier(
        **XGB_CONFIG,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")

    # Detailed test metrics
    test_cm = confusion_matrix(y_test, y_test_pred)
    print(f"  Test confusion matrix:")
    print(f"    [[{test_cm[0,0]:>4}, {test_cm[0,1]:>4}],")
    print(f"     [{test_cm[1,0]:>4}, {test_cm[1,1]:>4}]]")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': technical_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"  Top 5 features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")

    print()

    # Save model
    models[pair] = {
        'model': model,
        'features': technical_features,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_cm': test_cm
    }

    results.append({
        'pair': pair,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    })

# Summary
print("="*100)
print("TRAINING SUMMARY")
print("="*100)
print()

results_df = pd.DataFrame(results)
avg_test_acc = results_df['test_acc'].mean()

print(f"{'Pair':<10} {'Train Acc':>12} {'Test Acc':>12} {'Test Samples':>15}")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"{row['pair']:<10} {row['train_acc']*100:>11.2f}% {row['test_acc']*100:>11.2f}% {row['test_samples']:>15,}")

print()
print(f"Average test accuracy: {avg_test_acc*100:.2f}%")
print(f"Target (coin flip): 50.00%")
print(f"Edge: {(avg_test_acc - 0.5)*100:.2f} percentage points")
print()

if avg_test_acc > 0.50:
    print(f"SUCCESS: Model beats coin flip by {(avg_test_acc - 0.5)*100:.2f}%!")
else:
    print(f"WARNING: Model still at/below coin flip")

print()

# Save all models
models_file = os.path.join(OUTPUT_DIR, 'forex_xgb_per_pair_models.pkl')
with open(models_file, 'wb') as f:
    pickle.dump({
        'models': models,
        'prediction_horizon': PREDICTION_HORIZON,
        'test_split_date': TEST_SPLIT_DATE
    }, f)

print(f"Models saved to: {models_file}")
print()

print("="*100)
print("TRAINING COMPLETE")
print("="*100)
