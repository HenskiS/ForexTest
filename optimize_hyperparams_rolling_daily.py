"""
Optimize hyperparameters for rolling daily retraining approach.

Uses a single representative window (most recent) with 600/156 train/val split
to find optimal hyperparameters, then saves them for production use.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterSampler
import argparse
from tqdm import tqdm
import json

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--n-iter', type=int, default=30, help='Number of random hyperparameter combinations to test')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
TRAIN_DAYS = 600
VAL_DAYS = 156

print(f"Hyperparameter Optimization for Rolling Daily - {PAIR}")
print("="*80)
print(f"Train: {TRAIN_DAYS} days | Validation: {VAL_DAYS} days")
print(f"Random search iterations: {args.n_iter}")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

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

# Use most recent 756 days (600 train + 156 val)
if len(df_clean) < TRAIN_DAYS + VAL_DAYS:
    print(f"ERROR: Need at least {TRAIN_DAYS + VAL_DAYS} days, have {len(df_clean)}")
    exit(1)

recent_data = df_clean.iloc[-(TRAIN_DAYS + VAL_DAYS):].copy()
print(f"\nUsing most recent {len(recent_data)} days for optimization:")
print(f"  Date range: {recent_data.index[0]} to {recent_data.index[-1]}")

# Split into train and validation
train_data = recent_data.iloc[:TRAIN_DAYS]
val_data = recent_data.iloc[TRAIN_DAYS:]

print(f"  Train: {len(train_data)} days ({train_data.index[0]} to {train_data.index[-1]})")
print(f"  Val:   {len(val_data)} days ({val_data.index[0]} to {val_data.index[-1]})")

# Prepare data
X_train = train_data[technical_features].values
y_train = train_data[TARGET].values
X_val = val_data[technical_features].values
y_val = val_data[TARGET].values

print(f"\nTrain samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")

# Hyperparameter search space
param_distributions = {
    'n_estimators': [50, 100, 125, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'max_depth': [3, 5, 7, 8, 10, 12],
    'gamma': [0.0, 0.001, 0.01, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

# Generate random parameter combinations
param_list = list(ParameterSampler(param_distributions, n_iter=args.n_iter, random_state=42))

print(f"\n{'='*80}")
print(f"Testing {len(param_list)} hyperparameter combinations...")
print(f"{'='*80}\n")

results = []

for i, params in enumerate(tqdm(param_list, desc="Testing hyperparameters")):
    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train, verbose=False)

    # Evaluate on validation set
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    # Calculate metrics
    train_mae = np.mean(np.abs(y_train - y_pred_train))
    val_mae = np.mean(np.abs(y_val - y_pred_val))
    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
    val_corr = np.corrcoef(y_val, y_pred_val)[0, 1]

    results.append({
        'iteration': i + 1,
        'params': params,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_corr': train_corr,
        'val_corr': val_corr,
    })

# Sort by validation MAE (lower is better)
results_sorted = sorted(results, key=lambda x: x['val_mae'])

# Display top 10 results
print(f"\n{'='*80}")
print("TOP 10 HYPERPARAMETER COMBINATIONS")
print(f"{'='*80}\n")

print(f"{'Rank':<6} {'Val MAE':<12} {'Val Corr':<12} {'Hyperparameters'}")
print("-" * 100)

for i, result in enumerate(results_sorted[:10]):
    params_str = f"n_est={result['params']['n_estimators']}, lr={result['params']['learning_rate']}, " \
                 f"depth={result['params']['max_depth']}, gamma={result['params']['gamma']}"
    print(f"{i+1:<6} {result['val_mae']:<12.6f} {result['val_corr']:<12.4f} {params_str}")

# Best hyperparameters
best = results_sorted[0]
best_params = best['params']

print(f"\n{'='*80}")
print("BEST HYPERPARAMETERS")
print(f"{'='*80}")
print(f"Validation MAE: {best['val_mae']:.6f}")
print(f"Validation Correlation: {best['val_corr']:.4f}")
print(f"Train MAE: {best['train_mae']:.6f}")
print(f"Train Correlation: {best['train_corr']:.4f}")
print()
print("Parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print()

# Save results
output_file = f'hyperparams_rolling_daily_{PAIR}.json'
with open(output_file, 'w') as f:
    json.dump({
        'pair': PAIR,
        'best_params': best_params,
        'best_val_mae': best['val_mae'],
        'best_val_corr': best['val_corr'],
        'train_days': TRAIN_DAYS,
        'val_days': VAL_DAYS,
        'all_results': results_sorted[:10]  # Save top 10
    }, f, indent=2)

print(f"Results saved to: {output_file}")

# Also save in pickle format for easy loading in other scripts
pickle_file = f'hyperparams_rolling_daily_{PAIR}.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(best_params, f)

print(f"Best params saved to: {pickle_file}")
print()
print("="*80)
print("Use these hyperparameters for rolling daily production trading!")
print("="*80)
