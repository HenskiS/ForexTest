"""
Train XGBoost model on recent data only (2021-present) using walk-forward validation.

This tests whether the strategy works in the current market regime.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import pickle
from datetime import datetime

print("XGBOOST TRAINING - RECENT DATA (2021-PRESENT)")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Filter to 2021 and later
df = df[df.index >= '2021-01-01']
print(f"Filtered to 2021+: {len(df)} rows from {df.index.min()} to {df.index.max()}")

# Define features
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'target_5day_return'
]

# Clean data
df_clean = df.dropna(subset=technical_features)
print(f"After removing NaN: {len(df_clean)} rows")

# Configuration
TARGET = 'target_5day_return'
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

print(f"\nWindow Configuration:")
print(f"  Train: {TRAIN_DAYS} days")
print(f"  Validation: {VAL_DAYS} days")
print(f"  Test: {TEST_DAYS} days")
print(f"  Roll forward: {ROLL_DAYS} days")
print(f"  Total window size: {WINDOW_SIZE} days")


def generate_windows(df, window_size, roll_days):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'window_id': len(windows),
            'train_start': start_idx,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': test_end,
            'date_start': df.index[start_idx],
            'date_end': df.index[test_end - 1]
        }

        windows.append(window)
        start_idx += roll_days

    return windows


# Generate windows
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
print(f"\nGenerated {len(windows)} walk-forward windows:")
for i, w in enumerate(windows):
    print(f"  Window {i}: {w['date_start'].date()} to {w['date_end'].date()}")

if len(windows) == 0:
    print("\nERROR: Not enough data to create even one window!")
    print(f"Need at least {WINDOW_SIZE} days, but only have {len(df_clean)} days")
    exit(1)

# Feature columns
feature_cols = [col for col in technical_features if col != TARGET]

# Hyperparameter search space (same as original)
param_distributions = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.001, 0.1),
    'max_depth': randint(3, 15),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 2)
}

# Train models on each window
all_results = []
N_ITER = 20  # Same as original

for window_idx, window in enumerate(windows):
    print(f"\n{'='*80}")
    print(f"WINDOW {window_idx + 1}/{len(windows)}")
    print(f"Period: {window['date_start'].date()} to {window['date_end'].date()}")
    print(f"{'='*80}")

    # Split data
    train_data = df_clean.iloc[window['train_start']:window['train_end']]
    val_data = df_clean.iloc[window['val_start']:window['val_end']]
    test_data = df_clean.iloc[window['test_start']:window['test_end']]

    X_train = train_data[feature_cols].values
    y_train = train_data[TARGET].values
    X_val = val_data[feature_cols].values
    y_val = val_data[TARGET].values
    X_test = test_data[feature_cols].values
    y_test = test_data[TARGET].values

    print(f"Train: {len(X_train)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"Val:   {len(X_val)} samples ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    print(f"Test:  {len(X_test)} samples ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # Hyperparameter tuning with RandomizedSearchCV
    print(f"\nHyperparameter tuning ({N_ITER} iterations)...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"Best params: {best_params}")

    # Train final model with best params
    print("Training final model with best hyperparameters...")
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = final_model.predict(X_val)
    val_mse = np.mean((val_predictions - y_val) ** 2)
    val_rmse = np.sqrt(val_mse)

    # Evaluate on test set
    test_predictions = final_model.predict(X_test)
    test_mse = np.mean((test_predictions - y_test) ** 2)
    test_rmse = np.sqrt(test_mse)

    print(f"\nValidation RMSE: {val_rmse:.6f}")
    print(f"Test RMSE:       {test_rmse:.6f}")

    # Store results
    window_results = {
        'window_id': window_idx,
        'date_start': window['date_start'],
        'date_end': window['date_end'],
        'best_params': best_params,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'predictions': test_predictions.tolist(),
        'actuals': y_test.tolist(),
        'test_dates': test_data.index.tolist()
    }
    all_results.append(window_results)

# Save results
output_file = 'xgboost_results_recent_2021.pkl'
print(f"\n{'='*80}")
print(f"Saving results to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total windows trained: {len(all_results)}")
print(f"Time period: {windows[0]['date_start'].date()} to {windows[-1]['date_end'].date()}")
print(f"\nValidation RMSE by window:")
for result in all_results:
    print(f"  Window {result['window_id']}: {result['val_rmse']:.6f}")

avg_val_rmse = np.mean([r['val_rmse'] for r in all_results])
avg_test_rmse = np.mean([r['test_rmse'] for r in all_results])
print(f"\nAverage Validation RMSE: {avg_val_rmse:.6f}")
print(f"Average Test RMSE:       {avg_test_rmse:.6f}")

print(f"\nResults saved to: {output_file}")
print("="*80)
