"""
Optimize entry thresholds across ALL test years using rolling daily retraining.

Generates predictions for all 46 walk-forward windows (like train_rolling_daily_all_years.py),
then tests different threshold combinations across all predictions to find optimal entry points.

This is computationally expensive but gives robust threshold estimates.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
import json
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--use-cached', action='store_true', help='Use cached rolling daily results if available')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
TRAIN_WINDOW_SIZE = 756  # 600 train + 156 val

print(f"Optimizing Thresholds for {PAIR} - ALL YEARS Rolling Daily Retraining")
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

# Check if we have cached rolling daily results
cached_file = f'rolling_daily_results_{PAIR}_{TARGET}.pkl'
if args.use_cached and os.path.exists(cached_file):
    print(f"\nLoading cached rolling daily results from {cached_file}...")
    with open(cached_file, 'rb') as f:
        rolling_results = pickle.load(f)

    # Combine all predictions
    all_dates = []
    all_predictions = []
    all_actuals = []

    for result in rolling_results:
        dates = pd.to_datetime(result['dates'])
        preds = result['predictions']
        actuals = result['actuals']

        all_dates.extend(dates)
        all_predictions.extend(preds)
        all_actuals.extend(actuals)

    all_dates = pd.DatetimeIndex(all_dates)
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    print(f"Loaded {len(predictions)} predictions from cache")
    print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")

else:
    print("\nNo cached results found. Generating rolling daily predictions...")
    print("(This will take a while - ~30-60 minutes for all years)")
    print()

    # Load best hyperparameters
    hyperparam_file = f'hyperparams_rolling_daily_{PAIR}.pkl'
    if os.path.exists(hyperparam_file):
        with open(hyperparam_file, 'rb') as f:
            best_params = pickle.load(f)
        print(f"Loaded optimized hyperparameters from {hyperparam_file}")
        print(f"  {best_params}")
    else:
        # Try to load from static results as fallback
        try:
            with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
                static_results = pickle.load(f)
            best_params = static_results[-1]['best_params']
            print(f"Using hyperparameters from static results (not optimal for rolling daily!)")
            print(f"  {best_params}")
            print(f"  RECOMMENDATION: Run optimize_hyperparams_rolling_daily.py first!")
        except:
            best_params = {
                'n_estimators': 250,
                'learning_rate': 0.05,
                'max_depth': 10,
                'gamma': 0.001
            }
            print(f"Using default hyperparameters")
            print(f"  {best_params}")

    # Generate windows (same as train_rolling_daily_all_years.py)
    TRAIN_DAYS = 600
    VAL_DAYS = 156
    TEST_DAYS = 126
    ROLL_DAYS = 126
    WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

    def generate_windows(df, window_size, roll_days):
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

    windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
    print(f"\nGenerated {len(windows)} windows")

    # Generate predictions for all windows
    all_dates = []
    all_predictions = []
    all_actuals = []

    for window_idx in tqdm(range(len(windows)), desc="Processing windows"):
        window = windows[window_idx]

        # Get test period dates
        test_start_idx = window['test_start']
        test_end_idx = window['test_end']
        test_dates = df_clean.index[test_start_idx:test_end_idx]

        # For each day in test period, retrain with rolling window
        for date in test_dates:
            current_idx = df_clean.index.get_loc(date)

            # Check if we have enough history for 756-day window
            if current_idx < TRAIN_WINDOW_SIZE:
                continue

            # Get rolling 756-day window ending just before current day
            train_end_idx = current_idx
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
            X_today = scaler.transform(df_clean.loc[[date]][technical_features])
            y_pred = model.predict(X_today)[0]
            y_actual = df_clean.loc[date, TARGET]

            all_predictions.append(y_pred)
            all_actuals.append(y_actual)
            all_dates.append(date)

    all_dates = pd.DatetimeIndex(all_dates)
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    print(f"\nGenerated {len(predictions)} predictions across all years")
    print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")

print(f"\nPrediction range: {predictions.min():.6f} to {predictions.max():.6f}")
print(f"Prediction mean: {predictions.mean():.6f}")
print(f"Prediction std: {predictions.std():.6f}")

# Backtesting function
def backtest_with_thresholds(predictions, actuals, df_prices, dates,
                             lower_pct=48, upper_pct=52,
                             base_stop_loss_pct=0.0040,
                             base_take_profit_pct=0.0100,
                             loss_cooldown_days=1,
                             transaction_cost_pct=0.0002,
                             holding_period=5):
    """Backtest with specific threshold percentiles."""

    # Generate signals
    signals = np.zeros(len(predictions))
    lower_threshold = np.percentile(predictions, lower_pct)
    upper_threshold = np.percentile(predictions, upper_pct)
    signals[predictions >= upper_threshold] = 1
    signals[predictions <= lower_threshold] = -1

    # Get indices for dates
    test_indices = []
    for date in dates:
        try:
            idx = df_prices.index.get_loc(date)
            test_indices.append(idx)
        except:
            continue

    position = 0
    entry_price = 0.0
    cooldown_remaining = 0
    holding_days = 0

    equity = [1000]
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    atr = test_data['atr'].values
    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Volatility-adjusted stops
        if not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        if position != 0:
            holding_days += 1

            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            exit_triggered = False
            exit_price = None

            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)
            elif holding_days >= holding_period:
                exit_triggered = True
                exit_price = close_price

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                current_equity = equity[-1]
                new_equity = current_equity * (1 + net_return_pct / 100)
                equity.append(new_equity)

                trades.append({'net_return_pct': net_return_pct, 'outcome': outcome})

                if net_return_pct < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0
                holding_days = 0

        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    final_equity = equity[-1]
    total_return_pct = (final_equity - 1000) / 1000 * 100

    trades_df = pd.DataFrame(trades)
    win_rate = len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    n_hold = np.sum(signals == 0)

    # Calculate annual return
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    annual_return = (final_equity / 1000) ** (1 / years) - 1
    annual_return_pct = annual_return * 100

    return {
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'annual_return_pct': annual_return_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'n_long': n_long,
        'n_short': n_short,
        'n_hold': n_hold,
        'years': years,
    }

# Test different threshold combinations
print("\n" + "="*80)
print("Testing Threshold Combinations")
print("="*80)

threshold_pairs = [
    (25, 75), (30, 70), (35, 65), (40, 60), (42, 58), (45, 55),
    (46, 54), (47, 53), (48, 52), (49, 51), (50, 50)
]

results = []

for lower, upper in tqdm(threshold_pairs, desc="Testing thresholds"):
    result = backtest_with_thresholds(
        predictions, actuals, df_clean, all_dates,
        lower_pct=lower, upper_pct=upper,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002,
        holding_period=5
    )
    result['lower_pct'] = lower
    result['upper_pct'] = upper
    results.append(result)

# Sort by annual return
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('annual_return_pct', ascending=False)

# Display results
print("\n" + "="*80)
print(f"THRESHOLD OPTIMIZATION RESULTS - {PAIR} ALL YEARS")
print("="*80)
print()

print(f"{'Threshold':<15} {'Trades':<8} {'Win %':<8} {'Annual %':<12} {'Total %':<12} {'Final $':<12} {'Years':<8}")
print("-" * 100)

for _, row in results_df.iterrows():
    threshold_str = f"{int(row['lower_pct'])}th/{int(row['upper_pct'])}th"

    print(f"{threshold_str:<15} {int(row['total_trades']):<8} {row['win_rate']:>6.1f}% "
          f"{row['annual_return_pct']:>10.2f}% {row['total_return_pct']:>10.1f}% "
          f"${row['final_equity']:>10,.0f} {row['years']:>6.1f}")

# Highlight best threshold
best = results_df.iloc[0]
print("\n" + "="*80)
print(f"OPTIMAL THRESHOLD: {int(best['lower_pct'])}th/{int(best['upper_pct'])}th percentile")
print("="*80)
print(f"Annual Return: {best['annual_return_pct']:.2f}%")
print(f"Total Return: {best['total_return_pct']:.1f}%")
print(f"Final Equity: ${best['final_equity']:,.2f}")
print(f"Total Trades: {int(best['total_trades'])}")
print(f"Win Rate: {best['win_rate']:.1f}%")
print(f"Time Period: {best['years']:.1f} years")
print(f"Signal Distribution: {int(best['n_long'])} Long / {int(best['n_short'])} Short / {int(best['n_hold'])} Hold")
print()

# Save results
output_file = f'threshold_optimization_all_years_{PAIR}.json'
results_df.to_json(output_file, orient='records', indent=2)
print(f"Results saved to: {output_file}")
