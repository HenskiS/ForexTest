"""
Optimize entry thresholds for 2025 using rolling daily retraining.

Tests different percentile thresholds to find the optimal entry points
for the rolling daily retraining strategy.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
TRAIN_WINDOW_SIZE = 756  # 600 train + 156 val

print(f"Optimizing Thresholds for {PAIR} - 2025 Rolling Daily Retraining")
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

# Get 2025 data
df_2025 = df_clean[df_clean.index.year == 2025]
print(f"\n2025 data: {len(df_2025)} days ({df_2025.index.min()} to {df_2025.index.max()})")

# Get best hyperparameters from static results
try:
    with open(f'xgboost_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
        all_results = pickle.load(f)
    best_params = all_results[-1]['best_params']
    print(f"\nUsing hyperparameters: {best_params}")
except:
    best_params = {
        'n_estimators': 250,
        'learning_rate': 0.05,
        'max_depth': 10,
        'gamma': 0.001
    }
    print(f"\nUsing default hyperparameters: {best_params}")

# Rolling daily retraining for 2025
print(f"\nGenerating rolling daily predictions for 2025...")
print("="*80)

predictions = []
actuals = []
dates = []

for i, date in enumerate(tqdm(df_2025.index, desc="Training models")):
    current_idx = df_clean.index.get_loc(date)

    # Skip if not enough history
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

    predictions.append(y_pred)
    actuals.append(y_actual)
    dates.append(date)

predictions = np.array(predictions)
actuals = np.array(actuals)

print(f"\nGenerated {len(predictions)} predictions for 2025")
print(f"Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
print(f"Prediction mean: {predictions.mean():.6f}")
print(f"Prediction std: {predictions.std():.6f}")

# Backtesting function with thresholds
def backtest_with_thresholds(predictions, actuals, df_prices, test_indices,
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

        # Decrement cooldown
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

        # Check exits if we have a position
        if position != 0:
            holding_days += 1

            # Calculate P&L
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            # Check exits
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Stop loss
            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'Stop Loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Take profit
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                exit_reason = 'Target'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            # Time exit (5 days)
            elif holding_days >= holding_period:
                exit_triggered = True
                exit_reason = 'Time Exit'
                exit_price = close_price

            if exit_triggered:
                # Calculate return
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                # Update equity
                current_equity = equity[-1]
                new_equity = current_equity * (1 + net_return_pct / 100)
                equity.append(new_equity)

                # Record trade
                trades.append({
                    'net_return_pct': net_return_pct,
                    'outcome': outcome,
                    'exit_reason': exit_reason
                })

                # Set cooldown after losing trade
                if net_return_pct < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0
                holding_days = 0

        # Enter new position (only if not in cooldown)
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    # Calculate metrics
    final_equity = equity[-1]
    total_return_pct = (final_equity - 1000) / 1000 * 100

    trades_df = pd.DataFrame(trades)
    win_rate = len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    n_hold = np.sum(signals == 0)

    return {
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'n_long': n_long,
        'n_short': n_short,
        'n_hold': n_hold,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold,
    }

# Get test indices that match the prediction dates
test_indices = []
for date in dates:
    try:
        idx = df_clean.index.get_loc(date)
        test_indices.append(idx)
    except:
        continue

print(f"\nMatched {len(test_indices)} prediction dates to price data")

# Test different threshold combinations
print("\n" + "="*80)
print("Testing Threshold Combinations")
print("="*80)

threshold_pairs = [
    (30, 70), (35, 65), (40, 60), (42, 58), (45, 55),
    (46, 54), (47, 53), (48, 52), (49, 51), (50, 50)
]

results = []

for lower, upper in tqdm(threshold_pairs, desc="Testing thresholds"):
    result = backtest_with_thresholds(
        predictions, actuals, df_clean, test_indices,
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

# Convert to DataFrame and sort by return
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return_pct', ascending=False)

# Display results
print("\n" + "="*80)
print(f"THRESHOLD OPTIMIZATION RESULTS - {PAIR} 2025")
print("="*80)
print()

print(f"{'Threshold':<15} {'Trades':<8} {'Win %':<8} {'Return %':<12} {'Final $':<12} {'Signals (L/S/H)'}")
print("-" * 80)

for _, row in results_df.iterrows():
    threshold_str = f"{int(row['lower_pct'])}th/{int(row['upper_pct'])}th"
    signals_str = f"{int(row['n_long'])}/{int(row['n_short'])}/{int(row['n_hold'])}"

    print(f"{threshold_str:<15} {int(row['total_trades']):<8} {row['win_rate']:>6.1f}% "
          f"{row['total_return_pct']:>10.2f}% ${row['final_equity']:>10,.2f} {signals_str}")

# Highlight best threshold
best = results_df.iloc[0]
print("\n" + "="*80)
print(f"OPTIMAL THRESHOLD: {int(best['lower_pct'])}th/{int(best['upper_pct'])}th percentile")
print("="*80)
print(f"Total Return: {best['total_return_pct']:.2f}%")
print(f"Final Equity: ${best['final_equity']:,.2f}")
print(f"Total Trades: {int(best['total_trades'])}")
print(f"Win Rate: {best['win_rate']:.1f}%")
print(f"Signal Distribution: {int(best['n_long'])} Long / {int(best['n_short'])} Short / {int(best['n_hold'])} Hold")
print()

# Save results
output_file = f'threshold_optimization_2025_{PAIR}.json'
results_df.to_json(output_file, orient='records', indent=2)
print(f"Results saved to: {output_file}")

# Also save predictions for potential use in production
predictions_file = f'rolling_daily_predictions_2025_{PAIR}.pkl'
with open(predictions_file, 'wb') as f:
    pickle.dump({
        'dates': dates,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist(),
        'pair': PAIR
    }, f)
print(f"Predictions saved to: {predictions_file}")
