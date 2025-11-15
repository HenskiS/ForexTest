"""
Export 1-day EURUSD trades to CSV for leverage analysis
Using the optimal configuration: Vol-adj (0.40%/1.00%) + 1 day cooldown
"""

import pandas as pd
import numpy as np
import pickle

print("EXPORTING 1-DAY EURUSD TRADES TO CSV")
print("="*80)

# Load data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'target_5day_return'
]
df_clean = df.dropna(subset=technical_features)

# Load predictions
with open('xgboost_results_EURUSD_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

def generate_signals_q1q3(predictions):
    """Generate signals using Q1/Q3 thresholds (25th/75th percentile)."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals

def backtest_and_export_trades(actuals, predictions, signals, df_prices, test_indices,
                               base_stop_loss_pct=0.0040,
                               base_take_profit_pct=0.0100,
                               loss_cooldown_days=1,
                               transaction_cost_pct=0.0002,
                               holding_period=5):
    """Backtest and collect all trades for export."""

    position = 0
    entry_price = 0.0
    entry_date = None
    signal_date = None
    signal_value = 0
    prediction_value = 0
    cooldown_remaining = 0
    holding_days = 0

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
        pred = predictions[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        current_date = test_data.index[i]

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
                exit_reason = 'Time Exit (5 days)'
                exit_price = close_price

            if exit_triggered:
                # Calculate return
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                # Record trade
                trades.append({
                    'signal_date': signal_date,
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'signal': 'LONG' if signal_value == 1 else 'SHORT',
                    'prediction': prediction_value,
                    'entry_price': entry_price,
                    'stop_price': entry_price * (1 - stop_loss_pct) if position == 1 else entry_price * (1 + stop_loss_pct),
                    'target_price': entry_price * (1 + take_profit_pct) if position == 1 else entry_price * (1 - take_profit_pct),
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'stop_pct': stop_loss_pct * 100,
                    'target_pct': take_profit_pct * 100,
                    'raw_return_pct': raw_return_pct,
                    'net_return_pct': net_return_pct,
                    'outcome': outcome
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
            entry_date = current_date
            signal_date = test_data.index[i-1] if i > 0 else current_date
            signal_value = signal
            prediction_value = pred
            holding_days = 0

    return trades

# Collect all trades
print("\nProcessing trades with optimal configuration:")
print("  - Vol-adjusted stops (0.40% base SL / 1.00% base TP)")
print("  - 1 day loss cooldown")
print("  - 5 day holding period")
print()

all_trades = []

# Need to reconstruct windows to get test indices
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

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

windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)
print(f"Reconstructed {len(windows)} windows")

for window_idx, window_results in enumerate(all_results):
    predictions = window_results['predictions']
    actuals = window_results['actuals']

    # Get test indices from window
    window = windows[window_idx]
    test_indices = list(range(window['test_start'], window['test_end']))

    signals = generate_signals_q1q3(predictions)

    trades = backtest_and_export_trades(
        actuals,
        predictions,
        signals,
        df_clean,
        test_indices,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002,
        holding_period=5
    )

    all_trades.extend(trades)

# Create DataFrame and export
trades_df = pd.DataFrame(all_trades)
trades_df = trades_df.sort_values('exit_date')

# Export to CSV
output_file = 'xgboost_trades_EURUSD_1day.csv'
trades_df.to_csv(output_file, index=False)

print(f"Exported {len(trades_df)} trades to {output_file}")
print()

# Summary statistics
wins = trades_df[trades_df['outcome'] == 'WIN']
losses = trades_df[trades_df['outcome'] == 'LOSS']

print("SUMMARY STATISTICS")
print("="*80)
print(f"Total Trades: {len(trades_df)}")
print(f"Winning Trades: {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"Losing Trades: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")
print(f"Average Win: {wins['net_return_pct'].mean():.3f}%")
print(f"Average Loss: {losses['net_return_pct'].mean():.3f}%")
print(f"Best Trade: {trades_df['net_return_pct'].max():.3f}%")
print(f"Worst Trade: {trades_df['net_return_pct'].min():.3f}%")
print(f"Average Return per Trade: {trades_df['net_return_pct'].mean():.3f}%")
print()
print(f"Date Range: {trades_df['exit_date'].min()} to {trades_df['exit_date'].max()}")
print()
print("Exit Reasons:")
for reason in trades_df['exit_reason'].value_counts().items():
    print(f"  {reason[0]}: {reason[1]} ({reason[1]/len(trades_df)*100:.1f}%)")

print()
print("="*80)
print("Export complete!")
