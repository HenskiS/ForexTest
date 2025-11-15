"""
Analyze exit reasons across all currency pairs
Shows where profits come from: stop loss, take profit, or time exit
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("EXIT REASON ANALYSIS - ALL PAIRS")
print("="*80)

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
THRESHOLDS = {
    'EURUSD': (48, 52),
    'GBPUSD': (48, 52),
    'USDJPY': (35, 65),
    'AUDUSD': (48, 52)
}

TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

def generate_signals(predictions, lower_pct, upper_pct):
    """Generate trading signals from predictions."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_threshold = np.percentile(predictions, lower_pct)
    upper_threshold = np.percentile(predictions, upper_pct)
    signals[predictions >= upper_threshold] = 1
    signals[predictions <= lower_threshold] = -1

    return signals

def generate_windows(df, window_size, roll_days):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'test_start': val_end,
            'test_end': test_end,
        }
        windows.append(window)
        start_idx += roll_days

    return windows

def backtest_with_exit_tracking(actuals, predictions, signals, df_prices, test_indices):
    """Backtest and track exit reasons."""

    position = 0
    entry_price = 0.0
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

    base_stop_loss_pct = 0.0040
    base_take_profit_pct = 0.0100
    transaction_cost_pct = 0.0002

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
            elif holding_days >= 5:
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

                trades.append({
                    'net_return_pct': net_return_pct,
                    'outcome': outcome,
                    'exit_reason': exit_reason
                })

                if net_return_pct < 0:
                    cooldown_remaining = 1

                position = 0
                holding_days = 0

        # Enter new position
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    return trades

# Analyze each pair
all_pair_trades = []

for pair in PAIRS:
    print(f"\nAnalyzing {pair}...")

    # Load data
    df = pd.read_csv(f'data/{pair}_1day_with_features_FIXED_multitarget.csv',
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
    with open(f'xgboost_results_{pair}_target_5day_return.pkl', 'rb') as f:
        all_results = pickle.load(f)

    windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)

    lower_pct, upper_pct = THRESHOLDS[pair]

    for window_idx, window_results in enumerate(all_results):
        predictions = window_results['predictions']
        actuals = window_results['actuals']

        window = windows[window_idx]
        test_indices = list(range(window['test_start'], window['test_end']))

        signals = generate_signals(predictions, lower_pct, upper_pct)

        trades = backtest_with_exit_tracking(
            actuals, predictions, signals, df_clean, test_indices
        )

        for trade in trades:
            trade['pair'] = pair
            all_pair_trades.append(trade)

    print(f"  {len([t for t in all_pair_trades if t['pair'] == pair])} trades")

# Create DataFrame
trades_df = pd.DataFrame(all_pair_trades)

print("\n" + "="*80)
print("EXIT REASON ANALYSIS RESULTS")
print("="*80)
print(f"\nTotal Trades Across All Pairs: {len(trades_df)}")
print(f"Overall Win Rate: {len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100:.1f}%")

# Split by outcome
wins = trades_df[trades_df['outcome'] == 'WIN']
losses = trades_df[trades_df['outcome'] == 'LOSS']

print(f"\n{'='*80}")
print(f"WINNING TRADES ({len(wins)} total, {len(wins)/len(trades_df)*100:.1f}% win rate):")
print(f"{'='*80}")

for reason in ['Time Exit (5 days)', 'Target', 'Stop Loss']:
    reason_wins = wins[wins['exit_reason'] == reason]
    if len(reason_wins) > 0:
        pct_of_wins = len(reason_wins) / len(wins) * 100
        avg_return = reason_wins['net_return_pct'].mean()
        print(f"  {reason:20s}: {len(reason_wins):5d} trades ({pct_of_wins:4.1f}% of wins) | Avg: +{avg_return:.2f}%")

print(f"\n{'='*80}")
print(f"LOSING TRADES ({len(losses)} total, {len(losses)/len(trades_df)*100:.1f}% loss rate):")
print(f"{'='*80}")

for reason in ['Stop Loss', 'Time Exit (5 days)', 'Target']:
    reason_losses = losses[losses['exit_reason'] == reason]
    if len(reason_losses) > 0:
        pct_of_losses = len(reason_losses) / len(losses) * 100
        avg_return = reason_losses['net_return_pct'].mean()
        print(f"  {reason:20s}: {len(reason_losses):5d} trades ({pct_of_losses:4.1f}% of losses) | Avg: {avg_return:.2f}%")

# Summary statistics by exit type
print(f"\n{'='*80}")
print("RETURN STATISTICS BY EXIT TYPE")
print(f"{'='*80}")
print(f"{'Exit Type':<25} {'Count':>8} {'Win/Loss':>12} {'Mean %':>10} {'Min %':>10} {'Max %':>10}")
print("-"*80)

for reason in ['Target', 'Stop Loss', 'Time Exit (5 days)']:
    reason_trades = trades_df[trades_df['exit_reason'] == reason]
    if len(reason_trades) > 0:
        count = len(reason_trades)
        win_rate = len(reason_trades[reason_trades['outcome'] == 'WIN']) / count * 100
        mean_ret = reason_trades['net_return_pct'].mean()
        min_ret = reason_trades['net_return_pct'].min()
        max_ret = reason_trades['net_return_pct'].max()
        print(f"{reason:<25} {count:>8} {win_rate:>10.1f}% {mean_ret:>10.2f} {min_ret:>10.2f} {max_ret:>10.2f}")

avg_trade = trades_df['net_return_pct'].mean()
print(f"\n**Average return per trade: {avg_trade:+.3f}%**")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
