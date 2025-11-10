"""
Quick test of currency pair with optimal strategy.

Usage:
    python test_currency_pair.py GBPUSD
    python test_currency_pair.py AUDUSD
"""

import sys
import pandas as pd
import numpy as np
import pickle

if len(sys.argv) < 2:
    print("Usage: python test_currency_pair.py <CURRENCY_PAIR>")
    sys.exit(1)

CURRENCY_PAIR = sys.argv[1].upper()

print(f"\n{'='*80}")
print(f"TESTING {CURRENCY_PAIR} WITH OPTIMAL STRATEGY")
print(f"{'='*80}\n")

# Load data
df = pd.read_csv(f'data/{CURRENCY_PAIR}_1day_with_features_FIXED_multitarget.csv',
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

# Load model results
results_file = f'xgboost_results_{CURRENCY_PAIR}_target_5day_return.pkl'
with open(results_file, 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Configuration
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


def generate_signals_regression(predictions):
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals, q1, q3


def backtest_window_with_cooldown(actuals, signals, df_prices, test_indices,
                                   initial_capital,
                                   base_stop_loss_pct=0.0040,
                                   base_take_profit_pct=0.0100,
                                   loss_cooldown_days=1,
                                   transaction_cost_pct=0.0002):
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    cooldown_remaining = 0

    returns = []
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    atr = test_data['atr'].values
    dates = test_data.index

    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        date = dates[i]

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Adjust stops by volatility
        if not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        if position != 0:
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            exit_triggered = False
            exit_price = None
            exit_reason = None

            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            elif pct_high >= take_profit_pct:
                exit_triggered = True
                exit_reason = 'take_profit'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            if exit_triggered:
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = entry_capital * pnl_pct
                cost = capital * transaction_cost_pct
                pnl -= cost
                capital += pnl

                trade_return = pnl / (capital - pnl)
                returns.append(trade_return)
                trades.append({
                    'date': date,
                    'type': 'long' if position == 1 else 'short',
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                if pnl < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0

        if position == 0 and signal != 0 and cooldown_remaining == 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital

    # Close final position
    if position != 0:
        exit_price = closes[-1]

        if position == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl = entry_capital * pnl_pct
        cost = capital * transaction_cost_pct
        pnl -= cost
        capital += pnl

        trade_return = pnl / (capital - pnl)
        returns.append(trade_return)
        trades.append({
            'date': dates[-1],
            'type': 'long' if position == 1 else 'short',
            'pnl_pct': pnl_pct,
            'pnl': pnl,
            'return': trade_return,
            'exit_reason': 'end_of_period'
        })

    final_return = (capital - initial_capital) / initial_capital

    return {
        'trades': trades,
        'returns': returns,
        'final_capital': capital,
        'final_return': final_return
    }


# Generate windows
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)

# Run backtest with optimal strategy (1-day cooldown)
capital = 1000.0
all_trades = []

for window_idx, result in enumerate(all_results):
    window = windows[window_idx]
    test_start = window['test_start']
    test_end = window['test_end']
    test_indices = range(test_start, test_end)

    signals, q1, q3 = generate_signals_regression(result['predictions'])

    backtest = backtest_window_with_cooldown(
        result['actuals'],
        signals,
        df_clean,
        test_indices,
        capital,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002
    )

    all_trades.extend(backtest['trades'])
    capital = backtest['final_capital']

# Calculate overall statistics
print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)

trades_df = pd.DataFrame(all_trades)
total_trades = len(trades_df)
winning_trades = trades_df[trades_df['pnl'] > 0]
losing_trades = trades_df[trades_df['pnl'] <= 0]

print(f"\nTotal Trades: {total_trades}")
print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")

print(f"\nAverage Win: {winning_trades['pnl_pct'].mean()*100:.3f}%")
print(f"Average Loss: {losing_trades['pnl_pct'].mean()*100:.3f}%")
print(f"Best Trade: {trades_df['pnl_pct'].max()*100:.2f}%")
print(f"Worst Trade: {trades_df['pnl_pct'].min()*100:.2f}%")

# Calculate metrics
total_return = (capital - 1000) / 1000
years = (windows[-1]['date_end'] - windows[0]['date_start']).days / 365.25
annual_return = (1 + total_return) ** (1 / years) - 1

returns_arr = np.array(trades_df['return'].values)
sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(total_trades / years)

# Max drawdown
equity_curve = [1000.0]
running_capital = 1000.0
for ret in trades_df['return']:
    running_capital *= (1 + ret)
    equity_curve.append(running_capital)

equity_arr = np.array(equity_curve)
running_max = np.maximum.accumulate(equity_arr)
drawdown = (equity_arr - running_max) / running_max
max_dd = np.min(drawdown)

# Profit factor
total_wins = winning_trades['pnl'].sum()
total_losses = abs(losing_trades['pnl'].sum())
profit_factor = total_wins / total_losses if total_losses > 0 else 0

print(f"\n{'='*80}")
print("PERFORMANCE METRICS")
print(f"{'='*80}")
print(f"Initial Capital:      ${1000:.2f}")
print(f"Final Capital:        ${capital:.2f}")
print(f"Total Return:         {total_return*100:.2f}%")
print(f"Annualized Return:    {annual_return*100:.2f}%")
print(f"Sharpe Ratio:         {sharpe:.3f}")
print(f"Max Drawdown:         {max_dd*100:.2f}%")
print(f"Profit Factor:        {profit_factor:.3f}")
print(f"Total Trades:         {total_trades}")
print(f"Trades per year:      {total_trades/years:.0f}")
print(f"Time period:          {years:.2f} years")

# Compare to EURUSD
print(f"\n{'='*80}")
print(f"COMPARISON: {CURRENCY_PAIR} vs EURUSD")
print(f"{'='*80}")
print(f"{'Metric':<25} {CURRENCY_PAIR:<20} {'EURUSD':<20}")
print("-"*65)
print(f"{'Annualized Return':<25} {annual_return*100:>8.2f}% {10.52:>19.2f}%")
print(f"{'Sharpe Ratio':<25} {sharpe:>18.3f} {1.057:>19.3f}")
print(f"{'Max Drawdown':<25} {max_dd*100:>8.2f}% {-7.7:>19.2f}%")
print(f"{'Win Rate':<25} {len(winning_trades)/total_trades*100:>8.1f}% {39.0:>19.1f}%")
print(f"{'Profit Factor':<25} {profit_factor:>18.3f} {1.525:>19.3f}")
print(f"{'Total Return':<25} {total_return*100:>8.2f}% {954.7:>19.2f}%")

print("\n" + "="*80)
