"""
Analyze yearly performance of optimal strategy for any currency pair.

Usage:
    python analyze_yearly_performance_pair.py GBPUSD
    python analyze_yearly_performance_pair.py AUDUSD
"""

import sys
import pandas as pd
import numpy as np
import pickle

if len(sys.argv) < 2:
    print("Usage: python analyze_yearly_performance_pair.py <CURRENCY_PAIR>")
    sys.exit(1)

CURRENCY_PAIR = sys.argv[1].upper()

print(f"YEARLY PERFORMANCE ANALYSIS - {CURRENCY_PAIR}")
print("="*80)

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

with open(f'xgboost_results_{CURRENCY_PAIR}_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

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

    equity_curve = [capital]
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
                    'entry': entry_price,
                    'exit': exit_price,
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

        equity_curve.append(capital)

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
            'entry': entry_price,
            'exit': exit_price,
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

# Run backtest with 1-day cooldown
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

# Convert trades to DataFrame
trades_df = pd.DataFrame(all_trades)
trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year

# Calculate yearly statistics
print("\nYEARLY PERFORMANCE BREAKDOWN")
print("="*80)
print(f"{'Year':<6} {'Trades':<8} {'Win%':<8} {'Return':<12} {'Capital':<12} {'Best Trade':<12} {'Worst Trade':<12}")
print("-"*90)

yearly_capital = 1000.0
for year in sorted(trades_df['year'].unique()):
    year_trades = trades_df[trades_df['year'] == year]

    n_trades = len(year_trades)
    winning_trades = year_trades[year_trades['pnl'] > 0]
    win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0

    year_return = (1 + year_trades['return']).prod() - 1
    yearly_capital = yearly_capital * (1 + year_return)

    best_trade = year_trades['pnl_pct'].max() if n_trades > 0 else 0
    worst_trade = year_trades['pnl_pct'].min() if n_trades > 0 else 0

    print(f"{year:<6} {n_trades:<8} {win_rate*100:6.1f}% {year_return*100:10.2f}% ${yearly_capital:10.2f} {best_trade*100:10.2f}% {worst_trade*100:11.2f}%")

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

total_trades = len(trades_df)
winning_trades = trades_df[trades_df['pnl'] > 0]
losing_trades = trades_df[trades_df['pnl'] <= 0]

print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
print(f"Win Rate: {len(winning_trades)/total_trades*100:.1f}%")
print(f"Average Win: {winning_trades['pnl_pct'].mean()*100:.3f}%")
print(f"Average Loss: {losing_trades['pnl_pct'].mean()*100:.3f}%")
print(f"Best Trade: {trades_df['pnl_pct'].max()*100:.2f}%")
print(f"Worst Trade: {trades_df['pnl_pct'].min()*100:.2f}%")

# Verify capital calculation
compounded_capital = 1000.0
for ret in trades_df['return']:
    compounded_capital *= (1 + ret)

print(f"\nFinal Capital (from yearly compounding): ${yearly_capital:.2f}")
print(f"Final Capital (from all returns compounded): ${compounded_capital:.2f}")
print(f"Total Return (from yearly): {(yearly_capital - 1000)/1000*100:.2f}%")

# Calculate drawdown by year
print("\n" + "="*80)
print("DRAWDOWN BY YEAR")
print("="*80)
print(f"{'Year':<6} {'Max DD':<10} {'Recovery Days':<15}")
print("-"*40)

current_capital = 1000.0

for year in sorted(trades_df['year'].unique()):
    year_trades = trades_df[trades_df['year'] == year].sort_values('date')

    equity = [current_capital]
    for _, trade in year_trades.iterrows():
        current_capital = current_capital * (1 + trade['return'])
        equity.append(current_capital)

    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)

    # Find recovery days (rough estimate)
    dd_idx = np.argmin(drawdown)
    recovery_idx = np.where(equity[dd_idx:] >= running_max[dd_idx])[0]
    recovery_days = recovery_idx[0] * 3 if len(recovery_idx) > 0 else -1

    print(f"{year:<6} {max_dd*100:8.2f}% {recovery_days if recovery_days >= 0 else 'Not recovered':>14}")

print("\n" + "="*80)
