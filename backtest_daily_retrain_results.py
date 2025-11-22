"""
Backtest the daily retraining results for 2025.
"""

import pandas as pd
import numpy as np
import pickle
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD, USDJPY, AUDUSD)')
args = parser.parse_args()

PAIR = args.pair.upper()

print(f"Backtesting 2025 Daily Retrain Results: {PAIR}")
print("="*70)

# Load results - try rolling window first, then expanding window
try:
    results_file = f'rolling_daily_2025_{PAIR}.pkl'
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    retrain_type = "ROLLING DAILY"
except FileNotFoundError:
    try:
        results_file = f'rolling_weekly_2025_{PAIR}.pkl'
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        retrain_type = "ROLLING WEEKLY"
    except FileNotFoundError:
        try:
            results_file = f'weekly_retrain_2025_{PAIR}.pkl'
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            retrain_type = "EXPANDING WEEKLY"
        except FileNotFoundError:
            results_file = f'daily_retrain_2025_{PAIR}.pkl'
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            retrain_type = "EXPANDING DAILY"

# Load price data
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Convert dates back to datetime
dates = pd.to_datetime(results['dates'])
signals = np.array(results['signals'])
predictions = np.array(results['predictions'])
actuals = np.array(results['actuals'])

print(f"\nLoaded {len(dates)} days of predictions")
print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

# Backtest configuration (matching main backtest)
INITIAL_CAPITAL = 1000.0
BASE_STOP_LOSS = 0.004  # 0.40%
BASE_TAKE_PROFIT = 0.010  # 1.00%
MAX_HOLDING_DAYS = 5
LOSS_COOLDOWN_DAYS = 1
TRANSACTION_COST = 0.0002  # 0.02% or 2 pips

def backtest_strategy(df, dates, signals):
    """Backtest with volatility-adjusted stops and loss cooldown."""
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = [INITIAL_CAPITAL]

    cooldown_until = None

    for i, date in enumerate(dates):
        signal = signals[i]

        # Check if we're in cooldown
        if cooldown_until and date < cooldown_until:
            equity_curve.append(capital)
            continue

        # Manage existing position
        if position is not None:
            entry_date = position['entry_date']
            entry_price = position['entry_price']
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']

            current_price = df.loc[date, 'close']
            days_held = (date - entry_date).days

            # Calculate P&L
            if direction == 1:  # Long
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
            exit_reason = None
            if direction == 1:  # Long
                if current_price <= stop_loss:
                    exit_reason = 'stop_loss'
                elif current_price >= take_profit:
                    exit_reason = 'take_profit'
            else:  # Short
                if current_price >= stop_loss:
                    exit_reason = 'stop_loss'
                elif current_price <= take_profit:
                    exit_reason = 'take_profit'

            if days_held >= MAX_HOLDING_DAYS:
                exit_reason = 'max_holding'

            # Exit position
            if exit_reason:
                # Subtract transaction cost
                pnl_pct -= TRANSACTION_COST

                capital *= (1 + pnl_pct)

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'days_held': days_held
                })

                # Loss cooldown
                if pnl_pct < 0:
                    cooldown_until = date + pd.Timedelta(days=LOSS_COOLDOWN_DAYS)

                position = None

        # Enter new position if no current position and we have a signal
        if position is None and signal != 0:
            entry_price = df.loc[date, 'close']
            atr = df.loc[date, 'atr']
            median_atr = df.loc[:date, 'atr'].median()
            vol_multiplier = atr / median_atr if median_atr > 0 else 1.0

            # Adjust stops for volatility
            adj_stop = BASE_STOP_LOSS * vol_multiplier
            adj_tp = BASE_TAKE_PROFIT * vol_multiplier

            if signal == 1:  # Long
                stop_loss = entry_price * (1 - adj_stop)
                take_profit = entry_price * (1 + adj_tp)
            else:  # Short
                stop_loss = entry_price * (1 + adj_stop)
                take_profit = entry_price * (1 - adj_tp)

            # Subtract transaction cost for entry
            capital *= (1 - TRANSACTION_COST)

            position = {
                'entry_date': date,
                'entry_price': entry_price,
                'direction': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

        equity_curve.append(capital)

    return {
        'final_capital': capital,
        'trades': trades,
        'equity_curve': equity_curve
    }

# Run backtest
print("\nRunning backtest...")
backtest = backtest_strategy(df, dates, signals)

final_capital = backtest['final_capital']
trades = backtest['trades']

total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print(f"\n{'='*70}")
print(f"2025 {retrain_type} RETRAIN RESULTS")
print(f"{'='*70}")
print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
print(f"Final Capital:   ${final_capital:.2f}")
print(f"Total Return:    {total_return:.2f}%")
print(f"Total Trades:    {len(trades)}")

if trades:
    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    losing_trades = [t for t in trades if t['pnl_pct'] < 0]

    win_rate = len(winning_trades) / len(trades) * 100

    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) * 100 if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) * 100 if losing_trades else 0

    print(f"\nWin Rate:        {win_rate:.1f}%")
    print(f"Winning Trades:  {len(winning_trades)}")
    print(f"Losing Trades:   {len(losing_trades)}")
    print(f"Average Win:     {avg_win:.2f}%")
    print(f"Average Loss:    {avg_loss:.2f}%")

    if avg_loss != 0:
        profit_factor = (len(winning_trades) * avg_win) / (len(losing_trades) * abs(avg_loss))
        print(f"Profit Factor:   {profit_factor:.2f}")

    # Exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print(f"\nExit Reasons:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({100*count/len(trades):.1f}%)")

# Calculate max drawdown
equity = np.array(backtest['equity_curve'])
running_max = np.maximum.accumulate(equity)
drawdown = (equity - running_max) / running_max * 100
max_dd = drawdown.min()

print(f"\nMax Drawdown:    {max_dd:.2f}%")

# Annualize the return (2025 is ~10 months so far)
days_traded = (dates[-1] - dates[0]).days
years = days_traded / 365.25
annual_return = ((final_capital / INITIAL_CAPITAL) ** (1/years) - 1) * 100

print(f"Days Traded:     {days_traded}")
print(f"Annualized Return: {annual_return:.2f}%")

print(f"\n{'='*70}")
print("Backtest complete!")
print(f"{'='*70}")
