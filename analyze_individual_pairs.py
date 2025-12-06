"""
Analyze individual pair performance from actual spreads backtest
Uses the pickle file from 4500-day backtest
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

TEST_DAYS = 4500
pickle_file = f'backtest_results_actual_spreads_{TEST_DAYS}days.pkl'

if not Path(pickle_file).exists():
    print(f"ERROR: {pickle_file} not found")
    print("Please run backtest_multi_pair_actual_spreads.py first")
    exit(1)

print("="*100)
print("INDIVIDUAL PAIR PERFORMANCE ANALYSIS - WITH ACTUAL SPREADS")
print("="*100)
print()

# Load backtest results
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

pair_returns = data['pair_returns']
test_dates = pd.to_datetime(data['test_dates'])

print(f"Analyzing {len(test_dates)} days from {test_dates.min().date()} to {test_dates.max().date()}")
print()

# Calculate performance for each pair
pair_stats = []

for pair, daily_returns in pair_returns.items():
    # Convert to numpy array if not already
    daily_returns = np.array(daily_returns)

    # Calculate cumulative returns (starting with $1000)
    equity_curve = np.cumprod(1 + daily_returns) * 1000
    final_equity = equity_curve[-1]
    total_return = (final_equity - 1000) / 1000 * 100

    # Calculate annual return
    years = len(daily_returns) / 250
    annual_return = ((final_equity / 1000) ** (1 / years) - 1) * 100

    # Calculate Sharpe ratio
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Calculate max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax * 100
    max_dd = drawdowns.min()

    # Win rate
    winning_days = (daily_returns > 0).sum()
    win_rate = winning_days / len(daily_returns) * 100

    pair_stats.append({
        'pair': pair,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'final_equity': final_equity
    })

    print(f"{pair}:")
    print(f"  Total Return: {total_return:,.2f}%")
    print(f"  Annual Return: {annual_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Final Equity: ${final_equity:,.0f}")
    print()

# Sort by total return
pair_stats.sort(key=lambda x: x['total_return'], reverse=True)

print("="*100)
print("RANKING BY TOTAL RETURN:")
print("="*100)
print()

for i, stat in enumerate(pair_stats, 1):
    print(f"{i}. {stat['pair']}: {stat['total_return']:,.2f}% total return ({stat['annual_return']:.2f}% annual)")

print()
print("="*100)
print("MARKDOWN TABLE FORMAT:")
print("="*100)
print()
print("| Pair | Total Return | Annual Return | Sharpe | Max DD | Win Rate |")
print("|------|--------------|---------------|--------|--------|----------|")

for stat in pair_stats:
    print(f"| {stat['pair']} | {stat['total_return']:,.0f}% | "
          f"{stat['annual_return']:.2f}% | "
          f"{stat['sharpe']:.2f} | "
          f"{stat['max_dd']:.2f}% | "
          f"{stat['win_rate']:.1f}% |")

print()

# Calculate contribution percentages (normalized)
total_equity_gain = sum(stat['final_equity'] - 1000 for stat in pair_stats)

print("CONTRIBUTION TO PORTFOLIO:")
print("-" * 100)
for stat in pair_stats:
    contribution = ((stat['final_equity'] - 1000) / total_equity_gain) * 100
    print(f"{stat['pair']}: {contribution:.1f}%")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
