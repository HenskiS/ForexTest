"""
Analyze portfolio diversification effects on drawdown and daily winners
"""
import pandas as pd
import numpy as np
import sys
import io
from datetime import datetime, timedelta

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ASSETS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
]

ASSET_NAMES = {
    'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'AUDUSD': 'AUD/USD', 'USDJPY': 'USD/JPY',
    'XAUUSD': 'Gold', 'XAGUSD': 'Silver', 'XPTUSD': 'Platinum', 'XPDUSD': 'Palladium', 'XCUUSD': 'Copper',
    'SUGARUSD': 'Sugar', 'SPX500USD': 'S&P 500', 'DE30EUR': 'DAX', 'WTICOUSD': 'WTI Oil', 'BCOUSD': 'Brent Oil'
}

print("="*100)
print("PORTFOLIO DIVERSIFICATION ANALYSIS")
print("="*100)
print()

# Load all trade data
all_trades = {}
for asset in ASSETS:
    try:
        df = pd.read_csv(f'{asset}_backtest_trades_750days.csv')
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            df['date'] = df['entry_date'].dt.date
            all_trades[asset] = df
    except Exception as e:
        print(f"Warning: Could not load {asset}: {e}")

if not all_trades:
    print("No trade data found!")
    sys.exit(1)

# Part 1: Individual Asset Max Drawdowns
print("INDIVIDUAL ASSET MAX DRAWDOWNS (1x leverage)")
print("-"*100)

individual_dds = {}
for asset, df in all_trades.items():
    # Calculate cumulative equity curve
    df = df.sort_values('entry_date')
    df['cumulative_return'] = (1 + df['net_return_pct'] / 100).cumprod()
    df['peak'] = df['cumulative_return'].expanding().max()
    df['drawdown'] = (df['cumulative_return'] - df['peak']) / df['peak'] * 100
    max_dd = df['drawdown'].min()
    individual_dds[asset] = max_dd
    print(f"{ASSET_NAMES[asset]:20} Max DD: {max_dd:6.2f}%")

avg_individual_dd = np.mean(list(individual_dds.values()))
print(f"\nAverage Individual Max DD: {avg_individual_dd:.2f}%")

# Part 2: Portfolio Drawdown (10% allocation each)
print("\n")
print("="*100)
print("PORTFOLIO-LEVEL DRAWDOWN (10% allocation per asset, 1x leverage)")
print("-"*100)

# Combine all trades by date
all_dates = set()
for df in all_trades.values():
    all_dates.update(df['date'].unique())
all_dates = sorted(all_dates)

# Calculate daily portfolio returns
portfolio_equity = 1000.0  # Start with $1000
portfolio_values = [portfolio_equity]
dates_list = [all_dates[0]]

for date in all_dates:
    daily_return = 0
    for asset, df in all_trades.items():
        day_trades = df[df['date'] == date]
        if len(day_trades) > 0:
            # 10% allocation per asset
            daily_return += day_trades['net_return_pct'].sum() * 0.10

    portfolio_equity *= (1 + daily_return / 100)
    portfolio_values.append(portfolio_equity)
    dates_list.append(date)

# Calculate portfolio drawdown
portfolio_df = pd.DataFrame({
    'date': dates_list,
    'equity': portfolio_values
})
portfolio_df['peak'] = portfolio_df['equity'].expanding().max()
portfolio_df['drawdown'] = (portfolio_df['equity'] - portfolio_df['peak']) / portfolio_df['peak'] * 100
portfolio_max_dd = portfolio_df['drawdown'].min()

print(f"Portfolio Max Drawdown: {portfolio_max_dd:.2f}%")
print(f"Individual Asset Avg Max DD: {avg_individual_dd:.2f}%")
print(f"\nDiversification Benefit: {avg_individual_dd - portfolio_max_dd:.2f}% reduction in max DD")
print(f"Improvement: {(1 - portfolio_max_dd/avg_individual_dd)*100:.1f}% better than average asset")

# Part 3: Daily Winners Analysis
print("\n")
print("="*100)
print("DAILY WINNERS ANALYSIS")
print("-"*100)

# Count winners per day
daily_winners = {}
daily_trades = {}
for date in all_dates:
    winners = 0
    total = 0
    for asset, df in all_trades.items():
        day_trades = df[df['date'] == date]
        if len(day_trades) > 0:
            total += len(day_trades)
            winners += len(day_trades[day_trades['outcome'] == 'WIN'])
    daily_winners[date] = winners
    daily_trades[date] = total

winners_list = list(daily_winners.values())
trades_list = list(daily_trades.values())

print(f"Average Winners Per Day: {np.mean(winners_list):.2f}")
print(f"Average Trades Per Day: {np.mean(trades_list):.2f}")
print(f"Daily Win Rate: {np.mean(winners_list)/np.mean(trades_list)*100:.1f}%")
print(f"\nMin Winners In A Day: {np.min(winners_list)}")
print(f"Max Winners In A Day: {np.max(winners_list)}")
print(f"\nDays with 0 winners: {sum(1 for w in winners_list if w == 0)}")
print(f"Days with 5+ winners: {sum(1 for w in winners_list if w >= 5)}")
print(f"Days with 8+ winners: {sum(1 for w in winners_list if w >= 8)}")

# Part 4: Month-over-Month Projections for $500 account at 2x leverage
print("\n")
print("="*100)
print("MONTH-OVER-MONTH PROJECTION ($500 ACCOUNT, 2x LEVERAGE)")
print("-"*100)

# Calculate monthly returns from backtest
monthly_returns = {}
for date in all_dates:
    month_key = f"{date.year}-{date.month:02d}"
    if month_key not in monthly_returns:
        monthly_returns[month_key] = 0

    for asset, df in all_trades.items():
        day_trades = df[df['date'] == date]
        if len(day_trades) > 0:
            # 10% allocation per asset
            monthly_returns[month_key] += day_trades['net_return_pct'].sum() * 0.10

# Sort by date
sorted_months = sorted(monthly_returns.keys())

# Average monthly return (1x leverage)
monthly_return_1x = np.mean(list(monthly_returns.values()))
monthly_return_2x = monthly_return_1x * 2

print(f"\nAverage Monthly Return (1x leverage): {monthly_return_1x:.2f}%")
print(f"Average Monthly Return (2x leverage): {monthly_return_2x:.2f}%")

# Project 12 months forward
print(f"\nMonth-by-Month Projection (starting with $500):")
print("-"*100)
print(f"{'Month':<10} {'Balance':>12} {'Monthly Gain':>14} {'Monthly %':>12} {'Total Gain':>12}")
print("-"*100)

balance = 500.0
initial = 500.0

for month in range(1, 13):
    monthly_gain = balance * monthly_return_2x / 100
    balance += monthly_gain
    total_gain = balance - initial

    print(f"Month {month:<3}  ${balance:>10,.2f}  ${monthly_gain:>12,.2f}  {monthly_return_2x:>10.2f}%  ${total_gain:>10,.2f}")

print("-"*100)
print(f"Year End Balance: ${balance:,.2f}")
print(f"Total Gain: ${balance - initial:,.2f}")
print(f"Annual Return: {(balance/initial - 1)*100:.1f}%")

# Conservative and aggressive scenarios
print("\n")
print("="*100)
print("SCENARIO ANALYSIS")
print("="*100)

# Get actual monthly volatility
monthly_returns_list = list(monthly_returns.values())
monthly_std = np.std(monthly_returns_list)
monthly_min = np.min(monthly_returns_list)
monthly_max = np.max(monthly_returns_list)

print(f"\nHistorical Monthly Performance (1x leverage):")
print(f"  Average: {monthly_return_1x:.2f}%")
print(f"  Std Dev: {monthly_std:.2f}%")
print(f"  Best Month: {monthly_max:.2f}%")
print(f"  Worst Month: {monthly_min:.2f}%")

# Project conservative (avg - 1 std), average, and aggressive (avg + 1 std)
scenarios = {
    'Conservative': (monthly_return_1x - monthly_std) * 2,
    'Expected': monthly_return_1x * 2,
    'Optimistic': (monthly_return_1x + monthly_std) * 2,
}

print(f"\n12-Month Projections at 2x Leverage:")
print("-"*60)
print(f"{'Scenario':<15} {'Monthly %':>12} {'Year-End Balance':>20}")
print("-"*60)

for scenario, monthly_pct in scenarios.items():
    balance = 500 * (1 + monthly_pct/100)**12
    print(f"{scenario:<15} {monthly_pct:>10.2f}%  ${balance:>18,.2f}")

print("\n")
print("="*100)
