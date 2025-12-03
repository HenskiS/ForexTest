"""
Calculate TRUE portfolio max drawdown by simulating continuous equity curve
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ASSETS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',
    'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'
]

print("="*100)
print("TRUE PORTFOLIO MAX DRAWDOWN CALCULATION")
print("="*100)
print()

# Load all trades and get date range
all_trades = []
min_date = None
max_date = None

for asset in ASSETS:
    try:
        df = pd.read_csv(f'{asset}_backtest_trades_750days.csv')
        df['asset'] = asset
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        all_trades.append(df)

        if min_date is None or df['entry_date'].min() < min_date:
            min_date = df['entry_date'].min()
        if max_date is None or df['exit_date'].max() > max_date:
            max_date = df['exit_date'].max()
    except Exception as e:
        print(f"Error loading {asset}: {e}")

all_trades_df = pd.concat(all_trades, ignore_index=True)
print(f"Loaded {len(all_trades_df)} trades")
print(f"Date range: {min_date.date()} to {max_date.date()}")
print()

# Create daily equity curve
# Start with $1000, 10% allocation per asset ($100 each)
daily_equity = {}
current_date = min_date.date()
end_date = max_date.date()

equity = 1000.0
capital_per_asset = 100.0  # 10% each

print("Simulating day-by-day portfolio equity...")

day_count = 0
while current_date <= end_date:
    # Check each asset for trades that close today
    daily_pnl = 0

    for asset in ASSETS:
        asset_trades = all_trades_df[all_trades_df['asset'] == asset]
        # Trades that exit on this date
        closing_trades = asset_trades[asset_trades['exit_date'].dt.date == current_date]

        for _, trade in closing_trades.iterrows():
            # Calculate P&L for this trade with 10% allocation
            pnl_pct = trade['net_return_pct']
            pnl_dollars = capital_per_asset * (pnl_pct / 100)
            daily_pnl += pnl_dollars

    equity += daily_pnl
    daily_equity[current_date] = equity

    current_date += timedelta(days=1)
    day_count += 1

print(f"Processed {day_count} days")
print()

# Calculate drawdown from equity curve
equity_df = pd.DataFrame(list(daily_equity.items()), columns=['date', 'equity'])
equity_df = equity_df.sort_values('date')
equity_df['peak'] = equity_df['equity'].expanding().max()
equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100

max_dd = equity_df['drawdown'].min()
max_dd_date = equity_df[equity_df['drawdown'] == max_dd]['date'].iloc[0]

final_equity = equity_df['equity'].iloc[-1]
total_return = (final_equity / 1000 - 1) * 100

print("="*100)
print("RESULTS")
print("="*100)
print()
print(f"Starting Equity: $1,000.00")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print()
print(f"Portfolio Max Drawdown (1x leverage): {max_dd:.2f}%")
print(f"Max DD occurred on: {max_dd_date}")
print()

# Compare to individual assets
print("Individual Asset Max DDs (for comparison):")
print("-"*100)
individual_dds = []
for asset in ASSETS:
    asset_trades = all_trades_df[all_trades_df['asset'] == asset].copy()
    asset_trades = asset_trades.sort_values('entry_date')
    asset_trades['cum_return'] = (1 + asset_trades['net_return_pct'] / 100).cumprod()
    asset_trades['peak'] = asset_trades['cum_return'].expanding().max()
    asset_trades['dd'] = (asset_trades['cum_return'] - asset_trades['peak']) / asset_trades['peak'] * 100
    asset_dd = asset_trades['dd'].min()
    individual_dds.append(asset_dd)
    print(f"{asset:12} {asset_dd:6.2f}%")

avg_individual_dd = np.mean(individual_dds)
print(f"\nAverage Individual DD: {avg_individual_dd:.2f}%")
print(f"Portfolio DD: {max_dd:.2f}%")
print(f"Diversification Benefit: {avg_individual_dd - max_dd:.2f}% ({abs((max_dd/avg_individual_dd - 1)*100):.1f}% improvement)")

print()
print("="*100)
print("LEVERAGE SCENARIOS")
print("="*100)
print()
print(f"1x Leverage: Max DD = {max_dd:.2f}%")
print(f"2x Leverage: Max DD ≈ {max_dd * 2:.2f}%")
print(f"3x Leverage: Max DD ≈ {max_dd * 3:.2f}%")
print(f"5x Leverage: Max DD ≈ {max_dd * 5:.2f}%")
print(f"10x Leverage: Max DD ≈ {max_dd * 10:.2f}% ⚠️")
print()
print("Note: Keep max DD under 50% for safety. With portfolio DD of {:.1f}%, you could theoretically".format(abs(max_dd)))
print("      use up to {:.1f}x leverage, but 2-3x is recommended for safety margin.".format(50/abs(max_dd)))
print()
print("="*100)
