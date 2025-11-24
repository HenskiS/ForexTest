"""
Plot backtest trades on OANDA price chart.

Shows entry/exit points with P&L for all trades from backtest results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=250, help='Test days used in backtest')
args = parser.parse_args()

PAIR = args.pair.upper()
TEST_DAYS = args.test_days

print(f"Plotting {PAIR} Backtest Trades ({TEST_DAYS} days)")
print("="*80)

# Load OANDA price data
price_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(price_file):
    print(f"ERROR: {price_file} not found")
    sys.exit(1)

df_price = pd.read_csv(price_file)
df_price['date'] = pd.to_datetime(df_price['date'])
df_price = df_price.set_index('date')

# Load backtest trade history
trade_file = f'{PAIR}_backtest_trades_{TEST_DAYS}days.csv'
if not os.path.exists(trade_file):
    print(f"ERROR: {trade_file} not found")
    print(f"Run backtest first: python backtest_oanda_data.py --pair {PAIR} --test-days {TEST_DAYS}")
    sys.exit(1)

df_trades = pd.read_csv(trade_file)
df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])

print(f"Found {len(df_trades)} trades")
print(f"Date range: {df_trades['entry_date'].min().date()} to {df_trades['exit_date'].max().date()}")
print()

# Filter price data to backtest period (with some padding)
start_date = df_trades['entry_date'].min() - pd.Timedelta(days=30)
end_date = df_trades['exit_date'].max() + pd.Timedelta(days=30)
df_price_period = df_price.loc[start_date:end_date]

if len(df_price_period) == 0:
    print(f"No price data for backtest period")
    sys.exit(1)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

# Plot 1: Candlestick chart with trades
from matplotlib.patches import Rectangle

for idx, row in df_price_period.iterrows():
    date = idx
    open_price = row['open']
    high_price = row['high']
    low_price = row['low']
    close_price = row['close']

    # Color: green if close > open, red otherwise
    candle_color = 'green' if close_price >= open_price else 'red'

    # Draw high-low line (wick)
    ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.8, zorder=1)

    # Draw open-close rectangle (body)
    body_height = abs(close_price - open_price)
    body_bottom = min(open_price, close_price)

    # Width of candle (0.6 days)
    width = pd.Timedelta(hours=14)
    rect = Rectangle((date - width/2, body_bottom), width, body_height,
                     facecolor=candle_color, edgecolor='black', linewidth=0.5, alpha=0.8, zorder=2)
    ax1.add_patch(rect)

# Plot entry and exit points
for idx, trade in df_trades.iterrows():
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    direction = trade['direction']
    pnl_pct = trade['net_return_pct']
    outcome = trade['outcome']

    # Color based on outcome
    color = 'green' if outcome == 'WIN' else 'red'

    # Entry marker
    if direction == 'LONG':
        ax1.scatter(entry_date, entry_price, color=color, marker='^', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5, alpha=0.8)
    else:  # SHORT
        ax1.scatter(entry_date, entry_price, color=color, marker='v', s=200,
                   edgecolors='black', linewidths=1.5, zorder=5, alpha=0.8)

    # Exit marker
    ax1.scatter(exit_date, exit_price, color=color, marker='x', s=200,
               linewidths=2.5, zorder=5, alpha=0.8)

    # Connect entry to exit with line
    ax1.plot([entry_date, exit_date], [entry_price, exit_price],
            color=color, linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)

    # Add P&L label
    mid_date = entry_date + (exit_date - entry_date) / 2
    mid_price = (entry_price + exit_price) / 2
    ax1.text(mid_date, mid_price, f'{pnl_pct:+.2f}%',
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))

ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'{PAIR} Price', fontsize=12, fontweight='bold')
ax1.set_title(f'{PAIR} Backtest Performance ({TEST_DAYS} days)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Create legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='Bullish Candle'),
    Patch(facecolor='red', edgecolor='black', label='Bearish Candle'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
           markersize=10, markeredgecolor='black', label='Long Entry (Win)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='green',
           markersize=10, markeredgecolor='black', label='Short Entry (Win)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
           markersize=10, markeredgecolor='black', label='Long Entry (Loss)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
           markersize=10, markeredgecolor='black', label='Short Entry (Loss)'),
    Line2D([0], [0], marker='x', color='black', markersize=10,
           markeredgewidth=2.5, linestyle='None', label='Exit')
]
ax1.legend(handles=legend_elements, loc='best', fontsize=10)

# Plot 2: Equity Curve
df_trades_sorted = df_trades.sort_values('exit_date')

# Calculate equity curve starting at $1000
starting_equity = 1000
equity_curve = [starting_equity]

for pnl_pct in df_trades_sorted['net_return_pct']:
    new_equity = equity_curve[-1] * (1 + pnl_pct / 100)
    equity_curve.append(new_equity)

# Create dates for equity curve (add starting date)
equity_dates = [df_trades_sorted['entry_date'].iloc[0]] + df_trades_sorted['exit_date'].tolist()

ax2.plot(equity_dates, equity_curve,
        color='green', linewidth=2.5, marker='o', markersize=4)
ax2.axhline(y=starting_equity, color='gray', linestyle='--', linewidth=1, label='Starting Capital')
ax2.fill_between(equity_dates, equity_curve, starting_equity,
                 where=[e >= starting_equity for e in equity_curve],
                 alpha=0.3, color='green', interpolate=True)
ax2.fill_between(equity_dates, equity_curve, starting_equity,
                 where=[e < starting_equity for e in equity_curve],
                 alpha=0.3, color='red', interpolate=True)

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Account Balance ($)', fontsize=12, fontweight='bold')
ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.legend(loc='best')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add statistics text box
total_return = df_trades_sorted['net_return_pct'].sum()
win_rate = len(df_trades_sorted[df_trades_sorted['outcome'] == 'WIN']) / len(df_trades_sorted) * 100
avg_win = df_trades_sorted[df_trades_sorted['outcome'] == 'WIN']['net_return_pct'].mean()
avg_loss = df_trades_sorted[df_trades_sorted['outcome'] == 'LOSS']['net_return_pct'].mean()
total_trades = len(df_trades_sorted)

# Calculate annualized return
days_elapsed = (df_trades_sorted['exit_date'].max() - df_trades_sorted['entry_date'].min()).days
years = days_elapsed / 365.25
if years > 0:
    equity_multiplier = (1 + total_return / 100)
    annual_return = (equity_multiplier ** (1 / years) - 1) * 100
else:
    annual_return = 0

stats_text = f'Total Trades: {total_trades}\n'
stats_text += f'Win Rate: {win_rate:.1f}%\n'
stats_text += f'Total Return: {total_return:+.2f}%\n'
stats_text += f'Annual Return: {annual_return:+.2f}%\n'
stats_text += f'Avg Win: {avg_win:.2f}%\n'
stats_text += f'Avg Loss: {avg_loss:.2f}%'

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save plot
output_file = f'{PAIR}_backtest_trades_{TEST_DAYS}days.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Show plot
plt.show()

print("="*80)
print("✓ Visualization complete!")
