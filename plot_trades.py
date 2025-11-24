"""
Plot actual trades on OANDA price chart.

Shows entry/exit points with P&L for all trades logged in the trade history.
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
parser.add_argument('--year', type=int, default=2025, help='Year to plot (default: 2025)')
args = parser.parse_args()

PAIR = args.pair.upper()
YEAR = args.year

print(f"Plotting {PAIR} Trades for {YEAR}")
print("="*80)

# Load OANDA price data
price_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(price_file):
    print(f"ERROR: {price_file} not found")
    sys.exit(1)

df_price = pd.read_csv(price_file)
df_price['date'] = pd.to_datetime(df_price['date'])
df_price = df_price.set_index('date')

# Load trade history
trade_file = f'data/oanda_cache/{PAIR}_trades.csv'
if not os.path.exists(trade_file):
    print(f"ERROR: {trade_file} not found")
    print("No trades have been executed yet.")
    sys.exit(1)

df_trades = pd.read_csv(trade_file)
df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])

# Filter trades by year
df_trades_year = df_trades[
    (df_trades['entry_date'].dt.year == YEAR) |
    (df_trades['exit_date'].dt.year == YEAR)
]

if len(df_trades_year) == 0:
    print(f"No trades found for {YEAR}")
    sys.exit(0)

print(f"Found {len(df_trades_year)} trades in {YEAR}")
print()

# Filter price data to year
start_date = f'{YEAR}-01-01'
end_date = f'{YEAR}-12-31'
df_price_year = df_price.loc[start_date:end_date]

if len(df_price_year) == 0:
    print(f"No price data for {YEAR}")
    sys.exit(1)

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

# Plot 1: Price chart with trades
ax1.plot(df_price_year.index, df_price_year['close'],
         color='black', linewidth=1.5, label='Close Price', zorder=1)

# Plot entry and exit points
for idx, trade in df_trades_year.iterrows():
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    direction = trade['direction']
    pnl_pct = trade['pnl_pct']
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
ax1.set_title(f'{PAIR} Trading Performance - {YEAR}', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Create legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', linewidth=1.5, label='Close Price'),
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

# Plot 2: Cumulative P&L
df_trades_year_sorted = df_trades_year.sort_values('exit_date')
df_trades_year_sorted['cumulative_pnl'] = df_trades_year_sorted['pnl_pct'].cumsum()

# Create equity curve (starting at 100%)
equity_curve = [100]
for pnl in df_trades_year_sorted['pnl_pct']:
    equity_curve.append(equity_curve[-1] * (1 + pnl / 100))

ax2.plot(df_trades_year_sorted['exit_date'], df_trades_year_sorted['cumulative_pnl'],
        color='blue', linewidth=2, marker='o', markersize=6)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.fill_between(df_trades_year_sorted['exit_date'],
                 df_trades_year_sorted['cumulative_pnl'],
                 0, alpha=0.3, color='blue')

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative P&L (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Performance', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add statistics text box
total_return = df_trades_year_sorted['pnl_pct'].sum()
win_rate = len(df_trades_year_sorted[df_trades_year_sorted['outcome'] == 'WIN']) / len(df_trades_year_sorted) * 100
avg_win = df_trades_year_sorted[df_trades_year_sorted['outcome'] == 'WIN']['pnl_pct'].mean()
avg_loss = df_trades_year_sorted[df_trades_year_sorted['outcome'] == 'LOSS']['pnl_pct'].mean()
total_trades = len(df_trades_year_sorted)

stats_text = f'Total Trades: {total_trades}\n'
stats_text += f'Win Rate: {win_rate:.1f}%\n'
stats_text += f'Total Return: {total_return:+.2f}%\n'
stats_text += f'Avg Win: {avg_win:.2f}%\n'
stats_text += f'Avg Loss: {avg_loss:.2f}%'

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save plot
output_file = f'{PAIR}_trades_{YEAR}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Show plot
plt.show()

print("="*80)
print("✓ Visualization complete!")
