"""
Plot win rate evolution over time.

Shows rolling win rate and cumulative win rate to visualize strategy consistency.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=250, help='Test days used in backtest')
parser.add_argument('--rolling-window', type=int, default=20, help='Rolling window for win rate (trades)')
args = parser.parse_args()

PAIR = args.pair.upper()
TEST_DAYS = args.test_days
ROLLING_WINDOW = args.rolling_window

print(f"Plotting {PAIR} Win Rate Over Time")
print("="*80)

# Load backtest trade history
trade_file = f'{PAIR}_backtest_trades_{TEST_DAYS}days.csv'
if not os.path.exists(trade_file):
    print(f"ERROR: {trade_file} not found")
    print(f"Run backtest first: python backtest_oanda_data.py --pair {PAIR} --test-days {TEST_DAYS}")
    sys.exit(1)

df_trades = pd.read_csv(trade_file)
df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])

# Sort by exit date
df_trades = df_trades.sort_values('exit_date').reset_index(drop=True)

print(f"Found {len(df_trades)} trades")
print(f"Date range: {df_trades['entry_date'].min().date()} to {df_trades['exit_date'].max().date()}")
print()

# Create binary win/loss column
df_trades['win'] = (df_trades['outcome'] == 'WIN').astype(int)

# Calculate cumulative win rate
df_trades['trade_number'] = range(1, len(df_trades) + 1)
df_trades['cumulative_wins'] = df_trades['win'].cumsum()
df_trades['cumulative_win_rate'] = (df_trades['cumulative_wins'] / df_trades['trade_number']) * 100

# Calculate rolling win rate (last N trades)
df_trades['rolling_win_rate'] = df_trades['win'].rolling(window=ROLLING_WINDOW, min_periods=1).mean() * 100

# Create plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Cumulative Win Rate Over Time
ax1.plot(df_trades['exit_date'], df_trades['cumulative_win_rate'],
        color='blue', linewidth=2, label='Cumulative Win Rate')
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% (Breakeven)')
ax1.fill_between(df_trades['exit_date'], df_trades['cumulative_win_rate'], 50,
                 where=(df_trades['cumulative_win_rate'] >= 50), alpha=0.3, color='green', interpolate=True)
ax1.fill_between(df_trades['exit_date'], df_trades['cumulative_win_rate'], 50,
                 where=(df_trades['cumulative_win_rate'] < 50), alpha=0.3, color='red', interpolate=True)

ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
ax1.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title(f'Cumulative Win Rate Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Rolling Win Rate (Last N Trades)
ax2.plot(df_trades['exit_date'], df_trades['rolling_win_rate'],
        color='purple', linewidth=2, label=f'Rolling Win Rate ({ROLLING_WINDOW} trades)')
ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% (Breakeven)')
ax2.fill_between(df_trades['exit_date'], df_trades['rolling_win_rate'], 50,
                 where=(df_trades['rolling_win_rate'] >= 50), alpha=0.3, color='green', interpolate=True)
ax2.fill_between(df_trades['exit_date'], df_trades['rolling_win_rate'], 50,
                 where=(df_trades['rolling_win_rate'] < 50), alpha=0.3, color='red', interpolate=True)

ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
ax2.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title(f'Rolling Win Rate (Last {ROLLING_WINDOW} Trades)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Win/Loss Sequence
colors = ['green' if outcome == 'WIN' else 'red' for outcome in df_trades['outcome']]
ax3.bar(df_trades['trade_number'], df_trades['net_return_pct'],
       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax3.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('P&L (%)', fontsize=11, fontweight='bold')
ax3.set_title('Individual Trade P&L Sequence', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

# Plot 4: Win Rate by Month
df_trades['month'] = df_trades['exit_date'].dt.to_period('M')
monthly_stats = df_trades.groupby('month').agg({
    'win': ['sum', 'count']
}).reset_index()
monthly_stats.columns = ['month', 'wins', 'total']
monthly_stats['win_rate'] = (monthly_stats['wins'] / monthly_stats['total']) * 100
monthly_stats['month_str'] = monthly_stats['month'].astype(str)

bars = ax4.bar(range(len(monthly_stats)), monthly_stats['win_rate'],
              color=['green' if wr >= 50 else 'red' for wr in monthly_stats['win_rate']],
              alpha=0.7, edgecolor='black', linewidth=1)
ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
ax4.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Win Rate by Month', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(monthly_stats)))
ax4.set_xticklabels(monthly_stats['month_str'], rotation=45, ha='right')
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add trade count labels on bars
for i, (wr, total) in enumerate(zip(monthly_stats['win_rate'], monthly_stats['total'])):
    ax4.text(i, wr + 2, f'{total}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save plot
output_file = f'{PAIR}_win_rate_{TEST_DAYS}days.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Print statistics
print("\n" + "="*80)
print("WIN RATE STATISTICS")
print("="*80)
print(f"Overall Win Rate: {df_trades['cumulative_win_rate'].iloc[-1]:.1f}%")
print(f"Total Trades: {len(df_trades)}")
print(f"Wins: {df_trades['win'].sum()}")
print(f"Losses: {len(df_trades) - df_trades['win'].sum()}")
print()

print(f"Rolling Win Rate Statistics (last {ROLLING_WINDOW} trades):")
print(f"  Current: {df_trades['rolling_win_rate'].iloc[-1]:.1f}%")
print(f"  Average: {df_trades['rolling_win_rate'].mean():.1f}%")
print(f"  Highest: {df_trades['rolling_win_rate'].max():.1f}%")
print(f"  Lowest: {df_trades['rolling_win_rate'].min():.1f}%")
print()

print("Monthly Breakdown:")
for _, row in monthly_stats.iterrows():
    print(f"  {row['month_str']}: {row['win_rate']:.1f}% ({int(row['wins'])}/{int(row['total'])} trades)")

# Show plot
plt.show()

print("\n" + "="*80)
print("Visualization complete!")
print("="*80)
