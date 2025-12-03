"""
Analyze commodity backtest performance with equity charts and weekly stats.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
import io

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Commodities to analyze
COMMODITIES = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD']
NAMES = {
    'XAUUSD': 'Gold',
    'XAGUSD': 'Silver',
    'XPTUSD': 'Platinum',
    'XPDUSD': 'Palladium',
    'XCUUSD': 'Copper'
}

def analyze_commodity(pair):
    """Analyze a single commodity's trades"""
    try:
        trades_file = f'{pair}_backtest_trades_750days.csv'
        df = pd.read_csv(trades_file)

        # Parse dates
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df = df.sort_values('exit_date')

        # Calculate cumulative equity
        df['cumulative_return'] = (1 + df['net_return_pct'] / 100).cumprod()
        df['equity'] = 1000 * df['cumulative_return']

        # Weekly stats
        df['week'] = df['exit_date'].dt.to_period('W')
        weekly = df.groupby('week').agg({
            'net_return_pct': 'sum',
            'equity': 'last'
        }).reset_index()
        weekly['week_str'] = weekly['week'].astype(str)

        return df, weekly

    except Exception as e:
        print(f"Error analyzing {pair}: {e}")
        return None, None

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot equity curves for each commodity
for idx, pair in enumerate(COMMODITIES):
    print(f"\nAnalyzing {NAMES[pair]}...")

    df, weekly = analyze_commodity(pair)
    if df is None:
        continue

    # Plot equity curve
    row = idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    ax.plot(df['exit_date'], df['equity'], linewidth=2, color='blue', alpha=0.7)
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Starting Capital')
    ax.fill_between(df['exit_date'], 1000, df['equity'], alpha=0.3, color='green' if df['equity'].iloc[-1] > 1000 else 'red')

    ax.set_title(f'{NAMES[pair]} - Equity Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Equity ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add stats text
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity - 1000) / 1000 * 100
    wins = len(df[df['net_return_pct'] > 0])
    total = len(df)
    win_rate = wins / total * 100

    stats_text = f'Final: ${final_equity:,.0f}\nReturn: {total_return:.1f}%\nWin Rate: {win_rate:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Weekly performance stats
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  Final Equity: ${final_equity:,.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total Trades: {total}")

    # Weekly stats
    positive_weeks = len(weekly[weekly['net_return_pct'] > 0])
    total_weeks = len(weekly)
    avg_weekly_return = weekly['net_return_pct'].mean()
    best_week = weekly['net_return_pct'].max()
    worst_week = weekly['net_return_pct'].min()

    print(f"  Positive Weeks: {positive_weeks}/{total_weeks} ({positive_weeks/total_weeks*100:.1f}%)")
    print(f"  Avg Weekly Return: {avg_weekly_return:.2f}%")
    print(f"  Best Week: {best_week:.2f}%")
    print(f"  Worst Week: {worst_week:.2f}%")

# Add comparison plot in bottom row
ax_compare = fig.add_subplot(gs[2, :])

print("\n" + "="*80)
print("COMPARISON PLOT")
print("="*80)

for pair in COMMODITIES:
    df, _ = analyze_commodity(pair)
    if df is not None:
        # Normalize to percentage return
        normalized = (df['equity'] - 1000) / 1000 * 100
        ax_compare.plot(df['exit_date'], normalized, linewidth=2.5, label=NAMES[pair], alpha=0.8)

ax_compare.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax_compare.set_title('All Commodities - Normalized Returns (%)', fontsize=14, fontweight='bold')
ax_compare.set_xlabel('Date', fontsize=10)
ax_compare.set_ylabel('Return (%)', fontsize=10)
ax_compare.legend(loc='upper left', fontsize=10)
ax_compare.grid(True, alpha=0.3)

# Save figure
plt.savefig('commodity_equity_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved equity curves to: commodity_equity_curves.png")
plt.close()

# Create weekly performance table
print("\n" + "="*80)
print("WEEKLY PERFORMANCE SUMMARY")
print("="*80)

weekly_data = []
for pair in COMMODITIES:
    df, weekly = analyze_commodity(pair)
    if weekly is not None:
        positive_weeks = len(weekly[weekly['net_return_pct'] > 0])
        total_weeks = len(weekly)
        avg_weekly = weekly['net_return_pct'].mean()
        median_weekly = weekly['net_return_pct'].median()
        std_weekly = weekly['net_return_pct'].std()
        best_week = weekly['net_return_pct'].max()
        worst_week = weekly['net_return_pct'].min()

        weekly_data.append({
            'Commodity': NAMES[pair],
            'Positive Weeks': f"{positive_weeks}/{total_weeks} ({positive_weeks/total_weeks*100:.1f}%)",
            'Avg Week': f"{avg_weekly:.2f}%",
            'Median Week': f"{median_weekly:.2f}%",
            'Std Dev': f"{std_weekly:.2f}%",
            'Best Week': f"{best_week:.2f}%",
            'Worst Week': f"{worst_week:.2f}%"
        })

weekly_df = pd.DataFrame(weekly_data)
print(weekly_df.to_string(index=False))

print("\n" + "="*80)
