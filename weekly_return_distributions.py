"""
Analyze and visualize weekly return distributions for all commodities.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
COLORS = {
    'XAUUSD': '#FFD700',  # Gold color
    'XAGUSD': '#C0C0C0',  # Silver color
    'XPTUSD': '#E5E4E2',  # Platinum color
    'XPDUSD': '#CED0DD',  # Palladium color
    'XCUUSD': '#B87333'   # Copper color
}

def get_weekly_returns(pair):
    """Get weekly returns for a commodity"""
    try:
        trades_file = f'{pair}_backtest_trades_750days.csv'
        df = pd.read_csv(trades_file)

        # Parse dates
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df = df.sort_values('exit_date')

        # Group by week
        df['week'] = df['exit_date'].dt.to_period('W')
        weekly = df.groupby('week').agg({
            'net_return_pct': 'sum'
        }).reset_index()

        return weekly['net_return_pct'].values

    except Exception as e:
        print(f"Error loading {pair}: {e}")
        return None

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

print("="*80)
print("WEEKLY RETURN DISTRIBUTION ANALYSIS")
print("="*80)

# Plot individual distributions
for idx, pair in enumerate(COMMODITIES):
    row = idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    weekly_returns = get_weekly_returns(pair)
    if weekly_returns is None:
        continue

    name = NAMES[pair]
    color = COLORS[pair]

    # Create histogram
    n, bins, patches = ax.hist(weekly_returns, bins=30, alpha=0.7, color=color,
                                edgecolor='black', density=True)

    # Add vertical lines for mean and median
    mean_ret = np.mean(weekly_returns)
    median_ret = np.median(weekly_returns)
    ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.2f}%')
    ax.axvline(median_ret, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_ret:.2f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Calculate stats
    std_dev = np.std(weekly_returns)
    skewness = pd.Series(weekly_returns).skew()
    positive_pct = (weekly_returns > 0).sum() / len(weekly_returns) * 100

    # Title and labels
    ax.set_title(f'{name} - Weekly Return Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weekly Return (%)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add stats box
    stats_text = (f'Std Dev: {std_dev:.2f}%\n'
                  f'Skewness: {skewness:.2f}\n'
                  f'Positive: {positive_pct:.1f}%\n'
                  f'Range: [{weekly_returns.min():.2f}%, {weekly_returns.max():.2f}%]')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    print(f"\n{name}:")
    print(f"  Mean: {mean_ret:.2f}%")
    print(f"  Median: {median_ret:.2f}%")
    print(f"  Std Dev: {std_dev:.2f}%")
    print(f"  Skewness: {skewness:.2f}")
    print(f"  Min: {weekly_returns.min():.2f}%")
    print(f"  Max: {weekly_returns.max():.2f}%")
    print(f"  Positive weeks: {positive_pct:.1f}%")

# Create comparison plot (overlay all distributions)
ax_compare = fig.add_subplot(gs[2, :])

print("\n" + "="*80)
print("CREATING OVERLAY COMPARISON")
print("="*80)

for pair in COMMODITIES:
    weekly_returns = get_weekly_returns(pair)
    if weekly_returns is not None:
        name = NAMES[pair]
        color = COLORS[pair]

        # Create KDE (smooth distribution)
        from scipy import stats
        kde = stats.gaussian_kde(weekly_returns)
        x_range = np.linspace(weekly_returns.min() - 1, weekly_returns.max() + 1, 200)
        density = kde(x_range)

        ax_compare.plot(x_range, density, linewidth=2.5, label=name,
                       color=color, alpha=0.8)
        ax_compare.fill_between(x_range, 0, density, alpha=0.2, color=color)

ax_compare.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax_compare.set_title('All Commodities - Weekly Return Distribution Comparison',
                     fontsize=14, fontweight='bold')
ax_compare.set_xlabel('Weekly Return (%)', fontsize=10)
ax_compare.set_ylabel('Probability Density', fontsize=10)
ax_compare.legend(loc='upper right', fontsize=10)
ax_compare.grid(True, alpha=0.3)

# Save figure
plt.savefig('weekly_return_distributions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved distribution charts to: weekly_return_distributions.png")
plt.close()

# Create summary table
print("\n" + "="*80)
print("DISTRIBUTION STATISTICS SUMMARY")
print("="*80)

summary_data = []
for pair in COMMODITIES:
    weekly_returns = get_weekly_returns(pair)
    if weekly_returns is not None:
        summary_data.append({
            'Commodity': NAMES[pair],
            'Mean': f"{np.mean(weekly_returns):.2f}%",
            'Median': f"{np.median(weekly_returns):.2f}%",
            'Std Dev': f"{np.std(weekly_returns):.2f}%",
            'Skewness': f"{pd.Series(weekly_returns).skew():.2f}",
            'Min': f"{weekly_returns.min():.2f}%",
            'Max': f"{weekly_returns.max():.2f}%",
            'Sharpe (approx)': f"{np.mean(weekly_returns) / np.std(weekly_returns):.2f}"
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("• Higher mean = better average weekly performance")
print("• Lower std dev = more consistent (less risky)")
print("• Positive skewness = more extreme gains than losses (good!)")
print("• Higher Sharpe = better risk-adjusted returns")
print("="*80)
