"""
Analyze risk distributions (drawdowns and returns) from actual spreads backtest
Uses the pickle file from 4500-day backtest to generate yearly statistics
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
print("RISK DISTRIBUTION ANALYSIS - WITH ACTUAL SPREADS")
print("="*100)
print()

# Load backtest results
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

results = data['results']
test_dates = pd.to_datetime(data['test_dates'])

# Analyze for selected leverage levels
selected_leverages = [1.0, 2.0, 3.5, 5.0]

print("="*100)
print("DRAWDOWN DISTRIBUTION BY LEVERAGE")
print("="*100)
print()

drawdown_stats = []

for lev in selected_leverages:
    result = next((r for r in results if r['leverage'] == lev), None)
    if not result or result.get('blew_up', False):
        print(f"{lev}x: BLEW UP")
        continue

    equity_curve = np.array(result['equity_curve'])

    # Create dataframe with equity and dates
    equity_df = pd.DataFrame({
        'equity': equity_curve[1:],  # Skip initial $1000
        'date': test_dates
    })
    equity_df['year'] = equity_df['date'].dt.year

    # Calculate yearly max drawdowns
    yearly_max_dds = []
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) == 0:
            continue

        # Get start equity
        first_idx = equity_df[equity_df['year'] == year].index[0]
        if first_idx == 0:
            start_equity = 1000.0
        else:
            start_equity = equity_df.iloc[first_idx - 1]['equity']

        # Calculate max DD for the year
        year_equity_vals = np.concatenate([[start_equity], year_data['equity'].values])
        cummax = np.maximum.accumulate(year_equity_vals)
        drawdowns = (year_equity_vals - cummax) / cummax * 100
        max_dd = drawdowns.min()

        yearly_max_dds.append({
            'year': year,
            'max_dd': max_dd
        })

    dd_df = pd.DataFrame(yearly_max_dds)

    avg_dd = dd_df['max_dd'].mean()
    median_dd = dd_df['max_dd'].median()
    worst_dd = dd_df['max_dd'].min()
    worst_year = dd_df.loc[dd_df['max_dd'].idxmin(), 'year']

    drawdown_stats.append({
        'leverage': lev,
        'avg_dd': avg_dd,
        'median_dd': median_dd,
        'worst_dd': worst_dd,
        'worst_year': int(worst_year)
    })

    print(f"{lev}x Leverage:")
    print(f"  Average DD: {avg_dd:.2f}%")
    print(f"  Median DD: {median_dd:.2f}%")
    print(f"  Worst DD: {worst_dd:.2f}% ({int(worst_year)})")
    print()

print()
print("="*100)
print("RETURN DISTRIBUTION BY LEVERAGE")
print("="*100)
print()

return_stats = []

for lev in selected_leverages:
    result = next((r for r in results if r['leverage'] == lev), None)
    if not result or result.get('blew_up', False):
        continue

    equity_curve = np.array(result['equity_curve'])

    # Create dataframe with equity and dates
    equity_df = pd.DataFrame({
        'equity': equity_curve[1:],
        'date': test_dates
    })
    equity_df['year'] = equity_df['date'].dt.year

    # Calculate yearly returns
    yearly_returns = []
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) == 0:
            continue

        # Get start equity
        first_idx = equity_df[equity_df['year'] == year].index[0]
        if first_idx == 0:
            start_equity = 1000.0
        else:
            start_equity = equity_df.iloc[first_idx - 1]['equity']

        end_equity = year_data['equity'].iloc[-1]
        year_return = (end_equity - start_equity) / start_equity * 100

        yearly_returns.append({
            'year': year,
            'return': year_return
        })

    ret_df = pd.DataFrame(yearly_returns)

    avg_return = ret_df['return'].mean()
    median_return = ret_df['return'].median()
    worst_return = ret_df['return'].min()
    worst_year = ret_df.loc[ret_df['return'].idxmin(), 'year']
    best_return = ret_df['return'].max()
    best_year = ret_df.loc[ret_df['return'].idxmax(), 'year']
    std_dev = ret_df['return'].std()

    return_stats.append({
        'leverage': lev,
        'avg_return': avg_return,
        'median_return': median_return,
        'worst_return': worst_return,
        'worst_year': int(worst_year),
        'best_return': best_return,
        'best_year': int(best_year),
        'std_dev': std_dev
    })

    print(f"{lev}x Leverage:")
    print(f"  Average Annual: {avg_return:.2f}%")
    print(f"  Median Annual: {median_return:.2f}%")
    print(f"  Worst Year: {worst_return:.2f}% ({int(worst_year)})")
    print(f"  Best Year: {best_return:.2f}% ({int(best_year)})")
    print(f"  Std Dev: {std_dev:.2f}%")
    print()

print()
print("="*100)
print("MARKDOWN FORMAT - DRAWDOWN DISTRIBUTION")
print("="*100)
print()
print("| Leverage | Avg DD | Median DD | Worst DD | Year of Worst DD |")
print("|----------|--------|-----------|----------|------------------|")
for stat in drawdown_stats:
    print(f"| {stat['leverage']}x | {stat['avg_dd']:.2f}% | "
          f"{stat['median_dd']:.2f}% | {stat['worst_dd']:.2f}% | {stat['worst_year']} |")

print()
print("="*100)
print("MARKDOWN FORMAT - RETURN DISTRIBUTION")
print("="*100)
print()
print("| Leverage | Avg Annual | Median Annual | Worst Year | Best Year | Std Dev |")
print("|----------|-----------|---------------|------------|-----------|---------|")
for stat in return_stats:
    # Classify std dev
    if stat['std_dev'] < 20:
        std_label = "Low"
    elif stat['std_dev'] < 50:
        std_label = "Moderate"
    elif stat['std_dev'] < 100:
        std_label = "High"
    else:
        std_label = "Very High"

    print(f"| {stat['leverage']}x | {stat['avg_return']:.2f}% | "
          f"{stat['median_return']:.2f}% | "
          f"{stat['worst_return']:.2f}% ({stat['worst_year']}) | "
          f"{stat['best_return']:.2f}% ({stat['best_year']}) | "
          f"{std_label} |")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
