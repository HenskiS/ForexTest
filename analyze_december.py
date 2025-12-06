"""
Analyze December performance from actual spreads backtest
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
print("DECEMBER SEASONALITY ANALYSIS - WITH ACTUAL SPREADS")
print("="*100)
print()

# Load backtest results
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

results = data['results']
test_dates = pd.to_datetime(data['test_dates'])

# Analyze for selected leverage levels
selected_leverages = [1.0, 2.0, 3.5]

print(f"Analyzing December performance from {test_dates.min().date()} to {test_dates.max().date()}")
print()

# Create summary data
december_stats = []

for lev in selected_leverages:
    result = next((r for r in results if r['leverage'] == lev), None)
    if not result or result.get('blew_up', False):
        print(f"{lev}x Leverage: BLEW UP")
        continue

    equity_curve = np.array(result['equity_curve'])

    # Calculate daily returns
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]

    # Create dataframe
    df = pd.DataFrame({
        'date': test_dates,
        'daily_return': daily_returns,
        'equity': equity_curve[1:]
    })

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Filter to December only
    dec_data = df[df['month'] == 12]

    if len(dec_data) == 0:
        print(f"{lev}x: No December data available")
        continue

    # Calculate monthly returns for each December
    dec_returns = []
    dec_years = sorted(dec_data['year'].unique())

    for year in dec_years:
        year_dec = dec_data[dec_data['year'] == year]

        # Get starting equity (last day of November or first day of December)
        first_idx = df[df['date'] == year_dec['date'].iloc[0]].index[0]
        if first_idx == 0:
            start_equity = equity_curve[0]
        else:
            start_equity = df.iloc[first_idx - 1]['equity']

        end_equity = year_dec['equity'].iloc[-1]
        monthly_return = (end_equity - start_equity) / start_equity * 100

        dec_returns.append({
            'year': year,
            'return': monthly_return
        })

    dec_returns_df = pd.DataFrame(dec_returns)

    # Calculate statistics
    avg_return = dec_returns_df['return'].mean()
    best_return = dec_returns_df['return'].max()
    best_year = dec_returns_df.loc[dec_returns_df['return'].idxmax(), 'year']
    worst_return = dec_returns_df['return'].min()
    worst_year = dec_returns_df.loc[dec_returns_df['return'].idxmin(), 'year']
    win_rate = (dec_returns_df['return'] > 0).sum() / len(dec_returns_df) * 100

    december_stats.append({
        'leverage': lev,
        'avg_return': avg_return,
        'best_return': best_return,
        'best_year': int(best_year),
        'worst_return': worst_return,
        'worst_year': int(worst_year),
        'win_rate': win_rate,
        'num_decembers': len(dec_returns_df)
    })

    print(f"{lev}x Leverage:")
    print(f"  Average December: {avg_return:.2f}%")
    print(f"  Best December: {best_return:.2f}% ({int(best_year)})")
    print(f"  Worst December: {worst_return:.2f}% ({int(worst_year)})")
    print(f"  Win Rate: {win_rate:.1f}% ({int((dec_returns_df['return'] > 0).sum())}/{len(dec_returns_df)} years)")
    print()

print("="*100)
print("MARKDOWN TABLE FORMAT:")
print("="*100)
print()
print("| Leverage | Avg December | Best December | Worst December | Win Rate |")
print("|----------|--------------|---------------|----------------|----------|")

for stat in december_stats:
    print(f"| {stat['leverage']}x | {stat['avg_return']:.2f}% | "
          f"{stat['best_return']:.2f}% ({stat['best_year']}) | "
          f"{stat['worst_return']:.2f}% ({stat['worst_year']}) | "
          f"{stat['win_rate']:.1f}% |")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
