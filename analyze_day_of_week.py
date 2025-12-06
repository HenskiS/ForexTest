"""
Analyze day-of-week performance from actual spreads backtest
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
print("DAY OF WEEK PERFORMANCE ANALYSIS - WITH ACTUAL SPREADS")
print("="*100)
print()

# Load backtest results
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

results = data['results']
test_dates = pd.to_datetime(data['test_dates'])
pair_returns = data['pair_returns']

# Analyze for 2.0x leverage (primary interest)
selected_leverage = 2.0

result = next((r for r in results if r['leverage'] == selected_leverage), None)
if not result or result.get('blew_up', False):
    print(f"ERROR: {selected_leverage}x leverage not available or blew up")
    exit(1)

equity_curve = np.array(result['equity_curve'])

# Calculate daily returns
daily_returns = np.diff(equity_curve) / equity_curve[:-1]

# Create dataframe with returns and dates
df = pd.DataFrame({
    'date': test_dates,
    'daily_return': daily_returns
})

# Add day of week
df['day_of_week'] = df['date'].dt.day_name()
df['day_num'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

# Filter to trading days only (Monday-Friday)
df = df[df['day_num'] < 5]

print(f"Analyzing {len(df)} trading days from {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Leverage: {selected_leverage}x")
print()

# Calculate stats by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

print(f"{'Day':<12} {'Avg Daily Return':>18} {'Win Rate':>12} {'Trading Days':>15}")
print("-" * 100)

day_stats = []
for day in day_order:
    day_data = df[df['day_of_week'] == day]

    avg_return = day_data['daily_return'].mean() * 100
    win_rate = (day_data['daily_return'] > 0).sum() / len(day_data) * 100
    num_days = len(day_data)

    day_stats.append({
        'day': day,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'num_days': num_days
    })

    print(f"{day:<12} {avg_return:>17.3f}% {win_rate:>11.1f}% {num_days:>15}")

print("-" * 100)

# Overall stats
overall_avg = df['daily_return'].mean() * 100
overall_win_rate = (df['daily_return'] > 0).sum() / len(df) * 100
print(f"{'OVERALL':<12} {overall_avg:>17.3f}% {overall_win_rate:>11.1f}% {len(df):>15}")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print()

# Identify patterns
best_day = max(day_stats, key=lambda x: x['avg_return'])
worst_day = min(day_stats, key=lambda x: x['avg_return'])
best_win_rate = max(day_stats, key=lambda x: x['win_rate'])

print("KEY OBSERVATIONS:")
print(f"  - Best return day: {best_day['day']} ({best_day['avg_return']:.3f}%)")
print(f"  - Worst return day: {worst_day['day']} ({worst_day['avg_return']:.3f}%)")
print(f"  - Highest win rate: {best_win_rate['day']} ({best_win_rate['win_rate']:.1f}%)")
print()

# Show markdown table format
print("MARKDOWN TABLE FORMAT:")
print("-" * 100)
print("| Day | Avg Daily Return | Win Rate | Pattern |")
print("|-----|-----------------|----------|---------|")
for stat in day_stats:
    pattern = ""
    if stat['day'] == best_day['day']:
        pattern = "Strongest returns"
    elif stat['day'] == worst_day['day']:
        pattern = "Weakest day"
    elif stat['day'] == best_win_rate['day']:
        pattern = "Highest win rate"

    print(f"| {stat['day']} | {stat['avg_return']:.3f}% | {stat['win_rate']:.1f}% | {pattern} |")
print()
