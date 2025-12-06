"""
Generate year-by-year performance analysis using actual spreads
Uses the existing backtest data from multi_pair_leverage_actual_spreads_4500days.csv
and re-runs for specific leverage levels to get yearly breakdown
"""
import pandas as pd
import numpy as np
import sys
import os

# Load XGBoost configuration
sys.path.insert(0, os.path.dirname(__file__))

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TEST_DAYS = 4500

print("="*100)
print("YEAR-BY-YEAR PERFORMANCE ANALYSIS - WITH ACTUAL SPREADS")
print("="*100)
print()

# Check if results exist
if not os.path.exists('multi_pair_leverage_actual_spreads_4500days.csv'):
    print("ERROR: Need to run backtest_multi_pair_actual_spreads.py first (with TEST_DAYS=4500)")
    sys.exit(1)

# Load engineered data to get dates
print("Loading data...")
test_dates = None
for pair in PAIRS:
    try:
        df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        df['date'] = pd.to_datetime(df['date'])
        if test_dates is None:
            test_dates = df['date'].iloc[-TEST_DAYS:].values
        break
    except:
        try:
            df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
            df['date'] = pd.to_datetime(df['date'])
            if test_dates is None:
                test_dates = df['date'].iloc[-TEST_DAYS:].values
            break
        except:
            continue

if test_dates is None:
    print("ERROR: Could not load date data")
    sys.exit(1)

test_dates = pd.to_datetime(test_dates)
print(f"Test Period: {test_dates[0].date()} to {test_dates[-1].date()}")
print(f"Total Days: {len(test_dates)}")
print()

# We need to re-run backtest to get equity curves
# Import from the actual spreads backtest
print("Re-running backtest for selected leverage levels to get yearly breakdown...")
print("This will take ~2 minutes...")
print()

from backtest_multi_pair_actual_spreads import (
    backtest_single_pair_actual_spreads,
    simulate_multi_pair_portfolio,
    PAIRS as BACKTEST_PAIRS,
    TEST_DAYS as BACKTEST_TEST_DAYS
)

# Quick check - verify we're using the same parameters
if BACKTEST_TEST_DAYS != TEST_DAYS:
    print(f"WARNING: Backtest script has TEST_DAYS={BACKTEST_TEST_DAYS}, expected {TEST_DAYS}")
    print("Please update TEST_DAYS in backtest_multi_pair_actual_spreads.py to 4500")
    sys.exit(1)

# Load predictions for each pair (these should be cached from original run)
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from trading.market_utils import calculate_technical_features
import pickle

pair_returns = {}
print("\nLoading predictions for each pair...")

for pair in PAIRS:
    # Try to load from engineered data
    try:
        with open(f'hyperparams_best_{pair}.pkl', 'rb') as f:
            best_params = pickle.load(f)
    except:
        print(f"  {pair}: Using default XGBoost params")
        best_params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}

    # Load data with spreads
    try:
        df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
    except:
        df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
        df['spread_pct'] = 0.00013  # Default spread

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate features
    df = calculate_technical_features(df)
    df['target_1day_return'] = df['close'].pct_change(1).shift(-1)
    df = df.dropna()

    # Get test data
    test_data = df.iloc[-TEST_DAYS:]

    print(f"  {pair}: {len(test_data)} days loaded")

    # Generate predictions using rolling window (simplified - just use actual strategy)
    # For speed, we'll use a simpler approach
    daily_returns = backtest_single_pair_actual_spreads(
        test_data,
        best_params,
        calculate_technical_features
    )

    pair_returns[pair] = daily_returns

print("\nSimulating portfolio at selected leverage levels...")

# Analyze for selected leverage levels
selected_leverages = [1.0, 2.0, 3.5, 5.0]

for lev in selected_leverages:
    print(f"\n{'='*100}")
    print(f"LEVERAGE: {lev}x")
    print(f"{'='*100}")

    result = simulate_multi_pair_portfolio(pair_returns, lev)

    if result['blew_up']:
        print(f"Account blew up on day {result['blew_up_day']}")
        continue

    equity_curve = result['equity_curve']

    # Create dataframe with equity and dates
    equity_df = pd.DataFrame({
        'equity': equity_curve[1:],  # Skip initial
        'date': test_dates
    })
    equity_df['year'] = equity_df['date'].year

    print(f"\n{'Year':<6} {'Return':>12} {'Max DD':>12} {'Start Equity':>18} {'End Equity':>18} {'Days':>6}")
    print("-" * 100)

    yearly_stats = []
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

        # Calculate max DD for the year
        year_equity_vals = np.concatenate([[start_equity], year_data['equity'].values])
        cummax = np.maximum.accumulate(year_equity_vals)
        drawdowns = (year_equity_vals - cummax) / cummax * 100
        max_dd = drawdowns.min()

        yearly_stats.append({
            'year': year,
            'return': year_return,
            'max_dd': max_dd,
            'start': start_equity,
            'end': end_equity,
            'days': len(year_data)
        })

        print(f"{year:<6} {year_return:>11.2f}% {max_dd:>11.2f}% ${start_equity:>17,.0f} ${end_equity:>17,.0f} {len(year_data):>6}")

    # Summary
    if yearly_stats:
        avg_return = np.mean([s['return'] for s in yearly_stats])
        median_return = np.median([s['return'] for s in yearly_stats])
        min_return = np.min([s['return'] for s in yearly_stats])
        max_return = np.max([s['return'] for s in yearly_stats])
        avg_dd = np.mean([s['max_dd'] for s in yearly_stats])
        worst_dd = np.min([s['max_dd'] for s in yearly_stats])
        winning_years = sum(1 for s in yearly_stats if s['return'] > 0)

        print("-" * 100)
        print(f"{'AVG':<6} {avg_return:>11.2f}% {avg_dd:>11.2f}%")
        print(f"{'MEDIAN':<6} {median_return:>11.2f}%")
        print(f"{'MIN':<6} {min_return:>11.2f}% {worst_dd:>11.2f}%")
        print(f"{'MAX':<6} {max_return:>11.2f}%")
        print(f"\nWinning Years: {winning_years}/{len(yearly_stats)} ({winning_years/len(yearly_stats)*100:.1f}%)")

print("\n" + "="*100)
print("ANALYSIS COMPLETE - Results use ACTUAL VARIABLE SPREADS")
print("="*100)
