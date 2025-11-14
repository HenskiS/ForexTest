"""
Convert daily model results from list-of-dicts format to consolidated DataFrame format
like the 4-hour model uses, so we can test entry thresholds.

Usage:
    python convert_daily_results_format.py [PAIR]
    python convert_daily_results_format.py EURUSD
    python convert_daily_results_format.py GBPUSD
"""

import pickle
import pandas as pd
import numpy as np
import sys

# Get pair from command line or use EURUSD as default
PAIR = sys.argv[1].upper() if len(sys.argv) > 1 else 'EURUSD'

print("="*80)
print(f"CONVERTING {PAIR} DAILY MODEL RESULTS TO CONSOLIDATED FORMAT")
print("="*80)

# Determine input/output files based on pair
if PAIR == 'EURUSD':
    # Use the current target_5day_return.pkl (goes to 2025), not FIXED.pkl (only to 2022)
    input_file = 'xgboost_results_target_5day_return.pkl'
    data_file = 'data/EURUSD_1day_with_features_FIXED_multitarget.csv'
    output_file = 'xgboost_results_EURUSD_target_5day_return.pkl'
else:
    input_file = f'xgboost_results_{PAIR}_target_5day_return.pkl'
    data_file = f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv'
    output_file = f'xgboost_results_{PAIR}_target_5day_return.pkl'

# Load daily results (list of dicts)
print(f"\nLoading daily model results from {input_file}...")
daily_results = pickle.load(open(input_file, 'rb'))
print(f"Loaded {len(daily_results)} windows")

# Load the data to get full date index and OHLC
print(f"\nLoading price data from {data_file}...")
df = pd.read_csv(data_file, index_col='date', parse_dates=True)
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Build consolidated DataFrame
print("\nConsolidating predictions...")
consolidated_rows = []

for window_result in daily_results:
    window_id = window_result['window_id']
    predictions = window_result['predictions']
    actuals = window_result['actuals']
    date_start = pd.to_datetime(window_result['date_start'])
    date_end = pd.to_datetime(window_result['date_end'])

    # The date_start/date_end represent the entire window (train+val+test = 882 days)
    # But predictions are only for the test period (126 days at the end)
    # Test period starts at: date_start + 600 days (train) + 156 days (val)
    # Test period ends at: date_end

    # Get all dates in the window
    window_mask = (df.index >= date_start) & (df.index <= date_end)
    window_dates = df.index[window_mask]

    # Test period is the last 126 days of the window
    if len(window_dates) < len(predictions):
        print(f"ERROR: Window {window_id}: window has {len(window_dates)} dates but {len(predictions)} predictions")
        continue

    test_dates = window_dates[-len(predictions):]

    # Verify lengths match
    if len(test_dates) != len(predictions):
        print(f"WARNING: Window {window_id}: {len(test_dates)} dates vs {len(predictions)} predictions")
        # Use the shorter length
        min_len = min(len(test_dates), len(predictions))
        test_dates = test_dates[:min_len]
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

    # Get OHLC for these dates
    test_data = df.loc[test_dates, ['open', 'high', 'low', 'close']]

    # Create rows for this window
    for i, date in enumerate(test_dates):
        row = {
            'date': date,
            'window_id': window_id,
            'predicted_return': predictions[i],
            'actual_return': actuals[i],
            'open': test_data.iloc[i]['open'],
            'high': test_data.iloc[i]['high'],
            'low': test_data.iloc[i]['low'],
            'close': test_data.iloc[i]['close'],
        }
        consolidated_rows.append(row)

# Create DataFrame
print(f"\nCreating consolidated DataFrame with {len(consolidated_rows)} rows...")
consolidated_df = pd.DataFrame(consolidated_rows)
consolidated_df = consolidated_df.set_index('date')
consolidated_df = consolidated_df.sort_index()

print(f"\nConsolidated DataFrame:")
print(f"  Shape: {consolidated_df.shape}")
print(f"  Columns: {list(consolidated_df.columns)}")
print(f"  Date range: {consolidated_df.index.min()} to {consolidated_df.index.max()}")
print(f"\nFirst few rows:")
print(consolidated_df.head())
print(f"\nLast few rows:")
print(consolidated_df.tail())

# Save
print(f"\nSaving to: {output_file}")
consolidated_df.to_pickle(output_file)

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)
print(f"Now you can run: python test_entry_thresholds_daily.py --pair {PAIR}")
