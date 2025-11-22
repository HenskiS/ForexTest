"""
Analyze rolling daily results with the same backtest methodology as the static approach.

This will allow direct comparison between:
- Static: Retrain every 126 days
- Rolling Daily: Retrain every day with 756-day window
"""

import pandas as pd
import numpy as np
import pickle

PAIR = 'EURUSD'
TARGET = 'target_5day_return'

print(f"Analyzing Rolling Daily Results for {PAIR}")
print("="*70)

# Load rolling daily results
with open(f'rolling_daily_results_{PAIR}_{TARGET}.pkl', 'rb') as f:
    rolling_results = pickle.load(f)

print(f"Loaded {len(rolling_results)} windows")

# Load price data
df = pd.read_csv(f'data/{PAIR}_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Combine all predictions across windows
all_dates = []
all_predictions = []
all_actuals = []

for result in rolling_results:
    dates = pd.to_datetime(result['dates'])
    preds = result['predictions']
    actuals = result['actuals']

    all_dates.extend(dates)
    all_predictions.extend(preds)
    all_actuals.extend(actuals)

print(f"\nTotal predictions: {len(all_predictions)} days")
print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")

# Calculate overall MAE
mae = np.mean(np.abs(np.array(all_actuals) - np.array(all_predictions)))
correlation = np.corrcoef(all_predictions, all_actuals)[0, 1]
print(f"\nPrediction Quality:")
print(f"  MAE: {mae:.6f}")
print(f"  Correlation: {correlation:.4f}")

# Save combined results for backtesting
combined_results = {
    'dates': [str(d) for d in all_dates],
    'predictions': all_predictions,
    'actuals': all_actuals,
    'pair': PAIR,
    'method': 'rolling_daily'
}

output_file = f'rolling_daily_combined_{PAIR}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(combined_results, f)

print(f"\nCombined results saved to: {output_file}")
print("\nNow run backtest script to analyze trading performance!")
