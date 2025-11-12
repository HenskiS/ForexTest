"""
Analyze the correlation between predicted and actual returns.
This tells us if the model has ANY directional accuracy.
"""

import pandas as pd
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predictions', type=str, required=True,
                    help='Path to predictions pickle file')
args = parser.parse_args()

print("=" * 80)
print("PREDICTION CORRELATION ANALYSIS")
print("=" * 80)

# Load predictions
print(f"\nLoading predictions from {args.predictions}...")
with open(args.predictions, 'rb') as f:
    results_df = pickle.load(f)

print(f"Predictions loaded: {len(results_df)} rows")
print(f"Date range: {results_df.index.min()} to {results_df.index.max()}")

# Determine columns
if 'predicted_return' in results_df.columns:
    pred_col = 'predicted_return'
    actual_col = 'actual_return'
elif 'predicted_class' in results_df.columns:
    pred_col = 'predicted_class'
    actual_col = 'actual_class'
else:
    print(f"ERROR: No prediction column found!")
    exit(1)

print(f"\nUsing columns: {pred_col} vs {actual_col}")

# Overall correlation
overall_corr = results_df[pred_col].corr(results_df[actual_col])
print(f"\nOverall correlation: {overall_corr:.6f}")

# Correlation by year
results_df['year'] = results_df.index.year
yearly_corr = results_df.groupby('year').apply(
    lambda x: x[pred_col].corr(x[actual_col])
)

print(f"\nCorrelation by year:")
print(yearly_corr.to_string())

# Summary statistics
print(f"\nYearly correlation summary:")
print(f"  Mean:   {yearly_corr.mean():.6f}")
print(f"  Median: {yearly_corr.median():.6f}")
print(f"  Std:    {yearly_corr.std():.6f}")
print(f"  Min:    {yearly_corr.min():.6f} ({yearly_corr.idxmin()})")
print(f"  Max:    {yearly_corr.max():.6f} ({yearly_corr.idxmax()})")

# Positive correlation years
positive_years = yearly_corr[yearly_corr > 0]
print(f"\nYears with positive correlation: {len(positive_years)}/{len(yearly_corr)} ({len(positive_years)/len(yearly_corr)*100:.1f}%)")

# Directional accuracy
results_df['pred_sign'] = np.sign(results_df[pred_col])
results_df['actual_sign'] = np.sign(results_df[actual_col])
results_df['correct_direction'] = results_df['pred_sign'] == results_df['actual_sign']

directional_accuracy = results_df['correct_direction'].mean()
print(f"\nDirectional accuracy: {directional_accuracy*100:.2f}%")

# Directional accuracy by year
yearly_dir_acc = results_df.groupby('year')['correct_direction'].mean()
print(f"\nDirectional accuracy by year:")
print((yearly_dir_acc * 100).to_string())

print(f"\nYearly directional accuracy summary:")
print(f"  Mean:   {yearly_dir_acc.mean()*100:.2f}%")
print(f"  Median: {yearly_dir_acc.median()*100:.2f}%")
print(f"  Std:    {yearly_dir_acc.std()*100:.2f}%")
print(f"  Min:    {yearly_dir_acc.min()*100:.2f}% ({yearly_dir_acc.idxmin()})")
print(f"  Max:    {yearly_dir_acc.max()*100:.2f}% ({yearly_dir_acc.idxmax()})")

# Check prediction distribution
print("\n" + "=" * 80)
print("PREDICTION DISTRIBUTION")
print("=" * 80)
print(f"\nActual returns:")
print(f"  Mean:   {results_df[actual_col].mean():.6f}")
print(f"  Std:    {results_df[actual_col].std():.6f}")
print(f"  Min:    {results_df[actual_col].min():.6f}")
print(f"  Max:    {results_df[actual_col].max():.6f}")

print(f"\nPredicted returns:")
print(f"  Mean:   {results_df[pred_col].mean():.6f}")
print(f"  Std:    {results_df[pred_col].std():.6f}")
print(f"  Min:    {results_df[pred_col].min():.6f}")
print(f"  Max:    {results_df[pred_col].max():.6f}")

# Check if predictions are nearly constant
pred_range = results_df[pred_col].max() - results_df[pred_col].min()
actual_range = results_df[actual_col].max() - results_df[actual_col].min()
range_ratio = pred_range / actual_range

print(f"\nPrediction range: {pred_range:.6f}")
print(f"Actual range:     {actual_range:.6f}")
print(f"Range ratio:      {range_ratio:.4f}")

if range_ratio < 0.1:
    print(f"\nWARNING: Predictions have very narrow range ({range_ratio*100:.2f}% of actual range)")
    print(f"This suggests the model is predicting near-constant values.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
