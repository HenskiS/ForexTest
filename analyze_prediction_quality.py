"""
Analyze the quality of XGBoost predictions.
Tests:
1. Correlation between predictions and actual returns
2. Prediction accuracy by quartile
3. Direction accuracy (sign prediction)
"""

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt

print("PREDICTION QUALITY ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'target_5day_return'
]
df_clean = df.dropna(subset=technical_features)

with open('xgboost_results_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

# Aggregate all predictions and actuals
all_predictions = []
all_actuals = []

for result in all_results:
    all_predictions.extend(result['predictions'])
    all_actuals.extend(result['actuals'])

all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

print(f"\nTotal samples: {len(all_predictions)}")

# 1. Overall Correlation
correlation = np.corrcoef(all_predictions, all_actuals)[0, 1]
pearson_r, pearson_p = stats.pearsonr(all_predictions, all_actuals)
spearman_r, spearman_p = stats.spearmanr(all_predictions, all_actuals)

print(f"\n{'='*80}")
print(f"1. CORRELATION WITH ACTUAL RETURNS")
print(f"{'='*80}")
print(f"  Pearson Correlation:   {pearson_r:7.4f} (p={pearson_p:.2e})")
print(f"  Spearman Correlation:  {spearman_r:7.4f} (p={spearman_p:.2e})")

if pearson_p < 0.001:
    print(f"  >>> HIGHLY SIGNIFICANT correlation (p < 0.001)")
elif pearson_p < 0.05:
    print(f"  >>> Significant correlation (p < 0.05)")
else:
    print(f"  >>> NOT significant")

# 2. Direction Accuracy
pred_direction = np.sign(all_predictions)
actual_direction = np.sign(all_actuals)
direction_accuracy = np.mean(pred_direction == actual_direction)

print(f"\n{'='*80}")
print(f"2. DIRECTION ACCURACY")
print(f"{'='*80}")
print(f"  Accuracy: {direction_accuracy*100:.2f}%")
print(f"  (Random baseline: 50.00%)")
print(f"  Improvement: {(direction_accuracy - 0.5)*100:+.2f}%")

# Statistical test for direction accuracy
n = len(all_predictions)
null_hypothesis_accuracy = 0.5
z_score = (direction_accuracy - null_hypothesis_accuracy) / np.sqrt(null_hypothesis_accuracy * (1 - null_hypothesis_accuracy) / n)
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"\n  Z-score: {z_score:.2f}")
print(f"  P-value: {p_value:.2e}")
if p_value < 0.001:
    print(f"  >>> HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.05:
    print(f"  >>> Significant (p < 0.05)")

# 3. Quartile Analysis
q1 = np.percentile(all_predictions, 25)
q3 = np.percentile(all_predictions, 75)

quartile_labels = np.zeros(len(all_predictions))
quartile_labels[all_predictions <= q1] = 1  # Bottom quartile (SHORT)
quartile_labels[all_predictions >= q3] = 3  # Top quartile (LONG)
quartile_labels[(all_predictions > q1) & (all_predictions < q3)] = 2  # Middle

print(f"\n{'='*80}")
print(f"3. RETURNS BY PREDICTION QUARTILE")
print(f"{'='*80}")

for q in [1, 3]:
    if q == 1:
        label = "Bottom Quartile (SHORT signals)"
        mask = quartile_labels == 1
    else:
        label = "Top Quartile (LONG signals)"
        mask = quartile_labels == 3

    quartile_predictions = all_predictions[mask]
    quartile_actuals = all_actuals[mask]

    avg_prediction = np.mean(quartile_predictions)
    avg_actual = np.mean(quartile_actuals)
    median_actual = np.median(quartile_actuals)
    positive_pct = np.mean(quartile_actuals > 0) * 100

    print(f"\n{label}:")
    print(f"  Samples:              {len(quartile_actuals)}")
    print(f"  Avg Prediction:       {avg_prediction*100:7.4f}%")
    print(f"  Avg Actual Return:    {avg_actual*100:7.4f}%")
    print(f"  Median Actual Return: {median_actual*100:7.4f}%")
    print(f"  % Positive Returns:   {positive_pct:7.2f}%")

# 4. Information Coefficient (IC)
# IC is the correlation between predictions and actuals, calculated per window
ics = []
for result in all_results:
    ic = np.corrcoef(result['predictions'], result['actuals'])[0, 1]
    if not np.isnan(ic):
        ics.append(ic)

ics = np.array(ics)

print(f"\n{'='*80}")
print(f"4. INFORMATION COEFFICIENT (IC) PER WINDOW")
print(f"{'='*80}")
print(f"  Mean IC:    {np.mean(ics):7.4f}")
print(f"  Median IC:  {np.median(ics):7.4f}")
print(f"  Std IC:     {np.std(ics):7.4f}")
print(f"  % IC > 0:   {np.mean(ics > 0)*100:7.2f}%")
print(f"  Min IC:     {np.min(ics):7.4f}")
print(f"  Max IC:     {np.max(ics):7.4f}")

# Test if IC is significantly different from 0
t_stat, t_pval = stats.ttest_1samp(ics, 0)
print(f"\n  T-test (IC != 0):")
print(f"    T-statistic: {t_stat:.4f}")
print(f"    P-value:     {t_pval:.2e}")
if t_pval < 0.001:
    print(f"    >>> HIGHLY SIGNIFICANT (p < 0.001)")
elif t_pval < 0.05:
    print(f"    >>> Significant (p < 0.05)")

# 5. Prediction vs Actual by Bins
print(f"\n{'='*80}")
print(f"5. ACTUAL RETURNS BY PREDICTION BIN")
print(f"{'='*80}")

# Create 10 bins based on predictions
bins = np.percentile(all_predictions, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
bin_labels = np.digitize(all_predictions, bins) - 1

print(f"\n{'Prediction Bin':<20} {'Avg Prediction':>15} {'Avg Actual':>15} {'Win Rate':>12} {'N':>8}")
print("-" * 80)

for bin_idx in range(10):
    mask = bin_labels == bin_idx
    if np.sum(mask) == 0:
        continue

    bin_predictions = all_predictions[mask]
    bin_actuals = all_actuals[mask]

    avg_pred = np.mean(bin_predictions)
    avg_actual = np.mean(bin_actuals)
    win_rate = np.mean(bin_actuals > 0)
    n_samples = len(bin_actuals)

    percentile_low = bin_idx * 10
    percentile_high = (bin_idx + 1) * 10

    print(f"{percentile_low:3d}-{percentile_high:3d} percentile   "
          f"{avg_pred*100:12.4f}%   "
          f"{avg_actual*100:12.4f}%   "
          f"{win_rate*100:9.2f}%   "
          f"{n_samples:7d}")

# 6. Long-only vs Short-only performance
long_mask = all_predictions >= q3
short_mask = all_predictions <= q1

long_actuals = all_actuals[long_mask]
short_actuals = -all_actuals[short_mask]  # Invert for short positions

print(f"\n{'='*80}")
print(f"6. LONG vs SHORT SIGNAL QUALITY")
print(f"{'='*80}")

print(f"\nLONG signals (top quartile):")
print(f"  Samples:          {len(long_actuals)}")
print(f"  Avg Return:       {np.mean(long_actuals)*100:7.4f}%")
print(f"  Median Return:    {np.median(long_actuals)*100:7.4f}%")
print(f"  Win Rate:         {np.mean(long_actuals > 0)*100:7.2f}%")
print(f"  Sharpe (approx):  {np.mean(long_actuals) / np.std(long_actuals):7.4f}")

print(f"\nSHORT signals (bottom quartile, inverted):")
print(f"  Samples:          {len(short_actuals)}")
print(f"  Avg Return:       {np.mean(short_actuals)*100:7.4f}%")
print(f"  Median Return:    {np.median(short_actuals)*100:7.4f}%")
print(f"  Win Rate:         {np.mean(short_actuals > 0)*100:7.2f}%")
print(f"  Sharpe (approx):  {np.mean(short_actuals) / np.std(short_actuals):7.4f}")

# T-test to see if long and short returns are significantly different from 0
long_tstat, long_pval = stats.ttest_1samp(long_actuals, 0)
short_tstat, short_pval = stats.ttest_1samp(short_actuals, 0)

print(f"\nStatistical significance:")
print(f"  LONG signals t-test (return != 0):  t={long_tstat:.3f}, p={long_pval:.2e}")
print(f"  SHORT signals t-test (return != 0): t={short_tstat:.3f}, p={short_pval:.2e}")

# 7. Summary
print(f"\n{'='*80}")
print(f"SUMMARY: IS THE MODEL PREDICTIVE?")
print(f"{'='*80}")

checks = []

# Check 1: Overall correlation
if pearson_p < 0.001:
    checks.append("[PASS] Highly significant correlation with actual returns")
else:
    checks.append("[FAIL] No significant correlation")

# Check 2: Direction accuracy
if p_value < 0.001 and direction_accuracy > 0.5:
    checks.append("[PASS] Direction accuracy significantly better than random")
else:
    checks.append("[FAIL] Direction accuracy not better than random")

# Check 3: IC positive
if t_pval < 0.001 and np.mean(ics) > 0:
    checks.append("[PASS] Information Coefficient significantly positive")
else:
    checks.append("[FAIL] IC not significantly positive")

# Check 4: Top quartile returns
if long_pval < 0.05 and np.mean(long_actuals) > 0:
    checks.append("[PASS] Top quartile predictions have positive returns")
else:
    checks.append("[FAIL] Top quartile not predictive")

# Check 5: Bottom quartile returns
if short_pval < 0.05 and np.mean(short_actuals) > 0:
    checks.append("[PASS] Bottom quartile predictions have negative returns (good for shorts)")
else:
    checks.append("[FAIL] Bottom quartile not predictive")

print()
for check in checks:
    print(f"  {check}")

n_passed = sum(1 for c in checks if "[PASS]" in c)
print(f"\n  {n_passed}/{len(checks)} checks passed")

if n_passed >= 4:
    print(f"\n  >>> MODEL IS HIGHLY PREDICTIVE")
elif n_passed >= 3:
    print(f"\n  >>> MODEL HAS PREDICTIVE POWER")
else:
    print(f"\n  >>> MODEL MAY NOT BE PREDICTIVE")

print(f"\n{'='*80}")
