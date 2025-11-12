"""
Extract and analyze feature importance from trained XGBoost models.
Retrains models using saved best_params and aggregates feature importance across all windows.
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import json

print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)
print("Extracting feature importance from 46 trained windows")
print("="*80)

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

# Feature names (same order as training)
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

TARGET_COLUMN = 'target_5day_return'

# Load data
print("\nLoading data...")
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]
df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])

print(f"Data loaded: {df_clean.shape}")

# Load results
print("\nLoading trained model results...")
with open('xgboost_results_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows")


def generate_windows(df, window_size, roll_days, min_windows=40):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'window_id': len(windows),
            'train_start': start_idx,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': test_end,
            'date_start': df.index[start_idx],
            'date_end': df.index[test_end - 1]
        }

        windows.append(window)
        start_idx += roll_days

        if len(windows) >= min_windows:
            break

    return windows


def prepare_window_data(df, window, feature_cols, target_col):
    """Prepare train, validation, and test data for one window."""
    train_data = df.iloc[window['train_start']:window['train_end']]
    val_data = df.iloc[window['val_start']:window['val_end']]

    # Fit scaler on training data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    X_val = scaler.transform(val_data[feature_cols])

    y_train = train_data[target_col].values
    y_val = val_data[target_col].values

    return X_train, y_train, X_val, y_val


# Generate windows (match the number of results)
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=len(all_results))

# Extract feature importance for each window
print("\nExtracting feature importance from each window...")
all_importances = []

# Use only the number of windows available
n_windows = min(len(all_results), len(windows))

for window_idx in range(n_windows):
    result = all_results[window_idx]
    window = windows[window_idx]

    if (window_idx + 1) % 10 == 0:
        print(f"  Processing window {window_idx + 1}/{len(all_results)}...")

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_window_data(
        df_clean, window, FEATURE_COLS, TARGET_COLUMN
    )

    # Retrain model with best params
    best_params = result['best_params']

    model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        gamma=best_params['gamma'],
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Get feature importance (gain-based)
    importances = model.get_booster().get_score(importance_type='gain')

    # Convert to feature names (XGBoost uses f0, f1, f2, etc.)
    importance_dict = {}
    for key, value in importances.items():
        feature_idx = int(key[1:])  # Extract number from 'f0', 'f1', etc.
        feature_name = FEATURE_COLS[feature_idx]
        importance_dict[feature_name] = value

    # Add zeros for features not used
    for feature in FEATURE_COLS:
        if feature not in importance_dict:
            importance_dict[feature] = 0.0

    all_importances.append(importance_dict)

print(f"  Completed all {n_windows} windows")

# Aggregate feature importance
print("\nAggregating feature importance across all windows...")

# Calculate statistics for each feature
feature_stats = {}

for feature in FEATURE_COLS:
    values = [imp[feature] for imp in all_importances]

    feature_stats[feature] = {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'pct_nonzero': np.sum(np.array(values) > 0) / len(values) * 100
    }

# Sort by mean importance
sorted_features = sorted(feature_stats.items(),
                        key=lambda x: x[1]['mean'],
                        reverse=True)

# Display results
print("\n" + "="*80)
print("FEATURE IMPORTANCE RANKING (sorted by mean gain)")
print("="*80)
print(f"\n{'Rank':<6} {'Feature':<20} {'Mean':>12} {'Median':>12} {'Std':>12} {'% Used':>10}")
print("-" * 80)

for rank, (feature, stats) in enumerate(sorted_features, 1):
    print(f"{rank:<6} {feature:<20} {stats['mean']:11.2f}  {stats['median']:11.2f}  "
          f"{stats['std']:11.2f}  {stats['pct_nonzero']:8.1f}%")

# Group by feature type
print("\n" + "="*80)
print("FEATURE IMPORTANCE BY CATEGORY")
print("="*80)

categories = {
    'Price Action': ['momentum', 'avg_price', 'range', 'ohlc'],
    'EMAs': ['ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200'],
    'MACD': ['macd', 'macd_signal', 'macd_hist'],
    'Momentum': ['adx', 'plus_di', 'minus_di', 'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r'],
    'Bollinger Bands': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'],
    'Volatility': ['atr']
}

category_importances = {}
for category, features in categories.items():
    total_importance = sum(feature_stats[f]['mean'] for f in features if f in feature_stats)
    category_importances[category] = total_importance

# Sort categories by importance
sorted_categories = sorted(category_importances.items(),
                          key=lambda x: x[1],
                          reverse=True)

print(f"\n{'Category':<20} {'Total Importance':>20} {'% of Total':>15}")
print("-" * 60)

total_all = sum(category_importances.values())
for category, importance in sorted_categories:
    pct = (importance / total_all * 100) if total_all > 0 else 0
    print(f"{category:<20} {importance:18.2f}  {pct:13.1f}%")

# Top 10 features
print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*80)

top_10 = sorted_features[:10]
top_10_total = sum(stats['mean'] for _, stats in top_10)
total_importance = sum(stats['mean'] for _, stats in sorted_features)

print(f"\nThese 10 features account for {(top_10_total/total_importance)*100:.1f}% of total importance:\n")

for rank, (feature, stats) in enumerate(top_10, 1):
    pct_of_total = (stats['mean'] / total_importance * 100) if total_importance > 0 else 0
    print(f"{rank:2d}. {feature:<20} {stats['mean']:10.2f}  ({pct_of_total:5.1f}%)")

# Bottom 10 features
print("\n" + "="*80)
print("LEAST IMPORTANT FEATURES")
print("="*80)

bottom_10 = sorted_features[-10:]
print()
for rank, (feature, stats) in enumerate(bottom_10, 1):
    pct_of_total = (stats['mean'] / total_importance * 100) if total_importance > 0 else 0
    print(f"{rank:2d}. {feature:<20} {stats['mean']:10.2f}  ({pct_of_total:5.1f}%)")

# Consistency analysis
print("\n" + "="*80)
print("CONSISTENCY ANALYSIS")
print("="*80)
print("\nFeatures used consistently across windows (>90% of windows):\n")

consistent_features = [(f, s) for f, s in sorted_features if s['pct_nonzero'] > 90]
if consistent_features:
    for feature, stats in consistent_features:
        print(f"  {feature:<20} Used in {stats['pct_nonzero']:.1f}% of windows")
else:
    print("  No features used in >90% of windows")

print("\nFeatures rarely used (<50% of windows):\n")
rare_features = [(f, s) for f, s in sorted_features if s['pct_nonzero'] < 50]
if rare_features:
    for feature, stats in rare_features:
        print(f"  {feature:<20} Used in {stats['pct_nonzero']:.1f}% of windows")
else:
    print("  All features used in >50% of windows")

# Save results
output_file = 'feature_importance_analysis.json'
output_data = {
    'feature_stats': {
        feature: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in stats.items()}
        for feature, stats in sorted_features
    },
    'category_importances': {
        category: float(importance)
        for category, importance in sorted_categories
    },
    'top_10_features': [feature for feature, _ in top_10],
    'summary': {
        'total_windows': n_windows,
        'total_features': len(FEATURE_COLS),
        'top_10_pct_importance': float(top_10_total / total_importance * 100) if total_importance > 0 else 0
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}")

# Summary insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = []

# Most important feature
top_feature, top_stats = sorted_features[0]
insights.append(f"1. '{top_feature}' is the most important feature (mean gain: {top_stats['mean']:.2f})")

# Most important category
top_category, top_cat_importance = sorted_categories[0]
pct_cat = (top_cat_importance / total_all * 100) if total_all > 0 else 0
insights.append(f"2. '{top_category}' is the most important category ({pct_cat:.1f}% of importance)")

# Top 10 concentration
top_10_pct = (top_10_total / total_importance * 100) if total_importance > 0 else 0
insights.append(f"3. Top 10 features account for {top_10_pct:.1f}% of total importance")

# Most consistent feature
most_consistent = max(sorted_features, key=lambda x: x[1]['pct_nonzero'])
insights.append(f"4. '{most_consistent[0]}' is most consistent (used in {most_consistent[1]['pct_nonzero']:.1f}% of windows)")

print()
for insight in insights:
    print(f"  {insight}")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
