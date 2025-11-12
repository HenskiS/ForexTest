"""
Test XGBoost with the EXACT parameters from the study.
"""

import pandas as pd
import numpy as np
import xgboost as xgb

print("FEATURE IMPORTANCE ANALYSIS - STUDY'S EXACT PARAMETERS")
print("="*80)

# Load 4-hour EURUSD data
df = pd.read_csv('data/EURUSD_4hour_with_features.csv',
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

TARGET_COLUMN = 'target_1day_return'
df_clean = df.dropna(subset=technical_features + [TARGET_COLUMN])

print(f"Data loaded: {df_clean.shape}")
print(f"Features: {len(technical_features)}")

# Train on first window only
TRAIN_BARS = 1800
VAL_BARS = 468

train_end = TRAIN_BARS
val_end = train_end + VAL_BARS

X_train = df_clean[technical_features].iloc[:train_end]
y_train = df_clean[TARGET_COLUMN].iloc[:train_end]
X_val = df_clean[technical_features].iloc[train_end:val_end]
y_val = df_clean[TARGET_COLUMN].iloc[train_end:val_end]

print(f"\nTraining on window 1:")
print(f"  Train samples: {len(X_train)}")
print(f"  Val samples:   {len(X_val)}")

# TEST BOTH PARAMETER SETS FROM THE STUDY
configs = [
    {
        'name': 'Study 4H Config 1',
        'n_estimators': 125,
        'learning_rate': 0.10,
        'max_depth': 18,
        'gamma': 0.10,
    },
    {
        'name': 'Study 4H Config 2',
        'n_estimators': 200,
        'learning_rate': 0.11,
        'max_depth': 14,
        'gamma': 0.003,
    },
]

for config in configs:
    print("\n" + "="*80)
    print(f"TESTING: {config['name']}")
    print("="*80)

    # Train with exact study parameters
    model = xgb.XGBRegressor(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        gamma=config['gamma'],
        subsample=0.8,  # Typical default
        colsample_bytree=0.8,  # Typical default
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    print("\nModel parameters:")
    print(f"  n_estimators:    {config['n_estimators']}")
    print(f"  learning_rate:   {config['learning_rate']}")
    print(f"  max_depth:       {config['max_depth']}")
    print(f"  gamma:           {config['gamma']}")

    model.fit(X_train, y_train)

    # Check tree structure
    booster = model.get_booster()
    trees_str = booster.get_dump()
    print(f"\nTotal trees: {len(trees_str)}")

    # Check first tree
    first_tree = trees_str[0]
    tree_lines = first_tree.split('\n')
    actual_depth = max([line.count('\t') for line in tree_lines])
    print(f"First tree actual depth: {actual_depth}")
    print(f"First tree has {len([l for l in tree_lines if 'leaf' in l])} leaf nodes")

    # Validate
    y_val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(y_val - y_val_pred))

    # Metrics
    mean_pred_mae = np.mean(np.abs(y_val - y_val.mean()))
    improvement = (mean_pred_mae - val_mae) / mean_pred_mae * 100
    correlation = np.corrcoef(y_val, y_val_pred)[0, 1]

    print(f"\nPrediction Analysis:")
    print(f"  Actual - Min: {y_val.min():.6f}, Max: {y_val.max():.6f}, Mean: {y_val.mean():.6f}, Std: {y_val.std():.6f}")
    print(f"  Predicted - Min: {y_val_pred.min():.6f}, Max: {y_val_pred.max():.6f}, Mean: {y_val_pred.mean():.6f}, Std: {y_val_pred.std():.6f}")
    print(f"  Prediction range: {y_val_pred.max() - y_val_pred.min():.6f}")
    print(f"  Actual range: {y_val.max() - y_val.min():.6f}")

    print(f"\nPerformance:")
    print(f"  MAE (baseline - predict mean): {mean_pred_mae:.6f}")
    print(f"  MAE (model):                   {val_mae:.6f}")
    print(f"  Improvement over mean:         {improvement:+.2f}%")
    print(f"  Correlation (pred vs actual):  {correlation:.4f}")

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': technical_features,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    if importance_df['importance'].sum() > 0:
        importance_df['pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100

        print(f"\nTop 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:20s}: {row['pct']:5.2f}%")

        # Category analysis
        def categorize_feature(feat):
            if feat in ['momentum', 'avg_price', 'range', 'ohlc']:
                return 'Basic Price'
            elif 'ema' in feat:
                return 'Moving Averages'
            elif 'macd' in feat:
                return 'MACD'
            elif feat in ['adx', 'plus_di', 'minus_di']:
                return 'ADX/Directional'
            elif feat in ['rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r']:
                return 'Oscillators'
            elif 'bb_' in feat:
                return 'Bollinger Bands'
            elif feat == 'atr':
                return 'Volatility (ATR)'
            return 'Other'

        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        category_importance = importance_df.groupby('category')['pct'].sum().sort_values(ascending=False)

        print(f"\nFeature Categories:")
        for cat, pct in category_importance.items():
            print(f"  {cat:20s}: {pct:5.2f}%")
    else:
        print("\nWARNING: All feature importances are zero!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
