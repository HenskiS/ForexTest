"""
Quick test to compare old default params vs new config params across all 4 pairs.
Tests on last 250 days to be fast.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from trading.config import TradingConfig
from trading.market_utils import calculate_technical_features

def test_params_on_pair(pair, params, train_window=378, test_days=250):
    """Test params on a pair and return accuracy metrics"""

    # Load data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate features
    df = calculate_technical_features(df)
    df = df.dropna()

    # Use last test_days for testing
    if len(df) < train_window + test_days:
        return None

    test_data = df.iloc[-(train_window + test_days):]

    # Rolling predictions
    predictions = []
    actuals = []

    for i in range(len(test_data) - train_window - 1):
        train_end = train_window + i
        train = test_data.iloc[:train_end]

        # Target: next day return
        train_target = train['close'].shift(-1) / train['close'] - 1
        train_target = train_target.iloc[:-1]
        train_features = train[TradingConfig.TECHNICAL_FEATURES].iloc[:-1]

        # Scale
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_features)

        # Train
        model = xgb.XGBRegressor(**params)
        model.fit(train_scaled, train_target, verbose=False)

        # Predict next day
        test_features = test_data[TradingConfig.TECHNICAL_FEATURES].iloc[train_end:train_end+1]
        test_scaled = scaler.transform(test_features)
        pred = model.predict(test_scaled)[0]

        # Actual
        actual = test_data['close'].iloc[train_end+1] / test_data['close'].iloc[train_end] - 1

        predictions.append(pred)
        actuals.append(actual)

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Direction accuracy
    direction_accuracy = np.mean((predictions > 0) == (actuals > 0))

    # Mean absolute error
    mae = np.mean(np.abs(predictions - actuals))

    # Correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    return {
        'direction_accuracy': direction_accuracy,
        'mae': mae,
        'correlation': correlation,
        'n_predictions': len(predictions)
    }

# Test parameters
OLD_PARAMS = {
    'n_estimators': 250,
    'learning_rate': 0.05,
    'max_depth': 10,
    'gamma': 0.001,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

NEW_PARAMS = TradingConfig.XGBOOST_PARAMS.copy()

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

print("="*80)
print("COMPARING HYPERPARAMETERS ACROSS ALL 4 PAIRS")
print("="*80)
print(f"\nOLD PARAMS: {OLD_PARAMS}")
print(f"\nNEW PARAMS: {NEW_PARAMS}")
print("\nTesting on last 250 days per pair...")
print("="*80)

results = {}

for pair in PAIRS:
    print(f"\n{pair}:")
    print("-"*80)

    try:
        print("  Testing OLD params...", end=" ", flush=True)
        old_result = test_params_on_pair(pair, OLD_PARAMS, test_days=250)
        print("Done")

        print("  Testing NEW params...", end=" ", flush=True)
        new_result = test_params_on_pair(pair, NEW_PARAMS, test_days=250)
        print("Done")

        if old_result and new_result:
            results[pair] = {'old': old_result, 'new': new_result}

            print(f"\n  OLD: Accuracy={old_result['direction_accuracy']:.1%}, MAE={old_result['mae']:.5f}, Corr={old_result['correlation']:.3f}")
            print(f"  NEW: Accuracy={new_result['direction_accuracy']:.1%}, MAE={new_result['mae']:.5f}, Corr={new_result['correlation']:.3f}")

            # Which is better?
            if new_result['direction_accuracy'] > old_result['direction_accuracy']:
                print("  [OK] NEW params are BETTER")
            elif new_result['direction_accuracy'] < old_result['direction_accuracy']:
                print("  [WARNING] OLD params are better")
            else:
                print("  [NEUTRAL] Same performance")
    except Exception as e:
        print(f"  [ERROR] {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

avg_old_acc = np.mean([r['old']['direction_accuracy'] for r in results.values()])
avg_new_acc = np.mean([r['new']['direction_accuracy'] for r in results.values()])

print(f"\nAverage Direction Accuracy:")
print(f"  OLD params: {avg_old_acc:.1%}")
print(f"  NEW params: {avg_new_acc:.1%}")

if avg_new_acc > avg_old_acc:
    print(f"\n[RECOMMENDATION] Use NEW params (config) - {(avg_new_acc - avg_old_acc)*100:.2f}% better")
elif avg_old_acc > avg_new_acc:
    print(f"\n[RECOMMENDATION] Use OLD params - {(avg_old_acc - avg_new_acc)*100:.2f}% better")
else:
    print(f"\n[RECOMMENDATION] Both params perform similarly, use NEW params for consistency")
