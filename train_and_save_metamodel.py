"""
Train metamodel and save predictions for reuse.
This avoids rerunning the entire training process every time we want to test.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']

# Load all data
print("Loading data...")
pair_data = {}
for pair in ALL_PAIRS:
    with open(f'optimized_ann_predictions/predictions_{pair}.pkl', 'rb') as f:
        data = pickle.load(f)
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        df = df.join(spread_df[['spread_pct']], how='left')
    except:
        df['spread_pct'] = 0.00025
    df['spread_pct'] = df['spread_pct'].fillna(0.00025)

    pred_dates = data.get('dates', None)
    pair_data[pair] = {'predictions': data['predictions'], 'dates': pred_dates, 'df': df}

print("Data loaded!")
print()

# Trading parameters
HOLD_DAYS = 5
SL_PCT = 0.025
TRAIN_SIZE = 2500
LOOKBACK_TRADES = 20

def calculate_hypothetical_outcome(pair, pred_date, prediction, spread):
    """Calculate what outcome WOULD be if we took this trade"""
    df = pair_data[pair]['df']

    if pred_date not in df.index:
        return None

    pred_idx = df.index.get_loc(pred_date)
    entry_idx = pred_idx + 1

    if entry_idx >= len(df) - HOLD_DAYS:
        return None

    entry_row = df.iloc[entry_idx]
    direction = 1 if prediction > 0 else -1

    if direction == 1:
        actual_entry = entry_row['open'] * (1 + spread)
    else:
        actual_entry = entry_row['open'] * (1 - spread)

    # Check for SL
    for days in range(HOLD_DAYS + 1):
        check_idx = entry_idx + days
        if check_idx >= len(df):
            return None
        check_row = df.iloc[check_idx]

        if direction == 1:
            sl_price = actual_entry * (1 - SL_PCT)
            if check_row['low'] <= sl_price:
                exit_price = sl_price * (1 - spread)
                pnl = (exit_price / actual_entry) - 1
                return pnl > 0
        else:
            sl_price = actual_entry * (1 + SL_PCT)
            if check_row['high'] >= sl_price:
                exit_price = sl_price * (1 + spread)
                pnl = (actual_entry / exit_price) - 1
                return pnl > 0

    # Time exit
    exit_row = df.iloc[entry_idx + HOLD_DAYS]
    if direction == 1:
        exit_price = exit_row['open'] * (1 - spread)
        pnl = (exit_price / actual_entry) - 1
    else:
        exit_price = exit_row['open'] * (1 + spread)
        pnl = (actual_entry / exit_price) - 1

    return pnl > 0

def extract_features(pair, pred_idx, prediction, recent_outcomes, df):
    """Extract features for metamodel"""
    if pred_idx < 100:
        return None

    features = []

    # Prediction features
    features.append(prediction)
    features.append(abs(prediction))
    features.append(1 if prediction > 0 else -1)

    # Recent performance
    if len(recent_outcomes) >= 5:
        features.append(sum(recent_outcomes[-LOOKBACK_TRADES:]) / min(len(recent_outcomes), LOOKBACK_TRADES))
        features.append(sum(recent_outcomes[-5:]) / 5)
    else:
        features.append(0.5)
        features.append(0.5)

    # Market features
    current_row = df.iloc[pred_idx]
    features.append(current_row.get('atr', 0.01) if 'atr' in df.columns else 0.01)
    features.append(current_row['spread_pct'])

    # Price action
    if pred_idx >= 20:
        recent_prices = df.iloc[pred_idx-20:pred_idx]['close']
        features.append(recent_prices.pct_change().std())
        features.append((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1)
    else:
        features.append(0.01)
        features.append(0.0)

    # Day of week
    pred_date = df.index[pred_idx]
    features.append(pred_date.dayofweek)
    features.append(1 if pred_date.dayofweek == 0 else 0)
    features.append(1 if pred_date.dayofweek == 4 else 0)

    return features

# Step 1: Build training dataset
print("Building training dataset...")
X_train, y_train = [], []

for pair in ALL_PAIRS:
    predictions = pair_data[pair]['predictions']
    pred_dates = pair_data[pair]['dates']
    df = pair_data[pair]['df']
    recent_outcomes = []

    for i, (pred_date, prediction) in enumerate(zip(pred_dates[:TRAIN_SIZE], predictions[:TRAIN_SIZE])):
        if pred_date not in df.index:
            continue

        pred_idx = df.index.get_loc(pred_date)
        spread = df.iloc[pred_idx]['spread_pct']

        # Get hypothetical outcome
        outcome = calculate_hypothetical_outcome(pair, pred_date, prediction, spread)
        if outcome is None:
            continue

        # Extract features
        features = extract_features(pair, pred_idx, prediction, recent_outcomes, df)
        if features is not None:
            X_train.append(features)
            y_train.append(1 if outcome else 0)

        recent_outcomes.append(outcome)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training samples: {len(X_train)}")
print(f"Training win rate: {y_train.mean()*100:.1f}%")
print()

# Step 2: Train metamodel
print("Training metamodel...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)
print("Metamodel trained!")
print()

# Step 3: Generate predictions for all test data
print("Generating predictions for test data...")
all_predictions = []

for pair in ALL_PAIRS:
    predictions = pair_data[pair]['predictions'][TRAIN_SIZE:]
    pred_dates = pair_data[pair]['dates'][TRAIN_SIZE:]
    df = pair_data[pair]['df']
    recent_outcomes = []

    for idx, (pred_date, prediction) in enumerate(zip(pred_dates, predictions)):
        if pred_date not in df.index:
            continue

        pred_idx = df.index.get_loc(pred_date)
        spread = df.iloc[pred_idx]['spread_pct']

        # Extract features
        features = extract_features(pair, pred_idx, prediction, recent_outcomes, df)

        if features is not None:
            # Get metamodel confidence
            X_sample = scaler.transform([features])
            confidence = clf.predict_proba(X_sample)[0, 1]

            # Store prediction
            all_predictions.append({
                'pair': pair,
                'date': pred_date,
                'prediction': prediction,
                'confidence': confidence,
                'spread': spread,
                'features': features
            })

        # Calculate outcome for tracking
        hypo_outcome = calculate_hypothetical_outcome(pair, pred_date, prediction, spread)
        if hypo_outcome is not None:
            recent_outcomes.append(hypo_outcome)

print(f"Generated {len(all_predictions)} predictions")
print()

# Step 4: Save everything
print("Saving metamodel and predictions...")
save_data = {
    'clf': clf,
    'scaler': scaler,
    'predictions': all_predictions,
    'train_size': TRAIN_SIZE,
    'hold_days': HOLD_DAYS,
    'sl_pct': SL_PCT,
    'lookback_trades': LOOKBACK_TRADES,
    'feature_names': ['prediction', 'pred_abs', 'pred_sign', 'recent_wr', 'last_5_wr',
                      'atr', 'spread', 'volatility', 'trend', 'day_of_week', 'is_monday', 'is_friday']
}

with open('metamodel_predictions.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("Saved to metamodel_predictions.pkl")
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Trained on {len(X_train)} samples")
print(f"Generated {len(all_predictions)} test predictions")
print(f"Confidence range: {min(p['confidence'] for p in all_predictions):.3f} - {max(p['confidence'] for p in all_predictions):.3f}")
print()
print("Top feature importances:")
for name, imp in sorted(zip(save_data['feature_names'], clf.feature_importances_), key=lambda x: -x[1])[:5]:
    print(f"  {name:<20} {imp:.4f}")
print("=" * 90)
