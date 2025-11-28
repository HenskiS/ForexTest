"""
Analyze what predicts when each pair will perform well.

Instead of ATR, let's look at:
1. Signal strength (how far prediction is from threshold)
2. Recent model accuracy (has the model been right lately?)
3. Prediction magnitude (absolute value of prediction)
4. Market regime indicators (trending vs ranging)
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import sys

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = 1000

print(f"Signal Predictor Analysis")
print("=" * 80)
print(f"Finding what predicts when each pair will perform well...")
print("=" * 80)

def calculate_features(df):
    """Calculate all technical features"""
    df['momentum'] = df['close'].pct_change()
    df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['range'] = df['high'] - df['low']
    df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    for period in [10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    lowest_low = df['low'].rolling(window=14).min()
    highest_high = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)

    df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

    middle = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = middle + (std * 2)
    df['bb_middle'] = middle
    df['bb_lower'] = middle - (std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['atr'] = atr
    df['atr_pct'] = (df['atr'] / df['close']) * 100

    return df

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# Load data
print("\nLoading data...")
pair_data = {}
for pair in PAIRS:
    data_file = f'data/{pair}_1day_oanda.csv'
    df_raw = pd.read_csv(data_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')
    df = calculate_features(df_raw.copy())
    df[TARGET] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=technical_features + [TARGET])
    pair_data[pair] = df
    print(f"  {pair}: {len(df)} days")

best_params = {
    'n_estimators': 125,
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7
}

# Generate predictions with additional metrics
print("\nGenerating predictions and metrics...")
pair_trades = {}
transaction_cost_pct = 0.0002
stop_loss_pct = 0.0018
take_profit_pct = 0.0200

for pair in PAIRS:
    print(f"\n{pair}:")
    df = pair_data[pair]
    predictions = []
    test_indices = []
    start_idx = len(df) - TEST_DAYS

    for i in tqdm(range(start_idx, len(df)), desc=f"  Predicting"):
        if i < TRAIN_WINDOW_SIZE:
            continue

        train_end_idx = i - 1
        train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
        train_data = df.iloc[train_start_idx:train_end_idx]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_data[technical_features])
        y_train = train_data[TARGET].values

        model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, verbose=False)

        X_today = scaler.transform(df.iloc[[i]][technical_features])
        y_pred = model.predict(X_today)[0]

        predictions.append(y_pred)
        test_indices.append(i)

    predictions = np.array(predictions)
    test_data = df.iloc[test_indices]

    # Backtest with enhanced metrics
    trades = []
    for date_idx in range(len(predictions)):
        if date_idx < 50:
            continue

        pred_buffer = predictions[max(0, date_idx-200):date_idx+1]
        if len(pred_buffer) < 50:
            continue

        lower_threshold = np.percentile(pred_buffer, 48)
        upper_threshold = np.percentile(pred_buffer, 52)
        prediction = predictions[date_idx]

        # Calculate signal strength (distance from threshold)
        if prediction >= upper_threshold:
            signal = 1
            signal_strength = (prediction - upper_threshold) / abs(upper_threshold) if upper_threshold != 0 else 0
        elif prediction <= lower_threshold:
            signal = -1
            signal_strength = (lower_threshold - prediction) / abs(lower_threshold) if lower_threshold != 0 else 0
        else:
            continue

        # Calculate recent model accuracy (last 10 trades)
        if len(trades) >= 10:
            recent_trades = trades[-10:]
            recent_wins = sum(1 for t in recent_trades if t['net_return_pct'] > 0)
            recent_accuracy = recent_wins / 10
        else:
            recent_accuracy = 0.5  # Default

        # Get trade data
        row = test_data.iloc[date_idx]
        entry_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        atr_pct = row['atr_pct']
        adx = row['adx']

        # Calculate exit
        if signal == 1:
            pct_high = (high - entry_price) / entry_price
            pct_low = (low - entry_price) / entry_price
        else:
            pct_high = (entry_price - low) / entry_price
            pct_low = (entry_price - high) / entry_price

        if pct_low <= -stop_loss_pct:
            exit_price = entry_price * (1 - stop_loss_pct) if signal == 1 else entry_price * (1 + stop_loss_pct)
        elif pct_high >= take_profit_pct:
            exit_price = entry_price * (1 + take_profit_pct) if signal == 1 else entry_price * (1 - take_profit_pct)
        else:
            exit_price = close

        raw_return_pct = ((exit_price - entry_price) / entry_price if signal == 1
                          else (entry_price - exit_price) / entry_price) * 100
        net_return_pct = raw_return_pct - (transaction_cost_pct * 100)

        trades.append({
            'date': test_data.index[date_idx],
            'prediction': prediction,
            'prediction_abs': abs(prediction),
            'signal_strength': signal_strength,
            'recent_accuracy': recent_accuracy,
            'atr_pct': atr_pct,
            'adx': adx,
            'net_return_pct': net_return_pct
        })

    pair_trades[pair] = pd.DataFrame(trades)
    print(f"  Total trades: {len(trades)}")

# Analyze what predicts performance for each pair
print("\n" + "=" * 80)
print("SIGNAL PREDICTOR ANALYSIS")
print("=" * 80)

for pair in PAIRS:
    trades_df = pair_trades[pair]

    print(f"\n{pair}:")
    print("-" * 80)

    # Segment by signal strength
    trades_df['signal_strength_quartile'] = pd.qcut(trades_df['signal_strength'], q=4, labels=['Q1 (Weak)', 'Q2', 'Q3', 'Q4 (Strong)'], duplicates='drop')

    print("\nBy Signal Strength (distance from threshold):")
    for quartile in ['Q1 (Weak)', 'Q2', 'Q3', 'Q4 (Strong)']:
        subset = trades_df[trades_df['signal_strength_quartile'] == quartile]
        if len(subset) > 0:
            avg_return = subset['net_return_pct'].mean()
            win_rate = (subset['net_return_pct'] > 0).sum() / len(subset) * 100
            print(f"  {quartile:<15} Trades: {len(subset):>4}  Avg Return: {avg_return:>6.3f}%  Win Rate: {win_rate:>5.1f}%")

    # Segment by recent accuracy
    trades_df['recent_acc_quartile'] = pd.qcut(trades_df['recent_accuracy'], q=4, labels=['Q1 (Cold)', 'Q2', 'Q3', 'Q4 (Hot)'], duplicates='drop')

    print("\nBy Recent Model Accuracy (last 10 trades):")
    for quartile in ['Q1 (Cold)', 'Q2', 'Q3', 'Q4 (Hot)']:
        subset = trades_df[trades_df['recent_acc_quartile'] == quartile]
        if len(subset) > 0:
            avg_return = subset['net_return_pct'].mean()
            win_rate = (subset['net_return_pct'] > 0).sum() / len(subset) * 100
            print(f"  {quartile:<15} Trades: {len(subset):>4}  Avg Return: {avg_return:>6.3f}%  Win Rate: {win_rate:>5.1f}%")

    # Segment by prediction magnitude
    trades_df['pred_abs_quartile'] = pd.qcut(trades_df['prediction_abs'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')

    print("\nBy Prediction Magnitude (absolute value):")
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        subset = trades_df[trades_df['pred_abs_quartile'] == quartile]
        if len(subset) > 0:
            avg_return = subset['net_return_pct'].mean()
            win_rate = (subset['net_return_pct'] > 0).sum() / len(subset) * 100
            print(f"  {quartile:<15} Trades: {len(subset):>4}  Avg Return: {avg_return:>6.3f}%  Win Rate: {win_rate:>5.1f}%")

    # Segment by ADX (trend strength)
    trades_df['adx_quartile'] = pd.qcut(trades_df['adx'], q=4, labels=['Q1 (Ranging)', 'Q2', 'Q3', 'Q4 (Trending)'], duplicates='drop')

    print("\nBy ADX (trend strength):")
    for quartile in ['Q1 (Ranging)', 'Q2', 'Q3', 'Q4 (Trending)']:
        subset = trades_df[trades_df['adx_quartile'] == quartile]
        if len(subset) > 0:
            avg_return = subset['net_return_pct'].mean()
            win_rate = (subset['net_return_pct'] > 0).sum() / len(subset) * 100
            print(f"  {quartile:<15} Trades: {len(subset):>4}  Avg Return: {avg_return:>6.3f}%  Win Rate: {win_rate:>5.1f}%")

# Find best predictor for each pair
print("\n" + "=" * 80)
print("BEST PREDICTOR FOR EACH PAIR")
print("=" * 80)

for pair in PAIRS:
    trades_df = pair_trades[pair]

    # Calculate correlation between each metric and returns
    correlations = {
        'Signal Strength': trades_df['signal_strength'].corr(trades_df['net_return_pct']),
        'Recent Accuracy': trades_df['recent_accuracy'].corr(trades_df['net_return_pct']),
        'Prediction Magnitude': trades_df['prediction_abs'].corr(trades_df['net_return_pct']),
        'ADX': trades_df['adx'].corr(trades_df['net_return_pct']),
        'ATR%': trades_df['atr_pct'].corr(trades_df['net_return_pct'])
    }

    best_predictor = max(correlations, key=lambda k: abs(correlations[k]))

    print(f"\n{pair}:")
    for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        marker = " ***" if metric == best_predictor else ""
        print(f"  {metric:<25} Correlation: {corr:>7.4f}{marker}")

# Propose diversification strategy
print("\n" + "=" * 80)
print("PROPOSED DIVERSIFICATION STRATEGY")
print("=" * 80)
print("""
Based on the analysis above, here are potential strategies:

1. BEST SIGNAL STRENGTH: Trade the pair with the strongest signal (highest distance from threshold)
   - Theory: Confidence matters - trade when the model is most certain

2. HOT HAND: Trade the pair where the model has been most accurate recently
   - Theory: Momentum in model accuracy - ride the hot hand

3. HYBRID SCORE: Combine signal strength + recent accuracy + pair-specific predictor
   - Theory: Multi-factor model captures more information

4. ENSEMBLE: Trade top 2 pairs by score, split capital 50/50
   - Theory: Diversification reduces risk while capturing best opportunities

Next step: Backtest these strategies to see which actually works!
""")
