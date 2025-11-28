"""
Optimize the lookback window for recent performance.

Test different window lengths (5, 10, 15, 20, 30 trades) to see which
best predicts when each pair will perform well.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = 1000

print(f"Lookback Window Optimization")
print("=" * 80)
print(f"Testing different lookback windows for recent accuracy...")
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

best_params = {
    'n_estimators': 125,
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7
}

# Generate predictions
print("\nGenerating predictions...")
pair_predictions = {}
transaction_cost_pct = 0.0002
stop_loss_pct = 0.0018
take_profit_pct = 0.0200

for pair in PAIRS:
    print(f"  {pair}...")
    df = pair_data[pair]
    predictions = []
    test_indices = []
    start_idx = len(df) - TEST_DAYS

    for i in range(start_idx, len(df)):
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

    # Backtest
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

        if prediction >= upper_threshold:
            signal = 1
        elif prediction <= lower_threshold:
            signal = -1
        else:
            continue

        row = test_data.iloc[date_idx]
        entry_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

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
            'net_return_pct': net_return_pct
        })

    pair_predictions[pair] = {
        'trades': pd.DataFrame(trades)
    }

# Test different lookback windows
print("\n" + "=" * 80)
print("LOOKBACK WINDOW OPTIMIZATION")
print("=" * 80)

lookback_windows = [3, 5, 7, 10, 15, 20, 30]

for pair in PAIRS:
    print(f"\n{pair}:")
    print("-" * 80)

    trades_df = pair_predictions[pair]['trades'].copy()

    results = []

    for window in lookback_windows:
        # Calculate recent accuracy with this window
        trades_df[f'recent_acc_{window}'] = 0.0

        for i in range(len(trades_df)):
            if i >= window:
                recent_trades = trades_df.iloc[i-window:i]
                recent_wins = (recent_trades['net_return_pct'] > 0).sum()
                trades_df.at[trades_df.index[i], f'recent_acc_{window}'] = recent_wins / window

        # Segment by recent accuracy quartiles
        valid_trades = trades_df[trades_df[f'recent_acc_{window}'] > 0].copy()

        if len(valid_trades) < 100:
            continue

        try:
            valid_trades['quartile'] = pd.qcut(valid_trades[f'recent_acc_{window}'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except ValueError:
            # Not enough unique values for quartiles, skip this window
            continue

        # Calculate performance for hot (Q4) vs cold (Q1)
        q4 = valid_trades[valid_trades['quartile'] == 'Q4']
        q1 = valid_trades[valid_trades['quartile'] == 'Q1']

        if len(q4) > 0 and len(q1) > 0:
            q4_return = q4['net_return_pct'].mean()
            q1_return = q1['net_return_pct'].mean()
            spread = q4_return - q1_return

            q4_winrate = (q4['net_return_pct'] > 0).sum() / len(q4) * 100
            q1_winrate = (q1['net_return_pct'] > 0).sum() / len(q1) * 100

            results.append({
                'window': window,
                'q4_return': q4_return,
                'q1_return': q1_return,
                'spread': spread,
                'q4_winrate': q4_winrate,
                'q1_winrate': q1_winrate,
                'q4_trades': len(q4),
                'q1_trades': len(q1)
            })

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('spread', ascending=False)

        print(f"\n{'Window':<10} {'Q4 (Hot) Return':<18} {'Q1 (Cold) Return':<18} {'Spread':<12} {'Q4 WR':<10} {'Q1 WR':<10}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            marker = " ***" if row['spread'] == results_df['spread'].max() else ""
            print(f"{int(row['window']):<10} {row['q4_return']:>12.3f}%     {row['q1_return']:>12.3f}%     {row['spread']:>7.3f}%   {row['q4_winrate']:>6.1f}%  {row['q1_winrate']:>6.1f}%{marker}")

        best = results_df.iloc[0]
        print(f"\nBEST WINDOW: {int(best['window'])} trades")
        print(f"  When hot (top 25%): {best['q4_return']:.3f}% avg return, {best['q4_winrate']:.1f}% win rate")
        print(f"  When cold (bottom 25%): {best['q1_return']:.3f}% avg return, {best['q1_winrate']:.1f}% win rate")
        print(f"  Hot vs Cold spread: {best['spread']:.3f}%")

# Summary
print("\n" + "=" * 80)
print("SUMMARY - OPTIMAL LOOKBACK WINDOWS")
print("=" * 80)

for pair in PAIRS:
    trades_df = pair_predictions[pair]['trades'].copy()

    best_window = None
    best_spread = -999

    for window in lookback_windows:
        trades_df[f'recent_acc_{window}'] = 0.0

        for i in range(len(trades_df)):
            if i >= window:
                recent_trades = trades_df.iloc[i-window:i]
                recent_wins = (recent_trades['net_return_pct'] > 0).sum()
                trades_df.at[trades_df.index[i], f'recent_acc_{window}'] = recent_wins / window

        valid_trades = trades_df[trades_df[f'recent_acc_{window}'] > 0].copy()

        if len(valid_trades) < 100:
            continue

        try:
            valid_trades['quartile'] = pd.qcut(valid_trades[f'recent_acc_{window}'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        except ValueError:
            # Not enough unique values for quartiles, skip this window
            continue

        q4 = valid_trades[valid_trades['quartile'] == 'Q4']
        q1 = valid_trades[valid_trades['quartile'] == 'Q1']

        if len(q4) > 0 and len(q1) > 0:
            spread = q4['net_return_pct'].mean() - q1['net_return_pct'].mean()
            if spread > best_spread:
                best_spread = spread
                best_window = window

    if best_window:
        print(f"{pair:<10} Best window: {best_window:>2} trades  (Hot vs Cold spread: {best_spread:>6.3f}%)")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
Use these optimal windows to calculate "recent accuracy" for each pair.
Then trade the pair(s) with highest recent accuracy score.

Next step: Backtest the "Hot Hand" diversification strategy!
""")
