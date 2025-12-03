"""
Compare multiple trading strategies side-by-side.

Tests:
1. Original strategy (stop loss, take profit, fixed holding period)
2. Follow model strategy (stay as long as model predicts same direction)
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm
import os
import sys

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=250, help='Number of recent days to backtest')
parser.add_argument('--spread-pct', type=float, default=0.0, help='Spread cost per trade as percentage')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = args.test_days

print(f"Strategy Comparison - {PAIR}")
print("="*80)
print(f"Test period: Last {TEST_DAYS} days")
print("="*80)

# Load and prepare data
data_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(data_file):
    print(f"\nERROR: {data_file} not found")
    sys.exit(1)

df_raw = pd.read_csv(data_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.set_index('date')
print(f"\nLoaded {len(df_raw)} days of raw OANDA data")

# Calculate features
def calculate_features(df):
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

    return df

df = calculate_features(df_raw.copy())
df[TARGET] = df['close'].pct_change(1).shift(-1)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

df = df.dropna(subset=technical_features + [TARGET])

# Load hyperparameters
hyperparam_file = f'hyperparams_rolling_daily_{PAIR}.pkl'
if os.path.exists(hyperparam_file):
    with open(hyperparam_file, 'rb') as f:
        best_params = pickle.load(f)
else:
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
predictions = []
test_indices = []
start_idx = len(df) - TEST_DAYS

for i in tqdm(range(start_idx, len(df)), desc="Predictions"):
    if i < TRAIN_WINDOW_SIZE:
        continue

    train_end_idx = i - 1
    train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
    train_data = df.iloc[train_start_idx:train_end_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[technical_features])
    y_train = train_data[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        gamma=best_params['gamma'],
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=False)

    X_today = scaler.transform(df.iloc[[i]][technical_features])
    y_pred = model.predict(X_today)[0]

    predictions.append(y_pred)
    test_indices.append(i)

predictions = np.array(predictions)
test_data = df.iloc[test_indices]

print(f"\nGenerated {len(predictions)} predictions")

# Strategy 1: Original (stop loss, take profit, fixed holding)
def backtest_original(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0018,
                      base_take_profit_pct=0.0200,
                      transaction_cost_pct=0.0002,
                      holding_period=1,
                      buffer_warmup=50):

    prediction_buffer = list(predictions[:buffer_warmup])
    position = 0
    entry_price = 0.0
    holding_days = 0
    equity = [1000]
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values

    for i in range(len(predictions)):
        prediction = predictions[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]

        if i >= buffer_warmup:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        if len(prediction_buffer) >= buffer_warmup:
            buffer_array = np.array(prediction_buffer)
            lower_threshold = np.percentile(buffer_array, lower_pct)
            upper_threshold = np.percentile(buffer_array, upper_pct)

            if prediction >= upper_threshold:
                signal = 1
            elif prediction <= lower_threshold:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0

        if position != 0:
            holding_days += 1

            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            exit_triggered = False
            exit_price = None

            if pct_low <= -base_stop_loss_pct:
                exit_triggered = True
                exit_price = entry_price * (1 - base_stop_loss_pct) if position == 1 else entry_price * (1 + base_stop_loss_pct)
            elif pct_high >= base_take_profit_pct:
                exit_triggered = True
                exit_price = entry_price * (1 + base_take_profit_pct) if position == 1 else entry_price * (1 - base_take_profit_pct)
            elif holding_days >= holding_period:
                exit_triggered = True
                exit_price = close_price

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)

                current_equity = equity[-1]
                new_equity = current_equity * (1 + net_return_pct / 100)
                equity.append(new_equity)

                trades.append(net_return_pct)

                position = 0
                holding_days = 0

        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    return equity[-1], len(trades)

# Strategy 2: Follow model
def backtest_follow_model(predictions, df_prices, test_indices,
                          lower_pct=48, upper_pct=52,
                          transaction_cost_pct=0.0002,
                          buffer_warmup=50):

    prediction_buffer = list(predictions[:buffer_warmup])
    position = 0
    entry_price = 0.0
    equity = [1000]
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    closes = test_data['close'].values

    for i in range(len(predictions)):
        prediction = predictions[i]
        open_price = opens[i]
        close_price = closes[i]

        if i >= buffer_warmup:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        if len(prediction_buffer) >= buffer_warmup:
            buffer_array = np.array(prediction_buffer)
            lower_threshold = np.percentile(buffer_array, lower_pct)
            upper_threshold = np.percentile(buffer_array, upper_pct)

            if prediction >= upper_threshold:
                signal = 1
            elif prediction <= lower_threshold:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0

        # Exit if signal changes
        if position != 0 and signal != position:
            exit_price = open_price

            if position == 1:
                raw_return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                raw_return_pct = (entry_price - exit_price) / entry_price * 100

            net_return_pct = raw_return_pct - (transaction_cost_pct * 100)

            current_equity = equity[-1]
            new_equity = current_equity * (1 + net_return_pct / 100)
            equity.append(new_equity)

            trades.append(net_return_pct)

            position = 0

        # Enter if no position and signal present
        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price

    # Close any open position
    if position != 0:
        exit_price = closes[-1]
        if position == 1:
            raw_return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            raw_return_pct = (entry_price - exit_price) / entry_price * 100

        net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
        current_equity = equity[-1]
        new_equity = current_equity * (1 + net_return_pct / 100)
        equity.append(new_equity)
        trades.append(net_return_pct)

    return equity[-1], len(trades)

# Run both strategies
spread_cost = args.spread_pct / 100.0

print("\n" + "="*80)
print("Running Strategy 1: ORIGINAL (stop loss, take profit, 1-day hold)")
print("="*80)
equity1, trades1 = backtest_original(predictions, df, test_indices,
                                     transaction_cost_pct=0.0002 + spread_cost)

print("\n" + "="*80)
print("Running Strategy 2: FOLLOW MODEL (stay while model predicts)")
print("="*80)
equity2, trades2 = backtest_follow_model(predictions, df, test_indices,
                                         transaction_cost_pct=0.0002 + spread_cost)

# Display comparison
years = len(predictions) / 252

print("\n" + "="*80)
print("STRATEGY COMPARISON")
print("="*80)
print(f"\n{'Metric':<30} {'Original':<20} {'Follow Model':<20}")
print("-" * 70)
print(f"{'Final Equity':<30} ${equity1:>15,.2f}    ${equity2:>15,.2f}")
print(f"{'Total Return':<30} {(equity1-1000)/10:>15.2f}%    {(equity2-1000)/10:>15.2f}%")

annual1 = ((equity1 / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0
annual2 = ((equity2 / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0
print(f"{'Annual Return':<30} {annual1:>15.2f}%    {annual2:>15.2f}%")

print(f"{'Total Trades':<30} {trades1:>15}      {trades2:>15}")
print()

# Highlight winner
if equity2 > equity1:
    improvement = ((equity2 - equity1) / equity1) * 100
    print(f"WINNER: Follow Model strategy (+{improvement:.1f}% better)")
elif equity1 > equity2:
    improvement = ((equity1 - equity2) / equity2) * 100
    print(f"WINNER: Original strategy (+{improvement:.1f}% better)")
else:
    print("RESULT: Both strategies performed equally")

print("="*80)
