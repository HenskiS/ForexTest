"""
Test the model's raw predictive accuracy.

Strategy:
- Enter when model predicts direction (48/52 percentile thresholds)
- Exit after exactly 1 day (at close)
- NO stop loss
- NO take profit
- Pure model performance test
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

parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=250, help='Number of recent days to backtest')
parser.add_argument('--spread-pct', type=float, default=0.0, help='Spread cost per trade as percentage')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = args.test_days

print(f"Raw Model Performance Test - {PAIR}")
print("="*80)
print(f"Strategy: 1-day time exit ONLY (no stops)")
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

    model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, verbose=False)

    X_today = scaler.transform(df.iloc[[i]][technical_features])
    y_pred = model.predict(X_today)[0]

    predictions.append(y_pred)
    test_indices.append(i)

predictions = np.array(predictions)
test_data = df.iloc[test_indices]

# Backtest - TIME EXIT ONLY
print("\nRunning backtest...")

transaction_cost_pct = 0.0002 + (args.spread_pct / 100.0)

prediction_buffer = list(predictions[:50])
position = 0
entry_price = 0.0
holding_days = 0
equity = [1000]
trades = []

opens = test_data['open'].values
closes = test_data['close'].values

for i in range(len(predictions)):
    if i >= 50:
        prediction_buffer.append(predictions[i])
        if len(prediction_buffer) > 200:
            prediction_buffer = prediction_buffer[-200:]

    if len(prediction_buffer) >= 50:
        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, 48)
        upper_threshold = np.percentile(buffer_array, 52)
        signal = 1 if predictions[i] >= upper_threshold else (-1 if predictions[i] <= lower_threshold else 0)
    else:
        signal = 0

    if position != 0:
        holding_days += 1

        # TIME EXIT ONLY - no stops
        if holding_days >= 1:
            exit_price = closes[i]

            raw_return_pct = ((exit_price - entry_price) / entry_price if position == 1
                              else (entry_price - exit_price) / entry_price) * 100
            net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
            outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

            equity.append(equity[-1] * (1 + net_return_pct / 100))

            trades.append({
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'raw_return_pct': raw_return_pct,
                'net_return_pct': net_return_pct,
                'outcome': outcome
            })

            position = 0
            holding_days = 0

    if position == 0 and signal != 0:
        position = signal
        entry_price = opens[i]
        holding_days = 0

# Analysis
trades_df = pd.DataFrame(trades)

print("\n" + "="*80)
print("RAW MODEL PERFORMANCE (Time Exit Only)")
print("="*80)

total_trades = len(trades_df)
wins = trades_df[trades_df['outcome'] == 'WIN']
losses = trades_df[trades_df['outcome'] == 'LOSS']
win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

final_equity = equity[-1]
total_return = (final_equity - 1000) / 10
years = len(predictions) / 252
annual_return = ((final_equity / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0

print(f"\nTotal Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Wins: {len(wins)}")
print(f"Losses: {len(losses)}")
print()
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annual Return: {annual_return:.2f}%")

if len(wins) > 0 and len(losses) > 0:
    print()
    print(f"Average Win: {wins['net_return_pct'].mean():.2f}%")
    print(f"Average Loss: {losses['net_return_pct'].mean():.2f}%")
    print(f"Largest Win: {wins['net_return_pct'].max():.2f}%")
    print(f"Largest Loss: {losses['net_return_pct'].min():.2f}%")

    profit_factor = abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum())
    print(f"Profit Factor: {profit_factor:.2f}")

    # Expectancy
    avg_win = wins['net_return_pct'].mean()
    avg_loss = losses['net_return_pct'].mean()
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
    print(f"Expectancy: {expectancy:.2f}% per trade")

# Compare with strategy that has stops
print("\n" + "="*80)
print("COMPARISON WITH STOPS (0.18% SL / 3.0% TP)")
print("="*80)
print(f"Time Exit Only:    {total_return:.2f}% return, {win_rate:.1f}% win rate")
print(f"With Stops:        50.63% return, 43.9% win rate")
print()

if total_return > 50.63:
    diff = total_return - 50.63
    print(f"Time Exit Only is BETTER by {diff:.2f} percentage points")
    print("Finding: Stops are HURTING performance!")
elif total_return < 50.63:
    diff = 50.63 - total_return
    print(f"With Stops is BETTER by {diff:.2f} percentage points")
    print("Finding: Stops are HELPING performance by preventing worse losses")
else:
    print("Both strategies perform equally")

# Return distribution
print("\n" + "="*80)
print("RETURN DISTRIBUTION")
print("="*80)

bins = [
    ('-3% to -2%', -3, -2),
    ('-2% to -1%', -2, -1),
    ('-1% to 0%', -1, 0),
    ('0% to 1%', 0, 1),
    ('1% to 2%', 1, 2),
    ('2% to 3%', 2, 3),
    ('3%+', 3, 999)
]

for label, low, high in bins:
    count = len(trades_df[(trades_df['net_return_pct'] > low) & (trades_df['net_return_pct'] <= high)])
    if count > 0:
        pct = count / total_trades * 100
        print(f"{label:<15} {count:>3} trades ({pct:>5.1f}%)")

print("="*80)
