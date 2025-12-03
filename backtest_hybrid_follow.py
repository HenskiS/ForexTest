"""
Detailed backtest of the Hybrid Follow strategy:
- Enter when model predicts direction
- Stay as long as model continues predicting same direction
- Exit when: (1) Model changes, (2) Stop loss hit, or (3) Take profit hit
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
parser.add_argument('--lower-pct', type=float, default=48, help='Lower percentile threshold')
parser.add_argument('--upper-pct', type=float, default=52, help='Upper percentile threshold')
parser.add_argument('--stop-loss', type=float, default=0.18, help='Stop loss percentage')
parser.add_argument('--take-profit', type=float, default=2.0, help='Take profit percentage')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = args.test_days

print(f"Hybrid Follow Strategy Backtest - {PAIR}")
print("="*80)
print(f"Strategy: Follow model + SL/TP protection")
print(f"Thresholds: {args.lower_pct}th / {args.upper_pct}th percentile")
print(f"Stop Loss: {args.stop_loss}%")
print(f"Take Profit: {args.take_profit}%")
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

# Backtest
print("\nRunning backtest...")

stop_loss_pct = args.stop_loss / 100.0
take_profit_pct = args.take_profit / 100.0
transaction_cost_pct = 0.0002 + (args.spread_pct / 100.0)

prediction_buffer = list(predictions[:50])
position = 0
entry_price = 0.0
entry_date = None
holding_days = 0
equity = [1000]
trades = []

opens = test_data['open'].values
highs = test_data['high'].values
lows = test_data['low'].values
closes = test_data['close'].values
dates = test_data.index

for i in range(len(predictions)):
    if i >= 50:
        prediction_buffer.append(predictions[i])
        if len(prediction_buffer) > 200:
            prediction_buffer = prediction_buffer[-200:]

    if len(prediction_buffer) >= 50:
        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, args.lower_pct)
        upper_threshold = np.percentile(buffer_array, args.upper_pct)
        signal = 1 if predictions[i] >= upper_threshold else (-1 if predictions[i] <= lower_threshold else 0)
    else:
        signal = 0

    if position != 0:
        holding_days += 1

        # Check SL and TP
        if position == 1:
            pct_high = (highs[i] - entry_price) / entry_price
            pct_low = (lows[i] - entry_price) / entry_price
        else:
            pct_high = (entry_price - lows[i]) / entry_price
            pct_low = (entry_price - highs[i]) / entry_price

        exit_price = None
        exit_reason = None

        if pct_low <= -stop_loss_pct:
            exit_price = entry_price * (1 - stop_loss_pct) if position == 1 else entry_price * (1 + stop_loss_pct)
            exit_reason = 'STOP_LOSS'
        elif pct_high >= take_profit_pct:
            exit_price = entry_price * (1 + take_profit_pct) if position == 1 else entry_price * (1 - take_profit_pct)
            exit_reason = 'TAKE_PROFIT'
        elif signal != position:
            exit_price = opens[i]
            exit_reason = 'MODEL_CHANGE'

        if exit_price:
            raw_return_pct = ((exit_price - entry_price) / entry_price if position == 1
                              else (entry_price - exit_price) / entry_price) * 100
            net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
            outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

            equity.append(equity[-1] * (1 + net_return_pct / 100))

            trades.append({
                'entry_date': entry_date,
                'exit_date': dates[i],
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'net_return_pct': net_return_pct,
                'outcome': outcome,
                'exit_reason': exit_reason,
                'holding_days': holding_days
            })

            position = 0
            holding_days = 0

    if position == 0 and signal != 0:
        position = signal
        entry_price = opens[i]
        entry_date = dates[i]
        holding_days = 0

# Close any open position
if position != 0:
    raw_return_pct = ((closes[-1] - entry_price) / entry_price if position == 1
                      else (entry_price - closes[-1]) / entry_price) * 100
    net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
    outcome = 'WIN' if net_return_pct > 0 else 'LOSS'
    equity.append(equity[-1] * (1 + net_return_pct / 100))

    trades.append({
        'entry_date': entry_date,
        'exit_date': dates[-1],
        'direction': 'LONG' if position == 1 else 'SHORT',
        'entry_price': entry_price,
        'exit_price': closes[-1],
        'net_return_pct': net_return_pct,
        'outcome': outcome,
        'exit_reason': 'END_OF_TEST',
        'holding_days': holding_days
    })

# Results
final_equity = equity[-1]
total_return_pct = (final_equity - 1000) / 1000 * 100
trades_df = pd.DataFrame(trades)

years = len(predictions) / 252
annual_return_pct = ((final_equity / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return_pct:.2f}%")
print(f"Annual Return: {annual_return_pct:.2f}%")
print(f"Total Trades: {len(trades_df)}")

if len(trades_df) > 0:
    wins = trades_df[trades_df['outcome'] == 'WIN']
    losses = trades_df[trades_df['outcome'] == 'LOSS']
    win_rate = len(wins) / len(trades_df) * 100

    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: {wins['net_return_pct'].mean():.2f}%")
    print(f"Avg Loss: {losses['net_return_pct'].mean():.2f}%")
    print(f"Avg Holding: {trades_df['holding_days'].mean():.1f} days")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum())
        print(f"Profit Factor: {profit_factor:.2f}")

    # Exit reason breakdown
    print("\nExit Reason Breakdown:")
    for reason in ['STOP_LOSS', 'TAKE_PROFIT', 'MODEL_CHANGE', 'END_OF_TEST']:
        count = len(trades_df[trades_df['exit_reason'] == reason])
        if count > 0:
            pct = count / len(trades_df) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

    # Save trades
    output_file = f'{PAIR}_hybrid_follow_trades_{TEST_DAYS}days.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"\nTrades saved to: {output_file}")

print("="*80)
