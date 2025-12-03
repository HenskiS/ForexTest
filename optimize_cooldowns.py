"""
Optimize cooldown periods after wins and losses.

Tests all combinations of:
- Profit cooldown: 0-10 days
- Loss cooldown: 0-10 days

Using the original strategy with optimized parameters.
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

print(f"Cooldown Optimization - {PAIR}")
print("="*80)
print(f"Testing profit cooldowns: 0-10 days")
print(f"Testing loss cooldowns: 0-10 days")
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

# Backtest function with cooldowns
def backtest_with_cooldowns(predictions, df_prices, test_indices,
                            profit_cooldown=0, loss_cooldown=0,
                            lower_pct=48, upper_pct=52,
                            base_stop_loss_pct=0.0018,
                            base_take_profit_pct=0.0200,
                            transaction_cost_pct=0.0002,
                            holding_period=1):

    prediction_buffer = list(predictions[:50])
    position = 0
    entry_price = 0.0
    holding_days = 0
    cooldown_remaining = 0
    equity = [1000]
    trades_count = 0

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values

    for i in range(len(predictions)):
        if i >= 50:
            prediction_buffer.append(predictions[i])
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        if len(prediction_buffer) >= 50:
            buffer_array = np.array(prediction_buffer)
            lower_threshold = np.percentile(buffer_array, lower_pct)
            upper_threshold = np.percentile(buffer_array, upper_pct)
            signal = 1 if predictions[i] >= upper_threshold else (-1 if predictions[i] <= lower_threshold else 0)
        else:
            signal = 0

        # Decrease cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if position != 0:
            holding_days += 1

            if position == 1:
                pct_high = (highs[i] - entry_price) / entry_price
                pct_low = (lows[i] - entry_price) / entry_price
            else:
                pct_high = (entry_price - lows[i]) / entry_price
                pct_low = (entry_price - highs[i]) / entry_price

            exit_price = None
            if pct_low <= -base_stop_loss_pct:
                exit_price = entry_price * (1 - base_stop_loss_pct) if position == 1 else entry_price * (1 + base_stop_loss_pct)
            elif pct_high >= base_take_profit_pct:
                exit_price = entry_price * (1 + base_take_profit_pct) if position == 1 else entry_price * (1 - base_take_profit_pct)
            elif holding_days >= holding_period:
                exit_price = closes[i]

            if exit_price:
                raw_return_pct = ((exit_price - entry_price) / entry_price if position == 1
                                  else (entry_price - exit_price) / entry_price) * 100
                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                equity.append(equity[-1] * (1 + net_return_pct / 100))
                trades_count += 1

                # Set cooldown based on outcome
                if net_return_pct > 0:
                    cooldown_remaining = profit_cooldown
                else:
                    cooldown_remaining = loss_cooldown

                position = 0
                holding_days = 0

        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = opens[i]
            holding_days = 0

    return equity[-1], trades_count

# Test all cooldown combinations
print("\nTesting cooldown combinations...")
spread_cost = args.spread_pct / 100.0
cost = 0.0002 + spread_cost

results = []

# Test combinations
cooldown_values = range(0, 11)  # 0 to 10 days
total_tests = len(cooldown_values) ** 2

with tqdm(total=total_tests, desc="Testing cooldowns") as pbar:
    for profit_cooldown in cooldown_values:
        for loss_cooldown in cooldown_values:
            equity, trades = backtest_with_cooldowns(
                predictions, df, test_indices,
                profit_cooldown=profit_cooldown,
                loss_cooldown=loss_cooldown,
                transaction_cost_pct=cost
            )

            results.append({
                'profit_cooldown': profit_cooldown,
                'loss_cooldown': loss_cooldown,
                'final_equity': equity,
                'total_trades': trades,
                'return_pct': (equity - 1000) / 10
            })
            pbar.update(1)

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('final_equity', ascending=False)

# Display top 20 results
years = len(predictions) / 252

print("\n" + "="*80)
print("TOP 20 COOLDOWN COMBINATIONS")
print("="*80)
print(f"{'Rank':<6} {'Profit CD':<12} {'Loss CD':<12} {'Equity':<15} {'Return':<12} {'Trades':<10}")
print("-" * 80)

for idx, row in results_df.head(20).iterrows():
    annual_return = ((row['final_equity'] / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0
    print(f"{results_df.index.get_loc(idx)+1:<6} "
          f"{int(row['profit_cooldown']):<12} "
          f"{int(row['loss_cooldown']):<12} "
          f"${row['final_equity']:>10,.2f}    "
          f"{row['return_pct']:>7.2f}%    "
          f"{int(row['total_trades']):>5}")

# Save full results
output_file = f'{PAIR}_cooldown_optimization_{TEST_DAYS}days.csv'
results_df.to_csv(output_file, index=False)
print(f"\nFull results saved to: {output_file}")

# Show heatmap summary
print("\n" + "="*80)
print("HEATMAP SUMMARY (Return %)")
print("="*80)
pivot = results_df.pivot(index='loss_cooldown', columns='profit_cooldown', values='return_pct')
print(pivot.to_string(float_format=lambda x: f'{x:6.2f}'))

print("\n" + "="*80)
best = results_df.iloc[0]
print(f"BEST COMBINATION:")
print(f"  Profit Cooldown: {int(best['profit_cooldown'])} days")
print(f"  Loss Cooldown: {int(best['loss_cooldown'])} days")
print(f"  Final Equity: ${best['final_equity']:,.2f}")
print(f"  Total Return: {best['return_pct']:.2f}%")
print(f"  Total Trades: {int(best['total_trades'])}")
print("="*80)
