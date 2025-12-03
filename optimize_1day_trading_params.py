"""
Optimize trading parameters for 1-day prediction model.

Tests various combinations of:
- Holding periods (1-5 days)
- Stop loss percentages
- Take profit percentages

To find the best configuration for the 1-day forward return predictions.
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
from itertools import product

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=250, help='Number of recent days to backtest')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378  # Optimized for 1-day predictions
TEST_DAYS = args.test_days

print(f"Optimizing Trading Parameters for 1-Day Model - {PAIR}")
print("="*80)
print(f"Test period: Last {TEST_DAYS} days")
print("="*80)

# Load raw OANDA data and calculate features
data_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(data_file):
    print(f"\nERROR: {data_file} not found")
    print("Run: python oanda_data_fetcher.py")
    sys.exit(1)

df_raw = pd.read_csv(data_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.set_index('date')
print(f"\nLoaded {len(df_raw)} days of raw OANDA data")

# Calculate features (same as backtest script)
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

    return df

print("\nCalculating technical features...")
df = calculate_features(df_raw.copy())

# Calculate target
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
print(f"Clean data: {len(df)} days")

# Load hyperparameters
hyperparam_file = f'hyperparams_rolling_daily_{PAIR}.pkl'
if os.path.exists(hyperparam_file):
    with open(hyperparam_file, 'rb') as f:
        best_params = pickle.load(f)
    print(f"\nUsing optimized hyperparameters from {hyperparam_file}")
else:
    best_params = {
        'n_estimators': 125,
        'learning_rate': 0.1,
        'max_depth': 5,
        'gamma': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.7
    }
    print(f"\nUsing default hyperparameters")

# Generate predictions for test period
print(f"\nGenerating predictions for last {TEST_DAYS} days...")

if len(df) < TRAIN_WINDOW_SIZE + TEST_DAYS:
    print(f"ERROR: Need at least {TRAIN_WINDOW_SIZE + TEST_DAYS} days")
    sys.exit(1)

predictions = []
actuals = []
test_dates = []
test_indices = []

start_idx = len(df) - TEST_DAYS

for i in tqdm(range(start_idx, len(df)), desc="Generating predictions"):
    current_date = df.index[i]

    if i < TRAIN_WINDOW_SIZE:
        continue

    # Rolling window with 1-day gap
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
    y_actual = df.iloc[i][TARGET]

    predictions.append(y_pred)
    actuals.append(y_actual)
    test_dates.append(current_date)
    test_indices.append(i)

predictions = np.array(predictions)
actuals = np.array(actuals)
test_dates = pd.DatetimeIndex(test_dates)

print(f"\nGenerated {len(predictions)} predictions")

# Define parameter grid to search - fine-tuned around optimal zone
param_grid = {
    'holding_period': [1],  # Already confirmed optimal
    'base_stop_loss_pct': [0.0015, 0.0018, 0.0020, 0.0022, 0.0025],  # 0.15%, 0.18%, 0.20%, 0.22%, 0.25%
    'base_take_profit_pct': [0.0125, 0.0150, 0.0175, 0.0200, 0.0225]  # 1.25%, 1.50%, 1.75%, 2.00%, 2.25%
}

print("\n" + "="*80)
print("PARAMETER OPTIMIZATION")
print("="*80)
print(f"Testing {len(param_grid['holding_period']) * len(param_grid['base_stop_loss_pct']) * len(param_grid['base_take_profit_pct'])} combinations...")
print()

def backtest_strategy(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0040,
                      base_take_profit_pct=0.0100,
                      loss_cooldown_days=1,
                      transaction_cost_pct=0.0002,
                      holding_period=5,
                      buffer_warmup=50):
    """Backtest with given parameters"""

    prediction_buffer = list(predictions[:buffer_warmup])

    position = 0
    entry_price = 0.0
    entry_date = None
    cooldown_remaining = 0
    holding_days = 0

    equity = [1000]
    trades = []

    test_data = df_prices.iloc[test_indices]
    test_dates_array = test_data.index
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

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Fixed stops (no ATR adjustment)
        stop_loss_pct = base_stop_loss_pct
        take_profit_pct = base_take_profit_pct

        # Check for exit
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

            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)
            elif holding_days >= holding_period:
                exit_triggered = True
                exit_price = close_price

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                current_equity = equity[-1]
                new_equity = current_equity * (1 + net_return_pct / 100)
                equity.append(new_equity)

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': test_dates_array[i],
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_return_pct': net_return_pct,
                    'outcome': outcome
                })

                if net_return_pct < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0
                entry_date = None
                holding_days = 0

        # Check for entry
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            entry_date = test_dates_array[i]
            holding_days = 0

    final_equity = equity[-1]
    total_return_pct = (final_equity - 1000) / 1000 * 100

    trades_df = pd.DataFrame(trades)
    win_rate = len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    # Calculate Sharpe ratio (approximate)
    if len(equity) > 1:
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Calculate max drawdown
    equity_array = np.array(equity)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100

    return {
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': trades_df
    }

# Run grid search
results = []

combinations = list(product(
    param_grid['holding_period'],
    param_grid['base_stop_loss_pct'],
    param_grid['base_take_profit_pct']
))

for holding_period, stop_loss, take_profit in tqdm(combinations, desc="Testing combinations"):
    result = backtest_strategy(
        predictions, df, test_indices,
        lower_pct=48, upper_pct=52,
        base_stop_loss_pct=stop_loss,
        base_take_profit_pct=take_profit,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002,
        holding_period=holding_period,
        buffer_warmup=50
    )

    results.append({
        'holding_period': holding_period,
        'stop_loss_pct': stop_loss * 100,
        'take_profit_pct': take_profit * 100,
        'total_return': result['total_return_pct'],
        'total_trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe'],
        'max_drawdown': result['max_drawdown'],
        'final_equity': result['final_equity']
    })

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return', ascending=False)

# Display top 20 results
print("\n" + "="*80)
print(f"TOP 20 PARAMETER COMBINATIONS - {PAIR}")
print("="*80)
print(results_df.head(20).to_string(index=False))

# Display best by different metrics
print("\n" + "="*80)
print("BEST BY DIFFERENT METRICS")
print("="*80)

best_return = results_df.iloc[0]
print(f"\nBest Total Return: {best_return['total_return']:.2f}%")
print(f"  Holding Period: {best_return['holding_period']} days")
print(f"  Stop Loss: {best_return['stop_loss_pct']:.2f}%")
print(f"  Take Profit: {best_return['take_profit_pct']:.2f}%")
print(f"  Win Rate: {best_return['win_rate']:.1f}%")
print(f"  Trades: {best_return['total_trades']}")
print(f"  Sharpe: {best_return['sharpe']:.2f}")
print(f"  Max Drawdown: {best_return['max_drawdown']:.2f}%")

best_sharpe = results_df.sort_values('sharpe', ascending=False).iloc[0]
print(f"\nBest Sharpe Ratio: {best_sharpe['sharpe']:.2f}")
print(f"  Holding Period: {best_sharpe['holding_period']} days")
print(f"  Stop Loss: {best_sharpe['stop_loss_pct']:.2f}%")
print(f"  Take Profit: {best_sharpe['take_profit_pct']:.2f}%")
print(f"  Total Return: {best_sharpe['total_return']:.2f}%")
print(f"  Win Rate: {best_sharpe['win_rate']:.1f}%")

best_winrate = results_df.sort_values('win_rate', ascending=False).iloc[0]
print(f"\nBest Win Rate: {best_winrate['win_rate']:.1f}%")
print(f"  Holding Period: {best_winrate['holding_period']} days")
print(f"  Stop Loss: {best_winrate['stop_loss_pct']:.2f}%")
print(f"  Take Profit: {best_winrate['take_profit_pct']:.2f}%")
print(f"  Total Return: {best_winrate['total_return']:.2f}%")
print(f"  Sharpe: {best_winrate['sharpe']:.2f}")

# Save results
output_file = f'{PAIR}_1day_param_optimization_{TEST_DAYS}days.csv'
results_df.to_csv(output_file, index=False)
print(f"\n" + "="*80)
print(f"All results saved to: {output_file}")
print("="*80)
