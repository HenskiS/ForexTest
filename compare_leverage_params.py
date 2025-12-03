"""
Head-to-head comparison of two parameter sets with different leverage levels.
"""
import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tqdm import tqdm
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair')
parser.add_argument('--test-days', type=int, default=4500, help='Number of recent days to backtest')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = args.test_days

print(f"\nLeverage & Parameter Comparison - {PAIR}")
print("="*80)
print(f"Test period: Last {TEST_DAYS} days")
print("="*80)

# Load raw OANDA data
data_file = f'data/{PAIR}_1day_oanda.csv'
df_raw = pd.read_csv(data_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw.set_index('date', inplace=True)

print(f"\nLoaded {len(df_raw)} days of raw OANDA data")
print(f"Date range: {df_raw.index[0]} to {df_raw.index[-1]}")

# Calculate technical features
def calculate_features(df):
    df['momentum'] = df['close'].pct_change(5)
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['range'] = df['high'] - df['low']
    df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    for period in [10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    atr = df['range'].rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=20).mean()
    mad = (tp - sma).abs().rolling(window=20).mean()
    df['cci'] = (tp - sma) / (0.015 * mad)

    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
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

# Generate predictions
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

    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, verbose=False)

    X_test = scaler.transform(df.iloc[i:i+1][technical_features])
    y_pred = model.predict(X_test)[0]
    y_actual = df.iloc[i][TARGET]

    predictions.append(y_pred)
    actuals.append(y_actual)
    test_dates.append(current_date)
    test_indices.append(i)

predictions = np.array(predictions)
actuals = np.array(actuals)
test_dates = pd.DatetimeIndex(test_dates)

print(f"\nGenerated {len(predictions)} predictions")

# Define parameter sets to compare
param_sets = [
    {
        'name': 'Current (0.18% / 2.00%)',
        'stop_loss_pct': 0.0018,
        'take_profit_pct': 0.0200
    },
    {
        'name': 'Optimal (0.20% / 2.25%)',
        'stop_loss_pct': 0.0020,
        'take_profit_pct': 0.0225
    }
]

leverage_levels = [1, 4]

# Backtest strategy
def backtest_strategy(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0018,
                      base_take_profit_pct=0.0200,
                      holding_period=1,
                      leverage=1.0):
    """Backtest with leverage applied to returns"""

    lower_threshold = np.percentile(predictions, lower_pct)
    upper_threshold = np.percentile(predictions, upper_pct)

    position = 0
    entry_price = 0
    entry_date = None
    holding_days = 0

    equity = 1000  # Starting equity
    equity_curve = [equity]
    equity_dates = [df_prices.index[test_indices[0]]]

    trades = []
    signals_list = []

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

        if i < 30:
            signal = 0
        else:
            recent_preds = predictions[max(0, i-30):i]
            lower_threshold = np.percentile(recent_preds, lower_pct)
            upper_threshold = np.percentile(recent_preds, upper_pct)

            if prediction >= upper_threshold:
                signal = 1
            elif prediction <= lower_threshold:
                signal = -1
            else:
                signal = 0

        signals_list.append(signal)

        stop_loss_pct = base_stop_loss_pct
        take_profit_pct = base_take_profit_pct

        # Check for exit
        if position != 0:
            holding_days += 1

            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price

                if pct_high >= take_profit_pct:
                    exit_price = entry_price * (1 + take_profit_pct)
                    exit_return = take_profit_pct
                    outcome = 'WIN'
                    exit_reason = 'TAKE_PROFIT'
                elif pct_low <= -stop_loss_pct:
                    exit_price = entry_price * (1 - stop_loss_pct)
                    exit_return = -stop_loss_pct
                    outcome = 'LOSS'
                    exit_reason = 'STOP_LOSS'
                elif holding_days >= holding_period:
                    exit_price = close_price
                    exit_return = (exit_price - entry_price) / entry_price
                    outcome = 'WIN' if exit_return > 0 else 'LOSS'
                    exit_reason = 'TIME_EXIT'
                else:
                    continue
            else:  # Short
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price

                if pct_low <= -take_profit_pct:
                    exit_price = entry_price * (1 - take_profit_pct)
                    exit_return = take_profit_pct
                    outcome = 'WIN'
                    exit_reason = 'TAKE_PROFIT'
                elif pct_high >= stop_loss_pct:
                    exit_price = entry_price * (1 + stop_loss_pct)
                    exit_return = -stop_loss_pct
                    outcome = 'LOSS'
                    exit_reason = 'STOP_LOSS'
                elif holding_days >= holding_period:
                    exit_price = close_price
                    exit_return = (entry_price - exit_price) / entry_price
                    outcome = 'WIN' if exit_return > 0 else 'LOSS'
                    exit_reason = 'TIME_EXIT'
                else:
                    continue

            # Apply leverage to returns (not to position size)
            leveraged_return = exit_return * leverage

            # Update equity with leveraged return
            new_equity = equity * (1 + leveraged_return)

            trades.append({
                'entry_date': entry_date,
                'exit_date': test_dates_array[i],
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'base_return_pct': exit_return * 100,
                'leveraged_return_pct': leveraged_return * 100,
                'net_return_pct': leveraged_return * 100,
                'holding_days': holding_days,
                'outcome': outcome,
                'exit_reason': exit_reason
            })

            equity = new_equity
            equity_curve.append(equity)
            equity_dates.append(test_dates_array[i])

            position = 0
            holding_days = 0

        # Entry signal
        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price
            entry_date = test_dates_array[i]
            holding_days = 0

    # Close any open position at end
    if position != 0:
        exit_price = closes[-1]
        if position == 1:
            exit_return = (exit_price - entry_price) / entry_price
        else:
            exit_return = (entry_price - exit_price) / entry_price

        leveraged_return = exit_return * leverage

        outcome = 'WIN' if exit_return > 0 else 'LOSS'

        trades.append({
            'entry_date': entry_date,
            'exit_date': test_dates_array[-1],
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'base_return_pct': exit_return * 100,
            'leveraged_return_pct': leveraged_return * 100,
            'net_return_pct': leveraged_return * 100,
            'holding_days': holding_days + 1,
            'outcome': outcome,
            'exit_reason': 'END_OF_TEST'
        })

        equity = equity * (1 + leveraged_return)
        equity_curve.append(equity)
        equity_dates.append(test_dates_array[-1])

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {
            'total_return': 0,
            'win_rate': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'total_trades': 0
        }

    # Calculate metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100

    wins = trades_df[trades_df['outcome'] == 'WIN']
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    returns = trades_df['net_return_pct'].values
    avg_return = returns.mean()
    std_return = returns.std()

    trades_in_period = len(trades_df)
    if std_return > 0 and trades_in_period > 1:
        sharpe_ratio = (avg_return * trades_in_period) / (std_return * np.sqrt(trades_in_period))
    else:
        sharpe_ratio = 0

    # Max drawdown
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - peak) / peak * 100
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades_df),
        'final_equity': equity_curve[-1],
        'trades': trades_df,
        'equity_curve': equity_curve,
        'equity_dates': equity_dates
    }

# Run comparisons
print("\n" + "="*80)
print("HEAD-TO-HEAD COMPARISON")
print("="*80)

results = []

for param_set in param_sets:
    for leverage in leverage_levels:
        print(f"\nTesting {param_set['name']} with {leverage}x leverage...")

        result = backtest_strategy(
            predictions, df, test_indices,
            base_stop_loss_pct=param_set['stop_loss_pct'],
            base_take_profit_pct=param_set['take_profit_pct'],
            holding_period=1,
            leverage=leverage
        )

        results.append({
            'params': param_set['name'],
            'leverage': leverage,
            'stop_loss': param_set['stop_loss_pct'] * 100,
            'take_profit': param_set['take_profit_pct'] * 100,
            'total_return': result['total_return'],
            'final_equity': result['final_equity'],
            'trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'sharpe': result['sharpe'],
            'max_drawdown': result['max_drawdown']
        })

# Display results
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Calculate improvement from leveraging
print("\n" + "="*80)
print("LEVERAGE IMPACT ANALYSIS")
print("="*80)

for param_set in param_sets:
    unleveraged = results_df[(results_df['params'] == param_set['name']) & (results_df['leverage'] == 1)]
    leveraged = results_df[(results_df['params'] == param_set['name']) & (results_df['leverage'] == 4)]

    if len(unleveraged) > 0 and len(leveraged) > 0:
        print(f"\n{param_set['name']}:")
        print(f"  1x leverage: {unleveraged['total_return'].iloc[0]:.2f}% return, {unleveraged['max_drawdown'].iloc[0]:.2f}% max DD")
        print(f"  4x leverage: {leveraged['total_return'].iloc[0]:.2f}% return, {leveraged['max_drawdown'].iloc[0]:.2f}% max DD")
        print(f"  Return multiplier: {leveraged['total_return'].iloc[0] / unleveraged['total_return'].iloc[0]:.2f}x")
        print(f"  Drawdown multiplier: {leveraged['max_drawdown'].iloc[0] / unleveraged['max_drawdown'].iloc[0]:.2f}x")

# Save results
output_file = f'{PAIR}_leverage_comparison_{TEST_DAYS}days.csv'
results_df.to_csv(output_file, index=False)
print(f"\n" + "="*80)
print(f"Results saved to: {output_file}")
print("="*80)
