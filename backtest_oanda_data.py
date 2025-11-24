"""
Backtest rolling daily strategy on OANDA data.

Tests the exact production strategy (rolling daily retraining, percentile thresholds)
on OANDA historical data to verify performance matches expectations.
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
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_5day_return'
TRAIN_WINDOW_SIZE = 756
TEST_DAYS = args.test_days

print(f"Backtesting OANDA Data - {PAIR}")
print("="*80)
print(f"Test period: Last {TEST_DAYS} days")
print("="*80)

# Load raw OANDA data and calculate features fresh
data_file = f'data/{PAIR}_1day_oanda.csv'
if not os.path.exists(data_file):
    print(f"\nERROR: {data_file} not found")
    print("Run: python oanda_data_fetcher.py")
    sys.exit(1)

df_raw = pd.read_csv(data_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.set_index('date')
print(f"\nLoaded {len(df_raw)} days of raw OANDA data")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# Calculate features fresh (matching production trader)
print("\nCalculating technical features...")
def calculate_features(df):
    """Calculate all technical features (same as production trader)"""
    # Basic features
    df['momentum'] = df['close'].pct_change()
    df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['range'] = df['high'] - df['low']
    df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # EMAs
    for period in [10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ADX
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

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic
    lowest_low = df['low'].rolling(window=14).min()
    highest_high = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)

    # Williams %R
    df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

    # Bollinger Bands
    middle = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = middle + (std * 2)
    df['bb_middle'] = middle
    df['bb_lower'] = middle - (std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    df['atr'] = atr

    return df

df = calculate_features(df_raw.copy())

# Calculate target (5-day forward return)
df[TARGET] = df['close'].pct_change(5).shift(-5)

# Define feature list for subsetting
technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# Drop rows with missing features or targets
df = df.dropna(subset=technical_features + [TARGET])
print(f"Clean data: {len(df)} days ({df.index.min()} to {df.index.max()})")
print(f"(Last 5+ days excluded - no realized 5-day returns yet)")

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
    print(f"\nUsing default hyperparameters (optimized)")

print(f"  {best_params}")

# Generate predictions for test period
print(f"\nGenerating predictions for last {TEST_DAYS} days...")
print("-"*80)

if len(df) < TRAIN_WINDOW_SIZE + TEST_DAYS:
    print(f"ERROR: Need at least {TRAIN_WINDOW_SIZE + TEST_DAYS} days")
    print(f"Have: {len(df)} days")
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

    # Rolling 756-day window
    train_end_idx = i
    train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
    train_data = df.iloc[train_start_idx:train_end_idx]

    # Prepare data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[technical_features])
    y_train = train_data[TARGET].values

    # Train model
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

    # Predict
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
print(f"Date range: {test_dates[0]} to {test_dates[-1]}")

# Backtest with percentile thresholds
print("\n" + "="*80)
print("BACKTESTING WITH PERCENTILE THRESHOLDS")
print("="*80)

def backtest_strategy(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0040,
                      base_take_profit_pct=0.0100,
                      loss_cooldown_days=1,
                      transaction_cost_pct=0.0002,
                      holding_period=5,
                      buffer_warmup=50):
    """Backtest with rolling prediction buffer (mimics production)."""

    # Initialize buffer with first buffer_warmup predictions
    prediction_buffer = list(predictions[:buffer_warmup])

    position = 0
    entry_price = 0.0
    entry_date = None
    cooldown_remaining = 0
    holding_days = 0

    equity = [1000]
    trades = []
    signals_list = []

    test_data = df_prices.iloc[test_indices]
    test_dates_array = test_data.index
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    atr = test_data['atr'].values
    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(predictions)):
        prediction = predictions[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]

        # Update buffer (rolling window)
        if i >= buffer_warmup:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        # Calculate thresholds from buffer
        if len(prediction_buffer) >= buffer_warmup:
            buffer_array = np.array(prediction_buffer)
            lower_threshold = np.percentile(buffer_array, lower_pct)
            upper_threshold = np.percentile(buffer_array, upper_pct)

            # Generate signal
            if prediction >= upper_threshold:
                signal = 1
            elif prediction <= lower_threshold:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0  # No signal during warmup

        signals_list.append(signal)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Volatility-adjusted stops
        if not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
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

                # Store detailed trade info
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

    signals_array = np.array(signals_list)
    n_long = np.sum(signals_array == 1)
    n_short = np.sum(signals_array == -1)
    n_hold = np.sum(signals_array == 0)

    # Calculate annual return
    years = len(predictions) / 252  # Approximate trading days per year
    if years > 0:
        annual_return = (final_equity / 1000) ** (1 / years) - 1
        annual_return_pct = annual_return * 100
    else:
        annual_return_pct = 0

    return {
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'annual_return_pct': annual_return_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'n_long': n_long,
        'n_short': n_short,
        'n_hold': n_hold,
        'years': years,
        'trades': trades_df
    }

# Run backtest
result = backtest_strategy(predictions, df, test_indices,
                           lower_pct=48, upper_pct=52,
                           base_stop_loss_pct=0.0040,
                           base_take_profit_pct=0.0100,
                           loss_cooldown_days=1,
                           transaction_cost_pct=0.0002,
                           holding_period=5,
                           buffer_warmup=50)

# Display results
print("\n" + "="*80)
print(f"BACKTEST RESULTS - {PAIR} OANDA DATA")
print("="*80)
print(f"Test Period: {test_dates[0].date()} to {test_dates[-1].date()}")
print(f"Trading Days: {len(predictions)}")
print(f"Years: {result['years']:.2f}")
print()
print(f"Final Equity: ${result['final_equity']:,.2f}")
print(f"Total Return: {result['total_return_pct']:.2f}%")
print(f"Annual Return: {result['annual_return_pct']:.2f}%")
print()
print(f"Total Trades: {result['total_trades']}")
print(f"Win Rate: {result['win_rate']:.1f}%")
print()
print(f"Signal Distribution:")
print(f"  Long: {result['n_long']} ({result['n_long']/len(predictions)*100:.1f}%)")
print(f"  Short: {result['n_short']} ({result['n_short']/len(predictions)*100:.1f}%)")
print(f"  Hold: {result['n_hold']} ({result['n_hold']/len(predictions)*100:.1f}%)")
print()

# Trade statistics
if len(result['trades']) > 0:
    trades_df = result['trades']
    wins = trades_df[trades_df['outcome'] == 'WIN']
    losses = trades_df[trades_df['outcome'] == 'LOSS']

    print("Trade Statistics:")
    print(f"  Avg Win: {wins['net_return_pct'].mean():.2f}%")
    print(f"  Avg Loss: {losses['net_return_pct'].mean():.2f}%")
    print(f"  Largest Win: {wins['net_return_pct'].max():.2f}%")
    print(f"  Largest Loss: {losses['net_return_pct'].min():.2f}%")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum())
        print(f"  Profit Factor: {profit_factor:.2f}")

print("\n" + "="*80)
print("Backtest complete!")
print("="*80)

# Save trade details to CSV for visualization
if len(result['trades']) > 0:
    trades_output_file = f'{PAIR}_backtest_trades_{TEST_DAYS}days.csv'
    result['trades'].to_csv(trades_output_file, index=False)
    print(f"\nTrade details saved to: {trades_output_file}")
    print(f"Use plot_backtest_trades.py to visualize these trades")
