"""
Backtest "Most Volatile Pair" Strategy

Each day, calculates ATR% for all 4 pairs and trades whichever has highest volatility.
Compares against trading each pair individually and trading all pairs equally.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import sys

# Configuration
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = 1000

print(f"Most Volatile Pair Strategy Backtest")
print("=" * 80)
print(f"Strategy: Each day, trade the pair with highest ATR%")
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Test period: Last {TEST_DAYS} days")
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

# Load and prepare data for all pairs
print("\nLoading data for all pairs...")
pair_data = {}

for pair in PAIRS:
    data_file = f'data/{pair}_1day_oanda.csv'
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found")
        sys.exit(1)

    df_raw = pd.read_csv(data_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')

    df = calculate_features(df_raw.copy())
    df[TARGET] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=technical_features + [TARGET])

    pair_data[pair] = df
    print(f"  {pair}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

# Load hyperparameters (use defaults if not found)
best_params = {
    'n_estimators': 125,
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7
}

# Generate predictions for all pairs
print("\nGenerating predictions for all pairs...")
pair_predictions = {}

for pair in PAIRS:
    print(f"\n{pair}:")
    df = pair_data[pair]
    predictions = []
    test_indices = []
    start_idx = len(df) - TEST_DAYS

    for i in tqdm(range(start_idx, len(df)), desc=f"  {pair}"):
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

    pair_predictions[pair] = {
        'predictions': np.array(predictions),
        'test_indices': test_indices,
        'test_data': df.iloc[test_indices]
    }

print("\nPredictions complete!")

# Find common date range across all pairs
print("\nAligning dates across pairs...")
common_dates = None
for pair in PAIRS:
    dates = pair_predictions[pair]['test_data'].index
    if common_dates is None:
        common_dates = set(dates)
    else:
        common_dates = common_dates.intersection(set(dates))

common_dates = sorted(list(common_dates))
print(f"Common date range: {len(common_dates)} days ({common_dates[0].date()} to {common_dates[-1].date()})")

# Backtest: Most Volatile Pair Strategy
print("\nBacktesting 'Most Volatile Pair' strategy...")

transaction_cost_pct = 0.0002
stop_loss_pct = 0.0018
take_profit_pct = 0.0200

equity = [1000]
trades = []
pair_trade_counts = {pair: 0 for pair in PAIRS}

for date_idx, date in enumerate(tqdm(common_dates, desc="Trading")):
    # Find which pair has highest ATR% today
    atr_values = {}
    for pair in PAIRS:
        test_data = pair_predictions[pair]['test_data']
        if date in test_data.index:
            atr_values[pair] = test_data.loc[date, 'atr_pct']

    if not atr_values:
        continue

    most_volatile_pair = max(atr_values, key=atr_values.get)
    pair_trade_counts[most_volatile_pair] += 1

    # Get prediction and data for most volatile pair
    pred_data = pair_predictions[most_volatile_pair]
    test_data = pred_data['test_data']
    predictions = pred_data['predictions']

    # Find index for this date
    date_loc = test_data.index.get_loc(date)

    # Get prediction buffer for percentile calculation
    if date_idx >= 50:
        pred_buffer = predictions[max(0, date_idx-200):date_idx+1]
    else:
        pred_buffer = predictions[:date_idx+1]

    if len(pred_buffer) < 50:
        continue

    # Calculate signal
    lower_threshold = np.percentile(pred_buffer, 48)
    upper_threshold = np.percentile(pred_buffer, 52)
    prediction = predictions[date_idx]

    if prediction >= upper_threshold:
        signal = 1  # Long
    elif prediction <= lower_threshold:
        signal = -1  # Short
    else:
        continue  # No trade

    # Execute trade
    row = test_data.iloc[date_loc]
    entry_price = row['open']
    high = row['high']
    low = row['low']
    close = row['close']

    # Check stop loss and take profit
    if signal == 1:  # Long
        pct_high = (high - entry_price) / entry_price
        pct_low = (low - entry_price) / entry_price
    else:  # Short
        pct_high = (entry_price - low) / entry_price
        pct_low = (entry_price - high) / entry_price

    # Determine exit
    if pct_low <= -stop_loss_pct:
        exit_price = entry_price * (1 - stop_loss_pct) if signal == 1 else entry_price * (1 + stop_loss_pct)
        exit_reason = 'Stop Loss'
    elif pct_high >= take_profit_pct:
        exit_price = entry_price * (1 + take_profit_pct) if signal == 1 else entry_price * (1 - take_profit_pct)
        exit_reason = 'Take Profit'
    else:
        exit_price = close
        exit_reason = 'Time'

    # Calculate return
    raw_return_pct = ((exit_price - entry_price) / entry_price if signal == 1
                      else (entry_price - exit_price) / entry_price) * 100
    net_return_pct = raw_return_pct - (transaction_cost_pct * 100)

    equity.append(equity[-1] * (1 + net_return_pct / 100))

    trades.append({
        'date': date,
        'pair': most_volatile_pair,
        'atr_pct': atr_values[most_volatile_pair],
        'direction': 'LONG' if signal == 1 else 'SHORT',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'net_return_pct': net_return_pct,
        'exit_reason': exit_reason
    })

# Results
trades_df = pd.DataFrame(trades)
final_equity = equity[-1]
total_return = (final_equity - 1000) / 10
years = len(common_dates) / 252
annual_return = ((final_equity / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0

wins = trades_df[trades_df['net_return_pct'] > 0]
losses = trades_df[trades_df['net_return_pct'] <= 0]
win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0

print("\n" + "=" * 80)
print("MOST VOLATILE PAIR STRATEGY RESULTS")
print("=" * 80)
print(f"Test Period: {common_dates[0].date()} to {common_dates[-1].date()}")
print(f"Trading Days: {len(common_dates)}")
print(f"Years: {years:.2f}")
print()
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annual Return: {annual_return:.2f}%")
print()
print(f"Total Trades: {len(trades_df)}")
print(f"Win Rate: {win_rate:.1f}%")
print()
print("Pair Distribution:")
for pair in PAIRS:
    count = pair_trade_counts[pair]
    pct = count / len(common_dates) * 100
    print(f"  {pair}: {count} days ({pct:.1f}%)")

if len(wins) > 0 and len(losses) > 0:
    print()
    print(f"Avg Win: {wins['net_return_pct'].mean():.2f}%")
    print(f"Avg Loss: {losses['net_return_pct'].mean():.2f}%")
    profit_factor = abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum())
    print(f"Profit Factor: {profit_factor:.2f}")

print("\n" + "=" * 80)
print("COMPARISON WITH INDIVIDUAL PAIRS")
print("=" * 80)
print(f"{'Strategy':<30} {'Annual Return':<15} {'Total Trades':<15}")
print("-" * 80)
print(f"{'Most Volatile Pair':<30} {annual_return:>10.2f}%     {len(trades_df):>8}")
print(f"{'USDJPY (best individual)':<30} {'45.46':>10}%     {'966':>8}")
print(f"{'EURUSD':<30} {'33.31':>10}%     {'985':>8}")
print(f"{'AUDUSD':<30} {'31.47':>10}%     {'967':>8}")
print(f"{'GBPUSD':<30} {'28.59':>10}%     {'973':>8}")

# Calculate improvement
if annual_return > 45.46:
    improvement = annual_return - 45.46
    print(f"\n{'*' * 80}")
    print(f"IMPROVEMENT: +{improvement:.2f}% annually vs best individual pair (USDJPY)")
    print(f"{'*' * 80}")
else:
    diff = 45.46 - annual_return
    print(f"\nResult: -{diff:.2f}% vs USDJPY (not an improvement)")

print("=" * 80)

# Save results
output_file = f'most_volatile_pair_backtest_{TEST_DAYS}days.csv'
trades_df.to_csv(output_file, index=False)
print(f"\nTrade details saved to: {output_file}")
