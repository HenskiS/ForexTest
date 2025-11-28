"""
Find optimal ATR% range for each pair, then trade whichever pair is in its sweet spot.

Strategy:
1. For each pair, analyze performance across different ATR% ranges
2. Find the ATR% range where each pair performs best
3. Each day, check which pair(s) are in their optimal range
4. Trade the best available opportunity
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

print(f"Optimal Volatility Range Strategy")
print("=" * 80)
print(f"Finding best ATR% range for each pair...")
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
    print(f"  {pair}: {len(df)} days")

# Load hyperparameters
best_params = {
    'n_estimators': 125,
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7
}

# Generate predictions and backtest results for all pairs
print("\nGenerating predictions and analyzing ATR ranges...")
pair_predictions = {}
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

    # Generate predictions
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

    # Backtest and record trades with ATR values
    trades = []
    for date_idx in range(len(predictions)):
        if date_idx < 50:
            continue

        # Get prediction buffer for percentile calculation
        pred_buffer = predictions[max(0, date_idx-200):date_idx+1]
        if len(pred_buffer) < 50:
            continue

        # Calculate signal
        lower_threshold = np.percentile(pred_buffer, 48)
        upper_threshold = np.percentile(pred_buffer, 52)
        prediction = predictions[date_idx]

        if prediction >= upper_threshold:
            signal = 1
        elif prediction <= lower_threshold:
            signal = -1
        else:
            continue

        # Get trade data
        row = test_data.iloc[date_idx]
        entry_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        atr_pct = row['atr_pct']

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
            'atr_pct': atr_pct,
            'net_return_pct': net_return_pct
        })

    pair_trades[pair] = pd.DataFrame(trades)
    print(f"  Total trades: {len(trades)}")

# Analyze optimal ATR range for each pair
print("\n" + "=" * 80)
print("OPTIMAL ATR% RANGE ANALYSIS")
print("=" * 80)

optimal_ranges = {}

for pair in PAIRS:
    trades_df = pair_trades[pair]

    # Calculate performance across ATR percentile ranges
    atr_values = trades_df['atr_pct'].values
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    best_range = None
    best_return = -999999

    print(f"\n{pair} Performance by ATR% Range:")
    print(f"{'Range':<20} {'Trades':<10} {'Avg Return':<15} {'Total Return':<15}")
    print("-" * 60)

    for i in range(len(percentiles) - 1):
        lower_pct = percentiles[i]
        upper_pct = percentiles[i + 1]

        lower_atr = np.percentile(atr_values, lower_pct)
        upper_atr = np.percentile(atr_values, upper_pct)

        range_trades = trades_df[(trades_df['atr_pct'] >= lower_atr) & (trades_df['atr_pct'] < upper_atr)]

        if len(range_trades) > 0:
            avg_return = range_trades['net_return_pct'].mean()
            total_return = range_trades['net_return_pct'].sum()

            label = f"{lower_atr:.4f}% - {upper_atr:.4f}%"
            print(f"{label:<20} {len(range_trades):<10} {avg_return:>10.3f}%     {total_return:>10.2f}%")

            if total_return > best_return and len(range_trades) >= 20:  # Need minimum 20 trades
                best_return = total_return
                best_range = (lower_atr, upper_atr, lower_pct, upper_pct)

    if best_range:
        optimal_ranges[pair] = {
            'min_atr': best_range[0],
            'max_atr': best_range[1],
            'min_percentile': best_range[2],
            'max_percentile': best_range[3],
            'total_return': best_return
        }
        print(f"\n  ** OPTIMAL: {best_range[0]:.4f}% - {best_range[1]:.4f}% ({int(best_range[2])}-{int(best_range[3])}th %ile) = {best_return:.2f}% total")

# Now backtest the strategy: trade whichever pair is in its optimal range
print("\n" + "=" * 80)
print("BACKTESTING 'OPTIMAL RANGE' STRATEGY")
print("=" * 80)

# Find common dates
common_dates = None
for pair in PAIRS:
    dates = set(pair_trades[pair]['date'])
    if common_dates is None:
        common_dates = dates
    else:
        common_dates = common_dates.intersection(dates)

common_dates = sorted(list(common_dates))
print(f"Common dates: {len(common_dates)} days")

# Build a lookup for each pair's trades
pair_trade_lookup = {}
for pair in PAIRS:
    pair_trade_lookup[pair] = pair_trades[pair].set_index('date')

equity = [1000]
strategy_trades = []
pair_selection_counts = {pair: 0 for pair in PAIRS}

for date in tqdm(common_dates, desc="Trading"):
    # Find which pairs are in their optimal range
    candidates = []

    for pair in PAIRS:
        if pair not in optimal_ranges:
            continue

        trade_row = pair_trade_lookup[pair].loc[date]
        atr = trade_row['atr_pct']
        opt_range = optimal_ranges[pair]

        # Check if in optimal range
        if opt_range['min_atr'] <= atr < opt_range['max_atr']:
            candidates.append({
                'pair': pair,
                'return': trade_row['net_return_pct'],
                'atr': atr
            })

    if not candidates:
        continue  # No pairs in optimal range

    # Trade the pair with highest expected return (based on historical performance in that range)
    # For simplicity, just take the first candidate (could rank by various criteria)
    best_candidate = candidates[0]

    pair_selection_counts[best_candidate['pair']] += 1

    equity.append(equity[-1] * (1 + best_candidate['return'] / 100))
    strategy_trades.append({
        'date': date,
        'pair': best_candidate['pair'],
        'atr_pct': best_candidate['atr'],
        'net_return_pct': best_candidate['return']
    })

# Results
strategy_trades_df = pd.DataFrame(strategy_trades)
final_equity = equity[-1]
total_return = (final_equity - 1000) / 10
years = len(common_dates) / 252
annual_return = ((final_equity / 1000) ** (1 / years) - 1) * 100 if years > 0 else 0

wins = strategy_trades_df[strategy_trades_df['net_return_pct'] > 0]
losses = strategy_trades_df[strategy_trades_df['net_return_pct'] <= 0]
win_rate = len(wins) / len(strategy_trades_df) * 100 if len(strategy_trades_df) > 0 else 0

print("\n" + "=" * 80)
print("OPTIMAL RANGE STRATEGY RESULTS")
print("=" * 80)
print(f"Test Period: {common_dates[0].date()} to {common_dates[-1].date()}")
print(f"Years: {years:.2f}")
print()
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annual Return: {annual_return:.2f}%")
print()
print(f"Total Trades: {len(strategy_trades_df)}")
print(f"Win Rate: {win_rate:.1f}%")
print()
print("Pair Selection Distribution:")
for pair in PAIRS:
    count = pair_selection_counts[pair]
    pct = count / len(strategy_trades_df) * 100 if len(strategy_trades_df) > 0 else 0
    opt_info = ""
    if pair in optimal_ranges:
        opt = optimal_ranges[pair]
        opt_info = f"  [Optimal: {opt['min_atr']:.4f}%-{opt['max_atr']:.4f}%]"
    print(f"  {pair}: {count} trades ({pct:.1f}%){opt_info}")

if len(wins) > 0 and len(losses) > 0:
    print()
    print(f"Avg Win: {wins['net_return_pct'].mean():.2f}%")
    print(f"Avg Loss: {losses['net_return_pct'].mean():.2f}%")
    profit_factor = abs(wins['net_return_pct'].sum() / losses['net_return_pct'].sum())
    print(f"Profit Factor: {profit_factor:.2f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"{'Strategy':<35} {'Annual Return':<15}")
print("-" * 80)
print(f"{'Optimal Range Selection':<35} {annual_return:>10.2f}%")
print(f"{'USDJPY (best single pair)':<35} {'45.46':>10}%")
print(f"{'EURUSD':<35} {'33.31':>10}%")
print(f"{'Most Volatile (any ATR)':<35} {'-9.67':>10}%")

if annual_return > 45.46:
    improvement = annual_return - 45.46
    print(f"\n{'*' * 80}")
    print(f"SUCCESS: +{improvement:.2f}% annual vs best single pair!")
    print(f"{'*' * 80}")
else:
    diff = 45.46 - annual_return
    print(f"\nResult: -{diff:.2f}% vs USDJPY (not an improvement)")

print("=" * 80)

# Save results
output_file = f'optimal_range_strategy_{TEST_DAYS}days.csv'
strategy_trades_df.to_csv(output_file, index=False)
print(f"\nTrade details saved to: {output_file}")
