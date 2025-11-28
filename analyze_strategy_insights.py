"""
Comprehensive strategy analysis to improve model usage:
1. Time-of-week performance - identify best/worst trading days
2. Prediction magnitude analysis - does signal strength matter?
3. Drawdown analysis - risk metrics and recovery patterns
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
parser.add_argument('--test-days', type=int, default=1000, help='Number of recent days to backtest')
parser.add_argument('--spread-pct', type=float, default=0.0, help='Spread cost per trade as percentage')
args = parser.parse_args()

PAIR = args.pair.upper()
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = args.test_days

print(f"Strategy Insights Analysis - {PAIR}")
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

# Enhanced backtest that tracks everything
def detailed_backtest(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0018,
                      base_take_profit_pct=0.0200,
                      transaction_cost_pct=0.0002,
                      holding_period=1):

    prediction_buffer = list(predictions[:50])
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_prediction = 0.0
    holding_days = 0
    equity = [1000]
    equity_curve = []  # Track equity at each timestamp
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    dates = test_data.index
    atrs = test_data['atr'].values

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

            # Calculate signal strength (distance from threshold)
            if signal == 1:
                signal_strength = predictions[i] - upper_threshold
            elif signal == -1:
                signal_strength = lower_threshold - predictions[i]
            else:
                signal_strength = 0
        else:
            signal = 0
            signal_strength = 0
            lower_threshold = 0
            upper_threshold = 0

        if position != 0:
            holding_days += 1

            if position == 1:
                pct_high = (highs[i] - entry_price) / entry_price
                pct_low = (lows[i] - entry_price) / entry_price
            else:
                pct_high = (entry_price - lows[i]) / entry_price
                pct_low = (entry_price - highs[i]) / entry_price

            exit_price = None
            exit_reason = None

            if pct_low <= -base_stop_loss_pct:
                exit_price = entry_price * (1 - base_stop_loss_pct) if position == 1 else entry_price * (1 + base_stop_loss_pct)
                exit_reason = 'STOP_LOSS'
            elif pct_high >= base_take_profit_pct:
                exit_price = entry_price * (1 + base_take_profit_pct) if position == 1 else entry_price * (1 - base_take_profit_pct)
                exit_reason = 'TAKE_PROFIT'
            elif holding_days >= holding_period:
                exit_price = closes[i]
                exit_reason = 'TIME_EXIT'

            if exit_price:
                raw_return_pct = ((exit_price - entry_price) / entry_price if position == 1
                                  else (entry_price - exit_price) / entry_price) * 100
                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                equity.append(equity[-1] * (1 + net_return_pct / 100))

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': dates[i],
                    'entry_day_of_week': entry_date.dayofweek,  # 0=Monday, 4=Friday
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_prediction': entry_prediction,
                    'entry_signal_strength': abs(entry_prediction - (upper_threshold if position == 1 else lower_threshold)),
                    'entry_atr': atrs[max(0, i-1)],  # ATR at entry
                    'net_return_pct': net_return_pct,
                    'outcome': outcome,
                    'exit_reason': exit_reason,
                    'holding_days': holding_days
                })

                position = 0
                holding_days = 0

        # Track equity curve for drawdown calculation
        equity_curve.append({'date': dates[i], 'equity': equity[-1]})

        if position == 0 and signal != 0:
            position = signal
            entry_price = opens[i]
            entry_date = dates[i]
            entry_prediction = predictions[i]
            holding_days = 0

    return equity[-1], trades, equity_curve

# Run backtest
print("\nRunning detailed backtest...")
spread_cost = args.spread_pct / 100.0
cost = 0.0002 + spread_cost

final_equity, trades, equity_curve = detailed_backtest(
    predictions, df, test_indices,
    transaction_cost_pct=cost
)

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

print(f"\nTotal trades: {len(trades_df)}")

# ============================================================================
# ANALYSIS 1: TIME OF WEEK PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: TIME OF WEEK PERFORMANCE")
print("="*80)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Group by entry day of week
weekly_stats = []
for day in range(5):  # 0-4 (Mon-Fri)
    day_trades = trades_df[trades_df['entry_day_of_week'] == day]

    if len(day_trades) > 0:
        wins = day_trades[day_trades['outcome'] == 'WIN']
        losses = day_trades[day_trades['outcome'] == 'LOSS']

        win_rate = len(wins) / len(day_trades) * 100
        avg_return = day_trades['net_return_pct'].mean()
        total_return = day_trades['net_return_pct'].sum()

        weekly_stats.append({
            'day': day_names[day],
            'trades': len(day_trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'wins': len(wins),
            'losses': len(losses)
        })

weekly_df = pd.DataFrame(weekly_stats)
weekly_df = weekly_df.sort_values('total_return', ascending=False)

print(f"\n{'Day':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win Rate':<12} {'Avg Return':<12} {'Total Return':<12}")
print("-" * 80)

for _, row in weekly_df.iterrows():
    print(f"{row['day']:<12} {int(row['trades']):<8} {int(row['wins']):<6} {int(row['losses']):<8} "
          f"{row['win_rate']:>8.1f}%    {row['avg_return']:>8.3f}%    {row['total_return']:>10.2f}%")

best_day = weekly_df.iloc[0]
worst_day = weekly_df.iloc[-1]

print(f"\nBest Day: {best_day['day']} ({best_day['total_return']:.2f}% total return)")
print(f"Worst Day: {worst_day['day']} ({worst_day['total_return']:.2f}% total return)")

# ============================================================================
# ANALYSIS 2: PREDICTION MAGNITUDE (SIGNAL STRENGTH)
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: PREDICTION MAGNITUDE / SIGNAL STRENGTH")
print("="*80)

# Quartile analysis
trades_df['signal_strength_quartile'] = pd.qcut(
    trades_df['entry_signal_strength'],
    q=4,
    labels=['Q1 (Weakest)', 'Q2', 'Q3', 'Q4 (Strongest)'],
    duplicates='drop'
)

print("\nPerformance by Signal Strength Quartile:")
print(f"{'Quartile':<18} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Total Return':<12}")
print("-" * 80)

for quartile in ['Q1 (Weakest)', 'Q2', 'Q3', 'Q4 (Strongest)']:
    q_trades = trades_df[trades_df['signal_strength_quartile'] == quartile]

    if len(q_trades) > 0:
        wins = q_trades[q_trades['outcome'] == 'WIN']
        win_rate = len(wins) / len(q_trades) * 100
        avg_return = q_trades['net_return_pct'].mean()
        total_return = q_trades['net_return_pct'].sum()

        print(f"{quartile:<18} {len(q_trades):<8} {win_rate:>8.1f}%    {avg_return:>8.3f}%    {total_return:>10.2f}%")

# Analyze if filtering weak signals would help
print("\nWhat if we skipped weak signals?")
print(f"{'Min Strength':<15} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Total Return':<12}")
print("-" * 80)

# Original (no filter)
win_rate_all = len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100
avg_return_all = trades_df['net_return_pct'].mean()
total_return_all = trades_df['net_return_pct'].sum()
print(f"{'No filter':<15} {len(trades_df):<8} {win_rate_all:>8.1f}%    {avg_return_all:>8.3f}%    {total_return_all:>10.2f}%")

# Try different strength thresholds
for percentile in [25, 50, 75]:
    threshold = np.percentile(trades_df['entry_signal_strength'], percentile)
    filtered = trades_df[trades_df['entry_signal_strength'] >= threshold]

    if len(filtered) > 0:
        win_rate = len(filtered[filtered['outcome'] == 'WIN']) / len(filtered) * 100
        avg_return = filtered['net_return_pct'].mean()
        total_return = filtered['net_return_pct'].sum()

        print(f"Top {100-percentile}% ({threshold:.6f})  {len(filtered):<8} {win_rate:>8.1f}%    "
              f"{avg_return:>8.3f}%    {total_return:>10.2f}%")

# ============================================================================
# ANALYSIS 3: DRAWDOWN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: DRAWDOWN ANALYSIS")
print("="*80)

# Calculate running max and drawdown
equity_df['running_max'] = equity_df['equity'].expanding().max()
equity_df['drawdown_pct'] = (equity_df['equity'] - equity_df['running_max']) / equity_df['running_max'] * 100

# Find max drawdown
max_drawdown = equity_df['drawdown_pct'].min()
max_dd_idx = equity_df['drawdown_pct'].idxmin()
max_dd_date = equity_df.loc[max_dd_idx, 'date']
max_dd_equity = equity_df.loc[max_dd_idx, 'equity']

# Find when max was reached before drawdown
equity_before_dd = equity_df.loc[:max_dd_idx]
peak_idx = equity_before_dd['equity'].idxmax()
peak_date = equity_df.loc[peak_idx, 'date']
peak_equity = equity_df.loc[peak_idx, 'equity']

print(f"\nMax Drawdown: {max_drawdown:.2f}%")
print(f"  Peak: ${peak_equity:.2f} on {peak_date.date()}")
print(f"  Trough: ${max_dd_equity:.2f} on {max_dd_date.date()}")
print(f"  Duration: {(max_dd_date - peak_date).days} days")

# Recovery analysis
if max_dd_idx < len(equity_df) - 1:
    equity_after_dd = equity_df.loc[max_dd_idx:]
    recovered = equity_after_dd[equity_after_dd['equity'] >= peak_equity]

    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
        recovery_date = equity_df.loc[recovery_idx, 'date']
        recovery_days = (recovery_date - max_dd_date).days
        print(f"  Recovery: {recovery_days} days to reach new high")
    else:
        print(f"  Recovery: Still in drawdown (current: {equity_df.iloc[-1]['drawdown_pct']:.2f}%)")

# Consecutive losses
print("\nConsecutive Loss Streaks:")
streak = 0
max_streak = 0
max_streak_loss = 0
current_streak_loss = 0

for _, trade in trades_df.iterrows():
    if trade['outcome'] == 'LOSS':
        streak += 1
        current_streak_loss += trade['net_return_pct']
        if streak > max_streak:
            max_streak = streak
            max_streak_loss = current_streak_loss
    else:
        streak = 0
        current_streak_loss = 0

print(f"  Max consecutive losses: {max_streak}")
print(f"  Total loss during max streak: {max_streak_loss:.2f}%")

# Risk metrics
print("\nRisk Metrics:")
returns = trades_df['net_return_pct'].values
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
print(f"  Sharpe Ratio (annualized): {sharpe:.2f}")
print(f"  Win Rate: {len(trades_df[trades_df['outcome'] == 'WIN']) / len(trades_df) * 100:.1f}%")
print(f"  Avg Win: {trades_df[trades_df['outcome'] == 'WIN']['net_return_pct'].mean():.3f}%")
print(f"  Avg Loss: {trades_df[trades_df['outcome'] == 'LOSS']['net_return_pct'].mean():.3f}%")

wins = trades_df[trades_df['outcome'] == 'WIN']['net_return_pct'].sum()
losses = abs(trades_df[trades_df['outcome'] == 'LOSS']['net_return_pct'].sum())
profit_factor = wins / losses if losses > 0 else float('inf')
print(f"  Profit Factor: {profit_factor:.2f}")

# Save detailed results
output_file = f'{PAIR}_strategy_insights_{TEST_DAYS}days.csv'
trades_df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
