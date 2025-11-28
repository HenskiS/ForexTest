"""
Advanced strategy analysis:
4. Market regime analysis - Performance during high vs low volatility (ATR)
5. Entry timing optimization - Open vs close entry prices
6. Holding period flexibility - Testing 1, 2, and 3-day holding periods
7. Extended drawdown analysis - Recovery patterns and underwater periods
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

print(f"Advanced Strategy Insights - {PAIR}")
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

    # ATR as percentage of price for volatility regime
    df['atr_pct'] = (df['atr'] / df['close']) * 100

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

# Flexible backtest that can test different configurations
def flexible_backtest(predictions, df_prices, test_indices,
                      lower_pct=48, upper_pct=52,
                      base_stop_loss_pct=0.0018,
                      base_take_profit_pct=0.0200,
                      transaction_cost_pct=0.0002,
                      holding_period=1,
                      entry_type='open'):  # 'open' or 'close'

    prediction_buffer = list(predictions[:50])
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_atr_pct = 0.0
    holding_days = 0
    equity = [1000]
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    dates = test_data.index
    atr_pcts = test_data['atr_pct'].values

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
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_atr_pct': entry_atr_pct,
                    'net_return_pct': net_return_pct,
                    'outcome': outcome,
                    'exit_reason': exit_reason,
                    'holding_days': holding_days
                })

                position = 0
                holding_days = 0

        if position == 0 and signal != 0:
            position = signal
            entry_price = opens[i] if entry_type == 'open' else closes[i]
            entry_date = dates[i]
            entry_atr_pct = atr_pcts[i]
            holding_days = 0

    return equity[-1], trades

# ============================================================================
# ANALYSIS 4: MARKET REGIME (VOLATILITY) ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: MARKET REGIME - VOLATILITY (ATR)")
print("="*80)

print("\nRunning backtest...")
spread_cost = args.spread_pct / 100.0
cost = 0.0002 + spread_cost

final_equity, trades = flexible_backtest(predictions, df, test_indices, transaction_cost_pct=cost)
trades_df = pd.DataFrame(trades)

# Categorize by volatility regime at entry
trades_df['volatility_regime'] = pd.qcut(
    trades_df['entry_atr_pct'],
    q=3,
    labels=['Low Volatility', 'Medium Volatility', 'High Volatility'],
    duplicates='drop'
)

print("\nPerformance by Volatility Regime:")
print(f"{'Regime':<20} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Total Return':<12}")
print("-" * 80)

for regime in ['Low Volatility', 'Medium Volatility', 'High Volatility']:
    regime_trades = trades_df[trades_df['volatility_regime'] == regime]

    if len(regime_trades) > 0:
        wins = regime_trades[regime_trades['outcome'] == 'WIN']
        win_rate = len(wins) / len(regime_trades) * 100
        avg_return = regime_trades['net_return_pct'].mean()
        total_return = regime_trades['net_return_pct'].sum()

        print(f"{regime:<20} {len(regime_trades):<8} {win_rate:>8.1f}%    {avg_return:>8.3f}%    {total_return:>10.2f}%")

# ATR percentiles
atr_25 = trades_df['entry_atr_pct'].quantile(0.25)
atr_50 = trades_df['entry_atr_pct'].quantile(0.50)
atr_75 = trades_df['entry_atr_pct'].quantile(0.75)

print(f"\nATR Percentiles (as % of price):")
print(f"  25th: {atr_25:.4f}%")
print(f"  50th: {atr_50:.4f}%")
print(f"  75th: {atr_75:.4f}%")

# ============================================================================
# ANALYSIS 5: ENTRY TIMING - OPEN VS CLOSE
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: ENTRY TIMING - OPEN VS CLOSE")
print("="*80)

print("\nTesting entry at open vs close...")

equity_open, trades_open = flexible_backtest(
    predictions, df, test_indices,
    transaction_cost_pct=cost,
    entry_type='open'
)

equity_close, trades_close = flexible_backtest(
    predictions, df, test_indices,
    transaction_cost_pct=cost,
    entry_type='close'
)

trades_open_df = pd.DataFrame(trades_open)
trades_close_df = pd.DataFrame(trades_close)

print(f"\n{'Entry Type':<15} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Final Equity':<15} {'Total Return':<12}")
print("-" * 80)

for label, trades_data, equity in [('Open', trades_open_df, equity_open), ('Close', trades_close_df, equity_close)]:
    if len(trades_data) > 0:
        wins = trades_data[trades_data['outcome'] == 'WIN']
        win_rate = len(wins) / len(trades_data) * 100
        avg_return = trades_data['net_return_pct'].mean()
        total_return = (equity - 1000) / 10

        marker = " *" if equity == max(equity_open, equity_close) else ""
        print(f"{label:<15} {len(trades_data):<8} {win_rate:>8.1f}%    {avg_return:>8.3f}%    ${equity:>10,.2f}    {total_return:>8.2f}%{marker}")

diff = equity_close - equity_open
print(f"\nDifference (Close - Open): ${diff:+,.2f} ({diff/equity_open*100:+.2f}%)")

# ============================================================================
# ANALYSIS 6: HOLDING PERIOD FLEXIBILITY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 6: HOLDING PERIOD FLEXIBILITY")
print("="*80)

print("\nTesting 1, 2, and 3-day holding periods...")

holding_results = []

for holding_period in [1, 2, 3]:
    equity, trades_hp = flexible_backtest(
        predictions, df, test_indices,
        transaction_cost_pct=cost,
        holding_period=holding_period
    )

    trades_hp_df = pd.DataFrame(trades_hp)

    if len(trades_hp_df) > 0:
        wins = trades_hp_df[trades_hp_df['outcome'] == 'WIN']
        win_rate = len(wins) / len(trades_hp_df) * 100
        avg_return = trades_hp_df['net_return_pct'].mean()
        total_return = (equity - 1000) / 10

        holding_results.append({
            'holding_period': holding_period,
            'trades': len(trades_hp_df),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'final_equity': equity,
            'total_return': total_return
        })

holding_df = pd.DataFrame(holding_results)
holding_df = holding_df.sort_values('final_equity', ascending=False)

print(f"\n{'Holding':<10} {'Trades':<8} {'Win Rate':<12} {'Avg Return':<12} {'Final Equity':<15} {'Total Return':<12}")
print("-" * 80)

best_equity = holding_df.iloc[0]['final_equity']

for _, row in holding_df.iterrows():
    marker = " *" if row['final_equity'] == best_equity else ""
    print(f"{int(row['holding_period'])}-day     {int(row['trades']):<8} {row['win_rate']:>8.1f}%    "
          f"{row['avg_return']:>8.3f}%    ${row['final_equity']:>10,.2f}    {row['total_return']:>8.2f}%{marker}")

# Exit reason breakdown by holding period
print("\nExit Reason Breakdown by Holding Period:")
for holding_period in [1, 2, 3]:
    equity, trades_hp = flexible_backtest(
        predictions, df, test_indices,
        transaction_cost_pct=cost,
        holding_period=holding_period
    )
    trades_hp_df = pd.DataFrame(trades_hp)

    if len(trades_hp_df) > 0:
        print(f"\n{holding_period}-Day Hold:")
        for reason in ['STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT']:
            count = len(trades_hp_df[trades_hp_df['exit_reason'] == reason])
            pct = count / len(trades_hp_df) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

# ============================================================================
# ANALYSIS 7: EXTENDED DRAWDOWN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 7: EXTENDED DRAWDOWN ANALYSIS")
print("="*80)

# Build equity curve from trades
equity_curve = [1000]
dates_curve = [trades_df.iloc[0]['entry_date']]

for _, trade in trades_df.iterrows():
    equity_curve.append(equity_curve[-1] * (1 + trade['net_return_pct'] / 100))
    dates_curve.append(trade['exit_date'])

equity_series = pd.Series(equity_curve, index=dates_curve)

# Calculate drawdown metrics
running_max = equity_series.expanding().max()
drawdown = (equity_series - running_max) / running_max * 100

# Find all underwater periods (continuous time in drawdown)
underwater_periods = []
in_drawdown = False
drawdown_start = None
drawdown_peak = None

for i in range(len(drawdown)):
    if drawdown.iloc[i] < 0:
        if not in_drawdown:
            # Starting new drawdown
            in_drawdown = True
            drawdown_start = dates_curve[i]
            drawdown_peak = dates_curve[i-1] if i > 0 else dates_curve[i]
    else:
        if in_drawdown:
            # Ending drawdown
            in_drawdown = False
            drawdown_end = dates_curve[i]

            # Find the trough during this period
            period_mask = (equity_series.index >= drawdown_peak) & (equity_series.index <= drawdown_end)
            period_equity = equity_series[period_mask]
            trough_idx = period_equity.idxmin()
            max_dd_in_period = drawdown[equity_series.index == trough_idx].iloc[0]

            duration = (drawdown_end - drawdown_peak).days
            recovery = (drawdown_end - trough_idx).days

            underwater_periods.append({
                'peak_date': drawdown_peak,
                'trough_date': trough_idx,
                'recovery_date': drawdown_end,
                'max_drawdown': max_dd_in_period,
                'total_duration': duration,
                'recovery_duration': recovery
            })

# Sort by max drawdown
underwater_df = pd.DataFrame(underwater_periods)
if len(underwater_df) > 0:
    underwater_df = underwater_df.sort_values('max_drawdown', ascending=True)

    print("\nTop 5 Drawdown Periods:")
    print(f"{'Peak Date':<12} {'Trough Date':<12} {'Recovery':<12} {'Max DD':<10} {'Duration':<10} {'Recovery':<10}")
    print("-" * 80)

    for idx, row in underwater_df.head(5).iterrows():
        print(f"{row['peak_date'].date()}  {row['trough_date'].date()}  {row['recovery_date'].date()}  "
              f"{row['max_drawdown']:>7.2f}%  {int(row['total_duration']):>7}d   {int(row['recovery_duration']):>7}d")

    # Summary stats
    print("\nDrawdown Statistics:")
    print(f"  Number of drawdown periods: {len(underwater_df)}")
    print(f"  Avg drawdown: {underwater_df['max_drawdown'].mean():.2f}%")
    print(f"  Avg duration: {underwater_df['total_duration'].mean():.0f} days")
    print(f"  Avg recovery: {underwater_df['recovery_duration'].mean():.0f} days")
    print(f"  Max drawdown: {underwater_df['max_drawdown'].min():.2f}%")
    print(f"  Longest duration: {underwater_df['total_duration'].max():.0f} days")

# Time underwater
total_days = (dates_curve[-1] - dates_curve[0]).days
underwater_days = underwater_df['total_duration'].sum()
underwater_pct = underwater_days / total_days * 100

print(f"\n  Time underwater: {underwater_days:.0f} / {total_days:.0f} days ({underwater_pct:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
