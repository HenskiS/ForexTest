"""
Analyze daily returns and win/loss patterns from multi-pair strategy.

Shows what to expect day-to-day: average returns, win rates,
distribution of outcomes, and how many pairs win/lose each day.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = 750

# Trading parameters
STOP_LOSS_PCT = 0.0018
TAKE_PROFIT_PCT = 0.0200
TRANSACTION_COST_PCT = 0.0002
HOLDING_PERIOD = 1
LOWER_PCT = 48
UPPER_PCT = 52
BUFFER_WARMUP = 50

print("="*80)
print("DAILY RETURNS & WIN/LOSS ANALYSIS")
print("="*80)
print(f"Analyzing {TEST_DAYS} days of trading across {len(PAIRS)} pairs")
print("="*80)


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


def generate_predictions(pair, df, technical_features, best_params):
    """Generate rolling predictions for a pair"""
    predictions = []
    test_indices = []

    start_idx = len(df) - TEST_DAYS

    for i in tqdm(range(start_idx, len(df)), desc=f"  {pair}", leave=False):
        if i < TRAIN_WINDOW_SIZE:
            continue

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

        predictions.append(y_pred)
        test_indices.append(i)

    return np.array(predictions), test_indices


def backtest_single_pair(predictions, df_prices, test_indices):
    """Backtest a single pair and return detailed daily info"""
    prediction_buffer = list(predictions[:BUFFER_WARMUP])

    position = 0
    entry_price = 0.0
    holding_days = 0

    daily_returns = []
    daily_outcomes = []  # 'WIN', 'LOSS', 'NO_TRADE'
    daily_in_position = []

    test_data = df_prices.iloc[test_indices]
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

        if i >= BUFFER_WARMUP:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        if len(prediction_buffer) >= BUFFER_WARMUP:
            buffer_array = np.array(prediction_buffer)
            lower_threshold = np.percentile(buffer_array, LOWER_PCT)
            upper_threshold = np.percentile(buffer_array, UPPER_PCT)

            if prediction >= upper_threshold:
                signal = 1
            elif prediction <= lower_threshold:
                signal = -1
            else:
                signal = 0
        else:
            signal = 0

        day_return = 0.0
        outcome = 'NO_TRADE'

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

            if pct_low <= -STOP_LOSS_PCT:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                else:
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
            elif pct_high >= TAKE_PROFIT_PCT:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            elif holding_days >= HOLDING_PERIOD:
                exit_triggered = True
                exit_price = close_price

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price

                net_return_pct = raw_return_pct - TRANSACTION_COST_PCT
                day_return = net_return_pct
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                position = 0
                holding_days = 0

        daily_returns.append(day_return)
        daily_outcomes.append(outcome)
        daily_in_position.append(position != 0 or outcome != 'NO_TRADE')

        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    return np.array(daily_returns), daily_outcomes, daily_in_position


# Load data and generate predictions
print("\n" + "="*80)
print("STEP 1: LOADING DATA AND GENERATING PREDICTIONS")
print("="*80)

pair_predictions = {}
pair_data = {}

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

for pair in PAIRS:
    print(f"\n{pair}:")
    print("-" * 80)

    data_file = f'data/{pair}_1day_oanda.csv'
    if not os.path.exists(data_file):
        print(f"  ERROR: {data_file} not found")
        continue

    df_raw = pd.read_csv(data_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')

    df = calculate_features(df_raw.copy())
    df[TARGET] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=technical_features + [TARGET])

    hyperparam_file = f'hyperparams_rolling_daily_{pair}.pkl'
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

    predictions, test_indices = generate_predictions(pair, df, technical_features, best_params)

    pair_predictions[pair] = predictions
    pair_data[pair] = {
        'df': df,
        'test_indices': test_indices
    }

# Backtest each pair
print("\n" + "="*80)
print("STEP 2: BACKTESTING EACH PAIR")
print("="*80)

pair_returns = {}
pair_outcomes = {}

for pair in PAIRS:
    if pair not in pair_predictions:
        continue

    print(f"\n{pair}: Backtesting...")

    predictions = pair_predictions[pair]
    df = pair_data[pair]['df']
    test_indices = pair_data[pair]['test_indices']

    daily_returns, daily_outcomes, daily_in_position = backtest_single_pair(predictions, df, test_indices)
    pair_returns[pair] = daily_returns
    pair_outcomes[pair] = daily_outcomes

# Analyze daily statistics
print("\n" + "="*80)
print("STEP 3: ANALYZING DAILY STATISTICS")
print("="*80)

# Create dataframe for analysis
returns_df = pd.DataFrame(pair_returns)
outcomes_df = pd.DataFrame(pair_outcomes)

# Portfolio daily returns (equal weight, 1x leverage)
returns_df['portfolio'] = returns_df[PAIRS].mean(axis=1)

# Calculate daily portfolio statistics
print("\n" + "-"*80)
print("PORTFOLIO DAILY STATISTICS (1x leverage)")
print("-"*80)

portfolio_returns = returns_df['portfolio'].values
trading_days = (portfolio_returns != 0).sum()
total_days = len(portfolio_returns)

print(f"\nTotal days analyzed: {total_days}")
print(f"Days with trades: {trading_days} ({trading_days/total_days*100:.1f}%)")
print(f"Days with no activity: {total_days - trading_days} ({(total_days - trading_days)/total_days*100:.1f}%)")

# Daily return statistics
winning_days = portfolio_returns[portfolio_returns > 0]
losing_days = portfolio_returns[portfolio_returns < 0]
flat_days = portfolio_returns[portfolio_returns == 0]

print(f"\n{'='*80}")
print("DAILY RETURN BREAKDOWN")
print(f"{'='*80}")
print(f"Winning days: {len(winning_days)} ({len(winning_days)/total_days*100:.1f}%)")
print(f"Losing days: {len(losing_days)} ({len(losing_days)/total_days*100:.1f}%)")
print(f"Flat days: {len(flat_days)} ({len(flat_days)/total_days*100:.1f}%)")

print(f"\nAverage daily return: {portfolio_returns.mean()*100:.3f}%")
print(f"Average winning day: {winning_days.mean()*100:.3f}%")
print(f"Average losing day: {losing_days.mean()*100:.3f}%")
print(f"Best day: {portfolio_returns.max()*100:.3f}%")
print(f"Worst day: {portfolio_returns.min()*100:.3f}%")
print(f"Median day: {np.median(portfolio_returns)*100:.3f}%")

# Standard deviation
print(f"\nDaily volatility (std dev): {portfolio_returns.std()*100:.3f}%")

# Win/loss ratio
if len(winning_days) > 0 and len(losing_days) > 0:
    win_loss_ratio = abs(winning_days.mean() / losing_days.mean())
    print(f"Win/Loss ratio: {win_loss_ratio:.2f}x")

# Analyze how many pairs win/lose each day
print(f"\n{'='*80}")
print("PAIRS WINNING PER DAY")
print(f"{'='*80}")

pairs_winning_per_day = []
for idx in range(len(returns_df)):
    n_winning = sum(returns_df.iloc[idx][pair] > 0 for pair in PAIRS)
    pairs_winning_per_day.append(n_winning)

pairs_winning_counts = pd.Series(pairs_winning_per_day).value_counts().sort_index()

print("\nDistribution of winning pairs per day:")
for n_pairs, count in pairs_winning_counts.items():
    pct = count / len(returns_df) * 100
    print(f"  {n_pairs} pairs win: {count} days ({pct:.1f}%)")

# Percentiles
print(f"\n{'='*80}")
print("DAILY RETURN PERCENTILES (1x leverage)")
print(f"{'='*80}")

percentiles = [5, 10, 25, 50, 75, 90, 95]
print("\nWhat to expect on different days:")
for p in percentiles:
    val = np.percentile(portfolio_returns, p) * 100
    print(f"  {p}th percentile: {val:.3f}%")

print("\nInterpretation:")
print("  - 5th percentile = Bad day (only 5% of days are worse)")
print("  - 50th percentile = Typical day (median)")
print("  - 95th percentile = Great day (only 5% of days are better)")

# Leverage scaling
print(f"\n{'='*80}")
print("EXPECTED DAILY RETURNS AT DIFFERENT LEVERAGE")
print(f"{'='*80}")

leverage_levels = [1, 2, 3, 5, 10]
print("\nAverage daily return by leverage:")
for lev in leverage_levels:
    avg_daily = portfolio_returns.mean() * lev * 100
    monthly = ((1 + portfolio_returns.mean() * lev) ** 21 - 1) * 100
    print(f"  {lev}x leverage: {avg_daily:.3f}% per day (~{monthly:.2f}% per month)")

# Visualization
print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Daily return distribution
ax1 = axes[0, 0]
ax1.hist(portfolio_returns * 100, bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(portfolio_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {portfolio_returns.mean()*100:.3f}%')
ax1.axvline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Daily Return (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Daily Returns (1x leverage)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative returns over time
ax2 = axes[0, 1]
cumulative = (1 + portfolio_returns).cumprod()
ax2.plot(cumulative, linewidth=2)
ax2.set_xlabel('Days')
ax2.set_ylabel('Cumulative Return (Multiple)')
ax2.set_title('Portfolio Growth Over Time (1x leverage)')
ax2.grid(True, alpha=0.3)

# Plot 3: Pairs winning per day
ax3 = axes[1, 0]
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
bars = ax3.bar([str(i) for i in pairs_winning_counts.index],
               pairs_winning_counts.values,
               color=[colors[i] for i in pairs_winning_counts.index])
ax3.set_xlabel('Number of Pairs Winning')
ax3.set_ylabel('Number of Days')
ax3.set_title('How Many Pairs Win Each Day')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Rolling 20-day average return
ax4 = axes[1, 1]
rolling_avg = pd.Series(portfolio_returns).rolling(20).mean() * 100
ax4.plot(rolling_avg, linewidth=2)
ax4.axhline(portfolio_returns.mean() * 100, color='red', linestyle='--', linewidth=1, label='Overall Avg')
ax4.set_xlabel('Days')
ax4.set_ylabel('20-Day Avg Return (%)')
ax4.set_title('Rolling 20-Day Average Daily Return')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('daily_returns_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: daily_returns_analysis.png")

# Summary
print(f"\n{'='*80}")
print("WHAT TO EXPECT DAY-TO-DAY")
print(f"{'='*80}")

print(f"""
TYPICAL DAY (at 1x leverage):
- Average return: {portfolio_returns.mean()*100:.3f}%
- Median return: {np.median(portfolio_returns)*100:.3f}%
- You'll have activity ~{trading_days/total_days*100:.0f}% of days
- {len(winning_days)/total_days*100:.0f}% of days are winners
- {len(losing_days)/total_days*100:.0f}% of days are losers

NUMBER OF PAIRS WINNING:
- Most common: {pairs_winning_counts.idxmax()} pairs win ({pairs_winning_counts.max()/(len(returns_df))*100:.1f}% of days)
- All 4 pairs win: {pairs_winning_counts.get(4, 0)} days ({pairs_winning_counts.get(4, 0)/len(returns_df)*100:.1f}%)
- All 4 pairs lose: {pairs_winning_counts.get(0, 0)} days ({pairs_winning_counts.get(0, 0)/len(returns_df)*100:.1f}%)

AT 2x LEVERAGE:
- Average day: {portfolio_returns.mean()*2*100:.3f}%
- Good day (75th %ile): {np.percentile(portfolio_returns, 75)*2*100:.3f}%
- Bad day (25th %ile): {np.percentile(portfolio_returns, 25)*2*100:.3f}%
- Best day: {portfolio_returns.max()*2*100:.3f}%
- Worst day: {portfolio_returns.min()*2*100:.3f}%

AT 10x LEVERAGE:
- Average day: {portfolio_returns.mean()*10*100:.2f}%
- Good day (75th %ile): {np.percentile(portfolio_returns, 75)*10*100:.2f}%
- Bad day (25th %ile): {np.percentile(portfolio_returns, 25)*10*100:.2f}%
- Best day: {portfolio_returns.max()*10*100:.2f}%
- Worst day: {portfolio_returns.min()*10*100:.2f}%
""")

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
