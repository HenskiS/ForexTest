"""
Analyze correlation and diversification benefits across currency pairs.

Shows how often pairs lose together, correlation matrices, and why
diversification reduces portfolio drawdown so dramatically.
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
TEST_DAYS = 250

# Trading parameters
STOP_LOSS_PCT = 0.0018
TAKE_PROFIT_PCT = 0.0200
TRANSACTION_COST_PCT = 0.0002
HOLDING_PERIOD = 1
LOWER_PCT = 48
UPPER_PCT = 52
BUFFER_WARMUP = 50

print("="*80)
print("MULTI-PAIR CORRELATION & DIVERSIFICATION ANALYSIS")
print("="*80)
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Test period: Last {TEST_DAYS} days")
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
    """Backtest a single pair and return daily returns + trade outcomes"""
    prediction_buffer = list(predictions[:BUFFER_WARMUP])

    position = 0
    entry_price = 0.0
    holding_days = 0

    daily_returns = []
    trade_outcomes = []  # Track each trade outcome

    test_data = df_prices.iloc[test_indices]
    test_dates = test_data.index
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
        current_date = test_dates[i]

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
            exit_reason = None

            if pct_low <= -STOP_LOSS_PCT:
                exit_triggered = True
                exit_reason = 'STOP_LOSS'
                if position == 1:
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                else:
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
            elif pct_high >= TAKE_PROFIT_PCT:
                exit_triggered = True
                exit_reason = 'TAKE_PROFIT'
                if position == 1:
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            elif holding_days >= HOLDING_PERIOD:
                exit_triggered = True
                exit_reason = 'TIME_EXIT'
                exit_price = close_price

            if exit_triggered:
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price

                net_return_pct = raw_return_pct - TRANSACTION_COST_PCT
                day_return = net_return_pct

                trade_outcomes.append({
                    'date': current_date,
                    'return': net_return_pct,
                    'win': net_return_pct > 0
                })

                position = 0
                holding_days = 0

        daily_returns.append(day_return)

        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    return np.array(daily_returns), trade_outcomes


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
    print(f"  Loaded {len(df_raw)} days")

    df = calculate_features(df_raw.copy())
    df[TARGET] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=technical_features + [TARGET])
    print(f"  Clean data: {len(df)} days")

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
    print(f"  Generated {len(predictions)} predictions")

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
pair_trades = {}
pair_equity_curves = {}

for pair in PAIRS:
    if pair not in pair_predictions:
        continue

    print(f"\n{pair}: Backtesting...")

    predictions = pair_predictions[pair]
    df = pair_data[pair]['df']
    test_indices = pair_data[pair]['test_indices']

    daily_returns, trade_outcomes = backtest_single_pair(predictions, df, test_indices)
    pair_returns[pair] = daily_returns
    pair_trades[pair] = trade_outcomes

    # Calculate individual pair equity curve and drawdown
    equity_curve = [1.0]
    for ret in daily_returns:
        equity_curve.append(equity_curve[-1] * (1 + ret))

    equity_curve = np.array(equity_curve)
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax * 100
    max_dd = drawdowns.min()

    pair_equity_curves[pair] = equity_curve

    total_return = (equity_curve[-1] - 1) * 100
    n_trades = len(trade_outcomes)
    n_wins = sum(1 for t in trade_outcomes if t['win'])
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

    print(f"  Total return: {total_return:.2f}%")
    print(f"  Trades: {n_trades} (Win rate: {win_rate:.1f}%)")
    print(f"  Max Drawdown: {max_dd:.2f}%")

# Correlation analysis
print("\n" + "="*80)
print("STEP 3: CORRELATION ANALYSIS")
print("="*80)

# Daily return correlations
returns_df = pd.DataFrame(pair_returns)
correlation_matrix = returns_df.corr()

print("\nDaily Return Correlations:")
print(correlation_matrix.to_string())

print("\n" + "="*80)
print("CORRELATION INSIGHTS")
print("="*80)

# Average correlation
avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

print("\nWhat this means:")
if avg_corr < 0.3:
    print("  - LOW correlation: Pairs move independently (EXCELLENT diversification)")
elif avg_corr < 0.6:
    print("  - MODERATE correlation: Some relationship but still good diversification")
else:
    print("  - HIGH correlation: Pairs tend to move together (limited diversification)")

# Analyze simultaneous losses
print("\n" + "="*80)
print("STEP 4: SIMULTANEOUS LOSS ANALYSIS")
print("="*80)

# Count days where multiple pairs lost money
loss_days_by_pair = {}
for pair in PAIRS:
    loss_days_by_pair[pair] = (returns_df[pair] < 0)

# Days where N pairs lost simultaneously
n_days = len(returns_df)
simultaneous_losses = []

for n_pairs_losing in range(5):
    count = 0
    for idx in returns_df.index:
        n_losing = sum(loss_days_by_pair[pair][idx] for pair in PAIRS if pair in loss_days_by_pair)
        if n_losing == n_pairs_losing:
            count += 1
    simultaneous_losses.append(count)
    pct = count / n_days * 100
    print(f"{n_pairs_losing} pairs losing on same day: {count} days ({pct:.1f}%)")

# Consecutive losing days
print("\n" + "="*80)
print("STEP 5: PORTFOLIO DRAWDOWN ANALYSIS")
print("="*80)

# Calculate portfolio equity curve (equal weight, 1x leverage)
portfolio_equity = np.ones(len(next(iter(pair_returns.values()))) + 1)
for i in range(len(portfolio_equity) - 1):
    daily_portfolio_return = sum(pair_returns[pair][i] * 0.25 for pair in PAIRS if pair in pair_returns)
    portfolio_equity[i + 1] = portfolio_equity[i] * (1 + daily_portfolio_return)

# Calculate drawdowns
cummax = np.maximum.accumulate(portfolio_equity)
portfolio_drawdowns = (portfolio_equity - cummax) / cummax * 100
portfolio_max_dd = portfolio_drawdowns.min()

print(f"\nPortfolio Max Drawdown (1x leverage): {portfolio_max_dd:.2f}%")
print(f"\nIndividual Pair Max Drawdowns:")
for pair in PAIRS:
    if pair in pair_equity_curves:
        equity = pair_equity_curves[pair]
        cummax = np.maximum.accumulate(equity)
        dd = ((equity - cummax) / cummax * 100).min()
        print(f"  {pair}: {dd:.2f}%")

# Calculate diversification benefit
avg_individual_dd = np.mean([
    ((pair_equity_curves[pair] - np.maximum.accumulate(pair_equity_curves[pair])) /
     np.maximum.accumulate(pair_equity_curves[pair]) * 100).min()
    for pair in PAIRS if pair in pair_equity_curves
])

diversification_benefit = avg_individual_dd - portfolio_max_dd
diversification_benefit_pct = (diversification_benefit / abs(avg_individual_dd)) * 100

print(f"\nAverage Individual Pair DD: {avg_individual_dd:.2f}%")
print(f"Portfolio DD (diversified): {portfolio_max_dd:.2f}%")
print(f"Diversification Benefit: {diversification_benefit:.2f}% ({diversification_benefit_pct:.1f}% reduction)")

# Visualization
print("\n" + "="*80)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Correlation heatmap
ax1 = axes[0, 0]
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            ax=ax1, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
ax1.set_title('Daily Return Correlations Between Pairs')

# Plot 2: Individual equity curves
ax2 = axes[0, 1]
for pair in PAIRS:
    if pair in pair_equity_curves:
        ax2.plot(pair_equity_curves[pair], label=pair, linewidth=2, alpha=0.7)
ax2.plot(portfolio_equity, label='Portfolio (diversified)', linewidth=3, color='black', linestyle='--')
ax2.set_xlabel('Days')
ax2.set_ylabel('Equity (normalized to 1.0)')
ax2.set_title('Individual Pairs vs Diversified Portfolio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Simultaneous losses bar chart
ax3 = axes[1, 0]
x_labels = ['0 pairs', '1 pair', '2 pairs', '3 pairs', '4 pairs']
colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
ax3.bar(x_labels, simultaneous_losses, color=colors)
ax3.set_ylabel('Number of Days')
ax3.set_title('Simultaneous Loss Days')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Drawdown comparison
ax4 = axes[1, 1]
dd_data = []
dd_labels = []
for pair in PAIRS:
    if pair in pair_equity_curves:
        equity = pair_equity_curves[pair]
        cummax = np.maximum.accumulate(equity)
        dd = ((equity - cummax) / cummax * 100).min()
        dd_data.append(dd)
        dd_labels.append(pair)

dd_data.append(portfolio_max_dd)
dd_labels.append('Portfolio')

colors_dd = ['red'] * len(PAIRS) + ['green']
bars = ax4.barh(dd_labels, dd_data, color=colors_dd)
ax4.set_xlabel('Max Drawdown (%)')
ax4.set_title('Max Drawdown: Individual Pairs vs Portfolio')
ax4.axvline(0, color='black', linewidth=0.5)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('pair_correlation_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: pair_correlation_analysis.png")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY: WHY DIVERSIFICATION ROCKS")
print("="*80)

print(f"""
1. CORRELATION: Average correlation is {avg_corr:.3f}
   → Pairs don't move in lockstep = less coordinated losses

2. SIMULTANEOUS LOSSES: All 4 pairs lose on same day only {simultaneous_losses[4]} times
   → That's only {simultaneous_losses[4]/n_days*100:.1f}% of days!

3. DRAWDOWN REDUCTION: {diversification_benefit_pct:.1f}% smaller drawdown
   → Individual pairs: {avg_individual_dd:.2f}% max DD
   → Portfolio: {portfolio_max_dd:.2f}% max DD
   → You get {diversification_benefit_pct:.1f}% less pain for FREE!

4. LEVERAGE IMPLICATION:
   → At 1x leverage: {portfolio_max_dd:.2f}% max DD
   → At 5x leverage: ~{portfolio_max_dd * 5:.2f}% max DD (still manageable!)
   → At 10x leverage: ~{portfolio_max_dd * 10:.2f}% max DD

This is why you can safely use higher leverage with a diversified portfolio
than you ever could with a single pair. Math is beautiful! 📊
""")

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
