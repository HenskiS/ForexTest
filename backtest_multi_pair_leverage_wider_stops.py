"""
Multi-Pair Leverage Optimization

Backtests the 4-pair strategy (EURUSD, GBPUSD, AUDUSD, USDJPY) at different
leverage levels to find optimal leverage for maximum returns while managing risk.

Each pair trades independently with 25% capital allocation.
Tests leverage from 1x to 10x to see risk/reward tradeoff.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt

# Configuration
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TARGET = 'target_1day_return'
TRAIN_WINDOW_SIZE = 378
TEST_DAYS = 750  # Last 250 days

# Trading parameters (from production config)
STOP_LOSS_PCT = 0.0020
TAKE_PROFIT_PCT = 0.0200
TRANSACTION_COST_PCT = 0.0002
HOLDING_PERIOD = 1
LOWER_PCT = 48
UPPER_PCT = 52
BUFFER_WARMUP = 50

# Leverage levels to test
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print("="*80)
print("MULTI-PAIR LEVERAGE OPTIMIZATION")
print("="*80)
print(f"Pairs: {', '.join(PAIRS)}")
print(f"Test period: Last {TEST_DAYS} days")
print(f"Leverage levels: {LEVERAGE_LEVELS}")
print("="*80)


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


def generate_predictions(pair, df, technical_features, best_params):
    """Generate rolling predictions for a pair"""
    predictions = []
    actuals = []
    test_dates = []
    test_indices = []

    start_idx = len(df) - TEST_DAYS

    for i in tqdm(range(start_idx, len(df)), desc=f"  {pair}", leave=False):
        if i < TRAIN_WINDOW_SIZE:
            continue

        # Rolling window
        train_end_idx = i - 1
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
        test_dates.append(df.index[i])
        test_indices.append(i)

    return np.array(predictions), np.array(actuals), pd.DatetimeIndex(test_dates), test_indices


def backtest_single_pair(predictions, df_prices, test_indices):
    """Backtest a single pair and return daily returns"""
    prediction_buffer = list(predictions[:BUFFER_WARMUP])

    position = 0
    entry_price = 0.0
    holding_days = 0

    # Track daily equity changes (as percentage returns)
    daily_returns = []

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

        # Update buffer
        if i >= BUFFER_WARMUP:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 200:
                prediction_buffer = prediction_buffer[-200:]

        # Calculate thresholds
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

        # Track return for this day
        day_return = 0.0

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

                position = 0
                holding_days = 0

        daily_returns.append(day_return)

        # Check for entry
        if position == 0 and signal != 0:
            position = signal
            entry_price = open_price
            holding_days = 0

    return np.array(daily_returns)


def simulate_multi_pair_portfolio(pair_returns, leverage):
    """
    Simulate portfolio with multiple pairs at given leverage.

    Args:
        pair_returns: Dict of {pair: daily_returns_array}
        leverage: Leverage multiplier

    Returns:
        Dict with portfolio metrics
    """
    # Start with $1000
    equity = 1000.0
    equity_curve = [equity]

    # Each pair gets 25% of capital
    capital_per_pair = 0.25

    # Get all dates (assuming all pairs have same test period)
    n_days = len(next(iter(pair_returns.values())))

    for day in range(n_days):
        daily_portfolio_return = 0.0

        # Sum returns from all pairs (weighted by allocation and leverage)
        for pair in PAIRS:
            if pair in pair_returns:
                pair_return = pair_returns[pair][day]
                # Return contribution: allocation * leverage * pair_return
                daily_portfolio_return += capital_per_pair * leverage * pair_return

        # Update equity
        equity = equity * (1 + daily_portfolio_return)
        equity_curve.append(equity)

        # Check for blow up
        if equity <= 0:
            # Account blown
            return {
                'leverage': leverage,
                'final_equity': 0.0,
                'total_return_pct': -100.0,
                'max_drawdown_pct': -100.0,
                'sharpe_ratio': 0.0,
                'blew_up': True,
                'blew_up_day': day,
                'equity_curve': equity_curve
            }

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - 1000) / 1000 * 100

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax * 100
    max_drawdown_pct = drawdowns.min()

    # Sharpe ratio (annualized)
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Annual return
    years = n_days / 252
    if years > 0 and final_equity > 0:
        annual_return_pct = ((final_equity / 1000) ** (1 / years) - 1) * 100
    else:
        annual_return_pct = 0.0

    return {
        'leverage': leverage,
        'final_equity': final_equity,
        'total_return_pct': total_return_pct,
        'annual_return_pct': annual_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'blew_up': False,
        'equity_curve': equity_curve
    }


# Main execution
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

    # Load data
    data_file = f'data/{pair}_1day_oanda.csv'
    if not os.path.exists(data_file):
        print(f"  ERROR: {data_file} not found")
        continue

    df_raw = pd.read_csv(data_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index('date')
    print(f"  Loaded {len(df_raw)} days")

    # Calculate features
    df = calculate_features(df_raw.copy())
    df[TARGET] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=technical_features + [TARGET])
    print(f"  Clean data: {len(df)} days")

    # Load hyperparameters
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

    # Generate predictions
    predictions, actuals, test_dates, test_indices = generate_predictions(
        pair, df, technical_features, best_params
    )

    print(f"  Generated {len(predictions)} predictions")

    pair_predictions[pair] = predictions
    pair_data[pair] = {
        'df': df,
        'test_indices': test_indices,
        'test_dates': test_dates
    }

print("\n" + "="*80)
print("STEP 2: BACKTESTING EACH PAIR INDIVIDUALLY")
print("="*80)

pair_returns = {}

for pair in PAIRS:
    if pair not in pair_predictions:
        continue

    print(f"\n{pair}: Backtesting...")

    predictions = pair_predictions[pair]
    df = pair_data[pair]['df']
    test_indices = pair_data[pair]['test_indices']

    daily_returns = backtest_single_pair(predictions, df, test_indices)
    pair_returns[pair] = daily_returns

    # Basic stats for individual pair
    total_return = (np.prod(1 + daily_returns) - 1) * 100
    n_trades = np.sum(daily_returns != 0)
    print(f"  Total return (1x leverage): {total_return:.2f}%")
    print(f"  Trades: {n_trades}")

print("\n" + "="*80)
print("STEP 3: TESTING LEVERAGE LEVELS")
print("="*80)

results = []

for leverage in tqdm(LEVERAGE_LEVELS, desc="Testing leverage levels"):
    result = simulate_multi_pair_portfolio(pair_returns, leverage)
    results.append(result)

# Create results dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("LEVERAGE OPTIMIZATION RESULTS")
print("="*80)
print(f"\nTested leverage: {LEVERAGE_LEVELS[0]}x to {LEVERAGE_LEVELS[-1]}x")
print(f"Test period: {TEST_DAYS} days (~{TEST_DAYS/252:.2f} years)")
print()

# Display results table
print(results_df[['leverage', 'final_equity', 'total_return_pct', 'annual_return_pct',
                  'max_drawdown_pct', 'sharpe_ratio', 'blew_up']].to_string(index=False))

# Find optimal leverage
viable_results = results_df[results_df['blew_up'] == False]

if len(viable_results) > 0:
    best_sharpe_idx = viable_results['sharpe_ratio'].idxmax()
    best_return_idx = viable_results['total_return_pct'].idxmax()

    print("\n" + "="*80)
    print("OPTIMAL LEVERAGE RECOMMENDATIONS")
    print("="*80)

    print("\nBest Risk-Adjusted (Highest Sharpe Ratio):")
    best_sharpe = results_df.loc[best_sharpe_idx]
    print(f"  Leverage: {best_sharpe['leverage']:.1f}x")
    print(f"  Total Return: {best_sharpe['total_return_pct']:.2f}%")
    print(f"  Annual Return: {best_sharpe['annual_return_pct']:.2f}%")
    print(f"  Max Drawdown: {best_sharpe['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")

    print("\nHighest Returns (Max Total Return):")
    best_return = results_df.loc[best_return_idx]
    print(f"  Leverage: {best_return['leverage']:.1f}x")
    print(f"  Total Return: {best_return['total_return_pct']:.2f}%")
    print(f"  Annual Return: {best_return['annual_return_pct']:.2f}%")
    print(f"  Max Drawdown: {best_return['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_return['sharpe_ratio']:.2f}")

    # Current production setting
    current_leverage = 2.0
    if current_leverage in results_df['leverage'].values:
        current = results_df[results_df['leverage'] == current_leverage].iloc[0]
        print(f"\nCurrent Production Setting ({current_leverage}x):")
        print(f"  Total Return: {current['total_return_pct']:.2f}%")
        print(f"  Annual Return: {current['annual_return_pct']:.2f}%")
        print(f"  Max Drawdown: {current['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {current['sharpe_ratio']:.2f}")

# Check for blow-ups
blown_up = results_df[results_df['blew_up'] == True]
if len(blown_up) > 0:
    print("\n" + "="*80)
    print("WARNING: ACCOUNT BLOW-UPS DETECTED")
    print("="*80)
    for _, row in blown_up.iterrows():
        print(f"  {row['leverage']:.1f}x leverage: Blew up on day {row['blew_up_day']}")

# Save results
results_df.to_csv(f'multi_pair_leverage_optimization_{TEST_DAYS}days.csv', index=False)
print(f"\n\nResults saved to: multi_pair_leverage_optimization_{TEST_DAYS}days.csv")

# Plot results
print("\nGenerating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Total Return vs Leverage
ax1 = axes[0, 0]
viable = results_df[results_df['blew_up'] == False]
blown = results_df[results_df['blew_up'] == True]
ax1.plot(viable['leverage'], viable['total_return_pct'], 'o-', label='Viable', linewidth=2)
if len(blown) > 0:
    ax1.plot(blown['leverage'], blown['total_return_pct'], 'rx', label='Blown Up', markersize=10, linewidth=2)
ax1.axvline(2.0, color='green', linestyle='--', alpha=0.5, label='Current (2x)')
ax1.set_xlabel('Leverage')
ax1.set_ylabel('Total Return (%)')
ax1.set_title('Total Return vs Leverage')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Max Drawdown vs Leverage
ax2 = axes[0, 1]
ax2.plot(viable['leverage'], viable['max_drawdown_pct'], 'o-', linewidth=2)
ax2.axvline(2.0, color='green', linestyle='--', alpha=0.5, label='Current (2x)')
ax2.set_xlabel('Leverage')
ax2.set_ylabel('Max Drawdown (%)')
ax2.set_title('Max Drawdown vs Leverage')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sharpe Ratio vs Leverage
ax3 = axes[1, 0]
ax3.plot(viable['leverage'], viable['sharpe_ratio'], 'o-', linewidth=2)
ax3.axvline(2.0, color='green', linestyle='--', alpha=0.5, label='Current (2x)')
ax3.set_xlabel('Leverage')
ax3.set_ylabel('Sharpe Ratio')
ax3.set_title('Sharpe Ratio vs Leverage')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Equity Curves for Selected Leverage Levels
ax4 = axes[1, 1]
selected_leverages = [1.0, 2.0, 3.0, 5.0]
for lev in selected_leverages:
    if lev in results_df['leverage'].values:
        result = results_df[results_df['leverage'] == lev].iloc[0]
        if not result['blew_up']:
            ax4.plot(result['equity_curve'], label=f"{lev}x", linewidth=2)
ax4.set_xlabel('Days')
ax4.set_ylabel('Equity ($)')
ax4.set_title('Equity Curves at Different Leverage')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'multi_pair_leverage_optimization_{TEST_DAYS}days.png', dpi=150)
print(f"Chart saved to: multi_pair_leverage_optimization_{TEST_DAYS}days.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
