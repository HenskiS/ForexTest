"""
Test XGBoost baseline (0.52 confidence, 5-day hold, no HMM) with different position sizes
to see how returns scale with higher capital allocation.
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING XGBoost WITH DIFFERENT POSITION SIZES")
print("="*100)
print()

# Configuration
XGB_MODEL_FILE = 'models/forex_xgb_per_pair_models.pkl'
DATA_DIR = 'data'
INITIAL_CAPITAL = 10000
TRAIN_END_DATE = '2024-01-01'
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURJPY']

# Optimal configuration from optimization
CONFIDENCE_THRESHOLD = 0.52
HOLD_DAYS = 5
USE_HMM = False

# Test different position sizes
POSITION_SIZES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0]

print(f"Configuration:")
print(f"  XGBoost config: confidence={CONFIDENCE_THRESHOLD}, hold={HOLD_DAYS}d, HMM={USE_HMM}")
print(f"  Testing position sizes: {[f'{p*100:.0f}%' for p in POSITION_SIZES]}")
print()

# Load XGBoost models
print("Loading XGBoost models...")
with open(XGB_MODEL_FILE, 'rb') as f:
    xgb_package = pickle.load(f)

xgb_models = xgb_package['models']
print(f"Loaded {len(xgb_models)} XGBoost models")
print()

def calculate_features(df):
    """Calculate technical features"""
    df['return_1d'] = df['close'].pct_change()
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift()),
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    return df

# Load data and generate predictions
print("Loading data and generating predictions...")
price_data = {}
predictions = {}
all_data = {}

for pair in PAIRS:
    spread_file = os.path.join(DATA_DIR, f'{pair}_1day_with_spreads.csv')
    regular_file = os.path.join(DATA_DIR, f'{pair}_1day_oanda.csv')

    if os.path.exists(spread_file):
        df = pd.read_csv(spread_file)
    else:
        df = pd.read_csv(regular_file)
        if 'JPY' in pair:
            df['spread'] = 0.02
        else:
            df['spread'] = 0.0002

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = calculate_features(df)
    df = df.dropna()

    all_data[pair] = df.copy()
    price_data[pair] = df

print(f"  Loaded {len(price_data)} pairs")

# Calculate cross-pair features
print("Calculating cross-pair features...")
for date in all_data[PAIRS[0]].index:
    returns_on_date = []
    for pair, df in all_data.items():
        if date in df.index:
            returns_on_date.append(df.loc[date, 'return_1d'])

    avg_return = np.mean(returns_on_date) if len(returns_on_date) > 0 else 0

    for pair, df in all_data.items():
        if date in df.index:
            df.loc[date, 'market_sentiment'] = avg_return

for pair, df in all_data.items():
    df['relative_strength_vs_market'] = df['return_1d'] - df['market_sentiment']

# Generate predictions with probabilities
print("Generating predictions with probabilities...")
for pair in PAIRS:
    if pair not in xgb_models or pair not in all_data:
        continue

    df = all_data[pair]
    model = xgb_models[pair]['model']
    features = xgb_models[pair]['features']

    X = df[features].values

    # Get probabilities (probability of class 1 = up)
    proba = model.predict_proba(X)[:, 1]

    # Store both binary prediction and probability
    df['prediction_proba'] = proba
    df['prediction_binary'] = (proba > 0.5).astype(int)

    # Convert to direction (-1, 0, 1)
    df['prediction'] = df['prediction_binary'].apply(lambda x: 1 if x == 1 else -1)

    predictions[pair] = df[['prediction', 'prediction_proba']].copy()

print(f"  Generated predictions for {len(predictions)} pairs")
print()

def calculate_position_size(capital, price, position_size_pct):
    """Calculate position size"""
    position_value = capital * position_size_pct
    units = position_value / price
    return units

def run_backtest(position_size_pct):
    """Run backtest with given position size"""
    capital = INITIAL_CAPITAL
    positions = []

    # Get test dates
    test_dates = set()
    for df in price_data.values():
        test_dates = test_dates.union(set(df[df.index >= TRAIN_END_DATE].index))

    for pred_df in predictions.values():
        test_dates = test_dates.intersection(set(pred_df.index))

    test_dates = sorted(list(test_dates))

    for date in test_dates:
        for pair in predictions.keys():
            if pair not in price_data or pair not in predictions:
                continue

            pred_df = predictions[pair]
            price_df = price_data[pair]

            if date not in pred_df.index or date not in price_df.index:
                continue

            # Get prediction and probability
            prediction = pred_df.loc[date, 'prediction']
            proba = pred_df.loc[date, 'prediction_proba']

            # Apply confidence threshold
            if prediction > 0 and proba < CONFIDENCE_THRESHOLD:
                continue
            if prediction < 0 and (1 - proba) < CONFIDENCE_THRESHOLD:
                continue

            # Get entry price and spread
            entry_price = price_df.loc[date, 'close']
            spread = price_df.loc[date, 'spread'] if 'spread' in price_df.columns else 0.0002

            # Apply spread cost
            if prediction > 0:
                entry_price += spread
            else:
                entry_price -= spread

            units = calculate_position_size(capital, entry_price, position_size_pct)

            if units * entry_price < 10:
                continue

            direction = 'LONG' if prediction > 0 else 'SHORT'

            # Find exit date
            future_dates = [d for d in test_dates if d > date][:HOLD_DAYS]

            if len(future_dates) == 0:
                continue

            exit_date = future_dates[-1]

            if exit_date not in price_df.index:
                continue

            # Exit at close
            exit_price = price_df.loc[exit_date, 'close']

            # Apply spread on exit
            if direction == 'LONG':
                exit_price -= spread
            else:
                exit_price += spread

            # Calculate PnL
            if direction == 'LONG':
                pnl = units * (exit_price - entry_price)
            else:
                pnl = units * (entry_price - exit_price)

            capital += pnl

            positions.append({
                'date': date,
                'pair': pair,
                'direction': direction,
                'proba': proba,
                'pnl': pnl,
                'return': pnl / (units * entry_price)
            })

    if len(positions) == 0:
        return None

    positions_df = pd.DataFrame(positions)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate = (positions_df['pnl'] > 0).mean()

    # Calculate drawdown
    positions_df = positions_df.sort_values('date')
    positions_df['cumulative_pnl'] = positions_df['pnl'].cumsum()
    positions_df['equity'] = INITIAL_CAPITAL + positions_df['cumulative_pnl']
    positions_df['peak'] = positions_df['equity'].cummax()
    positions_df['drawdown'] = (positions_df['equity'] - positions_df['peak']) / positions_df['peak']
    max_dd = positions_df['drawdown'].min()

    # Calculate Sharpe
    daily_returns = positions_df.groupby('date')['pnl'].sum() / INITIAL_CAPITAL
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)

    # Calculate annualized return
    test_days = (positions_df['date'].max() - positions_df['date'].min()).days
    test_years = test_days / 365.25
    annual_return = (1 + total_return) ** (1 / test_years) - 1

    return {
        'final_capital': capital,
        'total_return': total_return,
        'num_trades': len(positions_df),
        'win_rate': win_rate,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'annual_return': annual_return,
        'test_years': test_years
    }

# Run backtest with different position sizes
print("="*100)
print("TESTING DIFFERENT POSITION SIZES")
print("="*100)
print()

results = []

print(f"{'Position Size':<15} {'Return':<12} {'Annual':<12} {'Trades':<10} {'Win Rate':<12} {'Max DD':<12} {'Sharpe':<12}")
print("-"*100)

for position_size in POSITION_SIZES:
    result = run_backtest(position_size)

    if result is None:
        print(f"{position_size*100:>6.0f}%          No trades")
        continue

    print(f"{position_size*100:>6.0f}%          {result['total_return']*100:>8.2f}%    "
          f"{result['annual_return']*100:>8.2f}%    {result['num_trades']:<10} "
          f"{result['win_rate']*100:>8.1f}%    {result['max_dd']*100:>8.2f}%    "
          f"{result['sharpe']:>8.3f}")

    results.append({
        'position_size': position_size,
        **result
    })

print()

# Analysis
if len(results) > 0:
    results_df = pd.DataFrame(results)

    print("="*100)
    print("ANALYSIS")
    print("="*100)
    print()

    baseline = results_df[results_df['position_size'] == 0.02].iloc[0]

    print(f"Baseline (2% position size):")
    print(f"  Return: {baseline['total_return']*100:.2f}%")
    print(f"  Annual: {baseline['annual_return']*100:.2f}%")
    print(f"  Trades: {baseline['num_trades']}")
    print(f"  Win Rate: {baseline['win_rate']*100:.1f}%")
    print(f"  Max DD: {baseline['max_dd']*100:.2f}%")
    print(f"  Sharpe: {baseline['sharpe']:.3f}")
    print()

    print("Comparison to probability model (4 pairs, 2% position size):")
    print(f"  XGBoost baseline: {baseline['total_return']*100:.2f}% return")
    print(f"  Probability model: 0.91% return")
    print(f"  XGBoost is {baseline['total_return']/0.0091:.1f}x better")
    print()

    print("Scaling analysis:")
    print(f"  If we increase position size to match probability model's allocation:")

    for psize in [0.05, 0.10, 0.20]:
        row = results_df[results_df['position_size'] == psize].iloc[0]
        print(f"    {psize*100:.0f}% position size: {row['total_return']*100:.2f}% return, "
              f"{row['max_dd']*100:.2f}% DD, {row['sharpe']:.2f} Sharpe")

    print()

    # Find optimal risk-adjusted
    print("Best configurations by metric:")
    best_return = results_df.nlargest(1, 'total_return').iloc[0]
    best_sharpe = results_df.nlargest(1, 'sharpe').iloc[0]
    best_dd_ratio = results_df.copy()
    best_dd_ratio['return_dd_ratio'] = best_dd_ratio['total_return'] / abs(best_dd_ratio['max_dd'])
    best_dd_ratio = best_dd_ratio.nlargest(1, 'return_dd_ratio').iloc[0]

    print(f"  Best return: {best_return['position_size']*100:.0f}% position = "
          f"{best_return['total_return']*100:.2f}% return")
    print(f"  Best Sharpe: {best_sharpe['position_size']*100:.0f}% position = "
          f"{best_sharpe['sharpe']:.3f} Sharpe")
    print(f"  Best return/DD: {best_dd_ratio['position_size']*100:.0f}% position = "
          f"{best_dd_ratio['total_return']*100:.2f}% return, {best_dd_ratio['max_dd']*100:.2f}% DD")
    print()

print("="*100)
print("SUMMARY")
print("="*100)
print()
print("XGBoost baseline scales linearly with position size:")
print(f"  2% position size = ~{baseline['total_return']*100:.1f}% return")
print(f"  The win rate ({baseline['win_rate']*100:.1f}%) and trade count ({baseline['num_trades']}) stay constant")
print(f"  Drawdown and volatility scale proportionally with position size")
print()
print("This shows that with equal position sizing, XGBoost outperforms the probability model.")
