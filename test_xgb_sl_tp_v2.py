"""
Test XGBoost with Stop-Loss and Take-Profit combinations
Based on verified working baseline script
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("TESTING XGBoost WITH SL/TP COMBINATIONS")
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
MAX_HOLD_DAYS = 5
POSITION_SIZE = 0.02  # Fixed 2% position sizing

# Test SL/TP combinations
SL_LEVELS = [None, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]  # None, 1%, 1.5%, 2%, 2.5%, 3%, 4%, 5%
TP_LEVELS = [None, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]  # None, 1%, 1.5%, 2%, 2.5%, 3%, 4%, 5%

print(f"Configuration:")
print(f"  Confidence: {CONFIDENCE_THRESHOLD}, Hold: {MAX_HOLD_DAYS}d, Position: {POSITION_SIZE*100:.0f}%")
print(f"  Testing SL: {['None' if s is None else f'{s*100:.1f}%' for s in SL_LEVELS]}")
print(f"  Testing TP: {['None' if t is None else f'{t*100:.1f}%' for t in TP_LEVELS]}")
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

def calculate_position_size(capital, price):
    """Calculate position size"""
    position_value = capital * POSITION_SIZE
    units = position_value / price
    return units

def run_backtest(sl_pct, tp_pct):
    """Run backtest with given SL/TP levels"""
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

            units = calculate_position_size(capital, entry_price)

            if units * entry_price < 10:
                continue

            direction = 'LONG' if prediction > 0 else 'SHORT'

            # Find future dates for holding period
            future_dates = [d for d in test_dates if d > date][:MAX_HOLD_DAYS]
            if len(future_dates) == 0:
                continue

            # Check each day for SL/TP/Time exit
            exit_price = None
            exit_date = None
            exit_reason = 'TIME'

            for check_date in future_dates:
                if check_date not in price_df.index:
                    continue

                day_data = price_df.loc[check_date]

                # Check stop loss
                if sl_pct is not None:
                    if direction == 'LONG':
                        sl_price = entry_price * (1 - sl_pct)
                        if day_data['low'] <= sl_price:
                            exit_price = sl_price - spread
                            exit_date = check_date
                            exit_reason = 'SL'
                            break
                    else:  # SHORT
                        sl_price = entry_price * (1 + sl_pct)
                        if day_data['high'] >= sl_price:
                            exit_price = sl_price + spread
                            exit_date = check_date
                            exit_reason = 'SL'
                            break

                # Check take profit
                if tp_pct is not None:
                    if direction == 'LONG':
                        tp_price = entry_price * (1 + tp_pct)
                        if day_data['high'] >= tp_price:
                            exit_price = tp_price - spread
                            exit_date = check_date
                            exit_reason = 'TP'
                            break
                    else:  # SHORT
                        tp_price = entry_price * (1 - tp_pct)
                        if day_data['low'] <= tp_price:
                            exit_price = tp_price + spread
                            exit_date = check_date
                            exit_reason = 'TP'
                            break

            # If no SL/TP hit, exit at max hold time
            if exit_price is None:
                exit_date = future_dates[-1]
                if exit_date not in price_df.index:
                    continue

                exit_price_raw = price_df.loc[exit_date, 'close']
                if direction == 'LONG':
                    exit_price = exit_price_raw - spread
                else:
                    exit_price = exit_price_raw + spread

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

# Run backtest with SL/TP combinations
print("="*100)
print("TESTING SL/TP COMBINATIONS")
print("="*100)
print()

results = []

print(f"{'SL':<10} {'TP':<10} {'Return':<12} {'Annual':<12} {'Trades':<10} {'Win Rate':<12} {'Max DD':<12} {'Sharpe':<12}")
print("-"*100)

for sl in SL_LEVELS:
    for tp in TP_LEVELS:
        sl_str = 'None' if sl is None else f'{sl*100:.1f}%'
        tp_str = 'None' if tp is None else f'{tp*100:.1f}%'

        result = run_backtest(sl, tp)

        if result is None:
            print(f"{sl_str:<10} {tp_str:<10} No trades")
            continue

        print(f"{sl_str:<10} {tp_str:<10} {result['total_return']*100:>8.2f}%    "
              f"{result['annual_return']*100:>8.2f}%    {result['num_trades']:<10} "
              f"{result['win_rate']*100:>8.1f}%    {result['max_dd']*100:>8.2f}%    "
              f"{result['sharpe']:>8.3f}")

        results.append({
            'sl': sl,
            'tp': tp,
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

    baseline = results_df[(results_df['sl'].isna()) & (results_df['tp'].isna())].iloc[0]

    print(f"Baseline (no SL/TP):")
    print(f"  Return: {baseline['total_return']*100:.2f}%")
    print(f"  Annual: {baseline['annual_return']*100:.2f}%")
    print(f"  Trades: {baseline['num_trades']}")
    print(f"  Win Rate: {baseline['win_rate']*100:.1f}%")
    print(f"  Max DD: {baseline['max_dd']*100:.2f}%")
    print(f"  Sharpe: {baseline['sharpe']:.3f}")
    print()

    # Find best SL/TP combination
    print("Best SL/TP combinations:")
    best_return = results_df.nlargest(1, 'total_return').iloc[0]
    best_sharpe = results_df.nlargest(1, 'sharpe').iloc[0]
    best_dd_ratio = results_df.copy()
    best_dd_ratio['return_dd_ratio'] = best_dd_ratio['total_return'] / abs(best_dd_ratio['max_dd'])
    best_dd_ratio = best_dd_ratio.nlargest(1, 'return_dd_ratio').iloc[0]

    sl_str = 'None' if pd.isna(best_return['sl']) else f"{best_return['sl']*100:.1f}%"
    tp_str = 'None' if pd.isna(best_return['tp']) else f"{best_return['tp']*100:.1f}%"
    print(f"  Best return: SL={sl_str}, TP={tp_str} = {best_return['total_return']*100:.2f}% return")

    sl_str = 'None' if pd.isna(best_sharpe['sl']) else f"{best_sharpe['sl']*100:.1f}%"
    tp_str = 'None' if pd.isna(best_sharpe['tp']) else f"{best_sharpe['tp']*100:.1f}%"
    print(f"  Best Sharpe: SL={sl_str}, TP={tp_str} = {best_sharpe['sharpe']:.3f} Sharpe")

    sl_str = 'None' if pd.isna(best_dd_ratio['sl']) else f"{best_dd_ratio['sl']*100:.1f}%"
    tp_str = 'None' if pd.isna(best_dd_ratio['tp']) else f"{best_dd_ratio['tp']*100:.1f}%"
    print(f"  Best return/DD: SL={sl_str}, TP={tp_str} = {best_dd_ratio['total_return']*100:.2f}% return")
    print()

print("="*100)
print("TESTING COMPLETE")
print("="*100)
