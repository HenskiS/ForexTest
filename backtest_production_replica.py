"""
Production-Replica Backtest

This backtest exactly replicates the production trading logic:
- Runs at 6am PST, Monday-Friday only
- Uses complete previous day's candle for predictions
- Holds position for 1 day, checks SL/TP on the holding candle
- Enters new positions based on prediction vs median threshold
- Uses actual spread data from CSVs
- Stop loss takes priority if both SL/TP could be hit same candle

Timeline:
- Day i: Prediction generated using this candle's features
- Day i+1: Entry at open (6am)
- Day i+1: Hold through this candle, SL/TP checked on high/low
- Day i+2 (6am): Exit at open if SL/TP not hit, then enter new position
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Must match production exactly
# =============================================================================
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
TRAIN_WINDOW = 378          # Days for training
BUFFER_SIZE = 200           # Prediction buffer size
BUFFER_WARMUP = 50          # Min predictions before trading

# Model-specific configs
MODEL_CONFIGS = {
    'ann': {
        'stop_loss': 0.0018,    # 0.18%
        'take_profit': 0.05,    # 5.0%
        'percentile': 50,       # Median split
    },
    'xgboost': {
        'stop_loss': 0.0018,    # 0.18%
        'take_profit': 0.03,    # 3.0%
        'percentile_lower': 48,
        'percentile_upper': 52,
    }
}

# ANN hyperparameters
ANN_PARAMS = {
    'hidden_layer_sizes': (13, 20, 31),
    'activation': 'tanh',
    'solver': 'sgd',
    'learning_rate_init': 0.001,
    'momentum': 0.4,
    'batch_size': 64,
    'max_iter': 20,
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'random_state': 42,
    'verbose': False,
    'early_stopping': False
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 125,
    'learning_rate': 0.1,
    'max_depth': 5,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# Feature order - must match production exactly
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
    'atr', 'volume_sma',
    'close_to_high', 'close_to_low',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10'
]


def calculate_features(df):
    """Calculate technical features - must match production exactly"""
    df = df.copy()

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
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
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
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    df['stoch_k'] = 100 * (df['close'] - low_20) / (high_20 - low_20)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR and Volume
    df['atr'] = atr
    df['volume_sma'] = df['volume'].rolling(window=20).mean()

    # Price position
    df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    return df


def load_data_with_spreads(pair):
    """Load price data and merge with spread data"""
    # Load price data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Load spread data if available
    spread_file = f'data/{pair}_1day_with_spreads.csv'
    if os.path.exists(spread_file):
        spread_df = pd.read_csv(spread_file)
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        df = df.join(spread_df[['spread_pct']], how='left')
    else:
        df['spread_pct'] = 0.00025  # Default spread

    # Fill missing spreads
    df['spread_pct'] = df['spread_pct'].fillna(0.00025)

    return df


def is_trading_day(date):
    """Check if date is a trading day (Monday-Friday)"""
    return date.dayofweek < 5  # 0=Monday, 4=Friday


def backtest_pair(pair, model_type='ann', start_date=None, end_date=None, verbose=False):
    """
    Backtest a single pair replicating production logic exactly.

    Timeline for each signal:
    - Day i: Generate prediction using candle i's features
    - Day i+1: Enter at open, hold through candle
    - Day i+1: Check candle's high/low for SL/TP
    - Day i+2 at open: Exit if SL/TP not hit

    Returns:
        dict with trades, daily_pnl, and metrics
    """
    config = MODEL_CONFIGS[model_type]
    STOP_LOSS_PCT = config['stop_loss']
    TAKE_PROFIT_PCT = config['take_profit']

    print(f"\n{'='*70}")
    print(f"BACKTESTING {pair} ({model_type.upper()})")
    print(f"SL: {STOP_LOSS_PCT*100:.2f}% | TP: {TAKE_PROFIT_PCT*100:.2f}%")
    print(f"{'='*70}")

    # Load data
    df = load_data_with_spreads(pair)
    df = calculate_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    if start_date:
        if df.index.tz is not None:
            start_date = start_date.tz_localize(df.index.tz)
        df = df[df.index >= start_date]
    if end_date:
        if df.index.tz is not None:
            end_date = end_date.tz_localize(df.index.tz)
        df = df[df.index <= end_date]

    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total rows: {len(df)}")

    # Initialize state
    prediction_buffer = []
    scaler = MinMaxScaler()

    # Results tracking
    trades = []
    daily_pnl = {}  # date -> pnl for that day

    # We need TRAIN_WINDOW + warmup before we can trade
    min_history = TRAIN_WINDOW + 250  # Extra for feature calculation

    print(f"Starting backtest loop...")

    for i in range(min_history, len(df) - 1):  # -1 because we need i+1 for entry
        signal_date = df.index[i]

        # Skip if entry day (i+1) is weekend
        entry_date = df.index[i + 1]
        if not is_trading_day(entry_date):
            continue

        # =====================================================================
        # STEP 1: Train model and generate prediction using candle i
        # =====================================================================
        train_end = i
        train_start = max(0, train_end - TRAIN_WINDOW)

        train_df = df.iloc[train_start:train_end+1].copy()
        train_df['target'] = train_df['close'].pct_change(1).shift(-1)
        train_df = train_df.dropna(subset=['target'])

        if len(train_df) < TRAIN_WINDOW - 50:
            continue

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['target'].values

        X_train_scaled = scaler.fit_transform(X_train)

        # Train model based on type
        if model_type == 'ann':
            model = MLPRegressor(**ANN_PARAMS)
            model.fit(X_train_scaled, y_train)
        else:  # xgboost
            model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X_train_scaled, y_train, verbose=False)

        # Predict using candle i's features
        X_pred = df.iloc[[i]][FEATURE_COLS].values
        X_pred_scaled = scaler.transform(X_pred)
        prediction = model.predict(X_pred_scaled)[0]

        # =====================================================================
        # STEP 2: Update prediction buffer and generate signal
        # =====================================================================
        prediction_buffer.append(prediction)
        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

        if len(prediction_buffer) < BUFFER_WARMUP:
            continue

        # Calculate threshold based on model type
        if model_type == 'ann':
            threshold = np.percentile(prediction_buffer, config['percentile'])
            # 50/50 split: always long or short
            if prediction >= threshold:
                direction = 1  # LONG
            else:
                direction = -1  # SHORT
        else:  # xgboost uses 48/52 percentile bands
            lower_threshold = np.percentile(prediction_buffer, config['percentile_lower'])
            upper_threshold = np.percentile(prediction_buffer, config['percentile_upper'])
            threshold = (lower_threshold + upper_threshold) / 2  # For logging
            if prediction >= upper_threshold:
                direction = 1  # LONG
            elif prediction <= lower_threshold:
                direction = -1  # SHORT
            else:
                continue  # No trade in dead zone

        # =====================================================================
        # STEP 3: Enter at day i+1 open, check SL/TP on day i+1 candle
        # =====================================================================
        entry_row = df.iloc[i + 1]
        entry_price_raw = entry_row['open']
        spread_pct = entry_row['spread_pct']

        if pd.isna(spread_pct):
            spread_pct = 0.00025

        # Calculate actual entry with spread
        if direction == 1:  # LONG
            actual_entry = entry_price_raw * (1 + spread_pct)
            stop_price = actual_entry * (1 - STOP_LOSS_PCT)
            target_price = actual_entry * (1 + TAKE_PROFIT_PCT)
        else:  # SHORT
            actual_entry = entry_price_raw * (1 - spread_pct)
            stop_price = actual_entry * (1 + STOP_LOSS_PCT)
            target_price = actual_entry * (1 - TAKE_PROFIT_PCT)

        # Check SL/TP on entry day's candle (day i+1)
        candle_high = entry_row['high']
        candle_low = entry_row['low']
        candle_close = entry_row['close']

        exit_price = None
        exit_reason = None

        if direction == 1:  # LONG
            # Stop loss first (conservative)
            if candle_low <= stop_price:
                exit_price = stop_price * (1 - spread_pct)
                exit_reason = 'STOP_LOSS'
            elif candle_high >= target_price:
                exit_price = target_price * (1 - spread_pct)
                exit_reason = 'TAKE_PROFIT'
            else:
                # Time exit at next day's open (= this candle's close)
                exit_price = candle_close * (1 - spread_pct)
                exit_reason = 'TIME_EXIT'
        else:  # SHORT
            # Stop loss first (conservative)
            if candle_high >= stop_price:
                exit_price = stop_price * (1 + spread_pct)
                exit_reason = 'STOP_LOSS'
            elif candle_low <= target_price:
                exit_price = target_price * (1 + spread_pct)
                exit_reason = 'TAKE_PROFIT'
            else:
                # Time exit at next day's open (= this candle's close)
                exit_price = candle_close * (1 + spread_pct)
                exit_reason = 'TIME_EXIT'

        # Calculate P&L
        if direction == 1:
            pnl_pct = (exit_price / actual_entry) - 1
        else:
            pnl_pct = (actual_entry / exit_price) - 1

        # Record trade
        trades.append({
            'pair': pair,
            'signal_date': signal_date,
            'entry_date': entry_date,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': actual_entry,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'prediction': prediction,
            'threshold': threshold
        })

        # Assign PnL to entry date
        if entry_date not in daily_pnl:
            daily_pnl[entry_date] = 0.0
        daily_pnl[entry_date] += pnl_pct

        if verbose:
            print(f"  {entry_date.date()}: {'LONG' if direction == 1 else 'SHORT'} @ {actual_entry:.5f} -> {exit_reason} @ {exit_price:.5f}, PnL: {pnl_pct*100:.3f}%")

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() * 100 if (trades_df['pnl_pct'] <= 0).any() else 0
        total_return = (1 + trades_df['pnl_pct']).prod() - 1

        # Exit reason breakdown
        exit_counts = trades_df['exit_reason'].value_counts()
        print(f"\nExit reasons:")
        for reason, count in exit_counts.items():
            print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        total_return = 0

    print(f"\nResults for {pair}:")
    print(f"  Total trades: {len(trades_df)}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Avg win: {avg_win:.3f}%")
    print(f"  Avg loss: {avg_loss:.3f}%")
    print(f"  Total return: {total_return*100:.2f}%")

    return {
        'pair': pair,
        'trades': trades_df,
        'daily_pnl': daily_pnl,
        'metrics': {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return
        }
    }


def run_multi_pair_backtest(pairs=PAIRS, model_type='ann', start_date=None, end_date=None, leverage=2.0):
    """
    Run backtest across multiple pairs with equal allocation.
    """
    config = MODEL_CONFIGS[model_type]

    print("="*100)
    print(f"PRODUCTION-REPLICA MULTI-PAIR BACKTEST ({model_type.upper()})")
    print("="*100)
    print(f"\nPairs: {', '.join(pairs)}")
    print(f"Model: {model_type.upper()}")
    print(f"Leverage: {leverage}x")
    print(f"SL: {config['stop_loss']*100:.2f}% | TP: {config['take_profit']*100:.2f}%")
    if model_type == 'ann':
        print(f"Percentile threshold: {config['percentile']} (median)")
    else:
        print(f"Percentile thresholds: {config['percentile_lower']}/{config['percentile_upper']}")

    # Run backtest for each pair
    pair_results = {}
    for pair in pairs:
        result = backtest_pair(pair, model_type, start_date, end_date)
        pair_results[pair] = result

    # Combine daily PnL across pairs (equal weight)
    all_dates = set()
    for result in pair_results.values():
        all_dates.update(result['daily_pnl'].keys())
    all_dates = sorted(all_dates)

    # Calculate portfolio daily returns
    allocation = 1.0 / len(pairs)
    portfolio_daily_pnl = []

    for date in all_dates:
        day_return = 0.0
        for pair in pairs:
            if date in pair_results[pair]['daily_pnl']:
                day_return += allocation * pair_results[pair]['daily_pnl'][date]
        portfolio_daily_pnl.append({'date': date, 'pnl': day_return})

    portfolio_df = pd.DataFrame(portfolio_daily_pnl)
    portfolio_df = portfolio_df.set_index('date')

    # Apply leverage
    portfolio_df['pnl_leveraged'] = portfolio_df['pnl'] * leverage

    # Calculate equity curve
    portfolio_df['equity'] = (1 + portfolio_df['pnl_leveraged']).cumprod() * 1000

    # Portfolio metrics
    total_return = portfolio_df['equity'].iloc[-1] / 1000 - 1
    years = len(portfolio_df) / 252
    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe ratio
    daily_returns = portfolio_df['pnl_leveraged']
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Max drawdown
    cummax = portfolio_df['equity'].cummax()
    drawdown = (portfolio_df['equity'] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Win rate (trading days with positive PnL)
    trading_days = portfolio_df[portfolio_df['pnl'] != 0]
    portfolio_win_rate = (trading_days['pnl'] > 0).mean() * 100 if len(trading_days) > 0 else 0

    # Summary
    print("\n" + "="*100)
    print("PORTFOLIO SUMMARY")
    print("="*100)
    print(f"\nDate range: {portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}")
    print(f"Trading days: {len(trading_days)}")
    print(f"\nPerformance ({leverage}x leverage):")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Annual return: {annual_return:.2f}%")
    print(f"  Sharpe ratio: {sharpe:.2f}")
    print(f"  Max drawdown: {max_dd:.2f}%")
    print(f"  Win rate: {portfolio_win_rate:.1f}%")
    print(f"  Final equity: ${portfolio_df['equity'].iloc[-1]:.2f}")

    # Per-pair summary
    print("\nPer-pair results:")
    for pair in pairs:
        m = pair_results[pair]['metrics']
        print(f"  {pair}: {m['total_trades']} trades, {m['win_rate']:.1f}% WR, {m['total_return']*100:.2f}% return")

    return {
        'pair_results': pair_results,
        'portfolio_df': portfolio_df,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': portfolio_win_rate
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Production-Replica Backtest')
    parser.add_argument('--pairs', nargs='+', default=PAIRS, help='Pairs to backtest')
    parser.add_argument('--model', type=str, default='ann', choices=['ann', 'xgboost'],
                        help='Model type: ann or xgboost')
    parser.add_argument('--leverage', type=float, default=2.0, help='Leverage multiplier')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Print trade details')
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start) if args.start else None
    end_date = pd.to_datetime(args.end) if args.end else None

    results = run_multi_pair_backtest(
        pairs=args.pairs,
        model_type=args.model,
        start_date=start_date,
        end_date=end_date,
        leverage=args.leverage
    )

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
