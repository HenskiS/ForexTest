#!/usr/bin/env python3
"""
Optimize percentile thresholds for 1-day prediction model.
Tests various buy/sell threshold combinations with optimal parameters:
- Stop Loss: 0.18%
- Take Profit: 2.00%
- Holding Period: 1 day
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_features(df):
    """Calculate technical indicators as features."""
    df = df.copy()

    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)

    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Price relative to MAs
    df['price_to_sma10'] = df['close'] / df['sma_10'] - 1
    df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['close'] / df['sma_50'] - 1

    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Momentum
    df['momentum_10d'] = df['close'] - df['close'].shift(10)
    df['momentum_20d'] = df['close'] - df['close'].shift(20)

    return df

def run_backtest(pair, buy_threshold, sell_threshold, cooldown_days=0, stop_loss_pct=0.0018, take_profit_pct=0.0200, holding_period=1):
    """Run backtest with specified thresholds and cooldown period."""

    # Load data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'time'})
    df = df.sort_values('time').reset_index(drop=True)

    # Calculate features
    df = calculate_features(df)

    # Calculate target (1-day forward return)
    TARGET = 'target_1day_return'
    df[TARGET] = df['close'].pct_change(1).shift(-1)

    # Feature columns
    feature_cols = [
        'return_1d', 'return_5d', 'return_20d',
        'price_to_sma10', 'price_to_sma20', 'price_to_sma50',
        'volatility_20d', 'rsi_14', 'atr_pct', 'bb_position',
        'volume_ratio', 'momentum_10d', 'momentum_20d'
    ]

    # Remove NaN values
    df_clean = df.dropna(subset=feature_cols + [TARGET]).copy()

    # Rolling window backtest (756-day training window with 1-day gap)
    # Only test last 250 days for speed
    TRAIN_WINDOW_SIZE = 756
    TEST_DAYS = 250
    predictions = []
    actuals = []
    dates = []

    # Start predictions for last TEST_DAYS days
    start_idx = len(df_clean) - TEST_DAYS
    total_predictions = TEST_DAYS
    for idx, i in enumerate(range(start_idx, len(df_clean))):
        # Print progress every 50 predictions
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total_predictions} predictions...", end='\r')

        # Training data: days (i - TRAIN_WINDOW_SIZE - 1) through (i - 1)
        train_end_idx = i - 1
        train_start_idx = train_end_idx - TRAIN_WINDOW_SIZE
        train_data = df_clean.iloc[train_start_idx:train_end_idx]

        # Prediction day: day i
        pred_day = df_clean.iloc[i:i+1]

        # Train model
        X_train = train_data[feature_cols]
        y_train = train_data[TARGET]

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict
        X_pred = pred_day[feature_cols]
        pred = model.predict(X_pred)[0]

        predictions.append(pred)
        actuals.append(pred_day[TARGET].values[0])
        dates.append(pred_day['time'].values[0])

    print()  # Clear progress line

    # Create results dataframe
    results_df = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'actual': actuals
    })
    results_df['date'] = pd.to_datetime(results_df['date']).dt.tz_localize(None)

    # Merge with original data for ATR
    df_clean['date'] = pd.to_datetime(df_clean['time']).dt.tz_localize(None)
    results_df = results_df.merge(
        df_clean[['date', 'close', 'atr_pct']],
        on='date',
        how='left'
    )

    # Generate signals using rolling percentile thresholds
    BUFFER_SIZE = 200
    results_df['signal'] = 0

    for i in range(BUFFER_SIZE, len(results_df)):
        buffer = results_df['prediction'].iloc[i-BUFFER_SIZE:i]
        buy_thresh = np.percentile(buffer, buy_threshold)
        sell_thresh = np.percentile(buffer, sell_threshold)

        pred = results_df['prediction'].iloc[i]

        if pred <= buy_thresh:
            results_df.loc[results_df.index[i], 'signal'] = 1  # Long
        elif pred >= sell_thresh:
            results_df.loc[results_df.index[i], 'signal'] = -1  # Short

    # Simulate trades
    trades = []
    current_position = None
    last_exit_date = None

    for idx, row in results_df.iterrows():
        date = row['date']
        signal = row['signal']
        close_price = row['close']
        atr_pct = row['atr_pct']

        # Check if we should exit current position
        if current_position is not None:
            entry_date, entry_price, direction, stop_loss, take_profit = current_position
            days_held = (date - entry_date).days

            exit_reason = None
            exit_price = close_price

            if direction == 1:  # Long position
                pnl_pct = (close_price - entry_price) / entry_price
                if close_price <= stop_loss:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss
                elif close_price >= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif days_held >= holding_period:
                    exit_reason = 'time_exit'
            else:  # Short position
                pnl_pct = (entry_price - close_price) / entry_price
                if close_price >= stop_loss:
                    exit_reason = 'stop_loss'
                    exit_price = stop_loss
                elif close_price <= take_profit:
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                elif days_held >= holding_period:
                    exit_reason = 'time_exit'

            if exit_reason:
                if direction == 1:
                    final_pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    final_pnl_pct = (entry_price - exit_price) / entry_price

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': 'long' if direction == 1 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': final_pnl_pct,
                    'exit_reason': exit_reason,
                    'days_held': days_held
                })
                current_position = None
                last_exit_date = date

        # Check if we should enter a new position (respecting cooldown period)
        if current_position is None and signal != 0:
            # Check cooldown period
            if last_exit_date is not None and cooldown_days > 0:
                days_since_exit = (date - last_exit_date).days
                if days_since_exit < cooldown_days:
                    continue  # Skip this signal due to cooldown
            direction = signal
            entry_price = close_price

            # Calculate volatility-adjusted stops
            vol_mult = min(max(atr_pct / 0.01, 0.5), 2.0) if not pd.isna(atr_pct) else 1.0
            adjusted_stop = stop_loss_pct * vol_mult
            adjusted_tp = take_profit_pct * vol_mult

            if direction == 1:  # Long
                stop_loss = entry_price * (1 - adjusted_stop)
                take_profit = entry_price * (1 + adjusted_tp)
            else:  # Short
                stop_loss = entry_price * (1 + adjusted_stop)
                take_profit = entry_price * (1 - adjusted_tp)

            current_position = (date, entry_price, direction, stop_loss, take_profit)

    # Close any remaining position
    if current_position is not None:
        entry_date, entry_price, direction, stop_loss, take_profit = current_position
        exit_price = results_df['close'].iloc[-1]
        exit_date = results_df['date'].iloc[-1]
        days_held = (exit_date - entry_date).days

        if direction == 1:
            final_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            final_pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': 'long' if direction == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': final_pnl_pct,
            'exit_reason': 'end_of_data',
            'days_held': days_held
        })

    if len(trades) == 0:
        return None

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    total_return = (1 + trades_df['pnl_pct']).prod() - 1
    num_trades = len(trades_df)
    win_rate = (trades_df['pnl_pct'] > 0).sum() / num_trades
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
    avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] <= 0).any() else 0

    # Annualized metrics
    start_date = trades_df['entry_date'].min()
    end_date = trades_df['exit_date'].max()
    years = (end_date - start_date).days / 365.25
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    # Sharpe ratio
    returns_std = trades_df['pnl_pct'].std()
    sharpe = (trades_df['pnl_pct'].mean() / returns_std) * np.sqrt(252/holding_period) if returns_std > 0 else 0

    # Max drawdown
    cumulative = (1 + trades_df['pnl_pct']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'cooldown_days': cooldown_days,
        'total_return': total_return,
        'annual_return': annual_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

def main():
    """Test various threshold combinations with cooldown periods."""

    pairs = ['EURUSD']  # Start with just EURUSD

    # Test just a few key thresholds with cooldowns (very fast)
    buy_thresholds = [48]  # Current production value
    sell_thresholds = [52]  # Current production value
    cooldown_periods = [0, 1, 2, 3]  # 0 = no cooldown, 1-3 = days to wait after exit

    for pair in pairs:
        print(f"\n{'='*80}")
        print(f"Testing {pair}")
        print(f"{'='*80}\n")

        results = []
        total_tests = len(buy_thresholds) * len(sell_thresholds) * len(cooldown_periods)
        test_num = 0

        for cooldown in cooldown_periods:
            for buy_thresh in buy_thresholds:
                for sell_thresh in sell_thresholds:
                    test_num += 1
                    print(f"Testing {test_num}/{total_tests}: Buy={buy_thresh}, Sell={sell_thresh}, Cooldown={cooldown}d", end='... ')

                    result = run_backtest(
                        pair=pair,
                        buy_threshold=buy_thresh,
                        sell_threshold=sell_thresh,
                        cooldown_days=cooldown,
                        stop_loss_pct=0.0018,
                        take_profit_pct=0.0200,
                        holding_period=1
                    )

                    if result:
                        results.append(result)
                        print(f"Annual Return: {result['annual_return']*100:.2f}%, Trades: {result['num_trades']}, Sharpe: {result['sharpe']:.2f}")
                    else:
                        print("No trades")

        # Sort by annual return
        results.sort(key=lambda x: x['annual_return'], reverse=True)

        print(f"\n{'='*80}")
        print(f"Top 10 Results for {pair}")
        print(f"{'='*80}\n")
        print(f"{'Buy':>4} {'Sell':>4} {'Cool':>4} {'Annual Return':>13} {'Trades':>7} {'Win Rate':>9} {'Sharpe':>7} {'Max DD':>8}")
        print(f"{'-'*4} {'-'*4} {'-'*4} {'-'*13} {'-'*7} {'-'*9} {'-'*7} {'-'*8}")

        for result in results[:10]:
            print(f"{result['buy_threshold']:>4} {result['sell_threshold']:>4} {result['cooldown_days']:>4} "
                  f"{result['annual_return']*100:>12.2f}% {result['num_trades']:>7} "
                  f"{result['win_rate']*100:>8.1f}% {result['sharpe']:>7.2f} "
                  f"{result['max_drawdown']*100:>7.1f}%")

        # Save full results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{pair}_threshold_optimization.csv', index=False)
        print(f"\nFull results saved to {pair}_threshold_optimization.csv")

if __name__ == '__main__':
    main()
