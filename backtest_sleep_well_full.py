"""
Full backtest of Sleep Well strategy on all available data.
Trains ANN with rolling window, saves predictions, shows yearly performance.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from trading.config import TradingConfig
from trading.market_utils import calculate_technical_features

warnings.filterwarnings('ignore')

# Sleep Well Config
LOWER_PCT = 10
UPPER_PCT = 90
HOLD_DAYS = 5
SL_PCT = 0.02
LEVERAGE = 1.5
TRAIN_WINDOW = 378
BUFFER_SIZE = 200
BUFFER_WARMUP = 50

# OPTIMIZED ANN Config - matches train_all_pairs_optimized_hyperparams.py
# This is the winning config from hyperparameter optimization
OPTIMIZED_ANN_PARAMS = {
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

def train_and_predict_all(pair):
    """Train ANN with rolling window and generate all predictions"""
    print(f"\n{'='*60}")
    print(f"Processing {pair}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Load spreads if available
    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        df = df.join(spread_df[['spread_pct']], how='left')
    except:
        df['spread_pct'] = 0.00025
    df['spread_pct'] = df['spread_pct'].fillna(0.00025)

    # Calculate features
    df = calculate_technical_features(df)
    df['target'] = df['close'].pct_change(1).shift(-1)
    df = df.dropna(subset=TradingConfig.TECHNICAL_FEATURES + ['target'])

    print(f"Data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    predictions = []
    test_indices = []

    # Rolling window training
    for i in range(TRAIN_WINDOW + BUFFER_WARMUP, len(df) - 1):
        train_end = i
        train_start = train_end - TRAIN_WINDOW

        train_data = df.iloc[train_start:train_end]
        test_row = df.iloc[i]

        X_train = train_data[TradingConfig.TECHNICAL_FEATURES].values
        y_train = train_data['target'].values

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = MLPRegressor(**OPTIMIZED_ANN_PARAMS)
        model.fit(X_train_scaled, y_train)

        X_test = test_row[TradingConfig.TECHNICAL_FEATURES].values.reshape(1, -1)
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled)[0]

        predictions.append(pred)
        test_indices.append(i)

        if len(predictions) % 500 == 0:
            print(f"  Processed {len(predictions)} predictions...")

    print(f"  Generated {len(predictions)} predictions")

    return {
        'predictions': predictions,
        'test_indices': test_indices,
        'df': df
    }

def backtest_sleep_well(pair_data):
    """Run Sleep Well backtest on predictions"""
    daily_pnl = {}
    all_trades = []

    for pair in pair_data:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []
        position_exit_idx = -1  # Track when current position exits (per pair)

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

            if len(prediction_buffer) < BUFFER_WARMUP:
                continue

            entry_idx = pred_idx + 1

            # CRITICAL: Skip if we're still in a position for this pair
            if entry_idx <= position_exit_idx:
                continue

            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)

            if prediction >= upper_thresh:
                direction = 1
            elif prediction <= lower_thresh:
                direction = -1
            else:
                continue

            if entry_idx + HOLD_DAYS >= len(df):
                continue

            entry_date = df.index[entry_idx]
            if entry_date.dayofweek >= 5:
                continue

            entry_row = df.iloc[entry_idx]
            entry_price = entry_row['open']
            spread = entry_row['spread_pct']

            if direction == 1:
                actual_entry = entry_price * (1 + spread)
                stop_price = actual_entry * (1 - SL_PCT) if SL_PCT else 0
            else:
                actual_entry = entry_price * (1 - spread)
                stop_price = actual_entry * (1 + SL_PCT) if SL_PCT else float('inf')

            exit_price = None
            exit_reason = 'TIME'
            actual_exit_idx = entry_idx + HOLD_DAYS - 1  # Default time exit

            for hold_day in range(HOLD_DAYS):
                check_idx = entry_idx + hold_day
                if check_idx >= len(df):
                    break

                check_row = df.iloc[check_idx]

                if direction == 1:
                    if SL_PCT and check_row['low'] <= stop_price:
                        exit_price = stop_price * (1 - spread)
                        exit_reason = 'SL'
                        actual_exit_idx = check_idx
                        break
                else:
                    if SL_PCT and check_row['high'] >= stop_price:
                        exit_price = stop_price * (1 + spread)
                        exit_reason = 'SL'
                        actual_exit_idx = check_idx
                        break

            if exit_price is None:
                final_idx = entry_idx + HOLD_DAYS - 1
                if final_idx < len(df):
                    final_row = df.iloc[final_idx]
                    if direction == 1:
                        exit_price = final_row['close'] * (1 - spread)
                    else:
                        exit_price = final_row['close'] * (1 + spread)

            if exit_price is None:
                continue

            # Update position exit index for this pair
            position_exit_idx = actual_exit_idx

            if direction == 1:
                pnl = (exit_price / actual_entry) - 1
            else:
                pnl = (actual_entry / exit_price) - 1

            all_trades.append({
                'pair': pair,
                'date': entry_date,
                'direction': direction,
                'pnl': pnl,
                'exit_reason': exit_reason
            })

            if entry_date not in daily_pnl:
                daily_pnl[entry_date] = 0.0
            daily_pnl[entry_date] += 0.25 * pnl

    return daily_pnl, all_trades

def analyze_results(daily_pnl, all_trades):
    """Analyze and display results"""
    daily_returns = pd.Series(daily_pnl).sort_index()
    leveraged_returns = daily_returns * LEVERAGE

    # Build equity curve
    equity = [1000]
    for ret in leveraged_returns:
        equity.append(equity[-1] * (1 + ret))
    equity = np.array(equity[1:])
    equity_series = pd.Series(equity, index=daily_returns.index)

    # Overall metrics
    total_return = equity[-1] / 1000 - 1
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365.25
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100
    sharpe = (leveraged_returns.mean() / leveraged_returns.std()) * np.sqrt(252)
    cummax = np.maximum.accumulate(equity)
    max_dd = ((equity - cummax) / cummax).min() * 100

    trades_df = pd.DataFrame(all_trades)
    win_rate = (trades_df['pnl'] > 0).mean() * 100

    print(f"\n{'='*70}")
    print("SLEEP WELL STRATEGY - FULL BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Period: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
    print(f"Total Years: {years:.1f}")
    print(f"\nOverall Performance:")
    print(f"  Total Return: {total_return*100:.1f}%")
    print(f"  Annual Return: {annual_return:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total Trades: {len(trades_df)}")

    # Yearly breakdown
    print(f"\n{'='*70}")
    print("YEARLY PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Year':<8} | {'Return':>10} | {'Sharpe':>8} | {'MaxDD':>8} | {'WinRate':>8} | {'Trades':>8}")
    print('-' * 70)

    yearly_results = []
    for year in sorted(daily_returns.index.year.unique()):
        year_mask = daily_returns.index.year == year
        year_returns = leveraged_returns[year_mask]

        if len(year_returns) < 20:
            continue

        year_equity = [1000]
        for ret in year_returns:
            year_equity.append(year_equity[-1] * (1 + ret))
        year_equity = np.array(year_equity[1:])

        year_total = year_equity[-1] / 1000 - 1
        year_sharpe = (year_returns.mean() / year_returns.std()) * np.sqrt(252) if year_returns.std() > 0 else 0
        year_cummax = np.maximum.accumulate(year_equity)
        year_dd = ((year_equity - year_cummax) / year_cummax).min() * 100

        year_trades = trades_df[trades_df['date'].dt.year == year]
        year_wr = (year_trades['pnl'] > 0).mean() * 100 if len(year_trades) > 0 else 0

        yearly_results.append({
            'year': year,
            'return': year_total * 100,
            'sharpe': year_sharpe,
            'max_dd': year_dd,
            'win_rate': year_wr,
            'trades': len(year_trades)
        })

        print(f"{year:<8} | {year_total*100:>9.1f}% | {year_sharpe:>8.2f} | {year_dd:>7.1f}% | {year_wr:>7.1f}% | {len(year_trades):>8}")

    # Monthly stats
    monthly = leveraged_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    print(f"\nMonthly Statistics:")
    print(f"  Average: {monthly.mean()*100:.2f}%")
    print(f"  Median: {monthly.median()*100:.2f}%")
    print(f"  Best: {monthly.max()*100:.2f}%")
    print(f"  Worst: {monthly.min()*100:.2f}%")
    print(f"  Positive: {(monthly > 0).mean()*100:.1f}%")

    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trades_df),
        'years': years,
        'yearly': yearly_results,
        'monthly_avg': monthly.mean() * 100,
        'monthly_median': monthly.median() * 100,
        'monthly_best': monthly.max() * 100,
        'monthly_worst': monthly.min() * 100,
        'monthly_positive': (monthly > 0).mean() * 100
    }

if __name__ == '__main__':
    # Train and generate predictions for all pairs
    pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
    pair_data = {}

    for pair in pairs:
        pair_data[pair] = train_and_predict_all(pair)

    # Save predictions
    print(f"\nSaving predictions...")
    for pair in pairs:
        with open(f'sleep_well_predictions_{pair}.pkl', 'wb') as f:
            pickle.dump(pair_data[pair], f)
    print("Predictions saved!")

    # Run backtest
    daily_pnl, all_trades = backtest_sleep_well(pair_data)

    # Analyze results
    results = analyze_results(daily_pnl, all_trades)

    # Save results for later
    with open('sleep_well_results.pkl', 'wb') as f:
        pickle.dump(results, f)
