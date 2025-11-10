"""
Monte Carlo simulation of the optimal trading strategy using trade resampling.

This script performs bootstrap resampling of historical trades to estimate:
- Distribution of final returns
- Confidence intervals for key metrics
- Risk of ruin probability
- Robustness to trade sequence
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

print("MONTE CARLO SIMULATION - TRADE RESAMPLING")
print("="*80)

# Load data and results
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

technical_features = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'target_5day_return'
]
df_clean = df.dropna(subset=technical_features)

with open('xgboost_results_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS


def generate_windows(df, window_size, roll_days, min_windows=40):
    """Generate rolling walk-forward windows."""
    windows = []
    start_idx = 0

    while start_idx + window_size <= len(df):
        train_end = start_idx + TRAIN_DAYS
        val_end = train_end + VAL_DAYS
        test_end = val_end + TEST_DAYS

        window = {
            'window_id': len(windows),
            'train_start': start_idx,
            'train_end': train_end,
            'val_start': train_end,
            'val_end': val_end,
            'test_start': val_end,
            'test_end': test_end,
            'date_start': df.index[start_idx],
            'date_end': df.index[test_end - 1]
        }

        windows.append(window)
        start_idx += roll_days

        if len(windows) >= min_windows:
            break

    return windows


def generate_signals_regression(predictions):
    """Generate signals from regression predictions."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals, q1, q3


def backtest_window_with_cooldown(actuals, signals, df_prices, test_indices,
                                   initial_capital,
                                   base_stop_loss_pct=0.0040,
                                   base_take_profit_pct=0.0100,
                                   loss_cooldown_days=1,
                                   transaction_cost_pct=0.0002):
    """Backtest with volatility-adjusted stops and loss cooldown."""
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    cooldown_remaining = 0

    equity_curve = [capital]
    returns = []
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    atr = test_data['atr'].values
    dates = test_data.index

    # Calculate median ATR for normalization
    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        date = dates[i]

        # Decrement cooldown timer
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Adjust stops by volatility
        if not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        # Check stops if we have a position
        if position != 0:
            # Calculate current P&L percentage
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            # Check exits
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check stop-loss
            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Check take-profit
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                exit_reason = 'take_profit'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            if exit_triggered:
                # Calculate P&L
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = entry_capital * pnl_pct
                cost = capital * transaction_cost_pct
                pnl -= cost
                capital += pnl

                trade_return = pnl / (capital - pnl)
                returns.append(trade_return)
                trades.append({
                    'date': date,
                    'type': 'long' if position == 1 else 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                # Set cooldown after losing trade
                if pnl < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0

        # Enter new position (only if not in cooldown)
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital

        equity_curve.append(capital)

    # Close final position
    if position != 0:
        exit_price = closes[-1]

        if position == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl = entry_capital * pnl_pct
        cost = capital * transaction_cost_pct
        pnl -= cost
        capital += pnl

        trade_return = pnl / (capital - pnl)
        returns.append(trade_return)
        trades.append({
            'date': dates[-1],
            'type': 'long' if position == 1 else 'short',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'pnl': pnl,
            'return': trade_return,
            'exit_reason': 'end_of_period'
        })

    final_return = (capital - initial_capital) / initial_capital

    return {
        'trades': trades,
        'returns': returns,
        'final_capital': capital,
        'final_return': final_return
    }


def resample_trades_monte_carlo(trades_df, n_simulations=10000, initial_capital=1000.0):
    """
    Monte Carlo simulation using bootstrap resampling of historical trades.

    For each simulation:
    1. Randomly sample trades with replacement (same number as original)
    2. Compound returns in the order sampled
    3. Calculate final capital, drawdown, Sharpe, etc.
    """
    print(f"\nRunning {n_simulations:,} Monte Carlo simulations...")
    print("Resampling {0} historical trades with replacement\n".format(len(trades_df)))

    n_trades = len(trades_df)
    trade_returns = trades_df['return'].values

    results = {
        'final_capital': [],
        'total_return': [],
        'max_drawdown': [],
        'sharpe_ratio': [],
        'win_rate': [],
        'total_trades': [],
        'winning_trades': [],
        'losing_trades': []
    }

    for sim in range(n_simulations):
        if (sim + 1) % 1000 == 0:
            print(f"Completed {sim + 1:,} / {n_simulations:,} simulations...")

        # Resample trades with replacement
        sampled_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        sampled_returns = trade_returns[sampled_indices]
        sampled_trades = trades_df.iloc[sampled_indices]

        # Compound returns
        capital = initial_capital
        equity_curve = [capital]
        for ret in sampled_returns:
            capital *= (1 + ret)
            equity_curve.append(capital)

        equity_curve = np.array(equity_curve)

        # Calculate metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital

        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdown)

        # Sharpe ratio (annualized)
        returns_arr = np.array(sampled_returns)
        if len(returns_arr) > 0:
            mean_return = np.mean(returns_arr)
            std_return = np.std(returns_arr, ddof=1)
            if std_return > 0:
                # Annualize assuming ~85 trades per year
                trades_per_year = 85
                sharpe = (mean_return / std_return) * np.sqrt(trades_per_year)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Win rate
        winning = sampled_trades[sampled_trades['pnl'] > 0]
        win_rate = len(winning) / len(sampled_trades) if len(sampled_trades) > 0 else 0

        # Store results
        results['final_capital'].append(final_capital)
        results['total_return'].append(total_return)
        results['max_drawdown'].append(max_dd)
        results['sharpe_ratio'].append(sharpe)
        results['win_rate'].append(win_rate)
        results['total_trades'].append(len(sampled_trades))
        results['winning_trades'].append(len(winning))
        results['losing_trades'].append(len(sampled_trades) - len(winning))

    return results


def calculate_statistics(results, initial_capital=1000.0):
    """Calculate comprehensive statistics from Monte Carlo results."""
    stats = {}

    for metric, values in results.items():
        values_arr = np.array(values)
        stats[metric] = {
            'mean': np.mean(values_arr),
            'median': np.median(values_arr),
            'std': np.std(values_arr),
            'min': np.min(values_arr),
            'max': np.max(values_arr),
            'p5': np.percentile(values_arr, 5),
            'p25': np.percentile(values_arr, 25),
            'p75': np.percentile(values_arr, 75),
            'p95': np.percentile(values_arr, 95)
        }

    # Calculate risk of ruin (probability of losing >50% of capital)
    final_capitals = np.array(results['final_capital'])
    risk_of_ruin = np.mean(final_capitals < initial_capital * 0.5) * 100

    # Probability of positive returns
    prob_profit = np.mean(np.array(results['total_return']) > 0) * 100

    # Probability of beating buy-and-hold (assume ~5% annual for 20 years = ~165% total)
    buyhold_return = 1.65
    prob_beat_buyhold = np.mean(np.array(results['total_return']) > buyhold_return) * 100

    stats['risk_of_ruin'] = risk_of_ruin
    stats['prob_profit'] = prob_profit
    stats['prob_beat_buyhold'] = prob_beat_buyhold

    return stats


# Generate windows and run original backtest to get trades
print("Generating baseline trades from optimal strategy...")
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=40)

capital = 1000.0
all_trades = []

for window_idx, result in enumerate(all_results):
    window = windows[window_idx]
    test_start = window['test_start']
    test_end = window['test_end']
    test_indices = range(test_start, test_end)

    signals, q1, q3 = generate_signals_regression(result['predictions'])

    backtest = backtest_window_with_cooldown(
        result['actuals'],
        signals,
        df_clean,
        test_indices,
        capital,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002
    )

    all_trades.extend(backtest['trades'])
    capital = backtest['final_capital']

# Convert trades to DataFrame
trades_df = pd.DataFrame(all_trades)
print(f"Baseline: {len(trades_df)} trades, Final capital: ${capital:.2f}")

# Run Monte Carlo simulation
mc_results = resample_trades_monte_carlo(trades_df, n_simulations=10000, initial_capital=1000.0)

# Calculate statistics
print("\n" + "="*80)
print("CALCULATING STATISTICS...")
stats = calculate_statistics(mc_results, initial_capital=1000.0)

# Print results
print("\n" + "="*80)
print("MONTE CARLO RESULTS - TRADE RESAMPLING (10,000 simulations)")
print("="*80)

print("\n1. FINAL CAPITAL DISTRIBUTION")
print("-" * 80)
fc_stats = stats['final_capital']
print(f"Mean:              ${fc_stats['mean']:,.2f}")
print(f"Median:            ${fc_stats['median']:,.2f}")
print(f"Std Dev:           ${fc_stats['std']:,.2f}")
print(f"Min:               ${fc_stats['min']:,.2f}")
print(f"Max:               ${fc_stats['max']:,.2f}")
print(f"\nConfidence Intervals:")
print(f"5th percentile:    ${fc_stats['p5']:,.2f}  (5% chance of ending below this)")
print(f"25th percentile:   ${fc_stats['p25']:,.2f}")
print(f"75th percentile:   ${fc_stats['p75']:,.2f}")
print(f"95th percentile:   ${fc_stats['p95']:,.2f}  (5% chance of ending above this)")

print("\n2. TOTAL RETURN DISTRIBUTION")
print("-" * 80)
tr_stats = stats['total_return']
print(f"Mean:              {tr_stats['mean']*100:.2f}%")
print(f"Median:            {tr_stats['median']*100:.2f}%")
print(f"Std Dev:           {tr_stats['std']*100:.2f}%")
print(f"Min:               {tr_stats['min']*100:.2f}%")
print(f"Max:               {tr_stats['max']*100:.2f}%")
print(f"\nConfidence Intervals:")
print(f"5th percentile:    {tr_stats['p5']*100:.2f}%")
print(f"25th percentile:   {tr_stats['p25']*100:.2f}%")
print(f"75th percentile:   {tr_stats['p75']*100:.2f}%")
print(f"95th percentile:   {tr_stats['p95']*100:.2f}%")

print("\n3. MAXIMUM DRAWDOWN DISTRIBUTION")
print("-" * 80)
dd_stats = stats['max_drawdown']
print(f"Mean:              {dd_stats['mean']*100:.2f}%")
print(f"Median:            {dd_stats['median']*100:.2f}%")
print(f"Std Dev:           {dd_stats['std']*100:.2f}%")
print(f"Best (smallest):   {dd_stats['min']*100:.2f}%")
print(f"Worst (largest):   {dd_stats['max']*100:.2f}%")
print(f"\nConfidence Intervals:")
print(f"5th percentile:    {dd_stats['p5']*100:.2f}%  (5% chance of drawdown worse than this)")
print(f"25th percentile:   {dd_stats['p25']*100:.2f}%")
print(f"75th percentile:   {dd_stats['p75']*100:.2f}%")
print(f"95th percentile:   {dd_stats['p95']*100:.2f}%")

print("\n4. SHARPE RATIO DISTRIBUTION")
print("-" * 80)
sr_stats = stats['sharpe_ratio']
print(f"Mean:              {sr_stats['mean']:.3f}")
print(f"Median:            {sr_stats['median']:.3f}")
print(f"Std Dev:           {sr_stats['std']:.3f}")
print(f"Min:               {sr_stats['min']:.3f}")
print(f"Max:               {sr_stats['max']:.3f}")
print(f"\nConfidence Intervals:")
print(f"5th percentile:    {sr_stats['p5']:.3f}")
print(f"25th percentile:   {sr_stats['p25']:.3f}")
print(f"75th percentile:   {sr_stats['p75']:.3f}")
print(f"95th percentile:   {sr_stats['p95']:.3f}")

print("\n5. WIN RATE DISTRIBUTION")
print("-" * 80)
wr_stats = stats['win_rate']
print(f"Mean:              {wr_stats['mean']*100:.2f}%")
print(f"Median:            {wr_stats['median']*100:.2f}%")
print(f"Std Dev:           {wr_stats['std']*100:.2f}%")
print(f"Min:               {wr_stats['min']*100:.2f}%")
print(f"Max:               {wr_stats['max']*100:.2f}%")
print(f"\nConfidence Intervals:")
print(f"5th percentile:    {wr_stats['p5']*100:.2f}%")
print(f"25th percentile:   {wr_stats['p25']*100:.2f}%")
print(f"75th percentile:   {wr_stats['p75']*100:.2f}%")
print(f"95th percentile:   {wr_stats['p95']*100:.2f}%")

print("\n6. RISK METRICS")
print("-" * 80)
print(f"Risk of Ruin (>50% loss):     {stats['risk_of_ruin']:.2f}%")
print(f"Probability of Profit:        {stats['prob_profit']:.2f}%")
print(f"Prob. of Beating Buy-Hold:    {stats['prob_beat_buyhold']:.2f}%")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"""
1. Central Tendency: The median final capital is ${stats['final_capital']['median']:,.2f},
   meaning 50% of simulations end above this level.

2. Worst Case (5th percentile): There's a 95% chance you'll end with more than
   ${stats['final_capital']['p5']:,.2f} (a {stats['total_return']['p5']*100:.1f}% return).

3. Best Case (95th percentile): There's a 95% chance you'll end with less than
   ${stats['final_capital']['p95']:,.2f} (a {stats['total_return']['p95']*100:.1f}% return).

4. Risk of Ruin: Only {stats['risk_of_ruin']:.2f}% chance of losing >50% of capital.

5. Profitability: {stats['prob_profit']:.1f}% of simulations end with positive returns.

6. Consistency: The strategy shows {"high" if stats['prob_profit'] > 90 else "moderate"} consistency across different trade sequences.
""")

# Save results
output_file = 'monte_carlo_results.json'
print(f"\nSaving results to {output_file}...")

output = {
    'simulation_config': {
        'n_simulations': 10000,
        'initial_capital': 1000.0,
        'n_trades': len(trades_df),
        'strategy': 'Vol-adj (0.40%/1.00%) + 1 day cooldown'
    },
    'statistics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats.items() if isinstance(v, dict)},
    'risk_metrics': {
        'risk_of_ruin': float(stats['risk_of_ruin']),
        'prob_profit': float(stats['prob_profit']),
        'prob_beat_buyhold': float(stats['prob_beat_buyhold'])
    }
}

with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {output_file}")
print("="*80)
