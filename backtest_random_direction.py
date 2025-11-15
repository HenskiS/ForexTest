"""
Test if returns come from model predictions or just the exit strategy.
Uses EXACT SAME TRADE TIMING as model (same days, same number of trades),
but RANDOMIZES the DIRECTION (50/50 long/short).

This isolates whether the model is good at:
1. Timing (when to trade) vs
2. Direction (which way to trade)
"""

import pandas as pd
import numpy as np
import pickle
import json

print("RANDOM DIRECTION BACKTEST - MODEL ATTRIBUTION TEST")
print("="*80)
print("Same trade timing as model, but random direction (50/50 long/short)")
print("="*80)

# Load data
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

with open('xgboost_results_EURUSD_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

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
    """Generate signals from regression predictions (model approach)."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    signals[predictions >= q3] = 1
    signals[predictions <= q1] = -1

    return signals, q1, q3


def randomize_signal_direction(signals, seed=None):
    """Take existing signals and randomize only the direction (1 vs -1)."""
    if seed is not None:
        np.random.seed(seed)

    random_signals = np.copy(signals)

    # Find where we have signals (non-zero)
    trade_mask = signals != 0

    # Randomly assign 50/50 long (+1) or short (-1)
    n_trades = np.sum(trade_mask)
    random_directions = np.random.choice([1, -1], size=n_trades)

    random_signals[trade_mask] = random_directions

    return random_signals


def backtest_advanced_exits(actuals, signals, df_prices, test_indices,
                           initial_capital,
                           base_stop_loss_pct=0.0040,
                           base_take_profit_pct=0.0100,
                           use_vol_adjustment=True,
                           vol_window=20,
                           use_trailing_stop=False,
                           trail_activate_pct=0.0075,
                           trail_distance_pct=0.0025,
                           loss_cooldown_days=1,
                           transaction_cost_pct=0.0002):
    """
    EXACT SAME backtest function as backtest_advanced_exits.py
    """
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    trailing_stop_price = None
    cooldown_remaining = 0

    equity_curve = [capital]
    returns = []
    trades = []
    yearly_returns = {}

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    dates = test_data.index

    # Get ATR for volatility adjustment
    if use_vol_adjustment:
        atr = test_data['atr'].values
        median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        current_date = dates[i]
        current_year = current_date.year

        # Decrement cooldown timer
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Adjust stops by volatility if enabled
        if use_vol_adjustment and not np.isnan(atr[i]):
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
                current_price_for_trail = high_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price
                current_price_for_trail = low_price

            # Update trailing stop if enabled and activated
            if use_trailing_stop and pct_high >= trail_activate_pct:
                if trailing_stop_price is None:
                    trailing_stop_price = entry_price
                else:
                    if position == 1:
                        new_stop = current_price_for_trail - (entry_price * trail_distance_pct)
                        trailing_stop_price = max(trailing_stop_price, new_stop)
                    else:
                        new_stop = current_price_for_trail + (entry_price * trail_distance_pct)
                        trailing_stop_price = min(trailing_stop_price, new_stop)

            # Check exits
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check trailing stop first (if active)
            if use_trailing_stop and trailing_stop_price is not None:
                if position == 1 and low_price <= trailing_stop_price:
                    exit_triggered = True
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'
                elif position == -1 and high_price >= trailing_stop_price:
                    exit_triggered = True
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'

            # Check regular stop-loss
            if not exit_triggered and pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'stop_loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Check take-profit
            elif not exit_triggered and pct_high >= take_profit_pct:
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
                    'type': 'long' if position == 1 else 'short',
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'return': trade_return,
                    'exit_reason': exit_reason,
                    'year': current_year
                })

                # Track by year
                if current_year not in yearly_returns:
                    yearly_returns[current_year] = []
                yearly_returns[current_year].append(trade_return)

                # Set cooldown after losing trade
                if pnl < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0
                trailing_stop_price = None

        # Enter new position (only if not in cooldown)
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            cost = capital * transaction_cost_pct
            capital -= cost
            entry_price = open_price
            position = signal
            entry_capital = capital
            trailing_stop_price = None

        equity_curve.append(capital)

    # Close final position
    if position != 0:
        exit_price = closes[-1]
        current_year = dates[-1].year

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
            'type': 'long' if position == 1 else 'short',
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'pnl': pnl,
            'return': trade_return,
            'exit_reason': 'end_of_period',
            'year': current_year
        })

        if current_year not in yearly_returns:
            yearly_returns[current_year] = []
        yearly_returns[current_year].append(trade_return)

    final_return = (capital - initial_capital) / initial_capital
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    exit_reasons = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'strategy_returns': returns,
        'final_capital': capital,
        'final_return': final_return,
        'n_trades': len(trades),
        'n_long': len([t for t in trades if t['type'] == 'long']),
        'n_short': len([t for t in trades if t['type'] == 'short']),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'exit_reasons': exit_reasons,
        'yearly_returns': yearly_returns
    }


def backtest_all_windows_random_direction(results, df_clean, windows, initial_capital=1000.0,
                                         transaction_cost=0.0002, seed=None):
    """Backtest all windows with RANDOM DIRECTION but same timing as model."""
    capital = initial_capital
    all_backtests = []

    # Use only the number of windows available
    n_windows = min(len(results), len(windows))

    for window_idx in range(n_windows):
        result = results[window_idx]
        test_start = windows[window_idx]['test_start']
        test_end = windows[window_idx]['test_end']
        test_indices = range(test_start, test_end)

        # Generate model signals (to get timing)
        model_signals, q1, q3 = generate_signals_regression(result['predictions'])

        # Randomize ONLY the direction, keep timing the same
        random_signals = randomize_signal_direction(model_signals, seed=seed + window_idx if seed else None)

        # Use BEST exit strategy from model
        backtest = backtest_advanced_exits(
            result['actuals'],
            random_signals,
            df_clean,
            test_indices,
            capital,
            base_stop_loss_pct=0.0040,
            base_take_profit_pct=0.0100,
            use_vol_adjustment=True,
            vol_window=20,
            use_trailing_stop=False,
            loss_cooldown_days=1,
            transaction_cost_pct=transaction_cost
        )
        all_backtests.append(backtest)
        capital = backtest['final_capital']

    return all_backtests


def calculate_performance_metrics(backtests):
    """Calculate comprehensive performance metrics."""
    all_returns = []
    for bt in backtests:
        all_returns.extend(bt['strategy_returns'])

    if len(all_returns) == 0:
        return None

    all_returns = np.array(all_returns)

    equity_curve = np.cumprod(1 + all_returns)
    total_return = equity_curve[-1] - 1

    n_days = len(backtests) * TEST_DAYS
    n_years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    volatility = np.std(all_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown)

    winning_trades = all_returns[all_returns > 0]
    losing_trades = all_returns[all_returns < 0]

    win_rate = len(winning_trades) / len(all_returns[all_returns != 0]) if len(all_returns[all_returns != 0]) > 0 else 0
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0

    total_wins = np.sum(winning_trades)
    total_losses = np.abs(np.sum(losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    all_exit_reasons = {}
    for bt in backtests:
        for reason, count in bt['exit_reasons'].items():
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + 1

    # Aggregate yearly returns
    yearly_performance = {}
    for bt in backtests:
        for year, year_returns in bt['yearly_returns'].items():
            if year not in yearly_performance:
                yearly_performance[year] = []
            yearly_performance[year].extend(year_returns)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'n_trades': len(all_returns[all_returns != 0]),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'exit_reasons': all_exit_reasons,
        'final_capital': backtests[-1]['final_capital'],
        'yearly_performance': yearly_performance
    }


# Generate windows (same as training)
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=40)

print(f"\n{'='*80}")
print(f"Running {len(windows)} windows with RANDOM DIRECTION")
print(f"Using BEST exit strategy: Vol-adj (0.40%/1.00%) + 1 day cooldown")
print(f"{'='*80}")

# Run multiple simulations with different random seeds
N_SIMULATIONS = 100
print(f"\nRunning {N_SIMULATIONS} simulations...")

simulation_results = []

for sim in range(N_SIMULATIONS):
    backtests = backtest_all_windows_random_direction(
        all_results,
        df_clean,
        windows,
        initial_capital=1000.0,
        transaction_cost=0.0002,
        seed=sim
    )

    metrics = calculate_performance_metrics(backtests)
    if metrics:
        simulation_results.append(metrics)

    if (sim + 1) % 10 == 0:
        print(f"  Completed {sim + 1}/{N_SIMULATIONS} simulations")

print(f"\nCompleted all {N_SIMULATIONS} simulations")

# Calculate statistics across simulations
returns = [r['total_return'] for r in simulation_results]
annual_returns = [r['annualized_return'] for r in simulation_results]
sharpes = [r['sharpe_ratio'] for r in simulation_results]
max_dds = [r['max_drawdown'] for r in simulation_results]
win_rates = [r['win_rate'] for r in simulation_results]
profit_factors = [r['profit_factor'] for r in simulation_results]

print(f"\n{'='*80}")
print(f"RANDOM DIRECTION RESULTS (100 simulations)")
print(f"{'='*80}")

print(f"\nTotal Return:")
print(f"  Median:  {np.median(returns)*100:7.2f}%")
print(f"  Mean:    {np.mean(returns)*100:7.2f}%")
print(f"  Std Dev: {np.std(returns)*100:7.2f}%")
print(f"  Min:     {np.min(returns)*100:7.2f}%")
print(f"  Max:     {np.max(returns)*100:7.2f}%")
print(f"  95% CI:  [{np.percentile(returns, 2.5)*100:7.2f}%, {np.percentile(returns, 97.5)*100:7.2f}%]")

print(f"\nAnnualized Return:")
print(f"  Median:  {np.median(annual_returns)*100:7.2f}%")
print(f"  Mean:    {np.mean(annual_returns)*100:7.2f}%")
print(f"  Std Dev: {np.std(annual_returns)*100:7.2f}%")
print(f"  95% CI:  [{np.percentile(annual_returns, 2.5)*100:7.2f}%, {np.percentile(annual_returns, 97.5)*100:7.2f}%]")

print(f"\nSharpe Ratio:")
print(f"  Median:  {np.median(sharpes):7.3f}")
print(f"  Mean:    {np.mean(sharpes):7.3f}")
print(f"  Std Dev: {np.std(sharpes):7.3f}")
print(f"  95% CI:  [{np.percentile(sharpes, 2.5):7.3f}, {np.percentile(sharpes, 97.5):7.3f}]")

print(f"\nMax Drawdown:")
print(f"  Median:  {np.median(max_dds)*100:7.2f}%")
print(f"  Mean:    {np.mean(max_dds)*100:7.2f}%")
print(f"  Worst:   {np.min(max_dds)*100:7.2f}%")

print(f"\nWin Rate:")
print(f"  Median:  {np.median(win_rates)*100:7.2f}%")
print(f"  Mean:    {np.mean(win_rates)*100:7.2f}%")

print(f"\nProfit Factor:")
print(f"  Median:  {np.median(profit_factors):7.3f}")
print(f"  Mean:    {np.mean(profit_factors):7.3f}")

# Probability of profit
prob_profit = np.sum(np.array(returns) > 0) / len(returns)
print(f"\nProbability of Profit: {prob_profit*100:.1f}%")

# Aggregate yearly returns across all simulations
all_yearly_returns = {}
for sim_result in simulation_results:
    for year, year_returns in sim_result['yearly_performance'].items():
        if year not in all_yearly_returns:
            all_yearly_returns[year] = []
        # Calculate annual return for this year in this simulation
        annual_ret = (np.prod(1 + np.array(year_returns)) - 1) if len(year_returns) > 0 else 0
        all_yearly_returns[year].append(annual_ret)

# Load model results for comparison
print(f"\n{'='*80}")
print(f"COMPARISON: RANDOM DIRECTION vs MODEL")
print(f"{'='*80}")

# Load advanced_exits_results.json to get model performance
try:
    with open('advanced_exits_results.json', 'r') as f:
        model_results = json.load(f)

    # Find the best configuration (Vol-adj + 1 day cooldown)
    best_model = None
    for r in model_results:
        if '1 day cooldown' in r['config']:
            best_model = r
            break

    if best_model:
        print(f"\nModel Strategy (Vol-adj 0.40%/1.00% + 1 day cooldown):")
        print(f"  Total Return:        {best_model['total_return']*100:7.2f}%")
        print(f"  Annualized Return:   {best_model['annualized_return']*100:7.2f}%")
        print(f"  Sharpe Ratio:        {best_model['sharpe_ratio']:7.3f}")
        print(f"  Max Drawdown:        {best_model['max_drawdown']*100:7.2f}%")
        print(f"  Win Rate:            {best_model['win_rate']*100:7.2f}%")
        print(f"  Profit Factor:       {best_model['profit_factor']:7.3f}")

        print(f"\nRandom Direction Strategy (median of 100 simulations):")
        print(f"  Total Return:        {np.median(returns)*100:7.2f}%")
        print(f"  Annualized Return:   {np.median(annual_returns)*100:7.2f}%")
        print(f"  Sharpe Ratio:        {np.median(sharpes):7.3f}")
        print(f"  Max Drawdown:        {np.median(max_dds)*100:7.2f}%")
        print(f"  Win Rate:            {np.median(win_rates)*100:7.2f}%")
        print(f"  Profit Factor:       {np.median(profit_factors):7.3f}")

        print(f"\n{'='*80}")
        print(f"MODEL VALUE-ADD:")
        print(f"{'='*80}")
        print(f"  Return Improvement:  {(best_model['total_return'] - np.median(returns))*100:+7.2f}%")
        print(f"  Annual Improvement:  {(best_model['annualized_return'] - np.median(annual_returns))*100:+7.2f}%")
        print(f"  Sharpe Improvement:  {(best_model['sharpe_ratio'] - np.median(sharpes)):+7.3f}")
        print(f"  Drawdown Reduction:  {(best_model['max_drawdown'] - np.median(max_dds))*100:+7.2f}%")

        # Calculate how many standard deviations the model is above random
        model_sharpe_zscore = (best_model['sharpe_ratio'] - np.mean(sharpes)) / np.std(sharpes)
        print(f"\n  Model Sharpe is {model_sharpe_zscore:.2f} std devs above random mean")

        if model_sharpe_zscore > 2:
            print(f"  [STATISTICALLY SIGNIFICANT] (p < 0.05)")
        else:
            print(f"  [NOT STATISTICALLY SIGNIFICANT]")

except FileNotFoundError:
    print("\nNote: Run backtest_advanced_exits.py first to compare against model results")

# Display yearly comparison
print(f"\n{'='*80}")
print(f"YEARLY PERFORMANCE COMPARISON")
print(f"{'='*80}")

# Get model yearly performance
try:
    # Run model backtest to get yearly returns
    from backtest_advanced_exits import backtest_all_windows_advanced

    config = {
        'name': 'Vol-adj (0.40%/1.00%) + 1 day cooldown',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'use_vol_adjustment': True,
        'vol_window': 20,
        'use_trailing_stop': False,
        'loss_cooldown_days': 1
    }

    model_backtests = backtest_all_windows_advanced(
        all_results,
        df_clean,
        windows,
        config,
        initial_capital=1000.0,
        transaction_cost=0.0002
    )

    # Get model yearly returns
    model_yearly_returns = {}
    for bt in model_backtests:
        for year, year_returns in bt['yearly_returns'].items():
            if year not in model_yearly_returns:
                model_yearly_returns[year] = []
            model_yearly_returns[year].extend(year_returns)

    # Calculate annual returns
    print(f"\n{'Year':<8} {'Model':>12} {'Random (median)':>18} {'Random (mean)':>16} {'Random (std)':>15} {'N Trades':>10}")
    print("-" * 95)

    for year in sorted(all_yearly_returns.keys()):
        if year in model_yearly_returns and len(model_yearly_returns[year]) > 0:
            model_annual = (np.prod(1 + np.array(model_yearly_returns[year])) - 1) * 100

            random_annuals = all_yearly_returns[year]
            random_median = np.median(random_annuals) * 100
            random_mean = np.mean(random_annuals) * 100
            random_std = np.std(random_annuals) * 100
            n_trades = len(model_yearly_returns[year])

            print(f"{year:<8} {model_annual:11.2f}% {random_median:17.2f}% {random_mean:15.2f}% {random_std:14.2f}% {n_trades:10d}")

except Exception as e:
    print(f"\nCould not calculate yearly comparison: {e}")

# Save results
output_file = 'random_direction_results.json'
results_json = {
    'n_simulations': N_SIMULATIONS,
    'summary_stats': {
        'total_return': {
            'median': float(np.median(returns)),
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
            'min': float(np.min(returns)),
            'max': float(np.max(returns))
        },
        'annualized_return': {
            'median': float(np.median(annual_returns)),
            'mean': float(np.mean(annual_returns)),
            'std': float(np.std(annual_returns))
        },
        'sharpe_ratio': {
            'median': float(np.median(sharpes)),
            'mean': float(np.mean(sharpes)),
            'std': float(np.std(sharpes))
        },
        'max_drawdown': {
            'median': float(np.median(max_dds)),
            'mean': float(np.mean(max_dds)),
            'worst': float(np.min(max_dds))
        },
        'win_rate': {
            'median': float(np.median(win_rates)),
            'mean': float(np.mean(win_rates))
        },
        'probability_of_profit': float(prob_profit)
    }
}

with open(output_file, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}")
