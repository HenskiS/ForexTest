"""
Test if market has a directional bias that explains returns.
Uses RANDOM TIMING with 70 trades/year (matching model frequency) but:
1. All LONG positions
2. All SHORT positions
3. 50/50 random (baseline)

This isolates whether EURUSD has been trending up/down vs model adding value.
"""

import pandas as pd
import numpy as np
import pickle
import json

print("MARKET BIAS TEST - DIRECTIONAL TREND ANALYSIS")
print("="*80)
print("Testing if market trend explains returns (70 trades/year, random timing)")
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

with open('xgboost_results_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
TRADES_PER_YEAR = 70  # Match model's approximate trade frequency


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


def generate_random_signals_with_frequency(n_days, trades_per_year, direction='random', seed=None):
    """
    Generate random signals matching a specific trade frequency.

    Args:
        n_days: Number of days in the period
        trades_per_year: Target number of trades per year
        direction: 'long', 'short', or 'random'
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    signals = np.zeros(n_days)

    # Calculate number of trades for this period
    n_years = n_days / 252
    n_trades = int(trades_per_year * n_years)

    # Randomly select days to trade
    trade_days = np.random.choice(n_days, size=min(n_trades, n_days), replace=False)

    # Assign direction
    if direction == 'long':
        signals[trade_days] = 1
    elif direction == 'short':
        signals[trade_days] = -1
    elif direction == 'random':
        signals[trade_days] = np.random.choice([1, -1], size=len(trade_days))

    return signals


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

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values

    # Get ATR for volatility adjustment
    if use_vol_adjustment:
        atr = test_data['atr'].values
        median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]

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
                    'exit_reason': exit_reason
                })

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
            'exit_reason': 'end_of_period'
        })

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
        'exit_reasons': exit_reasons
    }


def backtest_all_windows_market_bias(df_clean, windows, direction='random',
                                    initial_capital=1000.0,
                                    transaction_cost=0.0002, seed=None):
    """Backtest all windows with specific directional bias."""
    capital = initial_capital
    all_backtests = []

    for window_idx, window in enumerate(windows):
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)
        n_days = test_end - test_start

        # Generate signals with target frequency
        signals = generate_random_signals_with_frequency(
            n_days,
            TRADES_PER_YEAR,
            direction=direction,
            seed=seed + window_idx if seed else None
        )

        # Use BEST exit strategy from model
        backtest = backtest_advanced_exits(
            None,
            signals,
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
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count

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
        'final_capital': backtests[-1]['final_capital']
    }


# Generate windows (same as training)
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS, min_windows=40)

print(f"\n{'='*80}")
print(f"Testing market directional bias ({TRADES_PER_YEAR} trades/year)")
print(f"Using BEST exit strategy: Vol-adj (0.40%/1.00%) + 1 day cooldown")
print(f"{'='*80}")

# Test three scenarios
N_SIMULATIONS = 100
scenarios = {
    'All LONG': 'long',
    'All SHORT': 'short',
    '50/50 Random': 'random'
}

results_by_scenario = {}

for scenario_name, direction in scenarios.items():
    print(f"\n{'='*80}")
    print(f"Testing: {scenario_name} ({N_SIMULATIONS} simulations)")
    print(f"{'='*80}")

    simulation_results = []

    for sim in range(N_SIMULATIONS):
        backtests = backtest_all_windows_market_bias(
            df_clean,
            windows,
            direction=direction,
            initial_capital=1000.0,
            transaction_cost=0.0002,
            seed=sim
        )

        metrics = calculate_performance_metrics(backtests)
        if metrics:
            simulation_results.append(metrics)

        if (sim + 1) % 20 == 0:
            print(f"  Completed {sim + 1}/{N_SIMULATIONS} simulations")

    # Calculate statistics
    returns = [r['total_return'] for r in simulation_results]
    annual_returns = [r['annualized_return'] for r in simulation_results]
    sharpes = [r['sharpe_ratio'] for r in simulation_results]
    max_dds = [r['max_drawdown'] for r in simulation_results]
    win_rates = [r['win_rate'] for r in simulation_results]

    prob_profit = np.sum(np.array(returns) > 0) / len(returns)

    results_by_scenario[scenario_name] = {
        'total_return': {
            'median': np.median(returns),
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns)
        },
        'annualized_return': {
            'median': np.median(annual_returns),
            'mean': np.mean(annual_returns),
            'std': np.std(annual_returns)
        },
        'sharpe_ratio': {
            'median': np.median(sharpes),
            'mean': np.mean(sharpes),
            'std': np.std(sharpes)
        },
        'max_drawdown': {
            'median': np.median(max_dds),
            'mean': np.mean(max_dds)
        },
        'win_rate': {
            'median': np.median(win_rates),
            'mean': np.mean(win_rates)
        },
        'prob_profit': prob_profit
    }

# Display comparison
print(f"\n{'='*80}")
print(f"MARKET BIAS COMPARISON")
print(f"{'='*80}")

print(f"\n{'Strategy':<20} {'Annual Return':>15} {'Sharpe':>10} {'Max DD':>10} {'Win Rate':>10} {'Prob Profit':>12}")
print("-" * 95)

for scenario_name in ['All LONG', 'All SHORT', '50/50 Random']:
    r = results_by_scenario[scenario_name]
    print(f"{scenario_name:<20} "
          f"{r['annualized_return']['median']*100:12.2f}%   "
          f"{r['sharpe_ratio']['median']:8.3f}  "
          f"{r['max_drawdown']['median']*100:8.2f}%  "
          f"{r['win_rate']['median']*100:8.2f}%  "
          f"{r['prob_profit']*100:10.1f}%")

# Load model results for comparison
print(f"\n{'='*80}")
print(f"COMPARISON WITH MODEL")
print(f"{'='*80}")

try:
    with open('advanced_exits_results.json', 'r') as f:
        model_results = json.load(f)

    best_model = None
    for r in model_results:
        if '1 day cooldown' in r['config']:
            best_model = r
            break

    if best_model:
        print(f"\nModel Strategy:")
        print(f"  Annual Return:   {best_model['annualized_return']*100:7.2f}%")
        print(f"  Sharpe Ratio:    {best_model['sharpe_ratio']:7.3f}")
        print(f"  Max Drawdown:    {best_model['max_drawdown']*100:7.2f}%")
        print(f"  Win Rate:        {best_model['win_rate']*100:7.2f}%")

        # Compare to best random scenario
        all_long_annual = results_by_scenario['All LONG']['annualized_return']['median']
        all_short_annual = results_by_scenario['All SHORT']['annualized_return']['median']
        random_annual = results_by_scenario['50/50 Random']['annualized_return']['median']

        best_bias_annual = max(all_long_annual, all_short_annual, random_annual)
        best_bias_name = 'All LONG' if all_long_annual == best_bias_annual else ('All SHORT' if all_short_annual == best_bias_annual else '50/50 Random')

        print(f"\nBest Directional Bias ({best_bias_name}):")
        print(f"  Annual Return:   {best_bias_annual*100:7.2f}%")

        print(f"\nModel Improvement over Best Bias:")
        print(f"  {(best_model['annualized_return'] - best_bias_annual)*100:+7.2f}%")
        print(f"  ({best_model['annualized_return'] / best_bias_annual:.2f}x better)")

except FileNotFoundError:
    print("\nNote: Run backtest_advanced_exits.py first to compare against model results")

# Analyze market trend
print(f"\n{'='*80}")
print(f"MARKET TREND ANALYSIS")
print(f"{'='*80}")

long_return = results_by_scenario['All LONG']['annualized_return']['median']
short_return = results_by_scenario['All SHORT']['annualized_return']['median']

print(f"\nAll LONG:  {long_return*100:7.2f}% annual")
print(f"All SHORT: {short_return*100:7.2f}% annual")
print(f"Difference: {(long_return - short_return)*100:+7.2f}%")

if abs(long_return - short_return) < 0.02:  # Within 2%
    print(f"\n>>> Market is NEUTRAL - no strong directional bias")
    print(f">>> Model must be exploiting mean-reversion or volatility patterns")
elif long_return > short_return:
    print(f"\n>>> Market has UPWARD bias (+{(long_return - short_return)*100:.2f}% edge for longs)")
    print(f">>> But model ({best_model['annualized_return']*100:.2f}%) still significantly outperforms")
else:
    print(f"\n>>> Market has DOWNWARD bias (+{(short_return - long_return)*100:.2f}% edge for shorts)")
    print(f">>> But model ({best_model['annualized_return']*100:.2f}%) still significantly outperforms")

# Save results
output_file = 'market_bias_results.json'
with open(output_file, 'w') as f:
    json.dump(results_by_scenario, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print(f"{'='*80}")
