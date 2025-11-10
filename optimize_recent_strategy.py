"""
Optimize strategy specifically for recent market conditions (2021-2025).

Tests various configurations to find what works best in the current regime.
"""

import pandas as pd
import numpy as np
import pickle
import json

print("RECENT REGIME OPTIMIZATION (2021-2025)")
print("="*80)

# Load data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)

# Filter to 2021+
df = df[df.index >= '2021-01-01']

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

# Load model results
with open('xgboost_results_recent_2021.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"\nLoaded {len(all_results)} windows from 2021-present")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS


def generate_windows(df, window_size, roll_days):
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

    return windows


def generate_signals_regression(predictions, lower_percentile=25, upper_percentile=75):
    """Generate signals from regression predictions."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_thresh = np.percentile(predictions, lower_percentile)
    upper_thresh = np.percentile(predictions, upper_percentile)
    signals[predictions >= upper_thresh] = 1
    signals[predictions <= lower_thresh] = -1

    return signals, lower_thresh, upper_thresh


def backtest_window_with_cooldown(actuals, signals, df_prices, test_indices,
                                   initial_capital,
                                   base_stop_loss_pct=0.0040,
                                   base_take_profit_pct=0.0100,
                                   loss_cooldown_days=1,
                                   use_vol_adjustment=True,
                                   transaction_cost_pct=0.0002):
    """Backtest with configurable parameters."""
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
        'equity_curve': equity_curve,
        'final_capital': capital,
        'final_return': final_return
    }


def backtest_all_windows(results, df_prices, windows, config):
    """Backtest all windows with given configuration."""
    capital = 1000.0
    all_trades = []

    for window_idx, result in enumerate(results):
        window = windows[window_idx]
        test_start = window['test_start']
        test_end = window['test_end']
        test_indices = range(test_start, test_end)

        signals, _, _ = generate_signals_regression(
            result['predictions'],
            lower_percentile=config.get('lower_percentile', 25),
            upper_percentile=config.get('upper_percentile', 75)
        )

        backtest = backtest_window_with_cooldown(
            result['actuals'],
            signals,
            df_prices,
            test_indices,
            capital,
            base_stop_loss_pct=config['base_stop_loss_pct'],
            base_take_profit_pct=config['base_take_profit_pct'],
            loss_cooldown_days=config.get('loss_cooldown_days', 0),
            use_vol_adjustment=config.get('use_vol_adjustment', True),
            transaction_cost_pct=0.0002
        )

        all_trades.extend(backtest['trades'])
        capital = backtest['final_capital']

    return all_trades, capital


# Generate windows
windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)

# Test configurations optimized for recent regime
test_configs = [
    # Baseline (historical optimal)
    {
        'name': 'Historical Optimal',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },

    # Tighter signal thresholds (higher confidence)
    {
        'name': 'High Confidence (10/90)',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 10,
        'upper_percentile': 90
    },
    {
        'name': 'Very High Confidence (5/95)',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 5,
        'upper_percentile': 95
    },

    # Wider stops for choppy markets
    {
        'name': 'Wider Stops (0.50%/1.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },
    {
        'name': 'Much Wider Stops (0.60%/1.50%)',
        'base_stop_loss_pct': 0.0060,
        'base_take_profit_pct': 0.0150,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },

    # Longer cooldowns for choppier markets
    {
        'name': '2 Day Cooldown',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 2,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },
    {
        'name': '3 Day Cooldown',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 3,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },

    # Tighter take-profit (grab profits faster)
    {
        'name': 'Tighter TP (0.40%/0.80%)',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0080,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 25,
        'upper_percentile': 75
    },

    # Combined: High confidence + wider stops
    {
        'name': 'High Conf (10/90) + Wider (0.50%/1.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': True,
        'lower_percentile': 10,
        'upper_percentile': 90
    },

    # Combined: High confidence + longer cooldown
    {
        'name': 'High Conf (10/90) + 2 Day Cooldown',
        'base_stop_loss_pct': 0.0040,
        'base_take_profit_pct': 0.0100,
        'loss_cooldown_days': 2,
        'use_vol_adjustment': True,
        'lower_percentile': 10,
        'upper_percentile': 90
    },

    # No vol adjustment (fixed stops)
    {
        'name': 'Fixed Stops (0.50%/1.25%)',
        'base_stop_loss_pct': 0.0050,
        'base_take_profit_pct': 0.0125,
        'loss_cooldown_days': 1,
        'use_vol_adjustment': False,
        'lower_percentile': 25,
        'upper_percentile': 75
    },
]

print("\n" + "="*80)
print("TESTING CONFIGURATIONS ON RECENT DATA (2021-2025)")
print("="*80)

results_summary = []

for config in test_configs:
    print(f"\n{'-'*80}")
    print(f"Testing: {config['name']}")
    print(f"  Stop/TP: {config['base_stop_loss_pct']*100:.2f}%/{config['base_take_profit_pct']*100:.2f}%")
    print(f"  Signal thresholds: {config.get('lower_percentile', 25)}/{config.get('upper_percentile', 75)} percentile")
    print(f"  Cooldown: {config.get('loss_cooldown_days', 0)} days")
    print(f"  Vol adjustment: {config.get('use_vol_adjustment', True)}")
    print(f"{'-'*80}")

    all_trades, final_capital = backtest_all_windows(
        all_results,
        df_clean,
        windows,
        config
    )

    trades_df = pd.DataFrame(all_trades)
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    # Calculate metrics
    total_return = (final_capital - 1000) / 1000
    years = (windows[-1]['date_end'] - windows[0]['date_start']).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1

    if len(trades_df) > 0:
        returns_arr = np.array(trades_df['return'].values)
        sharpe = (np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(total_trades / years)

        win_rate = len(winning_trades) / total_trades

        # Max drawdown
        equity_curve = [1000.0]
        capital_running = 1000.0
        for ret in trades_df['return']:
            capital_running *= (1 + ret)
            equity_curve.append(capital_running)

        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        max_dd = np.min(drawdown)

        # Profit factor
        total_wins = winning_trades['pnl'].sum()
        total_losses = abs(losing_trades['pnl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        print(f"\nResults:")
        print(f"  Total Return: {total_return*100:.2f}%")
        print(f"  Annualized Return: {annual_return*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Total Trades: {total_trades} ({total_trades/years:.0f}/year)")
        print(f"  Profit Factor: {profit_factor:.3f}")
        print(f"  Final Capital: ${final_capital:.2f}")

        results_summary.append({
            'config': config['name'],
            'total_return': total_return,
            'annualized_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'trades_per_year': total_trades / years,
            'profit_factor': profit_factor,
            'final_capital': final_capital,
            'base_stop_loss_pct': config['base_stop_loss_pct'],
            'base_take_profit_pct': config['base_take_profit_pct'],
            'loss_cooldown_days': config.get('loss_cooldown_days', 0),
            'use_vol_adjustment': config.get('use_vol_adjustment', True),
            'lower_percentile': config.get('lower_percentile', 25),
            'upper_percentile': config.get('upper_percentile', 75)
        })

# Sort by Sharpe ratio
results_summary_sorted = sorted(results_summary, key=lambda x: x['sharpe_ratio'], reverse=True)

# Save results
output_file = 'recent_regime_optimization_results.json'
with open(output_file, 'w') as f:
    json.dump(results_summary_sorted, f, indent=2)

# Print summary
print("\n" + "="*80)
print("SUMMARY - RANKED BY SHARPE RATIO")
print("="*80)
print(f"{'Rank':<5} {'Configuration':<45} {'Sharpe':<8} {'Annual':<8} {'Total':<10} {'MaxDD':<8} {'WinRate':<9} {'Trades':<7}")
print("-"*110)

for i, result in enumerate(results_summary_sorted):
    print(f"{i+1:<5} {result['config']:<45} {result['sharpe_ratio']:>6.3f}  {result['annualized_return']*100:>6.2f}% {result['total_return']*100:>8.2f}% {result['max_drawdown']*100:>6.2f}% {result['win_rate']*100:>7.1f}% {result['total_trades']:>6}")

# Best configuration
best = results_summary_sorted[0]
print("\n" + "="*80)
print("BEST CONFIGURATION FOR RECENT REGIME (2021-2025)")
print("="*80)
print(f"Configuration: {best['config']}")
print(f"  Stop/TP: {best['base_stop_loss_pct']*100:.2f}%/{best['base_take_profit_pct']*100:.2f}%")
print(f"  Signal thresholds: {best['lower_percentile']}/{best['upper_percentile']} percentile")
print(f"  Cooldown: {best['loss_cooldown_days']} days")
print(f"  Vol adjustment: {best['use_vol_adjustment']}")
print(f"\nPerformance:")
print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
print(f"  Annualized Return: {best['annualized_return']*100:.2f}%")
print(f"  Total Return: {best['total_return']*100:.2f}%")
print(f"  Max Drawdown: {best['max_drawdown']*100:.2f}%")
print(f"  Win Rate: {best['win_rate']*100:.1f}%")
print(f"  Trades per year: {best['trades_per_year']:.0f}")
print(f"  Profit Factor: {best['profit_factor']:.3f}")
print(f"\nResults saved to: {output_file}")
print("="*80)
