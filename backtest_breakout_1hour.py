"""
Breakout strategy for 1-hour forex trading with volume confirmation.

Classic breakout approach:
- LONG: Price breaks above 20-bar high with volume > average
- SHORT: Price breaks below 20-bar low with volume > average

Usage:
    python backtest_breakout_1hour.py --pair EURUSD
    python backtest_breakout_1hour.py --pair GBPUSD --lookback 30
"""

import pandas as pd
import numpy as np
import json
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD)')
parser.add_argument('--lookback', type=int, default=20,
                    help='Lookback period for highs/lows (default: 20)')
parser.add_argument('--vol-threshold', type=float, default=1.2,
                    help='Volume multiplier threshold (default: 1.2x average)')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()
LOOKBACK = args.lookback
VOL_THRESHOLD = args.vol_threshold

print("="*80)
print(f"BREAKOUT STRATEGY BACKTEST - 1-HOUR {CURRENCY_PAIR}")
print("="*80)

# Load 1-hour data
print(f"\nLoading 1-hour {CURRENCY_PAIR} data...")
df = pd.read_csv(f'data/{CURRENCY_PAIR}_1hour_with_features.csv',
                 index_col='date', parse_dates=True)

required_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
df_clean = df.dropna(subset=required_cols)

print(f"Data shape: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
print(f"Total bars: {len(df_clean)}")


def generate_signals(df, lookback, vol_threshold, vol_window=50):
    """
    Generate breakout signals with volume confirmation.

    LONG: Close > highest high of last N bars AND volume > threshold * avg volume
    SHORT: Close < lowest low of last N bars AND volume > threshold * avg volume
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0

    # Calculate rolling highs/lows
    signals['high_breakout_level'] = df['high'].rolling(lookback).max().shift(1)
    signals['low_breakout_level'] = df['low'].rolling(lookback).min().shift(1)

    # Volume confirmation
    signals['avg_volume'] = df['volume'].rolling(vol_window).mean()
    signals['volume_confirmed'] = df['volume'] > (signals['avg_volume'] * vol_threshold)

    # Breakout conditions
    long_breakout = (df['close'] > signals['high_breakout_level']) & signals['volume_confirmed']
    short_breakout = (df['close'] < signals['low_breakout_level']) & signals['volume_confirmed']

    signals.loc[long_breakout, 'signal'] = 1
    signals.loc[short_breakout, 'signal'] = -1

    return signals


def backtest_strategy(df, signals, config, initial_capital=1000.0, transaction_cost=0.0002):
    """
    Backtest breakout strategy with vol-adjusted stops.
    """
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]

    max_hold_bars = config['max_hold_bars']
    stop_loss_pct = config['stop_loss_pct']
    take_profit_pct = config.get('take_profit_pct', None)
    use_vol_adjustment = config.get('use_vol_adjustment', False)
    vol_window = config.get('vol_window', 20)

    # Calculate volatility adjustment if needed
    if use_vol_adjustment:
        vol_multiplier_col = 'atr_vol_multiplier'
        df = df.copy()
        df[vol_multiplier_col] = df['atr'] / df['atr'].rolling(vol_window).median()
        df[vol_multiplier_col] = df[vol_multiplier_col].clip(0.5, 2.0)

    for i in range(len(df)):
        current_bar = df.iloc[i]
        current_date = df.index[i]

        # Check if we have an open position
        if position is not None:
            bars_held = i - position['entry_idx']
            entry_price = position['entry_price']
            direction = position['direction']

            # Calculate P&L
            if direction == 1:  # Long
                pnl_at_high = (current_bar['high'] - entry_price) / entry_price
                pnl_at_low = (current_bar['low'] - entry_price) / entry_price
            else:  # Short
                pnl_at_high = (entry_price - current_bar['low']) / entry_price
                pnl_at_low = (entry_price - current_bar['high']) / entry_price

            # Exit conditions
            exit_reason = None
            exit_price = None

            # Take profit check
            if take_profit_pct and pnl_at_high >= position['take_profit_pct']:
                exit_reason = 'take_profit'
                if direction == 1:
                    exit_price = entry_price * (1 + position['take_profit_pct'])
                else:
                    exit_price = entry_price * (1 - position['take_profit_pct'])

            # Stop loss check
            elif pnl_at_low <= -position['stop_loss_pct']:
                exit_reason = 'stop_loss'
                if direction == 1:
                    exit_price = entry_price * (1 - position['stop_loss_pct'])
                else:
                    exit_price = entry_price * (1 + position['stop_loss_pct'])

            # Time-based exit
            elif bars_held >= max_hold_bars:
                exit_reason = 'time_exit'
                exit_price = current_bar['close']

            # Exit at end of data
            elif i == len(df) - 1:
                exit_reason = 'end_of_period'
                exit_price = current_bar['close']

            # Close position if exit triggered
            if exit_reason:
                if direction == 1:
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price

                # Apply transaction costs
                pnl -= transaction_cost * 2

                capital *= (1 + pnl)

                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                    'capital_after': capital
                })

                position = None

        # Check for new entry signal (only if no position)
        if position is None and i < len(df) - 1:
            signal = signals.iloc[i]['signal']

            if signal != 0:
                # Adjust stops based on volatility if enabled
                if use_vol_adjustment:
                    vol_ratio = current_bar['atr_vol_multiplier']
                    adj_stop_loss = stop_loss_pct * vol_ratio
                    adj_take_profit = take_profit_pct * vol_ratio if take_profit_pct else None
                else:
                    adj_stop_loss = stop_loss_pct
                    adj_take_profit = take_profit_pct

                position = {
                    'entry_date': current_date,
                    'entry_idx': i,
                    'entry_price': current_bar['close'],
                    'direction': signal,
                    'stop_loss_pct': adj_stop_loss,
                    'take_profit_pct': adj_take_profit
                }

        equity_curve.append(capital)

    return trades, equity_curve


def calculate_metrics(trades, equity_curve, initial_capital=1000.0):
    """Calculate performance metrics."""
    if not trades:
        return None

    df_trades = pd.DataFrame(trades)

    # Returns
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    # Calculate returns from equity curve
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    # Years in backtest
    years = len(equity_curve) / (24 * 365)

    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    volatility = returns.std() * np.sqrt(252 * 24)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Drawdown
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Trade statistics
    winning_trades = df_trades[df_trades['pnl_pct'] > 0]
    losing_trades = df_trades[df_trades['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / len(df_trades)
    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

    total_wins = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Exit reason breakdown
    exit_reasons = df_trades['exit_reason'].value_counts().to_dict()

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
        'n_trades': len(df_trades),
        'n_winning': len(winning_trades),
        'n_losing': len(losing_trades),
        'trades_per_year': len(df_trades) / years if years > 0 else 0,
        'final_capital': equity_curve[-1],
        'exit_reasons': exit_reasons
    }


# Test configurations
test_configs = [
    # Vol-adjusted like successful ML strategy
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.3%/0.75%, 120 bar',
        'lookback': LOOKBACK,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0075,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.4%/1.0%, 120 bar',
        'lookback': LOOKBACK,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.004,
        'take_profit_pct': 0.010,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.5%/1.25%, 144 bar',
        'lookback': LOOKBACK,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.0125,
        'max_hold_bars': 144,
        'use_vol_adjustment': True,
        'vol_window': 20
    },

    # Different lookback periods
    {
        'name': f'15-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.3%/0.75%, 120 bar',
        'lookback': 15,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0075,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },
    {
        'name': f'30-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.3%/0.75%, 120 bar',
        'lookback': 30,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0075,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },

    # Different volume thresholds
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>1.5x, Vol-adj 0.3%/0.75%, 120 bar',
        'lookback': LOOKBACK,
        'vol_threshold': 1.5,
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0075,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>1.0x, Vol-adj 0.3%/0.75%, 120 bar',
        'lookback': LOOKBACK,
        'vol_threshold': 1.0,
        'stop_loss_pct': 0.003,
        'take_profit_pct': 0.0075,
        'max_hold_bars': 120,
        'use_vol_adjustment': True,
        'vol_window': 20
    },

    # Wider stops
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>{VOL_THRESHOLD}x, Vol-adj 0.6%/1.5%, 144 bar',
        'lookback': LOOKBACK,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.006,
        'take_profit_pct': 0.015,
        'max_hold_bars': 144,
        'use_vol_adjustment': True,
        'vol_window': 20
    },

    # Fixed stops for comparison
    {
        'name': f'{LOOKBACK}-bar breakout, Vol>{VOL_THRESHOLD}x, FIXED 0.5%/1.0%, 120 bar',
        'lookback': LOOKBACK,
        'vol_threshold': VOL_THRESHOLD,
        'stop_loss_pct': 0.005,
        'take_profit_pct': 0.010,
        'max_hold_bars': 120,
        'use_vol_adjustment': False
    },
]

print("\n" + "="*80)
print("TESTING BREAKOUT STRATEGIES")
print("="*80)

results_summary = []

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")

    # Generate signals
    signals = generate_signals(
        df_clean,
        config['lookback'],
        config['vol_threshold']
    )

    n_signals = (signals['signal'] != 0).sum()
    print(f"Generated {n_signals} signals")

    # Backtest
    trades, equity_curve = backtest_strategy(
        df_clean,
        signals,
        config,
        initial_capital=1000.0,
        transaction_cost=0.0002
    )

    metrics = calculate_metrics(trades, equity_curve)

    if metrics:
        print(f"\nPerformance:")
        print(f"  Final Capital:       ${metrics['final_capital']:.2f}")
        print(f"  Total Return:        {metrics['total_return']*100:8.2f}%")
        print(f"  Annualized Return:   {metrics['annualized_return']*100:8.2f}%")
        print(f"  Monthly Return:      {metrics['annualized_return']/12*100:8.2f}%")
        print(f"  Volatility (annual): {metrics['volatility']*100:8.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:8.3f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:8.2f}%")
        print(f"  Win Rate:            {metrics['win_rate']*100:8.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:8.3f}")
        print(f"  Total Trades:        {metrics['n_trades']}")
        print(f"  Trades/Year:         {metrics['trades_per_year']:8.1f}")

        print(f"\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            pct = count / metrics['n_trades'] * 100
            print(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")

        results_summary.append({
            'config': config['name'],
            **config,
            **metrics
        })
    else:
        print("\nNo trades generated!")

# Summary comparison
print("\n" + "="*80)
print("COMPARISON TABLE (Sorted by Sharpe Ratio)")
print("="*80)
print(f"\n{'Config':<75} {'Return':>8} {'Annual':>8} {'Sharpe':>8} {'Max DD':>8} {'Trades/Yr':>10}")
print("-"*130)

results_summary.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

for r in results_summary:
    print(f"{r['config']:<75} "
          f"{r['total_return']*100:7.1f}% "
          f"{r['annualized_return']*100:7.2f}% "
          f"{r['sharpe_ratio']:7.3f} "
          f"{r['max_drawdown']*100:7.1f}% "
          f"{r['trades_per_year']:9.1f}")

# Save results
output_file = f'breakout_1hour_{CURRENCY_PAIR}_results.json'
with open(output_file, 'w') as f:
    results_json = []
    for r in results_summary:
        r_copy = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                      float(v) if isinstance(v, (np.floating, np.float64)) else v)
                  for k, v in r.items() if k != 'exit_reasons'}
        r_copy['exit_reasons'] = {k: int(v) for k, v in r['exit_reasons'].items()}
        results_json.append(r_copy)
    json.dump(results_json, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_file}")
print("="*80)

if results_summary:
    best = results_summary[0]
    print(f"\nBEST CONFIGURATION (by Sharpe Ratio):")
    print(f"  {best['config']}")
    print(f"  Sharpe Ratio:        {best['sharpe_ratio']:.3f}")
    print(f"  Annualized Return:   {best['annualized_return']*100:.2f}%")
    print(f"  Monthly Return:      {best['annualized_return']/12*100:.2f}%")
    print(f"  Total Return:        {best['total_return']*100:.2f}%")
    print(f"  Max Drawdown:        {best['max_drawdown']*100:.2f}%")
    print(f"  Trades/Year:         {best['trades_per_year']:.1f}")
    print("="*80)
