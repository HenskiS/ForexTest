"""
Test random entries vs model entries on DAILY data with optimal exit rules.

Compare:
1. Model entries + Optimal exits (baseline)
2. Always Long (100% long positions)
3. Always Short (100% short positions)
4. Random entries (25% L, 25% S, 50% N)

This reveals if the daily model predictions add value, or if the exit strategy
alone creates the edge.
"""

import pandas as pd
import numpy as np
import pickle
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=str, default='EURUSD',
                    help='Currency pair (e.g., EURUSD, GBPUSD)')
args = parser.parse_args()

CURRENCY_PAIR = args.pair.upper()

print(f"BACKTEST DAILY {CURRENCY_PAIR} - RANDOM ENTRY TESTING")
print("="*80)

# Load data and results
df = pd.read_csv(f'data/{CURRENCY_PAIR}_1day_with_features_FIXED_multitarget.csv',
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

with open(f'xgboost_results_{CURRENCY_PAIR}_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} predictions")


def generate_signals_percentile(predictions, lower_pct, upper_pct):
    """Generate signals from regression predictions using custom percentiles."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_threshold = np.percentile(predictions, lower_pct)
    upper_threshold = np.percentile(predictions, upper_pct)
    signals[predictions >= upper_threshold] = 1
    signals[predictions <= lower_threshold] = -1

    return signals


def generate_random_signals(n_samples, long_pct=0.25, short_pct=0.25, random_seed=42):
    """Generate random trading signals."""
    np.random.seed(random_seed)
    signals = np.zeros(n_samples)

    # Randomly assign signals based on percentages
    random_vals = np.random.random(n_samples)

    signals[random_vals < long_pct] = 1
    signals[random_vals >= long_pct + (1 - long_pct - short_pct)] = -1

    return signals


def backtest_optimal_exits(actuals, signals, df_prices, test_indices,
                           initial_capital,
                           base_stop_loss_pct=0.0040,
                           base_take_profit_pct=0.0100,
                           loss_cooldown_days=1,
                           transaction_cost_pct=0.0002):
    """Backtest daily strategy with optimal exits (volatility-adjusted stops, loss cooldown)."""
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    cooldown_remaining = 0

    equity_curve = [capital]
    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    atr = test_data['atr'].values

    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]

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

        if position != 0:
            if position == 1:
                pct_high = (high_price - entry_price) / entry_price
                pct_low = (low_price - entry_price) / entry_price
            else:
                pct_high = (entry_price - low_price) / entry_price
                pct_low = (entry_price - high_price) / entry_price

            exit_triggered = False
            exit_price = None

            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)
            elif i == len(signals) - 1:
                exit_triggered = True
                exit_price = open_price

            if exit_triggered:
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl_pct -= transaction_cost_pct
                new_capital = entry_capital * (1 + pnl_pct)
                capital = new_capital

                trades.append({'pnl_pct': pnl_pct})

                if pnl_pct < 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0

        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            entry_capital = capital

        if position == 0:
            equity_curve.append(capital)
        else:
            if position == 1:
                unrealized_pnl = (open_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - open_price) / entry_price
            unrealized_pnl -= transaction_cost_pct
            current_equity = entry_capital * (1 + unrealized_pnl)
            equity_curve.append(current_equity)

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_capital': capital
    }


# Aggregate all test predictions and actuals from all windows
print(f"\nAggregating predictions from {len(all_results)} windows...")

# Configuration
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126

all_predictions = []
all_actuals = []
all_test_indices = []

for window_idx, result in enumerate(all_results):
    # Calculate window positions
    window_start = window_idx * ROLL_DAYS
    train_end = window_start + TRAIN_DAYS
    val_end = train_end + VAL_DAYS
    test_start = val_end
    test_end = test_start + TEST_DAYS

    # Get test indices for this window
    test_indices = range(test_start, test_end)

    # Append predictions and actuals
    all_predictions.extend(result['predictions'])
    all_actuals.extend(result['actuals'])
    all_test_indices.extend(test_indices)

# Convert to arrays
predictions = np.array(all_predictions)
actuals = np.array(all_actuals)
test_indices = all_test_indices

print(f"Total test samples: {len(predictions)}")

# Generate model-based signals (baseline)
initial_capital = 1000.0
model_signals = generate_signals_percentile(predictions, 25, 75)

print(f"\nModel signals:")
print(f"  Long:    {np.sum(model_signals == 1)} ({np.sum(model_signals == 1)/len(model_signals)*100:.1f}%)")
print(f"  Short:   {np.sum(model_signals == -1)} ({np.sum(model_signals == -1)/len(model_signals)*100:.1f}%)")
print(f"  Neutral: {np.sum(model_signals == 0)} ({np.sum(model_signals == 0)/len(model_signals)*100:.1f}%)")

print("\n" + "="*80)
print("TESTING ENTRY STRATEGIES")
print("="*80)
print("Using optimal exits: 0.40% SL / 1.00% TP (volatility-adjusted)")
print("Loss cooldown: 1 day")
print()

# Test configurations
entry_configs = [
    {"name": "Model Entries (Baseline)", "type": "model", "signals": model_signals},
    {"name": "Random Entries (25% L, 25% S, 50% N)", "type": "random", "long_pct": 0.25, "short_pct": 0.25},
    {"name": "Random Entries (50% L, 50% S, 0% N)", "type": "random", "long_pct": 0.50, "short_pct": 0.50},
    {"name": "Always Long (100% L, 0% S)", "type": "random", "long_pct": 1.00, "short_pct": 0.00},
    {"name": "Always Short (0% L, 100% S)", "type": "random", "long_pct": 0.00, "short_pct": 1.00},
]

results = []

for config in entry_configs:
    print(f"\n{'='*80}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*80}")

    # Generate appropriate signals
    if config['type'] == 'model':
        signals = config['signals']
    else:
        signals = generate_random_signals(
            len(test_indices),
            long_pct=config['long_pct'],
            short_pct=config['short_pct']
        )

        print(f"Signals generated:")
        print(f"  Long:    {np.sum(signals == 1)} ({np.sum(signals == 1)/len(signals)*100:.1f}%)")
        print(f"  Short:   {np.sum(signals == -1)} ({np.sum(signals == -1)/len(signals)*100:.1f}%)")
        print(f"  Neutral: {np.sum(signals == 0)} ({np.sum(signals == 0)/len(signals)*100:.1f}%)")

    # Backtest with optimal exits
    result = backtest_optimal_exits(
        actuals,
        signals,
        df_clean,
        test_indices,
        initial_capital,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002
    )

    # Calculate metrics
    trades_df = pd.DataFrame(result['trades'])
    equity_curve = np.array(result['equity_curve'])

    if len(trades_df) > 0:
        total_return = (result['final_capital'] - initial_capital) / initial_capital
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[returns != 0]

        total_days = len(predictions)
        years = total_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = np.min(drawdown)

        win_rate = (trades_df['pnl_pct'] > 0).mean()

        print(f"\nResults:")
        print(f"  Total Return:   {total_return*100:+.2f}%")
        print(f"  Annual Return:  {annual_return*100:.2f}%")
        print(f"  Sharpe Ratio:   {sharpe:.3f}")
        print(f"  Max Drawdown:   {max_dd*100:.2f}%")
        print(f"  Win Rate:       {win_rate*100:.1f}%")
        print(f"  Total Trades:   {len(trades_df)}")

        results.append({
            'Strategy': config['name'],
            'Annual Return': f"{annual_return*100:.2f}%",
            'Sharpe': f"{sharpe:.3f}",
            'Max DD': f"{max_dd*100:.2f}%",
            'Win Rate': f"{win_rate*100:.1f}%",
            'Trades': len(trades_df)
        })

print("\n" + "="*80)
print("ENTRY STRATEGY COMPARISON")
print("="*80)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("If model entries >> random/always entries:")
print("  -> Model predictions provide significant edge")
print("  -> Daily model is effective")
print()
print("If always long/short > model entries:")
print("  -> Exit strategy creates most of the edge")
print("  -> Directional bias + tight stops = profit")
print("  -> Model predictions dilute the trend-following edge")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
