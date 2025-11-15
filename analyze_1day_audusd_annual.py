"""
Analyze annual performance for 1-day AUDUSD model with optimized thresholds
Shows year-by-year results for 1:1 and 3:1 leverage
"""

import pandas as pd
import numpy as np
import pickle

print("1-DAY AUDUSD ANNUAL PERFORMANCE ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('data/AUDUSD_1day_with_features_FIXED_multitarget.csv',
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

# Load predictions
with open('xgboost_results_AUDUSD_target_5day_return.pkl', 'rb') as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} windows of predictions")

def generate_signals_optimized(predictions, lower_pct=48, upper_pct=52):
    """Generate signals using optimized thresholds (48th/52nd percentile for AUDUSD)."""
    predictions = np.array(predictions)
    signals = np.zeros(len(predictions))

    lower_threshold = np.percentile(predictions, lower_pct)
    upper_threshold = np.percentile(predictions, upper_pct)
    signals[predictions >= upper_threshold] = 1
    signals[predictions <= lower_threshold] = -1

    return signals

def backtest_with_dates(actuals, predictions, signals, df_prices, test_indices,
                        base_stop_loss_pct=0.0040,
                        base_take_profit_pct=0.0100,
                        loss_cooldown_days=1,
                        transaction_cost_pct=0.0002,
                        holding_period=5):
    """Backtest and return trades with dates for annual analysis."""

    position = 0
    entry_price = 0.0
    entry_date = None
    cooldown_remaining = 0
    holding_days = 0

    trades = []

    test_data = df_prices.iloc[test_indices]
    opens = test_data['open'].values
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    atr = test_data['atr'].values
    median_atr = np.median(atr[~np.isnan(atr)])

    for i in range(len(signals)):
        signal = signals[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        current_date = test_data.index[i]

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Volatility-adjusted stops
        if not np.isnan(atr[i]):
            vol_ratio = atr[i] / median_atr
            stop_loss_pct = base_stop_loss_pct * vol_ratio
            take_profit_pct = base_take_profit_pct * vol_ratio
        else:
            stop_loss_pct = base_stop_loss_pct
            take_profit_pct = base_take_profit_pct

        # Check exits if we have a position
        if position != 0:
            holding_days += 1

            # Calculate P&L
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

            # Stop loss
            if pct_low <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = 'Stop Loss'
                if position == 1:
                    exit_price = entry_price * (1 - stop_loss_pct)
                else:
                    exit_price = entry_price * (1 + stop_loss_pct)

            # Take profit
            elif pct_high >= take_profit_pct:
                exit_triggered = True
                exit_reason = 'Target'
                if position == 1:
                    exit_price = entry_price * (1 + take_profit_pct)
                else:
                    exit_price = entry_price * (1 - take_profit_pct)

            # Time exit (5 days)
            elif holding_days >= holding_period:
                exit_triggered = True
                exit_reason = 'Time Exit (5 days)'
                exit_price = close_price

            if exit_triggered:
                # Calculate return
                if position == 1:
                    raw_return_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    raw_return_pct = (entry_price - exit_price) / entry_price * 100

                net_return_pct = raw_return_pct - (transaction_cost_pct * 100)
                outcome = 'WIN' if net_return_pct > 0 else 'LOSS'

                # Record trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'exit_year': current_date.year,
                    'net_return_pct': net_return_pct,
                    'outcome': outcome,
                    'exit_reason': exit_reason
                })

                # Set cooldown after losing trade
                if net_return_pct < 0 and loss_cooldown_days > 0:
                    cooldown_remaining = loss_cooldown_days

                position = 0
                holding_days = 0

        # Enter new position (only if not in cooldown)
        if position == 0 and signal != 0 and cooldown_remaining == 0:
            position = signal
            entry_price = open_price
            entry_date = current_date
            holding_days = 0

    return trades

# Reconstruct windows
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

windows = generate_windows(df_clean, WINDOW_SIZE, ROLL_DAYS)

# Collect all trades with optimized thresholds
print("\nBacktesting with OPTIMIZED thresholds (48th/52nd percentile)...")
print("Configuration:")
print("  - Vol-adjusted stops (0.40% base SL / 1.00% base TP)")
print("  - 1 day loss cooldown")
print("  - 5 day holding period")
print()

all_trades = []

for window_idx, window_results in enumerate(all_results):
    predictions = window_results['predictions']
    actuals = window_results['actuals']

    # Get test indices from window
    window = windows[window_idx]
    test_indices = list(range(window['test_start'], window['test_end']))

    # Use OPTIMIZED thresholds (48/52 for AUDUSD)
    signals = generate_signals_optimized(predictions, lower_pct=48, upper_pct=52)

    trades = backtest_with_dates(
        actuals,
        predictions,
        signals,
        df_clean,
        test_indices,
        base_stop_loss_pct=0.0040,
        base_take_profit_pct=0.0100,
        loss_cooldown_days=1,
        transaction_cost_pct=0.0002,
        holding_period=5
    )

    all_trades.extend(trades)

trades_df = pd.DataFrame(all_trades)
print(f"Total trades: {len(trades_df)}")

# Analyze annual performance for 1:1 and 3:1 leverage
def calculate_annual_performance(trades_df, leverage=1.0, initial_capital=1000):
    """Calculate year-by-year performance with leverage."""
    capital = initial_capital
    yearly_stats = []

    years = sorted(trades_df['exit_year'].unique())

    for year in years:
        year_trades = trades_df[trades_df['exit_year'] == year]

        # Track capital throughout year for drawdown calculation
        year_capital = [capital]
        max_capital = capital
        max_dd = 0.0

        for _, trade in year_trades.iterrows():
            base_return = trade['net_return_pct'] / 100.0
            leveraged_return = base_return * leverage

            capital_before = capital
            pnl = capital * leveraged_return
            capital += pnl

            year_capital.append(capital)
            max_capital = max(max_capital, capital)

            # Track drawdown
            if max_capital > 0:
                dd = (capital - max_capital) / max_capital
                max_dd = min(max_dd, dd)

        # Calculate year stats
        year_start_cap = year_capital[0]
        year_end_cap = year_capital[-1]
        year_return_pct = (year_end_cap - year_start_cap) / year_start_cap * 100 if year_start_cap > 0 else 0

        wins = year_trades[year_trades['outcome'] == 'WIN']
        win_rate = len(wins) / len(year_trades) * 100 if len(year_trades) > 0 else 0

        yearly_stats.append({
            'year': year,
            'trades': len(year_trades),
            'win_rate': win_rate,
            'year_return_pct': year_return_pct,
            'start_capital': year_start_cap,
            'end_capital': year_end_cap,
            'max_dd_pct': max_dd * 100
        })

    return pd.DataFrame(yearly_stats), capital

# Calculate for 1:1 leverage
print("\n" + "="*80)
print("ANNUAL PERFORMANCE - 1:1 LEVERAGE (NO LEVERAGE)")
print("="*80)
print()

yearly_1x, final_1x = calculate_annual_performance(trades_df, leverage=1.0, initial_capital=1000)

print(f"{'Year':<6} {'Trades':<8} {'Win %':<8} {'Return %':<12} {'Capital':<15} {'Max DD %':<10}")
print("-" * 80)
for _, row in yearly_1x.iterrows():
    print(f"{row['year']:<6} {row['trades']:<8} {row['win_rate']:>6.1f}% {row['year_return_pct']:>10.2f}% "
          f"${row['end_capital']:>12,.2f} {row['max_dd_pct']:>8.2f}%")

print("-" * 80)
print(f"FINAL CAPITAL: ${final_1x:,.2f}")
print(f"TOTAL RETURN: {(final_1x - 1000) / 1000 * 100:.2f}%")

# Calculate compound annual return
years_elapsed = yearly_1x['year'].max() - yearly_1x['year'].min() + 1
annual_return_1x = (final_1x / 1000) ** (1 / years_elapsed) - 1
print(f"ANNUAL RETURN: {annual_return_1x * 100:.2f}%")
print()

# Calculate for 3:1 leverage
print("="*80)
print("ANNUAL PERFORMANCE - 3:1 LEVERAGE")
print("="*80)
print()

yearly_3x, final_3x = calculate_annual_performance(trades_df, leverage=3.0, initial_capital=1000)

print(f"{'Year':<6} {'Trades':<8} {'Win %':<8} {'Return %':<12} {'Capital':<15} {'Max DD %':<10}")
print("-" * 80)
for _, row in yearly_3x.iterrows():
    print(f"{row['year']:<6} {row['trades']:<8} {row['win_rate']:>6.1f}% {row['year_return_pct']:>10.2f}% "
          f"${row['end_capital']:>12,.2f} {row['max_dd_pct']:>8.2f}%")

print("-" * 80)
print(f"FINAL CAPITAL: ${final_3x:,.2f}")
print(f"TOTAL RETURN: {(final_3x - 1000) / 1000 * 100:.2f}%")

# Calculate compound annual return
annual_return_3x = (final_3x / 1000) ** (1 / years_elapsed) - 1
print(f"ANNUAL RETURN: {annual_return_3x * 100:.2f}%")
print()

# Summary comparison
print("="*80)
print("LEVERAGE COMPARISON SUMMARY")
print("="*80)
print()
print(f"{'Metric':<30} {'1:1 Leverage':<20} {'3:1 Leverage':<20}")
print("-" * 80)
print(f"{'Starting Capital':<30} {'$1,000':<20} {'$1,000':<20}")
print(f"{'Final Capital':<30} ${final_1x:<19,.2f} ${final_3x:<19,.2f}")
print(f"{'Total Return':<30} {(final_1x-1000)/1000*100:<19.2f}% {(final_3x-1000)/1000*100:<19.2f}%")
print(f"{'Annual Return':<30} {annual_return_1x*100:<19.2f}% {annual_return_3x*100:<19.2f}%")
print(f"{'Worst Annual DD':<30} {yearly_1x['max_dd_pct'].min():<19.2f}% {yearly_3x['max_dd_pct'].min():<19.2f}%")
print(f"{'Best Year':<30} {yearly_1x['year_return_pct'].max():<19.2f}% {yearly_3x['year_return_pct'].max():<19.2f}%")
print(f"{'Worst Year':<30} {yearly_1x['year_return_pct'].min():<19.2f}% {yearly_3x['year_return_pct'].min():<19.2f}%")
print()

# Find best and worst years
best_year_1x = yearly_1x.loc[yearly_1x['year_return_pct'].idxmax()]
worst_year_1x = yearly_1x.loc[yearly_1x['year_return_pct'].idxmin()]
best_year_3x = yearly_3x.loc[yearly_3x['year_return_pct'].idxmax()]
worst_year_3x = yearly_3x.loc[yearly_3x['year_return_pct'].idxmin()]

print(f"1:1 Best Year: {int(best_year_1x['year'])} (+{best_year_1x['year_return_pct']:.2f}%)")
print(f"1:1 Worst Year: {int(worst_year_1x['year'])} ({worst_year_1x['year_return_pct']:.2f}%)")
print()
print(f"3:1 Best Year: {int(best_year_3x['year'])} (+{best_year_3x['year_return_pct']:.2f}%)")
print(f"3:1 Worst Year: {int(worst_year_3x['year'])} ({worst_year_3x['year_return_pct']:.2f}%)")
print()

print("="*80)
print("Analysis complete!")
print("="*80)
