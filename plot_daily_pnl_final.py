"""
Plot daily P&L - exactly matching backtest_ann_multi_pair.py structure
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

print("="*100)
print("DAILY P&L VISUALIZATION - LAST 60 DAYS")
print("="*100)
print()

PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
SL_PCT = 0.0018
TP_PCT = 0.05
LEVERAGE = 2.0
STARTING_EQUITY = 500.0

def backtest_pair(pair, stop_loss, take_profit, hold_length=1, start_date=None, end_date=None):
    """Backtest a single pair - matches backtest_ann_multi_pair.py exactly"""

    # Load saved predictions
    with open(f'optimized_ann_predictions/predictions_{pair}.pkl', 'rb') as f:
        data = pickle.load(f)

    predictions = data['predictions']
    test_indices = data['test_indices']

    # Load price data
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Build test dataframe
    test_df = df.iloc[test_indices].copy()
    test_df['prediction'] = predictions

    # Filter to date range if specified
    if start_date is not None:
        test_df = test_df[test_df.index >= start_date]
    if end_date is not None:
        test_df = test_df[test_df.index <= end_date]

    # Load spread data
    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        test_df = test_df.join(spread_df[['spread_pct']], how='left')
    except FileNotFoundError:
        test_df['spread_pct'] = 0.00025

    # Backtest parameters
    LOWER_PCT = 50
    UPPER_PCT = 50
    BUFFER_SIZE = 200
    BUFFER_WARMUP = 50

    trades = []
    prediction_buffer = []
    daily_returns = []

    for i in range(len(test_df)):
        current_pred = test_df.iloc[i]['prediction']

        if pd.isna(current_pred):
            daily_returns.append(0)
            continue

        # Update buffer
        if i >= BUFFER_WARMUP:
            prediction_buffer.append(current_pred)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

        if len(prediction_buffer) < BUFFER_WARMUP:
            daily_returns.append(0)
            continue

        # Calculate thresholds
        buffer_array = np.array(prediction_buffer)
        lower_threshold = np.percentile(buffer_array, LOWER_PCT)
        upper_threshold = np.percentile(buffer_array, UPPER_PCT)

        # Generate signal
        if current_pred <= lower_threshold:
            direction = -1
        elif current_pred >= upper_threshold:
            direction = 1
        else:
            daily_returns.append(0)
            continue

        # Entry on next day
        if i + 1 >= len(test_df):
            break

        entry_price = test_df.iloc[i + 1]['open']
        spread_pct = test_df.iloc[i + 1]['spread_pct']
        if pd.isna(spread_pct):
            spread_pct = 0.00025

        # Check exit over hold period
        exit_day = min(i + 1 + hold_length, len(test_df) - 1)

        for hold_day in range(i + 2, exit_day + 1):
            if hold_day >= len(test_df):
                break

            exit_row = test_df.iloc[hold_day]

            if direction == 1:  # LONG
                actual_entry = entry_price * (1 + spread_pct)

                # Stop loss
                if exit_row['low'] <= actual_entry * (1 - stop_loss):
                    exit_price = actual_entry * (1 - stop_loss) * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # Take profit
                if exit_row['high'] >= actual_entry * (1 + take_profit):
                    exit_price = actual_entry * (1 + take_profit) * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # End of hold
                if hold_day == exit_day:
                    exit_price = exit_row['close'] * (1 - spread_pct)
                    pnl_pct = (exit_price / actual_entry) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

            else:  # SHORT
                actual_entry = entry_price * (1 - spread_pct)

                # Stop loss
                if exit_row['high'] >= actual_entry * (1 + stop_loss):
                    exit_price = actual_entry * (1 + stop_loss) * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # Take profit
                if exit_row['low'] <= actual_entry * (1 - take_profit):
                    exit_price = actual_entry * (1 - take_profit) * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break

                # End of hold
                if hold_day == exit_day:
                    exit_price = exit_row['close'] * (1 + spread_pct)
                    pnl_pct = (actual_entry / exit_price) - 1
                    trades.append(pnl_pct)
                    daily_returns.append(pnl_pct)
                    break
        else:
            # No trade taken
            daily_returns.append(0)

    return np.array(daily_returns), test_df.index

# Get actual date range from data
# First load one pair to find the actual last date
with open('optimized_ann_predictions/predictions_EURUSD.pkl', 'rb') as f:
    sample_data = pickle.load(f)

sample_df = pd.read_csv('data/EURUSD_1day_oanda.csv')
sample_df['date'] = pd.to_datetime(sample_df['date'])
sample_df = sample_df.set_index('date')
sample_test_df = sample_df.iloc[sample_data['test_indices']]

# Use actual last date from test data
actual_end_date = sample_test_df.index[-1]
if actual_end_date.tzinfo is None:
    actual_end_date = actual_end_date.tz_localize('UTC')

# Date range for last 180 days (6 months - to ensure we capture trades after buffer warmup)
end_date = actual_end_date
start_date = (end_date - pd.Timedelta(days=180))

print(f"Analysis period: {start_date.date()} to {end_date.date()}")
print()

# Backtest each pair
pair_returns = {}
print("Backtesting pairs...")
for pair in PAIRS:
    print(f"  {pair}...")
    returns, dates = backtest_pair(pair, SL_PCT, TP_PCT, hold_length=1,
                                    start_date=start_date, end_date=end_date)
    pair_returns[pair] = {'returns': returns, 'dates': dates}

print()

# Combine daily returns (equal weight)
n_pairs = len(PAIRS)
allocation = 1.0 / n_pairs

# Find common dates across all pairs
common_dates = None
for pair in PAIRS:
    if common_dates is None:
        common_dates = set(pair_returns[pair]['dates'])
    else:
        common_dates &= set(pair_returns[pair]['dates'])

common_dates = sorted(common_dates)

# Reindex all pairs to common dates
portfolio_returns = np.zeros(len(common_dates))

for pair in PAIRS:
    pair_dates = pair_returns[pair]['dates']
    pair_rets = pair_returns[pair]['returns']

    # Build a date->return mapping
    ret_dict = dict(zip(pair_dates, pair_rets))

    # Get returns for common dates
    aligned_returns = np.array([ret_dict.get(d, 0.0) for d in common_dates])
    portfolio_returns += allocation * aligned_returns

all_dates = common_dates

# Apply leverage
leveraged_returns = portfolio_returns * LEVERAGE

# Calculate equity curve with compounding
equity = STARTING_EQUITY
equity_curve = [equity]
daily_pnl = []

for ret in leveraged_returns:
    pnl = equity * ret
    equity += pnl
    equity_curve.append(equity)
    daily_pnl.append(pnl)

# Filter to only trading days (skip buffer warmup period with no trades)
# Find first and last day with non-zero P&L
daily_pnl_arr = np.array(daily_pnl)
non_zero_indices = np.where(daily_pnl_arr != 0)[0]

if len(non_zero_indices) > 0:
    first_trade_idx = non_zero_indices[0]
    last_trade_idx = non_zero_indices[-1] + 1  # +1 to include the last trade day

    # Crop to trading period
    all_dates = all_dates[first_trade_idx:last_trade_idx]
    daily_pnl = daily_pnl[first_trade_idx:last_trade_idx]
    equity_curve = equity_curve[first_trade_idx:last_trade_idx + 1]  # +1 because equity_curve has one extra element

    # Recalculate array after cropping
    daily_pnl_arr = np.array(daily_pnl)

# Calculate statistics
final_equity = equity_curve[-1]
total_return_pct = ((final_equity / STARTING_EQUITY) - 1) * 100
total_pnl = final_equity - STARTING_EQUITY

winning_days = (daily_pnl_arr > 0).sum()
losing_days = (daily_pnl_arr < 0).sum()
avg_daily = daily_pnl_arr.mean()
max_day = daily_pnl_arr.max()
min_day = daily_pnl_arr.min()

# Max drawdown
equity_curve_arr = np.array(equity_curve)
cummax = np.maximum.accumulate(equity_curve_arr)
drawdowns = (equity_curve_arr - cummax) / cummax * 100
max_drawdown_pct = drawdowns.min()

# Win rate (non-zero days)
non_zero = daily_pnl_arr[daily_pnl_arr != 0]
win_rate = (non_zero > 0).sum() / len(non_zero) * 100 if len(non_zero) > 0 else 0

print("Statistics:")
print(f"  Starting equity: ${STARTING_EQUITY:,.2f}")
print(f"  Ending equity: ${final_equity:,.2f}")
print(f"  Total return: {total_return_pct:+.2f}%")
print(f"  Total P&L: ${total_pnl:+.2f}")
print(f"  Average daily: ${avg_daily:+.2f}")
print(f"  Best day: ${max_day:+.2f}")
print(f"  Worst day: ${min_day:+.2f}")
print(f"  Max drawdown: {max_drawdown_pct:.2f}%")
print(f"  Trading days: {len(daily_pnl)}")
print(f"  Days with trades: {len(non_zero)}")
print(f"  Win rate (trading days): {win_rate:.1f}%")
print(f"  Winning days: {winning_days} ({100*winning_days/len(daily_pnl):.1f}%)")
print(f"  Losing days: {losing_days} ({100*losing_days/len(daily_pnl):.1f}%)")
print()

# Annualize return
years = len(daily_pnl) / 250
annual_return = ((final_equity / STARTING_EQUITY) ** (1 / years) - 1) * 100
print(f"  Annualized return: {annual_return:.2f}%")
print()

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Daily P&L bars
colors = ['green' if x > 0 else 'red' for x in daily_pnl]
ax1.bar(all_dates, daily_pnl, color=colors, alpha=0.7, width=0.8)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_title(f'Daily P&L - Last 60 Days (4-Pair ANN Strategy, {LEVERAGE}x Leverage, Median Split)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Daily P&L ($)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=avg_daily, color='blue', linestyle='--', linewidth=1,
            label=f'Average: ${avg_daily:.2f}')
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Equity curve
ax2.plot(all_dates, equity_curve[1:], color='blue', linewidth=2)
ax2.fill_between(all_dates, equity_curve[1:], STARTING_EQUITY,
                  where=(np.array(equity_curve[1:]) >= STARTING_EQUITY),
                  color='green', alpha=0.3)
ax2.fill_between(all_dates, equity_curve[1:], STARTING_EQUITY,
                  where=(np.array(equity_curve[1:]) < STARTING_EQUITY),
                  color='red', alpha=0.3)
ax2.axhline(y=STARTING_EQUITY, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Equity ($)', fontsize=12)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add stats box
stats_text = f'Start: ${STARTING_EQUITY:,.0f}\nEnd: ${final_equity:,.0f}\nReturn: {total_return_pct:+.1f}%\nAnnual: {annual_return:.1f}%\nMax DD: {max_drawdown_pct:.1f}%'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save plot
output_file = f'daily_pnl_60days_median_{LEVERAGE}x.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
print()

plt.show()

print("="*100)
