"""
Export trades from the best XGBoost strategy to CSV for manual verification.
Supports multiple currency pairs: EURUSD, GBPUSD, USDJPY, AUDUSD.
"""

import pandas as pd
import numpy as np
import pickle
import sys

# Allow user to specify pair, default to all
if len(sys.argv) > 1:
    PAIRS = [sys.argv[1].upper()]
else:
    PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

print("TRADE EXPORT FOR MANUAL VERIFICATION")
print("="*80)
print(f"Exporting trades for: {', '.join(PAIRS)}")
print("="*80)

# Configuration (matches backtest_advanced_exits.py)
TRAIN_DAYS = 600
VAL_DAYS = 156
TEST_DAYS = 126
ROLL_DAYS = 126
WINDOW_SIZE = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

SIGNAL_THRESHOLD = 0.003  # 0.3% threshold
BASE_STOP_PCT = 0.004     # 0.4% base stop
TARGET_MULTIPLE = 2.5     # 2.5:1 risk/reward
TRANSACTION_COST_PCT = 0.0002  # 0.02% per trade

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


def export_pair_trades(pair):
    """Export trades for a specific currency pair."""
    print(f"\n{'='*80}")
    print(f"Processing {pair}")
    print(f"{'='*80}")

    # Load data
    print(f"\nLoading {pair} data...")
    try:
        df = pd.read_csv(f'data/{pair}_1day_with_features_FIXED_multitarget.csv',
                        index_col='date', parse_dates=True)
    except FileNotFoundError:
        print(f"  ERROR: Data file not found for {pair}")
        return None

    # Load results
    print(f"Loading {pair} backtest results...")
    try:
        with open(f'xgboost_results_{pair}_target_5day_return.pkl', 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"  ERROR: Results file not found for {pair}")
        return None

    print(f"  Loaded {len(results)} windows")

    # Generate windows
    windows = generate_windows(df, WINDOW_SIZE, ROLL_DAYS)
    n_windows = min(len(results), len(windows))

    print(f"  Processing {n_windows} windows...")

    # Collect all trades
    all_trades = []

    for window_idx in range(n_windows):
        result = results[window_idx]
        window = windows[window_idx]

        # Get test period data
        test_start = window['test_start']
        test_end = window['test_end']
        test_data = df.iloc[test_start:test_end].copy()

        # Get predictions for this window
        predictions = np.array(result['predictions'])

        # Generate signals
        test_data['prediction'] = predictions
        test_data['signal'] = 0
        test_data.loc[test_data['prediction'] > SIGNAL_THRESHOLD, 'signal'] = 1  # Long
        test_data.loc[test_data['prediction'] < -SIGNAL_THRESHOLD, 'signal'] = -1  # Short

        # Calculate ATR for volatility adjustment
        high_low = test_data['high'] - test_data['low']
        high_close = np.abs(test_data['high'] - test_data['close'].shift())
        low_close = np.abs(test_data['low'] - test_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        test_data['atr'] = atr

        # Track cooldown
        last_loss_date = None

        # Process each day
        for i in range(len(test_data)):
            if test_data['signal'].iloc[i] == 0:
                continue

            signal_date = test_data.index[i]

            # Check if in cooldown
            if last_loss_date is not None:
                days_since_loss = (signal_date - last_loss_date).days
                if days_since_loss < 1:
                    continue

            # Entry on next day's open
            if i + 1 >= len(test_data):
                continue

            entry_date = test_data.index[i + 1]
            entry_price = test_data['open'].iloc[i + 1]
            signal = test_data['signal'].iloc[i]

            # Volatility-adjusted stop
            current_atr = test_data['atr'].iloc[i]
            if pd.notna(current_atr) and current_atr > 0:
                atr_multiple = current_atr / entry_price
                stop_pct = max(BASE_STOP_PCT, atr_multiple * 1.5)
            else:
                stop_pct = BASE_STOP_PCT

            target_pct = stop_pct * TARGET_MULTIPLE

            # Calculate stop and target prices
            if signal == 1:  # Long
                stop_price = entry_price * (1 - stop_pct)
                target_price = entry_price * (1 + target_pct)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct)
                target_price = entry_price * (1 - target_pct)

            # Simulate trade outcome
            exit_price = None
            exit_date = None
            exit_reason = None

            for j in range(i + 1, min(i + 6, len(test_data))):
                day_high = test_data['high'].iloc[j]
                day_low = test_data['low'].iloc[j]

                if signal == 1:  # Long
                    if day_low <= stop_price:
                        exit_price = stop_price
                        exit_date = test_data.index[j]
                        exit_reason = 'Stop Loss'
                        break
                    elif day_high >= target_price:
                        exit_price = target_price
                        exit_date = test_data.index[j]
                        exit_reason = 'Target'
                        break
                else:  # Short
                    if day_high >= stop_price:
                        exit_price = stop_price
                        exit_date = test_data.index[j]
                        exit_reason = 'Stop Loss'
                        break
                    elif day_low <= target_price:
                        exit_price = target_price
                        exit_date = test_data.index[j]
                        exit_reason = 'Target'
                        break

            # If no exit within 5 days, exit at day 5 close
            if exit_price is None:
                exit_idx = min(i + 5, len(test_data) - 1)
                exit_price = test_data['close'].iloc[exit_idx]
                exit_date = test_data.index[exit_idx]
                exit_reason = 'Time Exit (5 days)'

            # Calculate P&L
            if signal == 1:  # Long
                raw_return = (exit_price - entry_price) / entry_price
            else:  # Short
                raw_return = (entry_price - exit_price) / entry_price

            # Apply transaction costs
            net_return = raw_return - (2 * TRANSACTION_COST_PCT)

            # Record trade
            trade = {
                'pair': pair,
                'signal_date': signal_date.strftime('%Y-%m-%d'),
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'signal': 'LONG' if signal == 1 else 'SHORT',
                'prediction': test_data['prediction'].iloc[i],
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'stop_pct': stop_pct * 100,  # Convert to percentage
                'target_pct': target_pct * 100,  # Convert to percentage
                'raw_return_pct': raw_return * 100,
                'net_return_pct': net_return * 100,
                'outcome': 'WIN' if net_return > 0 else 'LOSS'
            }
            all_trades.append(trade)

            # Update cooldown
            if net_return < 0:
                last_loss_date = entry_date

    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)

    print(f"\n  Total trades extracted: {len(trades_df)}")
    print(f"  Date range: {trades_df['signal_date'].min()} to {trades_df['signal_date'].max()}")
    print(f"  Wins: {(trades_df['outcome'] == 'WIN').sum()}")
    print(f"  Losses: {(trades_df['outcome'] == 'LOSS').sum()}")
    print(f"  Win rate: {(trades_df['outcome'] == 'WIN').sum() / len(trades_df) * 100:.2f}%")
    print(f"  Average return: {trades_df['net_return_pct'].mean():.3f}%")

    # Export all trades
    output_file = f'xgboost_trades_{pair}_all.csv'
    trades_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n  All trades exported to: {output_file}")

    # Export 2024 trades separately (if available)
    trades_df['year'] = pd.to_datetime(trades_df['signal_date']).dt.year
    if 2024 in trades_df['year'].values:
        trades_2024 = trades_df[trades_df['year'] == 2024].copy()
        output_2024 = f'xgboost_trades_{pair}_2024.csv'
        trades_2024.to_csv(output_2024, index=False, float_format='%.6f')
        print(f"  2024 trades exported to: {output_2024} ({len(trades_2024)} trades)")

    # Export recent 50 trades for quick verification
    recent_trades = trades_df.tail(50).copy()
    output_recent = f'xgboost_trades_{pair}_recent_50.csv'
    recent_trades.to_csv(output_recent, index=False, float_format='%.6f')
    print(f"  Recent 50 trades exported to: {output_recent}")

    return trades_df


# Process all pairs
all_pair_results = {}

for pair in PAIRS:
    trades_df = export_pair_trades(pair)
    if trades_df is not None:
        all_pair_results[pair] = trades_df

# Create combined recent trades file
if all_pair_results:
    print(f"\n{'='*80}")
    print("Creating combined file with recent trades from all pairs...")
    print(f"{'='*80}")

    combined_recent = []
    for pair, trades_df in all_pair_results.items():
        recent = trades_df.tail(20)  # Get 20 most recent from each pair
        combined_recent.append(recent)

    combined_df = pd.concat(combined_recent, ignore_index=True)
    combined_df = combined_df.sort_values('entry_date', ascending=False)

    output_combined = 'xgboost_trades_all_pairs_recent.csv'
    combined_df.to_csv(output_combined, index=False, float_format='%.6f')
    print(f"\nCombined recent trades exported to: {output_combined}")
    print(f"  ({len(combined_df)} trades total)")

# Summary
print("\n" + "="*80)
print("EXPORT SUMMARY")
print("="*80)

for pair, trades_df in all_pair_results.items():
    print(f"\n{pair}:")
    print(f"  Total trades: {len(trades_df)}")
    print(f"  Win rate: {(trades_df['outcome'] == 'WIN').sum() / len(trades_df) * 100:.2f}%")
    print(f"  Avg return: {trades_df['net_return_pct'].mean():.3f}%")

print("\n" + "="*80)
print("VERIFICATION GUIDE")
print("="*80)
print("\nTo manually verify trades on TradingView:")
print("\n1. Go to TradingView.com and open the currency pair chart")
print("2. Set to Daily timeframe")
print("3. Use the date navigation or Replay mode")
print("4. For each trade in the CSV:")
print("   - Navigate to the 'signal_date' to see the prediction")
print("   - Check 'entry_date' (next day's open)")
print("   - Verify stop_price and target_price levels")
print("   - Confirm exit_price matches the actual high/low/close")
print("\n5. CSV Columns explained:")
print("   - signal_date: Day model generated signal")
print("   - entry_date: Next day (when trade entered at open)")
print("   - entry_price: Open price on entry_date")
print("   - stop_price: Stop loss level")
print("   - target_price: Take profit level")
print("   - exit_price: Actual exit price")
print("   - exit_reason: Target / Stop Loss / Time Exit")
print("   - net_return_pct: Final P&L after 0.02% transaction costs")

print("\n" + "="*80)
print("Export complete!")
print("="*80)
