"""
Test exit-on-opposite-signal approach vs blocking new trades
"""
import pandas as pd
import numpy as np
import pickle

# Load predictions from optimized_ann_predictions (the good ones)
pair_data = {}
for pair in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']:
    with open(f'optimized_ann_predictions/predictions_{pair}.pkl', 'rb') as f:
        data = pickle.load(f)

    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    try:
        spread_df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
        spread_df['date'] = pd.to_datetime(spread_df['date'])
        spread_df = spread_df.set_index('date')
        df = df.join(spread_df[['spread_pct']], how='left')
    except:
        df['spread_pct'] = 0.00025
    df['spread_pct'] = df['spread_pct'].fillna(0.00025)

    pair_data[pair] = {
        'predictions': data['predictions'],
        'test_indices': data['test_indices'],
        'df': df
    }

# Config
BUFFER_SIZE = 200
BUFFER_WARMUP = 50
LOWER_PCT = 10
UPPER_PCT = 90
HOLD_DAYS = 5
SL_PCT = 0.02
LEVERAGE = 1.5

def run_backtest(exit_on_opposite=False, allow_overlap=False):
    """Run backtest with optional exit-on-opposite-signal logic"""
    all_trades = []
    daily_pnl = {}

    for pair in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []

        # Position state for this pair
        in_position = False
        position_direction = 0
        position_entry_price = 0
        position_entry_idx = 0
        position_stop_price = 0
        position_entry_date = None

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

            if len(prediction_buffer) < BUFFER_WARMUP:
                continue

            current_idx = pred_idx + 1  # Next day (entry day)
            if current_idx >= len(df):
                continue

            current_date = df.index[current_idx]
            if current_date.dayofweek >= 5:
                continue

            current_row = df.iloc[current_idx]
            spread = current_row['spread_pct']

            # Check if existing position should exit (stop loss or time)
            if in_position:
                days_held = current_idx - position_entry_idx

                # Check stop loss
                stopped_out = False
                if position_direction == 1 and current_row['low'] <= position_stop_price:
                    exit_price = position_stop_price * (1 - spread)
                    stopped_out = True
                elif position_direction == -1 and current_row['high'] >= position_stop_price:
                    exit_price = position_stop_price * (1 + spread)
                    stopped_out = True

                # Check time exit
                if not stopped_out and days_held >= HOLD_DAYS:
                    if position_direction == 1:
                        exit_price = current_row['close'] * (1 - spread)
                    else:
                        exit_price = current_row['close'] * (1 + spread)
                    stopped_out = True  # Not really stopped, but exiting

                if stopped_out:
                    if position_direction == 1:
                        pnl = (exit_price / position_entry_price) - 1
                    else:
                        pnl = (position_entry_price / exit_price) - 1

                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': position_entry_date})
                    if position_entry_date not in daily_pnl:
                        daily_pnl[position_entry_date] = 0.0
                    daily_pnl[position_entry_date] += 0.25 * pnl

                    in_position = False

            # Get signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)

            if prediction >= upper_thresh:
                signal = 1
            elif prediction <= lower_thresh:
                signal = -1
            else:
                signal = 0

            # Handle exit on opposite signal
            if exit_on_opposite and in_position and signal != 0 and signal != position_direction:
                # Close current position at current close
                if position_direction == 1:
                    exit_price = current_row['close'] * (1 - spread)
                    pnl = (exit_price / position_entry_price) - 1
                else:
                    exit_price = current_row['close'] * (1 + spread)
                    pnl = (position_entry_price / exit_price) - 1

                all_trades.append({'pnl': pnl, 'pair': pair, 'date': position_entry_date, 'exit': 'OPPOSITE'})
                if position_entry_date not in daily_pnl:
                    daily_pnl[position_entry_date] = 0.0
                daily_pnl[position_entry_date] += 0.25 * pnl

                in_position = False

            # Enter new position if flat and have signal (or allow_overlap)
            can_enter = (not in_position and signal != 0) or (allow_overlap and signal != 0)
            if can_enter:
                entry_price = current_row['open']

                if signal == 1:
                    actual_entry = entry_price * (1 + spread)
                    stop_price = actual_entry * (1 - SL_PCT)
                else:
                    actual_entry = entry_price * (1 - spread)
                    stop_price = actual_entry * (1 + SL_PCT)

                if not allow_overlap:
                    in_position = True
                    position_direction = signal
                    position_entry_price = actual_entry
                    position_entry_idx = current_idx
                    position_stop_price = stop_price
                    position_entry_date = current_date
                else:
                    # For allow_overlap, immediately calculate PnL at end of hold period
                    exit_idx = current_idx + HOLD_DAYS - 1
                    if exit_idx < len(df):
                        exit_row = df.iloc[exit_idx]
                        exit_price = None

                        # Check stops during hold period
                        for hold_day in range(HOLD_DAYS):
                            check_idx = current_idx + hold_day
                            if check_idx >= len(df):
                                break
                            check_row = df.iloc[check_idx]

                            if signal == 1 and check_row['low'] <= stop_price:
                                exit_price = stop_price * (1 - spread)
                                break
                            elif signal == -1 and check_row['high'] >= stop_price:
                                exit_price = stop_price * (1 + spread)
                                break

                        if exit_price is None:
                            if signal == 1:
                                exit_price = exit_row['close'] * (1 - spread)
                            else:
                                exit_price = exit_row['close'] * (1 + spread)

                        if signal == 1:
                            pnl = (exit_price / actual_entry) - 1
                        else:
                            pnl = (actual_entry / exit_price) - 1

                        all_trades.append({'pnl': pnl, 'pair': pair, 'date': current_date})
                        if current_date not in daily_pnl:
                            daily_pnl[current_date] = 0.0
                        daily_pnl[current_date] += 0.25 * pnl

    return all_trades, daily_pnl

def run_backtest_multislot(num_slots=5):
    """
    Run backtest with multiple slots per pair.
    Each slot can hold one position independently.
    This allows capturing more signals without "overlapping" in a single position.
    """
    all_trades = []
    daily_pnl = {}

    for pair in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []

        # Track multiple slots per pair
        # Each slot: {'in_position': bool, 'direction': int, 'entry_price': float,
        #             'entry_idx': int, 'stop_price': float, 'entry_date': date}
        slots = [{'in_position': False} for _ in range(num_slots)]

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer = prediction_buffer[-BUFFER_SIZE:]

            if len(prediction_buffer) < BUFFER_WARMUP:
                continue

            current_idx = pred_idx + 1
            if current_idx >= len(df):
                continue

            current_date = df.index[current_idx]
            if current_date.dayofweek >= 5:
                continue

            current_row = df.iloc[current_idx]
            spread = current_row['spread_pct']

            # Check each slot for exits
            for slot in slots:
                if not slot['in_position']:
                    continue

                days_held = current_idx - slot['entry_idx']

                # Check stop loss
                exited = False
                if slot['direction'] == 1 and current_row['low'] <= slot['stop_price']:
                    exit_price = slot['stop_price'] * (1 - spread)
                    exited = True
                elif slot['direction'] == -1 and current_row['high'] >= slot['stop_price']:
                    exit_price = slot['stop_price'] * (1 + spread)
                    exited = True

                # Check time exit
                if not exited and days_held >= HOLD_DAYS:
                    if slot['direction'] == 1:
                        exit_price = current_row['close'] * (1 - spread)
                    else:
                        exit_price = current_row['close'] * (1 + spread)
                    exited = True

                if exited:
                    if slot['direction'] == 1:
                        pnl = (exit_price / slot['entry_price']) - 1
                    else:
                        pnl = (slot['entry_price'] / exit_price) - 1

                    # Allocation per slot = 25% / num_slots
                    alloc = 0.25 / num_slots
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': slot['entry_date']})
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += alloc * pnl

                    slot['in_position'] = False

            # Get signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)

            if prediction >= upper_thresh:
                signal = 1
            elif prediction <= lower_thresh:
                signal = -1
            else:
                signal = 0

            # Enter new position if signal and have free slot
            if signal != 0:
                free_slot = next((s for s in slots if not s['in_position']), None)
                if free_slot:
                    entry_price = current_row['open']

                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                        stop_price = actual_entry * (1 - SL_PCT)
                    else:
                        actual_entry = entry_price * (1 - spread)
                        stop_price = actual_entry * (1 + SL_PCT)

                    free_slot['in_position'] = True
                    free_slot['direction'] = signal
                    free_slot['entry_price'] = actual_entry
                    free_slot['entry_idx'] = current_idx
                    free_slot['stop_price'] = stop_price
                    free_slot['entry_date'] = current_date

    return all_trades, daily_pnl


def calc_metrics(trades, pnl_dict):
    """Calculate performance metrics"""
    if not trades or not pnl_dict:
        return None
    returns = pd.Series(pnl_dict).sort_index()
    leveraged = returns * LEVERAGE
    equity = [1000]
    for r in leveraged:
        equity.append(equity[-1] * (1 + r))
    total_return = equity[-1] / 1000 - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    annual = ((1 + total_return) ** (1/years) - 1) * 100
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100
    sharpe = (leveraged.mean() / leveraged.std()) * np.sqrt(252) if leveraged.std() > 0 else 0
    cummax = np.maximum.accumulate(equity[1:])
    max_dd = ((np.array(equity[1:]) - cummax) / cummax).min() * 100
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'annual': annual,
        'sharpe': sharpe,
        'max_dd': max_dd
    }

if __name__ == '__main__':
    print('=' * 70)
    print('COMPARING BACKTEST APPROACHES')
    print('=' * 70)

    # Version 1: No overlap (current fix)
    trades1, pnl1 = run_backtest(exit_on_opposite=False, allow_overlap=False)
    m1 = calc_metrics(trades1, pnl1)
    print(f'\n1. NO OVERLAP (block new trades while in position):')
    print(f'   Trades: {m1["trades"]}')
    print(f'   Win Rate: {m1["win_rate"]:.1f}%')
    print(f'   Annual Return: {m1["annual"]:.1f}%')
    print(f'   Sharpe: {m1["sharpe"]:.2f}')
    print(f'   Max DD: {m1["max_dd"]:.1f}%')

    # Version 2: Exit on opposite signal
    trades2, pnl2 = run_backtest(exit_on_opposite=True, allow_overlap=False)
    m2 = calc_metrics(trades2, pnl2)
    opposite_exits = sum(1 for t in trades2 if t.get('exit') == 'OPPOSITE')
    print(f'\n2. EXIT ON OPPOSITE SIGNAL:')
    print(f'   Trades: {m2["trades"]}')
    print(f'   Win Rate: {m2["win_rate"]:.1f}%')
    print(f'   Annual Return: {m2["annual"]:.1f}%')
    print(f'   Sharpe: {m2["sharpe"]:.2f}')
    print(f'   Max DD: {m2["max_dd"]:.1f}%')
    print(f'   Early exits (opposite signal): {opposite_exits}')

    # Version 3: Allow overlapping trades (original buggy behavior)
    trades3, pnl3 = run_backtest(exit_on_opposite=False, allow_overlap=True)
    m3 = calc_metrics(trades3, pnl3)
    print(f'\n3. ALLOW OVERLAP (original buggy behavior):')
    print(f'   Trades: {m3["trades"]}')
    print(f'   Win Rate: {m3["win_rate"]:.1f}%')
    print(f'   Annual Return: {m3["annual"]:.1f}%')
    print(f'   Sharpe: {m3["sharpe"]:.2f}')
    print(f'   Max DD: {m3["max_dd"]:.1f}%')

    print(f'\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"Approach":<35} {"Trades":>8} {"WinRate":>8} {"Annual":>10} {"Sharpe":>8} {"MaxDD":>8}')
    print('-' * 70)
    print(f'{"No overlap":<35} {m1["trades"]:>8} {m1["win_rate"]:>7.1f}% {m1["annual"]:>9.1f}% {m1["sharpe"]:>8.2f} {m1["max_dd"]:>7.1f}%')
    print(f'{"Exit on opposite":<35} {m2["trades"]:>8} {m2["win_rate"]:>7.1f}% {m2["annual"]:>9.1f}% {m2["sharpe"]:>8.2f} {m2["max_dd"]:>7.1f}%')
    print(f'{"Allow overlap (BUGGY)":<35} {m3["trades"]:>8} {m3["win_rate"]:>7.1f}% {m3["annual"]:>9.1f}% {m3["sharpe"]:>8.2f} {m3["max_dd"]:>7.1f}%')

    # Version 4-7: Multi-slot approaches
    print(f'\n' + '=' * 70)
    print('MULTI-SLOT APPROACHES (realistic overlapping)')
    print('=' * 70)

    slot_results = {}
    for num_slots in [2, 3, 5, 10]:
        trades_s, pnl_s = run_backtest_multislot(num_slots=num_slots)
        ms = calc_metrics(trades_s, pnl_s)
        slot_results[num_slots] = ms
        print(f'\n{num_slots} SLOTS per pair (allocation: {25/num_slots:.1f}% each):')
        print(f'   Trades: {ms["trades"]}')
        print(f'   Win Rate: {ms["win_rate"]:.1f}%')
        print(f'   Annual Return: {ms["annual"]:.1f}%')
        print(f'   Sharpe: {ms["sharpe"]:.2f}')
        print(f'   Max DD: {ms["max_dd"]:.1f}%')

    print(f'\n' + '=' * 70)
    print('FULL COMPARISON')
    print('=' * 70)
    print(f'{"Approach":<35} {"Trades":>8} {"WinRate":>8} {"Annual":>10} {"Sharpe":>8} {"MaxDD":>8}')
    print('-' * 70)
    print(f'{"1 slot (no overlap)":<35} {m1["trades"]:>8} {m1["win_rate"]:>7.1f}% {m1["annual"]:>9.1f}% {m1["sharpe"]:>8.2f} {m1["max_dd"]:>7.1f}%')
    for num_slots, ms in slot_results.items():
        print(f'{f"{num_slots} slots per pair":<35} {ms["trades"]:>8} {ms["win_rate"]:>7.1f}% {ms["annual"]:>9.1f}% {ms["sharpe"]:>8.2f} {ms["max_dd"]:>7.1f}%')
    print(f'{"Unlimited (buggy)":<35} {m3["trades"]:>8} {m3["win_rate"]:>7.1f}% {m3["annual"]:>9.1f}% {m3["sharpe"]:>8.2f} {m3["max_dd"]:>7.1f}%')

    print(f'\n' + '=' * 70)
    print('CONCLUSION')
    print('=' * 70)
    print(f'With 5 slots per pair, you can capture most of the signals (~{slot_results[5]["trades"]} trades)')
    print(f'while staying within realistic capital allocation constraints.')
    print(f'This gives ~{slot_results[5]["annual"]:.0f}% annual return vs {m3["annual"]:.0f}% from buggy backtest.')
