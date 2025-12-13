"""
Test DCA (Dollar Cost Average) stop loss approach for netting accounts.
When adding to a position, we average the stop losses.
"""
import pandas as pd
import numpy as np
import pickle

# Load predictions
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
    pair_data[pair] = {'predictions': data['predictions'], 'test_indices': data['test_indices'], 'df': df}

BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT, HOLD_DAYS, SL_PCT, LEVERAGE = 200, 50, 10, 90, 5, 0.02, 1.5


def run_netting_dca_stop(max_slots=5, full_size_per_trade=False):
    """
    Simulate netting account with DCA stop loss.
    When adding to position, average the entry and set SL 2% below average.
    Track each 'virtual slot' for time-based exits.

    If full_size_per_trade=True, each trade gets full 25% allocation (implicit leverage).
    If False, allocation is split: 25% / max_slots per trade.
    """
    all_trades = []
    daily_pnl = {}

    for pair in ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []

        # Track virtual positions (for time exits) but use combined stop
        virtual_positions = []

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

            # Process existing positions
            if virtual_positions:
                direction = virtual_positions[0]['direction']
                total_size = sum(p['size'] for p in virtual_positions)
                avg_entry = sum(p['entry_price'] * p['size'] for p in virtual_positions) / total_size

                # Combined stop loss at 2% from average entry
                if direction == 1:
                    combined_sl = avg_entry * (1 - SL_PCT)
                else:
                    combined_sl = avg_entry * (1 + SL_PCT)

                # Check if combined SL hit
                sl_hit = False
                if direction == 1 and current_row['low'] <= combined_sl:
                    sl_hit = True
                    exit_price = combined_sl * (1 - spread)
                elif direction == -1 and current_row['high'] >= combined_sl:
                    sl_hit = True
                    exit_price = combined_sl * (1 + spread)

                if sl_hit:
                    # Close ALL positions at combined SL
                    for vp in virtual_positions:
                        if direction == 1:
                            pnl = (exit_price / vp['entry_price']) - 1
                        else:
                            pnl = (vp['entry_price'] / exit_price) - 1
                        all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'SL'})
                        alloc = 0.25 * vp['size'] if full_size_per_trade else 0.25 / max_slots * vp['size']
                        if vp['entry_date'] not in daily_pnl:
                            daily_pnl[vp['entry_date']] = 0.0
                        daily_pnl[vp['entry_date']] += alloc * pnl
                    virtual_positions = []
                else:
                    # Check time exits for each virtual position
                    remaining = []
                    for vp in virtual_positions:
                        days_held = current_idx - vp['entry_idx']
                        if days_held >= HOLD_DAYS:
                            if direction == 1:
                                exit_price = current_row['close'] * (1 - spread)
                                pnl = (exit_price / vp['entry_price']) - 1
                            else:
                                exit_price = current_row['close'] * (1 + spread)
                                pnl = (vp['entry_price'] / exit_price) - 1
                            all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'TIME'})
                            alloc = 0.25 * vp['size'] if full_size_per_trade else 0.25 / max_slots * vp['size']
                            if vp['entry_date'] not in daily_pnl:
                                daily_pnl[vp['entry_date']] = 0.0
                            daily_pnl[vp['entry_date']] += alloc * pnl
                        else:
                            remaining.append(vp)
                    virtual_positions = remaining

            # Check for new signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)

            if signal != 0:
                can_add = True
                # Can't mix directions in netting
                if virtual_positions and virtual_positions[0]['direction'] != signal:
                    can_add = False
                # Max slots limit
                if len(virtual_positions) >= max_slots:
                    can_add = False

                if can_add:
                    entry_price = current_row['open']
                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                    else:
                        actual_entry = entry_price * (1 - spread)

                    virtual_positions.append({
                        'entry_price': actual_entry,
                        'entry_idx': current_idx,
                        'entry_date': current_date,
                        'direction': signal,
                        'size': 1
                    })

    return all_trades, daily_pnl


def calc_metrics(trades, pnl_dict, leverage=LEVERAGE):
    if not trades or not pnl_dict:
        return None
    returns = pd.Series(pnl_dict).sort_index()
    leveraged = returns * leverage
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
    print('DCA STOP WITH FULL 25% ALLOCATION PER TRADE (IMPLICIT LEVERAGE)')
    print('=' * 70)
    print()
    print('This matches the "buggy" backtest allocation:')
    print('  - Each signal gets full 25% allocation')
    print('  - Up to 5 overlapping trades per pair = 125% exposure per pair')
    print('  - 4 pairs × 5 trades = up to 500% total exposure (5x implicit leverage)')
    print('  - But with DCA stop loss (averaged across entries)')
    print()

    # Test full size per trade (matches buggy backtest allocation)
    for max_slots in [3, 5]:
        trades, pnl = run_netting_dca_stop(max_slots=max_slots, full_size_per_trade=True)
        m = calc_metrics(trades, pnl)

        sl_exits = sum(1 for t in trades if t.get('exit') == 'SL')
        time_exits = sum(1 for t in trades if t.get('exit') == 'TIME')

        print(f'{max_slots} SLOTS, FULL 25% per trade (DCA Stop):')
        print(f'   Trades: {m["trades"]}')
        print(f'   Win Rate: {m["win_rate"]:.1f}%')
        print(f'   Annual Return: {m["annual"]:.1f}%')
        print(f'   Sharpe: {m["sharpe"]:.2f}')
        print(f'   Max DD: {m["max_dd"]:.1f}%')
        print(f'   SL exits: {sl_exits} ({sl_exits/len(trades)*100:.1f}%)')
        print(f'   Time exits: {time_exits} ({time_exits/len(trades)*100:.1f}%)')
        print()

    print('=' * 70)
    print('COMPARISON')
    print('=' * 70)
    print()
    print('Original buggy backtest (individual stops, full allocation):')
    print('  3829 trades, 69.0% WR, 65.1% annual, -13.9% DD')
    print()
    print('DCA stop should give similar results since it uses same allocation,')
    print('just with averaged stop loss instead of individual stops.')
