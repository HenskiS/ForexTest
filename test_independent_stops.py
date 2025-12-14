"""
Test independent stops vs combined stops
"""
import pandas as pd
import numpy as np
import pickle
import os

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT, HOLD_DAYS, SL_PCT, LEVERAGE = 200, 50, 10, 90, 5, 0.02, 1.5

pair_data = {}
for pair in ALL_PAIRS:
    pred_file = f'optimized_ann_predictions/predictions_{pair}.pkl'
    if not os.path.exists(pred_file):
        continue
    with open(pred_file, 'rb') as f:
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


def run_independent_stops(pairs, max_slots=5):
    """Each position has its own independent stop loss"""
    all_trades = []
    daily_pnl = {}

    for pair in pairs:
        if pair not in pair_data:
            continue
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []
        positions = []  # List of independent positions

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

            # Check each position INDEPENDENTLY
            remaining_positions = []
            for pos in positions:
                days_held = current_idx - pos['entry_idx']
                direction = pos['direction']

                # Check THIS position's stop loss
                sl_hit = False
                if direction == 1:
                    sl_price = pos['entry_price'] * (1 - SL_PCT)
                    if current_row['low'] <= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 - spread)
                        pnl = (exit_price / pos['entry_price']) - 1
                else:
                    sl_price = pos['entry_price'] * (1 + SL_PCT)
                    if current_row['high'] >= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 + spread)
                        pnl = (pos['entry_price'] / exit_price) - 1

                if sl_hit:
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': pos['entry_date'], 'exit': 'SL'})
                    alloc = 0.25
                    if pos['entry_date'] not in daily_pnl:
                        daily_pnl[pos['entry_date']] = 0.0
                    daily_pnl[pos['entry_date']] += alloc * pnl
                    continue  # Position closed, don't add to remaining

                # Check time exit
                if days_held >= HOLD_DAYS:
                    if direction == 1:
                        exit_price = current_row['close'] * (1 - spread)
                        pnl = (exit_price / pos['entry_price']) - 1
                    else:
                        exit_price = current_row['close'] * (1 + spread)
                        pnl = (pos['entry_price'] / exit_price) - 1
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': pos['entry_date'], 'exit': 'TIME'})
                    alloc = 0.25
                    if pos['entry_date'] not in daily_pnl:
                        daily_pnl[pos['entry_date']] = 0.0
                    daily_pnl[pos['entry_date']] += alloc * pnl
                    continue  # Position closed

                remaining_positions.append(pos)

            positions = remaining_positions

            # Check for new signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)

            if signal != 0:
                # Check if we can add (same direction, under max slots)
                can_add = True
                if positions and positions[0]['direction'] != signal:
                    can_add = False
                if len(positions) >= max_slots:
                    can_add = False

                if can_add:
                    entry_price = current_row['open']
                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                    else:
                        actual_entry = entry_price * (1 - spread)

                    positions.append({
                        'entry_price': actual_entry,
                        'entry_idx': current_idx,
                        'entry_date': current_date,
                        'direction': signal
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
    annual = ((1 + total_return) ** (1/years) - 1) * 100 if years > 0 else 0
    win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100
    sharpe = (leveraged.mean() / leveraged.std()) * np.sqrt(252) if leveraged.std() > 0 else 0
    cummax = np.maximum.accumulate(equity[1:])
    max_dd = ((np.array(equity[1:]) - cummax) / cummax).min() * 100
    return {'trades': len(trades), 'win_rate': win_rate, 'annual': annual, 'sharpe': sharpe, 'max_dd': max_dd}


if __name__ == '__main__':
    print('='*70)
    print('INDEPENDENT STOPS vs COMBINED STOPS (8 pairs, 5 slots)')
    print('='*70)
    print()

    # Independent stops
    trades_ind, pnl_ind = run_independent_stops(ALL_PAIRS, max_slots=5)
    m_ind = calc_metrics(trades_ind, pnl_ind)

    # Count SL vs TIME exits
    sl_ind = sum(1 for t in trades_ind if t['exit'] == 'SL')
    time_ind = len(trades_ind) - sl_ind

    print(f'INDEPENDENT STOPS:')
    print(f'  Trades: {m_ind["trades"]} (SL: {sl_ind}, Time: {time_ind})')
    print(f'  Win Rate: {m_ind["win_rate"]:.1f}%')
    print(f'  Annual: {m_ind["annual"]:.1f}%')
    print(f'  Sharpe: {m_ind["sharpe"]:.2f}')
    print(f'  Max DD: {m_ind["max_dd"]:.1f}%')
    print()
    print('For comparison, COMBINED stops (from earlier):')
    print('  Trades: 7617, Win Rate: 63.6%, Annual: 112.8%, Max DD: -16.3%')
