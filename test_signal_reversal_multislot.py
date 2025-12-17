"""Test signal reversal with multi-slot system (implicit leverage via margin)"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT = 200, 50, 10, 90
SL_PCT, TP_PCT = 0.025, 0.025
LEVERAGE = 2.0

print('Loading predictions...')
pair_data = {}
for pair in ALL_PAIRS:
    try:
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
    except Exception as e:
        print(f'Could not load {pair}: {e}')


def run_strategy(max_slots, alloc_per_slot):
    """Run signal reversal with multi-slot system"""
    all_trades = []
    daily_pnl = {}

    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue

        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []
        slots = []  # List of {entry_price, entry_date, direction}

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

            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)

            if prediction >= upper_thresh:
                signal = 1
            elif prediction <= lower_thresh:
                signal = -1
            else:
                signal = 0

            # Check SL/TP for each slot independently
            still_open = []
            for slot in slots:
                exit_type = None
                exit_price = None

                if slot['direction'] == 1:
                    if current_row['low'] <= slot['entry_price'] * (1 - SL_PCT):
                        exit_type = 'SL'
                        exit_price = slot['entry_price'] * (1 - SL_PCT) * (1 - spread)
                    elif current_row['high'] >= slot['entry_price'] * (1 + TP_PCT):
                        exit_type = 'TP'
                        exit_price = slot['entry_price'] * (1 + TP_PCT) * (1 - spread)
                else:
                    if current_row['high'] >= slot['entry_price'] * (1 + SL_PCT):
                        exit_type = 'SL'
                        exit_price = slot['entry_price'] * (1 + SL_PCT) * (1 + spread)
                    elif current_row['low'] <= slot['entry_price'] * (1 - TP_PCT):
                        exit_type = 'TP'
                        exit_price = slot['entry_price'] * (1 - TP_PCT) * (1 + spread)

                if exit_type:
                    if slot['direction'] == 1:
                        pnl = (exit_price / slot['entry_price']) - 1
                    else:
                        pnl = (slot['entry_price'] / exit_price) - 1

                    all_trades.append({'entry_date': slot['entry_date'], 'pnl': pnl, 'exit_type': exit_type})
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += alloc_per_slot * pnl
                else:
                    still_open.append(slot)

            slots = still_open

            # Exit ALL slots on opposite signal (signal reversal)
            if slots and signal != 0 and signal != slots[0]['direction']:
                exit_price = current_row['open']
                for slot in slots:
                    if slot['direction'] == 1:
                        actual_exit = exit_price * (1 - spread)
                        pnl = (actual_exit / slot['entry_price']) - 1
                    else:
                        actual_exit = exit_price * (1 + spread)
                        pnl = (slot['entry_price'] / actual_exit) - 1

                    all_trades.append({'entry_date': slot['entry_date'], 'pnl': pnl, 'exit_type': 'REV'})
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += alloc_per_slot * pnl

                slots = []

            # Add new slot on signal (if room and same direction or empty)
            if signal != 0:
                can_add = True
                if slots and slots[0]['direction'] != signal:
                    can_add = False  # Opposite direction - should have been closed above
                if len(slots) >= max_slots:
                    can_add = False

                if can_add:
                    entry_price = current_row['open']
                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                    else:
                        actual_entry = entry_price * (1 - spread)

                    slots.append({
                        'direction': signal,
                        'entry_price': actual_entry,
                        'entry_date': current_date
                    })

    return all_trades, daily_pnl


def calc_stats(all_trades, daily_pnl, leverage):
    if not all_trades:
        return None
    trades_df = pd.DataFrame(all_trades)
    wins = (trades_df['pnl'] > 0).sum()
    total = len(trades_df)
    sl_exits = (trades_df['exit_type'] == 'SL').sum()
    tp_exits = (trades_df['exit_type'] == 'TP').sum()
    rev_exits = (trades_df['exit_type'] == 'REV').sum()

    daily_returns = pd.Series(daily_pnl).sort_index()
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365
    total_return = (1 + daily_returns * leverage).prod() - 1
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100

    cumulative = (1 + daily_returns * leverage).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    sharpe = (daily_returns.mean() * leverage * 252) / (daily_returns.std() * leverage * np.sqrt(252))

    # Calculate average slots open
    avg_daily_exposure = daily_returns.abs().mean() / 0.01  # rough estimate

    return {
        'trades': total, 'win_rate': wins/total*100, 'avg_pnl': trades_df['pnl'].mean()*100,
        'annual': annual_return, 'max_dd': max_dd, 'sharpe': sharpe,
        'sl_exits': sl_exits, 'tp_exits': tp_exits, 'rev_exits': rev_exits
    }


print()
print('='*100)
print('SIGNAL REVERSAL WITH MULTI-SLOT SYSTEM (2.5%/2.5% SL/TP, 2x leverage)')
print('='*100)
print()
print(f"{'Slots':<8} {'Alloc':<8} {'MaxExp':<10} {'Trades':<8} {'WinRate':<9} {'Annual':<12} {'MaxDD':<12} {'Sharpe':<8} {'Ret/DD':<8}")
print('-'*100)

# Test different slot configurations
configs = [
    # (max_slots, alloc_per_slot, description)
    (1, 1.0, "1 slot @ 100%"),      # Original single position
    (1, 0.5, "1 slot @ 50%"),       # Half size
    (2, 0.5, "2 slots @ 50%"),      # 2 slots, 100% max
    (3, 0.333, "3 slots @ 33%"),    # 3 slots, 100% max
    (5, 0.225, "5 slots @ 22.5%"),  # Same as Sleep Well
    (5, 0.20, "5 slots @ 20%"),     # 100% max
    (5, 0.15, "5 slots @ 15%"),     # 75% max
    (5, 0.10, "5 slots @ 10%"),     # 50% max
    (10, 0.10, "10 slots @ 10%"),   # 100% max, more granular
]

for max_slots, alloc, desc in configs:
    trades, pnl = run_strategy(max_slots, alloc)
    s = calc_stats(trades, pnl, LEVERAGE)
    if s:
        max_exposure = max_slots * alloc * 100
        ret_dd = abs(s['annual'] / s['max_dd'])
        print(f"{max_slots:<8} {alloc*100:<8.1f}% {max_exposure:<10.0f}% {s['trades']:<8} {s['win_rate']:<9.1f} {s['annual']:<12.1f} {s['max_dd']:<12.1f} {s['sharpe']:<8.2f} {ret_dd:<8.2f}")

print()
print('='*100)
print('COMPARISON: Single slot vs 5-slot @ same max exposure (112.5%)')
print('='*100)
print()

# Compare 1 slot @ 100% vs 5 slots @ 22.5%
for max_slots, alloc, label in [(1, 1.0, "1 slot @ 100%"), (5, 0.225, "5 slots @ 22.5%")]:
    trades, pnl = run_strategy(max_slots, alloc)
    s = calc_stats(trades, pnl, LEVERAGE)
    if s:
        print(f"{label}:")
        print(f"  Trades: {s['trades']}, Win Rate: {s['win_rate']:.1f}%")
        print(f"  Annual: {s['annual']:.1f}%, Max DD: {s['max_dd']:.1f}%, Sharpe: {s['sharpe']:.2f}")
        print(f"  Exits - SL: {s['sl_exits']}, TP: {s['tp_exits']}, Reversal: {s['rev_exits']}")
        print()
