"""Test signal reversal for FTMO challenge constraints - OPTIMIZED
- Max 5% daily drawdown
- Max 10% overall drawdown
"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
NUM_PAIRS = len(ALL_PAIRS)
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT = 200, 50, 10, 90
SL_PCT, TP_PCT = 0.025, 0.025

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


def run_strategy_raw(max_slots):
    """Run strategy once with 100% allocation per slot, return raw daily PnL"""
    daily_pnl = {}

    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue

        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']

        prediction_buffer = []
        slots = []

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

            # Check SL/TP for each slot
            still_open = []
            for slot in slots:
                exit_type = None

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

                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0.0
                    daily_pnl[current_date] += pnl  # Raw 100% allocation
                else:
                    still_open.append(slot)

            slots = still_open

            # Exit ALL slots on opposite signal
            if slots and signal != 0 and signal != slots[0]['direction']:
                exit_price = current_row['open']
                for slot in slots:
                    if slot['direction'] == 1:
                        actual_exit = exit_price * (1 - spread)
                        pnl = (actual_exit / slot['entry_price']) - 1
                    else:
                        actual_exit = exit_price * (1 + spread)
                        pnl = (slot['entry_price'] / actual_exit) - 1

                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0.0
                    daily_pnl[current_date] += pnl

                slots = []

            # Add new slot
            if signal != 0:
                can_add = True
                if slots and slots[0]['direction'] != signal:
                    can_add = False
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

    return pd.Series(daily_pnl).sort_index()


def calc_stats(raw_returns, alloc_per_slot, leverage):
    """Scale raw returns by allocation and leverage, calculate FTMO stats"""
    # Scale returns
    scaled = raw_returns * alloc_per_slot * leverage

    # Worst/best day
    worst_day = scaled.min() * 100
    best_day = scaled.max() * 100

    # Annual return
    years = (scaled.index[-1] - scaled.index[0]).days / 365
    cumulative = (1 + scaled).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual = ((1 + total_return) ** (1/years) - 1) * 100

    # Max DD
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Sharpe
    sharpe = (scaled.mean() * 252) / (scaled.std() * np.sqrt(252))

    return {
        'annual': annual, 'max_dd': max_dd, 'worst_day': worst_day,
        'best_day': best_day, 'sharpe': sharpe
    }


# Run strategy once per slot config
print('Running backtests (once per slot config)...')
raw_1slot = run_strategy_raw(1)
raw_2slot = run_strategy_raw(2)
raw_3slot = run_strategy_raw(3)
raw_5slot = run_strategy_raw(5)

print()
print('='*100)
print('SIGNAL REVERSAL - FTMO CONSTRAINTS (5% daily DD, 10% overall DD)')
print('='*100)
print()
print(f"{'Config':<35} {'Annual':<10} {'MaxDD':<10} {'WorstDay':<10} {'BestDay':<10} {'FTMO?':<8}")
print('-'*100)

# Test configs by scaling raw returns
slot_data = {1: raw_1slot, 2: raw_2slot, 3: raw_3slot, 5: raw_5slot}

ftmo_safe = []
for slots in [1, 2, 3, 5]:
    for alloc in [0.02, 0.025, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10, 0.125]:
        for lev in [1.0, 1.5, 2.0]:
            s = calc_stats(slot_data[slots], alloc, lev)

            is_safe = abs(s['worst_day']) < 5.0 and abs(s['max_dd']) < 10.0
            if is_safe:
                total_exp = NUM_PAIRS * slots * alloc * 100
                ftmo_safe.append((slots, alloc, lev, total_exp, s))

# Sort by annual return
ftmo_safe.sort(key=lambda x: x[4]['annual'], reverse=True)

# Show top 15 FTMO-safe configs
print("TOP 15 FTMO-SAFE CONFIGURATIONS:")
print()
for i, (slots, alloc, lev, exp, s) in enumerate(ftmo_safe[:15]):
    desc = f"{slots} slot @ {alloc*100:.1f}%, {lev}x lev, {exp:.0f}% exp"
    safe = "SAFE"
    print(f"{desc:<35} {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['worst_day']:<10.2f} {s['best_day']:<10.2f} {safe:<8}")

print()
print('='*100)
print('BEST FTMO CONFIG')
print('='*100)
if ftmo_safe:
    slots, alloc, lev, exp, s = ftmo_safe[0]
    print(f"\n{slots} slot(s) @ {alloc*100:.1f}% per pair, {lev}x leverage")
    print(f"  Total max exposure: {exp:.0f}%")
    print(f"  Annual return: {s['annual']:.1f}%")
    print(f"  Max drawdown: {s['max_dd']:.1f}%")
    print(f"  Worst single day: {s['worst_day']:.2f}%")
    print(f"  Best single day: {s['best_day']:.2f}%")
    print(f"  Sharpe: {s['sharpe']:.2f}")
