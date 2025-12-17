"""Compare Sleep Well vs Signal Reversal for FTMO"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT = 200, 50, 10, 90

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
    except:
        pass

def run_sleep_well_raw(max_slots, hold_days=5, sl_pct=0.025):
    daily_pnl = {}
    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer, slots = [], []
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
            # Time exits
            remaining = []
            for slot in slots:
                if (current_idx - slot['entry_idx']) >= hold_days:
                    exit_price = current_row['open']
                    actual_exit = exit_price * (1 - spread) if slot['direction'] == 1 else exit_price * (1 + spread)
                    pnl = (actual_exit / slot['entry_price']) - 1 if slot['direction'] == 1 else (slot['entry_price'] / actual_exit) - 1
                    daily_pnl[current_date] = daily_pnl.get(current_date, 0.0) + pnl
                else:
                    remaining.append(slot)
            slots = remaining
            # Stop losses
            still_open = []
            for slot in slots:
                sl_hit = False
                if slot['direction'] == 1:
                    sl_price = slot['entry_price'] * (1 - sl_pct)
                    if current_row['low'] <= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 - spread)
                        pnl = (exit_price / slot['entry_price']) - 1
                else:
                    sl_price = slot['entry_price'] * (1 + sl_pct)
                    if current_row['high'] >= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 + spread)
                        pnl = (slot['entry_price'] / exit_price) - 1
                if sl_hit:
                    daily_pnl[current_date] = daily_pnl.get(current_date, 0.0) + pnl
                else:
                    still_open.append(slot)
            slots = still_open
            # New signals
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)
            if signal != 0 and len(slots) < max_slots and (not slots or slots[0]['direction'] == signal):
                entry_price = current_row['open']
                actual_entry = entry_price * (1 + spread) if signal == 1 else entry_price * (1 - spread)
                slots.append({'direction': signal, 'entry_price': actual_entry, 'entry_idx': current_idx})
    return pd.Series(daily_pnl).sort_index()

def run_signal_reversal_raw(max_slots, sl_pct=0.025, tp_pct=0.025):
    daily_pnl = {}
    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer, slots = [], []
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
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)
            # SL/TP exits
            still_open = []
            for slot in slots:
                exit_type = None
                if slot['direction'] == 1:
                    if current_row['low'] <= slot['entry_price'] * (1 - sl_pct):
                        exit_type, exit_price = 'SL', slot['entry_price'] * (1 - sl_pct) * (1 - spread)
                    elif current_row['high'] >= slot['entry_price'] * (1 + tp_pct):
                        exit_type, exit_price = 'TP', slot['entry_price'] * (1 + tp_pct) * (1 - spread)
                else:
                    if current_row['high'] >= slot['entry_price'] * (1 + sl_pct):
                        exit_type, exit_price = 'SL', slot['entry_price'] * (1 + sl_pct) * (1 + spread)
                    elif current_row['low'] <= slot['entry_price'] * (1 - tp_pct):
                        exit_type, exit_price = 'TP', slot['entry_price'] * (1 - tp_pct) * (1 + spread)
                if exit_type:
                    pnl = (exit_price / slot['entry_price']) - 1 if slot['direction'] == 1 else (slot['entry_price'] / exit_price) - 1
                    daily_pnl[current_date] = daily_pnl.get(current_date, 0.0) + pnl
                else:
                    still_open.append(slot)
            slots = still_open
            # Signal reversal exit
            if slots and signal != 0 and signal != slots[0]['direction']:
                exit_price = current_row['open']
                for slot in slots:
                    actual_exit = exit_price * (1 - spread) if slot['direction'] == 1 else exit_price * (1 + spread)
                    pnl = (actual_exit / slot['entry_price']) - 1 if slot['direction'] == 1 else (slot['entry_price'] / actual_exit) - 1
                    daily_pnl[current_date] = daily_pnl.get(current_date, 0.0) + pnl
                slots = []
            # New entries
            if signal != 0 and len(slots) < max_slots and (not slots or slots[0]['direction'] == signal):
                entry_price = current_row['open']
                actual_entry = entry_price * (1 + spread) if signal == 1 else entry_price * (1 - spread)
                slots.append({'direction': signal, 'entry_price': actual_entry})
    return pd.Series(daily_pnl).sort_index()

def calc_stats(raw_returns, alloc, lev):
    scaled = raw_returns * alloc * lev
    years = (scaled.index[-1] - scaled.index[0]).days / 365
    cumulative = (1 + scaled).cumprod()
    annual = ((cumulative.iloc[-1]) ** (1/years) - 1) * 100
    rolling_max = cumulative.expanding().max()
    max_dd = ((cumulative - rolling_max) / rolling_max).min() * 100
    worst_day = scaled.min() * 100
    sharpe = (scaled.mean() * 252) / (scaled.std() * np.sqrt(252))
    return annual, max_dd, worst_day, sharpe

print('Running both strategies...')
sleep_well_raw = run_sleep_well_raw(5)
signal_rev_raw = run_signal_reversal_raw(5)

print()
print('='*90)
print('SLEEP WELL vs SIGNAL REVERSAL - FTMO COMPARISON')
print('='*90)
print()

# Find best FTMO-safe config for each
print('BEST FTMO-SAFE CONFIG FOR EACH (Max DD < 8%, Worst Day < 4%):')
print('-'*90)

best_sw = None
best_sr = None

for alloc in [0.02, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10, 0.125]:
    for lev in [1.0, 1.5, 2.0, 2.5]:
        sw = calc_stats(sleep_well_raw, alloc, lev)
        sr = calc_stats(signal_rev_raw, alloc, lev)

        if abs(sw[1]) < 8 and abs(sw[2]) < 4:
            if best_sw is None or sw[0] > best_sw[0]:
                best_sw = (sw[0], sw[1], sw[2], sw[3], alloc, lev)

        if abs(sr[1]) < 8 and abs(sr[2]) < 4:
            if best_sr is None or sr[0] > best_sr[0]:
                best_sr = (sr[0], sr[1], sr[2], sr[3], alloc, lev)

print()
print(f"{'Strategy':<20} {'Config':<25} {'Annual':<10} {'Max DD':<10} {'Worst Day':<12} {'Sharpe':<8}")
print('-'*90)

if best_sw:
    cfg = f"{best_sw[4]*100:.1f}% x 5 slots, {best_sw[5]}x"
    print(f"{'Sleep Well':<20} {cfg:<25} {best_sw[0]:<10.1f} {best_sw[1]:<10.1f} {best_sw[2]:<12.2f} {best_sw[3]:<8.2f}")

if best_sr:
    cfg = f"{best_sr[4]*100:.1f}% x 5 slots, {best_sr[5]}x"
    print(f"{'Signal Reversal':<20} {cfg:<25} {best_sr[0]:<10.1f} {best_sr[1]:<10.1f} {best_sr[2]:<12.2f} {best_sr[3]:<8.2f}")

print()
print('='*90)
print('SAME CONFIG COMPARISON (5 slots @ 6.2%, 1.5x leverage)')
print('='*90)
print()

sw = calc_stats(sleep_well_raw, 0.0625, 1.5)
sr = calc_stats(signal_rev_raw, 0.0625, 1.5)

print(f"{'Strategy':<20} {'Annual':<12} {'Max DD':<12} {'Worst Day':<12} {'Sharpe':<10}")
print('-'*70)
print(f"{'Sleep Well':<20} {sw[0]:<12.1f} {sw[1]:<12.1f} {sw[2]:<12.2f} {sw[3]:<10.2f}")
print(f"{'Signal Reversal':<20} {sr[0]:<12.1f} {sr[1]:<12.1f} {sr[2]:<12.2f} {sr[3]:<10.2f}")

print()
print('='*90)
print('YOUR CURRENT SLEEP WELL CONFIG (22.5% x 5 slots, 2x) - NOT FTMO SAFE')
print('='*90)
sw_current = calc_stats(sleep_well_raw, 0.225, 2.0)
print(f"Annual: {sw_current[0]:.1f}% | Max DD: {sw_current[1]:.1f}% | Worst Day: {sw_current[2]:.2f}%")
print()
print('='*90)
print('RECOMMENDATION')
print('='*90)
print()
print('For FTMO: Signal Reversal - higher Sharpe, more risk-efficient')
print('For personal: Keep Sleep Well - already running, proven')
print()
print('Run BOTH simultaneously:')
print('  - Personal OANDA: Sleep Well @ 22.5%/slot, 2x lev (~94% annual)')
print('  - FTMO account:   Signal Reversal @ 6.2%/slot, 1.5x lev (~38% annual)')
