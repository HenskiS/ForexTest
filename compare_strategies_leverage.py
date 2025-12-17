"""Compare Sleep Well vs Signal Reversal at different leverage levels"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT = 200, 50, 10, 90
SL_PCT = 0.025
ALLOC_PER_SLOT = 0.225
MAX_SLOTS = 5
HOLD_DAYS = 5

# Load data
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

def run_sleep_well():
    '''Current 5-day hold multi-slot strategy'''
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

            # Time exits
            remaining = []
            for slot in slots:
                days_held = (current_idx - slot['entry_idx'])
                if days_held >= HOLD_DAYS:
                    exit_price = current_row['open']
                    if slot['direction'] == 1:
                        actual_exit = exit_price * (1 - spread)
                        pnl = (actual_exit / slot['entry_price']) - 1
                    else:
                        actual_exit = exit_price * (1 + spread)
                        pnl = (slot['entry_price'] / actual_exit) - 1
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += ALLOC_PER_SLOT * pnl
                else:
                    remaining.append(slot)
            slots = remaining

            # Stop losses
            if slots:
                direction = slots[0]['direction']
                still_open = []
                for slot in slots:
                    if direction == 1:
                        sl_price = slot['entry_price'] * (1 - SL_PCT)
                        sl_hit = current_row['low'] <= sl_price
                    else:
                        sl_price = slot['entry_price'] * (1 + SL_PCT)
                        sl_hit = current_row['high'] >= sl_price
                    if sl_hit:
                        if direction == 1:
                            exit_price = sl_price * (1 - spread)
                            pnl = (exit_price / slot['entry_price']) - 1
                        else:
                            exit_price = sl_price * (1 + spread)
                            pnl = (slot['entry_price'] / exit_price) - 1
                        if slot['entry_date'] not in daily_pnl:
                            daily_pnl[slot['entry_date']] = 0.0
                        daily_pnl[slot['entry_date']] += ALLOC_PER_SLOT * pnl
                    else:
                        still_open.append(slot)
                slots = still_open

            # New signals
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)

            if signal != 0:
                can_add = True
                if slots and slots[0]['direction'] != signal:
                    can_add = False
                if len(slots) >= MAX_SLOTS:
                    can_add = False
                if can_add:
                    entry_price = current_row['open']
                    actual_entry = entry_price * (1 + spread) if signal == 1 else entry_price * (1 - spread)
                    slots.append({'entry_price': actual_entry, 'entry_idx': current_idx, 'entry_date': current_date, 'direction': signal})
    return daily_pnl

def run_signal_reversal():
    '''Signal reversal with 2.5%/2.5% SL/TP'''
    daily_pnl = {}
    SL, TP = 0.025, 0.025
    for pair in ALL_PAIRS:
        if pair not in pair_data:
            continue
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []
        position = None

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

            if position:
                exit_type = None
                if position['direction'] == 1:
                    if current_row['low'] <= position['entry_price'] * (1 - SL):
                        exit_type = 'SL'
                        exit_price = position['entry_price'] * (1 - SL) * (1 - spread)
                    elif current_row['high'] >= position['entry_price'] * (1 + TP):
                        exit_type = 'TP'
                        exit_price = position['entry_price'] * (1 + TP) * (1 - spread)
                else:
                    if current_row['high'] >= position['entry_price'] * (1 + SL):
                        exit_type = 'SL'
                        exit_price = position['entry_price'] * (1 + SL) * (1 + spread)
                    elif current_row['low'] <= position['entry_price'] * (1 - TP):
                        exit_type = 'TP'
                        exit_price = position['entry_price'] * (1 - TP) * (1 + spread)
                if exit_type:
                    if position['direction'] == 1:
                        pnl = (exit_price / position['entry_price']) - 1
                    else:
                        pnl = (position['entry_price'] / exit_price) - 1
                    if position['entry_date'] not in daily_pnl:
                        daily_pnl[position['entry_date']] = 0.0
                    daily_pnl[position['entry_date']] += pnl
                    position = None

            if position and signal != 0 and signal != position['direction']:
                exit_price = current_row['open']
                if position['direction'] == 1:
                    actual_exit = exit_price * (1 - spread)
                    pnl = (actual_exit / position['entry_price']) - 1
                else:
                    actual_exit = exit_price * (1 + spread)
                    pnl = (position['entry_price'] / actual_exit) - 1
                if position['entry_date'] not in daily_pnl:
                    daily_pnl[position['entry_date']] = 0.0
                daily_pnl[position['entry_date']] += pnl
                position = None

            if signal != 0 and position is None:
                entry_price = current_row['open']
                actual_entry = entry_price * (1 + spread) if signal == 1 else entry_price * (1 - spread)
                position = {'direction': signal, 'entry_price': actual_entry, 'entry_date': current_date}
    return daily_pnl

def calc_stats(daily_pnl, leverage):
    daily_returns = pd.Series(daily_pnl).sort_index()
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365
    cumulative = (1 + daily_returns * leverage).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    sharpe = (daily_returns.mean() * leverage * 252) / (daily_returns.std() * leverage * np.sqrt(252))
    return annual_return, max_dd, sharpe

print('Running backtests...')
sleep_well_pnl = run_sleep_well()
reversal_pnl = run_signal_reversal()

print()
print('='*90)
print('STRATEGY COMPARISON AT DIFFERENT LEVERAGE LEVELS')
print('='*90)
print()
print(f"{'Strategy':<25} {'Leverage':<10} {'Annual':<12} {'Max DD':<12} {'Sharpe':<10} {'Return/DD':<10}")
print('-'*90)

for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    sw_ann, sw_dd, sw_sharpe = calc_stats(sleep_well_pnl, lev)
    print(f"{'Sleep Well (5-day)':<25} {lev:<10.1f} {sw_ann:<12.1f} {sw_dd:<12.1f} {sw_sharpe:<10.2f} {abs(sw_ann/sw_dd):<10.2f}")

print()
for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    rev_ann, rev_dd, rev_sharpe = calc_stats(reversal_pnl, lev)
    print(f"{'Signal Reversal 2.5/2.5':<25} {lev:<10.1f} {rev_ann:<12.1f} {rev_dd:<12.1f} {rev_sharpe:<10.2f} {abs(rev_ann/rev_dd):<10.2f}")

print()
print('='*90)
print('MATCHED COMPARISONS')
print('='*90)
print()

# Match by annual return
print("To get ~94% annual (current Sleep Well @ 2x):")
for lev in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rev_ann, rev_dd, _ = calc_stats(reversal_pnl, lev)
    if 80 < rev_ann < 110:
        print(f"  Signal Reversal @ {lev}x: {rev_ann:.1f}% annual, {rev_dd:.1f}% max DD")

print()
print("To get ~464% annual (Signal Reversal @ 2x):")
for lev in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    sw_ann, sw_dd, _ = calc_stats(sleep_well_pnl, lev)
    if sw_ann >= 400:
        print(f"  Sleep Well @ {lev}x: {sw_ann:.1f}% annual, {sw_dd:.1f}% max DD")
        break

print()
print("To get ~35% max DD (Signal Reversal @ 2x):")
for lev in [3.0, 3.5, 4.0, 4.5, 5.0]:
    sw_ann, sw_dd, _ = calc_stats(sleep_well_pnl, lev)
    if abs(sw_dd) >= 30:
        print(f"  Sleep Well @ {lev}x: {sw_ann:.1f}% annual, {sw_dd:.1f}% max DD")
        break
