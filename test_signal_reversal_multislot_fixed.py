"""Test signal reversal with multi-slot system - FIXED for 8 pairs"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
NUM_PAIRS = len(ALL_PAIRS)
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

                    all_trades.append({'entry_date': slot['entry_date'], 'pnl': pnl, 'exit_type': 'REV'})
                    if slot['entry_date'] not in daily_pnl:
                        daily_pnl[slot['entry_date']] = 0.0
                    daily_pnl[slot['entry_date']] += alloc_per_slot * pnl

                slots = []

            # Add new slot on signal
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

    return all_trades, daily_pnl


def calc_stats(all_trades, daily_pnl, leverage):
    if not all_trades:
        return None
    trades_df = pd.DataFrame(all_trades)
    wins = (trades_df['pnl'] > 0).sum()
    total = len(trades_df)

    daily_returns = pd.Series(daily_pnl).sort_index()
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365
    total_return = (1 + daily_returns * leverage).prod() - 1
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100

    cumulative = (1 + daily_returns * leverage).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    sharpe = (daily_returns.mean() * leverage * 252) / (daily_returns.std() * leverage * np.sqrt(252))

    return {
        'trades': total, 'win_rate': wins/total*100,
        'annual': annual_return, 'max_dd': max_dd, 'sharpe': sharpe
    }


print()
print('='*110)
print('SIGNAL REVERSAL - PROPER ALLOCATION (8 pairs, 2x broker leverage)')
print('='*110)
print()
print('With 8 pairs, total exposure = 8 pairs × slots × alloc_per_slot')
print('E.g., 1 slot @ 12.5% per pair = 100% total account exposure (before 2x leverage)')
print()
print(f"{'Config':<25} {'Slots':<6} {'Alloc':<8} {'TotalExp':<10} {'Annual':<10} {'MaxDD':<10} {'Sharpe':<8} {'Ret/DD':<8}")
print('-'*110)

# Configs: (max_slots_per_pair, alloc_per_slot, description)
# Total exposure = 8 pairs × max_slots × alloc_per_slot
configs = [
    # Single slot configs
    (1, 0.125, "1 slot @ 12.5%"),     # 8 × 1 × 12.5% = 100%
    (1, 0.10, "1 slot @ 10%"),        # 8 × 1 × 10% = 80%
    (1, 0.0625, "1 slot @ 6.25%"),    # 8 × 1 × 6.25% = 50%

    # Multi-slot configs matching Sleep Well exposure
    (5, 0.025, "5 slots @ 2.5%"),     # 8 × 5 × 2.5% = 100%
    (5, 0.0225, "5 slots @ 2.25%"),   # 8 × 5 × 2.25% = 90% (like Sleep Well but per-pair capped)
    (5, 0.02, "5 slots @ 2%"),        # 8 × 5 × 2% = 80%

    # Higher implicit leverage (like Sleep Well's 22.5% × 5 = 112.5% per pair)
    (5, 0.05, "5 slots @ 5%"),        # 8 × 5 × 5% = 200%
    (5, 0.0625, "5 slots @ 6.25%"),   # 8 × 5 × 6.25% = 250%

    # Sleep Well equivalent: 22.5% per slot, but that's PER PAIR not total
    # Sleep Well max = 8 pairs × 5 slots × 22.5% = 900% !!!
    (5, 0.225, "5 slots @ 22.5% (SW)"), # Sleep Well config
]

for max_slots, alloc, desc in configs:
    trades, pnl = run_strategy(max_slots, alloc)
    s = calc_stats(trades, pnl, LEVERAGE)
    if s:
        total_exp = NUM_PAIRS * max_slots * alloc * 100
        ret_dd = abs(s['annual'] / s['max_dd'])
        print(f"{desc:<25} {max_slots:<6} {alloc*100:<8.2f}% {total_exp:<10.0f}% {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['sharpe']:<8.2f} {ret_dd:<8.2f}")

print()
print('='*110)
print('APPLES-TO-APPLES: Signal Reversal vs Sleep Well at SAME total exposure')
print('='*110)
print()

# Sleep Well typically has ~6.4 avg slots open × 22.5% = ~144% average exposure
# At 2x leverage = ~2.9x effective leverage
# Let's test signal reversal at similar exposures

print("Sleep Well reality check:")
print("  - 22.5% per slot × up to 5 slots × 8 pairs = 900% MAX exposure")
print("  - Average ~6.4 positions open = ~144% average exposure")
print("  - At 2x leverage = ~2.9x effective average leverage")
print()

# Fair comparison: same average exposure
print(f"{'Strategy':<35} {'AvgExp':<10} {'Annual':<10} {'MaxDD':<10} {'Sharpe':<8}")
print('-'*80)

# Signal reversal at ~100% exposure (conservative)
trades, pnl = run_strategy(1, 0.125)  # 1 slot @ 12.5% × 8 = 100%
s = calc_stats(trades, pnl, LEVERAGE)
print(f"{'Sig Rev: 1 slot @ 12.5%':<35} {'100%':<10} {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['sharpe']:<8.2f}")

# Signal reversal at ~200% exposure (moderate)
trades, pnl = run_strategy(2, 0.125)  # 2 slots @ 12.5% × 8 = 200%
s = calc_stats(trades, pnl, LEVERAGE)
print(f"{'Sig Rev: 2 slots @ 12.5%':<35} {'200%':<10} {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['sharpe']:<8.2f}")

# Signal reversal matching Sleep Well's slot config
trades, pnl = run_strategy(5, 0.225)  # Same as Sleep Well
s = calc_stats(trades, pnl, LEVERAGE)
print(f"{'Sig Rev: 5 slots @ 22.5% (SW cfg)':<35} {'900%':<10} {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['sharpe']:<8.2f}")
