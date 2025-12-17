"""Test signal reversal strategy with various SL/TP combinations"""
import pickle
import pandas as pd
import numpy as np

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT = 200, 50, 10, 90
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

def run_strategy(sl_pct, tp_pct):
    all_trades = []
    daily_pnl = {}

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

            if prediction >= upper_thresh:
                signal = 1
            elif prediction <= lower_thresh:
                signal = -1
            else:
                signal = 0

            # Check SL/TP if in position
            if position:
                exit_type = None
                exit_price = None

                if position['direction'] == 1:
                    if sl_pct and current_row['low'] <= position['entry_price'] * (1 - sl_pct):
                        exit_type = 'SL'
                        exit_price = position['entry_price'] * (1 - sl_pct) * (1 - spread)
                    elif tp_pct and current_row['high'] >= position['entry_price'] * (1 + tp_pct):
                        exit_type = 'TP'
                        exit_price = position['entry_price'] * (1 + tp_pct) * (1 - spread)
                else:
                    if sl_pct and current_row['high'] >= position['entry_price'] * (1 + sl_pct):
                        exit_type = 'SL'
                        exit_price = position['entry_price'] * (1 + sl_pct) * (1 + spread)
                    elif tp_pct and current_row['low'] <= position['entry_price'] * (1 - tp_pct):
                        exit_type = 'TP'
                        exit_price = position['entry_price'] * (1 - tp_pct) * (1 + spread)

                if exit_type:
                    if position['direction'] == 1:
                        pnl = (exit_price / position['entry_price']) - 1
                    else:
                        pnl = (position['entry_price'] / exit_price) - 1

                    all_trades.append({'entry_date': position['entry_date'], 'pnl': pnl, 'exit_type': exit_type})
                    if position['entry_date'] not in daily_pnl:
                        daily_pnl[position['entry_date']] = 0.0
                    daily_pnl[position['entry_date']] += pnl
                    position = None

            # Exit on opposite signal
            if position and signal != 0 and signal != position['direction']:
                exit_price = current_row['open']
                if position['direction'] == 1:
                    actual_exit = exit_price * (1 - spread)
                    pnl = (actual_exit / position['entry_price']) - 1
                else:
                    actual_exit = exit_price * (1 + spread)
                    pnl = (position['entry_price'] / actual_exit) - 1

                all_trades.append({'entry_date': position['entry_date'], 'pnl': pnl, 'exit_type': 'REV'})
                if position['entry_date'] not in daily_pnl:
                    daily_pnl[position['entry_date']] = 0.0
                daily_pnl[position['entry_date']] += pnl
                position = None

            # Enter on signal if no position
            if signal != 0 and position is None:
                entry_price = current_row['open']
                if signal == 1:
                    actual_entry = entry_price * (1 + spread)
                else:
                    actual_entry = entry_price * (1 - spread)

                position = {'direction': signal, 'entry_price': actual_entry, 'entry_date': current_date}

    return all_trades, daily_pnl

def calc_stats(all_trades, daily_pnl):
    if not all_trades:
        return None
    trades_df = pd.DataFrame(all_trades)
    wins = (trades_df['pnl'] > 0).sum()
    total = len(trades_df)
    sl_exits = (trades_df['exit_type'] == 'SL').sum()
    tp_exits = (trades_df['exit_type'] == 'TP').sum()

    daily_returns = pd.Series(daily_pnl).sort_index()
    years = (daily_returns.index[-1] - daily_returns.index[0]).days / 365
    total_return = (1 + daily_returns * LEVERAGE).prod() - 1
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100

    cumulative = (1 + daily_returns * LEVERAGE).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    sharpe = (daily_returns.mean() * LEVERAGE * 252) / (daily_returns.std() * LEVERAGE * np.sqrt(252))

    return {
        'trades': total, 'win_rate': wins/total*100, 'avg_pnl': trades_df['pnl'].mean()*100,
        'annual': annual_return, 'max_dd': max_dd, 'sharpe': sharpe, 'sl_exits': sl_exits, 'tp_exits': tp_exits
    }

print()
print('='*90)
print('SIGNAL REVERSAL WITH STOP LOSS AND TAKE PROFIT')
print('='*90)
print()
print(f"{'SL/TP':<12} {'Trades':<8} {'WinRate':<9} {'AvgPnL':<9} {'Annual':<10} {'MaxDD':<10} {'Sharpe':<8} {'SL':<6} {'TP':<6}")
print('-'*90)

configs = [
    (None, None, 'None/None'),
    (0.01, 0.01, '1%/1%'),
    (0.01, 0.02, '1%/2%'),
    (0.01, 0.03, '1%/3%'),
    (0.02, 0.02, '2%/2%'),
    (0.02, 0.03, '2%/3%'),
    (0.02, 0.04, '2%/4%'),
    (0.025, 0.025, '2.5%/2.5%'),
    (0.025, 0.03, '2.5%/3%'),
    (0.025, 0.04, '2.5%/4%'),
    (0.025, 0.05, '2.5%/5%'),
    (0.03, 0.03, '3%/3%'),
    (0.03, 0.04, '3%/4%'),
    (0.03, 0.05, '3%/5%'),
    (0.04, 0.04, '4%/4%'),
    (0.05, 0.05, '5%/5%'),
]

for sl, tp, label in configs:
    trades, pnl = run_strategy(sl, tp)
    s = calc_stats(trades, pnl)
    if s:
        print(f"{label:<12} {s['trades']:<8} {s['win_rate']:<9.1f} {s['avg_pnl']:<9.3f} {s['annual']:<10.1f} {s['max_dd']:<10.1f} {s['sharpe']:<8.2f} {s['sl_exits']:<6} {s['tp_exits']:<6}")
