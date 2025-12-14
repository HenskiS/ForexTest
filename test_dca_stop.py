"""
Test DCA (Dollar Cost Average) stop loss approach for netting accounts.
When adding to a position, we average the stop losses.
"""
import pandas as pd
import numpy as np
import pickle

# 8-pair Sleep Well portfolio
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']

# Load predictions
pair_data = {}
for pair in ALL_PAIRS:
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
ALLOC_PER_SLOT = 0.225  # 22.5% per slot for ~100% annual return with 8 pairs


def run_netting_dca_stop(max_slots=5, full_size_per_trade=True, close_on_opposite=False, tighten_sl=None):
    """
    Simulate netting account with independent stop losses per position.
    Each position has its own 2% SL from its entry price.
    Track each 'virtual slot' for time-based exits.

    If full_size_per_trade=True, each trade gets full allocation (default).
    If close_on_opposite=True, close all positions when opposite signal received.
    tighten_sl parameter kept for backwards compatibility but not used with independent stops.
    """
    all_trades = []
    daily_pnl = {}

    for pair in ALL_PAIRS:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []

        # Track virtual positions - each with independent stop
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

            # Process existing positions with INDEPENDENT stops
            remaining = []
            for vp in virtual_positions:
                direction = vp['direction']
                days_held = current_idx - vp['entry_idx']

                # Check individual stop loss for this position
                sl_hit = False
                if direction == 1:
                    sl_price = vp['entry_price'] * (1 - SL_PCT)
                    if current_row['low'] <= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                else:
                    sl_price = vp['entry_price'] * (1 + SL_PCT)
                    if current_row['high'] >= sl_price:
                        sl_hit = True
                        exit_price = sl_price * (1 + spread)
                        pnl = (vp['entry_price'] / exit_price) - 1

                if sl_hit:
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'SL'})
                    alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                    if vp['entry_date'] not in daily_pnl:
                        daily_pnl[vp['entry_date']] = 0.0
                    daily_pnl[vp['entry_date']] += alloc * pnl
                    continue  # Position closed, don't add to remaining

                # Check time exit
                if days_held >= HOLD_DAYS:
                    if direction == 1:
                        exit_price = current_row['close'] * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                    else:
                        exit_price = current_row['close'] * (1 + spread)
                        pnl = (vp['entry_price'] / exit_price) - 1
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'TIME'})
                    alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                    if vp['entry_date'] not in daily_pnl:
                        daily_pnl[vp['entry_date']] = 0.0
                    daily_pnl[vp['entry_date']] += alloc * pnl
                    continue  # Position closed

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
                    if close_on_opposite:
                        # Close all positions at current close price, then allow new entry
                        direction = virtual_positions[0]['direction']
                        for vp in virtual_positions:
                            if direction == 1:
                                exit_price = current_row['close'] * (1 - spread)
                                pnl = (exit_price / vp['entry_price']) - 1
                            else:
                                exit_price = current_row['close'] * (1 + spread)
                                pnl = (vp['entry_price'] / exit_price) - 1
                            all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'OPPOSITE'})
                            alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                            if vp['entry_date'] not in daily_pnl:
                                daily_pnl[vp['entry_date']] = 0.0
                            daily_pnl[vp['entry_date']] += alloc * pnl
                        virtual_positions = []
                        can_add = True
                    else:
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


def calc_yearly_metrics(trades, pnl_dict, leverage=LEVERAGE):
    """Calculate yearly performance metrics"""
    if not trades or not pnl_dict:
        return {}

    returns = pd.Series(pnl_dict).sort_index()
    leveraged = returns * leverage
    trades_df = pd.DataFrame(trades)

    yearly = {}
    for year in sorted(returns.index.year.unique()):
        year_mask = returns.index.year == year
        year_returns = leveraged[year_mask]

        if len(year_returns) < 20:
            continue

        # Build equity curve for the year
        equity = [1000]
        for r in year_returns:
            equity.append(equity[-1] * (1 + r))
        equity = np.array(equity[1:])

        year_total = equity[-1] / 1000 - 1
        year_sharpe = (year_returns.mean() / year_returns.std()) * np.sqrt(252) if year_returns.std() > 0 else 0
        cummax = np.maximum.accumulate(equity)
        year_dd = ((equity - cummax) / cummax).min() * 100

        year_trades = trades_df[trades_df['date'].dt.year == year]
        year_wr = (year_trades['pnl'] > 0).mean() * 100 if len(year_trades) > 0 else 0
        year_sl = sum(1 for _, t in year_trades.iterrows() if t.get('exit') == 'SL')

        yearly[year] = {
            'return': year_total * 100,
            'sharpe': year_sharpe,
            'max_dd': year_dd,
            'win_rate': year_wr,
            'trades': len(year_trades),
            'sl_exits': year_sl
        }

    return yearly


if __name__ == '__main__':
    print('=' * 70)
    print('8-PAIR SLEEP WELL BACKTEST (Independent Stops)')
    print('=' * 70)
    print()
    print('Config: 8 pairs, 22.5% alloc/slot, 1.5x leverage, 2% SL, 5-day hold')
    print()

    # Run backtest with independent stops
    print('Running backtest...')
    trades, pnl = run_netting_dca_stop(max_slots=5, full_size_per_trade=True)
    m = calc_metrics(trades, pnl)
    yearly = calc_yearly_metrics(trades, pnl)

    print()
    print('=' * 70)
    print('OVERALL RESULTS')
    print('=' * 70)
    print(f'Trades:     {m["trades"]:,}')
    print(f'Win Rate:   {m["win_rate"]:.1f}%')
    print(f'Annual:     {m["annual"]:.1f}%')
    print(f'Sharpe:     {m["sharpe"]:.2f}')
    print(f'Max DD:     {m["max_dd"]:.1f}%')

    print()
    print('=' * 70)
    print('YEARLY PERFORMANCE')
    print('=' * 70)
    print(f'{"Year":<6} {"Return":>10} {"MaxDD":>10} {"WinRate":>10} {"Trades":>8} {"SL Exits":>10}')
    print('-' * 70)

    for year in sorted(yearly.keys()):
        y = yearly[year]
        print(f'{year:<6} {y["return"]:>9.1f}% {y["max_dd"]:>9.1f}% {y["win_rate"]:>9.1f}% {y["trades"]:>8} {y["sl_exits"]:>10}')
