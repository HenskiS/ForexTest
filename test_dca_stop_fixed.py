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

BUFFER_SIZE, BUFFER_WARMUP, LOWER_PCT, UPPER_PCT, HOLD_DAYS, SL_PCT, LEVERAGE = 200, 50, 10, 90, 5, 0.025, 2.0
ALLOC_PER_SLOT = 0.225  # 22.5% per slot for ~100% annual return with 8 pairs


def run_netting_dca_stop(max_slots=5, full_size_per_trade=True, close_on_opposite=False, tighten_sl=None, check_entry_day_stop=True, buffer_size=None, buffer_warmup=None):
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

    # Use parameter values or defaults
    buf_size = buffer_size if buffer_size is not None else BUFFER_SIZE
    buf_warmup = buffer_warmup if buffer_warmup is not None else BUFFER_WARMUP

    for pair in ALL_PAIRS:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []

        # Track virtual positions - each with independent stop
        virtual_positions = []

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > buf_size:
                prediction_buffer = prediction_buffer[-buf_size:]
            if len(prediction_buffer) < buf_warmup:
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

                # Check time exit (exit at open since we execute at 6am)
                if days_held >= HOLD_DAYS:
                    if direction == 1:
                        exit_price = current_row['open'] * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                    else:
                        exit_price = current_row['open'] * (1 + spread)
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
                        # Close all positions at current open price (execute at 6am), then allow new entry
                        direction = virtual_positions[0]['direction']
                        for vp in virtual_positions:
                            if direction == 1:
                                exit_price = current_row['open'] * (1 - spread)
                                pnl = (exit_price / vp['entry_price']) - 1
                            else:
                                exit_price = current_row['open'] * (1 + spread)
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

                    # Check if entry-day stop is hit (same candle as entry)
                    entry_day_sl_hit = False
                    if check_entry_day_stop:
                        if signal == 1:
                            sl_price = actual_entry * (1 - SL_PCT)
                            if current_row['low'] <= sl_price:
                                entry_day_sl_hit = True
                                exit_price = sl_price * (1 - spread)
                                pnl = (exit_price / actual_entry) - 1
                        else:
                            sl_price = actual_entry * (1 + SL_PCT)
                            if current_row['high'] >= sl_price:
                                entry_day_sl_hit = True
                                exit_price = sl_price * (1 + spread)
                                pnl = (actual_entry / exit_price) - 1

                    if entry_day_sl_hit:
                        # Position stopped out on entry day - record trade but don't add to positions
                        all_trades.append({'pnl': pnl, 'pair': pair, 'date': current_date, 'exit': 'SL_DAY0'})
                        alloc = ALLOC_PER_SLOT if full_size_per_trade else ALLOC_PER_SLOT / max_slots
                        if current_date not in daily_pnl:
                            daily_pnl[current_date] = 0.0
                        daily_pnl[current_date] += alloc * pnl
                    else:
                        # No entry-day stop hit, add position to track
                        virtual_positions.append({
                            'entry_price': actual_entry,
                            'entry_idx': current_idx,
                            'entry_date': current_date,
                            'direction': signal,
                            'size': 1
                        })

    return all_trades, daily_pnl


def run_netting_averaged_stop(max_slots=5, full_size_per_trade=True, check_entry_day_stop=True, buffer_size=None, buffer_warmup=None):
    """
    Simulate netting account with AVERAGED (DCA) stop loss.
    All slots share ONE stop loss at 2.5% below the average entry price.
    This matches how OANDA netting accounts actually work.
    """
    all_trades = []
    daily_pnl = {}

    buf_size = buffer_size if buffer_size is not None else BUFFER_SIZE
    buf_warmup = buffer_warmup if buffer_warmup is not None else BUFFER_WARMUP

    for pair in ALL_PAIRS:
        predictions = pair_data[pair]['predictions']
        test_indices = pair_data[pair]['test_indices']
        df = pair_data[pair]['df']
        prediction_buffer = []
        virtual_positions = []

        for idx, (pred_idx, prediction) in enumerate(zip(test_indices, predictions)):
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > buf_size:
                prediction_buffer = prediction_buffer[-buf_size:]
            if len(prediction_buffer) < buf_warmup:
                continue

            current_idx = pred_idx + 1
            if current_idx >= len(df):
                continue
            current_date = df.index[current_idx]
            if current_date.dayofweek >= 5:
                continue
            current_row = df.iloc[current_idx]
            spread = current_row['spread_pct']

            # Process time exits first (before checking averaged stop)
            remaining = []
            for vp in virtual_positions:
                days_held = current_idx - vp['entry_idx']
                direction = vp['direction']

                if days_held >= HOLD_DAYS:
                    if direction == 1:
                        exit_price = current_row['open'] * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                    else:
                        exit_price = current_row['open'] * (1 + spread)
                        pnl = (vp['entry_price'] / exit_price) - 1
                    all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'TIME'})
                    alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                    if vp['entry_date'] not in daily_pnl:
                        daily_pnl[vp['entry_date']] = 0.0
                    daily_pnl[vp['entry_date']] += alloc * pnl
                else:
                    remaining.append(vp)
            virtual_positions = remaining

            # Check AVERAGED stop loss (all slots share one stop)
            if virtual_positions:
                direction = virtual_positions[0]['direction']
                # Calculate weighted average entry price
                total_size = sum(vp['size'] for vp in virtual_positions)
                avg_entry = sum(vp['entry_price'] * vp['size'] for vp in virtual_positions) / total_size

                # Single stop at 2.5% from average entry
                if direction == 1:
                    avg_sl_price = avg_entry * (1 - SL_PCT)
                    sl_hit = current_row['low'] <= avg_sl_price
                else:
                    avg_sl_price = avg_entry * (1 + SL_PCT)
                    sl_hit = current_row['high'] >= avg_sl_price

                if sl_hit:
                    # ALL positions close at the stop price
                    for vp in virtual_positions:
                        if direction == 1:
                            exit_price = avg_sl_price * (1 - spread)
                            pnl = (exit_price / vp['entry_price']) - 1
                        else:
                            exit_price = avg_sl_price * (1 + spread)
                            pnl = (vp['entry_price'] / exit_price) - 1
                        all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'SL_AVG'})
                        alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                        if vp['entry_date'] not in daily_pnl:
                            daily_pnl[vp['entry_date']] = 0.0
                        daily_pnl[vp['entry_date']] += alloc * pnl
                    virtual_positions = []  # All closed

            # Check for new signal
            lower_thresh = np.percentile(prediction_buffer, LOWER_PCT)
            upper_thresh = np.percentile(prediction_buffer, UPPER_PCT)
            signal = 1 if prediction >= upper_thresh else (-1 if prediction <= lower_thresh else 0)

            if signal != 0:
                can_add = True
                if virtual_positions and virtual_positions[0]['direction'] != signal:
                    can_add = False  # Can't mix directions
                if len(virtual_positions) >= max_slots:
                    can_add = False

                if can_add:
                    entry_price = current_row['open']
                    if signal == 1:
                        actual_entry = entry_price * (1 + spread)
                    else:
                        actual_entry = entry_price * (1 - spread)

                    # Check entry-day stop (using position's avg if exists, else own entry)
                    entry_day_sl_hit = False
                    if check_entry_day_stop:
                        if virtual_positions:
                            # Use new averaged stop including this entry
                            new_total = sum(vp['size'] for vp in virtual_positions) + 1
                            new_avg = (sum(vp['entry_price'] * vp['size'] for vp in virtual_positions) + actual_entry) / new_total
                            if signal == 1:
                                sl_price = new_avg * (1 - SL_PCT)
                                entry_day_sl_hit = current_row['low'] <= sl_price
                            else:
                                sl_price = new_avg * (1 + SL_PCT)
                                entry_day_sl_hit = current_row['high'] >= sl_price
                        else:
                            # No existing position, use own entry
                            if signal == 1:
                                sl_price = actual_entry * (1 - SL_PCT)
                                entry_day_sl_hit = current_row['low'] <= sl_price
                            else:
                                sl_price = actual_entry * (1 + SL_PCT)
                                entry_day_sl_hit = current_row['high'] >= sl_price

                    if entry_day_sl_hit:
                        # Entry day stop - close this entry AND all existing positions
                        if signal == 1:
                            exit_price = sl_price * (1 - spread)
                            pnl = (exit_price / actual_entry) - 1
                        else:
                            exit_price = sl_price * (1 + spread)
                            pnl = (actual_entry / exit_price) - 1
                        all_trades.append({'pnl': pnl, 'pair': pair, 'date': current_date, 'exit': 'SL_DAY0'})
                        alloc = ALLOC_PER_SLOT if full_size_per_trade else ALLOC_PER_SLOT / max_slots
                        if current_date not in daily_pnl:
                            daily_pnl[current_date] = 0.0
                        daily_pnl[current_date] += alloc * pnl

                        # Also close all existing positions at the stop
                        for vp in virtual_positions:
                            if signal == 1:
                                vp_pnl = (exit_price / vp['entry_price']) - 1
                            else:
                                vp_pnl = (vp['entry_price'] / exit_price) - 1
                            all_trades.append({'pnl': vp_pnl, 'pair': pair, 'date': vp['entry_date'], 'exit': 'SL_AVG'})
                            vp_alloc = ALLOC_PER_SLOT * vp['size'] if full_size_per_trade else ALLOC_PER_SLOT / max_slots * vp['size']
                            if vp['entry_date'] not in daily_pnl:
                                daily_pnl[vp['entry_date']] = 0.0
                            daily_pnl[vp['entry_date']] += vp_alloc * vp_pnl
                        virtual_positions = []
                    else:
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
        year_sl_day0 = sum(1 for _, t in year_trades.iterrows() if t.get('exit') == 'SL_DAY0')

        yearly[year] = {
            'return': year_total * 100,
            'sharpe': year_sharpe,
            'max_dd': year_dd,
            'win_rate': year_wr,
            'trades': len(year_trades),
            'sl_exits': year_sl,
            'sl_day0': year_sl_day0
        }

    return yearly


if __name__ == '__main__':
    print('=' * 90)
    print('INDEPENDENT vs AVERAGED (DCA) STOP LOSSES')
    print('=' * 90)
    print('Config: 8 pairs, 22.5% alloc/slot, 2.0x leverage, 2.5% SL, 5-day hold')
    print()

    # Compare independent stops vs averaged stops
    print('INDEPENDENT STOPS (backtest ideal - each slot has own SL):')
    trades_ind, pnl_ind = run_netting_dca_stop(max_slots=5, full_size_per_trade=True, check_entry_day_stop=True)
    m_ind = calc_metrics(trades_ind, pnl_ind)
    sl_ind = sum(1 for t in trades_ind if t.get('exit') in ['SL', 'SL_DAY0'])
    print(f'  Annual: {m_ind["annual"]:.1f}%, Sharpe: {m_ind["sharpe"]:.2f}, MaxDD: {m_ind["max_dd"]:.1f}%, WinRate: {m_ind["win_rate"]:.1f}%')
    print(f'  Trades: {m_ind["trades"]}, SL exits: {sl_ind}')

    print()
    print('AVERAGED STOPS (OANDA reality - all slots share one SL from avg price):')
    trades_avg, pnl_avg = run_netting_averaged_stop(max_slots=5, full_size_per_trade=True, check_entry_day_stop=True)
    m_avg = calc_metrics(trades_avg, pnl_avg)
    sl_avg = sum(1 for t in trades_avg if t.get('exit') in ['SL_AVG', 'SL_DAY0'])
    print(f'  Annual: {m_avg["annual"]:.1f}%, Sharpe: {m_avg["sharpe"]:.2f}, MaxDD: {m_avg["max_dd"]:.1f}%, WinRate: {m_avg["win_rate"]:.1f}%')
    print(f'  Trades: {m_avg["trades"]}, SL exits: {sl_avg}')

    print()
    print('=' * 90)
    print('DIFFERENCE:')
    print(f'  Annual: {m_avg["annual"] - m_ind["annual"]:+.1f}%')
    print(f'  Sharpe: {m_avg["sharpe"] - m_ind["sharpe"]:+.2f}')
    print(f'  MaxDD: {m_avg["max_dd"] - m_ind["max_dd"]:+.1f}%')
    print(f'  SL exits: {sl_avg - sl_ind:+d}')

    # Show yearly comparison
    print()
    print('=' * 90)
    print('YEARLY COMPARISON')
    print('=' * 90)
    yearly_ind = calc_yearly_metrics(trades_ind, pnl_ind)
    yearly_avg = calc_yearly_metrics(trades_avg, pnl_avg)

    print(f'{"Year":<6} {"Indep":>10} {"Avg":>10} {"Diff":>10} {"MaxDD Ind":>10} {"MaxDD Avg":>10}')
    print('-' * 70)
    for year in sorted(yearly_ind.keys()):
        y_ind = yearly_ind.get(year, {})
        y_avg = yearly_avg.get(year, {})
        ret_ind = y_ind.get('return', 0)
        ret_avg = y_avg.get('return', 0)
        dd_ind = y_ind.get('max_dd', 0)
        dd_avg = y_avg.get('max_dd', 0)
        print(f'{year:<6} {ret_ind:>9.1f}% {ret_avg:>9.1f}% {ret_avg - ret_ind:>+9.1f}% {dd_ind:>9.1f}% {dd_avg:>9.1f}%')

    # Show best years for each
    print()
    best_ind_year = max(yearly_ind.items(), key=lambda x: x[1]['return'])
    best_avg_year = max(yearly_avg.items(), key=lambda x: x[1]['return'])
    worst_ind_year = min(yearly_ind.items(), key=lambda x: x[1]['return'])
    worst_avg_year = min(yearly_avg.items(), key=lambda x: x[1]['return'])
    print(f'Best year (Independent): {best_ind_year[0]} at {best_ind_year[1]["return"]:.1f}%')
    print(f'Best year (Averaged):    {best_avg_year[0]} at {best_avg_year[1]["return"]:.1f}%')
    print(f'Worst year (Independent): {worst_ind_year[0]} at {worst_ind_year[1]["return"]:.1f}%')
    print(f'Worst year (Averaged):    {worst_avg_year[0]} at {worst_avg_year[1]["return"]:.1f}%')

    # Keep old yearly print for backward compatibility
    yearly = yearly_avg  # Use averaged stops results going forward
    print()
    print('=' * 90)
    print('YEARLY PERFORMANCE (Averaged Stops - for production)')
    print('=' * 90)
    print(f'{"Year":<6} {"Return":>10} {"MaxDD":>10} {"WinRate":>10} {"Trades":>8}')
    print('-' * 50)
    for year in sorted(yearly.keys()):
        y = yearly[year]
        print(f'{year:<6} {y["return"]:>9.1f}% {y["max_dd"]:>9.1f}% {y["win_rate"]:>9.1f}% {y["trades"]:>8}')
