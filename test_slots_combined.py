"""
COMBINED SLOT + SCALING STRATEGY
- Up to 5 slots per pair (independent stops)
- Position scaling by total concurrent positions across all pairs
- Proper leverage
- Exclude underperforming pairs (USDCHF, USDCAD)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Exclude the bad pairs identified earlier
GOOD_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'NZDUSD']
HORIZON = 20
MAX_SLOTS_PER_PAIR = 5

print("=" * 80)
print("COMBINED SLOT + SCALING STRATEGY")
print(f"Pairs: {len(GOOD_PAIRS)} (excluding USDCHF, USDCAD)")
print(f"Horizon: {HORIZON}d, Max {MAX_SLOTS_PER_PAIR} slots per pair")
print("=" * 80)
print()

# Load predictions
preds_df = pd.read_csv(f'walkforward_predictions_{HORIZON}d.csv')
preds_df['date'] = pd.to_datetime(preds_df['date'])

# Filter to good pairs only
preds_df = preds_df[preds_df['pair'].isin(GOOD_PAIRS)]
print(f"Loaded {len(preds_df)} predictions for {len(GOOD_PAIRS)} good pairs")
print()


def test_slots_with_scaling(confidence=0.55, stop_loss_pct=None, leverage=2.0,
                             base_position=0.10, scale_by_signals=True):
    """
    Test slot-based system with position scaling.

    - Each pair can have up to MAX_SLOTS_PER_PAIR concurrent positions
    - Each slot has independent entry price and optional stop loss
    - Position size is scaled by number of SAME-DAY signals (not all open positions)
    """
    SPREAD_PCT = 0.00015

    # Track open positions: {pair: [(entry_date, entry_price, slot_id, pos_size), ...]}
    open_positions = {pair: [] for pair in GOOD_PAIRS}

    equity = [10000]
    all_trades = []
    daily_pnl = []

    # Get unique dates sorted
    dates = sorted(preds_df['date'].unique())

    for current_date in dates:
        # Get today's data
        today = preds_df[preds_df['date'] == current_date]

        # 1. Check for exits (positions that have reached hold time)
        daily_realized = 0
        for pair in GOOD_PAIRS:
            positions_to_close = []
            for pos in open_positions[pair]:
                entry_date, entry_price, slot_id, pos_size = pos
                days_held = (current_date - entry_date).days

                if days_held >= HORIZON:
                    # Find exit price from predictions
                    exit_row = today[today['pair'] == pair]
                    if len(exit_row) > 0:
                        # Use entry_price from current row as proxy for exit
                        # (it's the open price of current day)
                        exit_price = exit_row.iloc[0]['entry_price']

                        # Calculate PnL
                        raw_return = (exit_price / entry_price) - 1
                        pnl_after_spread = raw_return - (SPREAD_PCT * 2)
                        trade_pnl = pnl_after_spread * pos_size * leverage

                        daily_realized += trade_pnl

                        all_trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'pair': pair,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'raw_return': raw_return * 100,
                            'pnl': trade_pnl * 100,
                            'exit_reason': 'hold_time',
                            'days_held': days_held
                        })

                        positions_to_close.append(pos)

            # Remove closed positions
            for pos in positions_to_close:
                open_positions[pair].remove(pos)

        # 2. Check stop losses on remaining positions
        if stop_loss_pct:
            for pair in GOOD_PAIRS:
                positions_to_close = []
                current_row = today[today['pair'] == pair]
                if len(current_row) == 0:
                    continue

                # Use low price as worst case for stop check
                # Approximate: entry_price * (1 - daily range) as low
                current_price = current_row.iloc[0]['entry_price']

                for pos in open_positions[pair]:
                    entry_date, entry_price, slot_id, pos_size = pos

                    # Check if stop triggered (simplified - using close as proxy)
                    current_return = (current_price / entry_price) - 1

                    if current_return <= -stop_loss_pct:
                        # Stop triggered
                        pnl_after_spread = -stop_loss_pct - (SPREAD_PCT * 2)
                        trade_pnl = pnl_after_spread * pos_size * leverage

                        daily_realized += trade_pnl

                        all_trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'pair': pair,
                            'entry_price': entry_price,
                            'exit_price': entry_price * (1 - stop_loss_pct),
                            'raw_return': -stop_loss_pct * 100,
                            'pnl': trade_pnl * 100,
                            'exit_reason': 'stop_loss',
                            'days_held': (current_date - entry_date).days
                        })

                        positions_to_close.append(pos)

                for pos in positions_to_close:
                    open_positions[pair].remove(pos)

        # 3. Count SAME-DAY signals for scaling (not all open positions!)
        # First identify all pairs with signals today
        todays_signals = []
        for pair in GOOD_PAIRS:
            pair_row = today[today['pair'] == pair]
            if len(pair_row) == 0:
                continue
            prob = pair_row.iloc[0]['prob']
            if prob >= confidence and len(open_positions[pair]) < MAX_SLOTS_PER_PAIR:
                todays_signals.append((pair, pair_row.iloc[0]))

        n_signals_today = len(todays_signals)

        # 4. Open new positions based on signals
        for pair, row in todays_signals:
            entry_price = row['entry_price']

            # Calculate position size - scale by same-day signals only
            if scale_by_signals and n_signals_today > 1:
                pos_size = base_position / n_signals_today
            else:
                pos_size = base_position

            slot_id = len(open_positions[pair])
            open_positions[pair].append((current_date, entry_price, slot_id, pos_size))

        # Update equity
        if daily_realized != 0:
            equity.append(equity[-1] * (1 + daily_realized))
            daily_pnl.append({'date': current_date, 'pnl': daily_realized * 100})
        else:
            equity.append(equity[-1])

    # Close any remaining positions at end
    # (simplified - just count them)
    remaining = sum(len(p) for p in open_positions.values())

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)

    # Calculate metrics
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_pnl = trades_df['pnl'].mean()

    years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
    total_return = equity[-1] / 10000 - 1
    annual = ((1 + total_return) ** (1/years) - 1) * 100 if years > 0 and total_return > -1 else -100

    # Max drawdown
    peak = 10000
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Stop loss stats
    n_stops = len(trades_df[trades_df['exit_reason'] == 'stop_loss']) if stop_loss_pct else 0

    return {
        'trades': n_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'annual': annual,
        'max_dd': max_dd * 100,
        'final': equity[-1],
        'n_stops': n_stops,
        'remaining': remaining,
        'trades_df': trades_df
    }


# ============================================================================
# Test 1: Compare scaling vs no scaling
# ============================================================================
print("=" * 80)
print("TEST 1: Position Scaling Impact (55% conf, 2x leverage)")
print("=" * 80)
print()

print(f"{'Scaling':>10} {'Stop':>8} {'Trades':>8} {'WinRate':>9} {'Annual':>10} {'MaxDD':>8} {'Final':>12}")
print("-" * 75)

for scaling in [False, True]:
    for stop in [None, 0.03]:
        result = test_slots_with_scaling(
            confidence=0.55,
            stop_loss_pct=stop,
            leverage=2.0,
            base_position=0.10,
            scale_by_signals=scaling
        )
        if result:
            scaling_str = "Yes" if scaling else "No"
            stop_str = f"{stop*100:.1f}%" if stop else "None"
            print(f"{scaling_str:>10} {stop_str:>8} {result['trades']:>8} {result['win_rate']:>8.1f}% "
                  f"{result['annual']:>9.1f}% {result['max_dd']:>7.1f}% ${result['final']:>10,.0f}")

print()


# ============================================================================
# Test 2: Leverage comparison with scaling
# ============================================================================
print("=" * 80)
print("TEST 2: Leverage Comparison (55% conf, with scaling)")
print("=" * 80)
print()

print(f"{'Leverage':>10} {'Stop':>8} {'Trades':>8} {'WinRate':>9} {'Annual':>10} {'MaxDD':>8} {'Final':>12}")
print("-" * 75)

for lev in [1.0, 2.0, 3.0, 5.0]:
    for stop in [None, 0.03]:
        result = test_slots_with_scaling(
            confidence=0.55,
            stop_loss_pct=stop,
            leverage=lev,
            base_position=0.10,
            scale_by_signals=True
        )
        if result:
            stop_str = f"{stop*100:.1f}%" if stop else "None"
            print(f"{lev:>9.1f}x {stop_str:>8} {result['trades']:>8} {result['win_rate']:>8.1f}% "
                  f"{result['annual']:>9.1f}% {result['max_dd']:>7.1f}% ${result['final']:>10,.0f}")

print()


# ============================================================================
# Test 3: Higher base position with scaling (since scaling divides it)
# ============================================================================
print("=" * 80)
print("TEST 3: Base Position Size (5x leverage, scaling ON)")
print("=" * 80)
print()

print(f"{'BasePos':>10} {'Stop':>8} {'Trades':>8} {'WinRate':>9} {'Annual':>10} {'MaxDD':>8} {'Final':>12}")
print("-" * 75)

for base_pos in [0.10, 0.20, 0.30, 0.50]:
    for stop in [None, 0.03]:
        result = test_slots_with_scaling(
            confidence=0.55,
            stop_loss_pct=stop,
            leverage=5.0,
            base_position=base_pos,
            scale_by_signals=True
        )
        if result:
            stop_str = f"{stop*100:.1f}%" if stop else "None"
            print(f"{base_pos*100:>9.0f}% {stop_str:>8} {result['trades']:>8} {result['win_rate']:>8.1f}% "
                  f"{result['annual']:>9.1f}% {result['max_dd']:>7.1f}% ${result['final']:>10,.0f}")

print()


# ============================================================================
# Test 4: Confidence thresholds with best config
# ============================================================================
print("=" * 80)
print("TEST 4: Confidence Thresholds (5x lev, 30% base pos, scaling)")
print("=" * 80)
print()

print(f"{'Conf':>8} {'Stop':>8} {'Trades':>8} {'WinRate':>9} {'Annual':>10} {'MaxDD':>8} {'Ret/DD':>8}")
print("-" * 70)

for conf in [0.50, 0.55, 0.60, 0.65]:
    for stop in [None, 0.03]:
        result = test_slots_with_scaling(
            confidence=conf,
            stop_loss_pct=stop,
            leverage=5.0,
            base_position=0.30,
            scale_by_signals=True
        )
        if result:
            stop_str = f"{stop*100:.1f}%" if stop else "None"
            ret_dd = result['annual'] / result['max_dd'] if result['max_dd'] > 0 else 0
            print(f"{conf*100:>7.0f}% {stop_str:>8} {result['trades']:>8} {result['win_rate']:>8.1f}% "
                  f"{result['annual']:>9.1f}% {result['max_dd']:>7.1f}% {ret_dd:>7.2f}")

print()


# ============================================================================
# Test 5: Best config detailed analysis
# ============================================================================
print("=" * 80)
print("BEST CONFIG ANALYSIS")
print("=" * 80)
print()

result = test_slots_with_scaling(
    confidence=0.55,
    stop_loss_pct=0.03,
    leverage=5.0,
    base_position=0.30,
    scale_by_signals=True
)

if result:
    print(f"Config: 55% conf, 3% stop, 5x leverage, 30% base position, scaling ON")
    print()
    print(f"Total Trades: {result['trades']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Avg PnL per trade: {result['avg_pnl']:.2f}%")
    print(f"Stops Triggered: {result['n_stops']}")
    print()
    print(f"Annual Return: {result['annual']:.1f}%")
    print(f"Max Drawdown: {result['max_dd']:.1f}%")
    print(f"Return/DD Ratio: {result['annual']/result['max_dd']:.2f}")
    print(f"Final: $10,000 -> ${result['final']:,.0f}")
    print()

    trades_df = result['trades_df']

    # By pair
    print("BY PAIR:")
    print(f"{'Pair':<8} {'Trades':>8} {'WinRate':>10} {'AvgPnL':>10} {'Stops':>8}")
    print("-" * 50)
    for pair in GOOD_PAIRS:
        pt = trades_df[trades_df['pair'] == pair]
        if len(pt) > 0:
            wr = (pt['pnl'] > 0).mean() * 100
            avg = pt['pnl'].mean()
            stops = len(pt[pt['exit_reason'] == 'stop_loss'])
            print(f"{pair:<8} {len(pt):>8} {wr:>9.1f}% {avg:>9.2f}% {stops:>8}")

    print()

    # By exit reason
    print("BY EXIT REASON:")
    for reason in trades_df['exit_reason'].unique():
        rt = trades_df[trades_df['exit_reason'] == reason]
        wr = (rt['pnl'] > 0).mean() * 100
        print(f"  {reason}: {len(rt)} trades, {wr:.1f}% win rate")

    print()

    # Average days held
    print(f"Average days held: {trades_df['days_held'].mean():.1f}")
    print(f"Hold time exits avg days: {trades_df[trades_df['exit_reason']=='hold_time']['days_held'].mean():.1f}")
    if result['n_stops'] > 0:
        print(f"Stop loss exits avg days: {trades_df[trades_df['exit_reason']=='stop_loss']['days_held'].mean():.1f}")

print()
print("=" * 80)
