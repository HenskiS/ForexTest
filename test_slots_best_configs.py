"""
BEST SLOT CONFIGS - Detailed analysis
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

GOOD_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'NZDUSD']
HORIZON = 20
MAX_SLOTS_PER_PAIR = 5
SPREAD_PCT = 0.00015

print("=" * 80)
print("BEST SLOT CONFIGURATIONS - Detailed Analysis")
print("=" * 80)
print()

preds_df = pd.read_csv(f'walkforward_predictions_{HORIZON}d.csv')
preds_df['date'] = pd.to_datetime(preds_df['date'])
preds_df = preds_df[preds_df['pair'].isin(GOOD_PAIRS)]


def test_config(confidence, stop_loss_pct, leverage, base_position):
    """Test a specific configuration."""
    open_positions = {pair: [] for pair in GOOD_PAIRS}
    equity = [10000]
    all_trades = []
    dates = sorted(preds_df['date'].unique())

    for current_date in dates:
        today = preds_df[preds_df['date'] == current_date]
        daily_realized = 0

        # Check exits
        for pair in GOOD_PAIRS:
            positions_to_close = []
            for pos in open_positions[pair]:
                entry_date, entry_price, slot_id, pos_size = pos
                days_held = (current_date - entry_date).days

                if days_held >= HORIZON:
                    exit_row = today[today['pair'] == pair]
                    if len(exit_row) > 0:
                        exit_price = exit_row.iloc[0]['entry_price']
                        raw_return = (exit_price / entry_price) - 1
                        pnl_after_spread = raw_return - (SPREAD_PCT * 2)
                        trade_pnl = pnl_after_spread * pos_size * leverage
                        daily_realized += trade_pnl
                        all_trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'pair': pair,
                            'raw_return': raw_return * 100,
                            'pnl': trade_pnl * 100,
                            'exit_reason': 'hold_time',
                            'days_held': days_held
                        })
                        positions_to_close.append(pos)

            for pos in positions_to_close:
                open_positions[pair].remove(pos)

        # Check stops
        if stop_loss_pct:
            for pair in GOOD_PAIRS:
                positions_to_close = []
                current_row = today[today['pair'] == pair]
                if len(current_row) == 0:
                    continue
                current_price = current_row.iloc[0]['entry_price']

                for pos in open_positions[pair]:
                    entry_date, entry_price, slot_id, pos_size = pos
                    current_return = (current_price / entry_price) - 1

                    if current_return <= -stop_loss_pct:
                        pnl_after_spread = -stop_loss_pct - (SPREAD_PCT * 2)
                        trade_pnl = pnl_after_spread * pos_size * leverage
                        daily_realized += trade_pnl
                        all_trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'pair': pair,
                            'raw_return': -stop_loss_pct * 100,
                            'pnl': trade_pnl * 100,
                            'exit_reason': 'stop_loss',
                            'days_held': (current_date - entry_date).days
                        })
                        positions_to_close.append(pos)

                for pos in positions_to_close:
                    open_positions[pair].remove(pos)

        # New signals
        todays_signals = []
        for pair in GOOD_PAIRS:
            pair_row = today[today['pair'] == pair]
            if len(pair_row) == 0:
                continue
            prob = pair_row.iloc[0]['prob']
            if prob >= confidence and len(open_positions[pair]) < MAX_SLOTS_PER_PAIR:
                todays_signals.append((pair, pair_row.iloc[0]))

        n_signals = len(todays_signals)
        for pair, row in todays_signals:
            pos_size = base_position / n_signals if n_signals > 1 else base_position
            open_positions[pair].append((current_date, row['entry_price'], len(open_positions[pair]), pos_size))

        if daily_realized != 0:
            equity.append(equity[-1] * (1 + daily_realized))
        else:
            equity.append(equity[-1])

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)
    years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
    total_return = equity[-1] / 10000 - 1
    annual = ((1 + total_return) ** (1/years) - 1) * 100 if years > 0 and total_return > -1 else -100

    peak = 10000
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': len(trades_df),
        'win_rate': (trades_df['pnl'] > 0).mean() * 100,
        'avg_pnl': trades_df['pnl'].mean(),
        'annual': annual,
        'max_dd': max_dd * 100,
        'final': equity[-1],
        'n_stops': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
        'trades_df': trades_df,
        'equity': equity
    }


# ============================================================================
# Find optimal configuration
# ============================================================================
print("=" * 80)
print("GRID SEARCH: Finding best return/DD ratio with DD < 25%")
print("=" * 80)
print()

results = []
for conf in [0.55, 0.60, 0.65, 0.70]:
    for stop in [None, 0.025, 0.03, 0.035]:
        for lev in [3.0, 4.0, 5.0]:
            for base in [0.20, 0.25, 0.30]:
                r = test_config(conf, stop, lev, base)
                if r and r['max_dd'] < 30:  # Filter out extreme DD
                    results.append({
                        'conf': conf,
                        'stop': stop,
                        'lev': lev,
                        'base': base,
                        **{k: v for k, v in r.items() if k not in ['trades_df', 'equity']}
                    })

results_df = pd.DataFrame(results)
results_df['ret_dd'] = results_df['annual'] / results_df['max_dd']

# Sort by return/DD ratio with DD constraint
good_results = results_df[results_df['max_dd'] < 25].sort_values('ret_dd', ascending=False)

print(f"{'Conf':>6} {'Stop':>6} {'Lev':>5} {'Base':>6} {'Trades':>7} {'WR':>6} {'Annual':>9} {'MaxDD':>7} {'Ret/DD':>7}")
print("-" * 75)

for _, row in good_results.head(15).iterrows():
    stop_str = f"{row['stop']*100:.1f}%" if row['stop'] else "None"
    print(f"{row['conf']*100:>5.0f}% {stop_str:>6} {row['lev']:>4.1f}x {row['base']*100:>5.0f}% "
          f"{row['trades']:>7.0f} {row['win_rate']:>5.1f}% {row['annual']:>8.1f}% {row['max_dd']:>6.1f}% {row['ret_dd']:>6.2f}")

print()


# ============================================================================
# Best config analysis
# ============================================================================
print("=" * 80)
print("TOP 3 CONFIGURATIONS - DETAILED")
print("=" * 80)

for i, (_, row) in enumerate(good_results.head(3).iterrows()):
    print()
    print(f"--- RANK {i+1} ---")
    r = test_config(row['conf'], row['stop'], row['lev'], row['base'])
    stop_str = f"{row['stop']*100:.1f}%" if row['stop'] else "None"
    print(f"Config: {row['conf']*100:.0f}% conf, {stop_str} stop, {row['lev']:.1f}x lev, {row['base']*100:.0f}% base")
    print(f"Trades: {r['trades']}, Win Rate: {r['win_rate']:.1f}%, Stops: {r['n_stops']}")
    print(f"Annual: {r['annual']:.1f}%, Max DD: {r['max_dd']:.1f}%, Return/DD: {r['annual']/r['max_dd']:.2f}")
    print(f"Final: $10,000 -> ${r['final']:,.0f}")

    # By pair
    print("\nBy Pair:")
    for pair in GOOD_PAIRS:
        pt = r['trades_df'][r['trades_df']['pair'] == pair]
        if len(pt) > 0:
            wr = (pt['pnl'] > 0).mean() * 100
            stops = len(pt[pt['exit_reason'] == 'stop_loss'])
            print(f"  {pair}: {len(pt):>3} trades, {wr:>5.1f}% WR, {stops:>2} stops")


# ============================================================================
# Compare with different DD targets
# ============================================================================
print()
print("=" * 80)
print("BEST CONFIGS BY MAX DD TARGET")
print("=" * 80)
print()

for dd_target in [15, 20, 25, 30]:
    subset = results_df[results_df['max_dd'] < dd_target].sort_values('annual', ascending=False)
    if len(subset) > 0:
        best = subset.iloc[0]
        stop_str = f"{best['stop']*100:.1f}%" if best['stop'] else "None"
        print(f"DD < {dd_target}%: {best['conf']*100:.0f}% conf, {stop_str} stop, {best['lev']:.1f}x lev, "
              f"{best['base']*100:.0f}% base -> {best['annual']:.1f}% annual, {best['max_dd']:.1f}% DD")

print()
print("=" * 80)
