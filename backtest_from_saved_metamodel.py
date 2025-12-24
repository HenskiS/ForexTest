"""
Load saved metamodel predictions and run quick backtest.
This allows testing different parameters without retraining.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load saved metamodel and predictions
print("Loading saved predictions...")
with open('metamodel_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

clf = data['clf']
scaler = data['scaler']
predictions = data['predictions']
HOLD_DAYS = data['hold_days']
SL_PCT = data['sl_pct']

print(f"Loaded {len(predictions)} predictions")
print()

# Load price data for backtesting
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
pair_dfs = {}
for pair in ALL_PAIRS:
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    pair_dfs[pair] = df

# Trading parameters
LEVERAGE = 2.0
ALLOC_PER_SLOT = 0.225

def run_backtest(confidence_threshold):
    """Run backtest using saved predictions"""
    all_trades = []
    daily_pnl = {}

    # Group predictions by pair
    pair_predictions = {}
    for pred in predictions:
        pair = pred['pair']
        if pair not in pair_predictions:
            pair_predictions[pair] = []
        pair_predictions[pair].append(pred)

    # Run backtest for each pair
    for pair, preds in pair_predictions.items():
        df = pair_dfs[pair]
        virtual_positions = []

        for pred_data in preds:
            pred_date = pred_data['date']
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            spread = pred_data['spread']

            if pred_date not in df.index:
                continue

            pred_idx = df.index.get_loc(pred_date)
            current_idx = pred_idx + 1

            if current_idx >= len(df):
                continue

            current_date = df.index[current_idx]
            if current_date.dayofweek >= 5:
                continue

            current_row = df.iloc[current_idx]

            # Check existing positions
            remaining = []
            for vp in virtual_positions:
                direction = vp['direction']
                days_held = current_idx - vp['entry_idx']

                # Check SL
                if direction == 1:
                    sl_price = vp['entry_price'] * (1 - SL_PCT)
                    sl_hit = current_row['low'] <= sl_price
                    if sl_hit:
                        exit_price = sl_price * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                    elif days_held >= HOLD_DAYS:
                        exit_price = current_row['open'] * (1 - spread)
                        pnl = (exit_price / vp['entry_price']) - 1
                    else:
                        remaining.append(vp)
                        continue
                else:
                    sl_price = vp['entry_price'] * (1 + SL_PCT)
                    sl_hit = current_row['high'] >= sl_price
                    if sl_hit:
                        exit_price = sl_price * (1 + spread)
                        pnl = (vp['entry_price'] / exit_price) - 1
                    elif days_held >= HOLD_DAYS:
                        exit_price = current_row['open'] * (1 + spread)
                        pnl = (vp['entry_price'] / exit_price) - 1
                    else:
                        remaining.append(vp)
                        continue

                # Record trade
                all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date']})
                if vp['entry_date'] not in daily_pnl:
                    daily_pnl[vp['entry_date']] = 0.0
                daily_pnl[vp['entry_date']] += ALLOC_PER_SLOT * pnl

            virtual_positions = remaining

            # Check for new signal
            if confidence >= confidence_threshold:
                # If opposite position exists, close it first
                if virtual_positions:
                    direction = 1 if prediction > 0 else -1
                    if virtual_positions[0]['direction'] != direction:
                        # Close all positions
                        for vp in virtual_positions:
                            if vp['direction'] == 1:
                                exit_price = current_row['open'] * (1 - spread)
                                pnl = (exit_price / vp['entry_price']) - 1
                            else:
                                exit_price = current_row['open'] * (1 + spread)
                                pnl = (vp['entry_price'] / exit_price) - 1

                            all_trades.append({'pnl': pnl, 'pair': pair, 'date': vp['entry_date']})
                            if vp['entry_date'] not in daily_pnl:
                                daily_pnl[vp['entry_date']] = 0.0
                            daily_pnl[vp['entry_date']] += ALLOC_PER_SLOT * pnl

                        virtual_positions = []

                # Enter new position
                direction = 1 if prediction > 0 else -1
                if direction == 1:
                    actual_entry = current_row['open'] * (1 + spread)
                else:
                    actual_entry = current_row['open'] * (1 - spread)

                virtual_positions.append({
                    'entry_price': actual_entry,
                    'entry_idx': current_idx,
                    'entry_date': current_date,
                    'direction': direction
                })

    return all_trades, daily_pnl

# Test different thresholds quickly
print("=" * 90)
print("BACKTESTING WITH SAVED PREDICTIONS")
print("=" * 90)
print()

thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
results = []

print(f'{"Threshold":<12} {"Annual":>8} {"Sharpe":>8} {"MaxDD":>8} {"WinRate":>8} {"Trades":>8}')
print('-' * 70)

for threshold in thresholds:
    trades, pnl = run_backtest(threshold)

    if not trades:
        print(f'{threshold:<12.2f} {"No trades":<55}')
        continue

    # Calculate metrics
    returns = pd.Series(pnl).sort_index()
    leveraged = returns * LEVERAGE

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

    results.append({
        'threshold': threshold,
        'annual': annual,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': len(trades),
        'years': years,
        'final_equity': equity[-1]
    })

    print(f'{threshold:<12.2f} {annual:>7.1f}% {sharpe:>8.2f} {max_dd:>7.1f}% {win_rate:>7.1f}% {len(trades):>8}')

print()
print("=" * 90)
print("BEST RESULT")
print("=" * 90)
if results:
    best = max(results, key=lambda x: x['annual'])
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Annual: {best['annual']:.1f}%")
    print(f"Sharpe: {best['sharpe']:.2f}")
    print(f"MaxDD: {best['max_dd']:.1f}%")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades: {best['trades']}")
    print(f"Test period: {best['years']:.1f} years")
    print(f"Final Equity: ${best['final_equity']:.2f}")
    print("=" * 90)
