"""
Load saved metamodel and run backtest with production-matching logic.
Rebuilds features with proper recent_outcomes history.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load saved metamodel
print("Loading saved metamodel...")
with open('metamodel_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

clf = data['clf']
scaler = data['scaler']
TRAIN_SIZE = data['train_size']
HOLD_DAYS = data['hold_days']
SL_PCT = data['sl_pct']
LOOKBACK_TRADES = data['lookback_trades']

print("Metamodel loaded!")
print(f"Train size: {TRAIN_SIZE}")
print()

# Load all data (base predictions + price data)
ALL_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']
pair_data = {}

for pair in ALL_PAIRS:
    with open(f'optimized_ann_predictions/predictions_{pair}.pkl', 'rb') as f:
        pred_data = pickle.load(f)

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

    pair_data[pair] = {
        'predictions': pred_data['predictions'],
        'dates': pred_data['dates'],
        'df': df
    }

# Trading parameters
LEVERAGE = 2.0
ALLOC_PER_SLOT = 0.225

def calculate_hypothetical_outcome(pair, pred_date, prediction, spread):
    """Calculate what outcome WOULD be if we took this trade"""
    df = pair_data[pair]['df']

    if pred_date not in df.index:
        return None

    pred_idx = df.index.get_loc(pred_date)
    entry_idx = pred_idx + 1

    if entry_idx >= len(df) - HOLD_DAYS:
        return None

    entry_row = df.iloc[entry_idx]
    direction = 1 if prediction > 0 else -1

    if direction == 1:
        actual_entry = entry_row['open'] * (1 + spread)
    else:
        actual_entry = entry_row['open'] * (1 - spread)

    # Check for SL
    for days in range(HOLD_DAYS + 1):
        check_idx = entry_idx + days
        if check_idx >= len(df):
            return None
        check_row = df.iloc[check_idx]

        if direction == 1:
            sl_price = actual_entry * (1 - SL_PCT)
            if check_row['low'] <= sl_price:
                exit_price = sl_price * (1 - spread)
                pnl = (exit_price / actual_entry) - 1
                return pnl > 0
        else:
            sl_price = actual_entry * (1 + SL_PCT)
            if check_row['high'] >= sl_price:
                exit_price = sl_price * (1 + spread)
                pnl = (actual_entry / exit_price) - 1
                return pnl > 0

    # Time exit
    exit_row = df.iloc[entry_idx + HOLD_DAYS]
    if direction == 1:
        exit_price = exit_row['open'] * (1 - spread)
        pnl = (exit_price / actual_entry) - 1
    else:
        exit_price = exit_row['open'] * (1 + spread)
        pnl = (actual_entry / exit_price) - 1

    return pnl > 0

def extract_features(pair, pred_idx, prediction, recent_outcomes, df):
    """Extract features for metamodel (production-matching)"""
    if pred_idx < 100:
        return None

    features = []

    # Prediction features
    features.append(prediction)
    features.append(abs(prediction))
    features.append(1 if prediction > 0 else -1)

    # Recent performance (this is what was wrong!)
    if len(recent_outcomes) >= 5:
        features.append(sum(recent_outcomes[-LOOKBACK_TRADES:]) / min(len(recent_outcomes), LOOKBACK_TRADES))
        features.append(sum(recent_outcomes[-5:]) / 5)
    else:
        features.append(0.5)
        features.append(0.5)

    # Market features
    current_row = df.iloc[pred_idx]
    features.append(current_row.get('atr', 0.01) if 'atr' in df.columns else 0.01)
    features.append(current_row['spread_pct'])

    # Price action
    if pred_idx >= 20:
        recent_prices = df.iloc[pred_idx-20:pred_idx]['close']
        features.append(recent_prices.pct_change().std())
        features.append((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1)
    else:
        features.append(0.01)
        features.append(0.0)

    # Day of week
    pred_date = df.index[pred_idx]
    features.append(pred_date.dayofweek)
    features.append(1 if pred_date.dayofweek == 0 else 0)
    features.append(1 if pred_date.dayofweek == 4 else 0)

    return features

def run_backtest(confidence_threshold):
    """Run backtest with production-matching logic"""
    all_trades = []
    daily_pnl = {}

    # Run backtest for each pair
    for pair in ALL_PAIRS:
        predictions = pair_data[pair]['predictions']
        pred_dates = pair_data[pair]['dates']
        df = pair_data[pair]['df']

        # Build recent_outcomes from the beginning (production logic!)
        recent_outcomes = []
        virtual_positions = []

        for idx, (pred_date, prediction) in enumerate(zip(pred_dates, predictions)):
            # Skip training period for trading, but build recent_outcomes
            if idx < TRAIN_SIZE:
                if pred_date in df.index:
                    spread = df.iloc[df.index.get_loc(pred_date)]['spread_pct']
                    hypo_outcome = calculate_hypothetical_outcome(pair, pred_date, prediction, spread)
                    if hypo_outcome is not None:
                        recent_outcomes.append(hypo_outcome)
                continue

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
            spread = current_row['spread_pct']

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

            # Extract features and get metamodel confidence (dynamically!)
            features = extract_features(pair, pred_idx, prediction, recent_outcomes, df)
            if features is None:
                # Still calculate outcome for recent_outcomes tracking
                hypo_outcome = calculate_hypothetical_outcome(pair, pred_date, prediction, spread)
                if hypo_outcome is not None:
                    recent_outcomes.append(hypo_outcome)
                continue

            # Get metamodel confidence
            X_sample = scaler.transform([features])
            confidence = clf.predict_proba(X_sample)[0, 1]

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

            # Calculate outcome for ALL signals (AFTER making decision)
            # This matches production logic - recent_outcomes tracks all signals
            hypo_outcome = calculate_hypothetical_outcome(pair, pred_date, prediction, spread)
            if hypo_outcome is not None:
                recent_outcomes.append(hypo_outcome)

    return all_trades, daily_pnl

# Test different thresholds quickly
print("=" * 90)
print("BACKTEST WITH PRODUCTION-MATCHING LOGIC")
print("=" * 90)
print("Using saved metamodel, but rebuilding features with proper recent_outcomes")
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
