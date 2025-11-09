"""
Test optimal percentage-based exit strategy on 2025 data only.
Train on 2000-2024, test on 2025 to verify no data leakage.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

print("2025 VALIDATION: PERCENTAGE EXITS STRATEGY")
print("="*80)

# Load data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)
df_clean = df.dropna()

print(f"Full data: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Split at 2025
split_date = '2025-01-01'
train_data = df_clean[df_clean.index < split_date]
test_data = df_clean[df_clean.index >= split_date]

print(f"\nTrain period: {train_data.index.min()} to {train_data.index.max()}")
print(f"  Size: {len(train_data)} days (~{len(train_data)/252:.1f} years)")
print(f"\nTest period: {test_data.index.min()} to {test_data.index.max()}")
print(f"  Size: {len(test_data)} days (~{len(test_data)/252:.1f} years)")

# Configuration
TARGET_COLUMN = 'target_5day_return'
FEATURE_COLS = [
    'momentum', 'avg_price', 'range', 'ohlc',
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'plus_di', 'minus_di',
    'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'atr'
]

# Prepare data
print("\nPreparing data...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data[FEATURE_COLS])
y_train = train_data[TARGET_COLUMN].values

X_test = scaler.transform(test_data[FEATURE_COLS])
y_test = test_data[TARGET_COLUMN].values

print(f"Train X: {X_train.shape}, y: {y_train.shape}")
print(f"Test X: {X_test.shape}, y: {y_test.shape}")

# Train model
best_params = {
    'n_estimators': 250,
    'learning_rate': 0.01,
    'max_depth': 12,
    'gamma': 0.002,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nTraining XGBoost...")
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train, verbose=False)

# Make predictions
y_pred = model.predict(X_test)

# Generate signals
q1 = np.percentile(y_pred, 25)
q3 = np.percentile(y_pred, 75)

signals = np.zeros(len(y_pred))
signals[y_pred >= q3] = 1   # Long
signals[y_pred <= q1] = -1  # Short

print(f"\nSignal distribution:")
print(f"  Buy:  {np.sum(signals == 1)} ({np.sum(signals == 1)/len(signals)*100:.1f}%)")
print(f"  Sell: {np.sum(signals == -1)} ({np.sum(signals == -1)/len(signals)*100:.1f}%)")
print(f"  Hold: {np.sum(signals == 0)} ({np.sum(signals == 0)/len(signals)*100:.1f}%)")

# Backtest with optimal stops: 0.50% loss, 1.25% profit
print("\nBacktesting with 0.50% stop-loss, 1.25% take-profit...")

STOP_LOSS_PCT = 0.0050
TAKE_PROFIT_PCT = 0.0125
INITIAL_CAPITAL = 1000.0
TRANSACTION_COST = 0.0002

capital = INITIAL_CAPITAL
position = 0
entry_price = 0.0
entry_capital = 0.0

equity_curve = [capital]
returns = []
trades = []

opens = test_data['open'].values
highs = test_data['high'].values
lows = test_data['low'].values
closes = test_data['close'].values

for i in range(len(signals)):
    signal = signals[i]
    open_price = opens[i]
    high_price = highs[i]
    low_price = lows[i]

    # Check stops if we have a position
    if position != 0:
        if position == 1:
            pct_high = (high_price - entry_price) / entry_price
            pct_low = (low_price - entry_price) / entry_price
        else:
            pct_high = (entry_price - low_price) / entry_price
            pct_low = (entry_price - high_price) / entry_price

        exit_triggered = False
        exit_price = None
        exit_reason = None

        if pct_low <= -STOP_LOSS_PCT:
            exit_triggered = True
            exit_reason = 'stop_loss'
            if position == 1:
                exit_price = entry_price * (1 - STOP_LOSS_PCT)
            else:
                exit_price = entry_price * (1 + STOP_LOSS_PCT)

        elif pct_high >= TAKE_PROFIT_PCT:
            exit_triggered = True
            exit_reason = 'take_profit'
            if position == 1:
                exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
            else:
                exit_price = entry_price * (1 - TAKE_PROFIT_PCT)

        if exit_triggered:
            if position == 1:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl = entry_capital * pnl_pct
            cost = capital * TRANSACTION_COST
            pnl -= cost
            capital += pnl

            trade_return = pnl / (capital - pnl)
            returns.append(trade_return)
            trades.append({
                'type': 'long' if position == 1 else 'short',
                'pnl': pnl,
                'exit_reason': exit_reason
            })

            position = 0

    # Enter new position
    if position == 0 and signal != 0:
        cost = capital * TRANSACTION_COST
        capital -= cost
        entry_price = open_price
        position = signal
        entry_capital = capital

    equity_curve.append(capital)

# Close final position
if position != 0:
    exit_price = closes[-1]

    if position == 1:
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    pnl = entry_capital * pnl_pct
    cost = capital * TRANSACTION_COST
    pnl -= cost
    capital += pnl

    trade_return = pnl / (capital - pnl)
    returns.append(trade_return)
    trades.append({
        'type': 'long' if position == 1 else 'short',
        'pnl': pnl,
        'exit_reason': 'end_of_period'
    })

# Calculate metrics
final_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
winning_trades = [t for t in trades if t['pnl'] > 0]
losing_trades = [t for t in trades if t['pnl'] <= 0]

n_years = len(test_data) / 252
annualized_return = (1 + final_return) ** (1 / n_years) - 1 if n_years > 0 else 0

if len(returns) > 0:
    returns = np.array(returns)
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
else:
    volatility = 0
    sharpe_ratio = 0

exit_reasons = {}
for t in trades:
    reason = t.get('exit_reason', 'unknown')
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

print(f"\n{'='*80}")
print("2025 VALIDATION RESULTS")
print(f"{'='*80}")
print(f"\nCapital:")
print(f"  Initial: ${INITIAL_CAPITAL:.2f}")
print(f"  Final:   ${capital:.2f}")
print(f"  Return:  {final_return*100:.2f}%")
print(f"  Annualized: {annualized_return*100:.2f}%")

print(f"\nRisk Metrics:")
print(f"  Volatility: {volatility*100:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")

print(f"\nTrading:")
print(f"  Total trades: {len(trades)}")
if len(trades) > 0:
    print(f"  Winning: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"  Losing: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    print(f"  Trades per year: {len(trades) / n_years:.1f}" if n_years > 0 else "  Trades per year: N/A")

    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items()):
        pct = count / len(trades) * 100
        print(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")

print(f"\n{'='*80}")
print("COMPARISON:")
print(f"{'='*80}")
print(f"  Walk-forward (2000-2025, retrain every 126 days):")
print(f"    Total: +223.5%, Annual: 6.05%, Sharpe: 0.464, Trades: 1863")
print(f"\n  2025-only (train 2000-2024, test 2025):")
print(f"    Total: {final_return*100:+.2f}%, Annual: {annualized_return*100:.2f}%, Sharpe: {sharpe_ratio:.3f}, Trades: {len(trades)}")

if final_return > 0 and sharpe_ratio > 0.1:
    print(f"\n  SUCCESS! Strategy works on 2025 data - NO DATA LEAKAGE!")
else:
    print(f"\n  Strategy performance degraded on 2025 data (needs retraining)")

print(f"{'='*80}")

# Save results
results = {
    'train_period': f"{train_data.index.min()} to {train_data.index.max()}",
    'test_period': f"{test_data.index.min()} to {test_data.index.max()}",
    'final_capital': capital,
    'total_return': final_return,
    'annualized_return': annualized_return,
    'sharpe_ratio': sharpe_ratio,
    'n_trades': len(trades),
    'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0,
    'exit_reasons': exit_reasons
}

with open('test_2025_percent_exits_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: test_2025_percent_exits_results.pkl")
