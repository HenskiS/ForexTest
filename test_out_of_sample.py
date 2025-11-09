"""
True out-of-sample test: Train on early period, test on late period.
NO retraining - test if model generalizes to completely unseen future.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

print("TRUE OUT-OF-SAMPLE TEST")
print("="*80)

# Load data
df = pd.read_csv('data/EURUSD_1day_with_features_FIXED_multitarget.csv',
                 index_col='date', parse_dates=True)
df_clean = df.dropna()

print(f"Full data: {df_clean.shape}")
print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")

# Split at 2015
split_date = '2015-01-01'
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

# Use median hyperparameters from our previous training
# (Could also do validation split, but let's use what worked)
best_params = {
    'n_estimators': 250,
    'learning_rate': 0.01,
    'max_depth': 12,
    'gamma': 0.002,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nTraining XGBoost with parameters:")
print(f"  {best_params}")

model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train, verbose=False)

print("\nModel trained!")

# Make predictions
y_pred = model.predict(X_test)
test_mae = np.mean(np.abs(y_test - y_pred))

print(f"\nPrediction Performance:")
print(f"  Test MAE: {test_mae:.6f}")
print(f"  Correlation: {np.corrcoef(y_pred, y_test)[0,1]:.4f}")

# Generate signals using quartile strategy
print("\nGenerating trading signals (quartile strategy)...")
q1 = np.percentile(y_pred, 25)
q3 = np.percentile(y_pred, 75)

signals = np.zeros(len(y_pred))
signals[y_pred >= q3] = 1   # Long
signals[y_pred <= q1] = -1  # Short

n_buy = np.sum(signals == 1)
n_sell = np.sum(signals == -1)
n_hold = np.sum(signals == 0)

print(f"Signal distribution:")
print(f"  Buy:  {n_buy} ({n_buy/len(signals)*100:.1f}%)")
print(f"  Sell: {n_sell} ({n_sell/len(signals)*100:.1f}%)")
print(f"  Hold: {n_hold} ({n_hold/len(signals)*100:.1f}%)")

# Backtest
print("\nBacktesting with forex mechanics...")

PIP_VALUE = 0.1
PIP_SIZE = 0.0001
INITIAL_CAPITAL = 1000.0
TRANSACTION_COST = 0.0002

capital = INITIAL_CAPITAL
position = 0
entry_price = 0.0
entry_lot_size = 0.0

equity_curve = [capital]
returns = []
trades = []

opens = test_data['open'].values
closes = test_data['close'].values

for i in range(len(signals)):
    signal = signals[i]
    open_price = opens[i]

    # Exit current position if signal changes
    if position != 0 and signal != position:
        exit_price = open_price

        if position == 1:
            pips = (exit_price - entry_price) / PIP_SIZE
        else:
            pips = (entry_price - exit_price) / PIP_SIZE

        pnl = pips * PIP_VALUE * entry_lot_size / 1000
        cost = capital * TRANSACTION_COST
        pnl -= cost
        capital += pnl

        trade_return = pnl / (capital - pnl)
        returns.append(trade_return)
        trades.append({
            'type': 'long' if position == 1 else 'short',
            'pips': pips,
            'pnl': pnl
        })

        position = 0

    # Enter new position
    if position == 0 and signal != 0:
        cost = capital * TRANSACTION_COST
        capital -= cost
        entry_price = open_price
        position = signal
        entry_lot_size = capital

    equity_curve.append(capital)

# Close final position
if position != 0:
    exit_price = closes[-1]

    if position == 1:
        pips = (exit_price - entry_price) / PIP_SIZE
    else:
        pips = (entry_price - exit_price) / PIP_SIZE

    pnl = pips * PIP_VALUE * entry_lot_size / 1000
    cost = capital * TRANSACTION_COST
    pnl -= cost
    capital += pnl

    trade_return = pnl / (capital - pnl)
    returns.append(trade_return)
    trades.append({
        'type': 'long' if position == 1 else 'short',
        'pips': pips,
        'pnl': pnl
    })

# Calculate metrics
final_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
winning_trades = [t for t in trades if t['pnl'] > 0]
losing_trades = [t for t in trades if t['pnl'] <= 0]

n_years = len(test_data) / 252
annualized_return = (1 + final_return) ** (1 / n_years) - 1

returns = np.array(returns)
volatility = np.std(returns) * np.sqrt(252)
sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

print(f"\n{'='*80}")
print("OUT-OF-SAMPLE BACKTEST RESULTS")
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
print(f"  Winning: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
print(f"  Losing: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
print(f"  Trades per year: {len(trades) / n_years:.1f}")

print(f"\n{'='*80}")
print("COMPARISON TO WALK-FORWARD RESULTS:")
print(f"{'='*80}")
print(f"  Walk-forward (retrain every 126 days): +106.77% total, 56.17% annualized")
print(f"  Out-of-sample (train once, no retrain): {final_return*100:+.2f}% total, {annualized_return*100:.2f}% annualized")

if final_return > 0:
    print(f"\n  SUCCESS! Model generalizes to unseen future data!")
else:
    print(f"\n  Model struggles without retraining on recent data")

print(f"{'='*80}")

# Save results
results = {
    'train_period': f"{train_data.index.min()} to {train_data.index.max()}",
    'test_period': f"{test_data.index.min()} to {test_data.index.max()}",
    'test_mae': test_mae,
    'final_capital': capital,
    'total_return': final_return,
    'annualized_return': annualized_return,
    'sharpe_ratio': sharpe_ratio,
    'n_trades': len(trades),
    'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0,
    'predictions': y_pred.tolist(),
    'actuals': y_test.tolist(),
    'signals': signals.tolist()
}

with open('out_of_sample_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: out_of_sample_results.pkl")
