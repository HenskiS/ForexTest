"""
WALK-FORWARD CLASSIFIER TEST
Proper validation: retrain periodically, predict next period.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
SPREAD_PCT = 0.00015
TRAIN_WINDOW = 300  # Days to train on
RETRAIN_EVERY = 1   # Retrain every day
SAVE_PREDICTIONS = True

print("=" * 80)
print("WALK-FORWARD CLASSIFIER TEST")
print(f"Train window: {TRAIN_WINDOW} days, Retrain every: {RETRAIN_EVERY} days")
print("=" * 80)
print()


def calculate_features(df):
    df = df.copy()

    for span in [10, 20, 50, 100, 200]:
        ema = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_{span}'] = (df['close'] / ema - 1) * 100

    df['ema_10_20'] = (df['close'].ewm(span=10).mean() / df['close'].ewm(span=20).mean() - 1) * 100
    df['ema_20_50'] = (df['close'].ewm(span=20).mean() / df['close'].ewm(span=50).mean() - 1) * 100

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema_12 - ema_26) / df['close'] * 100
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_pos'] = (df['close'] - bb_middle) / (bb_std * 2 + 1e-10)

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr_pct'] = (tr.rolling(14).mean() / df['close']) * 100

    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

    for period in [1, 3, 5, 10, 15, 20]:
        df[f'mom_{period}d'] = df['close'].pct_change(period) * 100

    df['dist_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100

    return df


print("Loading data...")
pair_data = {}
for pair in ALL_PAIRS:
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = calculate_features(df)
    df['spread_pct'] = SPREAD_PCT
    pair_data[pair] = df.dropna()

FEATURES = [
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'ema_10_20', 'ema_20_50',
    'macd', 'macd_signal', 'macd_hist',
    'rsi', 'stoch_k', 'stoch_d',
    'bb_pos', 'atr_pct', 'williams_r',
    'mom_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_15d', 'mom_20d',
    'dist_high_20',
]

print(f"Done - {len(FEATURES)} features")
print()


def walk_forward_test(horizon=15, confidence=0.5, test_days=500, save_preds=False):
    """
    Proper walk-forward test with NO lookahead bias.
    For each test day:
      - Train on previous TRAIN_WINDOW days, EXCLUDING last HORIZON days
        (because training targets extend HORIZON days into the future)
      - Predict direction for next HORIZON days
      - Move forward, retrain every RETRAIN_EVERY days
    """
    all_trades = []
    all_predictions = []  # Store ALL predictions, not just trades
    model = None
    scaler = None
    last_train_day = -RETRAIN_EVERY  # Force initial training

    # Get combined data with dates
    # REALISTIC: Entry at NEXT day's open, exit at open after horizon days
    combined = []
    for pair in ALL_PAIRS:
        df = pair_data[pair].copy()
        # Entry at tomorrow's open, exit at open after horizon days
        df['entry_price'] = df['open'].shift(-1)  # Next day's open
        df['exit_price'] = df['open'].shift(-1 - horizon)  # Open after horizon days
        df['future_return'] = df['exit_price'] / df['entry_price'] - 1
        df = df.dropna()
        df['pair'] = pair
        combined.append(df)

    combined = pd.concat(combined).sort_index()

    # Get unique dates for walk-forward
    unique_dates = combined.index.unique().sort_values()
    test_start_idx = len(unique_dates) - test_days

    print(f"Testing on {test_days} days, retraining every {RETRAIN_EVERY} days...")

    for day_idx in range(test_start_idx, len(unique_dates)):
        current_date = unique_dates[day_idx]

        # Retrain if needed
        if day_idx - last_train_day >= RETRAIN_EVERY:
            # CRITICAL FIX: Stop training HORIZON+1 days before test date
            # Because training targets extend HORIZON days into the future
            # The last training row's exit = feature_date + 1 + HORIZON
            # We need last exit < prediction date, so:
            # last_feature + 1 + HORIZON < day_idx
            # last_feature < day_idx - HORIZON - 1
            # Since train_end_idx is exclusive: train_end_idx = day_idx - HORIZON - 1
            train_end_idx = day_idx - horizon - 1  # Strict: no lookahead at all
            train_start_idx = max(0, train_end_idx - TRAIN_WINDOW)
            train_dates = unique_dates[train_start_idx:train_end_idx]

            train_data = combined[combined.index.isin(train_dates)]

            if len(train_data) < 100:
                continue

            X_train = train_data[FEATURES].values
            y_train = (train_data['future_return'] > 0).astype(int).values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
            model.fit(X_train_scaled, y_train)
            last_train_day = day_idx

        if model is None:
            continue

        # Get today's data for prediction
        today_data = combined[combined.index == current_date]

        for _, row in today_data.iterrows():
            X_test = row[FEATURES].values.reshape(1, -1)
            X_test_scaled = scaler.transform(X_test)

            prob = model.predict_proba(X_test_scaled)[0, 1]
            actual_ret = row['future_return']
            spread = row['spread_pct']

            # Save ALL predictions
            all_predictions.append({
                'date': current_date,
                'pair': row['pair'],
                'prob': prob,
                'actual_return': actual_ret * 100,
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
            })

            # Long only with confidence threshold for trades
            if prob >= confidence:
                pnl = (actual_ret - spread * 2) * 100
                all_trades.append({
                    'date': current_date,
                    'pair': row['pair'],
                    'prob': prob,
                    'pnl': pnl,
                    'actual': actual_ret * 100
                })

    # Save predictions if requested
    if save_preds and all_predictions:
        preds_df = pd.DataFrame(all_predictions)
        filename = f'walkforward_predictions_{horizon}d.csv'
        preds_df.to_csv(filename, index=False)
        print(f"Saved {len(preds_df)} predictions to {filename}")

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)

    # Metrics
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_pnl = trades_df['pnl'].mean()

    years = (trades_df['date'].max() - trades_df['date'].min()).days / 365.25

    # Equity curve (10% position, 1.5x leverage)
    equity = [10000]
    for pnl in trades_df['pnl'].values:
        trade_return = (pnl / 100) * 0.10 * 1.5
        equity.append(equity[-1] * (1 + trade_return))

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
        'trades': n_trades,
        'trades_per_year': n_trades / years if years > 0 else 0,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'annual': annual,
        'max_dd': max_dd * 100,
        'final': equity[-1],
        'retrains': (test_days // RETRAIN_EVERY),
        'trades_df': trades_df
    }


def analyze_predictions(filename, confidence_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70]):
    """Analyze saved predictions at different confidence thresholds."""
    preds_df = pd.read_csv(filename)
    preds_df['date'] = pd.to_datetime(preds_df['date'])

    print(f"{'Conf':>8} {'Trades':>8} {'Tr/Yr':>8} {'WinRate':>9} {'AvgPnL':>9} {'Annual':>10} {'MaxDD':>8}")
    print("-" * 70)

    years = (preds_df['date'].max() - preds_df['date'].min()).days / 365.25

    for conf in confidence_thresholds:
        trades = preds_df[preds_df['prob'] >= conf].copy()
        if len(trades) == 0:
            continue

        # Calculate PnL (with spread)
        trades['pnl'] = (trades['actual_return'] - SPREAD_PCT * 200)  # actual_return already in %

        n_trades = len(trades)
        win_rate = (trades['pnl'] > 0).mean() * 100
        avg_pnl = trades['pnl'].mean()

        # Equity curve (10% position, 1.5x leverage)
        equity = [10000]
        for pnl in trades['pnl'].values:
            trade_return = (pnl / 100) * 0.10 * 1.5
            equity.append(equity[-1] * (1 + trade_return))

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

        print(f"{conf*100:>7.0f}% {n_trades:>8} {n_trades/years:>7.0f} {win_rate:>8.1f}% "
              f"{avg_pnl:>8.3f}% {annual:>9.1f}% {max_dd*100:>7.1f}%")


# ============================================================================
# Generate predictions for 15d and 20d horizons (run once, analyze many times)
# ============================================================================
print("=" * 80)
print(f"WALK-FORWARD TEST - 8 PAIRS, {TRAIN_WINDOW}d train, daily retrain")
print("=" * 80)
print()

for horizon in [15, 20]:
    print(f"\n{'='*40}")
    print(f"HORIZON: {horizon} days")
    print(f"{'='*40}")

    # Generate predictions (saves to file)
    result = walk_forward_test(horizon=horizon, confidence=0.5, test_days=300, save_preds=SAVE_PREDICTIONS)

    if result:
        print(f"\nAt 50% confidence:")
        print(f"  Trades: {result['trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Annual: {result['annual']:.1f}%, Max DD: {result['max_dd']:.1f}%")

        # Analyze at different confidence thresholds from saved file
        print(f"\nConfidence threshold analysis:")
        analyze_predictions(f'walkforward_predictions_{horizon}d.csv')

print()
print("=" * 80)
print("Done - Predictions saved to walkforward_predictions_XXd.csv")
print("=" * 80)
