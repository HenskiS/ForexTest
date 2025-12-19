"""
CLASSIFIER HORIZON TEST
Test direction prediction at various horizons with long-only.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ALL_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY']
SPREAD_PCT = 0.00015

print("=" * 80)
print("CLASSIFIER HORIZON TEST - Long Only")
print("=" * 80)
print()


def calculate_features(df):
    df = df.copy()

    for span in [10, 20, 50, 100, 200]:
        ema = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_{span}'] = (df['close'] / ema - 1) * 100

    df['ema_10_20'] = (df['close'].ewm(span=10).mean() / df['close'].ewm(span=20).mean() - 1) * 100
    df['ema_20_50'] = (df['close'].ewm(span=20).mean() / df['close'].ewm(span=50).mean() - 1) * 100
    df['ema_50_200'] = (df['close'].ewm(span=50).mean() / df['close'].ewm(span=200).mean() - 1) * 100

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

    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['cci'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)

    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_pos'] = (df['close'] - bb_middle) / (bb_std * 2 + 1e-10)
    df['bb_width'] = (4 * bb_std / bb_middle) * 100

    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr_pct'] = (tr.rolling(14).mean() / df['close']) * 100

    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / (tr_14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).sum() / (tr_14 + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    for period in [1, 3, 5, 10, 15, 20, 30]:
        df[f'mom_{period}d'] = df['close'].pct_change(period) * 100

    df['daily_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['dist_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100

    return df


print("Loading data...")
pair_data = {}
for pair in ALL_PAIRS:
    df = pd.read_csv(f'data/{pair}_1day_oanda.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = calculate_features(df)
    df['spread_pct'] = SPREAD_PCT
    pair_data[pair] = df

FEATURES = [
    'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
    'ema_10_20', 'ema_20_50', 'ema_50_200',
    'macd', 'macd_signal', 'macd_hist',
    'rsi', 'stoch_k', 'stoch_d', 'cci',
    'bb_pos', 'bb_width', 'atr_pct', 'williams_r', 'adx',
    'mom_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_15d', 'mom_20d', 'mom_30d',
    'daily_range', 'dist_high_20', 'dist_high_50',
]

print(f"Done - {len(FEATURES)} features")
print()


def test_classifier(horizon, confidence_threshold=0.5, long_only=True):
    """Test classifier at specific horizon."""
    train_X, train_y, test_data = [], [], []

    for pair in ALL_PAIRS:
        df = pair_data[pair].copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df = df.dropna()

        df_sorted = df.sort_index()
        train_df = df_sorted.iloc[-1500:-750]  # 750 days train
        test_df = df_sorted.iloc[-750:]         # 750 days test

        if len(train_df) > 0:
            train_X.append(train_df[FEATURES].values)
            train_y.append((train_df['future_return'] > 0).astype(int).values)

        for idx, row in test_df.iterrows():
            test_data.append({
                'date': idx,
                'pair': pair,
                'features': row[FEATURES].values,
                'actual_return': row['future_return'],
                'spread_pct': row['spread_pct']
            })

    train_X = np.vstack(train_X)
    train_y = np.concatenate(train_y)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(np.array([d['features'] for d in test_data]))

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                          early_stopping=True, validation_fraction=0.1)
    model.fit(train_X_scaled, train_y)

    probs = model.predict_proba(test_X_scaled)[:, 1]  # Probability of up

    all_trades = []
    for i, d in enumerate(test_data):
        # Long only: only trade when model predicts up with confidence
        if long_only:
            if probs[i] >= confidence_threshold:
                pnl = (d['actual_return'] - d['spread_pct'] * 2) * 100
                all_trades.append({
                    'date': d['date'],
                    'pnl': pnl,
                    'prob': probs[i],
                    'actual': d['actual_return'] * 100
                })
        else:
            # Long/short based on probability
            if probs[i] >= confidence_threshold:
                pnl = (d['actual_return'] - d['spread_pct'] * 2) * 100
            elif probs[i] <= (1 - confidence_threshold):
                pnl = (-d['actual_return'] - d['spread_pct'] * 2) * 100
            else:
                continue
            all_trades.append({'date': d['date'], 'pnl': pnl, 'prob': probs[i]})

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_pnl = trades_df['pnl'].mean()
    n_trades = len(trades_df)
    years = (trades_df['date'].max() - trades_df['date'].min()).days / 365.25

    # Equity curve (10% position, 2x leverage)
    equity = [10000]
    for pnl in trades_df['pnl'].values:
        trade_return = (pnl / 100) * 0.10 * 2
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
        'max_dd': max_dd * 100
    }


# ============================================================================
# Test all horizons with long-only, 50% confidence
# ============================================================================
print("=" * 80)
print("HORIZON COMPARISON - Long Only, 50% confidence threshold")
print("=" * 80)
print()

print(f"{'Horizon':>8} {'Trades':>8} {'Tr/Yr':>8} {'WinRate':>9} {'AvgPnL':>9} {'Annual':>10} {'MaxDD':>8}")
print("-" * 70)

for horizon in [1, 2, 3, 5, 7, 10, 15, 20]:
    result = test_classifier(horizon, confidence_threshold=0.5, long_only=True)
    if result:
        print(f"{horizon:>7}d {result['trades']:>8} {result['trades_per_year']:>7.0f} {result['win_rate']:>8.1f}% "
              f"{result['avg_pnl']:>8.3f}% {result['annual']:>9.1f}% {result['max_dd']:>7.1f}%")

print()


# ============================================================================
# Test confidence thresholds for best horizon
# ============================================================================
print("=" * 80)
print("CONFIDENCE THRESHOLD TEST - 15d horizon, Long Only")
print("=" * 80)
print()

print(f"{'Conf':>8} {'Trades':>8} {'Tr/Yr':>8} {'WinRate':>9} {'AvgPnL':>9} {'Annual':>10} {'MaxDD':>8}")
print("-" * 70)

for conf in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    result = test_classifier(15, confidence_threshold=conf, long_only=True)
    if result:
        print(f"{conf*100:>7.0f}% {result['trades']:>8} {result['trades_per_year']:>7.0f} {result['win_rate']:>8.1f}% "
              f"{result['avg_pnl']:>8.3f}% {result['annual']:>9.1f}% {result['max_dd']:>7.1f}%")

print()


# ============================================================================
# Best config with position sizing
# ============================================================================
print("=" * 80)
print("POSITION SIZING - 15d horizon, 60% confidence, Long Only")
print("=" * 80)
print()

def test_with_sizing(horizon, conf, pos_size, leverage):
    """Test with specific position sizing."""
    train_X, train_y, test_data = [], [], []

    for pair in ALL_PAIRS:
        df = pair_data[pair].copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df = df.dropna()

        df_sorted = df.sort_index()
        train_df = df_sorted.iloc[-1500:-750]  # 750 days train
        test_df = df_sorted.iloc[-750:]         # 750 days test

        if len(train_df) > 0:
            train_X.append(train_df[FEATURES].values)
            train_y.append((train_df['future_return'] > 0).astype(int).values)

        for idx, row in test_df.iterrows():
            test_data.append({
                'date': idx,
                'features': row[FEATURES].values,
                'actual_return': row['future_return'],
                'spread_pct': row['spread_pct']
            })

    train_X = np.vstack(train_X)
    train_y = np.concatenate(train_y)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(np.array([d['features'] for d in test_data]))

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                          early_stopping=True, validation_fraction=0.1)
    model.fit(train_X_scaled, train_y)
    probs = model.predict_proba(test_X_scaled)[:, 1]

    equity = [10000]
    trades = 0
    wins = 0

    for i, d in enumerate(test_data):
        if probs[i] >= conf:
            pnl = d['actual_return'] - d['spread_pct'] * 2
            trade_return = pnl * pos_size * leverage
            equity.append(equity[-1] * (1 + trade_return))
            trades += 1
            if pnl > 0:
                wins += 1

    if trades == 0:
        return None

    total_return = equity[-1] / 10000 - 1
    years = 750 / 252  # Approximate trading days
    annual = ((1 + total_return) ** (1/years) - 1) * 100 if total_return > -1 else -100

    peak = 10000
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': trades,
        'win_rate': wins / trades * 100,
        'annual': annual,
        'max_dd': max_dd * 100,
        'final': equity[-1]
    }

print(f"{'Pos':>6} {'Lev':>5} {'Trades':>7} {'WinRate':>9} {'Annual':>10} {'MaxDD':>8} {'Final':>12}")
print("-" * 65)

for pos in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for lev in [1.0, 1.5, 2.0]:
        result = test_with_sizing(15, 0.60, pos, lev)
        if result:
            marker = "<--" if result['max_dd'] < 15 else ""
            print(f"{pos*100:>5.0f}% {lev:>4.1f}x {result['trades']:>7} {result['win_rate']:>8.1f}% "
                  f"{result['annual']:>9.1f}% {result['max_dd']:>7.1f}% ${result['final']:>10,.0f} {marker}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Classifier with long-only direction prediction.
Key: Find the right horizon and confidence threshold.
""")
