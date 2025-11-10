"""
Prepare and test any currency pair with the optimal XGBoost strategy.

Usage:
    python prepare_and_test_currency_pair.py GBPUSD
    python prepare_and_test_currency_pair.py AUDUSD
    python prepare_and_test_currency_pair.py USDJPY
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import FMPDataFetcher

def add_technical_features(df):
    """Add all technical indicators used in the study."""
    print("Starting feature engineering...")
    print(f"Initial shape: {df.shape}")

    df_features = df.copy()

    # Statistical Indicators
    print("\n[1/4] Calculating Statistical Indicators...")
    df_features['momentum'] = df_features['open'] - df_features['close']
    df_features['avg_price'] = (df_features['low'] + df_features['high']) / 2
    df_features['range'] = df_features['high'] - df_features['low']
    df_features['ohlc'] = (df_features['open'] + df_features['close'] +
                            df_features['low'] + df_features['high']) / 4
    print("   - Momentum, Average Price, Range, OHLC: Done")

    # Momentum Indicators
    print("\n[2/4] Calculating Momentum Indicators...")

    # EMAs
    for period in [10, 20, 50, 100, 200]:
        df_features[f'ema_{period}'] = df_features['close'].ewm(span=period, adjust=False).mean()
    print("   - EMA (10, 20, 50, 100, 200): Done")

    # MACD
    ema_fast = df_features['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_features['close'].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema_fast - ema_slow
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
    print("   - MACD (26, 12, 9): Done")

    # ADX
    high_diff = df_features['high'].diff()
    low_diff = -df_features['low'].diff()

    plus_dm = high_diff.copy()
    plus_dm[high_diff < low_diff] = 0
    plus_dm[plus_dm < 0] = 0

    minus_dm = low_diff.copy()
    minus_dm[low_diff < high_diff] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = df_features['high'] - df_features['low']
    tr2 = abs(df_features['high'] - df_features['close'].shift(1))
    tr3 = abs(df_features['low'] - df_features['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()

    plus_dm_smooth = plus_dm.rolling(window=14).mean()
    minus_dm_smooth = minus_dm.rolling(window=14).mean()

    plus_di = 100 * (plus_dm_smooth / atr_14)
    minus_di = 100 * (minus_dm_smooth / atr_14)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    df_features['adx'] = dx.rolling(window=14).mean()
    df_features['plus_di'] = plus_di
    df_features['minus_di'] = minus_di
    print("   - ADX (14): Done")

    # Oscillator Indicators
    print("\n[3/4] Calculating Oscillator Indicators...")

    # RSI
    delta = df_features['close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df_features['rsi'] = 100 - (100 / (1 + rs))
    print("   - RSI (14): Done")

    # Stochastic
    low_14 = df_features['low'].rolling(window=14).min()
    high_14 = df_features['high'].rolling(window=14).max()

    df_features['stoch_k'] = 100 * ((df_features['close'] - low_14) / (high_14 - low_14))
    df_features['stoch_d'] = df_features['stoch_k'].rolling(window=3).mean()
    print("   - Stochastic Oscillator (14, 3): Done")

    # CCI
    tp = (df_features['high'] + df_features['low'] + df_features['close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md = abs(tp - ma_tp).rolling(window=20).mean()
    df_features['cci'] = (tp - ma_tp) / (0.015 * md)
    print("   - CCI (20, 0.015): Done")

    # Williams %R
    high_14 = df_features['high'].rolling(window=14).max()
    low_14 = df_features['low'].rolling(window=14).min()
    df_features['williams_r'] = -100 * ((high_14 - df_features['close']) / (high_14 - low_14))
    print("   - Williams %R (14): Done")

    # Volatility Indicators
    print("\n[4/4] Calculating Volatility Indicators...")

    # Bollinger Bands
    sma_20 = df_features['close'].rolling(window=20).mean()
    std_20 = df_features['close'].rolling(window=20).std()

    df_features['bb_middle'] = sma_20
    df_features['bb_upper'] = sma_20 + (2 * std_20)
    df_features['bb_lower'] = sma_20 - (2 * std_20)
    df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
    df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / df_features['bb_width']
    print("   - Bollinger Bands (20, 2): Done")

    # ATR
    df_features['atr'] = atr_14
    print("   - ATR (14): Done")

    print("\n" + "="*60)
    print("Feature Engineering Complete!")
    print("="*60)
    print(f"Final shape: {df_features.shape}")
    print(f"Total features created: {df_features.shape[1] - df.shape[1]}")

    return df_features


def create_forward_return_targets(df):
    """Create forward-looking return targets."""
    print("\nCreating forward return targets...")

    df_with_targets = df.copy()

    # Create 5-day forward return (our optimal target)
    df_with_targets['target_5day_return'] = df_with_targets['close'].pct_change(5).shift(-5)

    print(f"  - target_5day_return: Done")

    # Remove rows with NaN targets
    before = len(df_with_targets)
    df_with_targets = df_with_targets.dropna(subset=['target_5day_return'])
    after = len(df_with_targets)
    print(f"\nRemoved {before - after} rows with NaN targets")
    print(f"Final dataset: {len(df_with_targets)} rows")

    return df_with_targets


def main(currency_pair):
    """Main pipeline to prepare and test currency pair."""

    print(f"\n{'='*80}")
    print(f"PREPARING {currency_pair} DATA")
    print(f"{'='*80}\n")

    # Step 1: Fetch historical data
    print("Step 1: Fetching historical data...")
    fetcher = FMPDataFetcher()

    # Fetch from 2000 to present for consistency with EURUSD
    from_date = '2000-01-01'
    to_date = datetime.now().strftime('%Y-%m-%d')

    df = fetcher.get_historical_data(
        currency_pair,
        from_date=from_date,
        to_date=to_date,
        timeframe='1day'
    )

    if df.empty:
        print(f"ERROR: No data fetched for {currency_pair}")
        return

    print(f"Fetched {len(df)} days from {df['date'].min()} to {df['date'].max()}")

    # Save raw data
    raw_filename = f'{currency_pair}_1day.csv'
    fetcher.save_data(df, raw_filename)

    # Step 2: Add technical features
    print(f"\nStep 2: Adding technical features...")
    df_features = add_technical_features(df)

    # Step 3: Create forward return targets
    print(f"\nStep 3: Creating forward return targets...")
    df_final = create_forward_return_targets(df_features)

    # Step 4: Save final dataset
    output_filename = f'{currency_pair}_1day_with_features_FIXED_multitarget.csv'
    output_path = os.path.join('data', output_filename)
    df_final.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"DATA PREPARATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nSaved to: {output_path}")
    print(f"Total rows: {len(df_final)}")
    print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"Features: {len(df_final.columns)}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS:")
    print(f"{'='*80}")
    print(f"1. Train model:")
    print(f"   python train_xgboost_multitarget.py --target target_5day_return --n_iter 20 --pair {currency_pair}")
    print(f"\n2. Backtest optimal strategy:")
    print(f"   python backtest_advanced_exits.py --pair {currency_pair}")
    print(f"\n3. Analyze yearly performance:")
    print(f"   python analyze_yearly_performance.py --pair {currency_pair}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_and_test_currency_pair.py <CURRENCY_PAIR>")
        print("Example: python prepare_and_test_currency_pair.py GBPUSD")
        sys.exit(1)

    currency_pair = sys.argv[1].upper()
    main(currency_pair)
