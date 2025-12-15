"""
Market Utilities

Functions for market hours checking and technical indicator calculations.
"""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime


def is_forex_market_open():
    """
    Check if forex market is currently open.
    Forex market is open Sunday 5:00 PM ET to Friday 5:00 PM ET.

    Returns:
        bool: True if market is open, False if closed
    """
    # Get current time in ET
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)

    # Get day of week (0 = Monday, 6 = Sunday)
    weekday = now_et.weekday()
    hour = now_et.hour
    minute = now_et.minute

    # Market is closed Friday 5:00 PM ET to Sunday 5:00 PM ET

    # Friday after 5:00 PM (17:00) - CLOSED
    if weekday == 4 and (hour > 17 or (hour == 17 and minute >= 0)):
        return False

    # Saturday all day - CLOSED
    if weekday == 5:
        return False

    # Sunday before 5:00 PM (17:00) - CLOSED
    if weekday == 6 and hour < 17:
        return False

    # All other times - OPEN
    return True


def calculate_technical_features(df):
    """
    Calculate all technical features for trading model.
    Uses the optimized 31-feature set from train_all_pairs_optimized_hyperparams.py

    Args:
        df: DataFrame with OHLC data (columns: open, high, low, close, volume)

    Returns:
        DataFrame with added technical indicator columns
    """
    df = df.copy()

    # Basic features
    df['momentum'] = df['close'].pct_change()
    df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['range'] = df['high'] - df['low']
    df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # EMAs
    for period in [10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(window=14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic (using 20-day window to match train script)
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    df['stoch_k'] = 100 * (df['close'] - low_20) / (high_20 - low_20)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR
    df['atr'] = atr

    # Volume indicator
    df['volume_sma'] = df['volume'].rolling(window=20).mean()

    # Price position features
    df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

    # Target for training (1-day forward return)
    df['target_1day_return'] = df['close'].pct_change(1).shift(-1)

    return df
