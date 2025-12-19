"""
THOROUGH BIAS VERIFICATION for Walk-Forward Test
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

HORIZON = 15
TRAIN_WINDOW = 300

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


print('='*80)
print('THOROUGH BIAS VERIFICATION')
print('='*80)
print()

# Load just EURUSD for detailed check
df = pd.read_csv('data/EURUSD_1day_oanda.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df_orig = df.copy()  # Keep original before features
df = calculate_features(df)

print('1. RAW DATA CHECK')
print('-'*40)
print(f'Data range: {df.index.min().date()} to {df.index.max().date()}')
print(f'Total rows: {len(df)}')
print()

# Add entry/exit prices
df['entry_price'] = df['open'].shift(-1)  # Next day's open
df['exit_price'] = df['open'].shift(-1 - HORIZON)  # Open after horizon days
df['future_return'] = df['exit_price'] / df['entry_price'] - 1
df_clean = df.dropna()

print('2. ENTRY/EXIT TIMING CHECK')
print('-'*40)
# Pick a specific date and verify
test_idx = -100  # 100 days from end
test_date = df_clean.index[test_idx]
test_row = df_clean.iloc[test_idx]

# Find actual dates in original data
orig_idx = df_orig.index.get_loc(test_date)
entry_date = df_orig.index[orig_idx + 1]  # Should be next day
exit_date = df_orig.index[orig_idx + 1 + HORIZON]  # Should be horizon days after entry

print(f'Signal date (features from): {test_date.date()}')
print(f'Entry date (next day open): {entry_date.date()}')
print(f'Exit date (+{HORIZON}d from entry): {exit_date.date()}')
print()

# Verify prices match
entry_price_expected = df_orig.loc[entry_date, 'open']
exit_price_expected = df_orig.loc[exit_date, 'open']
print(f'Entry price in row: {test_row["entry_price"]:.5f}')
print(f'Entry price from date: {entry_price_expected:.5f}')
print(f'Match: {abs(test_row["entry_price"] - entry_price_expected) < 0.00001}')
print()
print(f'Exit price in row: {test_row["exit_price"]:.5f}')
print(f'Exit price from date: {exit_price_expected:.5f}')
print(f'Match: {abs(test_row["exit_price"] - exit_price_expected) < 0.00001}')
print()

# Verify return calculation
calc_return = exit_price_expected / entry_price_expected - 1
print(f'Return in row: {test_row["future_return"]*100:.4f}%')
print(f'Calculated return: {calc_return*100:.4f}%')
print(f'Match: {abs(test_row["future_return"] - calc_return) < 0.000001}')
print()

print('3. FEATURE LOOKAHEAD CHECK')
print('-'*40)
# Check that features only use past data
test_date2 = df_clean.index[-50]
test_close = df_orig.loc[test_date2, 'close']

# Manually calculate EMA_10 up to test_date2
hist_data = df_orig.loc[:test_date2, 'close']
manual_ema10 = hist_data.ewm(span=10, adjust=False).mean().iloc[-1]
feature_ema10 = (test_close / manual_ema10 - 1) * 100

print(f'Date: {test_date2.date()}')
print(f'EMA_10 feature in data: {df_clean.loc[test_date2, "ema_10"]:.4f}')
print(f'EMA_10 calculated from history only: {feature_ema10:.4f}')
print(f'Match: {abs(df_clean.loc[test_date2, "ema_10"] - feature_ema10) < 0.0001}')
print()

print('4. TRAINING DATA TIMING CHECK')
print('-'*40)
# Simulate the walk-forward training
unique_dates = df_clean.index.unique().sort_values()
test_start_idx = len(unique_dates) - 50  # 50 days from end

for i, day_idx in enumerate([test_start_idx, test_start_idx + 10, test_start_idx + 20]):
    test_date = unique_dates[day_idx]
    train_start_idx = max(0, day_idx - TRAIN_WINDOW)
    train_dates = unique_dates[train_start_idx:day_idx]  # Exclusive of day_idx

    print(f'Test {i+1}:')
    print(f'  Prediction date: {test_date.date()}')
    print(f'  Training on: {train_dates[0].date()} to {train_dates[-1].date()}')
    print(f'  Training includes test date? {test_date in train_dates}')
    print(f'  Max train date < test date? {train_dates[-1] < test_date}')
    print()

print('5. CRITICAL: TARGET IN TRAINING CHECK')
print('-'*40)
print('Question: When training on row T, the target is the return from T+1 to T+1+horizon.')
print('Does this use any data beyond the test/prediction date?')
print()

test_date = unique_dates[test_start_idx]
train_dates = unique_dates[test_start_idx - TRAIN_WINDOW:test_start_idx]
last_train_date = train_dates[-1]

# For the last training row, when was the exit?
orig_idx = df_orig.index.get_loc(last_train_date)
last_train_exit_date = df_orig.index[orig_idx + 1 + HORIZON]

print(f'Test/prediction date: {test_date.date()}')
print(f'Last training date: {last_train_date.date()}')
print(f'Last training row exit date: {last_train_exit_date.date()}')
print(f'Exit date < test date? {last_train_exit_date < test_date}')
print()

if last_train_exit_date >= test_date:
    print('*** WARNING: Training target uses data that extends to or past test date! ***')
    print('This could be a source of lookahead bias!')
else:
    print('OK: All training targets use data strictly before the test date.')

print()
print('='*80)
print('6. DETAILED TIMELINE VISUALIZATION')
print('='*80)
print()
print('For a prediction on test date T:')
print()
print(f'  T-{TRAIN_WINDOW} ... T-1        = Training feature dates')
print(f'  T-{TRAIN_WINDOW-1}+1 ... T       = Training entry dates (T-{TRAIN_WINDOW-1} to T)')
print(f'  T-{TRAIN_WINDOW-1}+1+{HORIZON} ... T+{HORIZON} = Training exit dates')
print()
print(f'  T                  = Test feature date (we predict here)')
print(f'  T+1                = Test entry date')
print(f'  T+1+{HORIZON}             = Test exit date')
print()

# Check: does the training data exit overlap with test entry?
print('Critical check: Last training exit vs Test entry')
print(f'  Last train exit: T-1+1+{HORIZON} = T+{HORIZON}')
print(f'  Test entry: T+1')
print()
print(f'  T+{HORIZON} vs T+1: The last training exit (T+{HORIZON}) is AFTER test entry (T+1)!')
print()
print('='*80)
print('*** POTENTIAL BIAS FOUND ***')
print('='*80)
print()
print('The issue: When we train on the day before the test date (T-1),')
print(f'the target for that training row uses the exit price at T+{HORIZON}.')
print('But we are making a prediction on day T!')
print()
print('This means training data includes future price information')
print(f'(up to {HORIZON} days into the future relative to our prediction).')
