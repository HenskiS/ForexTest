"""
OANDA Production Trading Bot v3 - Optimized 1-Day Model
Implements rolling daily retraining with:
- 1-day target predictions (eliminates look-ahead bias)
- 1-day holding period with tight stops
- Optimized parameters: 0.18% stop loss, 2.00% take profit
- Volatility-adjusted stops
- No cooldown (immediate re-entry after exits)
- Position state persistence
- Realistic same-day exit then re-entry logic

WARNING: This bot trades real money. Test thoroughly on practice account first!
"""
import os
import sys
import requests
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

from oanda_data_fetcher import OandaDataFetcher
from notification_service import NotificationService
import pytz

load_dotenv()


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


class OandaTrader:
    """Production trading bot for OANDA with full backtest-matching logic"""

    def __init__(self, pair, practice=True, leverage=1.0):
        """
        Initialize OANDA trader.

        Args:
            pair: Forex pair to trade (e.g., 'EURUSD')
            practice: If True, use practice account. If False, use live account.
            leverage: Leverage multiplier (1.0 = no leverage, 2.0 = 2x, etc.)
        """
        self.pair = pair.upper()
        self.practice = practice

        # Initialize data fetcher
        self.fetcher = OandaDataFetcher(practice=practice)

        # API credentials
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in .env")

        # Set API endpoint
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Trading configuration (optimized 1-day model)
        self.TRAIN_WINDOW_SIZE = 756  # 756-day training window
        self.BASE_STOP_LOSS_PCT = 0.0018  # 0.18% (optimized via backtest)
        self.BASE_TAKE_PROFIT_PCT = 0.0200  # 2.00% (optimized via backtest)
        self.HOLDING_PERIOD = 1  # 1 day (optimized for 1-day predictions)
        self.LOSS_COOLDOWN_DAYS = 0  # No cooldown (optimized - immediate re-entry is best)
        self.LEVERAGE = leverage  # Leverage multiplier (1.0 = no leverage, 2.0 = 2x, etc.)

        # Pair-specific thresholds
        self.THRESHOLDS = {
            'EURUSD': (48, 52),
            'GBPUSD': (48, 52),
            'USDJPY': (35, 65),
            'AUDUSD': (48, 52),
        }

        # Technical features
        self.technical_features = [
            'momentum', 'avg_price', 'range', 'ohlc',
            'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'plus_di', 'minus_di',
            'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr'
        ]

        # State file for persistence
        self.state_file = f'data/oanda_cache/{self.pair}_state.json'
        os.makedirs('data/oanda_cache', exist_ok=True)

        # Trade log file
        self.trade_log_file = f'data/oanda_cache/{self.pair}_trades.csv'

        # Load state
        self.load_state()

        # Prediction buffer
        self.load_prediction_buffer()

        # Model and scaler (loaded on demand)
        self.model = None
        self.scaler = None

        # Load hyperparameters
        self.best_params = self.load_hyperparameters()

        # Cache directory
        self.cache_dir = 'data/oanda_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        # Notification service
        self.notifier = NotificationService()

    def load_state(self):
        """Load persisted state from disk"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.position = state.get('position', 0)
                self.entry_price = state.get('entry_price')
                self.entry_date = state.get('entry_date')
                if self.entry_date:
                    self.entry_date = datetime.fromisoformat(self.entry_date)
                self.trade_id = state.get('trade_id')
                self.position_size = state.get('position_size', 0)  # Loaded from state
                self.cooldown_until = state.get('cooldown_until')
                if self.cooldown_until:
                    self.cooldown_until = datetime.fromisoformat(self.cooldown_until)
                print(f"Loaded state: position={self.position}, entry_date={self.entry_date}")
        else:
            self.position = 0
            self.entry_price = None
            self.entry_date = None
            self.trade_id = None
            self.position_size = 0  # Will be calculated from account balance
            self.cooldown_until = None

    def save_state(self):
        """Persist state to disk"""
        state = {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'trade_id': self.trade_id,
            'position_size': self.position_size,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_prediction_buffer(self):
        """Load prediction buffer for percentile calculations"""
        buffer_file = f'data/oanda_cache/prediction_buffer_{self.pair}.pkl'
        if os.path.exists(buffer_file):
            with open(buffer_file, 'rb') as f:
                self.prediction_buffer = pickle.load(f)
            print(f"Loaded prediction buffer: {len(self.prediction_buffer)} predictions")
        else:
            self.prediction_buffer = []
            print("No prediction buffer found. Run initialize_oanda_buffer.py first.")

    def save_prediction_buffer(self):
        """Save prediction buffer to disk"""
        buffer_file = f'data/oanda_cache/prediction_buffer_{self.pair}.pkl'
        with open(buffer_file, 'wb') as f:
            pickle.dump(self.prediction_buffer, f)

    def load_hyperparameters(self):
        """Load optimized hyperparameters"""
        hyperparam_file = f'hyperparams_rolling_daily_{self.pair}.pkl'
        if os.path.exists(hyperparam_file):
            with open(hyperparam_file, 'rb') as f:
                params = pickle.load(f)
            print(f"Loaded hyperparameters from {hyperparam_file}")
            return params
        else:
            print(f"No hyperparameters found, using defaults")
            return {
                'n_estimators': 125,
                'learning_rate': 0.1,
                'max_depth': 5,
                'gamma': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.7
            }

    def fetch_latest_data(self):
        """Fetch latest data from OANDA"""
        main_oanda_file = f'data/{self.pair}_1day_oanda.csv'
        if os.path.exists(main_oanda_file):
            df_historical = pd.read_csv(main_oanda_file)
            df_historical['date'] = pd.to_datetime(df_historical['date'])
            df_historical = df_historical.set_index('date')
            print(f"Loaded OANDA historical data: {len(df_historical)} days ({df_historical.index[0].date()} to {df_historical.index[-1].date()})")

            # Fetch latest 5 candles from OANDA to get today's data
            print(f"Fetching latest candles from OANDA...")
            df_latest = self.fetcher.get_historical_data(
                self.pair,
                count=5,  # Fetch last 5 days to ensure we have today
                granularity='D'
            )

            if not df_latest.empty:
                df_latest = df_latest.set_index('date')

                # Find new candles (dates not in historical data)
                new_dates = df_latest.index.difference(df_historical.index)

                if len(new_dates) > 0:
                    df_new = df_latest.loc[new_dates]
                    print(f"Appending {len(new_dates)} new candle(s): {[d.date() for d in new_dates]}")

                    # Combine historical + new data
                    df = pd.concat([df_historical, df_new]).sort_index()
                    print(f"Total data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
                    return df
                else:
                    print(f"No new candles found (data up to date)")
                    return df_historical
            else:
                print(f"WARNING: Failed to fetch latest candles, using historical data only")
                return df_historical

        # Otherwise fetch from API
        print(f"Fetching latest data from OANDA...")
        df = self.fetcher.get_historical_data(
            self.pair,
            count=self.TRAIN_WINDOW_SIZE + 300,
            granularity='D'
        )

        if df.empty:
            raise ValueError("Failed to fetch data from OANDA")

        df = df.set_index('date')
        return df

    def calculate_features(self, df):
        """Calculate all technical features"""
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
        tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
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

        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma) / (0.015 * mad)

        # Williams %R
        df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

        # Bollinger Bands
        middle = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = middle + (std * 2)
        df['bb_middle'] = middle
        df['bb_lower'] = middle - (std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR
        df['atr'] = atr

        return df

    def train_model(self, df):
        """Train XGBoost model on last 756 days with 1-day gap"""
        print(f"\n{'='*70}")
        print(f"Training model on last {self.TRAIN_WINDOW_SIZE} days...")
        print(f"{'='*70}")

        # Calculate features
        df_with_features = self.calculate_features(df.copy())

        # Calculate target (1-day forward return)
        df_with_features['target_1day_return'] = df_with_features['close'].pct_change(1).shift(-1)

        # Drop NaN rows
        df_clean = df_with_features.dropna(subset=self.technical_features + ['target_1day_return'])

        if len(df_clean) < self.TRAIN_WINDOW_SIZE + 1:
            raise ValueError(f"Need {self.TRAIN_WINDOW_SIZE + 1} days, have {len(df_clean)}")

        # Take last 756 days with 1-day gap (train through yesterday, predict today)
        # This eliminates look-ahead bias
        train_data = df_clean.iloc[-(self.TRAIN_WINDOW_SIZE + 1):-1].copy()

        print(f"Training data: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"(1-day gap: training excludes today to prevent look-ahead bias)")

        # Prepare training data
        X_train = train_data[self.technical_features].values
        y_train = train_data['target_1day_return'].values

        # Scale features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.best_params.get('n_estimators', 125),
            learning_rate=self.best_params.get('learning_rate', 0.1),
            max_depth=self.best_params.get('max_depth', 5),
            gamma=self.best_params.get('gamma', 0.1),
            subsample=self.best_params.get('subsample', 0.9),
            colsample_bytree=self.best_params.get('colsample_bytree', 0.7),
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train, verbose=False)
        print("Model trained successfully")

        return df_clean

    def generate_signal(self, df_clean):
        """Generate trading signal using percentile thresholds"""
        print(f"\n{'='*70}")
        print("Generating Signal")
        print(f"{'='*70}")

        # Get latest features
        latest_features = df_clean.iloc[[-1]][self.technical_features].values
        X_scaled = self.scaler.transform(latest_features)

        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        print(f"Prediction: {prediction:.6f}")

        # Update prediction buffer
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > 200:
            self.prediction_buffer = self.prediction_buffer[-200:]
        self.save_prediction_buffer()

        # Calculate thresholds
        if len(self.prediction_buffer) < 50:
            print("Not enough predictions in buffer, signal = 0 (HOLD)")
            return 0, prediction

        buffer_array = np.array(self.prediction_buffer)
        lower_pct, upper_pct = self.THRESHOLDS.get(self.pair, (48, 52))
        lower_threshold = np.percentile(buffer_array, lower_pct)
        upper_threshold = np.percentile(buffer_array, upper_pct)

        print(f"Buffer: {len(self.prediction_buffer)} predictions")
        print(f"Thresholds: {lower_pct}th={lower_threshold:.6f}, {upper_pct}th={upper_threshold:.6f}")

        # Generate signal
        if prediction >= upper_threshold:
            signal = 1
        elif prediction <= lower_threshold:
            signal = -1
        else:
            signal = 0

        signal_str = {1: 'LONG', 0: 'HOLD', -1: 'SHORT'}[signal]
        print(f"Signal: {signal} ({signal_str})")

        return signal, prediction

    def check_and_close_position(self, df_clean, dry_run=False, prediction=None):
        """Check if position needs to be closed (1-day exit or stops)"""
        if self.position == 0:
            return None

        print(f"\n{'='*70}")
        print("Checking Open Position")
        print(f"{'='*70}")
        print(f"Position: {'LONG' if self.position == 1 else 'SHORT'}")
        print(f"Entry: {self.entry_price:.5f} on {self.entry_date.date()}")

        # Calculate days held
        days_held = (datetime.now() - self.entry_date).days
        print(f"Days held: {days_held}")

        # Get current price
        current_price_data = self.fetcher.get_current_price(self.pair)
        if not current_price_data:
            print("ERROR: Failed to get current price")
            return None

        current_price = current_price_data['mid']
        print(f"Current price: {current_price:.5f}")

        # Calculate P&L
        if self.position == 1:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

        print(f"P&L: {pnl_pct:.2f}%")

        # Get current ATR for volatility adjustment
        latest_atr = df_clean.iloc[-1]['atr']
        median_atr = df_clean['atr'].median()
        vol_ratio = latest_atr / median_atr if not np.isnan(latest_atr) else 1.0
        stop_loss_pct = self.BASE_STOP_LOSS_PCT * vol_ratio
        take_profit_pct = self.BASE_TAKE_PROFIT_PCT * vol_ratio

        print(f"Volatility ratio: {vol_ratio:.2f}x")
        print(f"Adjusted stop: {stop_loss_pct*100:.2f}%, take-profit: {take_profit_pct*100:.2f}%")

        # Check exit conditions
        should_close = False
        close_reason = None

        if self.position == 1:
            if pnl_pct <= -stop_loss_pct * 100:
                should_close = True
                close_reason = 'STOP_LOSS'
            elif pnl_pct >= take_profit_pct * 100:
                should_close = True
                close_reason = 'TAKE_PROFIT'
        else:
            if pnl_pct <= -stop_loss_pct * 100:
                should_close = True
                close_reason = 'STOP_LOSS'
            elif pnl_pct >= take_profit_pct * 100:
                should_close = True
                close_reason = 'TAKE_PROFIT'

        if days_held >= self.HOLDING_PERIOD:
            should_close = True
            close_reason = 'HOLDING_PERIOD'

        if should_close:
            print(f"\nClosing position: {close_reason}")
            if dry_run:
                print(f"[DRY RUN] Would close position at {current_price:.5f}")
                print(f"[DRY RUN] P&L: {pnl_pct:.2f}%")
                print(f"[DRY RUN] Would log trade: {close_reason}")
            else:
                # Save trade details before closing
                entry_date_save = self.entry_date
                entry_price_save = self.entry_price
                position_save = self.position
                position_size_save = self.position_size

                success = self.close_position(current_price)
                if success:
                    print(f"Position closed successfully")

                    # Calculate P&L in dollars
                    pnl_dollars = position_size_save * (pnl_pct / 100)

                    # Send notification
                    self.notifier.notify_trade_exit(
                        pair=self.pair,
                        direction='LONG' if position_save == 1 else 'SHORT',
                        entry_price=entry_price_save,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        pnl_dollars=pnl_dollars,
                        exit_reason=close_reason,
                        days_held=(datetime.now() - entry_date_save).days
                    )

                    # Log trade
                    self.log_trade(
                        entry_date=entry_date_save,
                        exit_date=datetime.now(),
                        direction=position_save,
                        entry_price=entry_price_save,
                        exit_price=current_price,
                        pnl_pct=pnl_pct,
                        exit_reason=close_reason,
                        position_size=position_size_save,
                        prediction=prediction,
                        signal=position_save
                    )

                    # Set cooldown if loss
                    if pnl_pct < 0 and self.LOSS_COOLDOWN_DAYS > 0:
                        self.cooldown_until = datetime.now() + timedelta(days=self.LOSS_COOLDOWN_DAYS)
                        print(f"Loss cooldown active until {self.cooldown_until.date()}")
                    return pnl_pct
                else:
                    print("ERROR: Failed to close position")
                    return None

        return None

    def log_trade(self, entry_date, exit_date, direction, entry_price, exit_price,
                  pnl_pct, exit_reason, position_size=None, prediction=None, signal=None):
        """Log trade to CSV file"""
        # Calculate P&L in absolute dollars if position size provided
        pnl_dollars = None
        if position_size is not None:
            pnl_dollars = position_size * (pnl_pct / 100)

        trade_record = {
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl_pct': round(pnl_pct, 4),
            'pnl_dollars': round(pnl_dollars, 2) if pnl_dollars is not None else None,
            'outcome': 'WIN' if pnl_pct > 0 else 'LOSS',
            'exit_reason': exit_reason,
            'days_held': (exit_date - entry_date).days,
            'prediction': round(prediction, 6) if prediction is not None else None,
            'signal': signal
        }

        # Create DataFrame and append to CSV
        trade_df = pd.DataFrame([trade_record])

        if os.path.exists(self.trade_log_file):
            trade_df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
        else:
            trade_df.to_csv(self.trade_log_file, mode='w', header=True, index=False)

        print(f"\nTrade logged to {self.trade_log_file}")

    def close_position(self, current_price):
        """Close current position via OANDA API"""
        instrument = self.fetcher.get_instrument_name(self.pair)

        # Close all positions for this instrument
        url = f"{self.base_url}/accounts/{self.account_id}/positions/{instrument}/close"

        data = {
            "longUnits": "ALL" if self.position == 1 else "NONE",
            "shortUnits": "ALL" if self.position == -1 else "NONE"
        }

        try:
            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()

            # Reset position state
            self.position = 0
            self.entry_price = None
            self.entry_date = None
            self.trade_id = None
            self.position_size = 0  # Will be recalculated on next entry
            self.save_state()

            return True

        except requests.exceptions.RequestException as e:
            print(f"Error closing position: {e}")
            return False

    def get_account_balance(self):
        """Get available account balance from OANDA"""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            # Get available balance (cash, no margin)
            balance = float(result['account']['balance'])
            print(f"Account balance: ${balance:.2f}")
            return balance

        except requests.exceptions.RequestException as e:
            print(f"Error fetching account balance: {e}")
            return None

    def get_open_positions(self):
        """Get all open positions from OANDA to verify no duplicate trades"""
        instrument = self.fetcher.get_instrument_name(self.pair)
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()

            # Check if there's an open position for this instrument
            for position in result.get('positions', []):
                if position['instrument'] == instrument:
                    long_units = float(position['long']['units'])
                    short_units = float(position['short']['units'])

                    if long_units != 0 or short_units != 0:
                        return {
                            'instrument': instrument,
                            'long_units': long_units,
                            'short_units': short_units,
                            'unrealized_pl': float(position.get('unrealizedPL', 0))
                        }

            return None  # No open position for this instrument

        except requests.exceptions.RequestException as e:
            print(f"Error fetching open positions: {e}")
            return None

    def place_order(self, signal, current_price, df_clean):
        """Place market order with volatility-adjusted stops"""
        instrument = self.fetcher.get_instrument_name(self.pair)

        # SAFETY CHECK: Verify no existing position at OANDA API level
        existing_position = self.get_open_positions()
        if existing_position:
            print(f"\n⚠️  WARNING: Open position already exists at OANDA!")
            print(f"  Long units: {existing_position['long_units']}")
            print(f"  Short units: {existing_position['short_units']}")
            print(f"  Unrealized P&L: ${existing_position['unrealized_pl']:.2f}")
            print(f"  Skipping new order to prevent duplicate trade")
            return False

        # Get current account balance
        account_balance = self.get_account_balance()
        if account_balance is None:
            print("ERROR: Failed to fetch account balance")
            return False

        # Calculate position size based on account balance and leverage
        position_size_dollars = account_balance * self.LEVERAGE
        if self.LEVERAGE > 1.0:
            print(f"Position size: ${position_size_dollars:.2f} ({self.LEVERAGE:.1f}x leverage on ${account_balance:.2f})")
        else:
            print(f"Position size: ${position_size_dollars:.2f} (100% of balance, no leverage)")

        # Calculate volatility-adjusted stops
        latest_atr = df_clean.iloc[-1]['atr']
        median_atr = df_clean['atr'].median()
        vol_ratio = latest_atr / median_atr if not np.isnan(latest_atr) else 1.0
        stop_loss_pct = self.BASE_STOP_LOSS_PCT * vol_ratio
        take_profit_pct = self.BASE_TAKE_PROFIT_PCT * vol_ratio

        # Calculate units based on position size
        units = int(position_size_dollars / current_price)

        # Calculate stop/target prices
        if signal == 1:  # Long
            stop_price = current_price * (1 - stop_loss_pct)
            target_price = current_price * (1 + take_profit_pct)
            order_units = abs(units)
        else:  # Short
            stop_price = current_price * (1 + stop_loss_pct)
            target_price = current_price * (1 - take_profit_pct)
            order_units = -abs(units)

        # Prepare order
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(order_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{stop_price:.5f}"
                },
                "takeProfitOnFill": {
                    "price": f"{target_price:.5f}"
                }
            }
        }

        url = f"{self.base_url}/accounts/{self.account_id}/orders"

        try:
            print(f"\nPlacing {'LONG' if signal == 1 else 'SHORT'} order:")
            print(f"  Units: {order_units}")
            print(f"  Entry: {current_price:.5f}")
            print(f"  Stop Loss: {stop_price:.5f} ({stop_loss_pct*100:.2f}%)")
            print(f"  Take Profit: {target_price:.5f} ({take_profit_pct*100:.2f}%)")
            print(f"  Volatility adjustment: {vol_ratio:.2f}x")

            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            result = response.json()

            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                self.position = signal
                self.entry_price = float(fill['price'])
                self.entry_date = datetime.now()
                self.trade_id = fill['id']
                self.position_size = position_size_dollars
                self.save_state()

                print(f"Order filled at {self.entry_price:.5f}")
                print(f"Position size: ${position_size_dollars}")
                print(f"Trade ID: {self.trade_id}")

                # Send notification
                self.notifier.notify_trade_entry(
                    pair=self.pair,
                    direction='LONG' if signal == 1 else 'SHORT',
                    entry_price=self.entry_price,
                    position_size=position_size_dollars,
                    stop_loss=stop_price,
                    take_profit=target_price
                )

                return True
            else:
                print(f"Order not filled: {result}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return False

    def run_daily_update(self, dry_run=False):
        """
        Run daily trading workflow:
        1. Check and close existing position if needed
        2. Generate new signal
        3. Place new order if signal and not in cooldown
        """
        print(f"\n{'='*70}")
        print(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pair: {self.pair}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"{'='*70}\n")

        # Check if market is open
        if not is_forex_market_open():
            et_tz = pytz.timezone('America/New_York')
            now_et = datetime.now(et_tz)
            print(f"Forex market is CLOSED (current time: {now_et.strftime('%A %Y-%m-%d %H:%M:%S %Z')})")
            print(f"Market hours: Sunday 5:00 PM ET to Friday 5:00 PM ET")
            print(f"Exiting without executing any trading logic.")
            return

        # Step 1: Fetch data and train model
        df = self.fetch_latest_data()
        df_clean = self.train_model(df)

        # Step 2: Generate signal (needed for logging)
        signal, prediction = self.generate_signal(df_clean)

        # Step 3: Check and close existing position (pass prediction for logging)
        if self.position != 0:
            self.check_and_close_position(df_clean, dry_run=dry_run, prediction=prediction)

        # Step 4: Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            print(f"\nIn cooldown until {self.cooldown_until.date()}, skipping new entry")
            return

        if signal == 0:
            print("\nNo trade signal (HOLD)")
            return

        # Step 5: Place order
        if self.position != 0:
            print("\nAlready in position, skipping new entry")
            return

        current_price_data = self.fetcher.get_current_price(self.pair)
        if not current_price_data:
            print("ERROR: Failed to get current price")
            return

        current_price = current_price_data['mid']

        if dry_run:
            # Get account balance for dry-run display
            account_balance = self.get_account_balance()
            if account_balance is None:
                print("ERROR: Failed to fetch account balance")
                return

            position_size_dollars = account_balance * self.LEVERAGE
            if self.LEVERAGE > 1.0:
                print(f"Position size: ${position_size_dollars:.2f} ({self.LEVERAGE:.1f}x leverage on ${account_balance:.2f})")
            else:
                print(f"Position size: ${position_size_dollars:.2f} (100% of balance, no leverage)")

            # Calculate volatility-adjusted stops for display
            latest_atr = df_clean.iloc[-1]['atr']
            median_atr = df_clean['atr'].median()
            vol_ratio = latest_atr / median_atr if not np.isnan(latest_atr) else 1.0
            stop_loss_pct = self.BASE_STOP_LOSS_PCT * vol_ratio
            take_profit_pct = self.BASE_TAKE_PROFIT_PCT * vol_ratio

            units = int(position_size_dollars / current_price)

            if signal == 1:  # Long
                stop_price = current_price * (1 - stop_loss_pct)
                target_price = current_price * (1 + take_profit_pct)
                order_units = abs(units)
            else:  # Short
                stop_price = current_price * (1 + stop_loss_pct)
                target_price = current_price * (1 - take_profit_pct)
                order_units = -abs(units)

            print(f"\n[DRY RUN] Would place {'LONG' if signal == 1 else 'SHORT'} order:")
            print(f"  Units: {order_units}")
            print(f"  Entry: {current_price:.5f}")
            print(f"  Stop Loss: {stop_price:.5f} ({stop_loss_pct*100:.2f}%)")
            print(f"  Take Profit: {target_price:.5f} ({take_profit_pct*100:.2f}%)")
            print(f"  Volatility adjustment: {vol_ratio:.2f}x")
        else:
            success = self.place_order(signal, current_price, df_clean)
            if success:
                print("\n[OK] Trade executed successfully!")
            else:
                print("\n[FAIL] Trade execution failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OANDA Production Trader v2')
    parser.add_argument('--pair', type=str, default='EURUSD', help='Currency pair to trade')
    parser.add_argument('--live', action='store_true', help='Use LIVE account')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only, do not place trades')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier (default: 1.0 = no leverage)')
    args = parser.parse_args()

    # Validate leverage
    if args.leverage < 1.0:
        print("ERROR: Leverage must be >= 1.0")
        sys.exit(1)
    if args.leverage > 50.0:
        print("ERROR: Leverage cannot exceed 50x (OANDA maximum)")
        sys.exit(1)

    # Confirm if using live account
    if args.live and not args.dry_run:
        print(f"\nWARNING: You are about to trade on a LIVE account")
        print(f"Leverage: {args.leverage:.1f}x")
        print(f"Pair: {args.pair}")
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    # Warn about leverage
    if args.leverage > 1.0 and not args.dry_run:
        print(f"\n⚠️  WARNING: Using {args.leverage:.1f}x leverage increases risk!")
        print(f"Max loss per trade: ~{0.004 * args.leverage * 100:.2f}% of account balance")
        confirm_leverage = input("Type 'YES' to confirm leverage: ")
        if confirm_leverage != 'YES':
            print("Aborted.")
            sys.exit(0)

    try:
        trader = OandaTrader(args.pair, practice=not args.live, leverage=args.leverage)
        trader.run_daily_update(dry_run=args.dry_run)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
