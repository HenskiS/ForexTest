"""
OANDA Production Trading Bot
Implements rolling daily retraining strategy with live OANDA execution.

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
import time
import json

from oanda_data_fetcher import OandaDataFetcher

load_dotenv()


class OandaTrader:
    """Production trading bot for OANDA"""

    def __init__(self, pair, practice=True):
        """
        Initialize OANDA trader.

        Args:
            pair: Forex pair to trade (e.g., 'EURUSD')
            practice: If True, use practice account. If False, use live account.
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

        # Trading configuration (from your backtest)
        self.TRAIN_WINDOW_SIZE = 756  # 600 train + 156 val
        self.BASE_STOP_LOSS_PCT = 0.0040  # 0.40%
        self.BASE_TAKE_PROFIT_PCT = 0.0100  # 1.00%
        self.TRANSACTION_COST_PCT = 0.0002  # 0.02%
        self.HOLDING_PERIOD = 5  # days
        self.LOSS_COOLDOWN_DAYS = 1

        # Pair-specific thresholds (from optimization)
        self.THRESHOLDS = {
            'EURUSD': (48, 52),
            'GBPUSD': (48, 52),
            'USDJPY': (35, 65),
            'AUDUSD': (48, 52),
        }

        # Technical features (must match training)
        self.technical_features = [
            'momentum', 'avg_price', 'range', 'ohlc',
            'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'plus_di', 'minus_di',
            'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr'
        ]

        # State tracking
        self.position = 0  # -1 = short, 0 = none, 1 = long
        self.entry_price = None
        self.entry_date = None
        self.trade_id = None
        self.cooldown_until = None

        # Prediction buffer for threshold calculation (rolling window)
        self.prediction_buffer = []
        self.prediction_buffer_size = 200  # Use last 200 predictions for percentiles

        # Load optimized hyperparameters
        self.best_params = self.load_hyperparameters()

        # Cache management
        self.cache_dir = 'data/oanda_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"OANDA Trader initialized for {self.pair} ({'PRACTICE' if practice else 'LIVE'})")

    def load_hyperparameters(self):
        """Load optimized hyperparameters for rolling daily approach"""
        # Try to load rolling daily optimized params first
        hyperparam_file = f'hyperparams_rolling_daily_{self.pair}.pkl'
        if os.path.exists(hyperparam_file):
            with open(hyperparam_file, 'rb') as f:
                best_params = pickle.load(f)
            print(f"Loaded optimized hyperparameters from {hyperparam_file}")
            return best_params

        # Fallback to static results
        try:
            with open(f'xgboost_results_{self.pair}_target_5day_return.pkl', 'rb') as f:
                static_results = pickle.load(f)
            best_params = static_results[-1]['best_params']
            print(f"Using hyperparameters from static results")
            return best_params
        except:
            # Default params
            best_params = {
                'n_estimators': 125,
                'learning_rate': 0.1,
                'max_depth': 5,
                'gamma': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.7
            }
            print(f"Using default hyperparameters")
            return best_params

    def load_prediction_buffer(self):
        """Load prediction buffer from disk"""
        buffer_file = f'{self.cache_dir}/prediction_buffer_{self.pair}.pkl'
        if os.path.exists(buffer_file):
            with open(buffer_file, 'rb') as f:
                self.prediction_buffer = pickle.load(f)
            print(f"Loaded prediction buffer: {len(self.prediction_buffer)} predictions")

    def save_prediction_buffer(self):
        """Save prediction buffer to disk"""
        buffer_file = f'{self.cache_dir}/prediction_buffer_{self.pair}.pkl'
        with open(buffer_file, 'wb') as f:
            pickle.dump(self.prediction_buffer, f)

    def update_prediction_buffer(self, prediction):
        """Add new prediction to buffer, maintaining fixed size"""
        self.prediction_buffer.append(prediction)
        # Keep only last N predictions
        if len(self.prediction_buffer) > self.prediction_buffer_size:
            self.prediction_buffer = self.prediction_buffer[-self.prediction_buffer_size:]
        self.save_prediction_buffer()

    def get_cached_data(self):
        """Load cached OANDA data if it exists and is recent"""
        cache_file = f'{self.cache_dir}/{self.pair}_daily.csv'

        if not os.path.exists(cache_file):
            return None

        # Check if cache is from today
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_mtime.date() < datetime.now().date():
            print("Cache is outdated, will fetch new data")
            return None

        # Load cached data
        df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
        print(f"Loaded cached data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        return df

    def save_cached_data(self, df):
        """Save data to cache"""
        cache_file = f'{self.cache_dir}/{self.pair}_daily.csv'
        df.to_csv(cache_file)
        print(f"Saved data to cache: {cache_file}")

    def fetch_latest_data(self):
        """Fetch latest data from OANDA, using cache when possible"""
        # First check if we have the main OANDA data file (5000 days)
        main_oanda_file = f'data/{self.pair}_1day_oanda.csv'
        if os.path.exists(main_oanda_file):
            df = pd.read_csv(main_oanda_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            print(f"Loaded OANDA data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            return df

        # Try to load cached data
        df_cached = self.get_cached_data()

        if df_cached is not None:
            # Check if we have enough data (need extra for feature warmup)
            # Need: 756 training + ~250 for feature warmup (EMA_200, etc.)
            if len(df_cached) >= self.TRAIN_WINDOW_SIZE + 250:
                return df_cached

        # Fetch new data from OANDA
        print(f"Fetching latest data from OANDA...")
        df = self.fetcher.get_historical_data(
            self.pair,
            count=self.TRAIN_WINDOW_SIZE + 300,  # Fetch extra for feature warmup
            granularity='D'
        )

        if df.empty:
            raise ValueError("Failed to fetch data from OANDA")

        # Set date as index
        df = df.set_index('date')

        # Save to cache
        self.save_cached_data(df)

        return df

    def calculate_features(self, df):
        """Calculate technical features (must match training exactly)"""
        from ta.trend import EMAIndicator, MACD, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.trend import CCIIndicator

        # Basic price features
        df['momentum'] = df['close'].pct_change()
        df['avg_price'] = (df['high'] + df['low']) / 2
        df['range'] = df['high'] - df['low']
        df['ohlc'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # EMAs
        for period in [10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()

        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['plus_di'] = adx.adx_pos()
        df['minus_di'] = adx.adx_neg()

        # RSI
        df['rsi'] = RSIIndicator(df['close']).rsi()

        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # CCI
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()

        # Williams %R
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

        # Bollinger Bands
        bb = BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        return df

    def train_model(self, df):
        """Train XGBoost model on most recent 756 days with optimized hyperparameters"""
        print(f"\n{'='*70}")
        print(f"Training model on last {self.TRAIN_WINDOW_SIZE} days...")
        print(f"{'='*70}")

        # Calculate features first on ALL available data (need extra for warmup)
        df_with_features = self.calculate_features(df.copy())

        # Calculate target (5-day forward return)
        df_with_features['target_5day_return'] = df_with_features['close'].pct_change(5).shift(-5)

        # Drop NaN rows (from indicators and target)
        df_clean = df_with_features.dropna(subset=self.technical_features + ['target_5day_return'])

        if len(df_clean) < self.TRAIN_WINDOW_SIZE:
            raise ValueError(f"After feature calculation, need {self.TRAIN_WINDOW_SIZE} days, have {len(df_clean)}")

        # NOW take last 756 clean days
        train_data = df_clean.iloc[-self.TRAIN_WINDOW_SIZE:].copy()

        print(f"Training data: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"Hyperparameters: {self.best_params}")

        # Prepare training data
        X_train = train_data[self.technical_features].values
        y_train = train_data['target_5day_return'].values

        # Scale features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model with optimized hyperparameters
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

        print(f"Model trained successfully!")
        print(f"{'='*70}\n")

    def generate_signal(self, df):
        """Generate trading signal for today using percentile-based thresholds"""
        # Calculate features for current day
        df_with_features = self.calculate_features(df.copy())
        df_clean = df_with_features.dropna(subset=self.technical_features)

        if len(df_clean) == 0:
            print("ERROR: No valid data after feature calculation")
            return 0

        # Get today's features
        X_today = df_clean[self.technical_features].iloc[-1:].values
        X_today_scaled = self.scaler.transform(X_today)

        # Make prediction
        prediction = self.model.predict(X_today_scaled)[0]

        print(f"Prediction: {prediction:.6f}")

        # Update prediction buffer
        self.update_prediction_buffer(prediction)

        # Get thresholds for this pair
        lower_pct, upper_pct = self.THRESHOLDS.get(self.pair, (48, 52))

        # Generate signal based on percentile thresholds
        if len(self.prediction_buffer) < 50:
            print(f"Warning: Only {len(self.prediction_buffer)} predictions in buffer (need 50+ for reliable thresholds)")
            print("Using absolute thresholds as fallback")
            # Fallback to absolute thresholds
            if prediction > 0.001:
                signal = 1
            elif prediction < -0.001:
                signal = -1
            else:
                signal = 0
        else:
            # Calculate percentile thresholds from buffer
            buffer_array = np.array(self.prediction_buffer)
            lower_threshold = np.percentile(buffer_array, lower_pct)
            upper_threshold = np.percentile(buffer_array, upper_pct)

            print(f"Buffer stats: mean={buffer_array.mean():.6f}, std={buffer_array.std():.6f}")
            print(f"Thresholds: {lower_pct}th={lower_threshold:.6f}, {upper_pct}th={upper_threshold:.6f}")

            # Generate signal
            if prediction >= upper_threshold:
                signal = 1  # Long
            elif prediction <= lower_threshold:
                signal = -1  # Short
            else:
                signal = 0  # Hold

        print(f"Signal: {signal} (Long={1}, Hold={0}, Short={-1})")

        return signal

    def place_order(self, signal, units, current_price):
        """Place market order with stop-loss and take-profit"""
        instrument = self.fetcher.get_instrument_name(self.pair)

        # Calculate volatility-adjusted stops
        # (In production, fetch recent ATR)
        stop_loss_pct = self.BASE_STOP_LOSS_PCT
        take_profit_pct = self.BASE_TAKE_PROFIT_PCT

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
                "timeInForce": "FOK",  # Fill or Kill
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

            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            result = response.json()

            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                self.position = signal
                self.entry_price = float(fill['price'])
                self.entry_date = datetime.now()
                self.trade_id = fill['id']

                print(f"✓ Order filled at {self.entry_price:.5f}")
                print(f"  Trade ID: {self.trade_id}")
                return True
            else:
                print(f"Order not filled: {result}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return False

    def check_open_positions(self):
        """Check if we have any open positions"""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            instrument = self.fetcher.get_instrument_name(self.pair)
            trades = [t for t in data.get('trades', []) if t['instrument'] == instrument]

            if trades:
                trade = trades[0]  # Should only be one per pair
                self.position = 1 if float(trade['currentUnits']) > 0 else -1
                self.entry_price = float(trade['price'])
                self.trade_id = trade['id']
                return True
            else:
                self.position = 0
                return False

        except requests.exceptions.RequestException as e:
            print(f"Error checking positions: {e}")
            return False

    def get_account_info(self):
        """Get account balance and summary"""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            account = data['account']
            return {
                'balance': float(account['balance']),
                'nav': float(account['NAV']),
                'unrealized_pl': float(account['unrealizedPL']),
                'margin_used': float(account['marginUsed']),
                'margin_available': float(account['marginAvailable']),
                'open_trades': int(account['openTradeCount']),
                'currency': account['currency']
            }

        except requests.exceptions.RequestException as e:
            print(f"Error getting account info: {e}")
            return None

    def run_daily_update(self, position_size_usd=1000, dry_run=False):
        """
        Run daily trading workflow:
        1. Fetch latest data (using cache)
        2. Retrain model on last 756 days
        3. Generate signal using percentile thresholds
        4. Execute trade if needed (unless dry_run=True)
        """
        print(f"\n{'='*70}")
        print(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pair: {self.pair}")
        print(f"{'='*70}\n")

        # Get account info
        account = self.get_account_info()
        if account:
            print(f"Account Balance: {account['balance']:.2f} {account['currency']}")
            print(f"Open Trades: {account['open_trades']}")
            print(f"Unrealized P&L: {account['unrealized_pl']:.2f}")
            print()

        # Check for existing positions
        has_position = self.check_open_positions()
        if has_position:
            print(f"⚠ Already have {['SHORT', 'NONE', 'LONG'][self.position+1]} position open")
            print(f"  Entry: {self.entry_price:.5f}")
            print(f"  Trade ID: {self.trade_id}")
            print("  Skipping new trade entry")
            return

        # Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            print(f"⚠ In cooldown until {self.cooldown_until}")
            print("  Skipping trade entry")
            return

        # Load prediction buffer
        self.load_prediction_buffer()

        # Step 1: Fetch latest data (using cache when possible)
        print("Fetching latest data...")
        df = self.fetch_latest_data()
        print(f"Data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

        # Step 2: Retrain model
        try:
            self.train_model(df)
        except Exception as e:
            print(f"ERROR: Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Step 3: Generate signal
        signal = self.generate_signal(df)

        if signal == 0:
            print("No trade signal (HOLD)")
            return

        # Step 4: Execute trade (or simulate if dry_run)
        current_price_data = self.fetcher.get_current_price(self.pair)
        if not current_price_data:
            print("ERROR: Failed to get current price")
            return

        current_price = current_price_data['mid']
        spread_pips = current_price_data['spread_pips']

        print(f"\nCurrent Price: {current_price:.5f}")
        print(f"Spread: {spread_pips:.1f} pips")

        # Calculate position size (in units)
        units = int(position_size_usd / current_price)

        if dry_run:
            print(f"\n[DRY RUN] Would place {'LONG' if signal == 1 else 'SHORT'} order:")
            print(f"  Units: {units if signal == 1 else -units}")
            print(f"  Entry: {current_price:.5f}")
            print(f"  Stop Loss: {current_price * (1 - self.BASE_STOP_LOSS_PCT if signal == 1 else 1 + self.BASE_STOP_LOSS_PCT):.5f}")
            print(f"  Take Profit: {current_price * (1 + self.BASE_TAKE_PROFIT_PCT if signal == 1 else 1 - self.BASE_TAKE_PROFIT_PCT):.5f}")
            print(f"\n[OK] Signal generated (no trade placed in dry-run mode)")
        else:
            # Place order
            success = self.place_order(signal, units, current_price)

            if success:
                print("✓ Trade executed successfully!")
            else:
                print("✗ Trade execution failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OANDA Production Trader')
    parser.add_argument('--pair', type=str, default='EURUSD', help='Forex pair to trade')
    parser.add_argument('--live', action='store_true', help='Use LIVE account (default: practice)')
    parser.add_argument('--position-size', type=float, default=1000, help='Position size in USD')
    parser.add_argument('--dry-run', action='store_true', help='Generate signals but do not place trades')
    args = parser.parse_args()

    # Confirm if using live account (skip confirmation in dry-run mode)
    if args.live and not args.dry_run:
        confirm = input("WARNING: You are about to trade on a LIVE account. Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            sys.exit(0)

    try:
        trader = OandaTrader(args.pair, practice=not args.live)

        if args.dry_run:
            print("\n[DRY RUN MODE] No trades will be placed\n")

        trader.run_daily_update(position_size_usd=args.position_size, dry_run=args.dry_run)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
