"""
Trading Configuration

Centralized configuration for trading parameters, model hyperparameters,
and file paths.
"""


class TradingConfig:
    """Trading configuration and parameters"""

    # Trading Parameters
    STOP_LOSS_PCT = 0.0018  # 0.18% stop loss
    TAKE_PROFIT_PCT = 0.0200  # 2.00% take profit
    TRANSACTION_COST_PCT = 0.0002  # 0.02% spread + commission

    # Model Parameters
    TRAIN_WINDOW_SIZE = 378  # Days of historical data for training
    PERCENTILE_LOWER = 48  # Lower percentile threshold for short signals
    PERCENTILE_UPPER = 52  # Upper percentile threshold for long signals
    PREDICTION_BUFFER_SIZE = 200  # Days to use for percentile calculation

    # XGBoost Hyperparameters
    XGBOOST_PARAMS = {
        'n_estimators': 125,
        'learning_rate': 0.1,
        'max_depth': 5,
        'gamma': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    # Technical Indicators
    TECHNICAL_FEATURES = [
        'momentum', 'avg_price', 'range', 'ohlc',
        'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
        'macd', 'macd_signal', 'macd_hist',
        'adx', 'plus_di', 'minus_di',
        'rsi', 'stoch_k', 'stoch_d', 'cci', 'williams_r',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr'
    ]

    # Position Management
    HOLDING_PERIOD_DAYS = 1  # Hold positions for 1 day

    # Multi-Pair Strategy Parameters
    DEFAULT_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']

    # 10-Asset Diversified Portfolio (4 Forex + 6 Commodities/Indices)
    DEFAULT_10_ASSETS = [
        'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',  # Forex (61% annual)
        'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',  # Metals (115-355% annual)
        'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'  # Commodities/Indices (70-220% annual)
    ]

    LEVERAGE = 2.0  # 2:1 leverage for multi-asset strategy
    CAPITAL_PER_PAIR_PCT = 0.10  # 10% capital allocation per asset (10 assets)

    # File Paths
    STATE_FILE_TEMPLATE = 'position_state_{pair}.pkl'
    TRADE_LOG_TEMPLATE = 'trades_{pair}.csv'
    MODEL_CACHE_TEMPLATE = 'model_cache_{pair}.pkl'

    @classmethod
    def get_state_file(cls, pair):
        """Get state file path for a pair"""
        return cls.STATE_FILE_TEMPLATE.format(pair=pair)

    @classmethod
    def get_trade_log(cls, pair):
        """Get trade log path for a pair"""
        return cls.TRADE_LOG_TEMPLATE.format(pair=pair)

    @classmethod
    def get_model_cache(cls, pair):
        """Get model cache path for a pair"""
        return cls.MODEL_CACHE_TEMPLATE.format(pair=pair)
