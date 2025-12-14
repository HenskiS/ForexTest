"""
Trading Configuration

Centralized configuration for trading parameters, model hyperparameters,
and file paths.
"""


class TradingConfig:
    """Trading configuration and parameters"""

    # Trading Parameters
    STOP_LOSS_PCT = 0.02  # 2% stop loss (Sleep Well config)
    TAKE_PROFIT_PCT = 0.0300  # 3.00% take profit (for XGBoost/commodities)
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

    # ANN Hyperparameters (Optimized via hyperparameter search)
    ANN_PARAMS = {
        'hidden_layer_sizes': (13, 20, 31),  # 3 hidden layers
        'activation': 'tanh',
        'solver': 'sgd',
        'learning_rate_init': 0.001,
        'momentum': 0.4,
        'batch_size': 64,
        'max_iter': 20,
        'alpha': 0.0001,  # L2 regularization
        'learning_rate': 'adaptive',
        'random_state': 42,
        'verbose': False,
        'warm_start': False,
        'early_stopping': False  # Keep training for all 20 epochs
    }

    # Model-specific parameters (Sleep Well config: 10/90, 5-day, 2% SL, no TP, 1.5x)
    ANN_TAKE_PROFIT_PCT = None  # No take profit for ANN (time-based exit only)
    ANN_PERCENTILE_LOWER = 10  # 10th percentile for short signals
    ANN_PERCENTILE_UPPER = 90  # 90th percentile for long signals

    # Technical Indicators (31 features - optimized set)
    # Order must match backtest's calculate_features() column order
    TECHNICAL_FEATURES = [
        'momentum', 'avg_price', 'range', 'ohlc',
        'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
        'macd', 'macd_signal', 'macd_hist',
        'adx', 'plus_di', 'minus_di',
        'rsi', 'stoch_k', 'stoch_d',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',  # Order matches backtest
        'atr', 'volume_sma',
        'close_to_high', 'close_to_low',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10'
    ]

    # Position Management
    HOLDING_PERIOD_DAYS = 5  # Hold positions for 5 days (Sleep Well config)

    # Multi-Pair Strategy Parameters (8-pair Sleep Well portfolio)
    DEFAULT_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'NZDUSD']

    # 10-Asset Diversified Portfolio (4 Forex + 6 Commodities/Indices)
    DEFAULT_10_ASSETS = [
        'EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY',  # Forex (61% annual)
        'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XCUUSD',  # Metals (115-355% annual)
        'SUGARUSD', 'SPX500USD', 'DE30EUR', 'WTICOUSD', 'BCOUSD'  # Commodities/Indices (70-220% annual)
    ]

    LEVERAGE = 1.5  # 1.5:1 leverage (Sleep Well config)
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
