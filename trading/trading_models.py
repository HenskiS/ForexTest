"""
Trading Models

XGBoost and ANN model training, prediction, and signal generation.
"""
import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from .config import TradingConfig
from .market_utils import calculate_technical_features


class TradingModel:
    """Trading model with support for XGBoost and ANN"""

    def __init__(self, pair, model_type='xgboost'):
        """
        Initialize trading model for a pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
            model_type: 'xgboost' or 'ann' (default: 'xgboost')
        """
        self.pair = pair.upper()
        self.model_type = model_type.lower()

        if self.model_type not in ['xgboost', 'ann']:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'xgboost' or 'ann'")

        # Model and scaler
        self.model = None
        self.scaler = None

        # Prediction buffer for percentile thresholds
        self.prediction_buffer = []
        self.buffer_file = f'data/oanda_cache/prediction_buffer_{self.model_type}_{self.pair}.pkl'

        # Load hyperparameters
        self.best_params = self.load_hyperparameters()

        # Load prediction buffer
        self.load_prediction_buffer()

        print(f"Initialized {self.model_type.upper()} model for {self.pair}")

    def load_hyperparameters(self):
        """Load optimized hyperparameters or use defaults"""
        if self.model_type == 'xgboost':
            hyperparam_file = f'hyperparams_rolling_daily_{self.pair}.pkl'
            if os.path.exists(hyperparam_file):
                with open(hyperparam_file, 'rb') as f:
                    params = pickle.load(f)
                print(f"Loaded XGBoost hyperparameters from {hyperparam_file}")
                return params
            else:
                print(f"No XGBoost hyperparameters found, using defaults")
                return TradingConfig.XGBOOST_PARAMS.copy()
        else:  # ann
            # Use optimized ANN params from config
            print(f"Using optimized ANN hyperparameters from config")
            return TradingConfig.ANN_PARAMS.copy()

    def load_prediction_buffer(self):
        """Load prediction buffer for percentile calculations"""
        if os.path.exists(self.buffer_file):
            with open(self.buffer_file, 'rb') as f:
                loaded = pickle.load(f)
            # Ensure it's a list (may have been saved as numpy array)
            if isinstance(loaded, np.ndarray):
                self.prediction_buffer = loaded.tolist()
            else:
                self.prediction_buffer = list(loaded)
            print(f"Loaded prediction buffer: {len(self.prediction_buffer)} predictions")
        else:
            self.prediction_buffer = []
            print("No prediction buffer found. Run initialize_oanda_buffer.py first.")

    def save_prediction_buffer(self):
        """Save prediction buffer to disk"""
        os.makedirs(os.path.dirname(self.buffer_file), exist_ok=True)
        with open(self.buffer_file, 'wb') as f:
            pickle.dump(self.prediction_buffer, f)

    def add_prediction_to_buffer(self, prediction):
        """
        Add prediction to buffer for percentile calculation.

        Args:
            prediction: Prediction value to add
        """
        self.prediction_buffer.append(prediction)

        # Keep last N predictions (e.g., 200 days)
        if len(self.prediction_buffer) > TradingConfig.PREDICTION_BUFFER_SIZE:
            self.prediction_buffer = self.prediction_buffer[-TradingConfig.PREDICTION_BUFFER_SIZE:]

        self.save_prediction_buffer()

    def train(self, df):
        """
        Train model on last N days with 1-day gap to prevent look-ahead bias.

        Args:
            df: DataFrame with OHLC data

        Returns:
            tuple: (trained model, scaler, clean dataframe with features)
        """
        if self.model_type == 'xgboost':
            return self.train_xgboost(df)
        else:  # ann
            return self.train_ann(df)

    def train_xgboost(self, df):
        """
        Train XGBoost model on last N days with 1-day gap to prevent look-ahead bias.

        Args:
            df: DataFrame with OHLC data

        Returns:
            tuple: (trained model, scaler, clean dataframe with features)
        """
        print(f"\n{'='*70}")
        print(f"Training XGBoost model on last {TradingConfig.TRAIN_WINDOW_SIZE} days...")
        print(f"{'='*70}")

        # Calculate features
        df_with_features = calculate_technical_features(df.copy())

        # Calculate target (1-day forward return)
        df_with_features['target_1day_return'] = df_with_features['close'].pct_change(1).shift(-1)

        # Drop rows with invalid features
        df_clean = df_with_features.dropna(subset=TradingConfig.TECHNICAL_FEATURES)

        # For training, only use rows with valid targets
        df_with_target = df_clean[df_clean['target_1day_return'].notna()].copy()

        if len(df_with_target) < TradingConfig.TRAIN_WINDOW_SIZE:
            raise ValueError(f"Need {TradingConfig.TRAIN_WINDOW_SIZE} days with targets, have {len(df_with_target)}")

        # Train on last N days that have targets (excludes today to prevent look-ahead)
        train_data = df_with_target.iloc[-TradingConfig.TRAIN_WINDOW_SIZE:].copy()

        print(f"Training data: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"(1-day gap: training excludes today to prevent look-ahead bias)")

        # Prepare training data
        X_train = train_data[TradingConfig.TECHNICAL_FEATURES].values
        y_train = train_data['target_1day_return'].values

        # Scale features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(X_train_scaled, y_train, verbose=False)

        print(f"XGBoost model trained successfully")

        return self.model, self.scaler, df_clean

    def train_ann(self, df):
        """
        Train ANN model on last N days with 1-day gap to prevent look-ahead bias.

        Args:
            df: DataFrame with OHLC data

        Returns:
            tuple: (trained model, scaler, clean dataframe with features)
        """
        print(f"\n{'='*70}")
        print(f"Training ANN model on last {TradingConfig.TRAIN_WINDOW_SIZE} days...")
        print(f"{'='*70}")

        # Calculate features
        df_with_features = calculate_technical_features(df.copy())

        # Calculate target (1-day forward return)
        df_with_features['target_1day_return'] = df_with_features['close'].pct_change(1).shift(-1)

        # Drop rows with invalid features
        df_clean = df_with_features.dropna(subset=TradingConfig.TECHNICAL_FEATURES)

        # For training, only use rows with valid targets
        df_with_target = df_clean[df_clean['target_1day_return'].notna()].copy()

        if len(df_with_target) < TradingConfig.TRAIN_WINDOW_SIZE:
            raise ValueError(f"Need {TradingConfig.TRAIN_WINDOW_SIZE} days with targets, have {len(df_with_target)}")

        # Train on last N days that have targets (excludes today to prevent look-ahead)
        train_data = df_with_target.iloc[-TradingConfig.TRAIN_WINDOW_SIZE:].copy()

        print(f"Training data: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"(1-day gap: training excludes today to prevent look-ahead bias)")

        # Prepare training data
        X_train = train_data[TradingConfig.TECHNICAL_FEATURES].values
        y_train = train_data['target_1day_return'].values

        # Scale features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = MLPRegressor(**self.best_params)
        self.model.fit(X_train_scaled, y_train)

        print(f"ANN model trained successfully")
        print(f"  Architecture: {self.best_params['hidden_layer_sizes']}")
        print(f"  Solver: {self.best_params['solver'].upper()}")
        print(f"  Learning rate: {self.best_params['learning_rate_init']}")

        return self.model, self.scaler, df_clean

    def predict(self, df_clean):
        """
        Generate prediction for today (last row in dataframe).

        Args:
            df_clean: DataFrame with features (output from train())

        Returns:
            float: Prediction value
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Get today's features (last row)
        X_today = df_clean.iloc[[-1]][TradingConfig.TECHNICAL_FEATURES].values
        X_today_scaled = self.scaler.transform(X_today)

        # Predict
        prediction = self.model.predict(X_today_scaled)[0]

        print(f"\nPrediction for today: {prediction:.6f}")

        return prediction

    def generate_signal(self, prediction, lower_percentile=None, upper_percentile=None):
        """
        Generate trading signal based on prediction and percentile thresholds.

        Args:
            prediction: Model prediction
            lower_percentile: Lower percentile threshold (default from config based on model type)
            upper_percentile: Upper percentile threshold (default from config based on model type)

        Returns:
            int: 1 for long, -1 for short, 0 for no signal
        """
        if len(self.prediction_buffer) < 50:
            print(f"WARNING: Prediction buffer too small ({len(self.prediction_buffer)} < 50)")
            print("Need more historical predictions for reliable signal generation")
            return 0

        # Use model-specific defaults if not specified
        if lower_percentile is None:
            if self.model_type == 'ann':
                lower_percentile = TradingConfig.ANN_PERCENTILE_LOWER
            else:
                lower_percentile = TradingConfig.PERCENTILE_LOWER
        if upper_percentile is None:
            if self.model_type == 'ann':
                upper_percentile = TradingConfig.ANN_PERCENTILE_UPPER
            else:
                upper_percentile = TradingConfig.PERCENTILE_UPPER

        # Calculate percentile thresholds
        # For 50/50 median split, both use the same median value
        lower_threshold = np.percentile(self.prediction_buffer, lower_percentile)
        upper_threshold = np.percentile(self.prediction_buffer, upper_percentile)

        print(f"\nSignal Generation ({self.model_type.upper()}):")
        print(f"  Prediction: {prediction:.6f}")
        if lower_percentile == upper_percentile:
            # Median split (50/50)
            print(f"  Threshold (median): {lower_threshold:.6f}")
        else:
            print(f"  Lower threshold ({lower_percentile}th percentile): {lower_threshold:.6f}")
            print(f"  Upper threshold ({upper_percentile}th percentile): {upper_threshold:.6f}")

        # Generate signal
        # For median split (50/50): SHORT if <= median, LONG if >= median
        if prediction >= upper_threshold:
            signal = 1  # Long
            print(f"  Signal: LONG (prediction >= threshold)")
        elif prediction <= lower_threshold:
            signal = -1  # Short
            print(f"  Signal: SHORT (prediction <= threshold)")
        else:
            signal = 0  # No signal
            print(f"  Signal: NONE (prediction between thresholds)")

        # Add prediction to buffer
        self.add_prediction_to_buffer(prediction)

        return signal
