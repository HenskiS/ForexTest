# LSTM Forex Trading Study

## Goal
Recreate the LSTM-based forex trading strategy from *"Predictive modeling of foreign exchange trading signals using machine learning techniques"* (Enkhbayar & Ślepaczuk, Expert Systems With Applications, 2025).

## Progress

### ✅ Step 1: Data Loading & Exploration (Complete)
- Loaded EURUSD daily data (6,743 observations, Jan 2000 - Nov 2025)
- Matches study's full timeframe (2000-2023) + additional recent data
- Calculated simple and log returns
- Compared statistics with study's Table 1
- Confirmed non-normal distribution (JB test)

### ✅ Step 2: Feature Engineering (Complete)
Implemented all 26 indicators from Table 3 with default settings:

**Statistical Indicators (4)**
- Momentum, Average Price, Range, OHLC

**Momentum Indicators (11)**
- EMA: 10, 20, 50, 100, 200
- MACD: line, signal, histogram (26, 12, 9)
- ADX: adx, +DI, -DI (14)

**Oscillator Indicators (5)**
- RSI (14)
- Stochastic: %K, %D (14, 3)
- CCI (20, 0.015)
- Williams %R (14)

**Volatility Indicators (6)**
- Bollinger Bands: upper, middle, lower, width, position (20, 2)
- ATR (14)

### ✅ Step 3: Target Variable Preparation (Complete)
- Target variables (simple_return, log_return) already in CSV from Step 1

### ✅ Step 4: Walk-Forward Optimization (Complete)
- 40 windows, 6-month rolling (126 days)
- Train: 600 days | Validation: 156 days | Test: 126 days
- Sequence length: 20 days lookback

### ✅ Step 5: LSTM Model Training (Ready to Run)
- Hyperparameter search: 20 iterations per window using RandomizedSearchCV
- Search space: layers [1-3], neurons [5-40], learning rate [0.001-0.05], batch [32/64/128], epochs [10/20/30]
- Loss: MAE, Activation: tanh, Dropout: 0.2
- **Note**: Full training (40 windows) will take several hours

### 🔄 Next Steps
- Step 6: Signal generation
- Step 7: Backtesting with transaction costs
- Step 8: Performance evaluation

## Key Details
- **Data**: EURUSD daily, 6,743 observations (2000-2025) — exceeds study's 6,171 obs
- **Framework**: Keras 3.0 + PyTorch backend
- **Notebooks**:
  - [forex_data_prep.ipynb](forex_data_prep.ipynb) — Steps 1-2: Data loading & feature engineering
  - [forex_lstm_model.ipynb](forex_lstm_model.ipynb) — Steps 3-8: Model training & evaluation
- **Output**: data/EURUSD_1day_with_features.csv
