# RunPod GPU Training Setup

## 1. On RunPod

### Start a Pod
1. Go to runpod.io
2. Select "Pods" → "GPU Pods"
3. Choose template: **PyTorch 2.0+** or **Jupyter Notebook**
4. Select GPU: RTX 3090 or A4000 (cheapest ~$0.40/hr)
5. Start pod

### Access Jupyter
1. Click "Connect" → "Jupyter Lab"
2. Opens Jupyter interface in browser

## 2. Upload Files

Upload these files to RunPod Jupyter:
- `forex_lstm_model.ipynb`
- `data/EURUSD_1day_with_features.csv`

## 3. Install Dependencies

In a Jupyter notebook cell, run:
```python
!pip install pandas numpy matplotlib seaborn scipy scikit-learn keras torch
```

## 4. Modify Training Cell

In cell 9 of `forex_lstm_model.ipynb`, change:
```python
n_windows = 3  # Quick test: 3 windows
```
to:
```python
n_windows = len(windows)  # Full run: 40 windows
```

## 5. Run Training

- Run all cells sequentially (Cells → Run All)
- Training should take ~1.5-2 hours on GPU
- Monitor progress in the output

## 6. Download Results

After training completes, download these files:
- `lstm_results.pkl`
- `lstm_results_summary.json`
- `lstm_results_{strategy}.pkl` (for each strategy)
- `performance_summary.json`
- `strategy_comparison.csv`

## 7. Continue Locally

Back on your local machine:
- Place downloaded files in your project folder
- Run cells 11-24 to generate signals, backtest, and evaluate
- (These cells are fast and don't need GPU)

## Tips

- **Cost**: ~$0.60-0.80 for full 40-window training
- **Monitor**: Check pod every 30 minutes to avoid overpaying
- **Save often**: Download results as soon as training completes
- **Stop pod**: Terminate immediately when done to stop billing
