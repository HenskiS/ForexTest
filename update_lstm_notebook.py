import json

# Read the notebook
with open('forex_lstm_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 1 (the imports cell) to add KERAS_BACKEND setting at the start
new_cell_1_source = """# Configure Keras to use PyTorch backend
import os
os.environ['KERAS_BACKEND'] = 'torch'

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML & Deep Learning
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

print(f"Keras version: {keras.__version__}")
print(f"Backend: {keras.backend.backend()}")
print("Libraries imported successfully!")"""

# Update the cell source
nb['cells'][1]['source'] = new_cell_1_source

# Clear outputs from cell 1 since we're changing it
nb['cells'][1]['outputs'] = []
nb['cells'][1]['execution_count'] = None

# Save the notebook
with open('forex_lstm_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated forex_lstm_model.ipynb - Added KERAS_BACKEND='torch' at the start")
print(f"Total cells: {len(nb['cells'])}")
