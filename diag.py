import pandas as pd
import numpy as np
import os

base = r'c:\Users\User\Desktop\TA_Rakan\training'

def analyze(name, path):
    df = pd.read_csv(path)
    print(f"\n{'='*50}")
    print(f"FILE: {name}  ({len(df)} rows)")
    print(f"{'='*50}")
    print(f"Columns: {df.columns.tolist()}")
    if 'label' in df.columns:
        print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    numeric = df.select_dtypes(include='number')
    print(f"\nNumeric stats:")
    print(numeric.describe().to_string())
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")

analyze('Normal.csv',           os.path.join(base, 'Normal.csv'))
analyze('Racing.csv',           os.path.join(base, 'Racing.csv'))
analyze('dataset_labeled.csv',  os.path.join(base, 'dataset_labeled.csv'))

# Check sliding window output
df = pd.read_csv(os.path.join(base, 'dataset_labeled.csv'))
FEATURES = ['speed_kmh', 'rpm', 'gear', 'throttle', 'brake', 'accel_ms2', 'jerk']
WINDOW_SIZE = 30
STEP_SIZE = 10
data_values = df[FEATURES].values
labels = df['label'].values

# Simulate windowing
X, y = [], []
for i in range(0, len(data_values) - WINDOW_SIZE, STEP_SIZE):
    X.append(data_values[i : i + WINDOW_SIZE])
    y.append(labels[i + WINDOW_SIZE])

X = np.array(X)
y = np.array(y)
print(f"\n{'='*50}")
print(f"WINDOWING RESULT (window=30, step=10)")
print(f"{'='*50}")
print(f"Total samples after windowing: {len(X)}")
if len(y) > 0:
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")
    
    # Check for label contamination (window spans both classes)
    df2 = pd.read_csv(os.path.join(base, 'dataset_labeled.csv'))
    df2['label_enc'] = (df2['label'] == 'racing').astype(int)
    label_vals = df2['label_enc'].values
    contaminated = 0
    for i in range(0, len(label_vals) - WINDOW_SIZE, STEP_SIZE):
        window_labels = label_vals[i : i + WINDOW_SIZE]
        if len(np.unique(window_labels)) > 1:
            contaminated += 1
    print(f"\nWindows with MIXED labels (contaminated): {contaminated} / {len(X)}")
