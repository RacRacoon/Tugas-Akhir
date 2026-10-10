# TA_Rakan — Driving Style Classification (AI Real-Time)

A real-time driving style classifier that listens to live UDP telemetry from racing/driving games,
feeds it into a trained hybrid CNN-LSTM model, and outputs whether the current driving behavior
is **NORMAL** or **RACING**.

---

## Project Structure

```
TA_Rakan/
│
├── Telemetry_Extract/          # Step 1 — Record raw telemetry from a game to CSV
│   ├── telext_basic.py         # Basic recorder (raw fields only)
│   └── telext_advanced.py      # Advanced recorder (+ accel & jerk computed)
│
├── training/                   # Step 2 & 3 — Prepare dataset and train the model
│   ├── Normal.csv              # Raw telemetry recorded while driving normally
│   ├── Racing.csv              # Raw telemetry recorded while driving aggressively
│   ├── Dataset.py              # Merges + cleans the two CSVs → dataset_labeled.csv
│   ├── dataset_labeled.csv     # Final labeled dataset used for training
│   └── training.py             # Trains the CNN-LSTM model
│
├── Models/                     # Step 3 output — saved model artifacts
│   ├── driver_style_model.h5   # Trained Keras model
│   ├── scaler.bin              # Fitted StandardScaler (joblib)
│   └── label_encoder.bin       # LabelEncoder (joblib)
│
├── Driving_Style_Classification/  # Step 4 — Real-time inference per game
│   ├── telemetry.py            # Classifier for BeamNG.drive (main/best version)
│   ├── teleForza.py            # Classifier for Forza Horizon 4
│   └── tele_f1.py              # Classifier for F1 2021
│
└── venv_ai/                    # Python virtual environment (TensorFlow, etc.)
```

---

## Pipeline Overview

```
Game (UDP) ──► Telemetry_Extract ──► training/Normal.csv
                                  └► training/Racing.csv
                                          │
                                     Dataset.py
                                          │
                                   dataset_labeled.csv
                                          │
                                     training.py
                                          │
                              Models/ (model + scaler)
                                          │
                         Driving_Style_Classification/
                          (telemetry.py / teleForza.py / tele_f1.py)
                                          │
                              Console output: NORMAL / RACING
```

---

## Step-by-Step Guide

### Step 1 — Record Training Data

Run one of the scripts in `Telemetry_Extract/` **while inside the game** to capture telemetry over UDP.
The game must have its UDP telemetry output enabled (BeamNG OutGauge on port 4444).

| Script | Description |
|---|---|
| `telext_basic.py` | Records raw fields: speed, rpm, gear, throttle, brake, clutch. Good for a quick first recording. |
| `telext_advanced.py` | Records all basic fields **plus** computed `accel_ms2` and `jerk`. This output matches the format expected by `Dataset.py` and the model. **Prefer this one.** |

Both scripts save to a CSV file (`driving_data_train.csv` or `Extracted.csv`) and stop on `Ctrl+C`.

Record **two separate sessions** and save them as:
- `training/Normal.csv` — calm, everyday driving
- `training/Racing.csv` — aggressive, high-speed driving

**Note:** The data in both CSVs was recorded from BeamNG.drive.

---

### Step 2 — Build the Labeled Dataset

```bash
python training/Dataset.py
```

This script:
1. Loads `training/Normal.csv` and `training/Racing.csv`
2. Applies noise filters to remove ambiguous data:
   - **Normal** data: keeps rows where `rpm ≤ 3500`, `throttle ≤ 65%`, `speed ≤ 110 km/h`
   - **Racing** data: removes rows where `speed < 15 km/h` (stationary car looks the same in both styles)
3. Adds a `label` column (`normal` / `racing`)
4. Merges both into `training/dataset_labeled.csv`

Output columns: `timestamp, car, speed_kmh, rpm, gear, throttle, brake, accel_ms2, jerk, label`

---

### Step 3 — Train the Model

```bash
# Activate the virtual environment first
venv_ai\Scripts\activate

python training/training.py
```

**What it does:**
- Loads `dataset_labeled.csv`
- Uses only 5 features for training: `speed_kmh, rpm, gear, throttle, brake`
  - *(accel and jerk are in the CSV but excluded in this version of training.py)*
- Applies a **sliding window** of 30 time steps, stepping by 10 each time
- Fits a `StandardScaler` on the training split only (no data leakage)
- Trains a **Hybrid CNN-LSTM** model:
  - `Conv1D(64)` → detects instantaneous input patterns
  - `LSTM(64)` → captures temporal/sequential context
  - `Dense(1, sigmoid)` → binary output (0 = Normal, 1 = Racing)
- Saves outputs to `Models/`:
  - `driver_style_model.h5`
  - `scaler.bin`
  - `label_encoder.bin`
- Prints a confusion matrix and classification report

**Key hyperparameters:**
| Parameter | Value |
|---|---|
| Window size | 30 frames |
| Step size | 10 frames |
| Epochs | 50 (with early stopping, patience=12) |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Class weights | Normal: 1.2, Racing: 1.0 |

---

### Step 4 — Run Real-Time Classification

Pick the script that matches your game:

#### BeamNG.drive — `Driving_Style_Classification/telemetry.py`

> **This is the most complete and robust version.**

- UDP port: **4444** (BeamNG OutGauge format)
- Features: `speed, rpm, gear, throttle, brake, accel, jerk` (7 features — computed live)
- Smoothing: 15-prediction moving average
- Hysteresis thresholds:
  - Switch to RACING if smoothed score > **0.85**
  - Switch back to NORMAL if smoothed score < **0.50**
- Output: color-coded bar + style label in terminal, updates at ~10 Hz

```bash
python Driving_Style_Classification/telemetry.py
```

#### Forza Horizon 4 — `Driving_Style_Classification/teleForza.py`

- UDP port: **5555** (FH4 "Data Out" format, 311/324-byte packets)
- Features: `speed, rpm, gear, throttle, brake, 0, 0` (accel/jerk hardcoded to 0 — model still expects 7 features)
- Smoothing: 15-prediction moving average
- Hysteresis thresholds: RACING > 0.7, NORMAL < 0.3
- Setup: In FH4 → Settings → HUD and Gameplay → Data Out: ON, IP: 127.0.0.1, Port: 5555

```bash
python Driving_Style_Classification/teleForza.py
```

#### F1 2021 — `Driving_Style_Classification/tele_f1.py`

- UDP port: **4444** (F1 2021 UDP telemetry format, Packet ID 6)
- Features: `speed, rpm, gear, throttle, brake, 0, 0` (no accel/jerk computed)
- No smoothing or hysteresis — uses a simple 0.5 threshold
- Note: This version loads `driver_style_model.h5` and `scaler.bin` from the **current working directory**, not from `Models/`

```bash
python Driving_Style_Classification/tele_f1.py
```

---

## Where You Left Off

The full pipeline appears complete:

- [x] Telemetry extraction scripts work for BeamNG (both basic and advanced)
- [x] Dataset has been recorded and cleaned (`dataset_labeled.csv` exists)
- [x] Model has been trained and saved (`Models/driver_style_model.h5` exists)
- [x] Real-time classifiers exist for all three target games

**Likely next steps / open issues to investigate:**

1. **Feature mismatch between training and inference**
   - `training.py` trains on **5 features** (`speed, rpm, gear, throttle, brake`)
   - `telemetry.py` (BeamNG) feeds **7 features** (adds `accel` and `jerk`) into the scaler
   - `teleForza.py` and `tele_f1.py` also use 7 features but with `accel=0, jerk=0`
   - This means the scaler and model shape may not match unless you retrain with 7 features. **Verify this.**

2. **tele_f1.py path issue**
   - It looks for `driver_style_model.h5` and `scaler.bin` in the working directory, not `Models/`. If you run it from the project root it will fail. Fix the paths to `Models/driver_style_model.h5` and `Models/scaler.bin`.

3. **Racing.csv starts with idle data**
   - The first rows of `Racing.csv` show `speed = 0.0` and `rpm = 744` — the car was stationary when recording started. `Dataset.py` already filters these out (`speed >= 15`), so this is handled.

4. **Potential improvement: add accel/jerk to training**
   - Currently `training.py` ignores `accel_ms2` and `jerk` even though `telext_advanced.py` records them and `telemetry.py` computes them live. Retraining with 7 features would make the pipeline fully consistent.

---

## Dependencies

All packages are installed in `venv_ai/`. Key ones:

| Package | Purpose |
|---|---|
| `tensorflow` | CNN-LSTM model training and inference |
| `numpy` | Array operations, windowing |
| `pandas` | CSV loading and dataset manipulation |
| `scikit-learn` | StandardScaler, LabelEncoder, train/test split, metrics |
| `joblib` | Saving/loading scaler and encoder |
| `matplotlib` / `seaborn` | Confusion matrix plot during training |

Activate the environment before running any script:

```bash
venv_ai\Scripts\activate
```
