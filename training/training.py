import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from Dataset import FEATURES  # satu sumber kebenaran daftar fitur dgn Dataset.py

# ============================================================
# KONFIGURASI
# ============================================================
WINDOW_SIZE = 30    # Jumlah time-step per sampel (10 Hz -> 3 detik)
STEP_SIZE   = 5     # Step lebih kecil = lebih banyak sampel dari data yang sama
EPOCHS      = 100
BATCH_SIZE  = 32
RANDOM_SEED = 42

# Pembagian data dilakukan per BLOK baris berurutan, bukan per window.
# Window yang overlap tidak boleh terpisah antara train dan test (kebocoran data).
BLOCK_ROWS = 200    # ~20 detik pada 10 Hz
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# FEATURES diimpor dari Dataset.py (lihat import di atas). Urutan ini juga
# harus dipakai saat inferensi (telemetry.py membaca urutan ini dari
# features.json, bukan hardcode, jadi otomatis ikut kalau FEATURES berubah).

# Batas clipping outlier — per kolom, karena satuan/skalanya beda-beda.
# accel_ms2/accel_lat: m/s^2, tidak realistis >15 untuk mobil jalan.
# steering_speed/steering_accel: turunan 1x & 2x dari steering (biasanya -1..1),
# jadi skalanya BUKAN m/s^2 — ini PLACEHOLDER, lihat catatan di README setelah
# rekam data asli: distribusi steering_accel jauh lebih liar daripada
# steering_speed (turunan ganda memperkuat noise, sama seperti kasus `jerk`
# yang dulu dihapus) — sesuaikan setelah lihat histogram data nyata.
CLIP_SPECS = {
    'accel_ms2':      15.0,
    'accel_lat':       15.0,
    'steering_speed':  5.0,
    'steering_accel': 20.0,
}
CLIP_COLS = list(CLIP_SPECS.keys())

# Path output default
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Models')


# ============================================================
# 1. BLOK -> SPLIT -> WINDOW (tanpa kebocoran, tanpa lintas segmen)
# ============================================================
def make_blocks(df: pd.DataFrame, le: LabelEncoder) -> list:
    """
    Memotong tiap segmen (baris berurutan murni satu kelas, lihat Dataset.py)
    menjadi blok ~BLOCK_ROWS baris. Window nanti hanya dibuat DI DALAM satu blok,
    jadi tidak ada window yang menyambung dua segmen atau dua split.
    """
    missing = [c for c in FEATURES + ['label', 'segment'] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}. Jalankan Dataset.py dengan data skema v2.")

    blocks = []
    for (label, _seg), grp in df.groupby(['label', 'segment'], sort=False):
        grp = grp.copy()
        for col, limit in CLIP_SPECS.items():
            grp[col] = grp[col].clip(-limit, limit)
        values = grp[FEATURES].values.astype(np.float32)
        if len(values) < WINDOW_SIZE:
            continue

        n_blocks = max(1, len(values) // BLOCK_ROWS)
        y = int(le.transform([label])[0])
        for chunk in np.array_split(values, n_blocks):
            blocks.append({'y': y, 'values': chunk})
    return blocks


def split_blocks(blocks: list, rng: np.random.Generator) -> dict:
    """Bagi blok ke train/val/test, terpisah per kelas agar rasio kelas terjaga."""
    split = {'train': [], 'val': [], 'test': []}
    for cls in sorted({b['y'] for b in blocks}):
        idx = rng.permutation([i for i, b in enumerate(blocks) if b['y'] == cls])
        n = len(idx)
        if n < 3:
            raise ValueError(
                f"Kelas {cls} hanya punya {n} blok; butuh minimal 3 untuk train/val/test. "
                "Rekam data lebih banyak."
            )
        n_test = max(1, round(n * TEST_FRAC))
        n_val  = max(1, round(n * VAL_FRAC))
        split['test'] += [blocks[i] for i in idx[:n_test]]
        split['val']  += [blocks[i] for i in idx[n_test:n_test + n_val]]
        split['train'] += [blocks[i] for i in idx[n_test + n_val:]]
    return split


def window_blocks(blocks: list) -> tuple:
    X, y = [], []
    for b in blocks:
        v = b['values']
        for i in range(0, len(v) - WINDOW_SIZE + 1, STEP_SIZE):
            X.append(v[i:i + WINDOW_SIZE])
            y.append(b['y'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(file_path: str) -> tuple:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    le = LabelEncoder()
    le.fit(df['label'].unique())
    if len(le.classes_) != 2:
        raise ValueError(f"Butuh tepat 2 label, ditemukan: {list(le.classes_)}")
    print(f"   Label ditemukan: {list(le.classes_)}")

    blocks = make_blocks(df, le)
    split  = split_blocks(blocks, np.random.default_rng(RANDOM_SEED))

    data = {}
    for name in ('train', 'val', 'test'):
        X, y = window_blocks(split[name])
        if len(X) == 0 or len(np.unique(y)) < 2:
            raise ValueError(f"Split '{name}' kosong atau hanya satu kelas. Tambah data.")
        if not np.isfinite(X).all():
            raise ValueError(f"Split '{name}' berisi NaN/inf.")
        data[name] = (X, y)
        counts = {c: int((y == i).sum()) for i, c in enumerate(le.classes_)}
        print(f"   [{name:<5}] {len(split[name]):>3} blok -> {len(X):>5} window {counts}")
    return data, le


# ============================================================
# 2. ARSITEKTUR MODEL — CNN + BiLSTM
#    Ukuran disesuaikan dengan ukuran dataset (~1000-2000 sampel)
# ============================================================
def build_model(win_len: int, feat_len: int) -> tf.keras.Model:
    """
    Arsitektur sederhana tapi efektif untuk dataset ~1000-2000 sampel:
      Conv1D  -> deteksi pola kontrol lokal
      BiLSTM  -> konteks temporal maju-mundur
      LSTM    -> representasi sekuensial final
      Dense   -> klasifikasi biner
    Ukuran layer sengaja dikecilkan untuk menghindari overfit pada dataset kecil.
    """
    inp = layers.Input(shape=(win_len, feat_len), name='input')

    # --- Blok CNN ---
    x = layers.Conv1D(
        filters=32, kernel_size=3, padding='causal',
        activation='linear', name='conv1'
    )(inp)
    x = layers.PReLU(shared_axes=[1], name='prelu_conv')(x)
    x = layers.BatchNormalization(name='bn_conv')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool')(x)
    x = layers.Dropout(0.3, name='drop_conv')(x)

    # --- Blok BiLSTM ---
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name='bilstm'
    )(x)
    x = layers.LayerNormalization(name='ln_bilstm')(x)

    # --- Blok LSTM kedua ---
    x = layers.LSTM(16, return_sequences=False, dropout=0.2, name='lstm2')(x)
    x = layers.LayerNormalization(name='ln_lstm2')(x)

    # --- Blok Dense ---
    x = layers.Dense(16, activation='linear', name='dense1')(x)
    x = layers.PReLU(name='prelu_dense')(x)
    x = layers.Dropout(0.3, name='drop_dense')(x)

    out = layers.Dense(1, activation='sigmoid', name='output')(x)

    mdl = models.Model(inp, out, name='DrivingStyleClassifier')
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return mdl


# ============================================================
# 3. MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.join(os.path.dirname(__file__), 'dataset_labeled.csv'))
    ap.add_argument('--out-dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()

    out_dir      = args.out_dir
    model_path   = os.path.join(out_dir, 'driver_style_model.h5')
    scaler_path  = os.path.join(out_dir, 'scaler.bin')
    encoder_path = os.path.join(out_dir, 'label_encoder.bin')
    config_path  = os.path.join(out_dir, 'features.json')

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} tidak ditemukan!")
        return

    os.makedirs(out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # --- Load, blok, split, window ---
    print("Memuat data (blok -> split -> window, tanpa kebocoran)...")
    data, le = prepare_data(args.dataset)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data['train'], data['val'], data['test']
    print(f"   Shape X train  : {X_train.shape}")

    if len(X_train) < 100:
        print("\nWARNING: Dataset terlalu kecil! Rekam lebih banyak data.")
        return

    # --- Scaling (fit hanya pada train) ---
    feat = X_train.shape[2]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(X_train.shape)
    X_val_s   = scaler.transform(X_val.reshape(-1, feat)).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape)

    joblib.dump(scaler, scaler_path)
    joblib.dump(le,     encoder_path)
    with open(config_path, 'w') as f:
        json.dump({'features': FEATURES, 'window_size': WINDOW_SIZE,
                   'clip_specs': CLIP_SPECS}, f, indent=2)
    print(f"\nScaler, encoder & features.json disimpan ke {out_dir}")

    # --- Bangun model ---
    model = build_model(WINDOW_SIZE, feat)
    model.summary()

    # --- Class weights (dari train saja) ---
    n_cls = [int((y_train == i).sum()) for i in range(len(le.classes_))]
    total = sum(n_cls)
    class_weights = {i: total / (2.0 * n) for i, n in enumerate(n_cls)}
    print(f"\nClass weights: " + ", ".join(f"{c}={class_weights[i]:.3f}" for i, c in enumerate(le.classes_)))

    # --- Callbacks (seleksi model memakai VAL, bukan test) ---
    cb_early = callbacks.EarlyStopping(
        monitor='val_auc', patience=20,
        restore_best_weights=True, mode='max', verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path, monitor='val_auc',
        save_best_only=True, mode='max', verbose=1
    )

    # --- Training ---
    print("\nMemulai training...")
    history = model.fit(
        X_train_s, y_train,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_s, y_val),
        class_weight=class_weights,
        callbacks=[cb_early, cb_reduce_lr, cb_checkpoint],
        verbose=1
    )

    # --- Evaluasi akhir pada TEST (tidak pernah dipakai untuk memilih model) ---
    print("\nEvaluasi pada data test (held-out)...")
    y_pred = (model.predict(X_test_s, verbose=0) > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test.astype(int), y_pred)

    print("\nLaporan Klasifikasi (test):")
    print(classification_report(y_test.astype(int), y_pred, target_names=le.classes_))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[0].set_title('Confusion Matrix (test)')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')

    ax2 = axes[1]
    ax2.plot(history.history['loss'],         label='Train Loss',   color='steelblue')
    ax2.plot(history.history['val_loss'],     label='Val Loss',     color='orange')
    ax2.set_ylabel('Loss', color='steelblue')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(history.history['accuracy'],     label='Train Acc',    color='green',      linestyle='--')
    ax3.plot(history.history['val_accuracy'], label='Val Acc',      color='red',        linestyle='--')
    ax3.plot(history.history['auc'],          label='Train AUC',    color='darkgreen',  linestyle=':')
    ax3.plot(history.history['val_auc'],      label='Val AUC',      color='darkred',    linestyle=':')
    ax3.set_ylabel('Accuracy / AUC')
    ax3.legend(loc='upper right')
    axes[1].set_title('Training History')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Grafik disimpan ke {plot_path}")
    print(f"\nModel terbaik disimpan di: {model_path}")


if __name__ == '__main__':
    main()
