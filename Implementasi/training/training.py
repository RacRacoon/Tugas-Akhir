import os

# Path CUDA nvvm/libdevice dari paket apt nvidia-cuda-toolkit — tanpa ini XLA
# gagal JIT-compile kernel GPU dengan error "libdevice.10.bc not found".
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_cuda_data_dir=/usr/lib/cuda')

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI
# ============================================================
WINDOW_SIZE = 30    # Jumlah time-step per sampel
STEP_SIZE   = 2     # Step lebih kecil = lebih banyak sampel (overlap) dari data yang sama
TEST_SIZE   = 0.2
EPOCHS      = 100
BATCH_SIZE  = 32
RANDOM_SEED = 42

# Augmentasi jitter — nambah variasi sampel training dari data terbatas
AUGMENT       = True
JITTER_STD    = 0.01   # std noise relatif (fraksi dari std tiap fitur)
AUGMENT_COPIES = 2      # jumlah salinan ter-jitter per sampel asli (hanya di train set)

# Fitur inti — jerk dihapus karena terlalu volatile (artefak numerik)
# accel dipertahankan karena bermakna secara fisik
# gear dihapus — mobil matic, klasifikasi fokus ke throttle & brake
FEATURES = ['speed_kmh', 'rpm', 'throttle', 'brake', 'accel_ms2']

# Batas clipping outlier untuk fitur volatile
ACCEL_CLIP = 15.0   # m/s^2 — lebih dari ini tidak realistis untuk mobil jalan

# Path output
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'Models')
MODEL_PATH   = os.path.join(OUTPUT_DIR, 'driver_style_model.h5')
SCALER_PATH  = os.path.join(OUTPUT_DIR, 'scaler.bin')
ENCODER_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.bin')


# ============================================================
# 1. WINDOWING PER LABEL (kunci utama)
# ============================================================
def window_dataframe(df: pd.DataFrame, label_str: str, le: LabelEncoder) -> tuple:
    """
    Menerapkan sliding window HANYA pada baris dengan label tertentu.
    Ini memastikan setiap window berisi data murni satu kelas — tidak ada kontaminasi.
    """
    subset = df[df['label'] == label_str].copy()

    # Clip outlier accel sebelum windowing
    if 'accel_ms2' in subset.columns:
        subset['accel_ms2'] = subset['accel_ms2'].clip(-ACCEL_CLIP, ACCEL_CLIP)

    missing = [f for f in FEATURES if f not in subset.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    data_values = subset[FEATURES].values.astype(np.float32)
    label_enc   = le.transform([label_str])[0]

    X, y = [], []
    for i in range(0, len(data_values) - WINDOW_SIZE, STEP_SIZE):
        X.append(data_values[i : i + WINDOW_SIZE])
        y.append(label_enc)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(file_path: str):
    """
    Membaca dataset, melakukan windowing per label, lalu menggabungkan dan shuffle.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    le = LabelEncoder()
    le.fit(df['label'].unique())

    labels_in_data = df['label'].unique().tolist()
    print(f"   Label ditemukan: {labels_in_data}")

    all_X, all_y = [], []
    for lbl in labels_in_data:
        X_lbl, y_lbl = window_dataframe(df, lbl, le)
        print(f"   [{lbl}] -> {len(X_lbl)} windows")
        all_X.append(X_lbl)
        all_y.append(y_lbl)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Shuffle setelah windowing — sekarang aman karena window sudah murni per kelas
    idx = np.random.default_rng(RANDOM_SEED).permutation(len(X))
    return X[idx], y[idx], le


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


def augment_jitter(X_scaled: np.ndarray, y: np.ndarray, n_copies: int, std: float, seed: int) -> tuple:
    """
    Menambahkan salinan ter-jitter (Gaussian noise kecil) dari tiap sampel training.
    Dilakukan SETELAH scaling — std di sini relatif ke skala standar (mean 0, std 1).
    Hanya dipakai untuk data training, tidak untuk test/validasi.
    """
    rng = np.random.default_rng(seed)
    X_aug = [X_scaled]
    y_aug = [y]
    for _ in range(n_copies):
        noise = rng.normal(0.0, std, size=X_scaled.shape).astype(np.float32)
        X_aug.append(X_scaled + noise)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ============================================================
# 3. MAIN
# ============================================================
def main():
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_labeled.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} tidak ditemukan!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load & windowing per label ---
    print("Memuat dan mempersiapkan data (windowing per label)...")
    X, y, le = prepare_data(dataset_path)
    print(f"   Sampel total   : {len(X)}")
    print(f"   Shape X        : {X.shape}")
    print(f"   Normal windows : {int((y == le.transform(['normal'])[0]).sum())}")
    print(f"   Racing windows : {int((y == le.transform(['racing'])[0]).sum())}")

    if len(X) < 100:
        print("\nWARNING: Dataset terlalu kecil! Rekam lebih banyak data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # --- Scaling (fit hanya pada train) ---
    feat = X_train.shape[2]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(X_train.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    print(f"\nScaler & Encoder disimpan ke {OUTPUT_DIR}")

    # --- Augmentasi jitter (hanya train set) ---
    if AUGMENT:
        n_before = len(X_train_s)
        X_train_s, y_train = augment_jitter(X_train_s, y_train, AUGMENT_COPIES, JITTER_STD, RANDOM_SEED)
        idx = np.random.default_rng(RANDOM_SEED).permutation(len(X_train_s))
        X_train_s, y_train = X_train_s[idx], y_train[idx]
        print(f"Augmentasi jitter: {n_before} -> {len(X_train_s)} sampel training")

    # --- Bangun model ---
    model = build_model(WINDOW_SIZE, feat)
    model.summary()

    # --- Class weights ---
    n_normal = int((y_train == le.transform(['normal'])[0]).sum())
    n_racing = int((y_train == le.transform(['racing'])[0]).sum())
    total    = n_normal + n_racing
    w_normal = total / (2.0 * n_normal) if n_normal > 0 else 1.0
    w_racing = total / (2.0 * n_racing) if n_racing > 0 else 1.0
    class_weights = {
        int(le.transform(['normal'])[0]): w_normal,
        int(le.transform(['racing'])[0]): w_racing,
    }
    print(f"\nClass weights: Normal={w_normal:.3f}, Racing={w_racing:.3f}")

    # --- Callbacks ---
    cb_early = callbacks.EarlyStopping(
        monitor='val_auc', patience=20,
        restore_best_weights=True, mode='max', verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_PATH, monitor='val_auc',
        save_best_only=True, mode='max', verbose=1
    )

    # --- Training ---
    print("\nMemulai training...")
    history = model.fit(
        X_train_s, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_s, y_test),
        class_weight=class_weights,
        callbacks=[cb_early, cb_reduce_lr, cb_checkpoint],
        verbose=1
    )

    # --- Evaluasi ---
    print("\nEvaluasi pada data test...")
    y_pred = (model.predict(X_test_s, verbose=0) > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test.astype(int), y_pred)

    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test.astype(int), y_pred, target_names=le.classes_))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[0].set_title('Confusion Matrix')
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
    plot_path = os.path.join(OUTPUT_DIR, 'training_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Grafik disimpan ke {plot_path}")
    print(f"\nModel terbaik disimpan di: {MODEL_PATH}")


if __name__ == '__main__':
    main()
