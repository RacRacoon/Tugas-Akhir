"""
telemetry.py (v2)
Klasifikasi gaya berkendara BeamNG.drive secara real-time.

Beda dari versi sebelumnya:
  - Baca OutGauge (4444) DAN OutSim (4123), bukan OutGauge saja.
  - accel_ms2 dari sensor OutSim, plus fitur baru accel_lat & yaw_rate.
  - Parsing paket di-import dari Telemetry_Extract/telext_v2.py (satu sumber
    kebenaran dengan skrip perekam, supaya tidak drift kalau sumbu OutSim
    diubah setelah --probe).
  - Sampling di jam tetap (RATE_HZ) sama seperti saat data dilatih, bukan
    tiap paket UDP tiba (~60 Hz) seperti dulu. Window 30 langkah sekarang
    = 3 detik, sama seperti saat training; dulu cuma ~0.5 detik.
  - Fitur dan urutannya dibaca dari Models/features.json (ditulis training.py),
    bukan di-hardcode, jadi tidak bisa diam-diam tidak-sinkron dengan model.
"""

import json
import os
import select
import sys
import time
import warnings
from collections import deque

import numpy as np
import joblib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, '..', 'Telemetry_Extract'))
import telext_v2 as rec  # noqa: E402  (sumber tunggal parsing OutGauge/OutSim)

MODEL_PATH    = os.path.join(_BASE, '..', 'Models', 'driver_style_model.h5')
SCALER_PATH   = os.path.join(_BASE, '..', 'Models', 'scaler.bin')
FEATURES_PATH = os.path.join(_BASE, '..', 'Models', 'features.json')

RATE_HZ          = 10.0   # HARUS sama dengan RATE_HZ di telext_v2.py saat data dilatih
STALE_S          = 0.5
PREDICT_EVERY_N  = 3       # prediksi tiap 3 tick (~0.3 s pada 10 Hz)
SMOOTHING_LEN    = 15
THRESHOLD_RACING = 0.85
THRESHOLD_NORMAL = 0.50

# ============================================================
# LOAD MODEL, SCALER & DAFTAR FITUR
# ============================================================
print('Memuat model AI...')
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH) as f:
    _cfg = json.load(f)
FEATURES = _cfg['features']

WINDOW_SIZE = model.input_shape[1]
N_FEATURES  = model.input_shape[2]
if N_FEATURES != len(FEATURES):
    sys.exit(
        f"Model butuh {N_FEATURES} fitur tapi features.json berisi {len(FEATURES)} "
        f"({FEATURES}). Model dan features.json tidak sinkron — retrain atau cek path."
    )
print(f'Model loaded | window={WINDOW_SIZE} | fitur={FEATURES}')


@tf.function(
    input_signature=[tf.TensorSpec(shape=(1, None, None), dtype=tf.float32)],
    reduce_retracing=True
)
def _predict(x):
    return model(x, training=False)


# ============================================================
# STATE
# ============================================================
data_buffer         = deque(maxlen=WINDOW_SIZE)
prediction_history  = deque(maxlen=SMOOTHING_LEN)
current_style       = 'NORMAL'
tick_counter        = 0


def make_row(latest_g: dict, latest_s: dict) -> dict:
    return {**latest_g, **latest_s}


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    global current_style, tick_counter

    g_sock = rec.make_socket(rec.OUTGAUGE_ADDR)
    s_sock = rec.make_socket(rec.OUTSIM_ADDR)

    interval  = 1.0 / RATE_HZ
    next_tick = time.perf_counter() + interval
    latest_g = latest_s = None
    t_g = t_s = 0.0

    print(f'Siap! Monitoring BeamNG — OutGauge:{rec.OUTGAUGE_ADDR[1]} '
          f'+ OutSim:{rec.OUTSIM_ADDR[1]} @ {RATE_HZ:.0f} Hz')
    print('Tekan Ctrl+C untuk berhenti.\n')

    try:
        while True:
            timeout = max(0.0, next_tick - time.perf_counter())
            ready, _, _ = select.select([g_sock, s_sock], [], [], timeout)

            for s in ready:
                raw, _addr = s.recvfrom(1024)
                now = time.perf_counter()
                if s is g_sock and len(raw) >= rec.OUTGAUGE_MIN_LEN:
                    latest_g, t_g = rec.parse_outgauge(raw), now
                elif s is s_sock and len(raw) >= rec.OUTSIM_MIN_LEN:
                    latest_s, t_s = rec.parse_outsim(raw), now

            now = time.perf_counter()
            if now < next_tick:
                continue
            next_tick = now + interval if now - next_tick > interval else next_tick + interval

            if latest_g is None or latest_s is None \
                    or now - t_g > STALE_S or now - t_s > STALE_S:
                print('\rMenunggu OutGauge+OutSim... (aktifkan di Options > Other)',
                      end='', flush=True)
                continue

            row = make_row(latest_g, latest_s)
            try:
                features = [row[name] for name in FEATURES]
            except KeyError as e:
                sys.exit(f"Fitur {e} dari features.json tidak tersedia dari telext_v2. "
                          "Model dilatih dengan fitur yang tidak dihasilkan skrip ini.")

            data_buffer.append(features)
            tick_counter += 1

            if len(data_buffer) < WINDOW_SIZE:
                continue
            if tick_counter % PREDICT_EVERY_N != 0:
                continue

            arr    = np.array(data_buffer, dtype=np.float32)
            scaled = scaler.transform(arr)
            tensor = tf.constant(scaled.reshape(1, WINDOW_SIZE, N_FEATURES))

            raw_score = float(_predict(tensor).numpy()[0][0])
            prediction_history.append(raw_score)
            smoothed  = float(np.mean(prediction_history))

            if current_style == 'NORMAL'  and smoothed > THRESHOLD_RACING:
                current_style = 'RACING'
            elif current_style == 'RACING' and smoothed < THRESHOLD_NORMAL:
                current_style = 'NORMAL'

            filled = int(np.clip(smoothed, 0.0, 1.0) * 20)
            bar    = '#' * filled + '-' * (20 - filled)
            color  = '\033[91m' if current_style == 'RACING' else '\033[92m'
            print(
                f'\r[{bar}] {color}{current_style:<7}\033[0m'
                f' | Score: {smoothed:.3f}'
                f' | Spd: {row["speed_kmh"]:>6.1f} km/h'
                f' | RPM: {row["rpm"]:>5.0f}',
                end='', flush=True
            )

    except KeyboardInterrupt:
        print('\n\nMonitoring dihentikan.')
    finally:
        g_sock.close()
        s_sock.close()


if __name__ == '__main__':
    main()
