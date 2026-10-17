"""
telemetry.py (v2)
Klasifikasi gaya berkendara BeamNG.drive secara real-time.

Beda dari versi sebelumnya:
  - Baca OutGauge (4444) dan MotionSim (4123), bukan OutGauge saja.
  - accel_ms2 dari sensor MotionSim, plus fitur accel_lat, yaw_rate.
  - steering diESTIMASI tiap tick dari yaw_rate+speed pakai model kinematik
    yang sama dengan telext_v2.estimate_steering() (mod custom protocol Lua
    yang sebelumnya dipakai bikin vehicle gagal load di BeamNG.drive, sudah
    didrop total -- lihat komentar di telext_v2.py).
  - steering_speed & steering_accel dihitung live di sini dari histori
    steering 3 tick terakhir dan RATE_HZ tetap (bukan timestamp per-paket,
    karena di real-time tidak ada konsep "segmen" seperti training — selama
    stream terus jalan di RATE_HZ yang sama ini konsisten dengan cara
    Dataset.py menghitungnya saat training).
  - Clip per-fitur (steering_accel dkk) diterapkan otomatis dari
    Models/features.json (`clip_specs`, ditulis training.py) — bukan
    di-hardcode di sini, jadi ikut kalau training.py diubah.
  - Parsing paket di-import dari Telemetry_Extract/telext_v2.py (satu sumber
    kebenaran dengan skrip perekam, supaya tidak drift kalau sumbu MotionSim
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
import telext_v2 as rec  # noqa: E402  (sumber tunggal parsing OutGauge/MotionSim)

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
FEATURES   = _cfg['features']
CLIP_SPECS = _cfg.get('clip_specs', {})
# Indeks posisi tiap kolom yang perlu di-clip, biar gak perlu cari index tiap tick.
CLIP_IDX = {FEATURES.index(col): limit for col, limit in CLIP_SPECS.items() if col in FEATURES}

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
steer_hist          = deque(maxlen=3)   # steering mentah 3 tick terakhir
steer_speed_hist    = deque(maxlen=2)   # steering_speed 2 tick terakhir (buat hitung accel)
current_style       = 'NORMAL'
tick_counter        = 0


def make_row(latest_g: dict, latest_s: dict) -> dict:
    row = {**latest_g, **latest_s}
    row['steering'] = rec.estimate_steering(row['speed_kmh'], row['yaw_rate'])
    return row


def update_steering_derivatives(steering: float) -> tuple:
    """
    steering_speed/steering_accel dari histori 3 tick terakhir, dengan
    interval tetap 1/RATE_HZ (lihat docstring modul). Dipanggil tepat 1x
    per tick tersimpan, urut waktu — jangan dipanggil ulang untuk tick yang
    sama, nanti histori kebanjiran nilai duplikat dan turunan jadi salah.
    """
    steer_hist.append(steering)
    if len(steer_hist) >= 2:
        speed = (steer_hist[-1] - steer_hist[-2]) * RATE_HZ
    else:
        speed = 0.0

    steer_speed_hist.append(speed)
    if len(steer_speed_hist) >= 2:
        accel = (steer_speed_hist[-1] - steer_speed_hist[-2]) * RATE_HZ
    else:
        accel = 0.0
    return speed, accel


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    global current_style, tick_counter

    g_sock = rec.make_socket(rec.OUTGAUGE_ADDR)
    s_sock = rec.make_socket(rec.MOTIONSIM_ADDR)

    interval  = 1.0 / RATE_HZ
    next_tick = time.perf_counter() + interval
    latest_g = latest_s = None
    t_g = t_s = 0.0

    print(f'Siap! Monitoring BeamNG — OutGauge:{rec.OUTGAUGE_ADDR[1]} '
          f'+ MotionSim:{rec.MOTIONSIM_ADDR[1]} @ {RATE_HZ:.0f} Hz')
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
                elif s is s_sock and rec.is_motionsim(raw):
                    latest_s, t_s = rec.parse_motionsim(raw), now

            now = time.perf_counter()
            if now < next_tick:
                continue
            next_tick = now + interval if now - next_tick > interval else next_tick + interval

            if latest_g is None or latest_s is None \
                    or now - t_g > STALE_S or now - t_s > STALE_S:
                print('\rMenunggu OutGauge+MotionSim... (aktifkan di Options > Other)',
                      end='', flush=True)
                continue

            row = make_row(latest_g, latest_s)
            row['steering_speed'], row['steering_accel'] = \
                update_steering_derivatives(row['steering'])

            try:
                features = [row[name] for name in FEATURES]
            except KeyError as e:
                sys.exit(f"Fitur {e} dari features.json tidak tersedia dari telext_v2. "
                          "Model dilatih dengan fitur yang tidak dihasilkan skrip ini.")

            for idx, limit in CLIP_IDX.items():
                features[idx] = float(np.clip(features[idx], -limit, limit))

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
                f' | RPM: {row["rpm"]:>5.0f}'
                f' | STEER~{row["steering"]:>5.2f}',
                end='', flush=True
            )

    except KeyboardInterrupt:
        print('\n\nMonitoring dihentikan.')
    finally:
        g_sock.close()
        s_sock.close()


if __name__ == '__main__':
    main()
