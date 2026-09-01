import os
import socket
import struct
import time
import warnings
from collections import deque

import numpy as np
import joblib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# KONFIGURASI
# ============================================================
UDP_IP   = '127.0.0.1'
UDP_PORT = 4444

_BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_BASE, '..', 'Models', 'driver_style_model.h5')
SCALER_PATH = os.path.join(_BASE, '..', 'Models', 'scaler.bin')

PREDICT_EVERY_N  = 6
SMOOTHING_LEN    = 15
THRESHOLD_RACING = 0.85
THRESHOLD_NORMAL = 0.50

# ============================================================
# LOAD MODEL & SCALER
# ============================================================
print('Memuat model AI...')
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

WINDOW_SIZE = model.input_shape[1]
N_FEATURES  = model.input_shape[2]
print(f'Model loaded  | Window={WINDOW_SIZE}, Features={N_FEATURES}')

@tf.function(
    input_signature=[tf.TensorSpec(shape=(1, None, None), dtype=tf.float32)],
    reduce_retracing=True
)
def _predict(x):
    return model(x, training=False)

# ============================================================
# STATE
# ============================================================
data_buffer        = deque(maxlen=WINDOW_SIZE)
prediction_history = deque(maxlen=SMOOTHING_LEN)
current_style      = 'NORMAL'
packet_counter     = 0
prev_speed         = 0.0
prev_accel         = 0.0
prev_time          = time.perf_counter()

# ============================================================
# PARSE BEAMNG OUTGAUGE PACKET
# ============================================================
def process_packet(raw: bytes) -> list:
    global prev_speed, prev_accel, prev_time

    physics = struct.unpack_from('<7f', raw, 12)
    inputs  = struct.unpack_from('<3f', raw, 44)
    gear    = struct.unpack_from('B',   raw, 10)[0] - 1

    now = time.perf_counter()
    dt  = max(now - prev_time, 0.001)

    speed_ms = physics[0]
    rpm      = physics[1]
    throttle = inputs[1] * 100.0
    brake    = inputs[2] * 100.0

    accel = (speed_ms - prev_speed) / dt
    jerk  = (accel    - prev_accel) / dt
    jerk  = float(np.clip(jerk, -500.0, 500.0))

    prev_speed = speed_ms
    prev_accel = accel
    prev_time  = now

    return [speed_ms * 3.6, rpm, gear, throttle, brake, accel, jerk]


# ============================================================
# MAIN LOOP
# ============================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

print(f'Siap! Monitoring BeamNG di {UDP_IP}:{UDP_PORT} (~60 Hz)')
print('Tekan Ctrl+C untuk berhenti.\n')

try:
    while True:
        raw, _addr = sock.recvfrom(1024)
        if len(raw) < 88:
            continue

        features = process_packet(raw)
        data_buffer.append(features)
        packet_counter += 1

        if len(data_buffer) < WINDOW_SIZE:
            continue
        if packet_counter % PREDICT_EVERY_N != 0:
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

        filled = int(smoothed * 20)
        bar    = '#' * filled + '-' * (20 - filled)
        color  = '\033[91m' if current_style == 'RACING' else '\033[92m'
        print(
            f'\r[{bar}] {color}{current_style:<7}\033[0m'
            f' | Score: {smoothed:.3f}'
            f' | Spd: {features[0]:>6.1f} km/h'
            f' | RPM: {features[1]:>5.0f}',
            end='', flush=True
        )

except KeyboardInterrupt:
    print('\n\nMonitoring dihentikan.')
finally:
    sock.close()
