"""
telext_advanced.py
Merekam telemetri BeamNG.drive via OutGauge (UDP) ke CSV.
Menghitung fitur turunan: accel_ms2 (percepatan) dan jerk (turunan percepatan).
Output CSV kompatibel langsung dengan Dataset.py dan training.py.
"""

import socket
import struct
import csv
import time
import os

# ============================================================
# KONFIGURASI
# ============================================================
UDP_IP         = '127.0.0.1'
UDP_PORT       = 4444
CSV_FILE       = 'Extracted.csv'
DATA_RATE_LIMIT = 0.05   # 20 Hz — lebih rapat, lebih banyak sampel per menit rekam

# ============================================================
# STATE GLOBAL
# ============================================================
_prev = {
    'time':  0.0,
    'speed': 0.0,
    'accel': 0.0,
}
_last_record_time = 0.0
_warmup_done      = False


def parse_outgauge(data: bytes):
    global _prev, _warmup_done

    curr_time = time.perf_counter()
    dt = curr_time - _prev['time']
    if dt <= 0:
        dt = 0.001

    car_name  = data[4:8].decode('ascii', errors='ignore').strip('\x00')
    gear_raw  = struct.unpack_from('B',   data, 10)[0]
    physics   = struct.unpack_from('<7f', data, 12)
    inputs    = struct.unpack_from('<3f', data, 44)

    speed_ms  = physics[0]
    rpm       = physics[1]
    speed_kmh = speed_ms * 3.6
    throttle  = inputs[1] * 100.0
    brake     = inputs[2] * 100.0

    accel = (speed_ms - _prev['speed']) / dt
    jerk  = (accel    - _prev['accel']) / dt

    _prev.update({'time': curr_time, 'speed': speed_ms, 'accel': accel})

    if not _warmup_done:
        _warmup_done = True
        return None

    return {
        'timestamp':  round(time.time(), 6),
        'car':        car_name,
        'speed_kmh':  round(speed_kmh, 2),
        'rpm':        int(rpm),
        'gear':       gear_raw - 1,
        'throttle':   round(throttle, 2),
        'brake':      round(brake, 2),
        'accel_ms2':  round(accel, 3),
        'jerk':       round(jerk, 3),
    }


def main():
    global _last_record_time, _prev

    _prev['time'] = time.perf_counter()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    fields = ['timestamp', 'car', 'speed_kmh', 'rpm', 'gear',
              'throttle', 'brake', 'accel_ms2', 'jerk']
    file_exists = os.path.isfile(CSV_FILE)

    print(f'Recording ke {CSV_FILE}  (interval: {DATA_RATE_LIMIT}s = {1/DATA_RATE_LIMIT:.0f} Hz)')
    print('Tekan Ctrl+C untuk berhenti.\n')

    with open(CSV_FILE, mode='a', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        try:
            while True:
                raw, _addr = sock.recvfrom(1024)
                curr_time  = time.perf_counter()

                if curr_time - _last_record_time < DATA_RATE_LIMIT:
                    continue
                if len(raw) < 88:
                    continue

                d = parse_outgauge(raw)
                if d is None:
                    continue

                writer.writerow(d)
                _last_record_time = curr_time
                print(
                    f"\rSAVED [{time.strftime('%H:%M:%S')}]"
                    f"  {d['speed_kmh']:>6.1f} km/h"
                    f"  | GAS {d['throttle']:>5.1f}%"
                    f"  | RPM {d['rpm']:>5}",
                    end='', flush=True
                )

        except KeyboardInterrupt:
            print('\nBerhenti. Data tersimpan.')
        finally:
            sock.close()


if __name__ == '__main__':
    main()
