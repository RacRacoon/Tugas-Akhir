"""
telext_basic.py
Merekam telemetri BeamNG.drive via OutGauge (UDP) ke CSV.
Versi sederhana: hanya menyimpan data mentah tanpa fitur turunan.
Gunakan telext_advanced.py untuk data yang siap training.
"""

import socket
import struct
import csv
import time
import os

# ============================================================
# KONFIGURASI
# ============================================================
UDP_IP   = '127.0.0.1'
UDP_PORT = 4444
CSV_FILE = 'driving_data_train.csv'


def parse_outgauge(data: bytes) -> dict:
    time_game   = struct.unpack_from('<I',  data, 0)[0]
    car_name    = data[4:8].decode('ascii', errors='ignore').strip('\x00')
    gear_raw    = struct.unpack_from('B',   data, 10)[0]
    physics     = struct.unpack_from('<7f', data, 12)
    dash_lights = struct.unpack_from('<I',  data, 40)[0]
    inputs      = struct.unpack_from('<3f', data, 44)

    return {
        'timestamp':  time.time(),
        'game_time':  time_game,
        'car':        car_name,
        'speed':      round(physics[0] * 3.6, 2),
        'rpm':        int(physics[1]),
        'gear':       gear_raw - 1,
        'throttle':   round(inputs[1] * 100.0, 2),
        'brake':      round(inputs[2] * 100.0, 2),
        'clutch':     round(inputs[0] * 100.0, 2),
        'lights':     dash_lights,
    }


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    fields = ['timestamp', 'game_time', 'car', 'speed', 'rpm',
              'gear', 'throttle', 'brake', 'clutch', 'lights']
    file_exists = os.path.isfile(CSV_FILE)

    print(f'Recording ke {CSV_FILE} ... Tekan Ctrl+C untuk berhenti.')

    with open(CSV_FILE, mode='a', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        try:
            while True:
                raw, _addr = sock.recvfrom(1024)
                if len(raw) < 88:
                    continue
                d = parse_outgauge(raw)
                writer.writerow(d)
                print(
                    f"\rREC: {d['speed']:>6.1f} km/h"
                    f" | GAS: {d['throttle']:>5.1f}%"
                    f" | RPM: {d['rpm']:>5}",
                    end='', flush=True
                )

        except KeyboardInterrupt:
            print(f'\nRecording selesai. Data disimpan di {CSV_FILE}')
        finally:
            sock.close()


if __name__ == '__main__':
    main()
