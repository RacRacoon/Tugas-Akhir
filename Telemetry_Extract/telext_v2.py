"""
telext_v2.py
Merekam telemetri BeamNG.drive ke CSV dari DUA sumber UDP:

  - OutGauge (port 4444): speed, rpm, gear, throttle, brake
  - OutSim   (port 4123): akselerasi longitudinal/lateral dan yaw rate (sensor asli)
                          CATATAN: struct di bawah ini masih asumsi gaya LFS/OutSim
                          klasik, BELUM diverifikasi terhadap protokol MotionSim resmi
                          BeamNG (beda struct, lihat dokumentasi). Belum diperbaiki —
                          sengaja ditunda, lihat riwayat chat.

Steering TIDAK direkam langsung. Percobaan sebelumnya pakai mod custom
protocol Lua (BeamNG_Mod/steertx/) gagal — mod itu bikin vehicle gagal load
di BeamNG.drive (sudah dikonfirmasi lewat tes isolasi oleh pengguna), jadi
didrop. Sebagai gantinya, 'steering' diESTIMASI tiap tick dari yaw_rate dan
speed pakai model kinematik sederhana (bicycle model):

    steering_rad ~= atan(WHEELBASE_M * yaw_rate / speed_ms)

Ini BUKAN pengukuran steering asli, dan secara matematis bukan informasi
baru di luar yaw_rate+speed yang sudah direkam (cuma reparameterisasi
non-linear keduanya) -- tujuannya cuma membantu model tidak perlu belajar
sendiri hubungan atan() itu. WHEELBASE_M WAJIB disesuaikan dengan mobil yang
dipakai merekam (lihat konstanta di bawah).

Beda dari telext_advanced.py:
  - accel_ms2 diambil dari sensor OutSim, bukan turunan speed (lebih bersih).
  - Tambah accel_lat (gaya samping), yaw_rate (kecepatan putar body), dan
    steering (estimasi, lihat di atas).
    steering_speed & steering_accel TIDAK dihitung di sini — itu turunan yang
    sensitif terhadap jitter, dihitung di Dataset.py pakai timestamp asli per
    segmen (lihat komentar di sana), bukan di recorder ini.
  - Perekaman dipicu jam tetap (default 10 Hz), bukan waktu tiba paket.
  - jerk dihapus (tidak dipakai training, hanya menambah noise).

Aktifkan dulu di BeamNG.drive: Options > Other > Protocols > OutGauge dan
OutSim. IP 127.0.0.1, port sesuai konstanta di bawah.

Mode probe untuk memverifikasi layout paket OutSim di mesin kamu:
    python telext_v2.py --probe
"""

import argparse
import csv
import math
import os
import select
import socket
import struct
import sys
import time

# ============================================================
# KONFIGURASI
# ============================================================
OUTGAUGE_ADDR = ('127.0.0.1', 4444)
OUTSIM_ADDR   = ('127.0.0.1', 4123)
CSV_FILE      = 'Extracted_v2.csv'
RATE_HZ       = 10.0
STALE_S       = 0.5     # paket lebih tua dari ini dianggap tidak ada

# Pemetaan sumbu OutSim (indeks 0..2 pada vektor Accel / AngVel).
# Asumsi awal mengikuti layout OutSim ala LFS: X kanan, Y depan, Z atas.
# WAJIB dicek dengan --probe, lalu ubah axis/tanda di sini bila perlu.
ACCEL_LON_AXIS, ACCEL_LON_SIGN = 1, 1.0
ACCEL_LAT_AXIS, ACCEL_LAT_SIGN = 0, 1.0
YAW_AXIS,       YAW_SIGN       = 2, 1.0

# Estimasi steering (bicycle model) -- lihat docstring modul.
WHEELBASE_M   = 2.7   # meter. PLACEHOLDER -- ganti sesuai mobil yang dipakai rekam.
MIN_SPEED_EST = 0.5   # m/s, floor penyebut biar gak div-by-zero saat nyaris diam

FIELDS = ['timestamp', 'car', 'speed_kmh', 'rpm', 'gear',
          'throttle', 'brake', 'accel_ms2', 'accel_lat', 'yaw_rate', 'steering']

OUTSIM_MIN_LEN   = 52   # uint time + 12 float
OUTGAUGE_MIN_LEN = 56   # sampai field brake


# ============================================================
# PARSER
# ============================================================
def parse_outgauge(data: bytes) -> dict:
    car_name = data[4:8].decode('ascii', errors='ignore').strip('\x00')
    gear_raw = struct.unpack_from('B',   data, 10)[0]
    speed_ms = struct.unpack_from('<f',  data, 12)[0]
    rpm      = struct.unpack_from('<f',  data, 16)[0]
    throttle = struct.unpack_from('<f',  data, 48)[0]
    brake    = struct.unpack_from('<f',  data, 52)[0]
    return {
        'car':       car_name,
        'speed_kmh': round(speed_ms * 3.6, 2),
        'rpm':       int(rpm),
        'gear':      gear_raw - 1,
        'throttle':  round(throttle * 100.0, 2),
        'brake':     round(brake * 100.0, 2),
    }


def parse_outsim(data: bytes) -> dict:
    # uint time, AngVel[3], Heading, Pitch, Roll, Accel[3], Vel[3], (Pos[3], ID)
    vals    = struct.unpack_from('<I12f', data, 0)
    ang_vel = vals[1:4]
    accel   = vals[7:10]
    return {
        'accel_ms2': round(ACCEL_LON_SIGN * accel[ACCEL_LON_AXIS], 3),
        'accel_lat': round(ACCEL_LAT_SIGN * accel[ACCEL_LAT_AXIS], 3),
        'yaw_rate':  round(YAW_SIGN * ang_vel[YAW_AXIS], 4),
    }


def estimate_steering(speed_kmh: float, yaw_rate: float) -> float:
    """Estimasi sudut setir (radian) dari model kinematik bicycle, lihat
    docstring modul. speed_kmh dikonversi ke m/s, dengan floor kecil di
    penyebut biar tidak div-by-zero saat mobil nyaris diam (baris begini
    toh nanti dibuang Dataset.py lewat MIN_SPEED_KMH filter)."""
    speed_ms = max(speed_kmh / 3.6, MIN_SPEED_EST)
    return round(math.atan2(WHEELBASE_M * yaw_rate, speed_ms), 4)


def make_socket(addr) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(addr)
    s.setblocking(False)
    return s


# ============================================================
# MODE PROBE
# ============================================================
def probe():
    g_sock = make_socket(OUTGAUGE_ADDR)
    s_sock = make_socket(OUTSIM_ADDR)
    print('PROBE: gas/rem/belok di game, lihat sumbu mana yang berubah. Ctrl+C untuk berhenti.\n')
    last = 0.0
    try:
        while True:
            ready, _, _ = select.select([g_sock, s_sock], [], [], 0.2)
            for s in ready:
                raw, _ = s.recvfrom(1024)
                now = time.perf_counter()
                if now - last < 0.5:
                    continue
                if s is s_sock and len(raw) >= OUTSIM_MIN_LEN:
                    v = struct.unpack_from('<I12f', raw, 0)
                    last = now
                    print(f'OutSim len={len(raw)} t={v[0]}\n'
                          f'  angvel = {v[1]:8.3f} {v[2]:8.3f} {v[3]:8.3f}\n'
                          f'  hpr    = {v[4]:8.3f} {v[5]:8.3f} {v[6]:8.3f}\n'
                          f'  accel  = {v[7]:8.3f} {v[8]:8.3f} {v[9]:8.3f}\n'
                          f'  vel    = {v[10]:8.3f} {v[11]:8.3f} {v[12]:8.3f}')
                elif s is g_sock and len(raw) >= OUTGAUGE_MIN_LEN:
                    d = parse_outgauge(raw)
                    print(f"OutGauge len={len(raw)} {d['speed_kmh']} km/h "
                          f"rpm={d['rpm']} thr={d['throttle']} brk={d['brake']}")
    except KeyboardInterrupt:
        print('\nProbe selesai.')
    finally:
        g_sock.close()
        s_sock.close()


# ============================================================
# REKAM
# ============================================================
def check_header(path: str):
    """Hentikan jika CSV lama punya skema kolom berbeda (hindari file campur)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline='') as f:
        header = next(csv.reader(f), [])
    if header != FIELDS:
        sys.exit(f'Header {path} beda dengan skema v2. Pakai nama file lain atau hapus file lama.')


def record():
    check_header(CSV_FILE)
    g_sock = make_socket(OUTGAUGE_ADDR)
    s_sock = make_socket(OUTSIM_ADDR)

    interval    = 1.0 / RATE_HZ
    latest_g    = latest_s = None
    t_g = t_s   = 0.0
    next_tick   = time.perf_counter() + interval
    saved = skipped = 0
    is_new = not os.path.isfile(CSV_FILE) or os.path.getsize(CSV_FILE) == 0

    print(f'Recording ke {CSV_FILE} ({RATE_HZ:.0f} Hz). Ctrl+C untuk berhenti.\n')

    with open(CSV_FILE, mode='a', newline='', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        try:
            while True:
                timeout = max(0.0, next_tick - time.perf_counter())
                ready, _, _ = select.select([g_sock, s_sock], [], [], timeout)

                for s in ready:
                    raw, _ = s.recvfrom(1024)
                    now = time.perf_counter()
                    if s is g_sock and len(raw) >= OUTGAUGE_MIN_LEN:
                        latest_g, t_g = parse_outgauge(raw), now
                    elif s is s_sock and len(raw) >= OUTSIM_MIN_LEN:
                        latest_s, t_s = parse_outsim(raw), now

                now = time.perf_counter()
                if now < next_tick:
                    continue
                next_tick = now + interval if now - next_tick > interval else next_tick + interval

                if latest_g is None or latest_s is None \
                        or now - t_g > STALE_S or now - t_s > STALE_S:
                    skipped += 1
                    print(f'\rMenunggu OutGauge+OutSim... (dilewati: {skipped})', end='', flush=True)
                    continue

                steering = estimate_steering(latest_g['speed_kmh'], latest_s['yaw_rate'])
                row = {'timestamp': round(time.time(), 6), **latest_g, **latest_s,
                       'steering': steering}
                writer.writerow(row)
                saved += 1
                print(f"\rSAVED {saved:>6} | {row['speed_kmh']:>6.1f} km/h"
                      f" | GAS {row['throttle']:>5.1f}%"
                      f" | LON {row['accel_ms2']:>6.2f}"
                      f" | LAT {row['accel_lat']:>6.2f}"
                      f" | STEER~{row['steering']:>6.3f}", end='', flush=True)

        except KeyboardInterrupt:
            print(f'\nBerhenti. {saved} baris tersimpan di {CSV_FILE}.')
        finally:
            g_sock.close()
            s_sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true', help='cetak paket mentah untuk cek sumbu OutSim')
    args = ap.parse_args()
    probe() if args.probe else record()


if __name__ == '__main__':
    main()
