import os

import numpy as np
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================
# Skema CSV v2 (telext_v2.py). Kolom ini wajib ada di Normal.csv dan Racing.csv.
RECORDED_FEATURES = ['speed_kmh', 'rpm', 'gear', 'throttle', 'brake',
                      'accel_ms2', 'accel_lat', 'yaw_rate', 'steering']

# steering_speed & steering_accel TIDAK direkam langsung — BeamNG cuma kasih
# posisi setir (steering), bukan kecepatan/akselerasinya. Dihitung di sini,
# bukan di telext_v2.py, supaya pakai timestamp ASLI per baris & terikat ke
# batas segmen (lihat add_steering_derivatives) — bukan asumsi dt tetap yang
# gampang goyah oleh jitter waktu kedatangan paket UDP.
DERIVED_FEATURES = ['steering_speed', 'steering_accel']

FEATURES = RECORDED_FEATURES + DERIVED_FEATURES

# Selisih timestamp antar baris (detik) yang dianggap sebagai putusnya rekaman.
# Perekaman 10 Hz -> jarak normal ~0.1 s. Lebih dari ini = sesi baru / ada bagian hilang.
GAP_S = 1.0

# Filter untuk membuang data ambigu. HANYA filter fisik netral (berlaku sama
# untuk kedua kelas) yang boleh ada di sini — bukan ambang rpm/throttle/speed
# per kelas. Label "normal" vs "racing" berasal dari SESI rekaman (niat
# pengemudi saat itu), bukan dari nilai fitur. Kalau kelas "normal" dipaksa
# tidak pernah punya rpm/throttle/speed tinggi lewat filter, model cukup
# belajar ambang itu, bukan pola perilaku — makanya ambang per kelas yang
# dulu ada di sini (NORMAL_MAX_RPM/THROTTLE/SPEED) dihapus.
MIN_SPEED_KMH = 15     # buang baris mobil diam; identik di kedua kelas


# ============================================================
# LOAD & SEGMENTASI
# ============================================================
def load_recording(path: str) -> tuple:
    """
    Membaca satu CSV rekaman, memvalidasi skema, dan membuang baris rusak (NaN/inf).
    Kolom `_row` menyimpan posisi baris asli agar celah akibat filter bisa dikenali.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required = ['timestamp'] + RECORDED_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: kolom {missing} tidak ada. "
            "Rekam ulang memakai telext_v2.py (skema v2, dengan mod steertx aktif)."
        )

    df = df[required].apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df['_row'] = np.arange(len(df))

    n_before = len(df)
    df = df.dropna(subset=required)
    return df, n_before - len(df)


def assign_segments(df: pd.DataFrame) -> pd.Series:
    """
    Segmen = potongan baris yang benar-benar berurutan di waktu.
    Segmen baru dimulai jika:
      - ada baris di antaranya yang dibuang (filter / NaN),
      - timestamp mundur atau melompat lebih dari GAP_S (sesi rekaman berbeda).
    Sliding window di training.py hanya boleh berada di dalam satu segmen,
    sehingga tidak ada window yang menyambung dua kejadian yang tidak berhubungan.
    """
    dt = df['timestamp'].diff()
    boundary = (df['_row'].diff() != 1) | (dt <= 0) | (dt > GAP_S)
    boundary.iloc[0] = True
    return boundary.cumsum() - 1


def add_steering_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    steering_speed = d(steering)/dt, steering_accel = d(steering_speed)/dt,
    dihitung PER SEGMEN pakai timestamp asli (df harus sudah terurut per
    segmen+waktu dan punya kolom 'segment'). Baris pertama & kedua tiap
    segmen tidak punya turunan penuh (NaN) -- dibuang oleh pemanggil.
    """
    df = df.sort_values(['segment', 'timestamp'])
    dt = df.groupby('segment')['timestamp'].diff()
    df['steering_speed'] = df.groupby('segment')['steering'].diff() / dt
    df['steering_accel'] = df.groupby('segment')['steering_speed'].diff() / dt
    return df


def prepare_class(path: str, label: str, keep_fn, seg_offset: int) -> tuple:
    df, n_invalid = load_recording(path)
    n_loaded = len(df) + n_invalid

    keep = keep_fn(df)
    n_filtered = int((~keep).sum())
    df = df[keep].copy()
    if df.empty:
        raise ValueError(f"{os.path.basename(path)}: semua baris terbuang oleh filter.")

    df['segment'] = assign_segments(df) + seg_offset
    df['label'] = label

    df = add_steering_derivatives(df)
    n_deriv_nan = int(df[DERIVED_FEATURES].isna().any(axis=1).sum())
    df = df.dropna(subset=DERIVED_FEATURES)
    if df.empty:
        raise ValueError(
            f"{os.path.basename(path)}: semua baris hilang setelah hitung turunan steering "
            "(segmen terlalu pendek?)."
        )

    seg_len = df.groupby('segment').size()
    stats = {
        'label': label, 'loaded': n_loaded, 'invalid': n_invalid,
        'filtered': n_filtered, 'deriv_nan': n_deriv_nan, 'kept': len(df),
        'segments': len(seg_len),
        'seg_min': int(seg_len.min()), 'seg_med': int(seg_len.median()),
        'seg_max': int(seg_len.max()),
    }
    return df, stats


def keep_idle_removed(df: pd.DataFrame) -> pd.Series:
    """Filter netral: buang baris mobil diam. Sama untuk kedua kelas, jadi
    tidak mendefinisikan 'normal'/'racing' lewat nilai fitur."""
    return df['speed_kmh'] >= MIN_SPEED_KMH


# Alias — dipakai per kelas di prepare_class() supaya nanti mudah diberi
# filter khusus kelas lagi kalau memang diperlukan (dengan alasan yang jelas,
# bukan ambang yang kebetulan sama dengan fitur model).
keep_normal = keep_idle_removed
keep_racing = keep_idle_removed


# ============================================================
# MERGE
# ============================================================
def merge_beamng_data(normal_path: str, racing_path: str, output_name: str = 'dataset_labeled.csv'):
    """
    Menggabungkan data Normal dan Racing (skema v2), membersihkan noise,
    memberi id `segment`, lalu menyimpan sebagai CSV siap training.

    PENTING:
      - Baris TIDAK di-shuffle dan TIDAK disambung lintas segmen.
      - Kolom `segment` dipakai training.py untuk membuat window dan membagi
        train/val/test tanpa kebocoran (window overlap antar split).
      - Label "normal"/"racing" murni dari sesi rekaman (nama file), bukan
        dari ambang nilai fitur — lihat keep_idle_removed() di atas.
    """
    for p in (normal_path, racing_path):
        if not os.path.exists(p):
            print(f"File {p} tidak ditemukan!")
            return None

    df_normal, st_n = prepare_class(normal_path, 'normal', keep_normal, seg_offset=0)
    df_racing, st_r = prepare_class(racing_path, 'racing', keep_racing,
                                    seg_offset=int(df_normal['segment'].max()) + 1)

    cols = ['timestamp', 'label', 'segment'] + FEATURES
    df_combined = pd.concat([df_normal[cols], df_racing[cols]], axis=0, ignore_index=True)

    print("-" * 60)
    print("STATS PEMBERSIHAN DATA:")
    for st in (st_n, st_r):
        print(f"  {st['label']:<7}: dimuat {st['loaded']:>6} | rusak {st['invalid']:>4} "
              f"| terfilter {st['filtered']:>5} | awal-segmen(turunan) {st['deriv_nan']:>3} "
              f"| dipakai {st['kept']:>6}")
        print(f"           segmen {st['segments']:>3} "
              f"(panjang min/median/maks = {st['seg_min']}/{st['seg_med']}/{st['seg_max']} baris)")
    print(f"  Total  : {len(df_combined):>6} baris")
    print("-" * 60)
    print("Distribusi label akhir:")
    print(df_combined['label'].value_counts().to_string())
    print("-" * 60)
    print("NOTE: Tidak di-shuffle. Window & split dilakukan per segmen di training.py.")

    df_combined.to_csv(output_name, index=False)
    print(f"Dataset tersimpan di: {output_name}")
    print(f"Kolom: {df_combined.columns.tolist()}")
    return df_combined


if __name__ == '__main__':
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    normal_path = os.path.join(base_dir, 'Normal.csv')
    racing_path = os.path.join(base_dir, 'Racing.csv')
    output_path = os.path.join(base_dir, 'dataset_labeled.csv')

    merge_beamng_data(normal_path, racing_path, output_name=output_path)
