import os

import numpy as np
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================
# Skema CSV v2 (telext_v2.py). Kolom ini wajib ada di Normal.csv dan Racing.csv.
FEATURES = ['speed_kmh', 'rpm', 'gear', 'throttle', 'brake',
            'accel_ms2', 'accel_lat', 'yaw_rate']

# Selisih timestamp antar baris (detik) yang dianggap sebagai putusnya rekaman.
# Perekaman 10 Hz -> jarak normal ~0.1 s. Lebih dari ini = sesi baru / ada bagian hilang.
GAP_S = 1.0

# Filter untuk membuang data ambigu
MIN_SPEED_KMH        = 15     # berlaku untuk KEDUA kelas (mobil diam terlihat sama di dua gaya)
NORMAL_MAX_RPM       = 4000
NORMAL_MAX_THROTTLE  = 70     # persen (0-100)
NORMAL_MAX_SPEED_KMH = 120


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

    required = ['timestamp'] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: kolom {missing} tidak ada. "
            "Rekam ulang memakai telext_v2.py (skema v2)."
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

    seg_len = df.groupby('segment').size()
    stats = {
        'label': label, 'loaded': n_loaded, 'invalid': n_invalid,
        'filtered': n_filtered, 'kept': len(df),
        'segments': len(seg_len),
        'seg_min': int(seg_len.min()), 'seg_med': int(seg_len.median()),
        'seg_max': int(seg_len.max()),
    }
    return df, stats


def keep_normal(df: pd.DataFrame) -> pd.Series:
    return (
        (df['rpm']       <= NORMAL_MAX_RPM)      &
        (df['throttle']  <= NORMAL_MAX_THROTTLE) &
        (df['speed_kmh'] <= NORMAL_MAX_SPEED_KMH) &
        (df['speed_kmh'] >= MIN_SPEED_KMH)
    )


def keep_racing(df: pd.DataFrame) -> pd.Series:
    return df['speed_kmh'] >= MIN_SPEED_KMH


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
      - Ambang filter Normal mendefinisikan apa itu "normal"; lihat konstanta di atas.
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
              f"| terfilter {st['filtered']:>5} | dipakai {st['kept']:>6}")
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
