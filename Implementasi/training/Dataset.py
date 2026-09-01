import pandas as pd
import os


def merge_beamng_data(normal_path: str, racing_path: str, output_name: str = 'dataset_labeled.csv'):
    """
    Menggabungkan data Normal dan Racing, membersihkan noise,
    lalu menyimpan sebagai CSV siap training.

    PENTING: File ini TIDAK di-shuffle. Shuffle dilakukan nanti di training.py
             setelah windowing per-label, agar setiap window tetap murni satu kelas.

    Filter Normal : rpm <= 4000, throttle <= 70%, speed <= 120 km/h
    Filter Racing : speed >= 15 km/h (buang data diam)
    """
    # ---- Load Normal ----
    if not os.path.exists(normal_path):
        print(f"File {normal_path} tidak ditemukan!")
        return None

    df_normal = pd.read_csv(normal_path)
    df_normal.columns = df_normal.columns.str.strip().str.lower()

    initial_normal = len(df_normal)
    df_normal = df_normal[
        (df_normal['rpm']       <= 4000) &
        (df_normal['throttle']  <= 70)   &
        (df_normal['speed_kmh'] <= 120)
    ].copy()
    df_normal['label'] = 'normal'
    removed_normal = initial_normal - len(df_normal)

    # ---- Load Racing ----
    if not os.path.exists(racing_path):
        print(f"File {racing_path} tidak ditemukan!")
        return None

    df_racing = pd.read_csv(racing_path)
    df_racing.columns = df_racing.columns.str.strip().str.lower()

    initial_racing = len(df_racing)
    df_racing = df_racing[df_racing['speed_kmh'] >= 15].copy()
    df_racing['label'] = 'racing'
    removed_racing = initial_racing - len(df_racing)

    # ---- Gabungkan — JANGAN di-shuffle di sini ----
    # Normal dulu, lalu Racing → urutan blok terjaga untuk windowing per-label
    df_combined = pd.concat([df_normal, df_racing], axis=0, ignore_index=True)

    # ---- Statistik ----
    print("-" * 50)
    print("STATS PEMBERSIHAN DATA:")
    print(f"  Normal : {len(df_normal):>6} baris  (dibuang {removed_normal} noise)")
    print(f"  Racing : {len(df_racing):>6} baris  (dibuang {removed_racing} noise)")
    print(f"  Total  : {len(df_combined):>6} baris")
    print("-" * 50)
    print("Distribusi label akhir:")
    print(df_combined['label'].value_counts().to_string())
    print("-" * 50)
    print("NOTE: File ini TIDAK di-shuffle. Shuffle dilakukan di training.py setelah windowing.")

    # ---- Simpan ----
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
