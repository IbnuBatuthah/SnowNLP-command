# =======================================================================
#         SKRIP ANALISIS SENTIMEN BAHASA MANDARIN DENGAN SNOWNLP
#                  (Versi 3 - Metode Penggabungan Tabel)
# =======================================================================

import numpy as np
from Orange.data import Table, Domain, ContinuousVariable

# Langkah 1: Cek apakah library SnowNLP sudah terinstal
try:
    from snownlp import SnowNLP
except ImportError:
    raise Exception("FATAL ERROR: Library 'snownlp' tidak ditemukan. \n\nSOLUSI: Buka Command Prompt (CMD) atau Terminal, lalu jalankan perintah 'pip install snownlp'. Setelah itu, restart Orange.")

# Langkah 2: Pastikan ada data yang masuk dari widget sebelumnya
if in_data is not None:
    
    # -------------------------------------------------------------------
    # PENGATURAN PENTING: Ubah nama kolom di bawah ini agar sesuai
    # dengan nama kolom komentar di file CSV Anda.
    NAMA_KOLOM_KOMENTAR = "Comment"  # <--- UBAH 'Comment' JIKA PERLU
    # -------------------------------------------------------------------
    
    print(f"INFO: Memulai skrip dengan target kolom: '{NAMA_KOLOM_KOMENTAR}'")

    # Langkah 3: Temukan lokasi kolom komentar
    col_idx = -1
    is_meta = False
    for i, attr in enumerate(in_data.domain.attributes):
        if attr.name == NAMA_KOLOM_KOMENTAR: col_idx = i; break
    if col_idx == -1:
        for i, meta in enumerate(in_data.domain.metas):
            if meta.name == NAMA_KOLOM_KOMENTAR: col_idx = i; is_meta = True; break
    if col_idx == -1:
        all_cols = [a.name for a in in_data.domain.attributes] + [m.name for m in in_data.domain.metas]
        raise Exception(f"FATAL ERROR: Kolom '{NAMA_KOLOM_KOMENTAR}' tidak ditemukan! \n\nSOLUSI: Periksa kembali ejaan di skrip. Kolom yang tersedia: {all_cols}")
    
    print(f"INFO: Kolom '{NAMA_KOLOM_KOMENTAR}' ditemukan. Memproses {len(in_data)} baris data...")

    # Langkah 4: Hitung semua skor sentimen dan kumpulkan dalam sebuah list
    scores = []
    for row in in_data:
        text = str(row.metas[col_idx]) if is_meta else str(row[col_idx])
        score = np.nan
        if text and text.strip():
            try:
                score = SnowNLP(text).sentiments
            except Exception:
                pass # Abaikan jika ada error pada satu baris
        scores.append(score)

    # Langkah 5: Buat tabel data baru HANYA untuk kolom skor
    scores_array = np.array(scores).reshape(-1, 1)
    sentiment_var = ContinuousVariable("Sentiment_SNLP")
    scores_domain = Domain([sentiment_var])
    scores_table = Table.from_numpy(scores_domain, X=scores_array)

    # Langkah 6: Gabungkan tabel asli dengan tabel skor
    # axis=1 berarti menggabungkan kolom-kolomnya
    out_data = Table.concatenate([in_data, scores_table], axis=1)
    
    print("INFO: Proses analisis sentimen selesai. Tabel berhasil digabungkan.")

else:
    print("INFO: Tidak ada data input yang diterima.")

# made with love by HarimauKicik
