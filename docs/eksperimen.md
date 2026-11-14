# Catatan Eksperimen

Dokumen ini merangkum perjalanan eksperimen dari eksplorasi awal sampai protokol yang dipakai sekarang.

## Eksperimen awal

Notebook di `ml/notebooks/` memuat eksperimen faktorial:

- Empat algoritma: KNN, Naive Bayes, Logistic Regression, Decision Tree.
- Varian fitur: TF-IDF dasar, ditambah 11 fitur dasar, dan ditambah indikator sentimen untuk task emosi.
- Penanganan imbalance: tanpa sampling, SMOTE, dan ADASYN.
- Hyperparameter tuning dengan GridSearchCV dan 5-fold Stratified K-Fold.
- Metrik utama macro F1, dengan inference time sebagai pertimbangan kedua.

Hasil terbaik saat itu: sentimen macro F1 0,9249 dan emosi 0,67.

## Tiga masalah pada protokol awal

Sebelum hasil dipakai, ada tiga hal yang perlu diperbaiki.

### 1. Pemilihan model memakai test set

Puluhan konfigurasi diranking berdasarkan F1 di test set, lalu konfigurasi terbaik dilaporkan pada test set yang sama. Test set ikut berperan dalam pemilihan, sehingga angkanya optimistis.

Perbaikan: seleksi memakai CV di data latih. Test set hanya dievaluasi sekali setelah model final dipilih.

### 2. Indikator sentimen memakai label asli

Model emosi memakai fitur indikator sentimen. Saat latihan nilainya diambil dari label asli, sedangkan saat serving nilainya berasal dari prediksi model sentimen. Model belajar pada kondisi yang lebih mudah daripada kondisi saat dipakai.

Perbaikan: indikator dihasilkan dengan `cross_val_predict`, sehingga nilainya out-of-fold dan setara dengan kondisi serving.

### 3. Duplikat teks

Ada 112 teks duplikat. Sebagian bisa terbagi ke data latih dan data uji, sehingga skor naik palsu.

Perbaikan: deduplikasi dilakukan sebelum split.

## Protokol baru

1. Deduplikasi teks, 5.400 menjadi 5.288 baris.
2. Split 80/20 dengan stratifikasi gabungan sentimen dan emosi.
3. TF-IDF di-fit hanya pada data latih.
4. SMOTE berada di dalam pipeline, sehingga hanya terjadi pada fold latihan.
5. Grid hyperparameter kecil, `C` pada 0,1, 1, dan 10.
6. Seleksi berdasarkan macro F1 dari 5-fold CV.
7. Holdout dievaluasi satu kali.
8. Baseline majority class dihitung sebagai pembanding.
9. Artifact disimpan sebagai bundle berversi beserta metadata.

## Hasil

| Task | CV macro F1 | Holdout macro F1 | Baseline |
|---|---|---|---|
| Sentimen | 0,9078 ± 0,0043 | 0,9214 | 0,3416 |
| Emosi | 0,6003 ± 0,0129 | 0,5818 | 0,0994 |

Sentimen tetap kuat. Emosi turun dari 0,67 menjadi 0,5818.

Penurunan itu bukan kemunduran model, melainkan efek protokol yang lebih jujur. Pada uji diagnostik dengan konfigurasi yang sama, model emosi dengan label sentimen asli mencapai 0,6387, sedangkan dengan prediksi out-of-fold 0,5958. Selisih itu adalah train-serve skew yang sebelumnya tidak terlihat. Model final dengan C=1 menghasilkan 0,5818. Angka lama juga memilih model di test set, sehingga lebih optimistis lagi.

## Mengapa Logistic Regression

- Cocok untuk TF-IDF sparse berdimensi 1.500 dan data berukuran sedang.
- Latensi inferensi sangat kecil, penting karena prediksi berjalan di dalam micro-batch Spark.
- Hasil CV konsisten dan model mudah dijelaskan lewat koefisien.
- KNN dan Decision Tree kalah pada emosi, Naive Bayes kalah pada sentimen.

Transformer seperti IndoBERT belum diuji pada protokol ini. Alasannya, inferensi berjalan di CPU dan dataset hanya 5.288 baris. Menambahkan model tersebut tanpa mengukur latensi dan biaya akan melemahkan cerita engineering, bukan menguatkannya. Ini kandidat eksperimen berikutnya.

## Analisis error emosi

- Happy paling mudah dikenali, F1 0,7766.
- Love cukup baik, F1 0,6687.
- Sadness dan Fear sering tertukar.
- Anger sering dianggap Sadness atau Happy.
- Fear punya recall lebih tinggi daripada precision, artinya model cenderung memprediksi Fear pada ulasan yang tidak jelas.

Penyebab utamanya adalah batas kelas yang kabur secara bahasa. Ulasan seperti "kecewa tapi masih mau beli lagi" bisa masuk beberapa kelas. Tanpa audit label dan pedoman anotasi, sulit menaikkan skor secara berarti.

## Pelajaran

- Memilih model di test set membuat angka terlihat lebih baik, tetapi tidak bisa dipertanggungjawabkan.
- Fitur yang tersedia saat latihan harus sama dengan yang tersedia saat serving.
- Baseline murah membuat klaim "model bekerja" punya konteks.
- Menyimpan metadata dan hash split membuat hasil bisa direproduksi.
- Angka yang lebih rendah tetapi jujur lebih berguna daripada angka tinggi yang rapuh.

## Reproduksi

```bash
cd ml
uv sync
uv run python -m affective_ml.train --config configs/default.yaml
```

Hasil akan menimpa `models/sentiment/1.0.0/` dan `models/emotion/1.0.0/`. Notebook eksplorasi lengkap ada di `ml/notebooks/`, sedangkan eksperimen lama yang memakai dataset lain diarsipkan di `ml/notebooks/arsip/`.
