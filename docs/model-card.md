# Model Card

## Ringkasan

Dua model klasifikasi teks Bahasa Indonesia untuk ulasan e-commerce:

| Task | Kelas | Model | Bundle |
|---|---|---|---|
| Sentimen | Positive, Negative | Logistic Regression | `models/sentiment/1.0.0` |
| Emosi | Happy, Love, Sadness, Fear, Anger | SMOTE + Logistic Regression | `models/emotion/1.0.0` |

Keduanya memakai TF-IDF 1500 fitur dan 11 fitur dasar. Model emosi menambahkan indikator sentimen positif.

## Penggunaan

Cocok untuk:

- Memantau sentimen dan emosi ulasan toko secara agregat.
- Mendeteksi lonjakan ulasan negatif atau penurunan rating.
- Menyaring ulasan yang perlu ditindaklanjuti manusia.

Tidak cocok untuk:

- Menilai satu orang atau mengambil keputusan otomatis terhadap pembeli.
- Klaim sebab akibat, misalnya "pengiriman terlambat menyebabkan pelanggan pergi".
- Domain di luar ulasan e-commerce berbahasa Indonesia tanpa evaluasi ulang.

## Data

| Properti | Nilai |
|---|---|
| Baris mentah | 5.400 |
| Duplikat dibuang | 112 |
| Baris dipakai | 5.288 |
| Split latih | 4.230 |
| Split uji | 1.058 |
| Stratifikasi | Gabungan sentimen dan emosi |
| Seed | 42 |
| Hash dataset | `05e330ffca7068...` |

Distribusi kelas emosi pada data mentah sekitar Happy 32,8%, Sadness 22,3%, Fear 17,0%, Love 15,0%, dan Anger 12,9%. Sentimen hampir seimbang, sekitar 52% negatif dan 48% positif.

Dataset berisi ulasan e-commerce yang sudah dilabeli. Kualitas label belum diaudit manual, sehingga ada kemungkinan label ambigu, terutama antara Fear, Anger, dan Sadness.

## Preprocessing

1. Normalisasi karakter berulang, misalnya `bagussss` menjadi `bagus`.
2. Normalisasi slang memakai kamus colloquial Indonesian.
3. Konversi emoji menjadi teks.
4. Tokenisasi NLTK.
5. Pembuangan stopword Bahasa Indonesia.
6. Stemming Sastrawi.
7. Penanganan negasi dengan prefiks `NEG_` pada kata setelah kata negasi.

## Fitur

- TF-IDF unigram, `max_features=1500`.
- 11 fitur dasar: jumlah tanda seru, tanda tanya, kata kapital, elipsis, pengulangan karakter, kata permintaan, kata ketidakpastian, kata kasar, kata keterikatan, frasa pembelian ulang, dan kata transaksional.
- Indikator sentimen positif untuk model emosi. Saat latihan nilainya berasal dari prediksi out-of-fold, bukan label asli, supaya sama dengan kondisi serving.

Urutan fitur disimpan di `metadata.json` dan wajib dipatuhi service inference.

## Pelatihan

- Grid hyperparameter: `C` pada 0,1, 1, dan 10.
- Validasi: 5-fold Stratified K-Fold di data latih.
- Metrik pemilihan: macro F1.
- Holdout dievaluasi satu kali setelah model dipilih.
- Model emosi memakai SMOTE `k_neighbors=3` di dalam pipeline, sehingga oversampling hanya terjadi pada fold latihan.
- Baseline: `DummyClassifier` strategi kelas mayoritas.

## Metrik

### Sentimen

| Metrik | Nilai |
|---|---|
| CV macro F1 | 0,9078 ± 0,0043 |
| Holdout accuracy | 0,9216 |
| Holdout macro F1 | 0,9214 |
| Baseline macro F1 | 0,3416 |

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0,9161 | 0,9344 | 0,9252 | 549 |
| Positive | 0,9277 | 0,9077 | 0,9176 | 509 |

### Emosi

| Metrik | Nilai |
|---|---|
| CV macro F1 | 0,6003 ± 0,0129 |
| Holdout accuracy | 0,6134 |
| Holdout macro F1 | 0,5818 |
| Baseline macro F1 | 0,0994 |

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anger | 0,4000 | 0,4889 | 0,4400 | 135 |
| Fear | 0,4439 | 0,4663 | 0,4548 | 178 |
| Happy | 0,7940 | 0,7600 | 0,7766 | 350 |
| Love | 0,6585 | 0,6792 | 0,6687 | 159 |
| Sadness | 0,6087 | 0,5339 | 0,5688 | 236 |

Emosi positif jauh lebih mudah dikenali. Kesalahan terbesar ada pada Anger yang sering tertukar dengan Fear dan Sadness, serta Fear yang punya recall lebih tinggi daripada precision.

## Limitasi

- Macro F1 emosi 0,5818. Model layak untuk triase, belum layak untuk keputusan otomatis.
- Label emosi berasal dari dataset dan belum diaudit, sehingga ada batas atas yang tidak diketahui.
- Model dilatih pada ulasan e-commerce umum. Saat dipakai untuk toko tertentu, kosakata dan gaya bahasa bisa berbeda.
- Kalibrasi probabilitas belum diuji.
- Model pembanding seperti IndoBERT atau sentence embedding belum dievaluasi pada protokol yang sama.
- Inferensi memakai CPU. Model dipilih dengan mempertimbangkan latensi, bukan hanya akurasi.

## Pertimbangan etis

- Ulasan adalah data publik, tetapi sampel demo tetap dianonimkan: email, nomor telepon, URL, dan mention diganti.
- Model tidak menyimpan data pribadi di luar teks ulasan.
- Prediksi bisa salah. Alert sebaiknya diperlakukan sebagai sinyal, bukan vonis.
- Jangan memakai model untuk memprofil pelanggan individual.
