# Arsitektur

Dokumen ini menjelaskan alur data, kontrak antar komponen, dan keputusan operasional pada `realtime-affective-analysis`.

## Tujuan

Sistem memantau ulasan toko di Shopee, lalu mengklasifikasi sentimen dan emosi setiap ulasan secara mendekati real-time. Hasilnya dipakai untuk dashboard dan alert.

## Topologi

Seluruh service berjalan di satu node lewat Docker Compose.

```text
Shopee API / sampel JSONL
        │
        ▼
   data-poller ──► Kafka "polled-data"
                        │
                        ▼
              data-preprocess (Spark Structured Streaming)
                        │
                        ▼
                 MongoDB replica set
                        │  Debezium MongoDB source
                        ▼
             Kafka "cdc.mongodb.ecommerce.reviews"
                        │
                        ▼
              ml-inference (Spark Structured Streaming)
                        │
                        ▼
                   PostgreSQL
                    ↙        ↘
              Streamlit    Trigger PL/Python ke Telegram
```

## Komponen

| Service | Teknologi | Peran | Port host |
|---|---|---|---|
| `data-poller` | Python, kafka-python | Mengambil ulasan dari API atau file replay, mengirim ke Kafka | - |
| `kafka` | Apache Kafka 4.1.2 (KRaft) | Message broker | 9092 |
| `kafka-ui` | Provectus Kafka UI | Inspeksi topic dan pesan | 8082 |
| `data-preprocess` | Spark 4.1.4, Spark ML | Normalisasi teks, TF-IDF, fitur dasar, tulis ke MongoDB | - |
| `mongodb` | MongoDB 8.0 replica set `rs0` | Document store dan sumber CDC | 27017 |
| `mongo-express` | Mongo Express | Inspeksi dokumen | 8081 |
| `debezium` | Debezium Connect 3.1.2 | Change data capture dari MongoDB | 8083 |
| `ml-inference` | Spark 4.1.4, scikit-learn | Prediksi sentimen dan emosi, tulis ke PostgreSQL | - |
| `spark-master` | Spark 4.1.4 | Master cluster | 8080, 7077 |
| `spark-worker` | Spark 4.1.4 | Worker cluster | - |
| `postgres` | PostgreSQL 18 + PL/Python | Menyimpan hasil inferensi, alert, dan metadata model | 5432 |
| `streamlit` | Streamlit 1.58 | Dashboard tiga halaman | 8501 |

## Alur data

1. `data-poller` mengirim JSON ulasan ke topic `polled-data`. Pada mode demo, sumbernya `data/sample/reviews_sample.jsonl`.
2. `data-preprocess` membaca topic tersebut, memvalidasi record, lalu menjalankan pipeline teks: normalisasi karakter berulang, normalisasi slang dari kamus, konversi emoji, tokenisasi, stopword Indonesia, stemming Sastrawi, dan penanganan negasi.
3. Teks yang sudah bersih divektorisasi dengan TF-IDF 1500 fitur. Sebelas fitur dasar ikut dihitung, yaitu jumlah tanda seru, tanda tanya, kata kapital, elipsis, pengulangan karakter, kata permintaan, kata ketidakpastian, kata kasar, kata keterikatan, frasa pembelian ulang, dan kata transaksional.
4. Dokumen hasil preprocessing ditulis ke MongoDB. `_id` diturunkan dari `md5(ctime + nama pengguna)`, sehingga penulisan ulang bersifat upsert.
5. Debezium menangkap perubahan MongoDB lewat replica set dan meneruskannya ke Kafka topic CDC.
6. `ml-inference` membaca CDC, memuat bundle model dari `models/`, merekonstruksi sparse vector, lalu memprediksi sentimen dan emosi.
7. Hasil prediksi di-upsert ke PostgreSQL berdasarkan `comment_id`. Ulasan negatif atau emosi negatif memicu baris di tabel `alerts`.
8. Dashboard membaca PostgreSQL setiap lima detik. Trigger PL/Python mengirim alert ke Telegram bila kredensial diisi.

## Kontrak data

### Pesan `polled-data`

```json
{
  "nama pengguna": "pengguna_001",
  "produk": "Nama produk",
  "review": "isi ulasan",
  "rating": 5,
  "waktu transaksi": "2026-11-14 06:00",
  "ctime": 1794592800
}
```

Field `review` dan `ctime` wajib valid. Record yang gagal validasi dikirim ke topic `dead-letter` beserta field `reason`, bukan ke MongoDB.

### Dokumen MongoDB

Dokumen berisi field masukan di atas ditambah:

- `review_preprocessed`
- `emotion_features_basic` (11 angka)
- `sentiment_vectorized` dan `emotion_vectorized`, berisi pasangan `{i, v}` untuk indeks dan nilai TF-IDF
- `preprocessed_at`

### Tabel PostgreSQL

- `reviews`: `comment_id` unik, teks, rating, sentimen, emosi, `create_time`, `processed_at`
- `alerts`: tipe alert, komentar, rating rata-rata, `review_id`, `triggered_at`
- `model_metadata`: nama model, versi, macro F1, waktu latih, status aktif, task

Indeks unik parsial `(alert_type, review_id)` mencegah alert ganda ketika batch diproses ulang.

## Kontrak model

```text
models/
├── manifest.json
├── sentiment/1.0.0/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metadata.json
└── emotion/1.0.0/
    ├── model.joblib
    ├── vectorizer.joblib
    └── metadata.json
```

`metadata.json` mencatat hash dataset, hash index test, urutan fitur, parameter model, metrik per kelas, dan versi library. Service membaca `manifest.json` untuk menentukan versi aktif, lalu mencatatnya ke `model_metadata`.

Urutan fitur yang harus dijaga saat serving:

- Sentimen: TF-IDF, lalu 11 fitur dasar
- Emosi: TF-IDF, 11 fitur dasar, lalu indikator sentimen positif

## Keandalan

- Checkpoint Spark disimpan di volume terpisah untuk preprocess, inference, dan alert rating.
- Semua penulisan ke PostgreSQL memakai upsert berdasarkan `comment_id`.
- Record tidak valid masuk ke `dead-letter` dengan alasan.
- Poller menulis penanda `replay_done` setelah mode replay selesai, supaya restart container tidak mengirim ulang sampel.
- Dashboard menampilkan waktu data terakhir dan warna kesegaran: hijau sampai 5 menit, kuning sampai 15 menit, merah setelahnya.
- Sink console hanya aktif jika `DEBUG_SINKS=true`, supaya tidak menggandakan pekerjaan UDF.

## Operasional

```bash
task demo    # nyalakan seluruh stack dan kirim sampel
task replay  # kirim ulang sampel
task down    # hentikan stack
task reset   # hapus volume dan mulai bersih
```

Kebutuhan sumber daya sekitar 6 GB RAM untuk seluruh stack. Mode demo memakai satu Spark master dan satu worker. Jika RAM mencukupi, worker bisa ditambah dengan `docker compose up -d --scale spark-worker=2`.
