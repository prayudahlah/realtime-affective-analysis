# Realtime Affective Analysis

Klasifikasi sentimen dan emosi ulasan e-commerce secara real-time, dari pengambilan data sampai dashboard dan notifikasi.

> Status: dalam pengerjaan. Struktur repo dan dokumentasi sedang disiapkan.

## Ringkasan

Ulasan produk dialirkan lewat Kafka, diproses dengan Spark Structured Streaming, disimpan di MongoDB, ditangkap perubahannya dengan Debezium, lalu diklasifikasi sentimen dan emosinya sebelum masuk PostgreSQL dan dashboard Streamlit.

Model dilatih terpisah di `ml/` dan dikemas sebagai artifact berversi di `models/`, sehingga pipeline hanya memuat bundle yang sudah teruji.

## Struktur

```text
ml/           pelatihan model, eksperimen, dan ekspor artifact
models/       bundle model berversi beserta metadata
streaming/    pipeline, service, dan compose
scripts/      utilitas pengembangan
docs/         arsitektur, model card, dan laporan
data/sample/  sampel data untuk demo
```

## Menjalankan

Belum tersedia. Perintah akan ditambahkan setelah pipeline dipindahkan pada tahap berikutnya.

## Lisensi

MIT, lihat [LICENSE](LICENSE).
