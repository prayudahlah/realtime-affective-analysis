# affective-ml

Paket pelatihan model sentimen dan emosi ulasan e-commerce.

## Instalasi

```bash
uv sync
```

## Menjalankan

```bash
uv run python -m affective_ml.train --config configs/default.yaml
```

Dataset diletakkan di `data/raw/ecommerce_reviews.csv`. Hasil training ditulis ke
`../models/<task>/<versi>/` dalam bentuk `model.joblib`, `vectorizer.joblib`, dan
`metadata.json`.

## Struktur

```text
configs/         konfigurasi eksperimen
notebooks/       narasi eksplorasi dan eksperimen
src/affective_ml fungsi data, teks, fitur, training, evaluasi, dan ekspor
tests/           unit test
```
