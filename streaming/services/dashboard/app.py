import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_autorefresh import st_autorefresh

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODE = os.getenv("DASHBOARD_MODE", "").strip().lower()
if MODE not in {"live", "snapshot"}:
    MODE = "live" if (os.getenv("INFERENCE_DB_HOST") or os.getenv("DATABASE_URL")) else "snapshot"

from components import (
    angka,
    chart_emosi,
    chart_produk,
    chart_sentimen,
    chart_tren,
    dataframe_alert,
    dataframe_ulasan,
    delta_html,
    inject_css,
    kartu,
    kpi,
    kosong,
    pagination,
    persen,
    rating,
    stats_alert,
    topbar,
)
from config import ALL_PRODUCT, AUTO_REFRESH_MS, EMOTIONS, PAGE_SIZE
if MODE == "snapshot":
    from db_snapshot import (
        ALERT_SORT,
        REVIEW_SORT,
        SNAPSHOT_DATE,
        get_active_models,
        get_alert_breakdown,
        get_alerts_page,
        get_data_freshness,
        get_emotion_distribution,
        get_kpi_metrics,
        get_products,
        get_reviews_page,
        get_sentiment_distribution,
        get_time_bounds,
        get_time_series_rating,
        get_top_products_by_rating,
        get_total_alerts_count,
        get_total_reviews_count,
    )
else:
    from db import (
        ALERT_SORT,
        REVIEW_SORT,
        get_active_models,
        get_alert_breakdown,
        get_alerts_page,
        get_data_freshness,
        get_emotion_distribution,
        get_kpi_metrics,
        get_products,
        get_reviews_page,
        get_sentiment_distribution,
        get_time_bounds,
        get_time_series_rating,
        get_top_products_by_rating,
        get_total_alerts_count,
        get_total_reviews_count,
    )

    SNAPSHOT_DATE = None

PLOT_CONFIG = {"displayModeBar": False}

st.set_page_config(
    page_title="ASUS Realtime Review Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()


if MODE == "snapshot" and SNAPSHOT_DATE is not None:
    DEFAULT_FROM = SNAPSHOT_DATE - timedelta(days=30)
    DEFAULT_TO = SNAPSHOT_DATE
else:
    DEFAULT_FROM = date.today() - timedelta(days=30)
    DEFAULT_TO = date.today()


def _init_state() -> None:
    st.session_state.setdefault("filter_date_from", DEFAULT_FROM)
    st.session_state.setdefault("filter_date_to", DEFAULT_TO)
    st.session_state.setdefault("filter_product", ALL_PRODUCT)
    st.session_state.setdefault("filter_sentiment", ["Positive", "Negative"])
    st.session_state.setdefault("review_page", 0)
    st.session_state.setdefault("alert_page", 0)
    st.session_state.setdefault("paused", MODE == "snapshot")
    st.session_state.setdefault("nav_halaman", "Ringkasan")
    st.session_state.setdefault("_last_filter_key", "")


def _reset_filters() -> None:
    st.session_state.filter_date_from = DEFAULT_FROM
    st.session_state.filter_date_to = DEFAULT_TO
    st.session_state.filter_product = ALL_PRODUCT
    st.session_state.filter_sentiment = ["Positive", "Negative"]
    st.session_state.review_page = 0
    st.session_state.alert_page = 0


def _sinkron_halaman() -> None:
    if st.session_state.get("nav_halaman"):
        st.query_params["halaman"] = st.session_state.nav_halaman


def _sinkron_jeda() -> None:
    st.query_params["jeda"] = "1" if st.session_state.paused else "0"


def _ensure_sentiment() -> None:
    if not st.session_state.filter_sentiment:
        st.session_state.filter_sentiment = ["Positive", "Negative"]


def _ganti_halaman(kunci: str, nilai: int) -> None:
    st.session_state[kunci] = nilai
    st.rerun()


_init_state()
if "nav_initialized" not in st.session_state:
    awal = st.query_params.get("halaman")
    if awal in ("Ringkasan", "Ulasan", "Alert"):
        st.session_state.nav_halaman = awal
    if st.query_params.get("jeda") == "1":
        st.session_state.paused = True
    st.session_state.nav_initialized = True

paused = st.session_state["paused"]
if MODE != "snapshot":
    st_autorefresh(interval=3_600_000 if paused else AUTO_REFRESH_MS, key="auto_refresh")

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
ALERT_LABELS = {
    "Sentimen Negatif": "sentiment_negative",
    "Penurunan Rating": "rating_drop",
}


def _tanggal_id(nilai: date) -> str:
    return f"{nilai.day} {BULAN[nilai.month - 1]} {nilai.year}"


sekarang = datetime.now(ZoneInfo("Asia/Jakarta"))
if MODE == "snapshot":
    info = get_data_freshness()
    tanggal_snapshot = info.get("snapshot_date") or SNAPSHOT_DATE
    status_data = "fresh"
    kanan_judul = f"Snapshot {_tanggal_id(tanggal_snapshot)}"
    kanan_sub = "Data statis"
else:
    freshness = get_data_freshness()
    processed_at = freshness.get("processed_at")
    if processed_at is not None:
        if processed_at.tzinfo is None:
            processed_at = processed_at.replace(tzinfo=ZoneInfo("Asia/Jakarta"))
        lag_menit = max(0, int((sekarang - processed_at.astimezone(ZoneInfo("Asia/Jakarta"))).total_seconds() // 60))
        status_data = "fresh" if lag_menit <= 5 else "warn" if lag_menit <= 15 else "stale"
        kanan_judul = f"Data terakhir {processed_at.astimezone(ZoneInfo('Asia/Jakarta')):%H:%M}"
        kanan_sub = "Refresh dijeda" if paused else f"{lag_menit} menit lalu"
    else:
        status_data = "stale"
        kanan_judul = "Belum ada data"
        kanan_sub = "Refresh dijeda" if paused else "Menunggu pipeline"

topbar(
    "ASUS Official Shop",
    f"{_tanggal_id(st.session_state.filter_date_from)} s.d. {_tanggal_id(st.session_state.filter_date_to)}",
    "Realtime Review Intelligence",
    kanan_judul,
    kanan_sub,
    paused=paused if MODE == "live" else False,
    status=status_data,
)

if MODE == "snapshot":
    st.markdown(
        "<div class='demo-banner'>Demo snapshot, data per "
        f"{_tanggal_id(SNAPSHOT_DATE)}. Pipeline lengkap berjalan lokal.</div>",
        unsafe_allow_html=True,
    )

navigasi, kontrol = st.columns([5, 1], vertical_alignment="center")
with navigasi:
    halaman = st.pills(
        "Halaman",
        ["Ringkasan", "Ulasan", "Alert"],
        label_visibility="collapsed",
        key="nav_halaman",
        on_change=_sinkron_halaman,
    )
with kontrol:
    if MODE == "snapshot":
        st.markdown(
            "<div class='snapshot-badge'>Mode demo</div>", unsafe_allow_html=True
        )
    else:
        st.toggle("Jeda refresh", key="paused", on_change=_sinkron_jeda)

with st.container(key="sheet"):
    c1, c2, c3, c4, c5 = st.columns([1.0, 1.0, 1.7, 1.2, 0.7], vertical_alignment="bottom")
    with c1:
        st.date_input("Dari", key="filter_date_from")
    with c2:
        st.date_input("Sampai", key="filter_date_to")
    with c3:
        produk = get_products()
        st.selectbox(
            "Produk",
            options=[ALL_PRODUCT, *produk],
            format_func=lambda nilai: "Semua Produk" if nilai == ALL_PRODUCT else nilai,
            key="filter_product",
        )
    with c4:
        st.pills(
            "Sentimen",
            ["Positive", "Negative"],
            selection_mode="multi",
            format_func=lambda nilai: "Positif" if nilai == "Positive" else "Negatif",
            key="filter_sentiment",
            on_change=_ensure_sentiment,
        )
    with c5:
        st.button("Reset", key="btn_reset", on_click=_reset_filters, width="stretch")

produk_terpilih = st.session_state.get("filter_product", ALL_PRODUCT)
sentimen_terpilih = st.session_state.get("filter_sentiment", ["Positive", "Negative"])
filters = {
    "date_from": st.session_state.get("filter_date_from"),
    "date_to": st.session_state.get("filter_date_to"),
    "product": None if produk_terpilih == ALL_PRODUCT else produk_terpilih,
    "sentiment": sentimen_terpilih,
}

if st.session_state.filter_date_from > st.session_state.filter_date_to:
    kosong(
        "Tanggal awal melewati tanggal akhir. Perbaiki rentang atau tekan Reset.",
        kotak=True,
    )
    st.button("Reset filter", key="reset_range", on_click=_reset_filters)
    st.stop()

filter_key = str(filters)
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state.review_page = 0
    st.session_state.alert_page = 0
    st.session_state["_last_filter_key"] = filter_key

metrics = get_kpi_metrics(filters)
model_info = get_active_models()
alert_counts = get_alert_breakdown(filters)
total_ulasan = get_total_reviews_count(filters)

if total_ulasan == 0:
    kosong(
        "Tidak ada ulasan untuk kombinasi filter ini. Longgarkan tanggal, produk, atau sentimen.",
        kotak=True,
    )
    st.button("Reset filter", key="reset_empty", on_click=_reset_filters)
    st.caption(
        "Filter aktif: "
        + "; ".join(
            [
                f"Periode {_tanggal_id(st.session_state.filter_date_from)} s.d. {_tanggal_id(st.session_state.filter_date_to)}",
                f"Produk {produk_terpilih if produk_terpilih != ALL_PRODUCT else 'semua'}",
                "Sentimen " + ", ".join("Positif" if v == "Positive" else "Negatif" for v in sentimen_terpilih),
            ]
        )
    )
    st.stop()


def halaman_ringkasan(
    filters: dict, total: int, metrics: dict, model_info: dict, alert_counts: dict
) -> None:
    kolom = st.columns(6, gap="small")
    with kolom[0]:
        sub_total = (
            "seluruhnya hari ini"
            if metrics["total_today"] == metrics["total_range"]
            else f"{angka(metrics['total_today'])} hari ini"
        )
        kpi("Total ulasan", angka(metrics["total_range"]), sub=sub_total)
    with kolom[1]:
        kpi("Sentimen negatif", persen(metrics["pct_negative"]), merah=True,
            sub=f"{angka(metrics['neg_range'])} dari {angka(metrics['total_range'])} ulasan")
    with kolom[2]:
        if metrics["avg_rating_now"] is None:
            kpi("Rating terakhir", "—", sub="belum ada data")
        else:
            delta = metrics["avg_rating_delta"]
            bucket = metrics.get("avg_rating_bucket")
            waktu_bucket = f"bucket {bucket:%H:%M}" if bucket is not None else "bucket terakhir"
            sub = waktu_bucket if delta is None else f"{waktu_bucket}, {delta_html(delta)}"
            kpi(
                "Rating terakhir",
                rating(metrics["avg_rating_now"]),
                sub=sub,
                sub_html=True,
            )
    with kolom[3]:
        kpi("Alert terkirim", angka(metrics["alert_count"]),
            sub=(f"sentimen {angka(alert_counts.get('sentiment_negative', 0))}, "
                 f"rating {angka(alert_counts.get('rating_drop', 0))}"))
    with kolom[4]:
        info = model_info.get("sentiment", {})
        kpi("F1 sentimen",
            f"{info.get('f1', 0):.4f}".replace(".", ",") if info.get("f1") is not None else "—",
            sub=f"macro F1, v{info.get('version', '—')}")
    with kolom[5]:
        info = model_info.get("emotion", {})
        kpi("F1 emosi",
            f"{info.get('f1', 0):.4f}".replace(".", ",") if info.get("f1") is not None else "—",
            sub=f"macro F1, v{info.get('version', '—')}")

    st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)

    awal_data, akhir_data = get_time_bounds(filters)
    if awal_data is None or akhir_data is None:
        rentang_hari = 0
    else:
        rentang_hari = max(0, (akhir_data - awal_data).days)
    if rentang_hari <= 1:
        bucket, label_bucket, tick = "10 minutes", "10 menit", "%H:%M"
    elif rentang_hari <= 7:
        bucket, label_bucket, tick = "1 hour", "jam", "%d/%m %H:%M"
    else:
        bucket, label_bucket, tick = "1 day", "hari", "%d/%m"

    kiri_tren, kanan_tren = st.columns([2, 1])
    with kiri_tren:
        with kartu("Tren Rating", f"Rata-rata per {label_bucket}"):
            df_tren = get_time_series_rating(filters, bucket=bucket)
            if df_tren.empty:
                kosong("Belum ada data rating pada rentang ini.")
            else:
                st.plotly_chart(
                    chart_tren(df_tren, tickformat=tick),
                    width="stretch",
                    config=PLOT_CONFIG,
                )
    with kanan_tren:
        with kartu("10 Menit Terakhir", "Rata-rata per menit"):
            df_recent = get_time_series_rating(
                filters, bucket="1 minute", since_minutes=10
            )
            if df_recent.empty:
                kosong("Belum ada ulasan dalam 10 menit terakhir.")
            else:
                st.plotly_chart(
                    chart_tren(df_recent, tickformat="%H:%M"),
                    width="stretch",
                    config=PLOT_CONFIG,
                )

    kiri, kanan = st.columns(2)
    with kiri:
        with kartu("Distribusi Sentimen", f"{angka(total)} ulasan pada rentang ini"):
            df_sentimen = get_sentiment_distribution(filters)
            if df_sentimen.empty:
                kosong("Belum ada data sentimen.")
            else:
                st.plotly_chart(
                    chart_sentimen(df_sentimen), width="stretch", config=PLOT_CONFIG
                )
    with kanan:
        with kartu("Distribusi Emosi", "Lima kategori emosi"):
            df_emosi = get_emotion_distribution(filters)
            if df_emosi.empty:
                kosong("Belum ada data emosi.")
            else:
                st.plotly_chart(
                    chart_emosi(df_emosi), width="stretch", config=PLOT_CONFIG
                )

    with st.expander("Definisi metrik dan sumber data"):
        st.markdown(
            """
| Metrik | Arti |
|---|---|
| Total ulasan | Ulasan pada rentang filter aktif, dengan jumlah hari ini sebagai keterangan |
| Sentimen negatif | Persentase ulasan negatif dari seluruh ulasan pada rentang filter |
| Rating terakhir | Rata-rata rating pada bucket 10 menit terakhir yang memiliki data, dibanding bucket sebelumnya |
| Alert terkirim | Notifikasi yang dipicu oleh sentimen negatif atau penurunan rating |
| F1 sentimen dan emosi | Macro F1 model aktif pada data uji, diambil dari metadata model |

Sumber data: ulasan toko ASUS di Shopee.
            """
        )


def halaman_ulasan(filters: dict, total: int) -> None:
    df_produk = get_top_products_by_rating(filters, min_reviews=2)
    with kartu(
        "Produk Rating Terendah",
        "Minimal 2 ulasan, lima produk dengan rating rata-rata terendah",
    ):
        if df_produk.empty:
            kosong("Belum ada produk dengan minimal 2 ulasan pada rentang ini.")
        else:
            st.plotly_chart(chart_produk(df_produk), width="stretch", config=PLOT_CONFIG)

    with kartu("Ulasan Terbaru", f"{angka(total)} ulasan pada rentang ini"):
        k1, k2, k3, k4 = st.columns(
            [1.8, 1.3, 1.1, 0.6], vertical_alignment="bottom"
        )
        with k1:
            cari = st.text_input(
                "Cari",
                placeholder="Ulasan atau produk",
                key="review_search",
            )
        with k2:
            emosi = st.multiselect(
                "Emosi",
                EMOTIONS,
                key="review_emotion",
                placeholder="Semua emosi",
            )
        with k3:
            urut = st.selectbox("Urutkan", list(REVIEW_SORT), key="review_sort")
        with k4:
            if st.button("Reset", key="review_table_reset", width="stretch"):
                st.session_state.review_search = ""
                st.session_state.review_emotion = []
                st.session_state.review_sort = "Terbaru"
                st.session_state.review_page = 0
                st.rerun()

        kunci = f"{cari}|{','.join(sorted(emosi))}|{urut}"
        if st.session_state.get("_review_table_key") != kunci:
            st.session_state.review_page = 0
            st.session_state["_review_table_key"] = kunci

        halaman = int(st.session_state.get("review_page", 0))
        total_tabel = get_total_reviews_count(filters, search=cari, emotions=emosi)
        df_ulasan = get_reviews_page(
            filters,
            limit=PAGE_SIZE,
            offset=halaman * PAGE_SIZE,
            search=cari,
            emotions=emosi,
            sort=urut,
        )
        if total_tabel == 0:
            kosong("Tidak ada ulasan yang cocok dengan pencarian atau emosi.")
        else:
            dataframe_ulasan(df_ulasan)
            sebelumnya, berikutnya = pagination(
                "review", halaman, total_tabel, PAGE_SIZE
            )
            if sebelumnya:
                _ganti_halaman("review_page", max(0, halaman - 1))
            if berikutnya:
                _ganti_halaman("review_page", halaman + 1)


def halaman_alert(filters: dict) -> None:
    with kartu(
        "Ringkasan Alert",
        "Alert penurunan rating tidak terkait ulasan tertentu, jadi selalu dihitung",
    ):
        stats_alert(
            [
                ("Sentimen negatif", angka(alert_counts.get("sentiment_negative", 0)), "ulasan pemicu"),
                ("Penurunan rating", angka(alert_counts.get("rating_drop", 0)), "rata-rata 10 menit di bawah ambang"),
            ]
        )

    total_alert = get_total_alerts_count(filters)
    with kartu("Riwayat Alert", f"{angka(total_alert)} alert pada rentang ini"):
        k1, k2, k3, k4 = st.columns(
            [1.8, 1.3, 1.1, 0.6], vertical_alignment="bottom"
        )
        with k1:
            cari = st.text_input(
                "Cari",
                placeholder="Ulasan atau produk",
                key="alert_search",
            )
        with k2:
            tipe_label = st.multiselect(
                "Tipe",
                list(ALERT_LABELS),
                key="alert_type",
                placeholder="Semua tipe",
            )
        with k3:
            urut = st.selectbox("Urutkan", list(ALERT_SORT), key="alert_sort")
        with k4:
            if st.button("Reset", key="alert_table_reset", width="stretch"):
                st.session_state.alert_search = ""
                st.session_state.alert_type = []
                st.session_state.alert_sort = "Terbaru"
                st.session_state.alert_page = 0
                st.rerun()

        tipe = [ALERT_LABELS[label] for label in tipe_label]
        kunci = f"{cari}|{','.join(sorted(tipe))}|{urut}"
        if st.session_state.get("_alert_table_key") != kunci:
            st.session_state.alert_page = 0
            st.session_state["_alert_table_key"] = kunci

        halaman = int(st.session_state.get("alert_page", 0))
        total_tabel = get_total_alerts_count(filters, search=cari, types=tipe)
        df_alert = get_alerts_page(
            filters,
            limit=PAGE_SIZE,
            offset=halaman * PAGE_SIZE,
            search=cari,
            types=tipe,
            sort=urut,
        )
        if total_tabel == 0:
            kosong("Tidak ada alert yang cocok dengan pencarian atau tipe.")
        else:
            dataframe_alert(df_alert)
            sebelumnya, berikutnya = pagination(
                "alert", halaman, total_tabel, PAGE_SIZE
            )
            if sebelumnya:
                _ganti_halaman("alert_page", max(0, halaman - 1))
            if berikutnya:
                _ganti_halaman("alert_page", halaman + 1)


if halaman == "Ulasan":
    halaman_ulasan(filters, total_ulasan)
elif halaman == "Alert":
    halaman_alert(filters)
else:
    halaman_ringkasan(filters, total_ulasan, metrics, model_info, alert_counts)
