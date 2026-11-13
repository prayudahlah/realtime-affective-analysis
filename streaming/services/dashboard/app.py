from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from components import (
    angka,
    chart_emosi,
    chart_produk,
    chart_sentimen,
    chart_tren,
    inject_css,
    kartu,
    kpi,
    kosong,
    pagination,
    persen,
    rating,
    stats_alert,
    tabel_alert,
    tabel_ulasan,
    topbar,
)
from config import ALL_PRODUCT, AUTO_REFRESH_MS, PAGE_SIZE
from db import (
    get_active_models,
    get_alert_breakdown,
    get_alerts_page,
    get_emotion_distribution,
    get_kpi_metrics,
    get_products,
    get_reviews_page,
    get_sentiment_distribution,
    get_time_series_rating,
    get_top_products_by_rating,
    get_total_alerts_count,
    get_total_reviews_count,
)

PLOT_CONFIG = {"displayModeBar": False}

st.set_page_config(
    page_title="ASUS Realtime Review Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()


def _init_state() -> None:
    st.session_state.setdefault("filter_date_from", date.today() - timedelta(days=30))
    st.session_state.setdefault("filter_date_to", date.today())
    st.session_state.setdefault("filter_product", ALL_PRODUCT)
    st.session_state.setdefault("filter_sentiment", ["Positive", "Negative"])
    st.session_state.setdefault("review_page", 0)
    st.session_state.setdefault("alert_page", 0)
    st.session_state.setdefault("paused", False)
    st.session_state.setdefault("nav_halaman", "Ringkasan")
    st.session_state.setdefault("_last_filter_key", "")


def _reset_filters() -> None:
    st.session_state.filter_date_from = date.today() - timedelta(days=30)
    st.session_state.filter_date_to = date.today()
    st.session_state.filter_product = ALL_PRODUCT
    st.session_state.filter_sentiment = ["Positive", "Negative"]
    st.session_state.review_page = 0
    st.session_state.alert_page = 0


def _sinkron_halaman() -> None:
    st.query_params["halaman"] = st.session_state.nav_halaman


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
    st.session_state.nav_initialized = True

paused = st.session_state["paused"]
st_autorefresh(interval=3_600_000 if paused else AUTO_REFRESH_MS, key="auto_refresh")

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def _tanggal_id(nilai: date) -> str:
    return f"{nilai.day} {BULAN[nilai.month - 1]} {nilai.year}"


sekarang = datetime.now(ZoneInfo("Asia/Jakarta"))
topbar(
    "ASUS Official Shop",
    f"{_tanggal_id(st.session_state.filter_date_from)} s.d. {_tanggal_id(st.session_state.filter_date_to)}",
    "Realtime Review Intelligence",
    sekarang.strftime("%H:%M:%S WIB"),
    "Refresh dijeda" if paused else "Diperbarui otomatis",
    paused=paused,
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
    st.toggle("Jeda refresh", key="paused")

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
        "Tidak ada ulasan untuk kombinasi filter ini. Longgarkan tanggal, produk, atau sentimen."
    )
    st.stop()


def halaman_ringkasan(
    filters: dict, total: int, metrics: dict, model_info: dict, alert_counts: dict
) -> None:
    kolom = st.columns(6, gap="small")
    with kolom[0]:
        kpi("Total ulasan hari ini", angka(metrics["total_today"]),
            sub=f"{metrics['total_delta']:+d} vs kemarin")
    with kolom[1]:
        kpi("Sentimen negatif", persen(metrics["pct_negative"]), merah=True,
            sub=f"{metrics['pct_negative_delta']:+.1f} pp vs kemarin".replace(".", ","))
    with kolom[2]:
        if metrics["avg_rating_now"] is None:
            kpi("Rating rata-rata 10 menit", "—", sub="belum ada data")
        else:
            delta = metrics["avg_rating_delta"]
            sub = "—" if delta is None else f"{delta:+.2f} vs 10 menit sebelumnya".replace(".", ",")
            kpi("Rating rata-rata 10 menit", rating(metrics["avg_rating_now"]), sub=sub)
    with kolom[3]:
        kpi("Alert terkirim", angka(metrics["alert_count"]),
            sub=(f"sentimen {angka(alert_counts.get('sentiment_negative', 0))} · "
                 f"rating {angka(alert_counts.get('rating_drop', 0))}"))
    with kolom[4]:
        info = model_info.get("sentiment", {})
        kpi("F1 sentimen",
            f"{info.get('f1', 0):.4f}".replace(".", ",") if info.get("f1") is not None else "—",
            sub=f"model v{info.get('version', '—')}")
    with kolom[5]:
        info = model_info.get("emotion", {})
        kpi("F1 emosi",
            f"{info.get('f1', 0):.4f}".replace(".", ",") if info.get("f1") is not None else "—",
            sub=f"model v{info.get('version', '—')}")

    df_tren = get_time_series_rating(filters)
    with kartu("Tren Rating", "Rata-rata rating per 10 menit"):
        if df_tren.empty:
            kosong("Belum ada data rating pada rentang ini.")
        else:
            st.plotly_chart(chart_tren(df_tren), width="stretch", config=PLOT_CONFIG)

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
| Total ulasan hari ini | Ulasan yang masuk pada tanggal berjalan, sesuai filter aktif |
| Sentimen negatif | Persentase ulasan dengan sentimen negatif dari seluruh ulasan terfilter |
| Rating rata-rata 10 menit | Rata-rata rating ulasan pada 10 menit terakhir, dibanding 10 menit sebelumnya |
| Alert terkirim | Notifikasi yang dipicu oleh sentimen negatif atau penurunan rating |
| F1 sentimen dan emosi | Macro F1 model aktif pada data uji, diambil dari metadata model |

Sumber data adalah ulasan toko yang dipantau, dialirkan lewat Kafka, diproses dengan Spark Structured Streaming, lalu disimpan di PostgreSQL.
            """
        )


def halaman_ulasan(filters: dict, total: int) -> None:
    df_produk = get_top_products_by_rating(filters)
    with kartu("Produk Rating Terendah", "Lima produk dengan rating rata-rata terendah"):
        if df_produk.empty:
            kosong("Belum ada data produk.")
        else:
            st.plotly_chart(chart_produk(df_produk), width="stretch", config=PLOT_CONFIG)

    halaman = int(st.session_state.get("review_page", 0))
    df_ulasan = get_reviews_page(filters, limit=PAGE_SIZE, offset=halaman * PAGE_SIZE)
    with kartu("Ulasan Terbaru", f"{angka(total)} ulasan pada rentang ini"):
        if df_ulasan.empty:
            kosong("Tidak ada ulasan pada halaman ini.")
        else:
            tabel_ulasan(df_ulasan)
            sebelumnya, berikutnya = pagination("review", halaman, total, PAGE_SIZE)
            if sebelumnya:
                _ganti_halaman("review_page", max(0, halaman - 1))
            if berikutnya:
                _ganti_halaman("review_page", halaman + 1)


def halaman_alert(filters: dict) -> None:
    with kartu("Ringkasan Alert", "Filter produk dan sentimen tidak memengaruhi halaman ini"):
        stats_alert(
            [
                ("Sentimen negatif", angka(alert_counts.get("sentiment_negative", 0)), "ulasan pemicu"),
                ("Penurunan rating", angka(alert_counts.get("rating_drop", 0)), "rata-rata 10 menit di bawah ambang"),
            ]
        )

    total_alert = get_total_alerts_count(filters)
    halaman = int(st.session_state.get("alert_page", 0))
    df_alert = get_alerts_page(filters, limit=PAGE_SIZE, offset=halaman * PAGE_SIZE)
    with kartu("Riwayat Alert", f"{angka(total_alert)} alert pada rentang ini"):
        if df_alert.empty:
            kosong("Belum ada alert pada rentang ini.")
        else:
            tabel_alert(df_alert)
            sebelumnya, berikutnya = pagination("alert", halaman, total_alert, PAGE_SIZE)
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
