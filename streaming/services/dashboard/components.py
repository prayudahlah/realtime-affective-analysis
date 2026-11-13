from __future__ import annotations

import html
from contextlib import contextmanager
from math import ceil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALERT_TYPE_LABELS,
    BLUE,
    EMOTION_COLORS,
    GREEN,
    INK,
    INK_SOFT,
    MUTED,
    RED,
    SENTIMENT_COLORS,
)

CSS = """
<style>
:root {
  --ink: #2F3E4E; --ink-soft: #4A5A6A; --muted: #7A8A9A;
  --line: #C9D6E2; --blue: #2F6DB5; --red: #C0392B;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background-color: #EDF1F6 !important;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
.block-container { padding: 0.7rem 1.4rem 2.5rem; max-width: 1420px; }

.st-key-sheet {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.7rem 0.9rem 0.35rem;
}

.crm-topbar {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1rem;
  background: #DCE6F0;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.5rem 0.85rem;
  margin-bottom: 0.55rem;
}
.crm-topbar .left { font-size: 0.67rem; color: var(--ink-soft); line-height: 1.25; }
.crm-topbar .left b { display: block; font-size: 0.78rem; color: var(--ink); font-weight: 600; }
.crm-topbar .center { font-size: 1.02rem; font-weight: 700; color: var(--ink); }
.crm-topbar .right { text-align: right; font-size: 0.67rem; color: var(--ink-soft); line-height: 1.25; }
.crm-topbar .right b { display: block; font-size: 0.78rem; color: var(--ink); font-weight: 600; }
.crm-topbar .live { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #2E7D4F; margin-right: 5px; vertical-align: middle; }
.crm-topbar .live.warn { background: #C58B00; }
.crm-topbar .live.stale { background: #C0392B; }
.crm-topbar .live.paused { background: #C0392B; }

.kpi-card {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.55rem 0.7rem;
  text-align: center;
  height: 100%;
}
.kpi-label { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.12rem; line-height: 1.2; }
.kpi-value { font-size: 1.4rem; font-weight: 600; color: var(--ink); line-height: 1.15; }
.kpi-value.red { color: var(--red); }
.kpi-sub { font-size: 0.72rem; color: var(--ink-soft); margin-top: 0.18rem; min-height: 1.1em; line-height: 1.3; }

.card-title { font-size: 0.82rem; font-weight: 600; color: var(--ink); margin: 0 0 0.15rem; }
.card-sub { font-size: 0.76rem; color: #5A6B7B; margin: 0 0 0.2rem; }

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF !important;
  border: 1px solid var(--line) !important;
  border-radius: 6px !important;
  padding: 0.5rem 0.7rem 0.35rem !important;
}

.table-wrap { overflow-x: auto; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
.data-table th { text-align: left; font-weight: 600; background: #F4F7FB;
  border-bottom: 1px solid var(--line); padding: 0.42rem 0.5rem; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--ink-soft); white-space: nowrap; }
.data-table td { border-bottom: 1px solid #E3EAF3; padding: 0.42rem 0.5rem;
  color: var(--ink-soft); vertical-align: top; line-height: 1.4; }
.data-table tr:last-child td { border-bottom: none; }
.data-table td.num { text-align: right; white-space: nowrap; color: var(--ink); font-weight: 600; }
.data-table td.nowrap { white-space: nowrap; }
.data-table td.prod { max-width: 190px; }
.data-table td.rev { color: var(--ink); }

.badge { display: inline-block; border-radius: 999px; padding: 1px 8px;
  font-size: 0.68rem; font-weight: 600; white-space: nowrap; }
.badge.pos { background: #E3EEF9; color: #2F6DB5; }
.badge.neg { background: #FBE7E5; color: #C0392B; }
.badge.happy { background: #E3EEF9; color: #2F6DB5; }
.badge.love { background: #E4F1E8; color: #2E7D4F; }
.badge.fear { background: #EDE7F5; color: #7A5AA6; }
.badge.sadness { background: #E3F0F4; color: #4C8FA8; }
.badge.anger { background: #FBE7E5; color: #C0392B; }

.page-info { text-align: center; font-size: 0.74rem; color: var(--muted); padding-top: 0.45rem; }

.alert-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin: 0.2rem 0 0.3rem; }
.alert-stat { background: #F4F7FB; border: 1px solid #E3EAF3; border-radius: 6px; padding: 0.5rem 0.7rem; }
.alert-stat span { display: block; font-size: 0.72rem; color: #7A8A9A; }
.alert-stat b { display: block; font-size: 1.25rem; color: #2F3E4E; line-height: 1.2; }
.alert-stat small { display: block; font-size: 0.7rem; color: #4A5A6A; }

.empty { background: #FFFFFF; border: 1px dashed var(--line); border-radius: 6px;
  padding: 1.6rem 1rem; text-align: center; color: var(--muted); font-size: 0.85rem; }

.def-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.2rem; }
.def-table th { text-align: left; font-weight: 600; background: #F4F7FB;
  border-bottom: 1px solid var(--line); padding: 0.42rem 0.55rem; font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--ink-soft); }
.def-table td { border-bottom: 1px solid #E3EAF3; padding: 0.45rem 0.55rem;
  vertical-align: top; color: var(--ink-soft); line-height: 1.45; }
.def-table tr:last-child td { border-bottom: none; }
.def-table td.metric { color: var(--blue); font-weight: 600; white-space: nowrap; }

div[data-testid="stPills"] button { font-size: 0.78rem; }

@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="column"] { flex: 1 1 46% !important; min-width: 46% !important; }
  div[data-testid="stPills"] > div { flex-wrap: wrap; }
  .crm-topbar { grid-template-columns: 1fr; text-align: center; gap: 0.35rem; }
  .crm-topbar .right { text-align: center; }
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stDataFrame"] { border: none; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def topbar(
    kiri_judul: str,
    kiri_sub: str,
    tengah: str,
    kanan_judul: str,
    kanan_sub: str,
    paused: bool = False,
    status: str = "fresh",
) -> None:
    kelas = f"live {status}"
    st.markdown(
        f"""
        <div class="crm-topbar">
          <div class="left"><b>{html.escape(kiri_judul)}</b>{html.escape(kiri_sub)}</div>
          <div class="center">{html.escape(tengah)}</div>
          <div class="right"><b><span class="{kelas}"></span>{html.escape(kanan_judul)}</b>{html.escape(kanan_sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, merah: bool = False, sub: str = "") -> None:
    kelas = "kpi-value red" if merah else "kpi-value"
    isi_sub = html.escape(sub) if sub else "&nbsp;"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="{kelas}">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{isi_sub}</div></div>',
        unsafe_allow_html=True,
    )


@contextmanager
def kartu(judul: str, sub: str = ""):
    with st.container(border=True):
        sub_html = f'<div class="card-sub">{html.escape(sub)}</div>' if sub else ""
        st.markdown(
            f'<div class="card-title">{html.escape(judul)}</div>{sub_html}',
            unsafe_allow_html=True,
        )
        yield


def angka(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def persen(value: float, desimal: int = 1) -> str:
    return f"{value:.{desimal}f}".replace(".", ",") + "%"


def rating(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def waktu(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%d/%m %H:%M")


def gaya(fig: go.Figure, tinggi: int = 300, legenda: bool = False) -> go.Figure:
    fig.update_layout(
        height=tinggi,
        margin=dict(l=8, r=18, t=8, b=8),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(size=12, color=INK_SOFT, family="Source Sans Pro, system-ui, sans-serif"),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, bordercolor="#C9D6E2"),
        bargap=0.35,
    )
    fig.update_xaxes(
        showline=True,
        linecolor="#D5DFE9",
        tickcolor="#D5DFE9",
        tickfont=dict(size=11, color=INK_SOFT),
        title_font=dict(size=11, color=MUTED),
        gridcolor="#E2EAF3",
        zeroline=False,
    )
    fig.update_yaxes(
        showline=False,
        tickcolor="#D5DFE9",
        tickfont=dict(size=11, color=INK_SOFT),
        title_font=dict(size=11, color=MUTED),
        gridcolor="#E2EAF3",
        zeroline=False,
    )
    return fig


def chart_tren(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["bucket"],
            y=df["avg_rating"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2.4),
            marker=dict(size=5, color=BLUE),
            fill="tozeroy",
            fillcolor="rgba(47,109,181,0.10)",
            hovertemplate="%{x|%d/%m %H:%M}<br>Rating %{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[1, 5], dtick=1)
    fig.update_xaxes(tickformat="%H:%M", nticks=8)
    return gaya(fig, tinggi=290)


def chart_sentimen(df: pd.DataFrame) -> go.Figure:
    raw = df["sentiment"].tolist()
    labels = ["Positif" if label == "Positive" else "Negatif" for label in raw]
    values = df["count"].tolist()
    colors = [SENTIMENT_COLORS.get(label, MUTED) for label in raw]
    total = sum(values)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            sort=False,
            marker=dict(colors=colors),
            textinfo="none",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{angka(total)}</b><br><span style='font-size:11px'>ulasan</span>",
        showarrow=False,
        font=dict(size=18, color=INK),
    )
    fig.update_layout(showlegend=True)
    return gaya(fig, tinggi=260, legenda=True)


def chart_emosi(df: pd.DataFrame) -> go.Figure:
    data = df.sort_values("count", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["count"],
            y=data["emotion"],
            orientation="h",
            marker=dict(color=[EMOTION_COLORS.get(label, MUTED) for label in data["emotion"]]),
            text=data["count"],
            textposition="outside",
            hovertemplate="%{y}: %{x} ulasan<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Jumlah ulasan")
    return gaya(fig, tinggi=260)


def chart_produk(df: pd.DataFrame) -> go.Figure:
    data = df.sort_values("avg_rating", ascending=False)
    labels = [str(name)[:42] for name in data["product_name"]]
    colors = [RED if value < 4 else BLUE for value in data["avg_rating"]]
    fig = go.Figure(
        go.Bar(
            x=data["avg_rating"],
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{rating(v)} · {int(n)}" for v, n in zip(data["avg_rating"], data["total_reviews"])],
            textposition="outside",
            customdata=data["total_reviews"],
            hovertemplate="%{y}<br>Rating %{x:.2f} dari %{customdata} ulasan<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 5.8], dtick=1, title_text="Rating rata-rata")
    return gaya(fig, tinggi=max(220, 42 * len(data)))


def _badge_sentiment(value: str) -> str:
    kelas = "pos" if value == "Positive" else "neg"
    label = "Positif" if value == "Positive" else "Negatif"
    return f'<span class="badge {kelas}">{label}</span>'


def _badge_emotion(value: str) -> str:
    kelas = str(value).lower()
    return f'<span class="badge {kelas}">{html.escape(str(value))}</span>'


def tabel_ulasan(df: pd.DataFrame) -> None:
    baris = []
    for row in df.itertuples(index=False):
        komentar = html.escape(str(row.comment or "")[:220])
        produk = html.escape(str(row.product_name or "—")[:60])
        baris.append(
            "<tr>"
            f"<td class='nowrap'>{waktu(row.create_time)}</td>"
            f"<td class='prod'>{produk}</td>"
            f"<td class='rev'>{komentar}</td>"
            f"<td class='num'>{rating(float(row.rating_star))}</td>"
            f"<td>{_badge_sentiment(row.sentiment)}</td>"
            f"<td>{_badge_emotion(row.emotion)}</td>"
            "</tr>"
        )
    st.markdown(
        "<div class='table-wrap'><table class='data-table'>"
        "<thead><tr><th>Waktu</th><th>Produk</th><th>Ulasan</th><th>Rating</th>"
        "<th>Sentimen</th><th>Emosi</th></tr></thead>"
        f"<tbody>{''.join(baris)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def tabel_alert(df: pd.DataFrame) -> None:
    baris = []
    for row in df.itertuples(index=False):
        tipe = ALERT_TYPE_LABELS.get(row.alert_type, row.alert_type)
        komentar = html.escape(str(row.comment or "")[:220])
        produk = html.escape(str(row.product_name or "—")[:50])
        nilai = "—"
        if row.review_rating is not None and not pd.isna(row.review_rating):
            nilai = rating(float(row.review_rating))
        elif row.rating_avg is not None and not pd.isna(row.rating_avg):
            nilai = "avg " + rating(float(row.rating_avg))
        baris.append(
            "<tr>"
            f"<td class='nowrap'>{waktu(row.triggered_at)}</td>"
            f"<td class='nowrap'>{html.escape(str(tipe))}</td>"
            f"<td class='prod'>{produk}</td>"
            f"<td class='rev'>{komentar}</td>"
            f"<td class='num'>{nilai}</td>"
            "</tr>"
        )
    st.markdown(
        "<div class='table-wrap'><table class='data-table'>"
        "<thead><tr><th>Waktu</th><th>Tipe</th><th>Produk</th><th>Ulasan</th>"
        "<th>Rating</th></tr></thead>"
        f"<tbody>{''.join(baris)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def pagination(key: str, page: int, total: int, size: int) -> tuple[bool, bool]:
    halaman_total = max(1, ceil(total / size)) if total else 1
    kiri, tengah, kanan = st.columns([1, 2, 1])
    with kiri:
        sebelumnya = st.button(
            "Sebelumnya", key=f"{key}_prev", disabled=page <= 0, width="stretch"
        )
    with tengah:
        st.markdown(
            f"<div class='page-info'>Halaman {page + 1} dari {halaman_total}</div>",
            unsafe_allow_html=True,
        )
    with kanan:
        berikutnya = st.button(
            "Berikutnya",
            key=f"{key}_next",
            disabled=page >= halaman_total - 1,
            width="stretch",
        )
    return sebelumnya, berikutnya


def stats_alert(items: list[tuple[str, str, str]]) -> None:
    isi = "".join(
        f"<div class='alert-stat'><span>{html.escape(label)}</span>"
        f"<b>{html.escape(value)}</b><small>{html.escape(sub)}</small></div>"
        for label, value, sub in items
    )
    st.markdown(f"<div class='alert-stats'>{isi}</div>", unsafe_allow_html=True)


def kosong(pesan: str) -> None:
    st.markdown(f"<div class='empty'>{html.escape(pesan)}</div>", unsafe_allow_html=True)
