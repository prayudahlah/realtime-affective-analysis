"""Backend data snapshot untuk demo Streamlit Cloud.

Modul ini meniru API `db.py` tetapi membaca satu file JSON, sehingga dashboard
bisa berjalan tanpa PostgreSQL dan tanpa pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SNAPSHOT_PATH = Path(os.getenv("SNAPSHOT_PATH", BASE / "snapshot.json"))

with SNAPSHOT_PATH.open(encoding="utf-8") as handle:
    _DATA = json.load(handle)

REVIEWS = pd.DataFrame(_DATA["reviews"])
ALERTS = pd.DataFrame(_DATA["alerts"])
MODELS = pd.DataFrame(_DATA["models"])
META = _DATA.get("meta", {})

for column in ("create_time", "processed_at"):
    if column in REVIEWS:
        REVIEWS[column] = pd.to_datetime(REVIEWS[column])

for column in ("triggered_at", "review_created_at"):
    if column in ALERTS:
        ALERTS[column] = pd.to_datetime(ALERTS[column], errors="coerce")

SNAPSHOT_NOW = (
    REVIEWS["create_time"].max()
    if not REVIEWS.empty
    else pd.to_datetime(META.get("snapshot_date"))
)
SNAPSHOT_DATE = SNAPSHOT_NOW.date()
EXPORTED_AT = pd.to_datetime(META.get("exported_at"), errors="coerce")

REVIEW_SORT = {
    "Terbaru": ("create_time", False),
    "Rating terendah": ("rating_star", True),
    "Rating tertinggi": ("rating_star", False),
    "Produk A-Z": ("product_name", True),
}

ALERT_SORT = {
    "Terbaru": ("triggered_at", False),
    "Terlama": ("triggered_at", True),
}


def _filter_reviews(filters: dict) -> pd.DataFrame:
    frame = REVIEWS
    if frame.empty:
        return frame
    if filters.get("date_from"):
        frame = frame[frame["create_time"] >= pd.Timestamp(filters["date_from"])]
    if filters.get("date_to"):
        end = pd.Timestamp(filters["date_to"]) + pd.Timedelta(days=1)
        frame = frame[frame["create_time"] < end]
    if filters.get("product"):
        frame = frame[frame["product_name"] == filters["product"]]
    if filters.get("sentiment"):
        frame = frame[frame["sentiment"].isin(list(filters["sentiment"]))]
    return frame


def _filter_alerts(filters: dict) -> pd.DataFrame:
    frame = ALERTS
    if frame.empty:
        return frame
    if filters.get("date_from"):
        frame = frame[frame["triggered_at"] >= pd.Timestamp(filters["date_from"])]
    if filters.get("date_to"):
        end = pd.Timestamp(filters["date_to"]) + pd.Timedelta(days=1)
        frame = frame[frame["triggered_at"] < end]
    if filters.get("product"):
        mask = frame["review_id"].isna() | (frame["product_name"] == filters["product"])
        frame = frame[mask]
    if filters.get("sentiment"):
        mask = frame["review_id"].isna() | frame["review_sentiment"].isin(
            list(filters["sentiment"])
        )
        frame = frame[mask]
    return frame


def get_kpi_metrics(filters: dict) -> dict:
    reviews = _filter_reviews(filters)
    alerts = _filter_alerts(filters)

    total_range = int(len(reviews))
    neg_range = int((reviews["sentiment"] == "Negative").sum())
    total_today = int((reviews["create_time"].dt.date == SNAPSHOT_DATE).sum())

    if reviews.empty:
        bucket = None
        avg_now = None
        avg_prev = None
    else:
        buckets = (
            reviews.assign(bucket=reviews["create_time"].dt.floor("10min"))
            .groupby("bucket")["rating_star"]
            .mean()
            .sort_index()
        )
        bucket = buckets.index[-1]
        avg_now = round(float(buckets.iloc[-1]), 2)
        avg_prev = round(float(buckets.iloc[-2]), 2) if len(buckets) > 1 else None

    active = MODELS[MODELS["is_active"].astype(bool)] if not MODELS.empty else MODELS
    f1_sentiment = None
    f1_emotion = None
    for row in active.itertuples(index=False):
        if row.task_type == "sentiment":
            f1_sentiment = float(row.f1_score_macro)
        elif row.task_type == "emotion":
            f1_emotion = float(row.f1_score_macro)

    pct_negative = (neg_range / total_range * 100) if total_range else 0.0

    return {
        "total_range": total_range,
        "neg_range": neg_range,
        "total_today": total_today,
        "pct_negative": pct_negative,
        "avg_rating_now": avg_now,
        "avg_rating_prev": avg_prev,
        "avg_rating_delta": (
            round(avg_now - avg_prev, 2)
            if avg_now is not None and avg_prev is not None
            else None
        ),
        "avg_rating_bucket": bucket,
        "alert_count": int(len(alerts)),
        "f1_sentiment": f1_sentiment,
        "f1_emotion": f1_emotion,
    }


ALLOWED_BUCKETS = {
    "1 minute": "1min",
    "10 minutes": "10min",
    "1 hour": "1h",
    "1 day": "1D",
}


def get_time_series_rating(
    filters: dict, bucket: str = "10 minutes", since_minutes: int | None = None
) -> pd.DataFrame:
    if bucket not in ALLOWED_BUCKETS:
        raise ValueError(f"Bucket tidak dikenal: {bucket}")
    frame = _filter_reviews(filters)
    if since_minutes:
        frame = frame[
            frame["create_time"] >= SNAPSHOT_NOW - pd.Timedelta(minutes=int(since_minutes))
        ]
    columns = ["bucket", "avg_rating", "review_count"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.assign(bucket=frame["create_time"].dt.floor(ALLOWED_BUCKETS[bucket]))
        .groupby("bucket")
        .agg(avg_rating=("rating_star", "mean"), review_count=("rating_star", "size"))
        .reset_index()
        .sort_values("bucket")
    )
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    return grouped[columns]


def get_sentiment_distribution(filters: dict) -> pd.DataFrame:
    frame = _filter_reviews(filters)
    if frame.empty:
        return pd.DataFrame(columns=["sentiment", "count"])
    return (
        frame.groupby("sentiment").size().reset_index(name="count")
    )


def get_emotion_distribution(filters: dict) -> pd.DataFrame:
    frame = _filter_reviews(filters)
    if frame.empty:
        return pd.DataFrame(columns=["emotion", "count"])
    return (
        frame.groupby("emotion")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def get_top_products_by_rating(
    filters: dict, limit: int = 5, min_reviews: int = 2
) -> pd.DataFrame:
    frame = _filter_reviews(filters)
    columns = ["product_name", "avg_rating", "total_reviews", "neg_count"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.groupby("product_name")
        .agg(
            avg_rating=("rating_star", "mean"),
            total_reviews=("rating_star", "size"),
            neg_count=("sentiment", lambda values: (values == "Negative").sum()),
        )
        .reset_index()
    )
    grouped = grouped[grouped["total_reviews"] >= min_reviews]
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    grouped = grouped.sort_values("avg_rating").head(limit)
    return grouped[columns]


def _table_filters(
    frame: pd.DataFrame, search: str, emotions: list[str] | None
) -> pd.DataFrame:
    if search:
        mask = frame["comment"].str.contains(search, case=False, na=False) | frame[
            "product_name"
        ].str.contains(search, case=False, na=False)
        frame = frame[mask]
    if emotions:
        frame = frame[frame["emotion"].isin(list(emotions))]
    return frame


def get_reviews_page(
    filters: dict,
    limit: int = 10,
    offset: int = 0,
    search: str = "",
    emotions: list[str] | None = None,
    sort: str = "Terbaru",
) -> pd.DataFrame:
    frame = _table_filters(_filter_reviews(filters), search, emotions)
    columns = ["create_time", "product_name", "comment", "rating_star", "sentiment", "emotion"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    kolom, naik = REVIEW_SORT.get(sort, REVIEW_SORT["Terbaru"])
    if kolom == "create_time":
        frame = frame.sort_values("create_time", ascending=naik)
    else:
        frame = frame.sort_values([kolom, "create_time"], ascending=[naik, False])
    return frame.iloc[offset : offset + limit][columns]


def get_total_reviews_count(
    filters: dict, search: str = "", emotions: list[str] | None = None
) -> int:
    return int(len(_table_filters(_filter_reviews(filters), search, emotions)))


def get_alerts_page(
    filters: dict,
    limit: int = 10,
    offset: int = 0,
    search: str = "",
    types: list[str] | None = None,
    sort: str = "Terbaru",
) -> pd.DataFrame:
    frame = _filter_alerts(filters)
    columns = [
        "triggered_at",
        "alert_type",
        "comment",
        "rating_avg",
        "product_name",
        "review_created_at",
        "review_rating",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if search:
        mask = frame["comment"].str.contains(search, case=False, na=False) | frame[
            "product_name"
        ].fillna("").str.contains(search, case=False, na=False)
        frame = frame[mask]
    if types:
        frame = frame[frame["alert_type"].isin(list(types))]
    kolom, naik = ALERT_SORT.get(sort, ALERT_SORT["Terbaru"])
    frame = frame.sort_values([kolom, "id"], ascending=[naik, False])
    return frame.iloc[offset : offset + limit][columns]


def get_total_alerts_count(
    filters: dict, search: str = "", types: list[str] | None = None
) -> int:
    frame = _filter_alerts(filters)
    if search:
        mask = frame["comment"].str.contains(search, case=False, na=False) | frame[
            "product_name"
        ].fillna("").str.contains(search, case=False, na=False)
        frame = frame[mask]
    if types:
        frame = frame[frame["alert_type"].isin(list(types))]
    return int(len(frame))


def get_products() -> list[str]:
    return sorted(
        str(product)
        for product in REVIEWS["product_name"].dropna().unique().tolist()
    )


def get_active_models() -> dict:
    result = {}
    for row in MODELS.itertuples(index=False):
        if bool(row.is_active):
            result[row.task_type] = {
                "version": row.version,
                "f1": float(row.f1_score_macro),
            }
    return result


def get_alert_breakdown(filters: dict) -> dict:
    frame = _filter_alerts(filters)
    if frame.empty:
        return {}
    return {
        str(alert_type): int(total)
        for alert_type, total in frame.groupby("alert_type").size().items()
    }


def get_data_freshness() -> dict:
    return {
        "processed_at": REVIEWS["processed_at"].max() if not REVIEWS.empty else None,
        "create_time": SNAPSHOT_NOW,
        "exported_at": EXPORTED_AT,
        "snapshot_date": SNAPSHOT_DATE,
    }


def get_time_bounds(filters: dict) -> tuple:
    frame = _filter_reviews(filters)
    if frame.empty:
        return None, None
    return frame["create_time"].min(), frame["create_time"].max()
