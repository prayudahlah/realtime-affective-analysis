import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

DB_USER     = os.getenv("INFERENCE_DB_USER",     "postgres")
DB_PASSWORD = os.getenv("INFERENCE_DB_PASSWORD", "postgres")
DB_HOST     = os.getenv("INFERENCE_DB_HOST",     "postgres")
DB_PORT     = int(os.getenv("INFERENCE_DB_HOST_PORT", "5432"))
DB_NAME     = os.getenv("INFERENCE_DB_NAME",     "postgres")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)


def _build_review_filter(filters: dict) -> tuple[str, dict]:
    conditions = []
    params = {}

    if filters.get("date_from"):
        conditions.append("create_time >= :date_from")
        params["date_from"] = filters["date_from"]

    if filters.get("date_to"):
        conditions.append("create_time < :date_to + INTERVAL '1 day'")
        params["date_to"] = filters["date_to"]

    if filters.get("product"):
        conditions.append("product_name = :product")
        params["product"] = filters["product"]

    if filters.get("sentiment"):
        conditions.append("sentiment = ANY(:sentiment)")
        params["sentiment"] = list(filters["sentiment"])

    clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return clause, params


def _build_alert_filter(filters: dict, date_column: str = "triggered_at") -> tuple[str, dict]:
    conditions = []
    params = {}

    if filters.get("date_from"):
        conditions.append(f"{date_column} >= :date_from")
        params["date_from"] = filters["date_from"]

    if filters.get("date_to"):
        conditions.append(f"{date_column} < :date_to + INTERVAL '1 day'")
        params["date_to"] = filters["date_to"]

    if filters.get("product"):
        conditions.append("(a.review_id IS NULL OR r.product_name = :product)")
        params["product"] = filters["product"]

    if filters.get("sentiment"):
        conditions.append("(a.review_id IS NULL OR r.sentiment = ANY(:sentiment))")
        params["sentiment"] = list(filters["sentiment"])

    clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return clause, params


@st.cache_data(ttl=10)
def get_kpi_metrics(filters: dict) -> dict:
    where, params = _build_review_filter(filters)
    alert_where, alert_params = _build_alert_filter(filters)
    sql = f"""
        WITH filtered AS (
            SELECT rating_star, sentiment, create_time
            FROM reviews
            {where}
        ),
        range_stats AS (
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg,
                   COUNT(*) FILTER (WHERE create_time::date = CURRENT_DATE) AS today
            FROM filtered
        ),
        buckets AS (
            SELECT date_bin('10 minutes', create_time, TIMESTAMP 'epoch') AS bucket,
                   AVG(rating_star) AS avg_rating
            FROM filtered
            GROUP BY 1
        ),
        latest AS (SELECT bucket, avg_rating FROM buckets ORDER BY bucket DESC LIMIT 1),
        prev AS (SELECT avg_rating FROM buckets ORDER BY bucket DESC OFFSET 1 LIMIT 1),
        alerts_filtered AS (
            SELECT COUNT(*) AS cnt
            FROM alerts a
            LEFT JOIN reviews r ON a.review_id = r.id
            {alert_where}
        ),
        active_models AS (
            SELECT task_type, f1_score_macro
            FROM model_metadata
            WHERE is_active = TRUE
        )
        SELECT
            (SELECT total FROM range_stats)                                            AS total_range,
            (SELECT neg FROM range_stats)                                              AS neg_range,
            (SELECT today FROM range_stats)                                            AS total_today,
            (SELECT bucket FROM latest)                                                AS bucket,
            (SELECT avg_rating FROM latest)                                            AS avg_now,
            (SELECT avg_rating FROM prev)                                              AS avg_prev,
            (SELECT cnt FROM alerts_filtered)                                          AS alert_count,
            (SELECT f1_score_macro FROM active_models WHERE task_type = 'sentiment')   AS f1_sentiment,
            (SELECT f1_score_macro FROM active_models WHERE task_type = 'emotion')     AS f1_emotion
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {**params, **alert_params}).fetchone()

    total_range = row[0] or 0
    neg_range = row[1] or 0
    total_today = row[2] or 0
    bucket = row[3]
    avg_now = float(row[4]) if row[4] is not None else None
    avg_prev = float(row[5]) if row[5] is not None else None
    alert_count = row[6] or 0
    f1_sentiment = float(row[7]) if row[7] is not None else None
    f1_emotion = float(row[8]) if row[8] is not None else None

    pct_negative = (neg_range / total_range * 100) if total_range > 0 else 0.0

    return {
        "total_range": total_range,
        "neg_range": neg_range,
        "total_today": total_today,
        "pct_negative": pct_negative,
        "avg_rating_now": avg_now,
        "avg_rating_prev": avg_prev,
        "avg_rating_delta": round(avg_now - avg_prev, 2) if avg_now is not None and avg_prev is not None else None,
        "avg_rating_bucket": bucket,
        "alert_count": alert_count,
        "f1_sentiment": f1_sentiment,
        "f1_emotion": f1_emotion,
    }


@st.cache_data(ttl=10)
def get_time_series_rating(filters: dict) -> pd.DataFrame:
    where, params = _build_review_filter(filters)
    sql = f"""
        SELECT
            date_bin('10 minutes', create_time, TIMESTAMP 'epoch') AS bucket,
            ROUND(AVG(rating_star)::numeric, 2)                    AS avg_rating,
            COUNT(*)                                               AS review_count
        FROM reviews
        {where}
        GROUP BY bucket
        ORDER BY bucket ASC
    """
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["bucket", "avg_rating", "review_count"],
        )
    df["avg_rating"] = df["avg_rating"].astype(float)
    return df


@st.cache_data(ttl=10)
def get_sentiment_distribution(filters: dict) -> pd.DataFrame:
    where, params = _build_review_filter(filters)
    sql = f"""
        SELECT sentiment, COUNT(*) AS count
        FROM reviews
        {where}
        GROUP BY sentiment
    """
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["sentiment", "count"],
        )
    return df



@st.cache_data(ttl=10)
def get_emotion_distribution(filters: dict) -> pd.DataFrame:
    where, params = _build_review_filter(filters)
    sql = f"""
        SELECT emotion, COUNT(*) AS count
        FROM reviews
        {where}
        GROUP BY emotion
        ORDER BY count DESC
    """
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["emotion", "count"],
        )
    return df



@st.cache_data(ttl=10)
def get_top_products_by_rating(
    filters: dict, limit: int = 5, min_reviews: int = 2
) -> pd.DataFrame:
    where, params = _build_review_filter(filters)
    sql = f"""
        SELECT
            product_name,
            ROUND(AVG(rating_star)::numeric, 2) AS avg_rating,
            COUNT(*)                             AS total_reviews,
            COUNT(*) FILTER (WHERE sentiment = 'Negative') AS neg_count
        FROM reviews
        {where}
        GROUP BY product_name
        HAVING COUNT(*) >= :min_reviews
        ORDER BY avg_rating ASC
        LIMIT :limit
    """
    params["limit"] = limit
    params["min_reviews"] = min_reviews
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["product_name", "avg_rating", "total_reviews", "neg_count"],
        )
    df["avg_rating"] = df["avg_rating"].astype(float)
    return df



def get_reviews_page(filters: dict, limit: int = 5, offset: int = 0) -> pd.DataFrame:
    where, params = _build_review_filter(filters)
    params["limit"]  = limit
    params["offset"] = offset
    sql = f"""
        SELECT create_time, product_name, comment, rating_star, sentiment, emotion
        FROM reviews
        {where}
        ORDER BY create_time DESC
        LIMIT :limit OFFSET :offset
    """
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["create_time", "product_name", "comment", "rating_star", "sentiment", "emotion"],
        )
    return df


def get_total_reviews_count(filters: dict) -> int:
    where, params = _build_review_filter(filters)
    sql = f"SELECT COUNT(*) FROM reviews {where}"
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar() or 0



def get_alerts_page(filters: dict, limit: int = 5, offset: int = 0) -> pd.DataFrame:
    where, params = _build_alert_filter(filters, date_column="a.triggered_at")
    params["limit"]  = limit
    params["offset"] = offset
    sql = f"""
        SELECT a.triggered_at, a.alert_type, a.comment, a.rating_avg,
               COALESCE(r.product_name, '—') AS product_name,
               r.create_time AS review_created_at,
               r.rating_star AS review_rating
        FROM alerts a
        LEFT JOIN reviews r ON a.review_id = r.id
        {where}
        ORDER BY a.triggered_at DESC
        LIMIT :limit OFFSET :offset
    """
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(sql), params).fetchall(),
            columns=["triggered_at", "alert_type", "comment", "rating_avg", "product_name", "review_created_at", "review_rating"],
        )
    return df


def get_total_alerts_count(filters: dict) -> int:
    where, params = _build_alert_filter(filters, date_column="a.triggered_at")
    sql = f"""
        SELECT COUNT(*)
        FROM alerts a
        LEFT JOIN reviews r ON a.review_id = r.id
        {where}
    """
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar() or 0


def get_products() -> list[str]:
    sql = "SELECT DISTINCT product_name FROM reviews ORDER BY product_name"
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return [row[0] for row in rows]

@st.cache_data(ttl=10)
def get_active_models() -> dict:
    sql = """
        SELECT task_type, version, f1_score_macro
        FROM model_metadata
        WHERE is_active = TRUE
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return {
        task: {"version": version, "f1": float(f1) if f1 is not None else None}
        for task, version, f1 in rows
    }


@st.cache_data(ttl=10)
def get_alert_breakdown(filters: dict) -> dict:
    where, params = _build_alert_filter(filters, date_column="a.triggered_at")
    sql = f"""
        SELECT a.alert_type, COUNT(*) AS total
        FROM alerts a
        LEFT JOIN reviews r ON a.review_id = r.id
        {where}
        GROUP BY a.alert_type
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return {alert_type: int(total) for alert_type, total in rows}


@st.cache_data(ttl=10)
def get_data_freshness() -> dict:
    sql = "SELECT MAX(processed_at), MAX(create_time) FROM reviews"
    with engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()
    return {"processed_at": row[0], "create_time": row[1]}
