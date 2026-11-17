"""Ekspor snapshot data dashboard dari PostgreSQL lokal ke JSON."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg2
import psycopg2.extras

OUTPUT = Path(
    os.getenv(
        "SNAPSHOT_OUTPUT",
        Path(__file__).resolve().parents[1]
        / "streaming"
        / "services"
        / "dashboard"
        / "snapshot.json",
    )
)


def _iso(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _rows(cursor, query: str) -> list[dict]:
    cursor.execute(query)
    return [{key: _iso(value) for key, value in row.items()} for row in cursor.fetchall()]


def main() -> None:
    conn = psycopg2.connect(
        host=os.getenv("INFERENCE_DB_HOST", "localhost"),
        port=int(os.getenv("INFERENCE_DB_HOST_PORT", "5432")),
        dbname=os.getenv("INFERENCE_DB_NAME", "postgres"),
        user=os.getenv("INFERENCE_DB_USER", "postgres"),
        password=os.getenv("INFERENCE_DB_PASSWORD", "postgres"),
    )
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            reviews = _rows(
                cur,
                """
                SELECT id, comment_id, buyer_username, product_name, comment,
                       rating_star, create_time, sentiment, emotion, processed_at
                FROM reviews
                ORDER BY id
                """,
            )
            alerts = _rows(
                cur,
                """
                SELECT a.id, a.alert_type, a.comment, a.rating_avg, a.review_id,
                       a.triggered_at, r.product_name, r.rating_star AS review_rating,
                       r.sentiment AS review_sentiment, r.create_time AS review_created_at
                FROM alerts a
                LEFT JOIN reviews r ON a.review_id = r.id
                ORDER BY a.id
                """,
            )
            models = _rows(
                cur,
                """
                SELECT id, model_name, version, f1_score_macro, trained_at,
                       is_active, task_type
                FROM model_metadata
                ORDER BY id
                """,
            )
    finally:
        conn.close()

    exported_at = datetime.now().isoformat(sep=" ", timespec="seconds")
    snapshot_date = max((row["create_time"] for row in reviews), default=exported_at)

    payload = {
        "meta": {
            "exported_at": exported_at,
            "snapshot_date": snapshot_date,
            "reviews": len(reviews),
            "alerts": len(alerts),
            "models": len(models),
        },
        "reviews": reviews,
        "alerts": alerts,
        "models": models,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Snapshot ditulis ke {OUTPUT}: {len(reviews)} ulasan, {len(alerts)} alert")


if __name__ == "__main__":
    main()
