import os
import sys
import logging
import json

from bundles import load_active_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MLInference")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "cdc.mongodb.ecommerce.reviews")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ml-inference-group")
SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
SPARK_TRIGGER_INTERVAL = os.getenv("SPARK_TRIGGER_INTERVAL", "1 seconds")
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/app/data/checkpoint")
MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models")
POSTGRES_HOST = os.getenv("INFERENCE_DB_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("INFERENCE_DB_HOST_PORT", "5432"))
POSTGRES_DB = os.getenv("INFERENCE_DB_NAME", "postgres")
POSTGRES_JDBC_URL = os.getenv(
    "POSTGRES_JDBC_URL",
    f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)
POSTGRES_USER = os.getenv("INFERENCE_DB_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("INFERENCE_DB_PASSWORD", "postgres")


def register_model_metadata(bundle: dict) -> None:
    import psycopg2

    metadata = bundle["metadata"]
    task = metadata.get("task", "unknown")
    version = bundle["version"]
    f1 = metadata.get("metrics", {}).get("holdout", {}).get("f1_macro")
    trained_at = metadata.get("trained_at")
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_metadata
                        (model_name, version, f1_score_macro, trained_at, is_active, task_type)
                    VALUES (%s, %s, %s, %s, TRUE, %s)
                    ON CONFLICT (model_name, version) DO UPDATE SET
                        f1_score_macro = EXCLUDED.f1_score_macro,
                        trained_at = EXCLUDED.trained_at,
                        is_active = TRUE,
                        task_type = EXCLUDED.task_type
                    """,
                    (task, version, f1, trained_at, task),
                )
                cur.execute(
                    """
                    UPDATE model_metadata
                    SET is_active = FALSE
                    WHERE task_type = %s
                      AND (version IS NULL OR version <> %s)
                    """,
                    (task, version),
                )
        finally:
            conn.close()
        logger.info("[MODEL] model_metadata %s v%s tercatat", task, version)
    except Exception as exc:
        logger.warning("[MODEL] Gagal mencatat model_metadata: %s", exc)


def main():
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import (
        from_json,
        col,
        udf,
        from_unixtime,
        coalesce,
        get_json_object,
        window,
        avg,
        expr,
    )
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType,
        ArrayType,
        LongType,
        TimestampType,
    )

    logger.info("StreamMLInference started (kafka=%s, spark=%s, pg=%s)", KAFKA_BOOTSTRAP_SERVERS, SPARK_MASTER_URL, POSTGRES_JDBC_URL)

    spark = (
        SparkSession.builder.appName("StreamMLInference")
        .master(SPARK_MASTER_URL)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.cores.max", "4")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("[SPARK] Session created")


    sentiment_bundle = load_active_bundle(MODEL_ROOT, "sentiment")
    emotion_bundle = load_active_bundle(MODEL_ROOT, "emotion")
    sentiment_model = sentiment_bundle["model"]
    emotion_model = emotion_bundle["model"]
    sentiment_n_features = len(sentiment_bundle["vectorizer"].get_feature_names_out())
    emotion_n_features = len(emotion_bundle["vectorizer"].get_feature_names_out())

    register_model_metadata(sentiment_bundle)
    register_model_metadata(emotion_bundle)

    logger.info(
        "[MODEL] Sentiment v%s f1=%s fitur=%d",
        sentiment_bundle["version"],
        sentiment_bundle["metadata"]["metrics"]["holdout"]["f1_macro"],
        sentiment_n_features,
    )
    logger.info(
        "[MODEL] Emotion v%s f1=%s fitur=%d",
        emotion_bundle["version"],
        emotion_bundle["metadata"]["metrics"]["holdout"]["f1_macro"],
        emotion_n_features,
    )

    sentiment_bc = spark.sparkContext.broadcast(sentiment_model)
    emotion_bc = spark.sparkContext.broadcast(emotion_model)

    after_schema = StructType(
        [
            StructField("_id", StringType(), True),
            StructField("nama pengguna", StringType(), True),
            StructField("produk", StringType(), True),
            StructField("review", StringType(), True),
            StructField("rating", IntegerType(), True),
            StructField("ctime", IntegerType(), True),
            StructField(
                "sentiment_vectorized", ArrayType(ArrayType(FloatType())), True
            ),
            StructField("emotion_vectorized", ArrayType(ArrayType(FloatType())), True),
            StructField("emotion_features_basic", ArrayType(IntegerType()), True),
        ]
    )

    # Schema for the Debezium envelope
    envelope_schema = StructType(
        [
            StructField("schema", StringType(), True),
            StructField(
                "payload",
                StructType(
                    [
                        StructField("before", StringType(), True),
                        StructField("after", StringType(), True),
                        StructField("patch", StringType(), True),
                        StructField("filter", StringType(), True),
                        StructField("updateDescription", StringType(), True),
                        StructField(
                            "source",
                            StructType(
                                [
                                    StructField("version", StringType(), True),
                                    StructField("connector", StringType(), True),
                                    StructField("name", StringType(), True),
                                    StructField("ts_ms", LongType(), True),
                                    StructField("snapshot", StringType(), True),
                                    StructField("db", StringType(), True),
                                    StructField("sequence", StringType(), True),
                                    StructField("collection", StringType(), True),
                                    StructField("ord", IntegerType(), True),
                                ]
                            ),
                            True,
                        ),
                        StructField("op", StringType(), True),
                        StructField("ts_ms", LongType(), True),
                        StructField("transaction", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    result_schema = StructType(
        [
            StructField("sentiment_label", StringType(), True),
            StructField("emotion_label", StringType(), True),
        ]
    )


    @udf(result_schema)
    def predict_udf(
        review: str,
        sentiment_vec,
        emotion_vec,
        basic_features,
    ):
        if not review or not review.strip():
            return (None, None)
        if sentiment_vec is None or emotion_vec is None:
            return None

        import numpy as np
        from scipy.sparse import csr_matrix, hstack

        s_indices = [int(p[0]) for p in sentiment_vec]
        s_values = [float(p[1]) for p in sentiment_vec]
        s_vec = csr_matrix(
            (s_values, ([0] * len(s_indices), s_indices)),
            shape=(1, sentiment_n_features),
        )

        if basic_features and len(basic_features) >= 11:
            basic_arr = np.array([float(basic_features[i]) for i in range(11)]).reshape(
                1, 11
            )
        else:
            basic_arr = np.zeros((1, 11))

        X_sent = hstack([s_vec, csr_matrix(basic_arr)])
        sent_label = str(sentiment_bc.value.predict(X_sent)[0])

        e_indices = [int(p[0]) for p in emotion_vec]
        e_values = [float(p[1]) for p in emotion_vec]
        e_vec = csr_matrix(
            (e_values, ([0] * len(e_indices), e_indices)),
            shape=(1, emotion_n_features),
        )

        sent_binary = np.array([[1.0 if sent_label == "Positive" else 0.0]])

        dense_part = csr_matrix(np.hstack([basic_arr, sent_binary]))
        X_emo = hstack([e_vec, dense_part])

        emo_label = str(emotion_bc.value.predict(X_emo)[0])

        return (sent_label, emo_label)


    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", "100")
        .load()
    )


    parsed_envelope = raw.select(
        from_json(col("value").cast("string"), envelope_schema).alias("envelope")
    ).select("envelope.payload.*")


    parsed = parsed_envelope.select(
        col("op"),
        col("ts_ms"),
        from_json(col("after"), after_schema).alias("doc"),
        col("after"),
    )


    valid = parsed.where(col("op").isin(["c", "r"])).where(col("doc").isNotNull())


    predictions = valid.select(
        coalesce(
            get_json_object(col("after"), "$._id.$oid"),
            col("doc._id"),
        ).alias("comment_id"),
        col("doc.nama pengguna").alias("buyer_username"),
        col("doc.produk").alias("product_name"),
        col("doc.review").alias("comment"),
        col("doc.rating").alias("rating_star"),
        col("doc.ctime").alias("create_time_ts"),
        predict_udf(
            col("doc.review"),
            col("doc.sentiment_vectorized"),
            col("doc.emotion_vectorized"),
            col("doc.emotion_features_basic"),
        ).alias("result"),
    ).select(
        col("comment_id"),
        col("buyer_username"),
        col("product_name"),
        col("comment"),
        col("rating_star"),
        from_unixtime(col("create_time_ts")).alias("create_time"),
        col("result.sentiment_label").alias("sentiment"),
        col("result.emotion_label").alias("emotion"),
    )


    def write_to_postgres(df: DataFrame, epoch_id: int):
        count = df.count()
        if count == 0:
            logger.info("[EPOCH %d] No records to write", epoch_id)
            return
        logger.info("[EPOCH %d] Writing %d predictions to PostgreSQL", epoch_id, count)
        import psycopg2

        rows = df.collect()
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        try:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO reviews
                            (comment_id, buyer_username, product_name, comment,
                             rating_star, create_time, sentiment, emotion)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (comment_id) DO UPDATE SET
                            sentiment = EXCLUDED.sentiment,
                            emotion = EXCLUDED.emotion,
                            processed_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """,
                        (
                            row.comment_id,
                            row.buyer_username,
                            row.product_name,
                            row.comment,
                            row.rating_star,
                            row.create_time,
                            row.sentiment,
                            row.emotion,
                        ),
                    )
                    review_id = cur.fetchone()[0]
                    if row.sentiment == "Negative" or row.emotion in (
                        "Fear",
                        "Anger",
                        "Sadness",
                    ):
                        logger.info(
                            "[ALERT] Sentimen negatif terdeteksi: %s", row.comment_id
                        )
                        cur.execute(
                            """
                            INSERT INTO alerts (alert_type, comment, review_id)
                            VALUES ('sentiment_negative', %s, %s)
                        """,
                            (row.comment, review_id),
                        )
            conn.commit()
            logger.info(
                "[EPOCH %d] Upserted %d rows, alerts: %s", epoch_id, len(rows), count
            )
        finally:
            conn.close()


    pg_query = (
        predictions.writeStream.foreachBatch(write_to_postgres)
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_DIR + "/postgres")
        .trigger(processingTime=SPARK_TRIGGER_INTERVAL)
        .start()
    )

    debug_query = (
        predictions.writeStream.format("console")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_DIR + "/console_debug")
        .trigger(processingTime=SPARK_TRIGGER_INTERVAL)
        .start()
    )

    recent_ratings = (
        predictions.withColumn("ts", col("create_time").cast(TimestampType()))
        .filter(col("ts") >= expr("current_timestamp() - INTERVAL '10 minutes'"))
        .withWatermark("ts", "1 minute")
        .groupBy(window(col("ts"), "10 minutes", "1 minute"))
        .agg(avg("rating_star").alias("avg_rating"))
        .filter(col("avg_rating") < 4.0)
    )

    def write_rating_alert(df: DataFrame, epoch_id: int):
        count = df.count()
        if count == 0:
            return
        rows = df.collect()
        import psycopg2

        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        try:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO alerts (alert_type, rating_avg)
                        SELECT 'rating_drop', %s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM alerts
                            WHERE alert_type = 'rating_drop'
                            AND triggered_at > NOW() - INTERVAL '10 minutes'
                        )
                    """,
                        (float(row.avg_rating),),
                    )
            conn.commit()
        finally:
            conn.close()

    rating_alert_query = (
        recent_ratings.writeStream.foreachBatch(write_rating_alert)
        .outputMode("update")
        .trigger(processingTime=SPARK_TRIGGER_INTERVAL)
        .option("checkpointLocation", CHECKPOINT_DIR + "/rating_alerts")
        .start()
    )

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
