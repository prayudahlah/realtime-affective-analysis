from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import Config, TaskConfig, load_config
from .data import (
    attach_text,
    dataset_sha256,
    deduplicate,
    load_raw,
    make_split,
)
from .evaluate import holdout_metrics, index_hash, out_of_fold_predictions
from .export import write_bundle, write_manifest
from .features import (
    BASIC_FEATURE_NAMES,
    basic_features_frame,
    build_vectorizer,
    compose_matrix,
)
from .text import build_slang_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_classifier(task: TaskConfig) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, solver="lbfgs")


def build_search(task: TaskConfig, seed: int, sampling: str | None):
    estimator = make_classifier(task)
    if sampling == "smote":
        config = dict(task.sampling or {})
        pipeline = ImbPipeline(
            [
                (
                    "sampler",
                    SMOTE(
                        random_state=seed,
                        k_neighbors=int(config.get("k_neighbors", 3)),
                    ),
                ),
                ("clf", estimator),
            ]
        )
    else:
        pipeline = Pipeline([("clf", estimator)])

    param_grid = {f"clf__{key}": values for key, values in task.grid.items()}
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=task.cv_folds, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=-1,
    )
    return search


def metadata_common(
    dataset_hash: str,
    rows_raw: int,
    rows_used: int,
    duplicates_removed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Config,
) -> dict:
    return {
        "dataset": {
            "path": config.data.path,
            "rows_raw": rows_raw,
            "rows_used": rows_used,
            "duplicates_removed": duplicates_removed,
            "sha256": dataset_hash,
        },
        "split": {
            "test_size": config.split.test_size,
            "seed": config.split.seed,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_index_hash": index_hash(test["text"]),
        },
    }


def train_sentiment(
    config: Config,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_hash: str,
    rows_raw: int,
    duplicates_removed: int,
    slang_dict: dict[str, str],
):
    y_train = train[config.data.sentiment_column].astype(str).to_numpy()
    y_test = test[config.data.sentiment_column].astype(str).to_numpy()

    vectorizer = build_vectorizer(config.sentiment.tfidf)
    x_train_tfidf = vectorizer.fit_transform(train["text_processed"])
    x_test_tfidf = vectorizer.transform(test["text_processed"])

    basic_train = basic_features_frame(train["text"])
    basic_test = basic_features_frame(test["text"])
    x_train = compose_matrix(x_train_tfidf, basic_train)
    x_test = compose_matrix(x_test_tfidf, basic_test)

    search = build_search(config.sentiment, config.split.seed, sampling=None)
    search.fit(x_train, y_train)
    best = search.best_estimator_
    predictions = best.predict(x_test)

    metrics = holdout_metrics(y_test, predictions)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_metrics = holdout_metrics(y_test, baseline.predict(x_test))

    clf = best.named_steps["clf"]
    metadata = {
        **metadata_common(
            dataset_hash, rows_raw, len(train) + len(test), duplicates_removed, train, test, config
        ),
        "features": {
            "order": ["tfidf"] + list(BASIC_FEATURE_NAMES),
            "tfidf": config.sentiment.tfidf,
            "basic11": True,
            "sentiment_indicator": False,
        },
        "model": {
            "algo": type(clf).__name__,
            "params": {
                key: value
                for key, value in clf.get_params().items()
                if key in ("C", "max_iter", "solver")
            },
            "sampling": None,
        },
        "metrics": {
            "cv": {
                "f1_macro_mean": round(float(search.best_score_), 4),
                "f1_macro_std": round(
                    float(search.cv_results_["std_test_score"][search.best_index_]), 4
                ),
                "folds": config.sentiment.cv_folds,
            },
            "holdout": metrics,
        },
        "baseline": {"strategy": "most_frequent", "holdout": baseline_metrics},
    }

    bundle = write_bundle(
        config.output.dir,
        "sentiment",
        config.sentiment.version,
        clf,
        vectorizer,
        metadata,
    )

    oof = out_of_fold_predictions(
        best, x_train, y_train, config.sentiment.cv_folds, config.split.seed
    )
    print(
        f"[sentiment] cv f1={search.best_score_:.4f} holdout f1={metrics['f1_macro']:.4f} "
        f"baseline f1={baseline_metrics['f1_macro']:.4f} -> {bundle}"
    )
    return {
        "vectorizer": vectorizer,
        "clf": clf,
        "oof_train": oof,
        "test_predictions": predictions,
    }


def train_emotion(
    config: Config,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_hash: str,
    rows_raw: int,
    duplicates_removed: int,
    sentiment_train_oof: np.ndarray,
    sentiment_test: np.ndarray,
):
    y_train = train[config.data.emotion_column].astype(str).to_numpy()
    y_test = test[config.data.emotion_column].astype(str).to_numpy()

    vectorizer = build_vectorizer(config.emotion.tfidf)
    x_train_tfidf = vectorizer.fit_transform(train["text_processed"])
    x_test_tfidf = vectorizer.transform(test["text_processed"])

    basic_train = basic_features_frame(train["text"])
    basic_test = basic_features_frame(test["text"])

    indicator_train = (sentiment_train_oof == "Positive").astype(int)
    indicator_test = (sentiment_test == "Positive").astype(int)
    x_train = compose_matrix(x_train_tfidf, basic_train, indicator_train)
    x_test = compose_matrix(x_test_tfidf, basic_test, indicator_test)

    sampling = (config.emotion.sampling or {}).get("method")
    search = build_search(config.emotion, config.split.seed, sampling=sampling)
    search.fit(x_train, y_train)
    best = search.best_estimator_
    predictions = best.predict(x_test)

    metrics = holdout_metrics(y_test, predictions)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_metrics = holdout_metrics(y_test, baseline.predict(x_test))

    clf = best.named_steps["clf"]
    metadata = {
        **metadata_common(
            dataset_hash, rows_raw, len(train) + len(test), duplicates_removed, train, test, config
        ),
        "features": {
            "order": ["tfidf"] + list(BASIC_FEATURE_NAMES) + ["sentiment_positive"],
            "tfidf": config.emotion.tfidf,
            "basic11": True,
            "sentiment_indicator": True,
            "sentiment_indicator_source": "out_of_fold",
        },
        "model": {
            "algo": type(clf).__name__,
            "params": {
                key: value
                for key, value in clf.get_params().items()
                if key in ("C", "max_iter", "solver")
            },
            "sampling": config.emotion.sampling,
        },
        "metrics": {
            "cv": {
                "f1_macro_mean": round(float(search.best_score_), 4),
                "f1_macro_std": round(
                    float(search.cv_results_["std_test_score"][search.best_index_]), 4
                ),
                "folds": config.emotion.cv_folds,
            },
            "holdout": metrics,
        },
        "baseline": {"strategy": "most_frequent", "holdout": baseline_metrics},
    }

    bundle = write_bundle(
        config.output.dir,
        "emotion",
        config.emotion.version,
        clf,
        vectorizer,
        metadata,
    )
    print(
        f"[emotion]   cv f1={search.best_score_:.4f} holdout f1={metrics['f1_macro']:.4f} "
        f"baseline f1={baseline_metrics['f1_macro']:.4f} -> {bundle}"
    )
    return {"clf": clf, "vectorizer": vectorizer}


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    base_dir = Path(config_path).resolve().parent.parent
    config.data.path = str(base_dir / config.data.path)
    config.data.cache_dir = str(base_dir / config.data.cache_dir)
    config.output.dir = str(base_dir / config.output.dir)

    set_seed(config.split.seed)

    raw = load_raw(config.data)
    rows_raw = len(raw)
    if config.data.deduplicate:
        raw, duplicates_removed = deduplicate(raw, config.data.text_column)
    else:
        duplicates_removed = 0

    dataset_hash = dataset_sha256(config.data.path)
    slang_dict = build_slang_dict(base_dir / "data/util/colloquial-indonesian-lexicon.csv")
    frame = attach_text(raw, config.data, slang_dict, dataset_hash)
    train, test = make_split(
        frame, config.split, config.data.sentiment_column, config.data.emotion_column
    )

    print(
        f"[data] raw={rows_raw} used={len(frame)} duplicates_removed={duplicates_removed} "
        f"train={len(train)} test={len(test)}"
    )

    sentiment = train_sentiment(
        config, train, test, dataset_hash, rows_raw, duplicates_removed, slang_dict
    )
    train_emotion(
        config,
        train,
        test,
        dataset_hash,
        rows_raw,
        duplicates_removed,
        sentiment["oof_train"],
        sentiment["test_predictions"],
    )

    write_manifest(
        config.output.dir,
        {"sentiment": config.sentiment.version, "emotion": config.emotion.version},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model sentimen dan emosi")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
