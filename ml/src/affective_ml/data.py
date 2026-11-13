from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import DataConfig, SplitConfig
from .text import preprocess_series


class DatasetError(Exception):
    pass


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(config: DataConfig) -> pd.DataFrame:
    path = Path(config.path)
    if not path.exists():
        raise DatasetError(
            f"Dataset tidak ditemukan di {path}. Letakkan ecommerce_reviews.csv di sana."
        )
    frame = pd.read_csv(path)
    missing = [
        column
        for column in (config.text_column, config.sentiment_column, config.emotion_column)
        if column not in frame.columns
    ]
    if missing:
        raise DatasetError(f"Kolom wajib tidak ada: {', '.join(missing)}")
    return frame


def deduplicate(frame: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, int]:
    normalized = frame[text_column].fillna("").astype(str).str.strip().str.lower()
    keep = ~normalized.duplicated(keep="first")
    removed = int((~keep).sum())
    return frame.loc[keep].reset_index(drop=True), removed


def attach_text(
    frame: pd.DataFrame,
    data_config: DataConfig,
    slang_dict: dict[str, str],
    dataset_hash: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["text"] = frame[data_config.text_column].fillna("").astype(str)

    cache_dir = Path(data_config.cache_dir)
    cache_path = cache_dir / f"preprocessed_{dataset_hash[:12]}.csv"
    cached: pd.DataFrame | None = None
    if cache_path.exists():
        candidate = pd.read_csv(cache_path)
        if len(candidate) == len(frame):
            cached = candidate

    if cached is not None:
        frame["text_processed"] = cached["text_processed"].fillna("").astype(str)
    else:
        frame["text_processed"] = preprocess_series(frame["text"], slang_dict)
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame[["text", "text_processed"]].to_csv(cache_path, index=False)

    return frame


def make_split(
    frame: pd.DataFrame, split: SplitConfig, sentiment_column: str, emotion_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    stratify = frame[sentiment_column].astype(str) + "_" + frame[emotion_column].astype(str)
    train, test = train_test_split(
        frame,
        test_size=split.test_size,
        random_state=split.seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
