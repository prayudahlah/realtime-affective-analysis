from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str = "data/raw/ecommerce_reviews.csv"
    text_column: str = "Customer Review"
    sentiment_column: str = "Sentiment"
    emotion_column: str = "Emotion"
    deduplicate: bool = True
    cache_dir: str = "data/intermediate"


@dataclass
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42


@dataclass
class TaskConfig:
    version: str = "1.0.0"
    tfidf: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    cv_folds: int = 5
    sampling: dict[str, Any] | None = None


@dataclass
class OutputConfig:
    dir: str = "../models"


@dataclass
class Config:
    seed: int
    data: DataConfig
    split: SplitConfig
    sentiment: TaskConfig
    emotion: TaskConfig
    output: OutputConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(**raw.get("data", {})),
        split=SplitConfig(**raw.get("split", {})),
        sentiment=TaskConfig(**raw["sentiment"]),
        emotion=TaskConfig(**raw["emotion"]),
        output=OutputConfig(**raw.get("output", {})),
    )
