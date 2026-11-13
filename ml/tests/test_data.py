import pandas as pd

from affective_ml.config import SplitConfig
from affective_ml.data import deduplicate, make_split


def make_frame() -> pd.DataFrame:
    texts = ["produk bagus", "produk bagus", "pengiriman lambat", "sesuai deskripsi"]
    return pd.DataFrame(
        {
            "Customer Review": texts,
            "Sentiment": ["Positive", "Positive", "Negative", "Positive"],
            "Emotion": ["Happy", "Happy", "Anger", "Happy"],
            "text": texts,
        }
    )


def test_deduplicate_removes_repeated_text():
    frame, removed = deduplicate(make_frame(), "Customer Review")
    assert removed == 1
    assert len(frame) == 3


def test_split_is_deterministic():
    frame = pd.concat([make_frame()] * 10, ignore_index=True)
    config = SplitConfig(test_size=0.25, seed=7)
    first = make_split(frame, config, "Sentiment", "Emotion")
    second = make_split(frame, config, "Sentiment", "Emotion")
    assert first[0]["text"].tolist() == second[0]["text"].tolist()
    assert first[1]["text"].tolist() == second[1]["text"].tolist()
