from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

BASIC_FEATURE_NAMES = (
    "n_exclamation",
    "n_question",
    "n_allcaps",
    "n_ellipsis",
    "max_char_repeat",
    "n_demands",
    "n_uncertainty",
    "n_swear",
    "n_attachment",
    "n_repurchase",
    "n_transactional",
)

DEMAND_WORDS = {
    "kembalikan",
    "ganti",
    "refund",
    "komplain",
    "keluhan",
    "kembali",
    "tolong",
    "mohon",
    "urus",
    "klarifikasi",
    "balas",
}
UNCERTAINTY_WORDS = {
    "mungkin",
    "khawatir",
    "takut",
    "was-was",
    "cemas",
    "ragu",
    "bimbang",
    "curiga",
    "sepertinya",
    "seolah",
    "antisipasi",
    "harap",
}
SWEAR_WORDS = {
    "anjing",
    "bangsat",
    "bodoh",
    "tolol",
    "jelek",
    "parah",
    "payah",
    "sampah",
    "busuk",
    "brengsek",
    "persetan",
    "keparat",
    "setan",
    "sial",
    "kacau",
}
ATTACHMENT_WORDS = {
    "cinta",
    "sayang",
    "suka",
    "gemas",
    "love",
    "favorit",
    "kesayangan",
    "favorite",
}
REPURCHASE_PHRASES = {
    "beli lagi",
    "order lagi",
    "repeat order",
    "langganan",
    "balik lagi",
    "pasti beli",
    "akan beli",
    "nanti beli",
    "rekomendasi",
    "recommend",
    "beli disini terus",
}
TRANSACTIONAL_WORDS = {
    "bagus",
    "mantap",
    "ok",
    "oke",
    "cocok",
    "puas",
    "sesuai",
    "recommended",
    "keren",
    "mantul",
    "top",
    "good",
    "nice",
    "great",
    "worth",
}


def _count_words(text: str, vocabulary: Iterable[str]) -> int:
    vocabulary = set(vocabulary)
    return sum(1 for word in text.lower().split() if word in vocabulary)


def basic_features_frame(series: pd.Series) -> pd.DataFrame:
    text = series.fillna("").astype(str)
    lower = text.str.lower()

    features = pd.DataFrame(index=series.index)
    features["n_exclamation"] = text.str.count("!")
    features["n_question"] = text.str.count(r"\?")
    features["n_allcaps"] = text.apply(
        lambda value: sum(1 for token in value.split() if token.isupper() and len(token) > 2)
    )
    features["n_ellipsis"] = text.str.count(r"\.\.")
    features["max_char_repeat"] = lower.apply(
        lambda value: max((len(list(group)) for _, group in itertools.groupby(value)), default=0)
    )
    features["n_demands"] = lower.apply(lambda value: _count_words(value, DEMAND_WORDS))
    features["n_uncertainty"] = lower.apply(lambda value: _count_words(value, UNCERTAINTY_WORDS))
    features["n_swear"] = lower.apply(lambda value: _count_words(value, SWEAR_WORDS))
    features["n_attachment"] = lower.apply(lambda value: _count_words(value, ATTACHMENT_WORDS))
    features["n_repurchase"] = lower.apply(
        lambda value: sum(1 for phrase in REPURCHASE_PHRASES if phrase in value)
    )
    features["n_transactional"] = lower.apply(
        lambda value: _count_words(value, TRANSACTIONAL_WORDS)
    )
    return features


def build_vectorizer(config: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=int(config.get("max_features", 1500)),
        ngram_range=tuple(config.get("ngram_range", (1, 1))),
        min_df=int(config.get("min_df", 1)),
    )


def compose_matrix(
    tfidf_matrix,
    basic_frame: pd.DataFrame,
    sentiment_indicator: np.ndarray | None = None,
):
    parts = [tfidf_matrix, csr_matrix(basic_frame.to_numpy(dtype=float))]
    if sentiment_indicator is not None:
        parts.append(csr_matrix(sentiment_indicator.reshape(-1, 1).astype(float)))
    return hstack(parts).tocsr()
