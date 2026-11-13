from __future__ import annotations

import re
import ssl
from pathlib import Path

import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

LEXICON_URL = (
    "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/"
    "colloquial-indonesian-lexicon.csv"
)

NEG_WORDS = {
    "tidak",
    "bukan",
    "belum",
    "tak",
    "ngga",
    "gak",
    "ga",
    "tdk",
    "enggak",
    "nggak",
    "kagak",
    "ndak",
    "ngg",
}

ONJUNGSI = {
    "dan",
    "atau",
    "tetapi",
    "namun",
    "sedangkan",
    "serta",
    "karena",
    "sehingga",
    "maka",
    "lalu",
    "kemudian",
    "setelah",
    "sebelum",
    "ketika",
    "sementara",
    "walaupun",
    "meskipun",
    "jika",
    "kalau",
    "apabila",
    "bahwa",
}
PREPOSISI = {
    "di",
    "ke",
    "dari",
    "pada",
    "dengan",
    "untuk",
    "bagi",
    "oleh",
    "tentang",
    "seperti",
    "sebagai",
    "tanpa",
    "dalam",
    "antara",
    "menurut",
    "sampai",
    "hingga",
}

_stemmer = None


def ensure_nltk() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context
    for resource in ("tokenizers/punkt_tab", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def build_slang_dict(lexicon_path: str | Path) -> dict[str, str]:
    path = Path(lexicon_path)
    if not path.exists():
        frame = pd.read_csv(LEXICON_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    valid = frame[frame["In-dictionary"] == 1]
    slang_dict: dict[str, str] = {}
    for _, row in valid.iterrows():
        slang = str(row["slang"]).strip().lower()
        formal = str(row["formal"]).strip().lower()
        if slang not in slang_dict or len(formal) < len(slang_dict[slang]):
            slang_dict[slang] = formal
    return slang_dict


def normalize_repetitive_chars(text: str) -> str:
    text = re.sub(r"(a)\1{2,}", "a", text)
    text = re.sub(r"(i)\1{2,}", "i", text)
    text = re.sub(r"(u)\1{2,}", "u", text)
    text = re.sub(r"(e)\1{2,}", "e", text)
    text = re.sub(r"(o)\1{2,}", "o", text)
    text = re.sub(r"([^aiueo])\1{2,}", r"\1", text)
    return text


def normalize_slang(text: str, slang_dict: dict[str, str]) -> str:
    if not slang_dict:
        return text
    words = text.split()
    normalized = []
    for word in words:
        lower = word.lower()
        stripped = lower.rstrip(".,!?;:")
        if lower in slang_dict:
            normalized.append(slang_dict[lower])
        elif stripped in slang_dict:
            punct = word[len(stripped) :]
            normalized.append(slang_dict[stripped] + punct)
        else:
            normalized.append(word)
    return " ".join(normalized)


def emoji_to_text(text: str) -> str:
    return emoji.demojize(text, language="id")


def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    tagged = []
    for token in tokens:
        lower = token.lower()
        if re.match(r'^[.,!?;:()\[\]{}"\'\-]$', token):
            tagged.append((token, "PUNCT"))
        elif re.match(r"^[0-9,.\-]+$", token):
            tagged.append((token, "NUM"))
        elif lower in ONJUNGSI:
            tagged.append((token, "CONJ"))
        elif lower in PREPOSISI:
            tagged.append((token, "ADP"))
        elif re.match(r"^(me|men|meng|meny|mem|di|ber|bel|ter|per)", lower):
            tagged.append((token, "VERB"))
        elif (
            re.match(r"^(pe|pen|pem|peng|ke)", lower)
            or lower.endswith("an")
            or lower.endswith("kan")
        ):
            tagged.append((token, "NOUN"))
        elif lower.endswith("i"):
            tagged.append((token, "VERB"))
        else:
            tagged.append((token, "NOUN"))
    return tagged


def handle_negation(tokens: list[str]) -> list[str]:
    result = []
    negate = False
    for word in tokens:
        clean = word.lower().strip(".,!?")
        if clean in NEG_WORDS:
            result.append(word)
            negate = True
        elif negate:
            result.append("NEG_" + word)
            negate = False
        else:
            result.append(word)
    return result


def full_pipeline(text: str, slang_dict: dict[str, str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = normalize_repetitive_chars(text)
    t = normalize_slang(t, slang_dict)
    t = emoji_to_text(t)
    tokens = word_tokenize(t)
    tokens = [tok for tok in tokens if tok.lower() not in set(stopwords.words("indonesian"))]
    pos_tag(tokens)
    stemmer = get_stemmer()
    tokens = [stemmer.stem(tok) for tok in tokens]
    tokens = handle_negation(tokens)
    return " ".join(tokens)


def preprocess_series(series: pd.Series, slang_dict: dict[str, str]) -> pd.Series:
    ensure_nltk()
    return series.fillna("").astype(str).apply(lambda text: full_pipeline(text, slang_dict))
