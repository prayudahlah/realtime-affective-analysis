"""Membuat sampel JSONL untuk mode replay dari dataset mentah."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"https?://\S+"), "[tautan]"),
    (re.compile(r"(?:\+62|62|0)8[0-9]{7,12}"), "[telepon]"),
    (re.compile(r"@\w+"), "@pengguna"),
    (re.compile(r"\b\d{6,}\b"), "[angka]"),
]


def scrub(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ml/data/raw/ecommerce_reviews.csv")
    parser.add_argument("--output", default="data/sample/reviews_sample.jsonl")
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.source)
    frame = frame.dropna(subset=["Customer Review", "Emotion", "Customer Rating"])
    frame = frame[frame["Customer Review"].astype(str).str.strip().str.len() > 0]
    frame = frame.drop_duplicates(subset=["Customer Review"])

    sampled = (
        frame.groupby("Emotion", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), args.per_class), random_state=args.seed))
        .sort_index()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (_, row) in enumerate(sampled.iterrows(), start=1):
            record = {
                "nama pengguna": f"pengguna_{index:03d}",
                "produk": str(row["Product Name"])[:80],
                "review": scrub(str(row["Customer Review"])),
                "rating": int(row["Customer Rating"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Ditulis {len(sampled)} baris ke {output}")


if __name__ == "__main__":
    main()
