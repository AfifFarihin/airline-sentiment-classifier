"""Dataset validation, privacy minimization, and chronological splitting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "tweet_id",
    "airline_sentiment",
    "airline",
    "text",
    "tweet_created",
}
VALID_LABELS = {"negative", "neutral", "positive"}


@dataclass(frozen=True)
class DatasetAudit:
    source_rows: int
    retained_rows: int
    duplicate_ids_removed: int
    duplicate_texts_removed: int
    first_timestamp: str
    last_timestamp: str


def normalize_text(text: str) -> str:
    """Replace direct identifiers while preserving sentiment-bearing text."""
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " url ", value)
    value = re.sub(r"@\w+", " user ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_airline_tweets(path: str | Path) -> tuple[pd.DataFrame, DatasetAudit]:
    """Load, validate, deduplicate, and minimize the airline tweet dataset."""
    source = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    source_rows = len(source)
    frame = source[sorted(REQUIRED_COLUMNS)].copy()
    if "name" in source.columns:
        author_values = source["name"].fillna(
            source["tweet_id"].map(lambda value: f"missing-{value}")
        )
        frame["author_group"] = author_values.map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
        )
    else:
        frame["author_group"] = source["tweet_id"].map(lambda value: f"row-{value}")
    frame["tweet_created"] = pd.to_datetime(frame["tweet_created"], utc=True, errors="raise")
    frame["airline_sentiment"] = frame["airline_sentiment"].astype(str).str.lower()
    unexpected = set(frame["airline_sentiment"].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"Unexpected sentiment labels: {sorted(unexpected)}")

    duplicate_ids = int(frame.duplicated("tweet_id").sum())
    frame = frame.drop_duplicates("tweet_id", keep="first")
    duplicate_texts = int(frame.duplicated("text").sum())
    frame = frame.drop_duplicates("text", keep="first")
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame = frame[frame["normalized_text"].str.len() > 0].copy()
    frame = frame.sort_values(["tweet_created", "tweet_id"]).reset_index(drop=True)

    audit = DatasetAudit(
        source_rows=source_rows,
        retained_rows=len(frame),
        duplicate_ids_removed=duplicate_ids,
        duplicate_texts_removed=duplicate_texts,
        first_timestamp=frame["tweet_created"].min().isoformat(),
        last_timestamp=frame["tweet_created"].max().isoformat(),
    )
    return frame, audit


def chronological_split(
    frame: pd.DataFrame, *, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest observations as a realistic future-message holdout."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for a chronological split.")

    ordered = frame.sort_values(["tweet_created", "tweet_id"]).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    if train["tweet_created"].max() > test["tweet_created"].min():
        raise AssertionError("Chronological split invariant failed.")
    return train, test
