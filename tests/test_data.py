from pathlib import Path

import pandas as pd
import pytest

from airline_sentiment.data import chronological_split, load_airline_tweets, normalize_text


def _row(tweet_id: int, text: str, day: int, label: str = "negative") -> dict:
    return {
        "tweet_id": tweet_id,
        "airline_sentiment": label,
        "airline": "Example Air",
        "text": text,
        "tweet_created": f"2015-02-{day:02d} 12:00:00 -0800",
        "name": f"user-{tweet_id}",
        "tweet_location": "not retained",
    }


def test_normalize_text_masks_direct_identifiers() -> None:
    result = normalize_text("@Example I loved this! https://example.com/a")

    assert result == "user i loved this! url"
    assert "@example" not in result
    assert "https://" not in result


def test_loader_validates_and_deduplicates(tmp_path: Path) -> None:
    rows = [
        _row(1, "first message", 17),
        _row(1, "duplicate id", 18),
        _row(3, "first message", 19),
        _row(4, "final message", 20, "positive"),
    ]
    path = tmp_path / "tweets.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame, audit = load_airline_tweets(path)

    assert len(frame) == 2
    assert audit.duplicate_ids_removed == 1
    assert audit.duplicate_texts_removed == 1
    assert "name" not in frame.columns
    assert "tweet_location" not in frame.columns


def test_loader_rejects_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "tweets.csv"
    pd.DataFrame([_row(1, "message", 17, "mixed")]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unexpected sentiment"):
        load_airline_tweets(path)


def test_chronological_split_reserves_latest_rows() -> None:
    frame = pd.DataFrame(
        [_row(index, f"message {index}", 17 + index % 8) for index in range(20)]
    )
    frame["tweet_created"] = pd.to_datetime(frame["tweet_created"], utc=True)

    train, test = chronological_split(frame, test_fraction=0.2)

    assert len(train) == 16
    assert len(test) == 4
    assert train["tweet_created"].max() <= test["tweet_created"].min()
