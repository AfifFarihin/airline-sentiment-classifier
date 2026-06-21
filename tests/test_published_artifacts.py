import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_summary_contract() -> None:
    summary = json.loads(
        (ROOT / "outputs/metrics/evaluation_summary.json").read_text(encoding="utf-8")
    )

    assert summary["selected_model"] in {"complement_nb", "logistic_regression"}
    assert summary["train_end"] <= summary["test_start"]
    assert summary["holdout_metrics"]["f1_macro"] > 0.5
    assert "never model features or published" in summary["privacy"].lower()
    assert len(summary["author_generalization_slices"]) == 2


def test_published_tables_contain_no_raw_text_or_identifiers() -> None:
    forbidden = {"text", "normalized_text", "name", "tweet_id", "tweet_location"}
    for path in (ROOT / "outputs/metrics").glob("*.csv"):
        columns = set(pd.read_csv(path).columns)
        assert not columns & forbidden
