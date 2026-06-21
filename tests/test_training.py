from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from airline_sentiment.data import normalize_text
from airline_sentiment.modeling import CandidateSpec
from airline_sentiment.training import train_and_evaluate


def test_training_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    label_text = {
        "negative": "awful delay and bad service",
        "neutral": "please share flight information",
        "positive": "excellent flight and great crew",
    }
    rows = []
    labels = list(label_text)
    for index in range(90):
        label = labels[index % len(labels)]
        rows.append(
            {
                "tweet_id": index,
                "airline_sentiment": label,
                "airline": "Example Air",
                "text": f"{label_text[label]} reference {index}",
                "tweet_created": pd.Timestamp("2015-02-17", tz="UTC")
                + pd.Timedelta(minutes=index),
                "name": f"author-{index % 30}",
            }
        )
    data_path = tmp_path / "tweets.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    estimator = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    min_df=1,
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    candidate = CandidateSpec(
        name="logistic_regression",
        estimator=estimator,
        parameters={"model__C": [1.0]},
    )

    summary = train_and_evaluate(
        data_path,
        tmp_path / "outputs",
        cv_splits=3,
        n_boot=20,
        candidates=[candidate],
    )

    assert summary["selected_model"] == "logistic_regression"
    assert summary["train_end"] <= summary["test_start"]
    assert (tmp_path / "outputs/metrics/generalization_slices.csv").exists()
    assert (tmp_path / "outputs/models/sentiment_pipeline.joblib").exists()
