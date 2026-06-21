"""Model candidates and leakage-safe search configuration."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from airline_sentiment.data import normalize_text


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    estimator: BaseEstimator
    parameters: dict[str, list[object]]


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        strip_accents="unicode",
        sublinear_tf=True,
        max_df=0.98,
    )


def candidate_specs(random_state: int = 42) -> list[CandidateSpec]:
    """Return predeclared candidates; vectorization stays inside every CV fold."""
    baseline = Pipeline(
        [
            ("tfidf", _tfidf()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    complement_nb = Pipeline(
        [
            ("tfidf", _tfidf()),
            ("model", ComplementNB()),
        ]
    )
    logistic = Pipeline(
        [
            ("tfidf", _tfidf()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return [
        CandidateSpec(
            "majority_baseline",
            baseline,
            {
                "tfidf__ngram_range": [(1, 1)],
                "tfidf__min_df": [2],
                "tfidf__max_features": [20_000],
            },
        ),
        CandidateSpec(
            "complement_nb",
            complement_nb,
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "tfidf__max_features": [30_000],
                "model__alpha": [0.1, 0.5, 1.0],
            },
        ),
        CandidateSpec(
            "logistic_regression",
            logistic,
            {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "tfidf__max_features": [30_000],
                "model__C": [0.5, 1.0, 2.0, 4.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]
