from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from airline_sentiment.modeling import candidate_specs


def test_candidates_keep_vectorization_inside_pipeline() -> None:
    candidates = candidate_specs()

    assert {candidate.name for candidate in candidates} == {
        "majority_baseline",
        "complement_nb",
        "logistic_regression",
    }
    assert all("tfidf" in candidate.estimator.named_steps for candidate in candidates)


def test_logistic_candidate_runs_cross_validation() -> None:
    texts = [
        "terrible delay and rude service",
        "late flight and lost bag",
        "awful cancellation",
        "flight information requested",
        "what time is boarding",
        "please confirm my gate",
        "excellent crew and smooth flight",
        "thank you for the great service",
        "wonderful airline experience",
    ] * 3
    labels = [
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "positive",
        "positive",
        "positive",
    ] * 3
    candidate = next(item for item in candidate_specs() if item.name == "logistic_regression")
    cv = TimeSeriesSplit(n_splits=3)

    scores = cross_val_score(candidate.estimator, texts, labels, cv=cv, scoring="f1_macro")

    assert len(scores) == 3
    assert scores.min() >= 0
