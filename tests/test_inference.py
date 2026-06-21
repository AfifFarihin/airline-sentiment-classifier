import pytest

from airline_sentiment.inference import predict_text
from airline_sentiment.modeling import candidate_specs


def _fitted_logistic_pipeline() -> object:
    texts = [
        "awful delay",
        "terrible service",
        "flight information",
        "where is the gate",
        "great flight",
        "excellent service",
    ] * 3
    labels = ["negative", "negative", "neutral", "neutral", "positive", "positive"] * 3
    candidate = next(item for item in candidate_specs() if item.name == "logistic_regression")
    model = candidate.estimator.set_params(tfidf__min_df=1)
    return model.fit(texts, labels)


def test_predict_text_returns_probabilities() -> None:
    result = predict_text(_fitted_logistic_pipeline(), "Thank you for the excellent flight")

    assert result["label"] in {"negative", "neutral", "positive"}
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_predict_text_rejects_blank_input() -> None:
    with pytest.raises(ValueError, match="Enter some text"):
        predict_text(_fitted_logistic_pipeline(), "   ")
