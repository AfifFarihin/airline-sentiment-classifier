"""Leakage-aware airline tweet sentiment classification."""

from airline_sentiment.data import chronological_split, load_airline_tweets, normalize_text
from airline_sentiment.inference import load_pipeline, predict_text
from airline_sentiment.training import train_and_evaluate

__all__ = [
    "chronological_split",
    "load_airline_tweets",
    "load_pipeline",
    "normalize_text",
    "predict_text",
    "train_and_evaluate",
]
