"""Offline inference helpers used by the recruiter-facing demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def load_pipeline(path: str | Path) -> object:
    """Load a locally trained sentiment pipeline."""
    return joblib.load(path)


def predict_text(model: object, text: str) -> dict[str, object]:
    """Predict one non-empty message and return class probabilities."""
    value = text.strip()
    if not value:
        raise ValueError("Enter some text before requesting a prediction.")
    label = str(model.predict([value])[0])
    if not hasattr(model, "predict_proba"):
        return {"label": label, "probabilities": {}}
    probability_values = np.asarray(model.predict_proba([value])[0], dtype=float)
    classes = [str(item) for item in model.classes_]
    return {
        "label": label,
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(classes, probability_values)
        },
    }
