"""Train, select, evaluate, and serialize the sentiment benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from airline_sentiment.data import chronological_split, load_airline_tweets
from airline_sentiment.modeling import CandidateSpec, candidate_specs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["negative", "neutral", "positive"]


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _grouped_bootstrap_intervals(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    n_boot: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap holdout metrics by author to preserve within-author dependence."""
    working = frame[["airline_sentiment", "author_group"]].copy()
    working["prediction"] = predictions
    groups = working["author_group"].unique()
    group_indices = {
        group: np.flatnonzero(working["author_group"].to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(random_state)
    metric_samples: dict[str, list[float]] = {
        key: [] for key in _metrics(working["airline_sentiment"], predictions)
    }
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sample_metrics = _metrics(
            working.iloc[indices]["airline_sentiment"],
            working.iloc[indices]["prediction"].to_numpy(),
        )
        for name, value in sample_metrics.items():
            metric_samples[name].append(value)

    observed = _metrics(working["airline_sentiment"], predictions)
    rows = []
    for name, values in metric_samples.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "metric": name,
                "observed": observed[name],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_boot": n_boot,
                "grouping": "author_group",
                "n_groups": int(len(groups)),
            }
        )
    return pd.DataFrame(rows)


def _generalization_slices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Report performance separately for repeat and previously unseen authors."""
    seen_authors = set(train["author_group"])
    working = test[["airline_sentiment", "author_group"]].copy()
    working["prediction"] = predictions
    working["author_segment"] = np.where(
        working["author_group"].isin(seen_authors),
        "seen_in_training",
        "unseen_in_training",
    )
    rows = []
    for segment, group in working.groupby("author_segment"):
        rows.append(
            {
                "author_segment": segment,
                "n_rows": int(len(group)),
                "n_author_groups": int(group["author_group"].nunique()),
                **_metrics(group["airline_sentiment"], group["prediction"].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("author_segment").reset_index(drop=True)


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axis.set_xticks(range(len(LABELS)), LABELS)
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Chronological holdout confusion matrix")
    fig.colorbar(image, ax=axis, label="Recall within actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_class_distribution(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    rows = []
    for split_name, frame in [("Earlier training period", train), ("Later holdout period", test)]:
        proportions = frame["airline_sentiment"].value_counts(normalize=True)
        for label in LABELS:
            rows.append({"split": split_name, "label": label, "proportion": proportions[label]})
    plot_data = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    width = 0.34
    x = np.arange(len(LABELS))
    for offset, split_name in zip([-width / 2, width / 2], plot_data["split"].unique()):
        values = plot_data[plot_data["split"] == split_name]["proportion"].to_numpy()
        axis.bar(x + offset, values, width, label=split_name)
    axis.set_xticks(x, LABELS)
    axis.set_ylabel("Share of rows")
    axis.set_ylim(0, 0.75)
    axis.set_title("Sentiment distribution over time")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _top_logistic_features(model: object, path: Path) -> None:
    if "model" not in model.named_steps or not isinstance(
        model.named_steps["model"], LogisticRegression
    ):
        return
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]
    names = np.asarray(vectorizer.get_feature_names_out())
    rows = []
    for class_name, coefficients in zip(classifier.classes_, classifier.coef_):
        for rank, index in enumerate(np.argsort(coefficients)[-15:][::-1], start=1):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": names[index],
                    "coefficient": float(coefficients[index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def train_and_evaluate(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_fraction: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
    n_boot: int = 2000,
    candidates: list[CandidateSpec] | None = None,
) -> dict[str, object]:
    """Run model selection on earlier data and evaluate once on later data."""
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    models_dir = output / "models"
    for directory in [metrics_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    frame, audit = load_airline_tweets(dataset_path)
    train, test = chronological_split(frame, test_fraction=test_fraction)
    x_train = train["text"]
    y_train = train["airline_sentiment"]
    x_test = test["text"]
    y_test = test["airline_sentiment"]
    cv = TimeSeriesSplit(n_splits=cv_splits)

    comparison_rows = []
    fitted_searches = {}
    for candidate in candidates or candidate_specs(random_state):
        search = GridSearchCV(
            clone(candidate.estimator),
            candidate.parameters,
            scoring={
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
                "accuracy": "accuracy",
            },
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        best_index = int(search.best_index_)
        row = {
            "model": candidate.name,
            "cv_f1_macro": float(search.best_score_),
            "cv_f1_weighted": float(search.cv_results_["mean_test_f1_weighted"][best_index]),
            "cv_accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        }
        comparison_rows.append(row)
        fitted_searches[candidate.name] = search

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_f1_macro", "model"], ascending=[False, True]
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_search = fitted_searches[selected_name]
    selected_model = selected_search.best_estimator_
    selected_predictions = selected_model.predict(x_test)

    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    report = pd.DataFrame(
        classification_report(
            y_test,
            selected_predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(metrics_dir / "classification_report.csv")
    uncertainty = _grouped_bootstrap_intervals(
        test,
        selected_predictions,
        n_boot=n_boot,
        random_state=random_state,
    )
    uncertainty.to_csv(metrics_dir / "holdout_uncertainty.csv", index=False)
    slices = _generalization_slices(train, test, selected_predictions)
    slices.to_csv(metrics_dir / "generalization_slices.csv", index=False)
    _top_logistic_features(selected_model, metrics_dir / "top_features.csv")
    _save_confusion_matrix(
        y_test, selected_predictions, figures_dir / "confusion_matrix.png"
    )
    _save_class_distribution(train, test, figures_dir / "class_distribution.png")
    joblib.dump(selected_model, models_dir / "sentiment_pipeline.joblib")

    summary = {
        "selection_rule": (
            "Highest expanding-window training-period macro F1; model name breaks exact ties. "
            "Only the selected model is evaluated on the final holdout."
        ),
        "selected_model": selected_name,
        "selected_parameters": selected_search.best_params_,
        "holdout_protocol": (
            "Latest 20% of deduplicated tweets by timestamp. Candidate selection does not "
            "inspect this holdout."
        ),
        "holdout_metrics": _metrics(y_test, selected_predictions),
        "holdout_uncertainty": {
            row["metric"]: {
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
            }
            for row in uncertainty.to_dict(orient="records")
        },
        "dataset_audit": asdict(audit),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "holdout_author_groups": int(test["author_group"].nunique()),
        "author_generalization_slices": slices.to_dict(orient="records"),
        "train_end": train["tweet_created"].max().isoformat(),
        "test_start": test["tweet_created"].min().isoformat(),
        "privacy": (
            "Raw usernames are pseudonymized in memory for author-grouped uncertainty and "
            "seen-versus-unseen-author evaluation; they are never model features or published. "
            "Published metrics and figures contain no tweet text, usernames, locations, or tweet IDs."
        ),
    }
    (metrics_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
