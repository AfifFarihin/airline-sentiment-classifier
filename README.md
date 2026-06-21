# Airline Sentiment Classifier

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)

A leakage-aware multiclass NLP benchmark for U.S. airline tweets. The project
compares a majority baseline, Complement Naive Bayes, and class-balanced
logistic regression using TF-IDF features, then evaluates the selected model on
strictly later messages.

![Normalized confusion matrix](outputs/figures/confusion_matrix.png)

## Why This Version Is Different

The original university exercise used one random train/test split and weighted
F1. This portfolio version makes the evaluation harder and more useful:

- Removes duplicate tweet IDs and duplicate text before splitting.
- Reserves the latest 20% of rows as a chronological holdout.
- Fits TF-IDF inside every cross-validation fold.
- Selects models by macro F1 so minority classes matter.
- Compares against an explicit majority-class baseline.
- Bootstraps uncertainty by author rather than treating every tweet as
  independent.
- Publishes aggregate artifacts only; raw tweets and direct identifiers stay
  out of Git.

## Results

The selected model is class-balanced logistic regression with word unigram and
bigram TF-IDF features.

| Model | Training CV macro F1 | Later holdout macro F1 | Holdout accuracy |
|---|---:|---:|---:|
| Logistic regression | 0.749 | **0.726** | **0.775** |
| Complement Naive Bayes | 0.733 | 0.701 | 0.759 |
| Majority baseline | 0.260 | 0.249 | 0.595 |

The selected model's author-grouped 95% interval is **0.707-0.744** for macro
F1 and **0.759-0.791** for accuracy. Per-class F1 is 0.854 for negative, 0.645
for neutral, and 0.679 for positive sentiment. Neutral language remains the
largest weakness.

![Class distribution over time](outputs/figures/class_distribution.png)

## Evaluation Design

```mermaid
flowchart LR
    A["Kaggle dataset"] --> B["Validate and deduplicate"]
    B --> C["Earlier 80%"]
    B --> D["Later 20% holdout"]
    C --> E["5-fold model selection by macro F1"]
    E --> F["Selected TF-IDF + logistic pipeline"]
    F --> D
    D --> G["Metrics, grouped uncertainty, and error matrix"]
```

The holdout begins seven seconds after the training period ends. It therefore
tests later messages from the same one-week collection, not performance in a
different year or platform.

## Quick Start

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/AfifFarihin/airline-sentiment-classifier.git
cd airline-sentiment-classifier
uv sync --extra notebooks --group dev
uv run python scripts/download_data.py
uv run airline-sentiment
```

The downloader retrieves the upstream Kaggle archive, extracts only
`Tweets.csv`, and verifies its SHA-256 checksum.

## Repository Layout

```text
src/airline_sentiment/   Reusable data, modelling, training, and CLI code
scripts/                 Verified data download and notebook generation
notebooks/               Portfolio analysis built from aggregate artifacts
outputs/metrics/         Model comparison, class report, uncertainty, features
outputs/figures/         Privacy-safe evaluation figures
tests/                   Data contracts, leakage guards, and artifact checks
```

## Checks

```powershell
uv run ruff check src scripts tests
uv run pytest -q
uv run pip-audit --skip-editable
```

## Data and Responsible Use

The raw Twitter US Airline Sentiment dataset is not committed. It is available
from Kaggle under CC BY-NC-SA 4.0. The source includes public tweet text,
usernames, and locations; this repository publishes none of those fields.

This is a historical portfolio benchmark, not a production customer-monitoring
system. See [DATA_LICENSES.md](DATA_LICENSES.md) and
[MODEL_CARD.md](MODEL_CARD.md) for provenance, intended use, and limitations.

## Author

Developed by [AfifFarihin](https://github.com/AfifFarihin) from a university
text-analytics exercise, then rebuilt as a reproducible NLP portfolio project.

## License

Original code and documentation are released under the [MIT License](LICENSE).
The upstream dataset retains its CC BY-NC-SA 4.0 terms.
