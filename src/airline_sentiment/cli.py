"""Command-line entrypoint for the complete training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airline_sentiment.training import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Tweets.csv"),
        help="Path to the Kaggle Tweets.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for aggregate metrics, figures, and the local model artifact.",
    )
    args = parser.parse_args()
    summary = train_and_evaluate(args.data, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
