"""Download the public Kaggle dataset and extract only Tweets.csv."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import requests

URL = "https://www.kaggle.com/api/v1/datasets/download/crowdflower/twitter-airline-sentiment"
EXPECTED_SHA256 = "ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "Tweets.csv"


def main() -> None:
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        payload = archive.read("Tweets.csv")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Dataset checksum changed. Review the upstream release before using it."
        )
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(payload)
    print(f"Saved verified dataset to {DESTINATION}")


if __name__ == "__main__":
    main()
