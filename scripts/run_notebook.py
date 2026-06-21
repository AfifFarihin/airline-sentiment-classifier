"""Execute the aggregate portfolio notebook without accessing raw tweets."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "airline_sentiment_analysis.ipynb"


def main() -> None:
    notebook = nbformat.read(PATH, as_version=4)
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(notebook, PATH)
    print(f"Executed {PATH}")


if __name__ == "__main__":
    main()
