"""Offline Streamlit demo for the trained sentiment pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from airline_sentiment.inference import load_pipeline, predict_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "models" / "sentiment_pipeline.joblib"
SUMMARY_PATH = ROOT / "outputs" / "metrics" / "evaluation_summary.json"

st.set_page_config(
    page_title="Airline Sentiment Classifier",
    page_icon=None,
    layout="centered",
)
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 900px; padding-top: 2.5rem; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe5ef;
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def model() -> object:
    return load_pipeline(MODEL_PATH)


@st.cache_data
def summary() -> dict[str, object]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


st.title("Airline Sentiment Classifier")
st.caption(
    "An offline TF-IDF and logistic-regression demo trained on historical "
    "U.S. airline tweets."
)

if not MODEL_PATH.exists():
    st.error("The local model has not been trained yet.")
    st.code(
        "uv run python scripts/download_data.py\n"
        "uv run airline-sentiment\n"
        "uv run streamlit run demo/app.py"
    )
    st.stop()

facts = summary()
left, middle, right = st.columns(3)
left.metric("Later-period macro F1", f"{facts['holdout_metrics']['f1_macro']:.3f}")
middle.metric("Later-period accuracy", f"{facts['holdout_metrics']['accuracy']:.3f}")
right.metric("Evaluation rows", f"{facts['test_rows']:,}")

text = st.text_area(
    "Enter an airline-related message",
    value="Thank you for the excellent crew and smooth flight!",
    height=130,
    help="The text is processed locally and is not sent to an external service.",
)

try:
    result = predict_text(model(), text)
except ValueError as error:
    st.warning(str(error))
else:
    prediction, confidence = st.columns(2)
    prediction.metric("Predicted sentiment", str(result["label"]).title())
    probabilities = result["probabilities"]
    selected_probability = float(probabilities.get(str(result["label"]), 0.0))
    confidence.metric("Model confidence", f"{selected_probability:.1%}")
    st.subheader("Class probabilities")
    for label, probability in probabilities.items():
        label_column, value_column = st.columns([4, 1])
        label_column.write(label.title())
        value_column.write(f"{probability:.1%}")
        st.progress(float(probability))

st.info(
    "Portfolio demonstration only. The model uses a one-week dataset from 2015 "
    "and should not drive customer, employee, or operational decisions."
)
st.caption(
    "Privacy: inference runs locally. The repository contains aggregate model "
    "evidence, not raw tweets, usernames, locations, or tweet IDs."
)
